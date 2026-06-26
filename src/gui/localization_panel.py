"""
Localization viewer — query photo + estimated camera position in the 3D model.

3D rendering strategy:
  - Static layer  (point cloud + DB cameras) drawn once in setup_localization().
  - Dynamic layer (query markers) removed/re-added on each selection change.
  This avoids redrawing 8 000+ points every time the user switches query.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QComboBox, QGroupBox, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def _parse_results(results_path: Path) -> dict:
    """Parse hloc results.txt -> {basename: camera_center_world}."""
    from src.utils.colmap_io import qvec2rotmat
    poses = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            try:
                qw, qx, qy, qz = (float(v) for v in parts[1:5])
                tx, ty, tz     = (float(v) for v in parts[5:8])
            except ValueError:
                continue
            R = qvec2rotmat(np.array([qw, qx, qy, qz]))
            poses[name] = -R.T @ np.array([tx, ty, tz])
    return poses


def _center_from_line(line: str) -> Optional[np.ndarray]:
    """Parse a single results line -> camera center in world coords."""
    from src.utils.colmap_io import qvec2rotmat
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    try:
        qw, qx, qy, qz = (float(v) for v in parts[1:5])
        tx, ty, tz     = (float(v) for v in parts[5:8])
    except ValueError:
        return None
    R = qvec2rotmat(np.array([qw, qx, qy, qz]))
    return -R.T @ np.array([tx, ty, tz])


class LocalizationPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._images_dir:    Optional[Path]        = None
        self._query_poses:  dict                  = {}   # name -> center (np.ndarray)
        self._query_times:  dict                  = {}   # name -> elapsed_ms
        self._db_centers:   Optional[np.ndarray]  = None
        self._db_names:     list                  = []   # same order as _db_centers
        self._points:       Optional[np.ndarray]  = None
        self._point_colors: Optional[np.ndarray]  = None
        self._image_index:  dict                  = {}   # basename -> absolute Path

        # Dynamic matplotlib artists (removed/re-added on update)
        self._artist_others      = None
        self._artist_selected    = None
        self._artist_db          = None   # pickable DB cameras scatter
        self._artist_db_highlight = None  # yellow ring on clicked DB camera

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: 3D viewer ──────────────────────────────────────────
        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._fig    = Figure(facecolor="#1a1a1a")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax     = self._fig.add_subplot(111, projection="3d")
        self._style_ax()

        ll.addWidget(self._make_view_toolbar())
        ll.addWidget(self._canvas, 1)

        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("pick_event",   self._on_pick)

        splitter.addWidget(left)

        # ── Right: photo + query selector ────────────────────────────
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        rl.setSpacing(8)
        right.setMinimumWidth(260)
        right.setMaximumWidth(480)

        self._photo = QLabel("Ejecutá el pipeline\npara ver queries aquí")
        self._photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._photo.setMinimumHeight(220)
        self._photo.setStyleSheet(
            "background:#1a1a1a; border:1px solid #444; color:#666; font-size:11px;"
        )
        rl.addWidget(self._photo, 1)

        grp = QGroupBox("Query localizada")
        gl  = QVBoxLayout(grp)

        self._combo = QComboBox()
        self._combo.setPlaceholderText("— sin resultados —")
        self._combo.currentTextChanged.connect(self._on_query_changed)
        gl.addWidget(self._combo)

        self._info = QLabel("—")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color:#aaa; font-size:10px; padding:2px;")
        gl.addWidget(self._info)

        nav = QHBoxLayout()
        btn_prev = QPushButton("◀  Anterior")
        btn_next = QPushButton("Siguiente  ▶")
        btn_prev.clicked.connect(lambda: self._step(-1))
        btn_next.clicked.connect(lambda: self._step(+1))
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        gl.addLayout(nav)
        rl.addWidget(grp)

        legend = QLabel("🔵 DB cameras   ★ Query seleccionada   · otras queries")
        legend.setStyleSheet("color:#555; font-size:9px;")
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(legend)

        splitter.addWidget(right)
        splitter.setSizes([680, 320])
        root.addWidget(splitter, 1)

    def _style_ax(self) -> None:
        self._ax.set_facecolor("#1a1a1a")
        self._ax.set_axis_off()

    def _make_view_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(30)
        bar.setStyleSheet("background:#222;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        BTN_STYLE = (
            "QPushButton { background:#333; color:#ccc; border:1px solid #555; "
            "border-radius:3px; padding:2px 8px; font-size:11px; }"
            "QPushButton:hover { background:#444; color:#fff; }"
        )

        def _btn(label: str, tip: str, elev: float, azim: float) -> QPushButton:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setStyleSheet(BTN_STYLE)
            b.clicked.connect(lambda: self._set_view(elev, azim))
            return b

        layout.addWidget(_btn("⌂",    "Reset (perspectiva)",    15,  45))
        layout.addWidget(_btn("Top",  "Vista desde arriba",     89,   0))
        layout.addWidget(_btn("Front","Vista frontal",           0,   0))
        layout.addWidget(_btn("Side", "Vista lateral",           0,  90))
        layout.addStretch()

        hint = QLabel("Rotar: drag  ·  Zoom: scroll")
        hint.setStyleSheet("color:#555; font-size:10px;")
        layout.addWidget(hint)
        return bar

    def _set_view(self, elev: float, azim: float) -> None:
        self._ax.view_init(elev=elev, azim=azim)
        self._canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        factor = 0.85 if event.step > 0 else 1.15
        ax = self._ax
        for get_lim, set_lim in [
            (ax.get_xlim3d, ax.set_xlim3d),
            (ax.get_ylim3d, ax.set_ylim3d),
            (ax.get_zlim3d, ax.set_zlim3d),
        ]:
            lo, hi = get_lim()
            mid  = (lo + hi) / 2
            half = (hi - lo) / 2 * factor
            set_lim(mid - half, mid + half)
        self._canvas.draw_idle()

    def _fit_view(self) -> None:
        all_pts = []
        if self._points is not None and len(self._points):
            all_pts.append(self._points)
        if self._db_centers is not None and len(self._db_centers):
            all_pts.append(self._db_centers)
        if not all_pts:
            return
        data = np.concatenate(all_pts, axis=0)
        lo   = data.min(axis=0)
        hi   = data.max(axis=0)
        mid  = (lo + hi) / 2.0
        half = (hi - lo).max() / 2.0 * 1.2
        ax   = self._ax
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def setup_localization(self, sfm_dir: str, images_dir: str) -> None:
        """
        Load the 3D model and draw the static background (point cloud + DB cameras).
        Called once before queries start arriving.
        """
        self._images_dir   = Path(images_dir).resolve()
        self._query_poses  = {}
        self._query_times  = {}
        self._artist_others   = None
        self._artist_selected = None

        # Build filename → path index
        self._image_index = {}
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            for p in self._images_dir.rglob(ext):
                self._image_index[p.name] = p

        # Load 3D model
        try:
            from src.utils.colmap_io import read_model, image_camera_center
            _, images, points3D = read_model(Path(sfm_dir))

            self._db_names   = [img.name for img in images.values()]
            self._db_centers = np.array([
                image_camera_center(img) for img in images.values()
            ])

            pts    = np.array([p.xyz        for p in points3D.values()])
            colors = np.array([p.rgb / 255. for p in points3D.values()])
            if len(pts) > 8000:
                idx    = np.random.choice(len(pts), 8000, replace=False)
                pts, colors = pts[idx], colors[idx]
            if len(pts):
                med  = np.median(pts, axis=0)
                mask = np.all(np.abs(pts - med) < 3 * (np.std(pts, axis=0) + 1e-9), axis=1)
                pts, colors = pts[mask], colors[mask]
            self._points        = pts
            self._point_colors  = colors
        except Exception:
            self._db_centers    = None
            self._points        = None
            self._point_colors  = None

        # Draw static background once
        ax = self._ax
        ax.cla()
        self._style_ax()

        if self._points is not None and len(self._points):
            sc = ax.scatter(
                self._points[:, 0], self._points[:, 1], self._points[:, 2],
                s=1.2, c=self._point_colors, alpha=0.65, linewidths=0,
            )
            sc.set_clip_on(False)

        if self._db_centers is not None and len(self._db_centers):
            sc = ax.scatter(
                self._db_centers[:, 0], self._db_centers[:, 1], self._db_centers[:, 2],
                s=22, c="#4a9eff", alpha=0.9, linewidths=0.6,
                edgecolors="white", zorder=5, picker=10,
            )
            sc.set_clip_on(False)
            self._artist_db = sc

        self._fit_view()
        ax.view_init(elev=15, azim=45)
        self._canvas.draw_idle()

        # Reset combo
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.blockSignals(False)
        self._info.setText("—")
        self._photo.setText("Reconstrucción lista\nEsperando queries…")

    def begin_localization(self) -> None:
        """Called when the localization loop starts (model already loaded)."""
        self._photo.setText("Localizando queries…")
        self._info.setText("Localizando queries…")

    def add_query_result(self, results_line: str, elapsed_ms: float) -> None:
        """
        Called for each query as it is localized (real-time).
        Parses the pose, adds to combo, updates dynamic markers.
        """
        parts = results_line.strip().split()
        if len(parts) < 8:
            return
        name   = parts[0]
        center = _center_from_line(results_line)
        if center is None:
            return

        self._query_poses[name] = center
        self._query_times[name] = elapsed_ms

        # Add to combo without triggering redraw; then auto-advance to latest
        self._combo.blockSignals(True)
        self._combo.addItem(name)
        self._combo.blockSignals(False)
        self._combo.setCurrentIndex(self._combo.count() - 1)
        # setCurrentIndex emits currentTextChanged → _on_query_changed

    def load_localization(self, sfm_dir: str, images_dir: str, results_path: str) -> None:
        """
        Batch load after pipeline completes (e.g. re-opening a previous run).
        Sets up the static background then adds all results at once.
        """
        self.setup_localization(sfm_dir, images_dir)
        poses = _parse_results(Path(results_path))
        for name, center in poses.items():
            self._query_poses[name] = center
            self._combo.blockSignals(True)
            self._combo.addItem(name)
            self._combo.blockSignals(False)

        if self._combo.count() > 0:
            self._combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _step(self, delta: int) -> None:
        n = self._combo.count()
        if n == 0:
            return
        self._combo.setCurrentIndex(
            max(0, min(n - 1, self._combo.currentIndex() + delta))
        )

    def _on_query_changed(self, name: str) -> None:
        if not name or name not in self._query_poses:
            return
        self._update_photo(name)
        self._update_3d(name)
        idx   = self._combo.currentIndex() + 1
        total = self._combo.count()
        ms    = self._query_times.get(name)
        time_str = f"  •  {ms:.1f} ms" if ms is not None else ""
        self._info.setText(f"[{idx}/{total}]  {name}{time_str}")

    def _update_photo(self, name: str) -> None:
        if self._images_dir is None:
            return
        img_path = self._images_dir / name
        if not img_path.exists():
            img_path = self._image_index.get(Path(name).name, img_path)
        if img_path.exists():
            pix = QPixmap(str(img_path))
            w   = max(self._photo.width(),  400)
            h   = max(self._photo.height(), 260)
            self._photo.setPixmap(
                pix.scaled(w, h,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self._photo.setText(f"No encontrada:\n{name}")

    def _update_3d(self, selected: str) -> None:
        ax = self._ax

        # Clear DB highlight — query selection takes over
        if self._artist_db_highlight is not None:
            try:
                self._artist_db_highlight.remove()
            except ValueError:
                pass
            self._artist_db_highlight = None

        # Remove previous dynamic artists
        if self._artist_others is not None:
            try:
                self._artist_others.remove()
            except ValueError:
                pass
            self._artist_others = None
        if self._artist_selected is not None:
            try:
                self._artist_selected.remove()
            except ValueError:
                pass
            self._artist_selected = None

        # Draw other queries (small gray dots)
        others = np.array([c for n, c in self._query_poses.items() if n != selected])
        if len(others):
            self._artist_others = ax.scatter(
                others[:, 0], others[:, 1], others[:, 2],
                s=8, c="#888", alpha=0.35, linewidths=0,
            )
            self._artist_others.set_clip_on(False)

        # Draw selected query (red star)
        sel = self._query_poses[selected]
        self._artist_selected = ax.scatter(
            [sel[0]], [sel[1]], [sel[2]],
            s=220, c="#ff3333", marker="*",
            edgecolors="white", linewidths=0.5,
            zorder=10,
        )
        self._artist_selected.set_clip_on(False)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # DB camera pick interaction
    # ------------------------------------------------------------------

    def _on_pick(self, event) -> None:
        if event.artist is not self._artist_db:
            return
        if not len(event.ind):
            return
        self._show_db_image(event.ind[0])

    def _show_db_image(self, idx: int) -> None:
        name = self._db_names[idx]

        # Clear query markers — DB click takes over the selection
        for attr in ("_artist_selected", "_artist_others"):
            artist = getattr(self, attr)
            if artist is not None:
                try:
                    artist.remove()
                except ValueError:
                    pass
                setattr(self, attr, None)

        # Move yellow highlight ring to this camera
        if self._artist_db_highlight is not None:
            try:
                self._artist_db_highlight.remove()
            except ValueError:
                pass
        pos = self._db_centers[idx]
        self._artist_db_highlight = self._ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            s=120, c="none", edgecolors="#ffcc00", linewidths=2.5, zorder=7,
        )
        self._artist_db_highlight.set_clip_on(False)
        self._canvas.draw_idle()

        # Show photo and update info
        self._update_photo(name)
        self._info.setText(f"[DB]  {Path(name).name}")
