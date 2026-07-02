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
    QLabel, QComboBox, QGroupBox, QPushButton, QFileDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
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
        self._sfm_dir:       Optional[Path]        = None
        self._out_dir:       Optional[Path]        = None
        self._query_poses:  dict                  = {}   # name -> center (np.ndarray)
        self._query_times:  dict                  = {}   # name -> elapsed_ms
        self._single_worker: Optional[QThread]    = None
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

        # ── Single image localization ─────────────────────────────────
        single_grp = QGroupBox("Localizar foto suelta")
        single_layout = QVBoxLayout(single_grp)
        self._single_btn = QPushButton("📷  Seleccionar imagen…")
        self._single_btn.setEnabled(False)
        self._single_btn.clicked.connect(self._pick_and_localize)
        self._single_status = QLabel("Cargá un mapa primero.")
        self._single_status.setWordWrap(True)
        self._single_status.setStyleSheet("color:#666; font-size:10px;")
        single_layout.addWidget(self._single_btn)
        single_layout.addWidget(self._single_status)
        rl.addWidget(single_grp)

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

        self._o3d_btn = QPushButton("🔲 3D viewer")
        self._o3d_btn.setToolTip("Abrir nube de puntos en viewer interactivo (ventana separada)")
        self._o3d_btn.setStyleSheet(BTN_STYLE)
        self._o3d_btn.setEnabled(False)
        self._o3d_btn.clicked.connect(self._open_in_open3d)
        layout.addWidget(self._o3d_btn)

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
        self._images_dir   = Path(images_dir).resolve() if images_dir else None
        self._sfm_dir      = Path(sfm_dir).resolve()
        self._out_dir      = self._sfm_dir.parent
        self._query_poses  = {}
        self._query_times  = {}
        self._artist_others   = None
        self._artist_selected = None

        # Build filename → path index
        self._image_index = {}
        if self._images_dir and self._images_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
                for p in self._images_dir.rglob(ext):
                    self._image_index[p.name] = p
            if not self._image_index:
                self._photo.setText(
                    f"Carpeta de imágenes vacía o no encontrada:\n{self._images_dir}"
                )
        elif images_dir:
            self._photo.setText(
                f"Carpeta de imágenes no encontrada:\n{images_dir}"
            )

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
        has_model = self._points is not None and len(self._points) > 0
        self._o3d_btn.setEnabled(has_model)
        self._single_btn.setEnabled(has_model)
        self._single_status.setText(
            "Seleccioná una foto para localizarla en el mapa." if has_model
            else "Cargá un mapa primero."
        )

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

        rp = Path(results_path) if results_path else None
        if rp and rp.is_file():
            try:
                poses = _parse_results(rp)
            except Exception:
                poses = {}
            for name, center in poses.items():
                self._query_poses[name] = center
                self._combo.blockSignals(True)
                self._combo.addItem(name)
                self._combo.blockSignals(False)

        if self._combo.count() > 0:
            self._combo.setCurrentIndex(-1)  # force change signal on next line
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

    def _remove_artist(self, artist) -> None:
        """Remove a matplotlib artist, handling versions that raise NotImplementedError."""
        try:
            artist.remove()
        except (ValueError, NotImplementedError):
            # Fallback: hide it so it doesn't render
            try:
                artist.set_visible(False)
            except Exception:
                pass

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
            self._remove_artist(self._artist_db_highlight)
            self._artist_db_highlight = None

        # Remove previous dynamic artists
        if self._artist_others is not None:
            self._remove_artist(self._artist_others)
            self._artist_others = None
        if self._artist_selected is not None:
            self._remove_artist(self._artist_selected)
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
    # Single-image localization
    # ------------------------------------------------------------------

    def _pick_and_localize(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)",
        )
        if not path:
            return

        conf_file = self._out_dir / "pipeline_conf.json"
        if not conf_file.exists():
            self._single_status.setText(
                "⚠ No se encontró pipeline_conf.json — "
                "re-ejecutá el pipeline una vez para guardarlo."
            )
            return

        import json
        conf = json.loads(conf_file.read_text())

        if self._single_worker and self._single_worker.isRunning():
            self._single_worker.terminate()
            self._single_worker.wait()

        self._single_btn.setEnabled(False)
        self._single_status.setText("⏳ Localizando…")

        self._single_worker = SingleQueryWorker(path, self._out_dir, self._sfm_dir, conf)
        self._single_worker.log.connect(self._single_status.setText)
        self._single_worker.done.connect(self._on_single_done)
        self._single_worker.error.connect(self._on_single_error)
        self._single_worker.start()

    def _on_single_done(self, results_line: str, center, original_path: str = "") -> None:
        self._single_btn.setEnabled(True)
        if center is None:
            self._single_status.setText("⚠ No se pudo localizar la imagen.")
            return

        import numpy as np
        name = results_line.split()[0] if results_line else "foto_suelta"
        self._query_poses[name] = np.array(center)

        # Register the original image so _update_photo can find it
        if original_path:
            self._image_index[name] = Path(original_path)

        self._combo.blockSignals(True)
        existing_idx = -1
        for i in range(self._combo.count()):
            if self._combo.itemText(i) == name:
                existing_idx = i
                break
        if existing_idx == -1:
            self._combo.addItem(name)
            existing_idx = self._combo.count() - 1
        self._combo.blockSignals(False)
        self._combo.setCurrentIndex(existing_idx)

        self._single_status.setText(f"✓ Localizado: {name}")

    def _on_single_error(self, msg: str) -> None:
        self._single_btn.setEnabled(True)
        self._single_status.setText(f"✗ Error: {msg.splitlines()[0]}")

    # ------------------------------------------------------------------
    # Open3D viewer
    # ------------------------------------------------------------------

    def _open_in_open3d(self) -> None:
        try:
            import pyvista  # noqa: F401
        except ImportError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "PyVista no instalado",
                "Instalá PyVista con:\n\npip install pyvista",
            )
            return

        points      = self._points
        colors      = self._point_colors
        db_centers  = self._db_centers
        query_poses = dict(self._query_poses)

        import threading
        t = threading.Thread(
            target=self._run_pyvista_window,
            args=(points, colors, db_centers, query_poses),
            daemon=True,
        )
        t.start()

    @staticmethod
    def _run_pyvista_window(points, colors, db_centers, query_poses) -> None:
        import numpy as np
        import pyvista as pv

        pl = pv.Plotter(window_size=(1280, 800), title="TP COLMAP — Nube de puntos")
        pl.set_background("#1a1a2e")

        # Point cloud
        if points is not None and len(points):
            cloud = pv.PolyData(points.astype(np.float32))
            if colors is not None:
                cloud["colors"] = (colors * 255).astype(np.uint8)
                pl.add_mesh(cloud, scalars="colors", rgb=True,
                            point_size=2.5, render_points_as_spheres=False,
                            style="points")
            else:
                pl.add_mesh(cloud, color="white", point_size=2.5, style="points")

        # DB cameras — blue spheres
        if db_centers is not None and len(db_centers):
            r = max(np.ptp(points, axis=0).max() * 0.008, 0.03) if points is not None and len(points) else 0.05
            db_cloud = pv.PolyData(db_centers.astype(np.float32))
            pl.add_mesh(
                db_cloud.glyph(geom=pv.Sphere(radius=r), scale=False, orient=False),
                color="#4a9eff", smooth_shading=True, label="Cámaras DB",
            )

        # Query poses — red spheres (larger)
        if query_poses:
            q_pts   = np.array(list(query_poses.values()), dtype=np.float32)
            r_q     = max(np.ptp(points, axis=0).max() * 0.013, 0.05) if points is not None and len(points) else 0.08
            q_cloud = pv.PolyData(q_pts)
            pl.add_mesh(
                q_cloud.glyph(geom=pv.Sphere(radius=r_q), scale=False, orient=False),
                color="#ff3333", smooth_shading=True, label="Queries",
            )

        pl.add_legend(bcolor="#111111", border=True, size=(0.15, 0.1))
        pl.show()

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
                self._remove_artist(artist)
                setattr(self, attr, None)

        # Move yellow highlight ring to this camera
        if self._artist_db_highlight is not None:
            self._remove_artist(self._artist_db_highlight)
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


# ---------------------------------------------------------------------------
# Single-image localization worker
# ---------------------------------------------------------------------------

class SingleQueryWorker(QThread):
    log   = Signal(str)
    done  = Signal(str, object, str)   # results_line, center, original_image_path
    error = Signal(str)

    def __init__(self, image_path: str, out_dir: Path, sfm_dir: Path, conf: dict):
        super().__init__()
        self._image_path = Path(image_path)
        self._out_dir    = Path(out_dir)
        self._sfm_dir    = Path(sfm_dir)
        self._conf       = conf

    def run(self) -> None:
        try:
            self._execute()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _execute(self) -> None:
        import shutil, time, json
        import numpy as np
        from hloc import extract_features, match_features, pairs_from_exhaustive
        from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
        from hloc.utils.parsers import parse_image_lists, parse_retrieval
        import pycolmap
        from src.utils.colmap_io import qvec2rotmat

        ext_key       = self._conf["extractor"]
        mat_key       = self._conf["matcher"]
        cal           = self._conf.get("calibration")
        feature_conf  = extract_features.confs[ext_key]
        matcher_conf  = match_features.confs[mat_key]
        features_name = feature_conf["output"]

        # 1. Copy image to isolated subdir so its h5 key is just the filename
        single_dir = self._out_dir / "single_query"
        single_dir.mkdir(exist_ok=True)
        dest = single_dir / self._image_path.name
        shutil.copy2(self._image_path, dest)

        # 2. Extract features for the single image
        self.log.emit(f"Extrayendo features de {self._image_path.name}…")
        feature_path = extract_features.main(
            feature_conf, single_dir,
            export_dir=self._out_dir,
            image_list=[self._image_path.name],
            overwrite=True,
        )

        # 3. Load SfM model
        sfm_model    = pycolmap.Reconstruction(str(self._sfm_dir))
        db_images    = [img.name for img in sfm_model.images.values()]
        db_name_to_id = {img.name: i for i, img in sfm_model.images.items()}
        self.log.emit(f"Modelo: {len(db_images)} imágenes DB")

        # 4. Exhaustive pairs: single query vs all DB
        pairs_file = self._out_dir / "pairs_single.txt"
        pairs_from_exhaustive.main(
            pairs_file,
            image_list=[self._image_path.name],
            ref_list=db_images,
        )

        # 5. Match features
        self.log.emit("Matching…")
        matches_path = match_features.main(
            matcher_conf, pairs_file,
            features=features_name,
            export_dir=self._out_dir,
            overwrite=True,
        )

        # 6. Write query intrinsics file
        if cal:
            model_name = cal["model"]
            w, h       = cal["width"], cal["height"]
            params_str = " ".join(f"{p:.6f}" for p in cal["params"])
        else:
            cam = next(iter(sfm_model.cameras.values()))
            try:
                model_name = cam.model_name
            except AttributeError:
                model_name = str(cam.model).split(".")[-1]
            w, h       = cam.width, cam.height
            params_str = " ".join(f"{p:.6f}" for p in cam.params)

        queries_txt = self._out_dir / "single_query_list.txt"
        queries_txt.write_text(
            f"{self._image_path.name} {model_name} {w} {h} {params_str}\n"
        )

        # 7. Localize
        self.log.emit("Localizando…")
        queries    = parse_image_lists(queries_txt, with_intrinsics=True)
        retrieval  = parse_retrieval(pairs_file)
        config     = {"estimation": {"ransac": {"max_error": 12}}}
        localizer  = QueryLocalizer(sfm_model, config)

        for qname, qcam in queries:
            candidates = retrieval.get(qname, [])
            db_ids = [db_name_to_id[n] for n in candidates if n in db_name_to_id]
            if not db_ids:
                self.error.emit("Sin candidatas DB para la imagen.")
                return

            t0 = time.perf_counter()
            ret, _ = pose_from_cluster(
                localizer, qname, qcam, db_ids, feature_path, matches_path
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if ret is not None:
                pose = ret["cam_from_world"]
                self.log.emit(f"✓ Localizado en {elapsed_ms:.1f} ms")
            else:
                pose = sfm_model.images[db_ids[0]].cam_from_world()
                self.log.emit(f"⚠ Pose aproximada (cámara DB más cercana) — {elapsed_ms:.1f} ms")

            q    = pose.rotation.quat           # [qx, qy, qz, qw]
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            tx, ty, tz     = pose.translation
            R              = qvec2rotmat(np.array([qw, qx, qy, qz]))
            center         = -R.T @ np.array([tx, ty, tz])

            line = f"{self._image_path.name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"
            self.done.emit(line, center.tolist(), str(self._image_path))
            return

        self.error.emit("No se encontraron queries en el archivo generado.")
