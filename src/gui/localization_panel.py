"""
Localization viewer — shows query photo + estimated camera position in the 3D model.
"""
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QComboBox, QGroupBox, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


def _parse_results(results_path: Path) -> dict:
    """Parse hloc results.txt -> {name: camera_center_world (np.ndarray shape (3,))}"""
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


class LocalizationPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._images_dir: Path | None = None
        self._query_poses: dict       = {}
        self._db_centers: np.ndarray | None = None
        self._points:     np.ndarray | None = None
        self._image_index: dict       = {}   # basename -> absolute Path
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
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._fig    = Figure(facecolor="#1e1e1e")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax     = self._fig.add_subplot(111, projection="3d")
        self._style_ax()

        toolbar = NavigationToolbar2QT(self._canvas, left)
        toolbar.setStyleSheet("background:#2a2a2a; color:white;")
        ll.addWidget(toolbar)
        ll.addWidget(self._canvas, 1)
        splitter.addWidget(left)

        # ── Right: photo + query selector ────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
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
        ax = self._ax
        ax.set_facecolor("#1e1e1e")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#444")
        ax.tick_params(colors="#888", labelsize=6)
        ax.set_xlabel("X", color="#888", fontsize=7)
        ax.set_ylabel("Y", color="#888", fontsize=7)
        ax.set_zlabel("Z", color="#888", fontsize=7)

    # ------------------------------------------------------------------
    # Public slot — called from MainWindow after localization finishes
    # ------------------------------------------------------------------

    def load_localization(self, sfm_dir: str, images_dir: str, results_path: str) -> None:
        self._images_dir  = Path(images_dir).resolve()
        self._query_poses = _parse_results(Path(results_path))

        # hloc's write_poses strips subdirectory → build filename→path index
        self._image_index = {}
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            for p in self._images_dir.rglob(ext):
                self._image_index[p.name] = p

        try:
            from src.utils.colmap_io import read_model, image_camera_center
            _, images, points3D = read_model(Path(sfm_dir))

            self._db_centers = np.array([
                image_camera_center(img) for img in images.values()
            ])

            pts = np.array([p.xyz for p in points3D.values()])
            if len(pts) > 8000:
                pts = pts[np.random.choice(len(pts), 8000, replace=False)]
            if len(pts):
                med  = np.median(pts, axis=0)
                mask = np.all(np.abs(pts - med) < 3 * (np.std(pts, axis=0) + 1e-9), axis=1)
                pts  = pts[mask]
            self._points = pts
        except Exception:
            self._db_centers = None
            self._points     = None

        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(sorted(self._query_poses.keys()))
        self._combo.blockSignals(False)

        if self._combo.count() > 0:
            self._combo.setCurrentIndex(0)
            self._on_query_changed(self._combo.currentText())

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
        self._info.setText(f"[{idx}/{total}]  {name}")

    def _update_photo(self, name: str) -> None:
        if self._images_dir is None:
            return
        # Try direct path first; fall back to index (hloc strips subdirectory prefix)
        img_path = self._images_dir / name
        if not img_path.exists():
            basename = Path(name).name
            img_path = self._image_index.get(basename, img_path)
        if img_path.exists():
            pix = QPixmap(str(img_path))
            w = max(self._photo.width(),  400)
            h = max(self._photo.height(), 260)
            self._photo.setPixmap(
                pix.scaled(w, h,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self._photo.setText(f"No encontrada:\n{name}")

    def _update_3d(self, selected: str) -> None:
        ax = self._ax
        ax.cla()
        self._style_ax()

        if self._points is not None and len(self._points):
            ax.scatter(
                self._points[:, 0], self._points[:, 1], self._points[:, 2],
                s=0.4, c="#777", alpha=0.25, linewidths=0,
            )

        if self._db_centers is not None and len(self._db_centers):
            ax.scatter(
                self._db_centers[:, 0], self._db_centers[:, 1], self._db_centers[:, 2],
                s=12, c="#4a9eff", alpha=0.6, linewidths=0, label="DB",
            )

        others = np.array([c for n, c in self._query_poses.items() if n != selected])
        if len(others):
            ax.scatter(
                others[:, 0], others[:, 1], others[:, 2],
                s=8, c="#888", alpha=0.35, linewidths=0,
            )

        sel = self._query_poses[selected]
        ax.scatter(
            [sel[0]], [sel[1]], [sel[2]],
            s=220, c="#ff3333", marker="*",
            edgecolors="white", linewidths=0.5,
            zorder=10, label="Query",
        )

        ax.legend(facecolor="#333", labelcolor="white", fontsize=7, loc="upper left")
        self._canvas.draw_idle()
