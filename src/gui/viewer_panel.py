"""
3D reconstruction viewer — embedded matplotlib canvas inside Qt.
Shows the point cloud and camera positions from a COLMAP sparse model.
"""
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QGroupBox,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class ViewerPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._cameras = {}
        self._images  = {}
        self._points3D = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Matplotlib figure
        self._fig = Figure(figsize=(7, 5), facecolor="#1e1e1e")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._style_axes()

        toolbar = NavigationToolbar2QT(self._canvas, self)
        toolbar.setStyleSheet("background: #2a2a2a; color: white;")

        # Controls bar
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 2, 4, 2)

        self._info_label = QLabel("Esperando reconstrucción...")
        self._info_label.setStyleSheet("color: #888; font-size: 10px;")

        btn_reset = QPushButton("Reset vista")
        btn_reset.setFixedWidth(90)
        btn_reset.clicked.connect(self._reset_view)

        self._pt_slider = QSlider(Qt.Orientation.Horizontal)
        self._pt_slider.setRange(1, 100)
        self._pt_slider.setValue(50)
        self._pt_slider.setFixedWidth(100)
        self._pt_slider.setToolTip("Densidad de puntos (%)")
        self._pt_slider.valueChanged.connect(self._refresh)

        ctrl.addWidget(self._info_label, 1)
        ctrl.addWidget(QLabel("Puntos:"))
        ctrl.addWidget(self._pt_slider)
        ctrl.addWidget(btn_reset)

        layout.addWidget(toolbar)
        layout.addLayout(ctrl)
        layout.addWidget(self._canvas, 1)

    def _style_axes(self) -> None:
        ax = self._ax
        ax.set_facecolor("#1e1e1e")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#444")
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlabel("X", color="#888", fontsize=8)
        ax.set_ylabel("Y", color="#888", fontsize=8)
        ax.set_zlabel("Z", color="#888", fontsize=8)

    # ------------------------------------------------------------------
    # Public slot — called by runner signal
    # ------------------------------------------------------------------

    def load_model(self, sfm_dir: str) -> None:
        from src.utils.colmap_io import read_model
        try:
            self._cameras, self._images, self._points3D = read_model(Path(sfm_dir))
            n_imgs = len(self._images)
            n_pts  = len(self._points3D)
            self._info_label.setText(
                f"{n_imgs} cámaras  |  {n_pts:,} puntos 3D  |  Rotá con click + drag"
            )
            self._info_label.setStyleSheet("color: #4af; font-size: 10px;")
            self._refresh()
        except Exception as e:
            self._info_label.setText(f"Error al cargar modelo: {e}")
            self._info_label.setStyleSheet("color: #f44; font-size: 10px;")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if not self._images:
            return
        self._ax.cla()
        self._style_axes()
        self._draw_points()
        self._draw_cameras()
        self._canvas.draw_idle()

    def _draw_points(self) -> None:
        if not self._points3D:
            return

        pts   = np.array([p.xyz for p in self._points3D.values()])
        colors = np.array([p.rgb / 255.0 for p in self._points3D.values()])

        # Remove statistical outliers (beyond 3σ from median)
        median = np.median(pts, axis=0)
        std    = np.std(pts, axis=0) + 1e-9
        mask   = np.all(np.abs(pts - median) < 3 * std, axis=1)
        pts, colors = pts[mask], colors[mask]

        # Sub-sample according to slider
        pct = self._pt_slider.value() / 100.0
        n   = max(1, int(len(pts) * pct))
        idx = np.random.choice(len(pts), n, replace=False)

        self._ax.scatter(
            pts[idx, 0], pts[idx, 1], pts[idx, 2],
            c=colors[idx], s=0.5, alpha=0.6, linewidths=0,
        )

    def _draw_cameras(self) -> None:
        from src.utils.colmap_io import image_camera_center

        centers = np.array([
            image_camera_center(img) for img in self._images.values()
        ])
        self._ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c="orange", s=25, zorder=5, label=f"Cámaras ({len(centers)})",
        )
        self._ax.legend(
            facecolor="#333", labelcolor="white", fontsize=8,
            loc="upper left",
        )

    def _reset_view(self) -> None:
        self._ax.view_init(elev=20, azim=45)
        self._canvas.draw_idle()
