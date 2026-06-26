from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QRadioButton, QButtonGroup,
    QLabel, QPushButton, QFileDialog, QHBoxLayout, QLineEdit,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, Signal

from .calibration_widget import CalibrationWidget

VIDEO_EXTS = "Video (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv *.MP4 *.AVI *.MOV)"

DATASET_OPTIONS = [
    ("Cambridge — KingsCollege",  "cambridge",    "KingsCollege"),
    ("Cambridge — ShopFacade",    "cambridge",    "ShopFacade"),
    ("Cambridge — OldHospital",   "cambridge",    "OldHospital"),
    ("7-Scenes — Chess",          "seven_scenes", "chess"),
    ("7-Scenes — Fire",           "seven_scenes", "fire"),
    ("7-Scenes — Office",         "seven_scenes", "office"),
    ("COLMAP — South Building",   "colmap_dem",   "south-building"),
    ("COLMAP — Gerrard Hall",     "colmap_dem",   "gerrard-hall"),
    ("COLMAP — Person Hall",      "colmap_dem",   "person-hall"),
    ("COLMAP — Graham Hall",      "colmap_dem",   "graham-hall"),
    ("Dataset propio",            "custom",       None),
]

DESCRIPTIONS = {
    "cambridge":    "Localización outdoor. Auto-descarga ~250 MB. Tiene ground truth.",
    "seven_scenes": "Localización indoor RGB-D. Auto-descarga ~0.3-3 GB. Sin GT.",
    "colmap_dem":   "Datasets demo de COLMAP. Auto-descarga. South Building ~150 MB, Graham Hall ~1.5 GB.",
    "custom":       "Tus propias fotos o video. Split 80% DB / 20% query automático.",
}

_RB_STYLE = """
QRadioButton {
    color: #aaa;
    spacing: 6px;
    padding: 2px 0;
}
QRadioButton:checked {
    color: #ffffff;
    font-weight: bold;
}
QRadioButton::indicator {
    width: 13px;
    height: 13px;
    border-radius: 7px;
    border: 2px solid #555;
    background: #2a2a2a;
}
QRadioButton::indicator:checked {
    background: #0078d4;
    border-color: #0078d4;
}
QRadioButton::indicator:hover {
    border-color: #888;
}
"""


class DatasetPanel(QWidget):
    load_previous_requested = Signal()

    def __init__(self):
        super().__init__()
        self._custom_path: Optional[Path] = None
        self._is_video:    bool           = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Dataset")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(3)

        self._btn_group = QButtonGroup(self)

        for i, (label, module, _) in enumerate(DATASET_OPTIONS):
            rb = QRadioButton(label)
            rb.setStyleSheet(_RB_STYLE)
            if i == 0:
                rb.setChecked(True)
            self._btn_group.addButton(rb, i)
            group_layout.addWidget(rb)

            if module == "custom":
                # Folder / video picker row
                self._custom_row = QWidget()
                row = QHBoxLayout(self._custom_row)
                row.setContentsMargins(20, 0, 0, 0)
                self._path_edit = QLineEdit()
                self._path_edit.setPlaceholderText("Carpeta de imágenes o archivo de video...")
                self._path_edit.setReadOnly(True)
                btn_folder = QPushButton("📁")
                btn_folder.setFixedWidth(32)
                btn_folder.setToolTip("Seleccionar carpeta de imágenes")
                btn_folder.clicked.connect(self._browse_folder)
                btn_video = QPushButton("📹")
                btn_video.setFixedWidth(32)
                btn_video.setToolTip("Seleccionar archivo de video")
                btn_video.clicked.connect(self._browse_video)
                row.addWidget(self._path_edit)
                row.addWidget(btn_folder)
                row.addWidget(btn_video)
                group_layout.addWidget(self._custom_row)
                self._custom_row.setEnabled(False)

                # FPS row — visible only when video selected
                self._fps_row = QWidget()
                fps_layout = QHBoxLayout(self._fps_row)
                fps_layout.setContentsMargins(20, 0, 0, 2)
                fps_layout.addWidget(QLabel("Extraer:"))
                self._fps_spin = QDoubleSpinBox()
                self._fps_spin.setRange(0.5, 10.0)
                self._fps_spin.setValue(2.0)
                self._fps_spin.setSingleStep(0.5)
                self._fps_spin.setSuffix(" fps")
                self._fps_spin.setFixedWidth(90)
                fps_layout.addWidget(self._fps_spin)
                fps_layout.addWidget(QLabel("del video"))
                fps_layout.addStretch()
                group_layout.addWidget(self._fps_row)
                self._fps_row.setVisible(False)

        # ── Previous run status ───────────────────────────────────────
        group_layout.addSpacing(6)

        self._run_status = QLabel()
        self._run_status.setWordWrap(True)
        self._run_status.setStyleSheet("color: #555; font-size: 10px; padding: 0 2px;")
        group_layout.addWidget(self._run_status)

        self._load_prev_btn = QPushButton("📂  Cargar ejecución anterior")
        self._load_prev_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px 8px; background: #1f3020; "
            "color: #80c080; border: 1px solid #305030; border-radius: 3px; }"
            "QPushButton:hover { background: #2a4a2a; color: #aaffaa; }"
        )
        self._load_prev_btn.setVisible(False)
        self._load_prev_btn.clicked.connect(self.load_previous_requested)
        group_layout.addWidget(self._load_prev_btn)

        layout.addWidget(group)

        # Description
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px 4px;")
        layout.addWidget(self._desc_label)

        # Calibration widget — only for custom dataset
        self._cal_widget = CalibrationWidget()
        self._cal_widget.setVisible(False)
        layout.addWidget(self._cal_widget)

        self._btn_group.buttonToggled.connect(self._on_toggle)
        self._update_description()
        self._check_previous_run()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_toggle(self, btn, checked: bool) -> None:
        if not checked:
            return
        idx = self._btn_group.id(btn)
        _, module, _ = DATASET_OPTIONS[idx]
        is_custom = module == "custom"
        self._custom_row.setEnabled(is_custom)
        self._cal_widget.setVisible(is_custom and not self._is_video)
        self._update_description()
        self._check_previous_run()

    def _update_description(self) -> None:
        idx = self._btn_group.checkedId()
        if idx < 0:
            return
        _, module, _ = DATASET_OPTIONS[idx]
        self._desc_label.setText(DESCRIPTIONS.get(module, ""))

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if path:
            self._custom_path = Path(path)
            self._is_video    = False
            self._path_edit.setText(path)
            self._fps_row.setVisible(False)
            self._cal_widget.setVisible(True)
            self._cal_widget.set_image_folder(self._custom_path)
            self._check_previous_run()

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de video", "", VIDEO_EXTS
        )
        if path:
            self._custom_path = Path(path)
            self._is_video    = True
            self._path_edit.setText(path)
            self._fps_row.setVisible(True)
            self._cal_widget.setVisible(False)
            self._check_previous_run()

    # ------------------------------------------------------------------
    # Previous run detection
    # ------------------------------------------------------------------

    def _check_previous_run(self) -> None:
        dataset = self.get_selected_dataset()
        if dataset is None:
            self._run_status.setText("")
            self._load_prev_btn.setVisible(False)
            return

        out_dir      = Path("outputs") / dataset.name
        has_sfm     = (out_dir / "sparse").exists() and any((out_dir / "sparse").iterdir())
        has_results  = (out_dir / "results.txt").exists()

        if has_results:
            n_queries = sum(1 for _ in open(out_dir / "results.txt") if _.strip())
            self._run_status.setText(
                f"✓  Ejecución disponible — {n_queries} queries localizadas"
            )
            self._run_status.setStyleSheet("color: #7ec87e; font-size: 10px; padding: 0 2px;")
            self._load_prev_btn.setVisible(True)
        elif has_sfm:
            self._run_status.setText("⚠  Reconstrucción disponible — sin localización")
            self._run_status.setStyleSheet("color: #e0c07e; font-size: 10px; padding: 0 2px;")
            self._load_prev_btn.setVisible(False)
        else:
            self._run_status.setText("—  Sin ejecución previa")
            self._run_status.setStyleSheet("color: #555; font-size: 10px; padding: 0 2px;")
            self._load_prev_btn.setVisible(False)

    def refresh_run_status(self) -> None:
        """Call after a pipeline run finishes to update the status badge."""
        self._check_previous_run()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_selected_dataset(self):
        idx = self._btn_group.checkedId()
        if idx < 0:
            return None
        _, module, scene = DATASET_OPTIONS[idx]

        if module == "cambridge":
            from src.datasets.cambridge import CambridgeDataset
            return CambridgeDataset(scene)
        if module == "seven_scenes":
            from src.datasets.seven_scenes import SevenScenesDataset
            return SevenScenesDataset(scene)
        if module == "colmap_dem":
            from src.datasets.colmap_dem import ColmapDemDataset
            return ColmapDemDataset(scene)
        if module == "custom":
            if not self._custom_path:
                return None
            if self._is_video:
                from src.datasets.video import VideoDataset
                return VideoDataset(self._custom_path, fps=self._fps_spin.value())
            from src.datasets.custom import CustomDataset
            return CustomDataset(self._custom_path)
        return None

    def get_calibration(self) -> Optional[dict]:
        idx = self._btn_group.checkedId()
        if idx < 0:
            return None
        _, module, _ = DATASET_OPTIONS[idx]
        if module != "custom":
            return None
        return self._cal_widget.get_calibration()
