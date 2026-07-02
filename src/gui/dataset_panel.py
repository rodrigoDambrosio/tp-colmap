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
    ("COLMAP — South Building",   "colmap_dem",   "south-building"),
    ("COLMAP — Gerrard Hall",     "colmap_dem",   "gerrard-hall"),
    ("COLMAP — Person Hall",      "colmap_dem",   "person-hall"),
    ("COLMAP — Graham Hall",      "colmap_dem",   "graham-hall"),
    ("Dataset propio",            "custom",       None),
]

DESCRIPTIONS: dict = {}

_RB_STYLE = """
QRadioButton {
    color: #999;
    spacing: 8px;
    padding: 2px 0;
}
QRadioButton:checked {
    color: #ffffff;
    font-weight: 600;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #6a6a6a;
    background: #333333;
}
QRadioButton::indicator:checked {
    border: 3px solid #0078d4;
    background: #0078d4;
}
QRadioButton::indicator:hover {
    border-color: #aaa;
    background: #3c3c3c;
}
"""


class DatasetPanel(QWidget):
    load_previous_requested = Signal()
    browse_runs_requested   = Signal()  # MainWindow opens the dialog

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
                crow = QVBoxLayout(self._custom_row)
                crow.setContentsMargins(0, 2, 0, 2)
                crow.setSpacing(4)

                self._path_edit = QLineEdit()
                self._path_edit.setPlaceholderText("Carpeta o video seleccionado...")
                self._path_edit.setReadOnly(True)
                crow.addWidget(self._path_edit)

                btn_row = QHBoxLayout()
                btn_row.setSpacing(6)
                btn_folder = QPushButton("📁  Carpeta de imágenes")
                btn_video  = QPushButton("📹  Archivo de video")
                btn_folder.clicked.connect(self._browse_folder)
                btn_video.clicked.connect(self._browse_video)
                btn_row.addWidget(btn_folder)
                btn_row.addWidget(btn_video)
                crow.addLayout(btn_row)

                # Run name field
                name_row = QHBoxLayout()
                name_row.setSpacing(4)
                name_row.addWidget(QLabel("Nombre:"))
                self._name_edit = QLineEdit()
                self._name_edit.setPlaceholderText("nombre_del_run (auto)")
                self._name_edit.setToolTip(
                    "Nombre de la carpeta en outputs/. Podés reusar el mismo nombre "
                    "para continuar una ejecución anterior."
                )
                name_row.addWidget(self._name_edit, 1)
                crow.addLayout(name_row)

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

        self._browse_runs_btn = QPushButton("🗂  Explorar runs anteriores")
        self._browse_runs_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px 6px; color: #aaa; "
            "background: #2a2a2a; border: 1px solid #444; border-radius: 3px; }"
            "QPushButton:hover { color: #fff; background: #353535; }"
        )
        self._browse_runs_btn.clicked.connect(self._open_past_runs)
        group_layout.addWidget(self._browse_runs_btn)

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

        # Calibration widget — only for custom dataset
        self._cal_widget = CalibrationWidget()
        self._cal_widget.setVisible(False)
        layout.addWidget(self._cal_widget)

        self._btn_group.buttonToggled.connect(self._on_toggle)
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
        # Cal widget only shows once a folder is actually selected
        self._cal_widget.setVisible(
            is_custom and not self._is_video and self._custom_path is not None
        )
        self._check_previous_run()

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if path:
            self._custom_path = Path(path)
            self._is_video    = False
            self._path_edit.setText(path)
            self._fps_row.setVisible(False)
            if not self._name_edit.text().strip():
                self._name_edit.setText(f"custom_{self._custom_path.name}")
            self._cal_widget.setVisible(True)
            self._cal_widget.set_image_folder(self._custom_path)
            self._check_previous_run()

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de video", "", VIDEO_EXTS
        )
        if not path:
            return
        self._custom_path = Path(path)
        self._is_video    = True
        self._path_edit.setText(path)
        self._fps_row.setVisible(True)
        if not self._name_edit.text().strip():
            self._name_edit.setText(f"video_{self._custom_path.stem}")
        self._cal_widget.set_video_source(self._custom_path)
        self._cal_widget._detect(silent=True)
        self._cal_widget.setVisible(True)
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

    def _open_past_runs(self) -> None:
        self.browse_runs_requested.emit()

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
        if module == "colmap_dem":
            from src.datasets.colmap_dem import ColmapDemDataset
            return ColmapDemDataset(scene)
        if module == "custom":
            if not self._custom_path:
                return None
            run_name = self._name_edit.text().strip()
            if self._is_video:
                from src.datasets.video import VideoDataset
                return VideoDataset(self._custom_path, fps=self._fps_spin.value(), run_name=run_name)
            from src.datasets.custom import CustomDataset
            return CustomDataset(self._custom_path, run_name=run_name)
        return None

    def get_calibration(self) -> Optional[dict]:
        idx = self._btn_group.checkedId()
        if idx < 0:
            return None
        _, module, _ = DATASET_OPTIONS[idx]
        if module != "custom":
            return None
        return self._cal_widget.get_calibration()
