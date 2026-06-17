from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QRadioButton, QButtonGroup,
    QLabel, QPushButton, QFileDialog, QHBoxLayout, QLineEdit,
)
from PySide6.QtCore import Qt

# (display label, module, scene/key)
DATASET_OPTIONS = [
    ("Cambridge — KingsCollege",  "cambridge",    "KingsCollege"),
    ("Cambridge — ShopFacade",    "cambridge",    "ShopFacade"),
    ("Cambridge — OldHospital",   "cambridge",    "OldHospital"),
    ("7-Scenes — Chess",          "seven_scenes", "chess"),
    ("7-Scenes — Fire",           "seven_scenes", "fire"),
    ("7-Scenes — Office",         "seven_scenes", "office"),
    ("Dataset propio",            "custom",       None),
]

DESCRIPTIONS = {
    "cambridge":    "Localización outdoor. Auto-descarga ~250 MB. Tiene ground truth.",
    "seven_scenes": "Localización indoor RGB-D. Auto-descarga ~0.3-3 GB. Sin GT en formato COLMAP.",
    "custom":       "Usá tus propias fotos. Se divide 80% DB / 20% query automáticamente.",
}


class DatasetPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._custom_path: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        group = QGroupBox("Seleccionar dataset")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        self._btn_group = QButtonGroup(self)

        for i, (label, module, _) in enumerate(DATASET_OPTIONS):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            self._btn_group.addButton(rb, i)
            group_layout.addWidget(rb)

            if module == "custom":
                self._custom_row = QWidget()
                row = QHBoxLayout(self._custom_row)
                row.setContentsMargins(20, 0, 0, 0)
                self._path_edit = QLineEdit()
                self._path_edit.setPlaceholderText("Ruta a carpeta de imágenes...")
                self._path_edit.setReadOnly(True)
                btn = QPushButton("...")
                btn.setFixedWidth(32)
                btn.clicked.connect(self._browse)
                row.addWidget(self._path_edit)
                row.addWidget(btn)
                group_layout.addWidget(self._custom_row)
                self._custom_row.setEnabled(False)

        layout.addWidget(group)

        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #777; font-size: 11px; padding: 4px;")
        layout.addWidget(self._desc_label)

        self._btn_group.buttonToggled.connect(self._on_toggle)
        self._update_description()

    def _on_toggle(self, btn, checked: bool) -> None:
        if not checked:
            return
        idx = self._btn_group.id(btn)
        _, module, _ = DATASET_OPTIONS[idx]
        self._custom_row.setEnabled(module == "custom")
        self._update_description()

    def _update_description(self) -> None:
        idx = self._btn_group.checkedId()
        if idx < 0:
            return
        _, module, _ = DATASET_OPTIONS[idx]
        self._desc_label.setText(DESCRIPTIONS.get(module, ""))

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if path:
            self._custom_path = Path(path)
            self._path_edit.setText(path)

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
        if module == "custom":
            if not self._custom_path:
                return None
            from src.datasets.custom import CustomDataset
            return CustomDataset(self._custom_path)
        return None
