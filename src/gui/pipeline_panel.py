from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox,
    QPushButton, QFormLayout, QCheckBox, QLabel, QSpinBox,
)
from PySide6.QtCore import Qt, Signal

# (label, extractor_key, matcher_key)
FEATURE_COMBOS = [
    ("DISK + LightGlue (recomendado)",     "disk",              "disk+lightglue"),
    ("SuperPoint + LightGlue (benchmark)", "superpoint_aachen", "superpoint+lightglue"),
    ("ALIKED + LightGlue (más rápido)",    "aliked-n16",        "aliked+lightglue"),
    ("SIFT + NN-ratio (sin GPU)",          "sift",              "NN-ratio"),
]

COMBO_DESCRIPTIONS = ["", "", "", ""]

RETRIEVAL = [
    "netvlad",
    "megaloc",
    "openibl",
    "exhaustive (dataset pequeño)",
]

_RETRIEVAL_DESC: dict = {}


class PipelinePanel(QWidget):
    run_requested   = Signal(dict)
    clear_requested = Signal()

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        # Feature combo (extractor + matcher juntos para evitar combos inválidas)
        feat_group = QGroupBox("Features y Matching")
        feat_form  = QFormLayout(feat_group)
        self._combo_box = QComboBox()
        self._combo_box.addItems([c[0] for c in FEATURE_COMBOS])
        feat_form.addRow("Combo:", self._combo_box)
        layout.addWidget(feat_group)

        # Retrieval
        ret_group = QGroupBox("Image Retrieval (para pares de matching)")
        ret_form  = QFormLayout(ret_group)
        self._ret_combo = QComboBox()
        self._ret_combo.addItems(RETRIEVAL)
        ret_form.addRow("Método:", self._ret_combo)
        layout.addWidget(ret_group)

        # Options
        opts_group  = QGroupBox("Opciones")
        opts_layout = QVBoxLayout(opts_group)
        opts_form   = QFormLayout()
        opts_form.setSpacing(4)

        self._num_matched = QSpinBox()
        self._num_matched.setRange(5, 100)
        self._num_matched.setValue(20)
        self._num_matched.setSuffix(" imgs")
        self._num_matched.setToolTip("")
        opts_form.addRow("Retrieval top-N:", self._num_matched)
        opts_layout.addLayout(opts_form)

        self._skip_sfm  = QCheckBox("Saltar reconstrucción SfM (solo localizar)")
        self._fast_sfm  = QCheckBox()  # hidden, kept for API compat
        self._use_gpu   = QCheckBox("Usar GPU (CUDA)")
        self._use_gpu.setChecked(True)
        opts_layout.addWidget(self._skip_sfm)
        opts_layout.addWidget(self._use_gpu)
        layout.addWidget(opts_group)

        # Run button
        self._run_btn = QPushButton("▶   Ejecutar pipeline")
        self._run_btn.setStyleSheet(
            "QPushButton { font-size: 13px; padding: 9px; background: #0078d4; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #106ebe; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        self._run_btn.clicked.connect(self._emit_run)
        layout.addWidget(self._run_btn)

        # Clear button
        self._clear_btn = QPushButton("🗑   Limpiar ejecución anterior")
        self._clear_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 5px; background: #3c1f1f; color: #e08080; border: 1px solid #6b3030; border-radius: 4px; }"
            "QPushButton:hover { background: #5a2a2a; color: #ffaaaa; }"
            "QPushButton:disabled { background: #333; color: #666; border-color: #444; }"
        )
        self._clear_btn.clicked.connect(self.clear_requested)
        layout.addWidget(self._clear_btn)

    def _emit_run(self) -> None:
        idx = self._combo_box.currentIndex()
        _, extractor, matcher = FEATURE_COMBOS[idx]
        self.run_requested.emit({
            "extractor":           extractor,
            "matcher":             matcher,
            "retrieval":           self._ret_combo.currentText(),
            "num_matched":         self._num_matched.value(),
            "skip_reconstruction": self._skip_sfm.isChecked(),
            "fast_reconstruction": self._fast_sfm.isChecked(),
            "use_gpu":             self._use_gpu.isChecked(),
        })

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._clear_btn.setEnabled(not running)
        self._run_btn.setText("⏳   Ejecutando..." if running else "▶   Ejecutar pipeline")
