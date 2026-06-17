from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox,
    QPushButton, QFormLayout, QCheckBox, QLabel,
)
from PySide6.QtCore import Qt, Signal

# (label, extractor_key, matcher_key)
FEATURE_COMBOS = [
    ("DISK + LightGlue (recomendado)",   "disk",       "disk+lightglue"),
    ("ALIKED + LightGlue (más rápido)",  "aliked-n16", "aliked+lightglue"),
    ("SIFT + NN-ratio (sin GPU)",        "sift",       "NN-ratio"),
]

COMBO_DESCRIPTIONS = [
    "DISK es robusto ante cambios de iluminación. LightGlue es el matcher moderno más rápido y preciso.",
    "ALIKED es ~3x más rápido que DISK con calidad similar. Ideal para datasets grandes.",
    "SIFT clásico. No necesita GPU, funciona en cualquier máquina. Útil como baseline.",
]

RETRIEVAL = [
    "netvlad",
    "dir",
    "openibl",
    "exhaustive (dataset pequeño)",
]

_RETRIEVAL_DESC = {
    "netvlad":  "NetVLAD — retrieval global clásico, robusto y bien benchmarkeado.",
    "dir":      "DIRe — basado en redes con atención, buen recall en outdoor.",
    "openibl":  "OpenIBL — alternativa open-source a NetVLAD.",
    "exhaustive (dataset pequeño)": "Sin retrieval: compara todos contra todos. Ideal para < 200 imágenes o para empezar.",
}


class PipelinePanel(QWidget):
    run_requested = Signal(dict)

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
        self._combo_desc = QLabel(COMBO_DESCRIPTIONS[0])
        self._combo_desc.setWordWrap(True)
        self._combo_desc.setStyleSheet("color: #666; font-size: 10px;")
        self._combo_box.currentIndexChanged.connect(
            lambda i: self._combo_desc.setText(COMBO_DESCRIPTIONS[i])
        )
        feat_form.addRow("Combo:", self._combo_box)
        feat_form.addRow(self._combo_desc)
        layout.addWidget(feat_group)

        # Retrieval
        ret_group = QGroupBox("Image Retrieval (para pares de matching)")
        ret_form  = QFormLayout(ret_group)
        self._ret_combo = QComboBox()
        self._ret_combo.addItems(RETRIEVAL)
        self._ret_desc  = QLabel(_RETRIEVAL_DESC["netvlad"])
        self._ret_desc.setWordWrap(True)
        self._ret_desc.setStyleSheet("color: #666; font-size: 10px;")
        self._ret_combo.currentTextChanged.connect(
            lambda t: self._ret_desc.setText(_RETRIEVAL_DESC.get(t, ""))
        )
        ret_form.addRow("Método:", self._ret_combo)
        ret_form.addRow(self._ret_desc)
        layout.addWidget(ret_group)

        # Options
        opts_group  = QGroupBox("Opciones")
        opts_layout = QVBoxLayout(opts_group)
        self._skip_sfm = QCheckBox("Saltar reconstrucción SfM (solo localizar)")
        self._use_gpu  = QCheckBox("Usar GPU (CUDA)")
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

    def _emit_run(self) -> None:
        idx = self._combo_box.currentIndex()
        _, extractor, matcher = FEATURE_COMBOS[idx]
        self.run_requested.emit({
            "extractor":           extractor,
            "matcher":             matcher,
            "retrieval":           self._ret_combo.currentText(),
            "skip_reconstruction": self._skip_sfm.isChecked(),
            "use_gpu":             self._use_gpu.isChecked(),
        })

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("⏳   Ejecutando..." if running else "▶   Ejecutar pipeline")
