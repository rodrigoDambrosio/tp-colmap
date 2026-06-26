"""
CalibrationWidget — optional camera intrinsics panel for custom datasets.
Supports EXIF auto-detection and manual entry.
"""
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt


class CalibrationWidget(QGroupBox):
    """
    Shows estimated or manually entered camera intrinsics.
    Call set_image_folder() when the user picks a dataset folder.
    Call get_calibration() to retrieve the current params (None = use AUTO).
    """

    def __init__(self):
        super().__init__("Calibración de cámara (opcional)")
        self._calibration: Optional[dict] = None
        self._image_folder: Optional[Path] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # Status / source label
        self._status = QLabel("Sin calibración — se usará estimación automática.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self._status)

        # EXIF button
        self._exif_btn = QPushButton("🔍  Detectar desde EXIF")
        self._exif_btn.setEnabled(False)
        self._exif_btn.clicked.connect(self._detect_exif)
        root.addWidget(self._exif_btn)

        # Parameter fields
        form = QFormLayout()
        form.setSpacing(4)

        self._spin_f  = self._make_spin(1.0, 99999.0, 2, "px")
        self._spin_cx = self._make_spin(1.0, 99999.0, 1, "px")
        self._spin_cy = self._make_spin(1.0, 99999.0, 1, "px")
        self._spin_k1 = self._make_spin(-1.0, 1.0,   4, "")

        form.addRow("Focal (f):", self._spin_f)
        form.addRow("cx:", self._spin_cx)
        form.addRow("cy:", self._spin_cy)
        form.addRow("Distorsión k1:", self._spin_k1)
        root.addLayout(form)

        for sp in (self._spin_f, self._spin_cx, self._spin_cy, self._spin_k1):
            sp.valueChanged.connect(self._on_manual_edit)

        # Clear button
        clear_row = QHBoxLayout()
        self._clear_btn = QPushButton("✕  Usar AUTO")
        self._clear_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px 6px; color: #aaa; "
            "background: #2a2a2a; border: 1px solid #444; border-radius: 3px; }"
            "QPushButton:hover { color: #fff; }"
        )
        self._clear_btn.clicked.connect(self._clear)
        clear_row.addStretch()
        clear_row.addWidget(self._clear_btn)
        root.addLayout(clear_row)

        self._set_fields_enabled(False)

    def _make_spin(self, lo, hi, decimals, suffix) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSuffix(f" {suffix}" if suffix else "")
        sp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return sp

    def _set_fields_enabled(self, enabled: bool) -> None:
        for w in (self._spin_f, self._spin_cx, self._spin_cy, self._spin_k1):
            w.setEnabled(enabled)
        self._clear_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image_folder(self, folder: Path) -> None:
        self._image_folder = folder
        self._exif_btn.setEnabled(True)
        # Auto-attempt EXIF on first image found
        self._detect_exif(silent=True)

    def get_calibration(self) -> Optional[dict]:
        """Returns calibration dict or None (→ pipeline uses AUTO)."""
        return self._calibration

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _detect_exif(self, *, silent: bool = False) -> None:
        if self._image_folder is None:
            return
        from src.utils.calibration import read_exif_calibration

        # Find first image in the folder
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}
        candidate = None
        for p in sorted(self._image_folder.rglob("*")):
            if p.suffix in exts and p.is_file():
                candidate = p
                break

        if candidate is None:
            if not silent:
                self._status.setText("No se encontraron imágenes en la carpeta.")
            return

        cal = read_exif_calibration(candidate)
        if cal is None:
            if not silent:
                self._status.setText(
                    f"No se encontró información EXIF en:\n{candidate.name}\n"
                    "Ingresá los parámetros manualmente."
                )
            return

        self._apply_calibration(cal)

    def _apply_calibration(self, cal: dict) -> None:
        self._calibration = cal
        f, cx, cy, k1 = cal["params"]
        w, h = cal["width"], cal["height"]

        self._spin_f.blockSignals(True)
        self._spin_cx.blockSignals(True)
        self._spin_cy.blockSignals(True)
        self._spin_k1.blockSignals(True)

        self._spin_f.setValue(f)
        self._spin_cx.setValue(cx)
        self._spin_cy.setValue(cy)
        self._spin_k1.setValue(k1)

        self._spin_f.blockSignals(False)
        self._spin_cx.blockSignals(False)
        self._spin_cy.blockSignals(False)
        self._spin_k1.blockSignals(False)

        src = cal.get("source", "desconocido")
        self._status.setText(
            f"✓ Fuente: {src}  |  {w}×{h}px\n"
            f"f={f:.1f}px  cx={cx:.1f}  cy={cy:.1f}  k1={k1:.4f}"
        )
        self._status.setStyleSheet("color: #7ec87e; font-size: 10px;")
        self._set_fields_enabled(True)

    def _on_manual_edit(self) -> None:
        if self._calibration is None:
            # User started typing manually — create a calibration dict
            self._calibration = {
                "model": "SIMPLE_RADIAL",
                "width": 0, "height": 0,
                "params": [0.0, 0.0, 0.0, 0.0],
                "source": "manual",
            }
            self._set_fields_enabled(True)

        self._calibration["params"] = [
            self._spin_f.value(),
            self._spin_cx.value(),
            self._spin_cy.value(),
            self._spin_k1.value(),
        ]
        self._calibration["source"] = "manual"
        self._status.setText("Parámetros manuales configurados.")
        self._status.setStyleSheet("color: #e0c07e; font-size: 10px;")

    def _clear(self) -> None:
        self._calibration = None
        self._set_fields_enabled(False)
        self._status.setText("Sin calibración — se usará estimación automática.")
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        for sp in (self._spin_f, self._spin_cx, self._spin_cy, self._spin_k1):
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
