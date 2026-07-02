"""
CalibrationWidget — optional camera intrinsics panel for custom datasets.
Supports EXIF auto-detection (folder mode) and video metadata detection (video mode).
"""
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QClipboard


class CalibrationWidget(QGroupBox):
    def __init__(self):
        super().__init__("Calibración de cámara (opcional)")
        self._calibration:  Optional[dict] = None
        self._image_folder: Optional[Path] = None
        self._video_path:   Optional[Path] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        self._status = QLabel("Sin calibración — se usará estimación automática.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._detect_btn = QPushButton("🔍  Detectar desde EXIF")
        self._detect_btn.setEnabled(False)
        self._detect_btn.clicked.connect(self._detect)
        self._manual_btn = QPushButton("✏  Manual")
        self._manual_btn.setEnabled(False)
        self._manual_btn.setToolTip("Ingresar parámetros manualmente")
        self._manual_btn.clicked.connect(self._enable_manual_entry)
        btn_row.addWidget(self._detect_btn, 1)
        btn_row.addWidget(self._manual_btn)
        root.addLayout(btn_row)

        # ffprobe install hint — only shown when needed
        self._ffprobe_box = QGroupBox()
        self._ffprobe_box.setStyleSheet(
            "QGroupBox { border: 1px solid #5a3a1a; border-radius: 3px; "
            "background: #2a1f0f; padding: 6px; margin: 0; }"
        )
        fb_layout = QVBoxLayout(self._ffprobe_box)
        fb_layout.setSpacing(4)
        fb_layout.setContentsMargins(6, 6, 6, 6)
        fb_lbl = QLabel(
            "ffprobe no encontrado. Instalá ffmpeg para detectar\n"
            "la focal length del video automáticamente:"
        )
        fb_lbl.setWordWrap(True)
        fb_lbl.setStyleSheet("color: #cc9944; font-size: 10px;")
        fb_layout.addWidget(fb_lbl)
        cmd_row = QHBoxLayout()
        self._cmd_lbl = QLabel("winget install --id=Gyan.FFmpeg -e")
        self._cmd_lbl.setStyleSheet(
            "color: #eee; font-family: Consolas, monospace; font-size: 10px; "
            "background: #1a1a1a; padding: 3px 6px; border-radius: 2px;"
        )
        self._cmd_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        copy_btn = QPushButton("📋")
        copy_btn.setFixedSize(26, 26)
        copy_btn.setToolTip("Copiar comando")
        copy_btn.clicked.connect(self._copy_install_cmd)
        cmd_row.addWidget(self._cmd_lbl, 1)
        cmd_row.addWidget(copy_btn)
        fb_layout.addLayout(cmd_row)
        root.addWidget(self._ffprobe_box)
        self._ffprobe_box.setVisible(False)

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
        self._video_path   = None
        self._detect_btn.setText("🔍  Detectar desde EXIF")
        self._detect_btn.setEnabled(True)
        self._manual_btn.setEnabled(True)
        self._ffprobe_box.setVisible(False)
        self._detect(silent=True)

    def set_video_source(self, video_path: Path) -> None:
        self._video_path   = video_path
        self._image_folder = None
        self._detect_btn.setText("🔍  Detectar desde video")
        self._detect_btn.setEnabled(True)
        self._manual_btn.setEnabled(True)

    def set_calibration(self, cal: dict) -> None:
        self._apply_calibration(cal)

    def get_calibration(self) -> Optional[dict]:
        return self._calibration

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, *, silent: bool = False) -> None:
        self._detect_btn.setEnabled(False)
        self._detect_btn.setText("⏳  Detectando…")
        QGuiApplication.processEvents()

        try:
            if self._video_path is not None:
                self._detect_from_video(silent=silent)
            elif self._image_folder is not None:
                self._detect_from_exif(silent=silent)
        finally:
            label = "Detectar desde video" if self._video_path else "Detectar desde EXIF"
            self._detect_btn.setText(f"🔍  {label}")
            self._detect_btn.setEnabled(True)

    def _detect_from_exif(self, *, silent: bool = False) -> None:
        from src.utils.calibration import read_exif_calibration

        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}
        candidate = None
        for p in sorted(self._image_folder.rglob("*")):
            if p.suffix in exts and p.is_file():
                candidate = p
                break

        if candidate is None:
            if not silent:
                self._set_status("No se encontraron imágenes en la carpeta.", "#e07e7e")
            return

        cal = read_exif_calibration(candidate)
        if cal is None:
            if not silent:
                self._set_status(
                    f"Sin datos EXIF en {candidate.name}. Ingresá los parámetros manualmente.",
                    "#e07e7e",
                )
            return

        self._apply_calibration(cal)
        self._check_resolution_mismatch(candidate, cal)

    def _detect_from_video(self, *, silent: bool = False) -> None:
        from src.datasets.video import read_calibration_from_video, ffprobe_available

        if not ffprobe_available():
            self._ffprobe_box.setVisible(True)
            if not silent:
                self._set_status(
                    "ffprobe no encontrado — instalá ffmpeg o ingresá la focal manualmente.",
                    "#cc9944",
                )
            return

        self._ffprobe_box.setVisible(False)
        log_lines: list[str] = []

        def _log(msg: str) -> None:
            print(f"[ffprobe] {msg}")
            log_lines.append(msg)

        cal = read_calibration_from_video(self._video_path, log_fn=_log)

        if cal is None:
            if not silent:
                summary = "\n".join(log_lines[-6:]) if log_lines else "Sin output"
                self._set_status(summary, "#e07e7e")
            return

        self._apply_calibration(cal)

    def _copy_install_cmd(self) -> None:
        QGuiApplication.clipboard().setText(self._cmd_lbl.text())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, color: str) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-size: 10px;")

    def _apply_calibration(self, cal: dict) -> None:
        self._calibration = cal
        f, cx, cy, k1 = cal["params"]
        w, h = cal["width"], cal["height"]

        for sp, val in zip(
            (self._spin_f, self._spin_cx, self._spin_cy, self._spin_k1),
            (f, cx, cy, k1),
        ):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

        src = cal.get("source", "desconocido")
        self._set_status(
            f"✓ {src}  |  {w}×{h}px\n"
            f"f={f:.1f}px  cx={cx:.1f}  cy={cy:.1f}  k1={k1:.4f}",
            "#7ec87e",
        )
        self._set_fields_enabled(True)

    def _enable_manual_entry(self) -> None:
        w, h = 0, 0
        if self._video_path is not None:
            try:
                import cv2
                cap = cv2.VideoCapture(str(self._video_path))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception:
                pass
        elif self._image_folder is not None:
            exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}
            for p in sorted(self._image_folder.rglob("*")):
                if p.suffix in exts and p.is_file():
                    try:
                        from PIL import Image as _PIL
                        with _PIL.open(p) as im:
                            w, h = im.size
                    except Exception:
                        pass
                    break

        if self._calibration is None:
            self._calibration = {
                "model": "SIMPLE_RADIAL",
                "width": w, "height": h,
                "params": [0.0, w / 2.0, h / 2.0, 0.0],
                "source": "manual",
            }
            for sp, val in zip(
                (self._spin_f, self._spin_cx, self._spin_cy, self._spin_k1),
                (0.0, w / 2.0, h / 2.0, 0.0),
            ):
                sp.blockSignals(True)
                sp.setValue(val)
                sp.blockSignals(False)

        res_str = f"  |  {w}×{h}px" if w and h else ""
        self._set_status(
            f"Ingresá los parámetros manualmente{res_str}.\n"
            f"cx={w//2}  cy={h//2}  (centro de imagen)",
            "#e0c07e",
        )
        self._set_fields_enabled(True)

    def _check_resolution_mismatch(self, img_path: Path, cal: dict) -> None:
        try:
            from PIL import Image as _PIL
            with _PIL.open(img_path) as im:
                actual_w, actual_h = im.size
        except Exception:
            return
        exif_w, exif_h = cal["width"], cal["height"]
        if actual_w == exif_w and actual_h == exif_h:
            return
        f = cal["params"][0]
        scaled_f = f * actual_w / exif_w
        self._set_status(
            f"⚠ EXIF dice {exif_w}×{exif_h} pero las imágenes son {actual_w}×{actual_h}.\n"
            f"Si son frames de video la focal EXIF ({f:.0f}px) es incorrecta → "
            f"f correcta ≈ {scaled_f:.0f}px.\n"
            f"Usá '✕ Usar AUTO' para que COLMAP la estime, o ajustá manualmente.",
            "#e0a040",
        )

    def _on_manual_edit(self) -> None:
        if self._calibration is None:
            self._calibration = {
                "model": "SIMPLE_RADIAL",
                "width": 0, "height": 0,
                "params": [0.0, 0.0, 0.0, 0.0],
                "source": "manual",
            }

        self._calibration["params"] = [
            self._spin_f.value(),
            self._spin_cx.value(),
            self._spin_cy.value(),
            self._spin_k1.value(),
        ]
        self._calibration["source"] = "manual"
        self._set_status("Parámetros manuales configurados.", "#e0c07e")

    def _clear(self) -> None:
        self._calibration = None
        self._set_fields_enabled(False)
        self._set_status("Sin calibración — se usará estimación automática.", "#888")
        for sp in (self._spin_f, self._spin_cx, self._spin_cy, self._spin_k1):
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
