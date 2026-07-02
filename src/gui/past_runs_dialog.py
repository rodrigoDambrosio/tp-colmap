"""
PastRunsDialog — browse and load previous pipeline runs from outputs/.
"""
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

OUTPUT_DIR = Path("outputs")


def _scan_runs() -> list[dict]:
    """Return metadata for every run folder found under outputs/."""
    runs = []
    if not OUTPUT_DIR.exists():
        return runs
    for d in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        has_sfm     = (d / "sparse").exists() and any((d / "sparse").iterdir())
        has_results = (d / "results.txt").exists()
        if not has_sfm and not has_results:
            continue
        n_queries = 0
        if has_results:
            try:
                n_queries = sum(1 for _ in open(d / "results.txt") if _.strip())
            except Exception:
                pass
        images_dir = ""
        idf = d / "images_dir.txt"
        if idf.exists():
            p = Path(idf.read_text().strip())
            images_dir = str(p) if p.exists() else f"⚠ {p.name} (no encontrado)"

        runs.append({
            "name":        d.name,
            "path":        d,
            "has_sfm":     has_sfm,
            "has_results": has_results,
            "n_queries":   n_queries,
            "images_dir":  images_dir,
            "disabled":    (d / ".hidden").exists(),
        })
    return runs


class PastRunsDialog(QDialog):
    load_requested = Signal(str, str, str)  # sfm_dir, images_dir, results_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Runs anteriores")
        self.setMinimumSize(480, 360)
        self._runs = _scan_runs()
        # Filled when user clicks "Cargar"; read by caller after exec() returns
        self.result_sfm_dir    = ""
        self.result_images_dir = ""
        self.result_results    = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        if not self._runs:
            root.addWidget(QLabel("No hay ejecuciones guardadas en outputs/."))
            close = QPushButton("Cerrar")
            close.clicked.connect(self.reject)
            root.addWidget(close)
            return

        hint = QLabel("Seleccioná un run para cargarlo en el visualizador:")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(hint)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background: #1e1e1e; border: 1px solid #444; }"
            "QListWidget::item { padding: 6px 4px; border-bottom: 1px solid #2a2a2a; }"
            "QListWidget::item:selected { background: #0e3a5a; }"
        )
        for run in self._runs:
            if run["disabled"]:
                item = QListWidgetItem(f"  {run['name']}\n  — desactivado")
                item.setForeground(QColor("#555"))
            elif run["has_results"]:
                item = QListWidgetItem(f"  {run['name']}\n  ✓  {run['n_queries']} queries localizadas")
                item.setForeground(QColor("#7ec87e"))
            else:
                item = QListWidgetItem(f"  {run['name']}\n  ⚠  Solo reconstrucción (sin localización)")
                item.setForeground(QColor("#e0c07e"))
            self._list.addItem(item)
        self._list.currentRowChanged.connect(self._on_select)
        root.addWidget(self._list, 1)

        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        root.addWidget(self._detail)

        btn_row = QHBoxLayout()
        self._folder_btn = QPushButton("📁  Especificar carpeta de imágenes")
        self._folder_btn.setEnabled(False)
        self._folder_btn.setStyleSheet("font-size: 10px; padding: 3px 6px;")
        self._folder_btn.clicked.connect(self._pick_images_dir)
        btn_row.addWidget(self._folder_btn)

        self._toggle_btn = QPushButton("⏸  Desactivar")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px 8px; color: #aaa; "
            "background: #2a2a2a; border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { color: #fff; background: #383838; }"
            "QPushButton:disabled { color: #444; background: #222; border-color: #333; }"
        )
        self._toggle_btn.clicked.connect(self._toggle_run)
        btn_row.addWidget(self._toggle_btn)

        btn_row.addStretch()
        self._load_btn = QPushButton("📂  Cargar run seleccionado")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._load)
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

        if self._runs:
            self._list.setCurrentRow(0)

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self._runs):
            self._load_btn.setEnabled(False)
            self._folder_btn.setEnabled(False)
            return
        run = self._runs[row]
        parts = [f"Carpeta: outputs/{run['name']}"]
        if run["has_sfm"]:
            parts.append("Reconstrucción SfM: ✓")
        if run["has_results"]:
            parts.append(f"Localización: ✓ ({run['n_queries']} queries)")
        if run["images_dir"]:
            parts.append(f"Imágenes: {run['images_dir']}")
        else:
            parts.append("⚠ Carpeta de imágenes no guardada — usá '📁 Especificar' para verlas")
        self._detail.setText("\n".join(parts))
        disabled = run["disabled"]
        self._load_btn.setEnabled(run["has_sfm"] and not disabled)
        self._folder_btn.setEnabled(run["has_sfm"] and not disabled)
        self._toggle_btn.setEnabled(True)
        self._toggle_btn.setText("▶  Activar" if disabled else "⏸  Desactivar")

    def _pick_images_dir(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        row = self._list.currentRow()
        if row < 0:
            return
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if folder:
            self._runs[row]["images_dir"] = folder
            (self._runs[row]["path"] / "images_dir.txt").write_text(folder)
            self._on_select(row)

    def _toggle_run(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        run = self._runs[row]
        marker = run["path"] / ".hidden"
        if run["disabled"]:
            marker.unlink(missing_ok=True)
            run["disabled"] = False
            if run["has_results"]:
                self._list.item(row).setText(f"  {run['name']}\n  ✓  {run['n_queries']} queries localizadas")
                self._list.item(row).setForeground(QColor("#7ec87e"))
            else:
                self._list.item(row).setText(f"  {run['name']}\n  ⚠  Solo reconstrucción (sin localización)")
                self._list.item(row).setForeground(QColor("#e0c07e"))
        else:
            marker.touch()
            run["disabled"] = True
            self._list.item(row).setText(f"  {run['name']}\n  — desactivado")
            self._list.item(row).setForeground(QColor("#555"))
        self._on_select(row)

    def _load(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        run = self._runs[row]
        self.result_sfm_dir    = str(run["path"] / "sparse")
        self.result_results    = str(run["path"] / "results.txt") if run["has_results"] else ""
        self.result_images_dir = run["images_dir"] or self._find_images_dir(run["path"])
        self.accept()  # close dialog; caller reads result_* attributes

    def _find_images_dir(self, run_path: Path) -> str:
        # Prefer the saved path from the pipeline run
        saved = run_path / "images_dir.txt"
        if saved.exists():
            p = Path(saved.read_text().strip())
            if p.exists():
                return str(p)
        return ""
