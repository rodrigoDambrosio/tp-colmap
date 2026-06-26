import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QScrollArea, QStatusBar, QMessageBox,
)
from PySide6.QtCore import Qt

from .dataset_panel import DatasetPanel
from .pipeline_panel import PipelinePanel
from .progress_panel import ProgressPanel
from .localization_panel import LocalizationPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP COLMAP — Visual Localization")
        self.setMinimumSize(1200, 700)
        self._runner = None
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left: dataset + pipeline config in a single scroll area ────
        self.dataset_panel  = DatasetPanel()
        self.pipeline_panel = PipelinePanel()

        left_container = QWidget()
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        inner = QWidget()
        inner_layout = QHBoxLayout(inner)  # horizontal so scroll width works
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(8)

        stack = QWidget()
        from PySide6.QtWidgets import QVBoxLayout
        stack_layout = QVBoxLayout(stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(12)
        stack_layout.addWidget(self.dataset_panel)
        stack_layout.addWidget(self.pipeline_panel)
        stack_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(stack)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        left_layout.addWidget(scroll)

        # ── Right: progress (top) + unified 3D viewer (bottom) ────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.progress_panel = ProgressPanel()
        self.loc_panel      = LocalizationPanel()

        right_splitter.addWidget(self.progress_panel)
        right_splitter.addWidget(self.loc_panel)
        right_splitter.setSizes([220, 480])
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)

        root.addWidget(left_container)
        root.addWidget(right_splitter, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo  —  Seleccioná un dataset y presioná Ejecutar.")

        self.pipeline_panel.run_requested.connect(self._on_run)
        self.pipeline_panel.clear_requested.connect(self._on_clear)
        self.dataset_panel.load_previous_requested.connect(self._on_load)

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #252526; color: #ccc; }
            QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
            QScrollArea { background: transparent; }
            QScrollBar:vertical { background: #2a2a2a; width: 8px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QComboBox, QLineEdit { background: #3c3c3c; border: 1px solid #555; border-radius: 3px; padding: 3px; color: #ddd; }
            QComboBox::drop-down { border: none; }
            QCheckBox { color: #ccc; }
            QLabel { color: #ccc; }
            QStatusBar { color: #888; font-size: 10px; }
            QSplitter::handle { background: #444; }
            QPushButton { background: #3c3c3c; border: 1px solid #555; border-radius: 3px; padding: 4px 8px; color: #ddd; }
            QPushButton:hover { background: #4a4a4a; }
        """)

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def _on_run(self, config: dict) -> None:
        from src.pipeline.runner import PipelineRunner

        dataset = self.dataset_panel.get_selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "Dataset", "Seleccioná un dataset o ruta de imágenes primero.")
            return

        if self._runner and self._runner.isRunning():
            self._runner.terminate()
            self._runner.wait()

        self.pipeline_panel.set_running(True)
        self.progress_panel.clear()
        self.statusBar().showMessage(f"Ejecutando pipeline sobre: {dataset.name}")

        config["calibration"] = self.dataset_panel.get_calibration()
        self._runner = PipelineRunner(dataset, config)
        self._runner.log_message.connect(self.progress_panel.append_log)
        self._runner.progress_updated.connect(self.progress_panel.set_progress)
        self._runner.stage_changed.connect(self.progress_panel.set_stage)
        self._runner.reconstruction_ready.connect(self._on_reconstruction_ready)
        self._runner.localization_started.connect(self._on_localization_started)
        self._runner.query_localized.connect(self.loc_panel.add_query_result)
        self._runner.localization_ready.connect(self._on_localization_ready)
        self._runner.finished.connect(self._on_finished)
        self._runner.start()

    def _on_reconstruction_ready(self, sfm_dir: str, images_dir: str) -> None:
        self.loc_panel.setup_localization(sfm_dir, images_dir)

    def _on_localization_started(self, sfm_dir: str, images_dir: str) -> None:
        self.loc_panel.begin_localization()

    def _on_localization_ready(self, sfm_dir: str, images_dir: str, results_path: str) -> None:
        pass

    def _on_finished(self, success: bool) -> None:
        self.pipeline_panel.set_running(False)
        self.dataset_panel.refresh_run_status()
        if success:
            self.statusBar().showMessage("Pipeline completado con éxito.")
        else:
            self.statusBar().showMessage("Pipeline falló — revisá los logs.")

    def _on_load(self) -> None:
        dataset = self.dataset_panel.get_selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "Cargar", "Seleccioná un dataset primero.")
            return
        out_dir      = Path("outputs") / dataset.name
        sfm_dir      = out_dir / "sparse"
        results_path = out_dir / "results.txt"
        images_dir   = dataset.get_images_dir()

        if not sfm_dir.exists():
            QMessageBox.information(self, "Cargar", f"No hay reconstrucción guardada para {dataset.name}.")
            return
        if not results_path.exists():
            QMessageBox.information(self, "Cargar", f"No hay resultados de localización para {dataset.name}.")
            return

        self.loc_panel.load_localization(str(sfm_dir), str(images_dir), str(results_path))
        self.statusBar().showMessage(f"Ejecución anterior cargada: {dataset.name}")

    def _on_clear(self) -> None:
        dataset = self.dataset_panel.get_selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "Limpiar", "Seleccioná un dataset primero.")
            return
        out_dir = Path("outputs") / dataset.name
        if not out_dir.exists():
            QMessageBox.information(self, "Limpiar", f"No hay ejecución previa para {dataset.name}.")
            return
        reply = QMessageBox.question(
            self, "Confirmar limpieza",
            f"¿Borrar todo en:\n{out_dir.resolve()}?\n\nEsto elimina features, matches y la reconstrucción.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        shutil.rmtree(out_dir)
        self.progress_panel.clear()
        self.dataset_panel.refresh_run_status()
        self.statusBar().showMessage(f"Limpiado: {out_dir}")

    def closeEvent(self, event) -> None:
        if self._runner and self._runner.isRunning():
            self._runner.terminate()
            self._runner.wait()
        super().closeEvent(event)
