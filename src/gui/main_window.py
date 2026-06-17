from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QTabWidget, QStatusBar, QMessageBox,
)
from PySide6.QtCore import Qt

from .dataset_panel import DatasetPanel
from .pipeline_panel import PipelinePanel
from .progress_panel import ProgressPanel
from .viewer_panel import ViewerPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP COLMAP — Visual Localization")
        self.setMinimumSize(1100, 680)
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

        # ── Left panel: tabs for dataset / pipeline config ─────────────
        self._tabs = QTabWidget()
        self._tabs.setMaximumWidth(360)
        self._tabs.setMinimumWidth(300)

        self.dataset_panel  = DatasetPanel()
        self.pipeline_panel = PipelinePanel()
        self._tabs.addTab(self.dataset_panel,  "📁  Dataset")
        self._tabs.addTab(self.pipeline_panel, "⚙️  Pipeline")

        # ── Right panel: progress (top) + 3D viewer (bottom) ───────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.progress_panel = ProgressPanel()
        self.viewer_panel   = ViewerPanel()

        right_splitter.addWidget(self.progress_panel)
        right_splitter.addWidget(self.viewer_panel)
        right_splitter.setSizes([220, 460])
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)

        root.addWidget(self._tabs)
        root.addWidget(right_splitter, 1)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo  —  Seleccioná un dataset y presioná Ejecutar.")

        # Connect signals
        self.pipeline_panel.run_requested.connect(self._on_run)

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #252526; color: #ccc; }
            QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #2d2d30; color: #aaa; padding: 6px 14px; }
            QTabBar::tab:selected { background: #3e3e42; color: white; }
            QComboBox, QLineEdit { background: #3c3c3c; border: 1px solid #555; border-radius: 3px; padding: 3px; color: #ddd; }
            QComboBox::drop-down { border: none; }
            QCheckBox { color: #ccc; }
            QRadioButton { color: #ccc; }
            QLabel { color: #ccc; }
            QStatusBar { color: #888; font-size: 10px; }
            QSplitter::handle { background: #444; }
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

        # Stop any previous run
        if self._runner and self._runner.isRunning():
            self._runner.terminate()
            self._runner.wait()

        self.pipeline_panel.set_running(True)
        self.progress_panel.clear()
        self.statusBar().showMessage(f"Ejecutando pipeline sobre: {dataset.name}")

        self._runner = PipelineRunner(dataset, config)
        self._runner.log_message.connect(self.progress_panel.append_log)
        self._runner.progress_updated.connect(self.progress_panel.set_progress)
        self._runner.stage_changed.connect(self.progress_panel.set_stage)
        self._runner.reconstruction_ready.connect(self.viewer_panel.load_model)
        self._runner.finished.connect(self._on_finished)
        self._runner.start()

    def _on_finished(self, success: bool) -> None:
        self.pipeline_panel.set_running(False)
        if success:
            self.statusBar().showMessage("Pipeline completado con éxito.")
            self._tabs.setCurrentIndex(1)  # Switch to pipeline tab to see results
        else:
            self.statusBar().showMessage("Pipeline falló — revisá los logs.")

    def closeEvent(self, event) -> None:
        if self._runner and self._runner.isRunning():
            self._runner.terminate()
            self._runner.wait()
        super().closeEvent(event)
