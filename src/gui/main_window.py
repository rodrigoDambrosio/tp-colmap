from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QTabWidget, QStatusBar, QMessageBox,
)
from PySide6.QtCore import Qt

from .dataset_panel import DatasetPanel
from .pipeline_panel import PipelinePanel
from .progress_panel import ProgressPanel
from .viewer_panel import ViewerPanel
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

        # ── Left panel: dataset / pipeline config ──────────────────────
        self._left_tabs = QTabWidget()
        self._left_tabs.setMaximumWidth(360)
        self._left_tabs.setMinimumWidth(300)

        self.dataset_panel  = DatasetPanel()
        self.pipeline_panel = PipelinePanel()
        self._left_tabs.addTab(self.dataset_panel,  "📁  Dataset")
        self._left_tabs.addTab(self.pipeline_panel, "⚙️  Pipeline")

        # ── Right panel: progress (top) + viewer tabs (bottom) ─────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.progress_panel = ProgressPanel()

        # Viewer tabs: reconstruction 3D | localization
        self._view_tabs = QTabWidget()
        self._view_tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.viewer_panel = ViewerPanel()
        self.loc_panel    = LocalizationPanel()

        self._view_tabs.addTab(self.viewer_panel, "🗺  Reconstrucción 3D")
        self._view_tabs.addTab(self.loc_panel,    "📍  Localización")

        right_splitter.addWidget(self.progress_panel)
        right_splitter.addWidget(self._view_tabs)
        right_splitter.setSizes([220, 480])
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)

        root.addWidget(self._left_tabs)
        root.addWidget(right_splitter, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo  —  Seleccioná un dataset y presioná Ejecutar.")

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

        self._runner = PipelineRunner(dataset, config)
        self._runner.log_message.connect(self.progress_panel.append_log)
        self._runner.progress_updated.connect(self.progress_panel.set_progress)
        self._runner.stage_changed.connect(self.progress_panel.set_stage)
        self._runner.reconstruction_ready.connect(self._on_reconstruction_ready)
        self._runner.localization_ready.connect(self._on_localization_ready)
        self._runner.finished.connect(self._on_finished)
        self._runner.start()

    def _on_reconstruction_ready(self, sfm_dir: str) -> None:
        self.viewer_panel.load_model(sfm_dir)
        self._view_tabs.setCurrentIndex(0)

    def _on_localization_ready(self, sfm_dir: str, images_dir: str, results_path: str) -> None:
        self.loc_panel.load_localization(sfm_dir, images_dir, results_path)
        self._view_tabs.setCurrentIndex(1)

    def _on_finished(self, success: bool) -> None:
        self.pipeline_panel.set_running(False)
        if success:
            self.statusBar().showMessage("Pipeline completado con éxito.")
        else:
            self.statusBar().showMessage("Pipeline falló — revisá los logs.")

    def closeEvent(self, event) -> None:
        if self._runner and self._runner.isRunning():
            self._runner.terminate()
            self._runner.wait()
        super().closeEvent(event)
