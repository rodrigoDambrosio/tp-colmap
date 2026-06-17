from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QTextEdit, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QTextCursor


class ProgressPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(4)

        # Header row: stage label + progress bar + clear button
        header = QHBoxLayout()
        self._stage_label = QLabel("En espera")
        self._stage_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self._stage_label.setMinimumWidth(220)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; background: #2a2a2a; }"
            "QProgressBar::chunk { background: #0078d4; border-radius: 3px; }"
        )

        btn_clear = QPushButton("Limpiar")
        btn_clear.setFixedWidth(70)
        btn_clear.clicked.connect(self.clear)

        header.addWidget(self._stage_label)
        header.addWidget(self._progress_bar, 1)
        header.addWidget(btn_clear)
        layout.addLayout(header)

        # Log area
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Cascadia Code, Consolas, Courier New", 9))
        self._log.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #444; }"
        )
        layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # Public slots (connected from MainWindow)
    # ------------------------------------------------------------------

    def append_log(self, message: str) -> None:
        # Colorize ERROR lines in red, warnings in yellow
        if "ERROR" in message or "error" in message.lower():
            self._log.append(f'<span style="color:#f44;">{message}</span>')
        elif "Advertencia" in message or "WARNING" in message:
            self._log.append(f'<span style="color:#fa0;">{message}</span>')
        elif message.startswith("[") and "---" in message:
            self._log.append(f'<span style="color:#4af;">{message}</span>')
        else:
            self._log.append(message)
        # Auto-scroll
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def set_progress(self, value: int) -> None:
        self._progress_bar.setValue(value)

    def set_stage(self, stage: str) -> None:
        self._stage_label.setText(stage)

    def clear(self) -> None:
        self._log.clear()
        self._progress_bar.setValue(0)
        self._stage_label.setText("En espera")
