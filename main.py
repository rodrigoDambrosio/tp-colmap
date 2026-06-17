import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from src.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TP COLMAP — Visual Localization")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
