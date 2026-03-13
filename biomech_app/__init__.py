"""
Biomechanics SLS Analysis – แอปหลัก
"""
import os
os.environ.setdefault("QT_API", "pyside6")

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from biomech_app.theme import apply_dark_theme
from biomech_app.main_window import BiomechMainWindow


def main() -> int:
    """รันแอป – return exit code"""
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    app.setFont(QFont("Segoe UI", 10))

    window = BiomechMainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
