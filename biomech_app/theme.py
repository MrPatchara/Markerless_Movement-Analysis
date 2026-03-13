"""
ธีมและสีสำหรับแอป
"""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

COLORS = {
    "bg_dark": "#0D1B2A",
    "bg_card": "#1B263B",
    "bg_elevated": "#415A77",
    "accent": "#00D4AA",
    "text": "#E0E1DD",
    "text_dim": "#778DA9",
}


def apply_dark_theme(app: QApplication) -> None:
    """ใช้ธีมมืดให้แอป"""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg_dark"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    app.setPalette(palette)
    app.setStyleSheet("""
        QMainWindow, QWidget { background-color: #0D1B2A; color: #E0E1DD; }
        QPushButton {
            background-color: #1B263B; color: #E0E1DD;
            border: 1px solid #415A77; border-radius: 6px;
            padding: 8px 16px; min-height: 24px;
        }
        QPushButton:hover { background-color: #415A77; border-color: #00A080; }
        QPushButton:pressed { background-color: #00A080; color: #0D1B2A; }
        QPushButton:disabled {
            background-color: #2a3544; color: #5a6575;
            border-color: #3a4a5a;
        }
        QPushButton#openButton { background-color: #00D4AA; color: #0D1B2A; font-weight: 600; }
        QPushButton#openButton:disabled {
            background-color: #2a3544; color: #5a6575; border-color: #3a4a5a;
        }
        QPushButton#closeButton {
            background-color: #C53030; color: #FFFFFF; font-weight: 600;
            border: 1px solid #9B2C2C;
        }
        QPushButton#closeButton:hover { background-color: #E53E3E; border-color: #C53030; }
        QPushButton#closeButton:pressed { background-color: #9B2C2C; }
        QPushButton#closeButton:disabled {
            background-color: #2a3544; color: #5a6575; border-color: #3a4a5a;
        }
        QPushButton#aboutButton {
            background-color: transparent;
            border: 1px solid #415A77;
            color: #E0E1DD;
            padding: 6px 12px;
            font-size: 11px;
        }
        QPushButton#aboutButton:hover {
            background-color: #1B263B;
            border-color: #00D4AA;
        }
        QPushButton#aboutButton:pressed {
            background-color: #00D4AA;
            color: #0D1B2A;
        }
        QGroupBox { font-weight: 600; border: 1px solid #415A77; border-radius: 8px; margin-top: 12px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #00D4AA; }
        QStatusBar { background-color: #1B263B; color: #778DA9; }
        QLabel { color: #E0E1DD; }
    """)
