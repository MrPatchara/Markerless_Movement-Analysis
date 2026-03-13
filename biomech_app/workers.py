"""
Background workers (โหลดไฟล์แบบ async)
"""
from PySide6.QtCore import QThread, Signal

from biomech.c3d_loader import load_motion_file


class LoadWorker(QThread):
    """โหลดไฟล์ C3D/JSON ใน background"""
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            data = load_motion_file(self.path)
            self.finished.emit(data, self.path)
        except Exception as e:
            self.error.emit(str(e))
