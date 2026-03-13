"""
PyInstaller runtime hook: โหลด numpy ก่อนทุกอย่างเพื่อป้องกัน
"ImportError: cannot load module more than once per process"
เมื่อใช้ --onefile
"""
import sys
# บังคับให้ numpy ถูกโหลดครั้งเดียวใน process ก่อนที่ main จะ import
import numpy  # noqa: F401
