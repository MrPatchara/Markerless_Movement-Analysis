# -*- mode: python ; coding: utf-8 -*-
# ก่อน build ต้องติดตั้ง: pip install pyvista pyvistaqt opencv-python
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# โฟลเดอร์โปรเจกต์ (ใช้ตอน build จากที่อื่นได้)
project_dir = os.path.dirname(os.path.abspath(SPEC))

# รวม numpy ทั้งหมด (datas, binaries, hiddenimports) เพื่อป้องกัน ModuleNotFoundError
numpy_datas, numpy_binaries, numpy_hidden = collect_all('numpy')

# PyVista + PyVistaQt + VTK สำหรับ 3D view (ถ้าติดตั้งแล้วจะถูก bundle)
def _collect(name):
    try:
        return collect_all(name)
    except Exception:
        return ([], [], [])

pyvista_datas, pyvista_binaries, pyvista_hidden = _collect('pyvista')
pyvistaqt_datas, pyvistaqt_binaries, pyvistaqt_hidden = _collect('pyvistaqt')
vtk_datas, vtk_binaries, vtk_hidden = _collect('vtk')
cv2_datas, cv2_binaries, cv2_hidden = _collect('cv2')

# รวม datas / binaries / hiddenimports
all_datas = list(numpy_datas) + list(pyvista_datas) + list(pyvistaqt_datas) + list(vtk_datas) + list(cv2_datas)
all_binaries = list(numpy_binaries) + list(pyvista_binaries) + list(pyvistaqt_binaries) + list(vtk_binaries) + list(cv2_binaries)
all_hidden = list(numpy_hidden) + list(pyvista_hidden) + list(pyvistaqt_hidden) + list(vtk_hidden) + list(cv2_hidden)

# ไฟล์โลโก้/ไอคอน (ถ้ามี) ให้รวมใน exe
for name in ['logo.png', 'logo2.png']:
    p = os.path.join(project_dir, name)
    if os.path.isfile(p):
        all_datas.append((p, '.'))

# ไอคอน exe (ใช้ your_icon.ico หรือ logo2.png แปลงเป็น .ico)
icon_file = os.path.join(project_dir, 'your_icon.ico')
if not os.path.isfile(icon_file):
    icon_file = os.path.join(project_dir, 'logo2.png')
if not os.path.isfile(icon_file):
    icon_file = None

a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'cv2',
        'pyvista',
        'pyvistaqt',
        'vtk',
    ] + all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'PyQt6', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
        'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# onedir: เปิดโปรแกรมเร็วกว่า onefile (ไม่ต้องแตกไฟล์ทุกครั้ง)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MarkerlessMovementAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[icon_file] if icon_file else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MarkerlessMovementAnalysis',
)
