@echo off
REM Build exe: ใช้ Python ตัวเดียวกับที่รันโปรแกรม (ถ้าใช้ 3.14 ใน Cursor ให้ใช้ 3.14 ตอน build)
echo Python used for build:
python --version
echo.
echo Installing dependencies...
python -m pip install -r requirements.txt
python -m pip install pyinstaller
echo.
echo Building exe...
python -m PyInstaller main.spec
echo.
echo Done. Run: dist\MarkerlessMovementAnalysis\MarkerlessMovementAnalysis.exe
