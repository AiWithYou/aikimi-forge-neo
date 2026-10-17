@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Please run the Forge Neo setup first.
    exit /b 1
)
"venv\Scripts\python.exe" -X utf8 "tools\setup_moss_image.py" %*
exit /b %errorlevel%
