@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" tools\setup_flux3_klein.py %*
) else (
  py -3.13 tools\setup_flux3_klein.py %*
)
if errorlevel 1 exit /b 1
endlocal
