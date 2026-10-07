@echo off
setlocal
cd /d "%~dp0"
echo [Aikimi] Clef: separate runtime, pre-quantized Flash INT8 + Clef NF4 releases.
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" tools\setup_clef.py %*
) else (
  py -3.13 tools\setup_clef.py %*
)
if errorlevel 1 exit /b 1
endlocal
