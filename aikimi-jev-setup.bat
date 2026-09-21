@echo off
setlocal
cd /d "%~dp0"
echo [Aikimi] Setting up the separate Jev SDK. Model weights are not downloaded.
echo [Aikimi] Add --h3 to install the node in an existing H3 runtime.
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" tools\setup_jev_sparse.py --sdk %*
) else (
  py -3.13 tools\setup_jev_sparse.py --sdk %*
)
if errorlevel 1 (
  echo [Aikimi] Setup failed. See the error above.
  exit /b 1
)
endlocal
