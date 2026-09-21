@echo off
setlocal
cd /d "%~dp0"
echo [Aikimi] Setting up a separate Jev SDK and H3 experiment runtime.
echo [Aikimi] Normal Forge/H3 environments and model weights are not replaced.
py -3.12 tools\setup_jev_sparse.py --sdk --create-h3-runtime %*
if errorlevel 1 (
  echo [Aikimi] Setup failed. See the error above.
  exit /b 1
)
endlocal
