@echo off
setlocal
cd /d "%~dp0"
echo [Aikimi] Iris: normal / downloadable INT8, generation / depth / restoration.
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" tools\setup_iris.py %*
) else (
  py -3.13 tools\setup_iris.py %*
)
if errorlevel 1 exit /b 1
endlocal
