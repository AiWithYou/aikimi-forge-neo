@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Python environment not found. Run aikimi-setup.bat first.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" "tools\download_background_models.py" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo Background model download failed. See the message above.
pause
exit /b %RESULT%
