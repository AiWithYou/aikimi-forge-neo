@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_sensenova_u15_int8.ps1" -ModelOnly
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" (
    echo SenseNova U1.5 model download failed. See the message above.
) else (
    echo SenseNova U1.5 model download completed.
)
pause
exit /b %EXITCODE%
