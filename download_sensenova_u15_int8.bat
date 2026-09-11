@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_sensenova_u15_int8.ps1"
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo SenseNova U1.5 setup failed. See the message above.
) else (
    echo SenseNova U1.5 final INT8 ConvRot setup completed.
    echo Start webui-user.bat and open the SenseNova U1.5 tab.
)
echo.
pause
exit /b %EXITCODE%
