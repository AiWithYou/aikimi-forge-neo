@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_anima38_v11_int8_convrot_models.ps1"
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo Anima 3.8B v1.1 setup failed. See the message above.
) else (
    echo Anima 3.8B v1.1 INT8 ConvRot setup completed.
    echo Start webui-user.bat next.
)
echo.
pause
exit /b %EXITCODE%
