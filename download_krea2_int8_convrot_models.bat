@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_krea2_int8_convrot_models.ps1"
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo Krea2 model download failed. See the message above.
) else (
    echo Krea2 INT8 ConvRot model download completed.
    echo Start webui-user.bat next.
)
echo.
pause
exit /b %EXITCODE%
