@echo off
setlocal
cd /d "%~dp0"
call python -I -c "import sys; assert sys.version_info[:2] >= (3, 12), 'Python 3.12 or newer is required'"
if errorlevel 1 (
    echo Python 3.12 or newer is required. See README.md.
    pause
    exit /b 1
)
call python -B "%~dp0tools\setup_minimax_h3_w4a8.py" %*
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" (
    echo MiniMax H3 W4A8 download failed. See the message above.
) else (
    echo MiniMax H3 W4A8 preparation completed.
)
pause
exit /b %EXITCODE%
