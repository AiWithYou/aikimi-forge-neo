@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -X utf8 tools\setup_qwen_image21.py %*
) else (
  py -3.13 -X utf8 tools\setup_qwen_image21.py %*
)
set "setup_exit=%errorlevel%"
if not "%setup_exit%"=="0" echo Qwen Image 2.1 setup failed. See the message above.
if /I not "%QWEN_SETUP_NO_PAUSE%"=="1" pause
exit /b %setup_exit%
