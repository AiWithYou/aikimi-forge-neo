@echo off
setlocal
cd /d "%~dp0"
py -3.12 -X utf8 modules_forge\yue2_studio\setup.py %*
if errorlevel 1 (
  echo.
  echo YuE2 setup did not finish. Python 3.12 and Git are required.
  echo See extensions-builtin\yue2-studio\README.md
)
pause
