@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -X utf8 tools\setup_irodori.py %*
) else (
  py -3 -X utf8 tools\setup_irodori.py %*
)
if errorlevel 1 (
  echo.
  echo Irodori TTS setup did not finish. Python, Git and an NVIDIA GPU are required.
  echo See docs\irodori-tts.md
)
pause
