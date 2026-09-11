@echo off
setlocal
if "%~1"=="" (
  pwsh -NoProfile -File "%~dp0aikimi-launch.ps1"
  exit /b %ERRORLEVEL%
)
pwsh -NoProfile -File "%~dp0aikimi-launch.ps1" -Profile "%~1"
exit /b %ERRORLEVEL%
