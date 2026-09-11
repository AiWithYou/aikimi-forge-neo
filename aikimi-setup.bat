@echo off
setlocal
where pwsh.exe >nul 2>nul
if errorlevel 1 (
  echo PowerShell 7 is required. See README.md.
  pause
  exit /b 2
)
pwsh.exe -NoProfile -File "%~dp0aikimi-setup.ps1" %*
exit /b %ERRORLEVEL%
