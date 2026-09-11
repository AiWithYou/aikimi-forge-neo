@echo off
:: Copy this file to webui-user.local.bat, then edit only non-secret local
:: preferences. The tracked webui-user.bat loads that ignored file.

set "VENV_DIR=%~dp0venv"
set "COMMANDLINE_ARGS=--uv --bnb --api --server-name 127.0.0.1 --port 7861 --theme dark --tiled-conv2d 128 --cuda-malloc"

if exist "%~dp0forge_neo_model_paths.yaml" (
  set "COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --forge-ref-comfy-yaml forge_neo_model_paths.yaml"
)

:: Remote credentials belong in ignored files under secrets\ and must be used
:: only with the explicit LANAuthenticated profile documented in README.md.
