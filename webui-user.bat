@echo off
cd /d "%~dp0"

:: Keep personal settings in the ignored webui-user.local.bat. Never put
:: passwords, API keys, or tunnel tokens in this tracked compatibility wrapper.
if exist "%~dp0webui-user.local.bat" call "%~dp0webui-user.local.bat"

if not defined VENV_DIR set "VENV_DIR=%~dp0venv"
set "_AIKIMI_DEFAULT_ARGS=0"
if not defined COMMANDLINE_ARGS (
  set "COMMANDLINE_ARGS=--uv --bnb --api --server-name 127.0.0.1 --port 7861 --theme dark --tiled-conv2d 128 --cuda-malloc"
  set "_AIKIMI_DEFAULT_ARGS=1"
)
if "%_AIKIMI_DEFAULT_ARGS%"=="1" if exist "%~dp0forge_neo_model_paths.yaml" set "COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --forge-ref-comfy-yaml forge_neo_model_paths.yaml"

call webui.bat %*
