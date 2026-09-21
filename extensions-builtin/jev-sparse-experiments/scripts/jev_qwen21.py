"""Qwen 2.1 opt-in routing; no extra tabs or legacy-Qwen processor patches."""
from modules import script_callbacks
from modules_forge.jev_sparse import qwen21_integration

script_callbacks.on_before_ui(qwen21_integration.install)
script_callbacks.on_script_unloaded(qwen21_integration.uninstall)
