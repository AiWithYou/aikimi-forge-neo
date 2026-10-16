"""Register Irodori without importing its GPU inference dependencies."""

from modules import script_callbacks
from modules_forge.irodori.ui import build_ui


def on_ui_tabs():
    return [(build_ui(), "Irodori TTS", "irodori_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
