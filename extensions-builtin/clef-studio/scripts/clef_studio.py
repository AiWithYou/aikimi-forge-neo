"""Register the local Clef decision Studio without importing torch in the tab."""

import gradio as gr

from modules import script_callbacks
from modules_forge.clef.ui import build_ui


def on_ui_tabs():
    with gr.Blocks() as tab:
        build_ui()
    return [(tab, "Clef", "clef_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
