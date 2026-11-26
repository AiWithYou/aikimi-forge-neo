import gradio as gr

from modules import script_callbacks


def gallery_tab():
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.HTML("", elem_id="aikimi-gallery-workspace")
    return [(tab, "ギャラリー", "aikimi_gallery")]


script_callbacks.on_ui_tabs(gallery_tab)
