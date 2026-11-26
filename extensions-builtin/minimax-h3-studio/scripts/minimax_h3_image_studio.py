import gradio as gr

from modules import script_callbacks
from modules.paths import data_path
from modules_forge.image_workspace import descendants


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Tabs(elem_id="h3-image-tabs"):
            with gr.Tab("P-H3 Image", elem_id="p_h3_image") as p_h3_tab:
                from modules_forge.p_h3_image_ui import render as render_p_h3
                render_p_h3(tab, data_directory=data_path)

                for component in descendants(p_h3_tab):
                    component.custom_script_source = "p_h3_image"
            with gr.Tab("MiniMax-H3-Image", elem_id="official_h3_image") as official_tab:
                from modules_forge.minimax_h3_image_ui import render

                render(data_directory=data_path)
                for component in descendants(official_tab):
                    component.custom_script_source = "minimax_h3_image_official"
    return [(tab, "H3 Image", "minimax_h3_image_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
