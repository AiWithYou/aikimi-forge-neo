import os

import gradio as gr

from modules import scripts_postprocessing
from modules.background_removal import DEFAULT_MODEL, MODELS, BackgroundRemover
from modules.ui_components import InputAccordion


class ScriptPostprocessingBackgroundRemoval(scripts_postprocessing.ScriptPostprocessing):
    name = "Background Removal"
    order = 20000

    def __init__(self):
        self.remover = BackgroundRemover()

    def ui(self):
        with InputAccordion(
            False, label="Background Removal / 背景除去", elem_id="extras_background_removal"
        ) as enable:
            model_id = gr.Dropdown(
                choices=[(spec.label, key) for key, spec in MODELS.items()],
                value=DEFAULT_MODEL,
                label="モデル",
                elem_id="extras_background_model",
            )
            output_mask = gr.Checkbox(False, label="マスクも保存", elem_id="extras_background_mask")
        return {"enable": enable, "model_id": model_id, "output_mask": output_mask}

    def process(self, pp, enable=False, model_id=DEFAULT_MODEL, output_mask=False):
        if not enable:
            return
        if model_id not in MODELS:
            raise gr.Error("背景除去モデルを選択してください。")

        from backend import memory_management
        from modules import devices, paths, shared

        if shared.state.interrupted or shared.state.stopping_generation:
            return
        shared.state.textinfo = f"背景除去: {MODELS[model_id].label}"
        memory_management.unload_all_models()
        try:
            output, mask = self.remover.remove(
                pp.image,
                model_id,
                os.path.join(paths.models_path, "background_removal"),
                memory_management.get_torch_device(),
            )
        finally:
            devices.torch_gc()
        if shared.state.interrupted or shared.state.stopping_generation:
            return
        pp.image = output
        pp.info[self.name] = MODELS[model_id].label
        if output_mask:
            pp.extra_images.append(pp.create_copy(mask, nametags=["mask"], disable_processing=True))
