# https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy_extras/nodes_slg.py

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.args import dynamic_args
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling import sampling_function
from modules import scripts
from modules.shared import opts

_calc_cond_uncond_batch = sampling_function.calc_cond_uncond_batch


def parse_layers(layers: str) -> list[int]:
    return [int(layer) for layer in re.findall(r"\d+", layers)]


class SkipLayerGuidance(scripts.Script):
    def title(self):
        return "Skip Layer Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process_before_every_sampling(self, p: "StableDiffusionProcessing", *args, **kwargs):
        self.on_process_cleanup(p)
        if not (getattr(dynamic_args, "moss", False) and opts.slg_enable):
            return

        single_layers = parse_layers(opts.slg_single_layers)

        if not single_layers:
            return

        p_start = max(min(opts.slg_start_percent, 1.0), 0.0)
        p_end = max(min(opts.slg_end_percent, 1.0), 0.0)

        unet = p.sd_model.forge_objects.unet
        predictor = unet.model.predictor
        sigma_start = predictor.percent_to_sigma(p_start)
        sigma_end = predictor.percent_to_sigma(p_end)

        calc_batch_function = self.make_calc_cond_batch_function(single_layers, sigma_start, sigma_end)

        m = unet.clone()
        m.set_model_sampler_calc_cond_batch_function(calc_batch_function)
        p._slg_patch = (unet, m)
        p.sd_model.forge_objects.unet = m

        p.extra_generation_params.update(
            {
                "Skip Layer Guidance": True,
                "SLG Single Layers": ", ".join(map(str, single_layers)) if single_layers else None,
                "SLG Start Percent": p_start,
                "SLG End Percent": p_end,
            }
        )

    def on_process_cleanup(self, p, *args):
        saved = getattr(p, "_slg_patch", None)
        if saved is not None:
            original, patched = saved
            if p.sd_model.forge_objects.unet is patched:
                p.sd_model.forge_objects.unet = original
            del p._slg_patch

    def postprocess(self, p, processed, *args):
        self.on_process_cleanup(p)

    @staticmethod
    def make_calc_cond_batch_function(single_layers: list[int], sigma_start: float, sigma_end: float):

        def skip(args, extra_args):
            return args

        def calc_batch_function(model, cond, uncond, x, sigma, model_options, **kwargs):
            slg_model_options = model_options.copy()

            for layer in single_layers:
                slg_model_options = set_model_options_patch_replace(slg_model_options, skip, "dit", "single_block", layer)

            sigma_ = sigma[0].item()

            if sigma_end <= sigma_ <= sigma_start and uncond is not None:
                cond_out, _ = _calc_cond_uncond_batch(model, cond, None, x, sigma, model_options, **kwargs)
                _, uncond_out = _calc_cond_uncond_batch(model, [], uncond, x, sigma, slg_model_options, **kwargs)
                return [cond_out, uncond_out]

            return _calc_cond_uncond_batch(model, cond, uncond, x, sigma, model_options, **kwargs)

        return calc_batch_function
