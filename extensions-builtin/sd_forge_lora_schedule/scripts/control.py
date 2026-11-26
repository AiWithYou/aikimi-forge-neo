import logging
import math
import re
from itertools import pairwise
from typing import TYPE_CHECKING

from networks import available_network_aliases

from backend.args import dynamic_args
from backend.logging import setup_logger
from backend.patcher.base import ModelPatcher, OnlineLoRAPatch
from modules import scripts, shared

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser, on_script_unloaded

logger = logging.getLogger("lora_ctl")
setup_logger(logger)

lora_ctl = re.compile(r"<lora:([^\:\>]+):\[([^\]\>]+)\]>")


def schedule_weight(points, t):
    for (w1, t1), (w2, t2) in pairwise(points):
        if t <= t2:
            return w1 + (w2 - w1) * max(t - t1, 0) / (t2 - t1)
    return points[-1][0]


class LoRAControl(scripts.Script):
    mapping: dict[str, list[tuple[float, float]]] = {}

    originals = {}

    def title(self):
        return "LoRA Control Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    @staticmethod
    def _parse_schedule(syntax: str) -> list[tuple[float, float]]:
        points = [tuple(map(float, chunk.split("@"))) for chunk in syntax.split(",")]
        if len(points) < 2 or any(not math.isfinite(w) or not 0 <= t <= 1 for w, t in points):
            raise ValueError("Invalid LoRA schedule")
        if any(a[1] >= b[1] for a, b in pairwise(points)):
            raise ValueError("LoRA schedule times must increase")
        return points

    @classmethod
    def restore(cls):
        for patch, weight in cls.originals.values():
            patch[0][0] = weight
        cls.originals.clear()
        cls.mapping.clear()

    def before_process(self, p: "StableDiffusionProcessing"):
        self.restore()
        prompts = p.prompt if isinstance(p.prompt, list) else [p.prompt]
        signatures = [lora_ctl.findall(prompt) for prompt in prompts]
        if not any(signatures):
            return
        uniform = all(tags == signatures[0] for tags in signatures)
        if not uniform:
            logger.warning("Skipped scheduled LoRAs because the prompts in this batch use different schedules")

        def schedule(match):
            if not uniform:
                return ""
            alias, syntax = match.groups()
            entry = available_network_aliases.get(alias)
            if entry is None or not dynamic_args.online_lora:
                logger.warning("Skipped scheduled LoRA %s (requires an installed LoRA and online patching)", alias)
                return ""
            try:
                self.mapping[entry.filename] = self._parse_schedule(syntax)
            except ValueError as exc:
                logger.warning("Invalid schedule for %s: %s", alias, exc)
                return ""
            return f"<lora:{alias}:1>"

        p._lora_schedule_prompt = p.prompt
        converted = [lora_ctl.sub(schedule, prompt) for prompt in prompts]
        p.prompt = converted if isinstance(p.prompt, list) else converted[0]

    def on_process_cleanup(self, p, *args):
        self.restore()
        if "_lora_schedule_prompt" in vars(p):
            p.prompt = vars(p).pop("_lora_schedule_prompt")

    def postprocess(self, p, processed, *args):
        self.restore()

    @classmethod
    def adjust_lora(cls, params: CFGDenoiserParams):
        if not cls.mapping or not params.denoiser.total_steps:
            return
        model: ModelPatcher = shared.sd_model.forge_objects.unet
        t = min(1.0, params.denoiser.step / max(1, params.denoiser.total_steps - 1))
        for wrappers in model.weight_wrapper_patches.values():
            for wrapper in wrappers:
                if isinstance(wrapper, OnlineLoRAPatch) and wrapper.name in cls.mapping:
                    cls.originals.setdefault(id(wrapper), (wrapper.patch, wrapper.patch[0][0]))
                    wrapper.patch[0][0] = schedule_weight(cls.mapping[wrapper.name], t)


on_cfg_denoiser(LoRAControl.adjust_lora)
on_script_unloaded(LoRAControl.restore)
