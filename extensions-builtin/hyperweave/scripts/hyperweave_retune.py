"""Forge img2img Script entrypoint for generation-free HyperWeave retuning."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules import images, processing
from modules.api import api
from modules.shared import opts

from hyperweave.forge_adapter import ProcessingSnapshot
from hyperweave.retune import RETUNE_VERSION, RetuneSettings, retune_image


def _normalize_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return api.decode_base64_to_image(value)
    if isinstance(value, dict):
        candidate = value.get("composite") or value.get("background")
        if isinstance(candidate, Image.Image):
            return candidate
    raise TypeError(f"Unsupported HyperWeave Retune image argument: {type(value).__name__}")


class HyperWeaveRetuneScript(scripts.Script):
    def title(self):
        return "HyperWeave · 生成せず再調整"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.Markdown(
            "既存のHyperWeave結果を固定し、元のimg2img入力との差分だけをCPUで再調整します。"
            "生成モデル・VAE・テキストエンコーダーは呼びません。白い保護マスク領域は既存結果を維持します。"
        )
        generated_image = gr.Image(
            label="再調整するHyperWeave結果",
            type="pil",
            image_mode="RGBA",
            sources=["upload", "clipboard"],
        )
        with gr.Row():
            high_detail = gr.Slider(
                label="High detail",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
            )
            mid_detail = gr.Slider(
                label="Mid detail",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
            )
            low_detail = gr.Slider(
                label="Low-frequency generated residual",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
            )
            chroma_detail = gr.Slider(
                label="Generated chroma",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
            )
        with gr.Row():
            protection_mask = gr.Image(
                label="Retune Protection Mask · white keeps current result",
                type="pil",
                image_mode="RGBA",
                sources=["upload", "clipboard"],
            )
            mask_channel = gr.Radio(
                label="Protection mask channel",
                choices=["Luminance", "Alpha"],
                value="Luminance",
            )
            tile_size = gr.Dropdown(
                label="CPU retune tile",
                choices=[512, 768, 1024, 1536, 2048],
                value=1024,
            )
        self.infotext_fields = [
            (high_detail, "HyperWeave Retune high"),
            (mid_detail, "HyperWeave Retune mid"),
            (low_detail, "HyperWeave Retune low"),
            (chroma_detail, "HyperWeave Retune chroma"),
        ]
        return [
            generated_image,
            high_detail,
            mid_detail,
            low_detail,
            chroma_detail,
            protection_mask,
            mask_channel,
            tile_size,
        ]

    def run(
        self,
        p,
        generated_image,
        high_detail,
        mid_detail,
        low_detail,
        chroma_detail,
        protection_mask,
        mask_channel,
        tile_size,
    ):
        if not getattr(p, "init_images", None):
            raise ValueError("HyperWeave Retune requires an img2img source image.")
        generated = _normalize_image(generated_image)
        if generated is None:
            raise ValueError("再調整するHyperWeave結果を指定してください。")
        protection = _normalize_image(protection_mask)
        source = p.init_images[0].copy()
        settings = RetuneSettings(
            high_detail=float(high_detail),
            mid_detail=float(mid_detail),
            low_detail=float(low_detail),
            chroma_detail=float(chroma_detail),
            tile_size=int(tile_size),
            mask_channel=str(mask_channel),
        )
        result, metrics = retune_image(
            source,
            generated,
            settings,
            protection_mask=protection,
        )

        snapshot = ProcessingSnapshot(p)
        original_save_samples = p.save_samples()
        try:
            p.width, p.height = result.size
            extra = dict(getattr(p, "extra_generation_params", {}) or {})
            extra.update(
                {
                    "Script": self.title(),
                    "HyperWeave Retune version": RETUNE_VERSION,
                    "HyperWeave Retune high": settings.high_detail,
                    "HyperWeave Retune mid": settings.mid_detail,
                    "HyperWeave Retune low": settings.low_detail,
                    "HyperWeave Retune chroma": settings.chroma_detail,
                    "HyperWeave Retune tile": settings.tile_size,
                    "HyperWeave Retune changed pixels": round(
                        metrics.changed_pixel_fraction, 6
                    ),
                    "HyperWeave Retune time": round(metrics.elapsed_seconds, 3),
                    "HyperWeave Retune generator calls": 0,
                }
            )
            p.extra_generation_params = extra
            prompt = p.prompt[0] if isinstance(p.prompt, list) else p.prompt
            negative = (
                p.negative_prompt[0]
                if isinstance(p.negative_prompt, list)
                else p.negative_prompt
            )
            seed = p.seed[0] if isinstance(p.seed, list) else p.seed
            subseed = p.subseed[0] if isinstance(p.subseed, list) else p.subseed
            seed_value = -1 if seed is None else int(seed)
            subseed_value = 0 if subseed is None else int(subseed)
            p.all_prompts = [prompt]
            p.all_negative_prompts = [negative]
            p.all_seeds = [seed_value]
            p.all_subseeds = [subseed_value]
            info = processing.create_infotext(
                p,
                p.all_prompts,
                p.all_seeds,
                p.all_subseeds,
                comments=[],
            )
            retune_json = json.dumps(
                {"metrics": asdict(metrics)},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            result.info["parameters"] = info
            result.info["hyperweave_retune"] = retune_json
            if original_save_samples:
                images.save_image(
                    result,
                    p.outpath_samples,
                    "",
                    seed_value,
                    prompt,
                    opts.samples_format,
                    info=info,
                    p=p,
                    existing_info={"hyperweave_retune": retune_json},
                )
            processed = processing.Processed(
                p,
                [result],
                seed=seed_value,
                info=info,
                infotexts=[info],
            )
            processed.width, processed.height = result.size
            processed.extra_generation_params = extra
            return processed
        finally:
            snapshot.restore(p)
