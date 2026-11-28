"""Native pinned Comfy Ming Image inference, isolated from the Forge process."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, PngImagePlugin

from . import checkpoints, core, references


class Cancelled(RuntimeError):
    """The job was stopped before publishing a complete artifact."""

    public_error_code = "cancelled"


class Runner:
    def __init__(self, root, runtime=None):
        self.root = Path(root)
        self._runtime = runtime

    def close(self):
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def run(self, directory, request, *, work_dir):
        request = core.validate_request(request, internal=True)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        def check():
            if (work_dir / "cancel").exists():
                raise Cancelled("Ming Image generation was cancelled")

        def progress(stage, current=0, total=1):
            check()
            core.atomic_json(work_dir / "progress.json", {"stage": stage, "current": current, "total": total})

        check()
        progress("verifying")
        if not core.model_ready(self.root, request):
            raise core.MingImageError(
                "選択したMing Imageの保存形式が未導入か検証できません。セットアップを実行してください。"
            )
        images = references.load(work_dir, request)
        started = time.monotonic()
        if self._runtime is None:
            progress("initializing")
            self._runtime = ComfyRuntime(self.root)
        completed = False
        temporary = directory / ".output.png.tmp"
        try:
            image = (
                self._runtime.generate(request, progress, check, images=images)
                if images
                else self._runtime.generate(request, progress, check)
            )
            check()
            references.load(work_dir, request)
            checkpoints.bind(self.root, request)
            if not isinstance(image, Image.Image) or image.size != (request["width"], request["height"]):
                raise RuntimeError("Ming Image returned an invalid image")
            if request["transparent"] and image.mode != "RGBA":
                raise core.MingImageError("透過指定には RGBA 出力が必要です。")
            metadata = {
                **request,
                "model_id": core.MODEL_ID,
                "model_revision": core.MODEL_REVISION,
                "comfy_revision": core.manifest()["runtime"]["comfy_revision"],
                "runtime_backend": "ComfyUI",
                "sampler": "euler",
                "scheduler": "simple",
                "quant_revision": core.manifest()["profiles"][request["profile"]]["revision"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            metadata = core.png_metadata(metadata)
            info = PngImagePlugin.PngInfo()
            info.add_text("ming_image", json.dumps(metadata, ensure_ascii=False, allow_nan=False))
            image_path = directory / "output.png"
            image.save(temporary, format="PNG", pnginfo=info)
            check()
            os.replace(temporary, image_path)
            result = {**metadata, "image_path": image_path.name}
            core.atomic_json(directory / "result.json", result)
            progress("complete")
            completed = True
            return result
        finally:
            temporary.unlink(missing_ok=True)
            if not completed:
                (directory / "output.png").unlink(missing_ok=True)
                (directory / "result.json").unlink(missing_ok=True)


def output_image(pixels, transparent):
    import numpy as np

    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4) or not np.isfinite(pixels).all():
        raise ValueError("VAEの出力画像が不正です。")
    if transparent and pixels.shape[-1] != 4:
        raise core.MingImageError("透過指定には RGBA 出力が必要です。")
    image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    return image if transparent else image.convert("RGB")


class ComfyRuntime:
    def __init__(self, root):
        self.root = root
        sys.path.insert(0, str(core.runtime_paths(root)["comfy"]))
        import comfy.sd
        import torch

        if not torch.cuda.is_available():
            raise core.MingImageError("CUDA GPUが必要です。")
        self.sd = comfy.sd
        self.models = None
        self.identity = None

    def close(self):
        import comfy.model_management as management

        self.models = None
        self.identity = None
        management.unload_all_models()
        management.soft_empty_cache()

    def generate(self, request, progress, check, *, images=()):
        import comfy.model_management as management
        import comfy.sample
        import comfy.utils
        import numpy as np
        import torch
        from comfy_extras.nodes_ming import TextEncodeMingImageEdit
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux

        paths = core.asset_paths(self.root, request)
        identity = (paths, request.get("checkpoint_identity"), request.get("text_checkpoint_identity"))
        if self.identity != identity:
            self.close()
            progress("loading")
            model = self.sd.load_diffusion_model(str(paths["dit"]), disable_dynamic=True)
            clip = self.sd.load_clip(
                ckpt_paths=[str(paths["text"])], clip_type=self.sd.CLIPType.QWEN_IMAGE, disable_dynamic=True
            )
            vae = self.sd.VAE(sd=comfy.utils.load_torch_file(str(paths["vae"]), safe_load=True))
            self.models = model, clip, vae
            self.identity = identity
        model, clip, vae = self.models
        width, height = request["width"], request["height"]
        prompt = request["prompt"]
        if request["transparent"]:
            try:
                structured = json.loads(prompt)
            except ValueError:
                structured = None
            if isinstance(structured, dict):
                prompt = json.dumps({**structured, "background": "transparent, RGBA, 4-channel"}, ensure_ascii=False)
            else:
                prompt += ", RGBA, 4-channel, transparent background"
        with torch.inference_mode():
            progress("encoding")
            inputs = {
                f"image_{i}": torch.from_numpy(np.array(image.resize((width, height), Image.Resampling.LANCZOS)))
                .float()
                .unsqueeze(0)
                / 255
                for i, image in enumerate(images, 1)
            }

            def encode(text):
                if inputs:
                    return TextEncodeMingImageEdit.execute(clip, text, inputs, vae).result[0]
                return clip.encode_from_tokens_scheduled(clip.tokenize(text))

            positive = encode(prompt)
            negative = encode(request["negative_prompt"]) if request["cfg"] > 1 else positive
            (model,) = ModelSamplingFlux().patch(model, request["shift"], 0.5, width, height)
            latent = torch.zeros((1, 4, height // 8, width // 8), device=management.intermediate_device())
            latent = comfy.sample.fix_empty_latent_channels(model, latent, downscale_ratio_spacial=8)
            noise = comfy.sample.prepare_noise(latent, request["seed"])
            check()
            samples = comfy.sample.sample(
                model,
                noise,
                request["steps"],
                request["cfg"],
                "euler",
                "simple",
                positive,
                negative,
                latent,
                seed=request["seed"],
                callback=lambda step, x0, x, total: progress("sampling", step + 1, total),
            )
            progress("decoding")
            decoded = vae.decode(samples)
            if decoded.ndim == 5:
                decoded = decoded.flatten(0, 1)
            if decoded.ndim != 4 or len(decoded) != 1:
                raise ValueError("VAEの出力枚数が一致しません。")
            pixels = decoded[0].float().cpu().numpy()
            return output_image(pixels, request["transparent"])
