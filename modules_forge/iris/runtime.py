"""Official Iris inference with sequential text/model residency and progress."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .core import (
    CODE_REVISION,
    SOURCE_REPO,
    SOURCE_REVISION,
    TEXT_REVISION,
    atomic_json,
    checkpoint_identity,
    model_directory,
    model_files,
    sha256,
)
from .quantization import load_int8


class Cancelled(RuntimeError):
    pass


def check_model_output(module, args, output):
    if not torch.isfinite(output.x).all():
        raise ValueError("モデル出力に非有限値が含まれています。")


class PreparedText:
    def __init__(self, positive, negative):
        self.positive, self.negative = positive, negative

    def encode(self, prompts):
        return self.positive

    def null(self, prompt=""):
        return self.negative


def load_model(root, precision, task, checkpoint="base"):
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    from omegaconf import OmegaConf

    weights, config = model_files(root, precision, task, checkpoint)
    raw = OmegaConf.to_container(OmegaConf.load(config))
    settings = raw.pop("task", {})
    cfg = inference_config(raw, [])
    if checkpoint != "base":

        def check_keys(value, schema, prefix):
            for name, item in value.items():
                if not hasattr(schema, name):
                    raise ValueError(f"非対応の推論設定です: {prefix}.{name}")
                if isinstance(item, dict):
                    check_keys(item, getattr(schema, name), f"{prefix}.{name}")

        for section in ("model", "text_encoder", "flow"):
            if section in raw:
                check_keys(raw[section], getattr(cfg, section), section)
    if task == "depth":
        from iris3b.downstream.depth import IrisDepth

        with torch.device("meta"):
            model = IrisDepth(cfg.model)
        model.num_train_timesteps = cfg.flow.num_train_timesteps
    else:
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
    if precision == "int8":
        model = load_int8(model, weights)
    elif precision == "w4a8":
        from .w4a8 import load_w4a8

        model = load_w4a8(model, weights)
    else:
        model.load_state_dict(load_file(weights), strict=True, assign=True)
    if precision != "w4a8":
        # Colab full fine-tunes are stored in BF16. Upstream keeps master
        # parameters in FP32 while autocast handles the inference computation.
        model.float()
    model.eval().requires_grad_(False)
    return model, cfg, settings


class Runner:
    def __init__(self, root, precision, task, checkpoint="base"):
        if precision == "w4a8":
            from .w4a8 import require_native_cuda

            require_native_cuda()
        self.root, self.precision, self.task = Path(root), precision, task
        self.checkpoint = checkpoint
        self.model, self.cfg, self.settings = load_model(root, precision, task, checkpoint)
        self.encoder = None
        self.source_identity = None
        self.loras = []

    def set_loras(self, loras):
        from .lora import apply, verify_base, verify_sources

        if any(item["metadata"].get("base_sha256") for item in loras):
            if self.source_identity is None:
                self.source_identity = checkpoint_identity(self.root, self.precision, self.task, self.checkpoint)
            verify_base(loras, self.source_identity, self.precision)
        apply(self.model, loras)
        verify_sources(loras)
        self.loras = loras

    def close(self):
        self.model = self.encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    def offload(self):
        if self.model is not None:
            self.model.cpu()
            # Upstream caches RoPE as plain tensors, outside registered buffers.
            for module in self.model.modules():
                for name in ("_rope_img", "_rope_txt", "_rope_pix"):
                    cache = getattr(module, name, None)
                    if cache is not None:
                        cache.clear()
        if self.encoder is not None:
            self.encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def prepare_text(self, request):
        from iris3b.text.base import TextEncoding
        from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

        cfg = self.cfg.text_encoder
        cfg.pretrained = str(self.root / "text-encoder")
        cfg.null_embed_dir = str(self.root / "text-cache")
        if self.encoder is None:
            if self.precision == "w4a8":
                from .text import load_text_encoder

                self.encoder = load_text_encoder(self.root, cfg)
            else:
                self.encoder = Qwen3VLTextEncoder(cfg, device="cpu")
        self.encoder.to("cuda")
        try:
            positive = self.encoder.encode([request["prompt"]])
            # Same upstream template as null(); avoid its cache filename built
            # from an absolute Windows model path (which contains a colon).
            negative = (
                self.encoder.encode([request["negative_prompt"]])
                if request.get("sampling", {}).get("mode") not in {"turbo", "official_turbo"} or request["cfg"] != 1
                else None
            )

            def cpu(encoding):
                return TextEncoding(encoding.embeddings.cpu(), encoding.mask.cpu())

            return PreparedText(cpu(positive), cpu(negative) if negative is not None else None)
        finally:
            self.encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def run(self, directory, request=None):
        directory = Path(directory)
        if request is None:
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        from .lora import verify_sources

        loras = getattr(self, "loras", [])
        if request.get("loras", []) != loras:
            raise ValueError("読み込み済みLoRAと実行条件が一致しません。")
        verify_sources(loras)
        if self.source_identity is None:
            self.source_identity = checkpoint_identity(self.root, self.precision, self.task, self.checkpoint)
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        calls = 0
        total = request.get("steps", 1) if self.task == "generate" else 1
        image_mode = request.get("image_mode", "txt2img") if self.task == "generate" else None
        image_plan = None
        if image_mode == "img2img":
            from .img2img import image_schedule

            if request.get("sampling", {}).get("mode") in {"turbo", "official_turbo"}:
                raise ValueError("img2imgではTurbo LoRAを外して通常サンプリングを使ってください。")
            image_plan = image_schedule(request["steps"], request["strength"], self.cfg.flow.shift)
            total = image_plan.steps
        if request.get("sampling", {}).get("mode") == "turbo" and request["cfg"] != 1:
            total *= 2

        def progress(module, args):
            nonlocal calls
            if (directory / "cancel").exists():
                raise Cancelled("中断しました。")
            calls += 1
            atomic_json(
                directory / "progress.json",
                {"stage": "実行中", "current": calls, "total": total, "seconds": time.perf_counter() - begin},
            )

        hook = self.model.register_forward_pre_hook(progress)
        finite_hook = self.model.register_forward_hook(check_model_output) if self.task != "depth" else None
        files = []
        try:
            if self.task == "generate":
                from iris3b.sampling import generate

                input_image = None
                if image_plan is not None:
                    from .img2img import image_pixels, resized_input

                    input_image = resized_input(request["image"], request["width"], request["height"])
                if image_plan is not None and image_plan.steps == 0:
                    image = input_image
                else:
                    atomic_json(directory / "progress.json", {"stage": "プロンプトを読み込み中"})
                    text = self.prepare_text(request)
                    self.model.cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if request.get("sampling", {}).get("mode") == "official_turbo":
                            from .official_turbo import generate as generate_official_turbo

                            pixels = generate_official_turbo(
                                self.model,
                                text,
                                [request["prompt"]],
                                request["height"],
                                request["width"],
                                request["sampling"]["sigmas"],
                                torch.Generator().manual_seed(request["seed"]),
                                num_train_timesteps=self.cfg.flow.num_train_timesteps,
                                prediction=self.cfg.flow.prediction,
                            )
                        elif request.get("sampling", {}).get("mode") == "turbo":
                            from .turbo import generate_turbo

                            if self.cfg.flow.prediction != "v" or self.cfg.flow.num_train_timesteps != 1000:
                                raise ValueError("Turbo LoRAには標準のIris flow設定が必要です。")
                            pixels = generate_turbo(
                                self.model,
                                text,
                                [request["prompt"]],
                                request["height"],
                                request["width"],
                                request["steps"],
                                request["sampling"]["shift"],
                                torch.Generator(device="cuda").manual_seed(request["seed"]),
                                cfg_scale=request["cfg"],
                                negative_prompt=request["negative_prompt"],
                            )
                        elif image_plan is not None:
                            from .img2img import generate_img2img

                            pixels = generate_img2img(
                                self.model,
                                text,
                                [request["prompt"]],
                                image_pixels(input_image),
                                image_plan,
                                cfg_scale=request["cfg"],
                                cfg_interval=tuple(self.cfg.sample.cfg_interval),
                                negative_prompt=request["negative_prompt"],
                                generator=torch.Generator(device="cuda").manual_seed(request["seed"]),
                                num_train_timesteps=self.cfg.flow.num_train_timesteps,
                                prediction=self.cfg.flow.prediction,
                            )
                        else:
                            pixels = generate(
                                self.model,
                                text,
                                [request["prompt"]],
                                height=request["height"],
                                width=request["width"],
                                steps=request["steps"],
                                order=2,
                                cfg_scale=request["cfg"],
                                cfg_interval=tuple(self.cfg.sample.cfg_interval),
                                shift=self.cfg.flow.shift,
                                negative_prompt=request["negative_prompt"],
                                generator=torch.Generator(device="cuda").manual_seed(request["seed"]),
                                num_train_timesteps=self.cfg.flow.num_train_timesteps,
                                prediction=self.cfg.flow.prediction,
                            )
                    if not torch.isfinite(pixels).all():
                        raise ValueError("生成結果に非有限値が含まれています。")
                    image = Image.fromarray(
                        ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                    )
                if image_plan is not None:
                    files.append(request["image"])
            else:
                with Image.open(request["image"]) as source:
                    image = source.convert("RGB")
                self.model.cuda()
                prompt = load_file(model_directory(self.root, self.precision, self.task) / "empty_prompt.safetensors")
                if self.task == "depth":
                    from iris3b.downstream.depth import DepthPredictor, colorize

                    predictor = DepthPredictor.__new__(DepthPredictor)
                    predictor.device, predictor.model, predictor.patch = (
                        torch.device("cuda"),
                        self.model,
                        self.cfg.model.patch_size,
                    )
                    predictor.embeddings, predictor.mask = prompt["embeddings"].cuda(), prompt["mask"].cuda()
                    depth = predictor(image, max_side=1024)
                    if not np.isfinite(depth).all():
                        raise ValueError("深度結果に非有限値が含まれています。")
                    np.save(directory / "depth.npy", depth, allow_pickle=False)
                    files.append(str(directory / "depth.npy"))
                    image = colorize(depth)
                else:
                    from iris3b.downstream.restoration import Restorer, fit_budget, tile_positions

                    restorer = Restorer.__new__(Restorer)
                    restorer.device, restorer.model, restorer.patch = (
                        torch.device("cuda"),
                        self.model,
                        self.cfg.model.patch_size,
                    )
                    restorer.embeddings, restorer.mask = prompt["embeddings"].cuda(), prompt["mask"].cuda()
                    restorer.tile, restorer.sigma = self.settings["tile"], self.settings["sigma"]
                    restorer.time = restorer.sigma * self.cfg.flow.num_train_timesteps
                    image = fit_budget(image)
                    total = len(tile_positions(image.height * 4, restorer.tile, restorer.tile // 2)) * len(
                        tile_positions(image.width * 4, restorer.tile, restorer.tile // 2)
                    )
                    image = restorer(image)
            if checkpoint_identity(self.root, self.precision, self.task, self.checkpoint) != self.source_identity:
                raise ValueError("実行中にモデルが更新されました。再実行してください。")
            verify_sources(loras)
            atomic_json(directory / "progress.json", {"stage": "結果を保存中"})
            image.save(directory / "result.png")
            files.insert(0, str(directory / "result.png"))
            torch.cuda.synchronize()
            result = {
                "status": "complete",
                "task": self.task,
                "precision": self.precision,
                "image": files[0],
                "files": files + [str(directory / "request.json")],
                "size": list(image.size),
                "seconds": time.perf_counter() - begin,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "seed": request.get("seed"),
            }
            metadata = {
                "task": self.task,
                "precision": self.precision,
                "source_repo": SOURCE_REPO if self.checkpoint == "base" else None,
                "source_revision": SOURCE_REVISION if self.checkpoint == "base" else None,
                "model_source": self.source_identity,
                "code_revision": CODE_REVISION,
                "text_revision": TEXT_REVISION if self.task == "generate" else None,
                "settings": {k: v for k, v in request.items() if k != "image"},
                "output_size": result["size"],
                "seconds": result["seconds"],
                "peak_allocated_mib": result["peak_allocated_mib"],
                "peak_reserved_mib": result["peak_reserved_mib"],
            }
            if request.get("image"):
                with Image.open(request["image"]) as input_image:
                    metadata["input"] = {
                        "name": "input.png",
                        "sha256": sha256(request["image"]),
                        "size": list(input_image.size),
                    }
            if image_plan is not None:
                metadata["schedule"] = image_plan.metadata()
                metadata["schedule"].update(
                    forward_calls=calls,
                    prediction=self.cfg.flow.prediction,
                    num_train_timesteps=self.cfg.flow.num_train_timesteps,
                    cfg_interval=list(self.cfg.sample.cfg_interval),
                )
                metadata["input"]["resize"] = {"mode": "resize", "resample": "bicubic", "size": result["size"]}
                result.update(image_mode=image_mode, strength=request["strength"], executed_steps=total)
            elif request.get("sampling", {}).get("mode") == "official_turbo":
                metadata["schedule"] = {
                    **request["sampling"],
                    "forward_calls": calls,
                    "prediction": self.cfg.flow.prediction,
                    "num_train_timesteps": self.cfg.flow.num_train_timesteps,
                }
            atomic_json(directory / "progress.json", {"stage": "終了処理中"})
        finally:
            hook.remove()
            if finite_hook:
                finite_hook.remove()
            self.offload()
        result["seconds"] = metadata["seconds"] = time.perf_counter() - begin
        atomic_json(directory / "metadata.json", metadata)
        result["files"].append(str(directory / "metadata.json"))
        atomic_json(directory / "result.json", result)
        return result
