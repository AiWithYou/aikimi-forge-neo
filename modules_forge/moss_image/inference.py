"""Native pinned Comfy MOSS Image inference, isolated from the Forge process."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from PIL import Image, ImageOps, PngImagePlugin

from modules_forge import gpu_fault

from . import checkpoints, core, lora, references


class Cancelled(RuntimeError):
    """The job was stopped before publishing a complete artifact."""
    public_error_code = "cancelled"


def require_finite(value, stage):
    import torch

    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"MOSS Image {stage} contains non-finite values")


@contextmanager
def import_arguments(*, cpu=False):
    previous = sys.argv
    sys.argv = [previous[0], "--fp32-vae"] + (["--cpu"] if cpu else [])
    try:
        yield
    finally:
        sys.argv = previous


def verify_module_origin(module, repository):
    filename = getattr(module, "__file__", None)
    locations = [filename] if filename else list(getattr(module, "__path__", []))
    repository = Path(repository).resolve()
    if not locations or any(not Path(path).resolve().is_relative_to(repository) for path in locations):
        raise RuntimeError("a Comfy module was imported from a different runtime")


def read_hf_weights(directory, stem, utils):
    """Merge verified local HF shards for the native Comfy component loader."""
    directory = core.safe_model_root(directory)
    index = directory / f"{stem}.safetensors.index.json"
    if not index.is_file():
        return utils.load_torch_file(str(directory / f"{stem}.safetensors"), safe_load=True)
    mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items())
    ):
        raise core.MossImageError("MOSS Imageの重み一覧が不正です。")
    state = {}
    for filename in sorted(set(mapping.values())):
        path = core.safe_model_root(directory / filename)
        if not path.is_relative_to(directory):
            raise core.MossImageError("MOSS Imageの重みファイルが保存先の外を参照しています。")
        values = utils.load_torch_file(str(path), safe_load=True)
        if values.keys() != {key for key, file in mapping.items() if file == filename}:
            raise core.MossImageError("MOSS Imageの重みと一覧が一致しません。")
        state.update(values)
    return state


def weight_sources(request):
    value = core.manifest()
    result = {}
    for name, selection in (("dit", request["precision"]), ("text", request["text_precision"])):
        if name == "dit" and request.get("checkpoint_identity"):
            result[name] = dict(request["checkpoint_identity"])
            continue
        if selection == "bf16":
            result[name] = {"repo_id": core.MODEL_ID, "revision": core.MODEL_REVISION, "precision": "bf16"}
        else:
            spec = value["assets"][name + "_" + selection]
            result[name] = {key: spec[key] for key in ("repo_id", "revision", "filename", "sha256")}
    return result


@contextmanager
def finite_vae_decode(vae):
    """Check raw decoder outputs before Comfy's output clamp can hide infinity."""
    model = getattr(vae, "first_stage_model", None)
    if model is None:
        yield
        return
    original = model.decode
    had_local = "decode" in model.__dict__

    def decode(*args, **kwargs):
        pixels = original(*args, **kwargs)
        require_finite(pixels, "raw VAE pixels")
        return pixels

    model.decode = decode
    try:
        yield
    finally:
        if had_local:
            model.decode = original
        else:
            del model.decode


class Runner:
    def __init__(self, root, runtime=None):
        self.root = Path(root)
        self._runtime = runtime

    def close(self):
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def run(self, directory, request):
        request = core.validate_request(request, internal=True)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        def check():
            if (directory / "cancel").exists():
                raise Cancelled("MOSS Image generation was cancelled")

        def progress(stage, current=0, total=1):
            check()
            core.atomic_json(directory / "progress.json", {"stage": stage, "current": current, "total": total})

        check()
        progress("verifying")
        if not core.model_ready(self.root, request):
            raise core.MossImageError(
                "選択したMOSS Imageの保存形式が未導入か検証できません。セットアップを実行してください。"
            )
        lora.bind(self.root, request)
        images = references.load(directory, request)
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
            references.load(directory, request)
            checkpoints.bind(self.root, request)
            if not isinstance(image, Image.Image) or image.size != (request["width"], request["height"]):
                raise RuntimeError("MOSS Image returned an invalid image")
            lora.bind(self.root, request)
            metadata = {
                **getattr(self._runtime, "last_metadata", {}),
                **request,
                "model_id": core.MODEL_ID,
                "model_revision": core.MODEL_REVISION,
                "comfy_revision": core.manifest()["runtime"]["comfy_revision"],
                "runtime_backend": "ComfyUI",
                "weight_sources": weight_sources(request),
                "vae_precision": "fp32",
                "lora": lora.record(request),
                "lora_applied_targets": getattr(self._runtime, "last_metadata", {}).get("lora_applied_targets", 0),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            info = PngImagePlugin.PngInfo()
            info.add_text("moss_image", json.dumps(metadata, ensure_ascii=False, allow_nan=False))
            image_path = directory / "output.png"
            image.save(temporary, format="PNG", pnginfo=info)
            check()
            os.replace(temporary, image_path)
            result = {**metadata, "image_path": str(image_path), "files": [str(image_path)]}
            core.atomic_json(directory / "result.json", result)
            progress("complete")
            completed = True
            return result
        finally:
            temporary.unlink(missing_ok=True)
            if not completed:
                (directory / "output.png").unlink(missing_ok=True)
                (directory / "result.json").unlink(missing_ok=True)


class ComfyRuntime:
    """Import only the dedicated, verified Comfy checkout; no server or extra nodes."""

    def __init__(self, root, *, cpu=False):
        self.root = core.safe_model_root(root)
        if not core.runtime_ready(self.root):
            raise core.MossImageError("MOSS Imageの専用環境をセットアップしてください。")
        repository = core.runtime_paths(self.root)["comfy"].resolve()
        sys.path.insert(0, str(repository))
        self.torch = importlib.import_module("torch")
        self.cpu = cpu
        if not cpu and not self.torch.cuda.is_available():
            raise core.MossImageError("MOSS ImageにはNVIDIA CUDA GPUが必要です。")

        def load(name):
            module = importlib.import_module(name)
            verify_module_origin(module, repository)
            return module

        with import_arguments(cpu=cpu):
            load("comfy")
            options = load("comfy.options")
            previous_parsing = options.args_parsing
            options.args_parsing = True
            try:
                cli = load("comfy.cli_args")
                if bool(cli.args.cpu) != bool(cpu):
                    raise RuntimeError("Comfy was initialized with a different device mode")
                self.management = load("comfy.model_management")
                self.sd = load("comfy.sd")
                self.utils = load("comfy.utils")
                self.sample_module = load("comfy.sample")
                self.samplers = load("comfy.samplers")
                self.mapper = load("comfy.lora")
                self.moss_model = load("comfy.ldm.moss_image.model")
                self.advanced = load("comfy_extras.nodes_model_advanced")
                self.slg = load("comfy_extras.nodes_slg")
            finally:
                options.args_parsing = previous_parsing
        self.last_metadata = {}
        self.failed = False
        self.policy = None
        if not cpu:
            from .safety import InferencePolicy

            self.policy = InferencePolicy.install(self.torch, load("comfy.ops"))
        self.kitchen_version = importlib.metadata.version("comfy-kitchen")

    def load_clip(self, path, precision, *, require_vision=False):
        if precision == "bf16":
            state = read_hf_weights(Path(path) / "text_encoder", "model", self.utils)
            options = {"dtype": self.torch.bfloat16}
        elif precision in ("int8", "w4a8"):
            state, metadata = self.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
            state, _ = self.utils.convert_old_quants(state, model_prefix="", metadata=metadata)
            options = {}
        else:
            raise core.MossImageError("テキストエンコーダーの保存形式が不正です。")
        normalized = {}
        for key, value in state.items():
            name = "model." + key if key.startswith(("language_model.", "visual.")) else key
            if name in normalized:
                raise core.MossImageError("MOSS Imageのテキスト重みの対象名が重複しています。")
            normalized[name] = value
        if self.sd.detect_te_model(normalized) != self.sd.TEModel.QWEN35_9B:
            raise core.MossImageError("テキストエンコーダーがMOSS用Qwen3.5-9Bと一致しません。")
        if require_vision and not any(key.startswith("model.visual.") for key in normalized):
            raise core.MossImageError("参照画像には画像理解用の重みを含むQwen3.5エンコーダーを導入してください。")
        clip = self.sd.load_text_encoder_state_dicts(
            [normalized], clip_type=self.sd.CLIPType.MOSS_IMAGE, model_options=options
        )
        if not hasattr(getattr(clip, "tokenizer", None), "qwen35_9b"):
            raise core.MossImageError(
                "MOSS Imageのトークナイザーを読み込めませんでした。実行環境をセットアップしてください。"
            )
        return clip

    def encode_text(self, clip, prompt, *, images=()):
        # Use the native underlying tokenizer before any higher-level cap. This
        # is the pinned MOSS chat template, including its raw-chat escape hatch.
        if images and prompt.startswith("<|im_start|>"):
            raise core.MossImageError("参照画像を使う場合は通常の指示文を入力してください。")
        image_prompt = "".join(
            f"Picture {index + 1}: <|vision_start|><|image_pad|><|vision_end|>" for index in range(len(images))
        )
        text = (
            prompt
            if prompt.startswith("<|im_start|>")
            else (f"<|im_start|>user\n{image_prompt}{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        )
        rows = clip.tokenizer.qwen35_9b.tokenize_with_weights(text, disable_weights=True)
        if not isinstance(rows, list) or not rows or any(not isinstance(row, list) for row in rows):
            raise RuntimeError("MOSS Image tokenizer returned invalid token rows")
        count = sum(len(row) for row in rows)
        image_index = 0
        for row in rows:
            for index, token in enumerate(row):
                if token[0] != 248056:
                    continue
                if image_index >= len(images):
                    raise core.MossImageError("画像プレースホルダーと参照画像の数が一致しません。")
                image = images[image_index]
                row[index] = ({"type": "image", "data": image, "original_type": "image"}, *token[1:])
                height, width = image.shape[1:3]
                nh, nw = round(height / 32) * 32, round(width / 32) * 32
                if nh * nw < 3136:
                    scale = math.sqrt(3136 / (height * width))
                    nh, nw = math.ceil(height * scale / 32) * 32, math.ceil(width * scale / 32) * 32
                count += nh * nw // 1024 - 1
                image_index += 1
        if image_index != len(images):
            raise core.MossImageError("参照画像の入力トークンを生成できませんでした。")
        if count > 4096:
            raise core.MossImageError("プロンプトが4096トークンを超えています。短くして再実行してください。")
        if count < 1:
            raise RuntimeError("MOSS Image tokenizer returned empty tokens")
        tokens = {"qwen35_9b": rows}
        with self.torch.inference_mode():
            encoded = clip.encode_from_tokens_scheduled(tokens)
        for tensor, metadata in encoded:
            require_finite(tensor, "text features")
            if tensor.shape[-2] > 4096:
                raise core.MossImageError("参照画像を含むプロンプトが4096トークンを超えています。短くしてください。")
            for value in metadata.values():
                require_finite(value, "text features")
        return [
            [
                tensor.cpu(),
                {key: value.cpu() if self.torch.is_tensor(value) else value for key, value in metadata.items()},
            ]
            for tensor, metadata in encoded
        ], count

    def load_dit(self, path, precision):
        if Path(path).is_relative_to(self.root / "checkpoints") and Path(path).is_file():
            state, metadata = self.utils.load_torch_file(str(path), return_metadata=True)
            for key, value in state.items():
                require_finite(value, key)
            options = {"dtype": self.torch.bfloat16} if precision == "bf16" else {}
            model = self.sd.load_diffusion_model_state_dict(state, model_options=options, metadata=metadata)
        elif precision == "bf16":
            state = read_hf_weights(Path(path) / "transformer", "diffusion_pytorch_model", self.utils)
            model = self.sd.load_diffusion_model_state_dict(state, model_options={"dtype": self.torch.bfloat16})
        elif precision == "int8":
            model = self.sd.load_diffusion_model(str(path))
        else:
            raise core.MossImageError("画像生成本体の保存形式が不正です。")
        if model is None or not isinstance(model.model.diffusion_model, self.moss_model.MOSSImage):
            raise core.MossImageError("読み込まれたモデルがMOSS Imageと一致しません。")
        return model

    def sampling_sigmas(self, model, request):
        steps = request["steps"]
        denoise = request.get("denoise", 1.0)
        total = steps if denoise >= 1 else max(steps, int(steps / denoise))
        return self.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "sgm_uniform", total)[
            -(steps + 1) :
        ].to(device="cpu", dtype=self.torch.float32)

    def sample(self, model, positive, negative, request, progress, check, *, initial_latent=None):
        torch = self.torch
        # Native MOSS uses Flux's timestep_embedding, which already multiplies
        # by 1000; its sampling configuration therefore has multiplier=1.
        model = self.advanced.ModelSamplingSD3().patch(model, shift=request["shift"], multiplier=1.0)[0]
        rope = request.get("rope", {"scale_x": 1.0, "scale_y": 1.0})
        if any(value != 1 for value in rope.values()):
            model.model_options.setdefault("transformer_options", {})["rope_options"] = dict(rope)
        if request["skip_layer"]:
            model = self.slg.SkipLayerGuidanceDiTSimple.execute(
                model,
                start_percent=0.0,
                end_percent=1.0,
                double_layers="",
                single_layers="22",
            )[0]
        sigmas = self.sampling_sigmas(model, request)
        self.last_metadata.update({"actual_steps": len(sigmas) - 1, "sampler": "euler", "scheduler": "sgm_uniform"})
        latent = torch.zeros(
            (1, 128, request["height"] // 16, request["width"] // 16), device="cpu", dtype=torch.float32
        )
        if initial_latent is not None:
            if initial_latent.shape != latent.shape:
                raise core.MossImageError("元画像の潜在サイズが出力サイズと一致しません。")
            latent = initial_latent.to(device="cpu", dtype=torch.float32)
            require_finite(latent, "input latent")
        noise = torch.randn(
            latent.shape,
            generator=torch.Generator(device="cpu").manual_seed(request["seed"]),
            device="cpu",
            dtype=torch.float32,
        )

        def sampled(step, denoised, current, total):
            check()
            require_finite(denoised, "denoised latent")
            require_finite(current, "Euler latent")
            progress("sampling", step + 1, total)

        check()
        with torch.inference_mode():
            result = self.sample_module.sample_custom(
                model,
                noise,
                request["cfg"],
                self.samplers.sampler_object("euler"),
                sigmas,
                positive,
                negative,
                latent,
                callback=sampled,
                disable_pbar=True,
                seed=request["seed"],
            )
        check()
        require_finite(result, "sampled latent")
        return result

    def load_vae(self, path):
        state, metadata = self.utils.load_torch_file(str(path), return_metadata=True)
        vae = self.sd.VAE(sd=state, metadata=metadata, dtype=self.torch.float32)
        vae.throw_exception_if_invalid()
        return vae

    def _vae_tile(self, vae, shape):
        budget = int(self.management.get_free_memory(vae.device) * 0.8)
        if vae.memory_used_decode(shape, vae.vae_dtype) <= budget:
            return None
        for tile in (64, 32, 16):
            peaks = []
            for height, width in ((tile, tile), (tile // 2, tile * 2), (tile * 2, tile // 2)):
                bounded = (*shape[:-2], min(shape[-2], height), min(shape[-1], width))
                peaks.append(vae.memory_used_decode(bounded, vae.vae_dtype))
            if max(peaks) <= budget:
                return tile
        raise RuntimeError("insufficient free memory for MOSS Image FP32 VAE decode")

    def decode(self, vae, latent):
        import numpy as np

        tile = self._vae_tile(vae, latent.shape)
        with self.torch.inference_mode(), finite_vae_decode(vae):
            pixels = (
                vae.decode(latent)
                if tile is None
                else vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=tile // 4)
            )[0].float()
        require_finite(pixels, "decoded pixels")
        return Image.fromarray(np.clip(pixels.cpu().numpy() * 255, 0, 255).astype(np.uint8), "RGB")

    def encode_image(self, vae, image):
        import numpy as np

        pixels = self.torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
        with self.torch.inference_mode():
            # The pinned native VAE encoder retries OOM with tiled encoding.
            latent = vae.encode(pixels).to(device="cpu", dtype=self.torch.float32)
        require_finite(latent, "upscaled VAE latent")
        return latent

    def sampling_record(self, request):
        return {
            **{key: request[key] for key in ("width", "height", "steps", "denoise", "seed", "cfg", "shift", "skip_layer")},
            **{key: self.last_metadata[key] for key in ("actual_steps", "sampler", "scheduler")},
            "rope": dict(request.get("rope", {"scale_x": 1.0, "scale_y": 1.0})),
        }

    def generate(self, request, progress, check, *, images=()):
        policy = getattr(self, "policy", None)
        with policy.enforce() if policy is not None else nullcontext():
            try:
                return self._generate(request, progress, check, images=images)
            finally:
                if policy is not None:
                    self.last_metadata["inference_policy"] = policy.snapshot()

    def _generate(self, request, progress, check, *, images=()):
        self.last_metadata = {
            "generator_device": "cpu",
            "initial_noise_precision": "fp32",
            "latent_update_precision": "fp32",
            "token_limit": 4096,
            "prompt_truncated": False,
            "comfy_kitchen_version": getattr(self, "kitchen_version", "unknown"),
            "dit_storage_precision": request["precision"],
            "text_storage_precision": request["text_precision"],
        }
        paths = core.asset_paths(self.root, request)
        try:
            check()
            vision, ref_latents, initial_latent = [], [], None
            if images:
                import numpy as np

                progress("encoding_references")
                prepared = (
                    [ImageOps.fit(images[0], (request["width"], request["height"]), Image.Resampling.LANCZOS)]
                    if request["edit_method"] == "redraw"
                    else references.prepare(images)
                )
                vision_images = (
                    [
                        image.resize(references.fit_size(image.size, 384**2, snap=32), Image.Resampling.LANCZOS)
                        for image in images
                    ]
                    if request["edit_method"] == "reference"
                    else []
                )
                vision = [
                    self.torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
                    for image in vision_images
                ]
                self.last_metadata.update(
                    reference_sizes=[list(image.size) for image in prepared],
                    vision_sizes=[list(image.size) for image in vision_images],
                )
                vae = self.load_vae(paths["vae"])
                with self.torch.inference_mode():
                    for image in prepared:
                        check()
                        pixels = self.torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
                        latent = vae.encode(pixels).cpu()
                        require_finite(latent, "reference VAE latent")
                        ref_latents.append(latent)
                self.offload()
                del vae
                if request["edit_method"] == "redraw":
                    initial_latent, ref_latents = ref_latents[0], []
                self.last_metadata["reference_latent_tokens"] = [
                    int(item.shape[-1] * item.shape[-2]) for item in ref_latents
                ]
            progress("loading_text_encoder")
            clip = (
                self.load_clip(paths["text_encoder"], request["text_precision"], require_vision=True)
                if vision
                else self.load_clip(paths["text_encoder"], request["text_precision"])
            )
            text_dtypes = getattr(getattr(clip, "cond_stage_model", None), "dtypes", set())
            self.last_metadata["text_compute_precision"] = (
                ",".join(sorted(str(dtype).removeprefix("torch.") for dtype in text_dtypes)) or "unknown"
            )
            progress("encoding_prompt")
            positive, count = (
                self.encode_text(clip, request["prompt"], images=vision)
                if vision
                else self.encode_text(clip, request["prompt"])
            )
            self.last_metadata["prompt_tokens"] = count
            self.last_metadata["text_conditioning_precision"] = str(positive[0][0].dtype).removeprefix("torch.")
            negative = positive
            if request["cfg"] != 1:
                check()
                negative, count = (
                    self.encode_text(clip, request["negative_prompt"], images=vision)
                    if vision
                    else self.encode_text(clip, request["negative_prompt"])
                )
                self.last_metadata["negative_prompt_tokens"] = count
            if ref_latents:
                for conditioning in (positive, negative):
                    for _, metadata in conditioning:
                        metadata["reference_latents"] = ref_latents
            self.offload()
            del clip
            check()
            progress("loading_diffusion")
            model = self.load_dit(paths["dit"], request["precision"])
            actual = getattr(model, "model", None)
            dtype = getattr(actual, "get_dtype_inference", getattr(actual, "get_dtype", lambda: "unknown"))()
            self.last_metadata.update(
                {
                    "dit_compute_precision": str(dtype).removeprefix("torch."),
                    "dit_patcher_class": type(model).__name__,
                    "dit_patcher_mode": "dynamic" if getattr(model, "is_dynamic", lambda: False)() else "standard",
                }
            )
            lora.bind(self.root, request)
            model, count = lora.apply(model, self.root, request, self.mapper, self.torch)
            self.last_metadata["lora_applied_targets"] = count
            if ref_latents:
                from vendor.moss_image_ostris_edit import MOSSImageOstrisEditModelPatch

                model = MOSSImageOstrisEditModelPatch().patch(model, kv_cache=request["kv_cache"])[0]
            hires = request.get("hires")
            first_request = (
                {**request, "width": hires["base_width"], "height": hires["base_height"]} if hires else request
            )
            progress("sampling", 0, first_request["steps"])
            latent = (
                self.sample(model, positive, negative, first_request, progress, check, initial_latent=initial_latent)
                if initial_latent is not None
                else self.sample(model, positive, negative, first_request, progress, check)
            )
            if hires:
                passes = [self.sampling_record(first_request)]
                self.offload()
                progress("hires_upscaling")
                check()
                vae = self.load_vae(paths["vae"])
                base_image = self.decode(vae, latent)
                check()
                enlarged = base_image.resize((request["width"], request["height"]), Image.Resampling.LANCZOS)
                progress("hires_encoding")
                check()
                initial_latent = self.encode_image(vae, enlarged)
                self.offload()
                del vae, base_image, enlarged
                check()
                finish_request = {**request, "steps": hires["steps"], "denoise": hires["denoise"]}

                def finish_progress(stage, current=0, total=1):
                    progress("hires_sampling" if stage == "sampling" else stage, current, total)

                progress("hires_sampling", 0, finish_request["steps"])
                latent = self.sample(
                    model, positive, negative, finish_request, finish_progress, check, initial_latent=initial_latent
                )
                passes.append(self.sampling_record(finish_request))
                self.last_metadata.update(
                    sampling_passes=passes,
                    actual_steps=sum(item["actual_steps"] for item in passes),
                    upscaler="lanczos",
                )
            require_finite(latent, "sampled latent")
            self.offload()
            del model
            check()
            progress("decoding")
            vae = self.load_vae(paths["vae"])
            image = self.decode(vae, latent)
            check()
            lora.bind(self.root, request)
            return image
        except BaseException as exc:
            if gpu_fault.classify(exc):
                self.failed = True
            raise
        finally:
            if not getattr(self, "failed", False):
                self.offload()

    def offload(self):
        if getattr(self, "failed", False):
            return
        try:
            self.management.unload_all_models()
            if not self.cpu and self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            self.management.soft_empty_cache()
        except BaseException as exc:
            if gpu_fault.classify(exc):
                self.failed = True
            raise

    def close(self):
        self.offload()
