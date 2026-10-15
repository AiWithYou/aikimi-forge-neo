"""Direct, isolated ComfyUI FLUX 3 inference; no server or shared Forge imports."""

import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, PngImagePlugin

from . import conditioning
from .paths import canonical_path


class Cancelled(RuntimeError):
    """The job was cancelled before a completed result could be published."""


def require_finite(value, stage):
    """Reject numerical failures before they can become plausible PNG pixels."""
    import torch

    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"FLUX 3 {stage} contains non-finite values")


def model_paths(root, request):
    from .core import model_paths as resolve

    return resolve(root, request)


def official_sigmas(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 200:
        raise ValueError("steps must be an integer in 1..200")
    times = [1 - step / steps for step in range(steps + 1)]
    return [6.93 * t / (1 + 5.93 * t) for t in times]


def _presence_scores(scores, seen, penalty):
    result = scores.float().clone()
    if penalty and seen:
        result[..., sorted(set(seen))] -= penalty
    return result


def greedy_with_presence(scores, seen, penalty):
    """vLLM's output-token presence penalty, applied before greedy argmax."""
    return _presence_scores(scores, seen, penalty).argmax(dim=-1)


@contextmanager
def greedy_logits(model, penalty, check):
    """Pinned Comfy's greedy branch bypasses penalties; adjust only this PUP instance.

    MTP must be disabled by the caller. Passing presence_penalty=0 to generate
    avoids a second penalty if upstream later changes its greedy implementation.
    """
    original = model.logits
    had_local = "logits" in model.__dict__
    seen = []

    def logits(hidden):
        check()
        scores = _presence_scores(original(hidden), seen, penalty)
        require_finite(scores, "prompt-upscaler logits")
        seen.append(int(scores.argmax(dim=-1).reshape(-1)[0].item()))
        return scores

    model.logits = logits
    try:
        yield
    finally:
        if had_local:
            model.logits = original
        else:
            del model.logits


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def verify_module_origin(module, repository):
    """Comfy's top-level package is a namespace package without __file__."""
    filename = getattr(module, "__file__", None)
    locations = [filename] if filename else list(getattr(module, "__path__", []))
    try:
        repository = canonical_path(repository)
        if not locations or any(not canonical_path(path).is_relative_to(repository) for path in locations):
            raise ValueError("module outside the pinned checkout")
    except (OSError, ValueError):
        raise RuntimeError("a Comfy module was imported from a different runtime") from None


class Runner:
    def __init__(self, root, runtime=None):
        self.root = Path(root)
        self._runtime = runtime

    def offload(self):
        if self._runtime is not None:
            self._runtime.offload()

    def close(self):
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def run(self, directory, request):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        def check():
            if (directory / "cancel").exists():
                raise Cancelled("FLUX 3 generation was cancelled")

        def progress(stage, current=0, total=1):
            check()
            _write_json(directory / "progress.json", {"stage": stage, "current": current, "total": total})

        check()
        model_kind = request.get("model", "distilled")
        if model_kind not in ("base", "distilled"):
            raise ValueError("model must be base or distilled")
        steps = request.get("steps", 50 if model_kind == "base" else 16)
        sigmas = official_sigmas(steps)
        cfg = request.get("cfg", 3.5 if model_kind == "base" else 1.0)
        if model_kind == "distilled" and cfg != 1:
            raise ValueError("distilled FLUX 3 samples without CFG (cfg=1)")
        seed = request.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        prompt = request.get("prompt", "")
        boxes = request.get("boxes") or []
        if boxes:
            conditioning.to_canvas(boxes)
        paths = model_paths(self.root, request)
        images, references = [], []
        for index, path in enumerate(request.get("images") or []):
            check()
            path = Path(path)
            with Image.open(path) as opened:
                image = opened.copy()
                image.info = opened.info.copy()
            images.append(image)
            references.append(
                {"index": index, "name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        prepared = conditioning.prepare_images(images, request.get("width"), request.get("height"))
        if self._runtime is None:
            progress("initializing")
            self._runtime = ComfyRuntime(self.root)
        runtime = self._runtime
        completed = False
        try:
            runtime.ensure_precision(request.get("precision", "int8"))
            runtime.verify_checkpoint(paths["dit"], model_kind)
            caption = prompt
            upsampled = False
            if request.get("pup", False) and not (not boxes and conditioning.is_finished(prompt)):
                progress("upsampling")
                spec = conditioning.pup_request(prompt, images, boxes)
                prompts = json.loads(Path(paths["pup_prompts"]).read_text(encoding="utf-8"))
                pup_clip = runtime.load_clip(paths["pup"], "pup")
                caption = runtime.upsample(pup_clip, spec, prompts[spec.system_key], check).strip()
                if not caption:
                    raise RuntimeError("the prompt upsampler returned an empty caption")
                if spec.locked:
                    caption = conditioning.reconcile(caption, spec.locked)
                upsampled = True
                runtime.offload()
                del pup_clip
            elif boxes:
                caption = f"{prompt.rstrip()} {json.dumps(conditioning.to_canvas(boxes), ensure_ascii=False)}"
            progress("encoding_text")
            clip = runtime.load_clip(paths["text_encoder"], "flux3")
            positive = runtime.encode_text(clip, caption)
            negative = runtime.encode_text(clip, "") if model_kind == "base" else None
            runtime.offload()
            del clip
            ref_latents = []
            vae = None
            if prepared.references:
                vae = runtime.load_vae(paths["vae"])
                for index, image in enumerate(prepared.references):
                    progress("encoding_references", index, len(prepared.references))
                    ref_latents.append(runtime.encode_reference(vae, image))
                runtime.offload()
            progress("loading_model")
            model = runtime.load_dit(paths["dit"], model_kind)
            progress("sampling", 0, steps)
            latent = runtime.sample(
                model,
                positive,
                negative,
                ref_latents,
                prepared.width,
                prepared.height,
                sigmas,
                cfg,
                seed,
                lambda step, total: progress("sampling", step + 1, total),
            )
            require_finite(latent, "sampled latent")
            check()
            runtime.offload()
            del model
            progress("decoding")
            if vae is None:
                vae = runtime.load_vae(paths["vae"])
            image = runtime.decode(vae, latent)
            if image.size != (prepared.width, prepared.height):
                raise RuntimeError("the decoded image size does not match the sampling canvas")
            if prepared.crop is not None:
                image = image.crop(prepared.crop)
            check()
            # Release the GPU before publishing the success marker/result JSON.
            runtime.offload()
            from . import core

            record = {
                "model": model_kind,
                "precision": request.get("precision", "int8"),
                "model_revision": core.MODEL_REVISION,
                "comfy_revision": core.COMFY_REVISION,
                "input_prompt": prompt,
                "prompt": caption,
                "boxes": boxes,
                "width": image.width,
                "height": image.height,
                "sampling_width": prepared.width,
                "sampling_height": prepared.height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "upsampled": upsampled,
                "references": references,
                "reference_hashes": [reference["sha256"] for reference in references],
            }
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("flux3_klein", json.dumps(record, ensure_ascii=False, allow_nan=False))
            metadata.add_text("Software", "Forge Neo / FLUX 3 Image Klein")
            temporary = directory / "output.tmp.png"
            image.save(temporary, format="PNG", pnginfo=metadata)
            check()
            os.replace(temporary, directory / "output.png")
            result = {
                **record,
                "image_path": str((directory / "output.png").resolve()),
                "result_path": str((directory / "result.json").resolve()),
            }
            check()
            _write_json(directory / "result.json", result)
            progress("complete", 1, 1)
            completed = True
            return result
        finally:
            try:
                runtime.offload()
            except BaseException:
                completed = False
                raise
            finally:
                if not completed:
                    for name in ("output.tmp.png", "output.png", "result.json", "result.json.tmp"):
                        (directory / name).unlink(missing_ok=True)


class ComfyRuntime:
    """The pinned Comfy checkout is imported only inside the dedicated worker."""

    def __init__(self, root):
        from .core import comfy_path

        self.root = canonical_path(root)
        repository = canonical_path(comfy_path(self.root))
        if not (repository / "comfy" / "ldm" / "flux3" / "model.py").is_file():
            raise RuntimeError("the pinned FLUX 3 Comfy runtime has not been installed")
        sys.path.insert(0, str(repository))
        self.torch = importlib.import_module("torch")
        self.nodes = importlib.import_module("nodes")
        for name in ("nodes", "comfy"):
            module = importlib.import_module(name)
            verify_module_origin(module, repository)
        self.management = importlib.import_module("comfy.model_management")
        self.sd = importlib.import_module("comfy.sd")
        self.utils = importlib.import_module("comfy.utils")
        self.sample_module = importlib.import_module("comfy.sample")
        self.samplers = importlib.import_module("comfy.samplers")
        folder_paths = importlib.import_module("folder_paths")
        for category in ("diffusion_models", "text_encoders", "vae"):
            folder_paths.add_model_folder_path(category, str(self.root / category), is_default=True)

    def ensure_precision(self, precision):
        if precision not in ("int8", "bf16", "nvfp4"):
            raise ValueError("unsupported FLUX 3 precision")
        if precision == "nvfp4":
            if (
                getattr(self.torch.version, "hip", None)
                or not self.torch.cuda.is_available()
                or self.torch.cuda.get_device_capability()[0] < 10
            ):
                raise ValueError("official NVFP4 inference requires an NVIDIA Blackwell GPU")

    def verify_checkpoint(self, path, model):
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as file:
            keys = set(file.keys())
        marker = "ref_time_in.in_layer.weight"
        if (marker in keys) != (model == "distilled") or "txt_in.weight" not in keys:
            raise ValueError("the checkpoint structure does not match the selected FLUX 3 variant")

    def load_clip(self, path, kind):
        loader = self.nodes.NODE_CLASS_MAPPINGS["CLIPLoader"]()
        return loader.load_clip(Path(path).name, type="stable_diffusion" if kind == "pup" else "flux3")[0]

    def upsample(self, clip, spec, system_prompt, check):
        import numpy as np

        tensors = [self.torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255 for image in spec.images]
        # Leading <|im_start|> makes native Qwen35 tokenize this as a raw chat
        # transcript; the generic Comfy thinking suffix is not appended.
        text = conditioning.pup_chat_text(spec, system_prompt)
        tokens = clip.tokenize(text, images=tensors, thinking=False, min_length=1)
        transformer = clip.cond_stage_model.qwen35_9b.transformer
        with self.torch.inference_mode(), greedy_logits(transformer, spec.presence_penalty, check):
            generated = clip.generate(
                tokens,
                do_sample=False,
                max_length=spec.max_tokens,
                temperature=1.0,
                top_k=20,
                top_p=0.95,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                seed=0,
                mtp=False,
            )
        caption = clip.decode(generated).strip()
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if caption.endswith(stop):
                caption = caption[: -len(stop)].rstrip()
        return caption

    def encode_text(self, clip, prompt):
        tokens = conditioning.bound_text_tokens(clip.tokenize(prompt))
        with self.torch.inference_mode():
            encoded = clip.encode_from_tokens_scheduled(tokens)
        for tensor, metadata in encoded:
            require_finite(tensor, "text features")
            for value in metadata.values():
                require_finite(value, "text features")
        return [
            [
                tensor.cpu(),
                {key: value.cpu() if self.torch.is_tensor(value) else value for key, value in metadata.items()},
            ]
            for tensor, metadata in encoded
        ]

    def load_vae(self, path):
        # The official VAE and its normalization statistics stay FP32. Casting
        # a default BF16 VAE back afterward would not restore rounded values.
        path = str(path)
        state, metadata = self.utils.load_torch_file(path, return_metadata=True)
        vae = self.sd.VAE(sd=state, metadata=metadata, dtype=self.torch.float32)
        vae.throw_exception_if_invalid()
        vae.patcher.cached_patcher_init = (self.sd.load_vae_patcher, (path, metadata, self.torch.float32))
        return vae

    def encode_reference(self, vae, image):
        import numpy as np

        pixels = self.torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
        tile = self._vae_tile(vae, pixels.movedim(-1, 1).shape, "encode", (1024, 512, 256))
        with self.torch.inference_mode():
            if tile is None:
                latent = vae.encode(pixels)
            else:
                latent = vae.encode_tiled(pixels, tile_x=tile, tile_y=tile, overlap=tile // 4)
        require_finite(latent, "reference latent")
        return latent.cpu()

    def load_dit(self, path, model):
        loaded = self.nodes.NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(Path(path).name, "default")[0]
        actual = loaded.model.diffusion_model
        if actual.__class__.__module__ != "comfy.ldm.flux3.model" or actual.flow_map != (model == "distilled"):
            raise ValueError("Comfy loaded a different diffusion architecture")
        return loaded

    def sample(self, model, positive, negative, references, width, height, sigmas, cfg, seed, callback):
        torch = self.torch

        def with_references(values):
            return [[tensor, {**metadata, "reference_latents": references}] for tensor, metadata in values]

        positive = with_references(positive)
        negative = with_references(negative) if negative is not None else positive
        height, width = height // 16, width // 16
        latent = torch.zeros((1, 128, height, width), device="cpu", dtype=torch.float32)
        # The reference draws token-major noise on CPU and rounds values to BF16.
        noise = torch.randn((height * width, 128), generator=torch.Generator().manual_seed(seed))
        noise = noise.to(torch.bfloat16).float().T.reshape(1, 128, height, width)
        schedule = torch.tensor(sigmas, dtype=torch.float32)

        def sampled(step, denoised, x, total):
            callback(step, total)

        with torch.inference_mode():
            return self.sample_module.sample_custom(
                model,
                noise,
                cfg,
                self.samplers.sampler_object("euler"),
                schedule,
                positive,
                negative,
                latent,
                callback=sampled,
                disable_pbar=True,
                seed=seed,
            )

    def decode(self, vae, latent):
        import numpy as np

        tile = self._vae_tile(vae, latent.shape, "decode", (64, 32, 16))
        with self.torch.inference_mode():
            if tile is None:
                pixels = vae.decode(latent)[0].float()
            else:
                pixels = vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=tile // 4)[0].float()
        require_finite(pixels, "decoded pixels")
        pixels = pixels.cpu().numpy()
        return Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8), "RGB")

    def _vae_tile(self, vae, shape, operation, sizes):
        # Windows can spill to shared memory before raising OOM. Decide before
        # attempting the full FP32 forward, retaining headroom for VAE weights.
        estimate = getattr(vae, f"memory_used_{operation}")
        budget = int(self.management.get_free_memory(vae.device) * 0.8)
        if estimate(shape, vae.vae_dtype) <= budget:
            return None
        for tile in sizes:
            peaks = []
            # The native 2D tiler averages square, wide and tall passes.
            for height, width in ((tile, tile), (tile // 2, tile * 2), (tile * 2, tile // 2)):
                bounded = (*shape[:-2], min(shape[-2], height), min(shape[-1], width))
                peaks.append(estimate(bounded, vae.vae_dtype))
            if max(peaks) <= budget:
                return tile
        raise RuntimeError(f"insufficient free memory for FLUX 3 FP32 VAE {operation}")

    def offload(self):
        self.management.unload_all_models()
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.management.soft_empty_cache()

    def close(self):
        self.offload()
