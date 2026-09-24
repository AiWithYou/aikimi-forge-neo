"""Isolated, local-only Qwen-Image 2.1 Diffusers inference worker.

The resident process owns its model cache. No CUDA or model libraries are
imported until a validated job has acquired the shared GPU lease.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.qwen_image21.quantized_cache import INT8_SKIP_MODULES  # noqa: E402
from modules_forge.qwen_image21.regular_gguf import MODEL_ID as REGULAR_MODEL_ID  # noqa: E402
from modules_forge.qwen_image21.regular_gguf import REVISION as REGULAR_REVISION  # noqa: E402
from modules_forge.qwen_image21.turbo import PROFILES as TURBO_PROFILES  # noqa: E402

DIFFUSERS_REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
MODEL_ID = "Qwen/Qwen-Image-2.1"
PROFILES = {**TURBO_PROFILES, "base_q4_k_m": (REGULAR_MODEL_ID, REGULAR_REVISION)}
EVENT_PREFIX = "QWEN_IMAGE21_EVENT "
# Keep the small input/output and timestep projections in BF16. Attention and
# feed-forward projections inside every denoising block use LLM.int8().
_RESIDENT_RUNTIME: dict[str, Any] | None = None
_RESIDENT_KEY: tuple | None = None


class GenerationCancelled(RuntimeError):
    """A cooperative cancellation invalidates any partially used model cache."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    from modules_forge.yue2_studio.core import atomic_json

    atomic_json(path, payload)


def _progress(job: Path, stage: str, message: str, progress: float, **extra: Any) -> None:
    event = {"stage": stage, "message": message, "progress": max(0.0, min(1.0, progress)), **extra}
    _atomic_json(job / "progress.json", event)
    sys.stdout.write(EVENT_PREFIX + json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _check_cancel(job: Path) -> None:
    if (job / "cancel").exists():
        raise GenerationCancelled("Qwen-Image 2.1 generation cancelled.")


def clear_runtime() -> None:
    global _RESIDENT_RUNTIME, _RESIDENT_KEY
    _RESIDENT_RUNTIME = None
    _RESIDENT_KEY = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _cache_key(model_path: Path, precision: str, memory_mode: str) -> tuple:
    # Include weight shards as well as configs: overwriting files in the same
    # directory must not silently reuse an older resident checkpoint.
    metadata = tuple(
        (str(path.relative_to(model_path)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(model_path.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(model_path).parts
    )
    manifests = tuple(
        (name, path.stat().st_size, path.stat().st_mtime_ns)
        for name in ("runtime.json", "model-files.json", "turbo-files.json", "regular-gguf-files.json")
        if (path := model_path.parent / name).is_file()
    )
    turbo = tuple(
        (str(path.relative_to(model_path.parent)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted((model_path.parent / "turbo").rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(model_path.parent).parts
    )
    regular = tuple(
        (str(path.relative_to(model_path.parent)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted((model_path.parent / "regular-gguf").rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(model_path.parent).parts
    )
    return str(model_path), precision, memory_mode, metadata, manifests, turbo, regular


def _model_revision(model_path: Path) -> str | None:
    manifest = model_path.parent / "model-files.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("revision")


def _read_request(payload: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    job = Path(payload["job_dir"]).resolve()
    _check_cancel(job)
    request = json.loads((job / "request.json").read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request.json must contain an object.")
    operation = request.get("operation", "generate")
    if operation not in {"generate", "prepare"}:
        raise ValueError("Unsupported operation")
    for key, default, allowed in (
        ("precision", "int8", {"int8", "bf16", "w4a8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"}),
        ("memory_mode", "offload", {"offload", "gpu"}),
    ):
        value = payload.get(key, request.get(key, default))
        if value not in allowed:
            raise ValueError(f"Unsupported {key}: {value}")
        if key in payload and key in request and payload[key] != request[key]:
            raise ValueError(f"Worker payload and request disagree about {key}.")
        request[key] = value
    model_path = Path(payload["model_path"]).resolve()
    if not model_path.is_dir():
        raise ValueError(f"Local Qwen-Image 2.1 model directory was not found: {model_path}")
    index = json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("The local model is not a QwenImage21Pipeline checkpoint.")
    if not isinstance(request.get("prompt"), str) or not request["prompt"].strip():
        raise ValueError("A nonempty prompt is required.")
    for key in ("width", "height"):
        value = request.get(key, 1024)
        if isinstance(value, bool) or not isinstance(value, int) or value < 256 or value > 4096 or value % 32:
            raise ValueError(f"{key} must be a multiple of 32 between 256 and 4096.")
        request[key] = value
    steps = request.get("steps", 40)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer between 1 and 100.")
    request["steps"] = steps
    if request["precision"].startswith("turbo_"):
        if steps != 4:
            raise ValueError("Viggle Turbo requires exactly 4 steps.")
        from modules_forge.qwen_image21.turbo import turbo_manifest

        turbo_manifest(model_path.parent, request["precision"])
    elif request["precision"] == "base_q4_k_m":
        from modules_forge.qwen_image21.regular_gguf import regular_manifest

        regular_manifest(model_path.parent)
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2^63 - 1.")
    request["seed"] = seed
    if not isinstance(request.get("transparent", False), bool):
        raise ValueError("transparent must be a boolean.")
    if not isinstance(request.get("rewrite_prompt", False), bool):
        raise ValueError("rewrite_prompt must be a boolean.")
    if not isinstance(request.get("rewrite_edit_prompt", False), bool):
        raise ValueError("rewrite_edit_prompt must be a boolean.")
    if not isinstance(request.get("preserve_unmasked", False), bool):
        raise ValueError("preserve_unmasked must be a boolean.")
    images = request.get("input_images", [])
    if not isinstance(images, list) or len(images) > 10:
        raise ValueError("input_images must contain at most ten local image paths.")
    for path in images:
        if not isinstance(path, str) or not Path(path).is_absolute() or not Path(path).is_file():
            raise ValueError("Each input image must be an existing absolute local file path.")
    request["input_images"] = images
    if operation == "prepare" and request["precision"] not in {"int8", "w4a8"}:
        raise ValueError("Only INT8 and W4A8 models need conversion")
    return job, model_path, request


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "diffusers", "transformers", "accelerate", "bitsandbytes")
    }


def _int8_layers(model: Any) -> int:
    from bitsandbytes.nn import Linear8bitLt

    return sum(isinstance(module, Linear8bitLt) for module in model.modules())


def _int8_offload_apply(layer, fn, recurse=True):
    """Move bitsandbytes 0.50.2's unregistered tensors with the parent model.

    Parent Module.to() recurses through _apply(), bypassing Linear8bitLt.to().
    PyTorch's default parameter conversion can also keep the original
    Int8Params attributes while replacing only its .data. Preserve whether
    quantization state has moved from weight to state (first forward), and
    re-alias CB to the moved weight instead of allocating a second weight copy.
    """
    import torch

    weight_cb = layer.weight.CB is not None
    weight_scb = layer.weight.SCB
    state_tensors = {name: value for name, value in vars(layer.state).items() if isinstance(value, torch.Tensor)}
    result = torch.nn.Module._apply(layer, fn, recurse=recurse)
    device = layer.weight.device
    layer.weight.CB = layer.weight.data if weight_cb else None
    layer.weight.SCB = weight_scb.to(device=device) if weight_scb is not None else None
    for name, value in state_tensors.items():
        # Scales remain FP32; a device move must not cast quantization metadata.
        setattr(layer.state, name, layer.weight.data if name == "CB" else value.to(device=device))
    return result


def _install_int8_offload_fix(model: Any) -> None:
    from bitsandbytes.nn import Linear8bitLt

    # Confined to this worker's instances and its validated 0.50.2 dependency.
    # No shared library files or global torch/bitsandbytes classes are changed.
    for module in model.modules():
        if isinstance(module, Linear8bitLt):
            module._apply = MethodType(_int8_offload_apply, module)


def _load_runtime(model_path: Path, request: dict[str, Any], job: Path) -> dict[str, Any]:
    # Disable Hub traffic before importing libraries as a second guard behind
    # local_files_only=True. Setup/downloads are a separate, explicit action.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
        os.environ[name] = "1"
    from modules_forge.qwen_image21_environment import validate_running_versions

    validate_running_versions()
    if request["precision"] in {"base_q4_k_m", "turbo_q4_k_m"}:
        try:
            gguf_version = importlib.metadata.version("gguf")
        except importlib.metadata.PackageNotFoundError:
            gguf_version = None
        if gguf_version != "0.19.0":
            raise RuntimeError("Q4_K_Mにはgguf==0.19.0が必要です。aikimi-qwen-image21-setup.bat を実行してください。")
    _check_cancel(job)
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from transformers import Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen-Image 2.1 requires a CUDA GPU with BF16 support.")
    _progress(job, "loading", "Qwen-Image 2.1 モデルを読み込み中", 0.05)
    started = time.monotonic()
    components: dict[str, Any] = {}
    counts: dict[str, int] = {}
    w4a8_stats: dict[str, dict] = {}
    disk_cache: dict[str, dict] = {}
    from modules_forge.qwen_image21.quantized_cache import (
        InvalidQuantizedCheckpoint,
        component_identity,
        load_or_create,
    )

    if request["precision"] in {"int8", "base_q4_k_m", "turbo_q4_k_m"}:
        loaders = [
            ("text_encoder", Qwen3VLForConditionalGeneration, TransformersBitsAndBytesConfig(load_in_8bit=True)),
        ]
        if request["precision"] == "int8":
            loaders.insert(
                0,
                (
                    "transformer",
                    QwenImage21Transformer2DModel,
                    DiffusersBitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=list(INT8_SKIP_MODULES),
                    ),
                ),
            )
        for name, model_class, quantization in loaders:
            _check_cancel(job)
            _progress(job, "loading", f"{name} を INT8 で読み込み中", 0.08 if name == "transformer" else 0.17)
            identity = component_identity(
                model_path, name, "int8", skip_modules=INT8_SKIP_MODULES if name == "transformer" else ()
            )

            def load_int8(folder, *, cached=False, model_class=model_class, name=name, quantization=quantization):
                loaded = model_class.from_pretrained(
                    str(folder),
                    **({} if cached else {"subfolder": name, "quantization_config": quantization}),
                    torch_dtype=torch.bfloat16,
                    device_map={"": "cuda:0"},
                    local_files_only=True,
                    use_safetensors=True,
                )
                counts[name] = _int8_layers(loaded)
                if counts[name] == 0 or not getattr(loaded, "is_loaded_in_8bit", False):
                    raise InvalidQuantizedCheckpoint(f"{name} was not loaded as bitsandbytes INT8.")
                _install_int8_offload_fix(loaded)
                # Keep quantization scales aliased correctly across save/reload
                # and GPU -> CPU -> GPU, before the next component is loaded.
                loaded.to("cpu")
                return loaded

            component, disk_cache[name] = load_or_create(
                model_path,
                identity,
                lambda path: load_int8(path, cached=True),
                lambda: load_int8(model_path),
                lambda model, path: model.save_pretrained(path, safe_serialization=True, max_shard_size="2GB"),
                check_cancel=lambda: _check_cancel(job),
                progress=lambda message, stage_end=0.17 if name == "transformer" else 0.27: _progress(
                    job, "loading", message, stage_end
                ),
            )
            components[name] = component
            torch.cuda.empty_cache()
            _check_cancel(job)
        if request["precision"] in {"base_q4_k_m", "turbo_q4_k_m"}:
            if request["precision"] == "base_q4_k_m":
                from modules_forge.qwen_image21.regular_gguf import load_gguf_transformer

                label = "通常版"
            else:
                from modules_forge.qwen_image21.turbo import load_gguf_transformer

                label = "Turbo"
            _progress(job, "loading", f"{label} Q4_K_M GGUF Transformerを読み込み中", 0.27)
            components["transformer"] = load_gguf_transformer(model_path)
            _check_cancel(job)
    elif request["precision"] == "w4a8":
        from modules_forge.qwen_image21.w4a8 import load_model, load_saved_model, save_model, validate_dependency

        validate_dependency()
        for name, model_class in (
            ("transformer", QwenImage21Transformer2DModel),
            ("text_encoder", Qwen3VLForConditionalGeneration),
        ):
            _check_cancel(job)
            component_progress = 0.08 if name == "transformer" else 0.18
            _progress(job, "loading", f"{name} の W4A8 モデルを確認中", component_progress)
            folder = model_path / name
            if name == "transformer":
                config = model_class.load_config(str(folder), local_files_only=True)
                factory = partial(model_class.from_config, config)
            else:
                config = model_class.config_class.from_pretrained(str(folder), local_files_only=True)
                factory = partial(model_class, config)

            last_progress = -1

            def packing_progress(done, total, component_name=name, base_progress=component_progress):
                nonlocal last_progress
                _check_cancel(job)
                if done != last_progress and (done % 16 == 0 or done == total):
                    _progress(
                        job, "loading", f"{component_name} W4A8変換 {done}/{total}", base_progress + 0.10 * done / total
                    )
                    last_progress = done

            identity = component_identity(model_path, name, "w4a8")
            saved, disk_cache[name] = load_or_create(
                model_path,
                identity,
                partial(load_saved_model, component=name, factory=factory, check_cancel=lambda: _check_cancel(job)),
                partial(load_model, folder, name, factory, progress=packing_progress),
                lambda result, path: save_model(result[0], path, result[1], check_cancel=lambda: _check_cancel(job)),
                check_cancel=lambda: _check_cancel(job),
                progress=lambda message, stage_end=component_progress + 0.10: _progress(
                    job, "loading", message, stage_end
                ),
            )
            component, w4a8_stats[name] = saved
            components[name] = component
            torch.cuda.empty_cache()
            _check_cancel(job)
    elif request["precision"] == "turbo_bf16":
        _progress(job, "loading", "Turbo BF16 Transformerを読み込み中", 0.10)
        components["transformer"] = QwenImage21Transformer2DModel.from_pretrained(
            str(model_path.parent / "turbo" / "bf16"),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
        )
        _check_cancel(job)
    pipe = QwenImage21Pipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
        **components,
    )
    if request["precision"].startswith("turbo_"):
        from diffusers import FlowMatchEulerDiscreteScheduler

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(model_path.parent / "turbo"), subfolder="scheduler", local_files_only=True
        )
        if pipe.scheduler.config.shift_terminal is not None:
            raise RuntimeError("Turbo scheduler must have shift_terminal=null.")
    _check_cancel(job)
    if request["memory_mode"] == "offload":
        pipe.enable_model_cpu_offload(gpu_id=0, device="cuda")
    else:
        pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    _check_cancel(job)
    versions = _versions()
    if request["precision"] in {"base_q4_k_m", "turbo_q4_k_m"}:
        versions["gguf"] = importlib.metadata.version("gguf")
    if w4a8_stats:
        versions["comfy-kitchen"] = importlib.metadata.version("comfy-kitchen")
    return {
        "pipe": pipe,
        "int8_layers": counts,
        "w4a8": w4a8_stats,
        "disk_cache": disk_cache,
        "versions": versions,
        "load_seconds": time.monotonic() - started,
    }


def _runtime_for_request(model_path: Path, request: dict[str, Any], job: Path) -> tuple[dict[str, Any], bool]:
    global _RESIDENT_RUNTIME, _RESIDENT_KEY
    key = _cache_key(model_path, request["precision"], request["memory_mode"])
    if _RESIDENT_RUNTIME is not None and key == _RESIDENT_KEY:
        _progress(job, "loaded", "読み込み済みモデルを再利用", 0.30)
        return _RESIDENT_RUNTIME, True
    clear_runtime()
    _RESIDENT_RUNTIME = _load_runtime(model_path, request, job)
    _RESIDENT_KEY = key
    return _RESIDENT_RUNTIME, False


def _load_images(paths: list[str]) -> list[Any]:
    from PIL import Image, ImageOps

    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGBA"))
    return images


def _rewrite_for_request(model_path: Path, request: dict[str, Any], job: Path) -> dict:
    editing = bool(request["input_images"])
    enabled = request.get("rewrite_edit_prompt" if editing else "rewrite_prompt", False)
    if not enabled:
        legacy_edit = editing and request.get("rewrite_prompt", False)
        return {"enabled": bool(legacy_edit), "applied": False, "reason": "image_edit" if legacy_edit else "disabled"}
    from modules_forge.qwen_image21.prompt_rewriter import rewrite_prompt

    # Preserve a reusable image pipeline on CPU while the small helper owns CUDA.
    # The same GPU lease covers rewriting, diffusion, cancellation, and cleanup.
    restore_gpu = False
    if _RESIDENT_RUNTIME is not None:
        if _RESIDENT_KEY[2] == "offload":
            _RESIDENT_RUNTIME["pipe"].maybe_free_model_hooks()
        else:
            _RESIDENT_RUNTIME["pipe"].to("cpu")
            restore_gpu = True
    result = rewrite_prompt(
        model_path.parent,
        request["prompt"].strip(),
        request["width"],
        request["height"],
        request["seed"],
        lambda: _check_cancel(job),
        lambda message: _progress(job, "rewriting", message, 0.03),
        **({"images": _load_images(request["input_images"])} if editing else {}),
    )
    _check_cancel(job)
    if restore_gpu:
        _RESIDENT_RUNTIME["pipe"].to("cuda:0")
    return result


def _idle_cuda_memory(torch, memory_mode: str) -> dict:
    """Release only unused allocator blocks; resident model tensors stay intact."""
    if not torch.cuda.is_initialized():
        return {}
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if memory_mode == "offload":
        torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    return {
        "peak_allocated_mib": round(peak / 2**20, 1),
        "peak_reserved_mib": round(peak_reserved / 2**20, 1),
        "idle_allocated_mib": round(allocated / 2**20, 1),
        "idle_reserved_before_mib": round(before / 2**20, 1),
        "idle_reserved_after_mib": round(after / 2**20, 1),
        "released_cache_mib": round(max(0, before - after) / 2**20, 1),
        "note": "PyTorch allocator only; peak includes this request's model loading; not total device memory or system RAM.",
    }


def _bf16_safetensors_bytes(path: Path) -> int:
    """Read tensor sizes from a single-file checkpoint without loading weights."""
    with path.open("rb") as source:
        prefix = source.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < length <= 64 * 1024**2:
            raise ValueError("Invalid safetensors header size.")
        header = json.loads(source.read(length))
    total = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        dtype = tensor["dtype"]
        if dtype == "BF16" or dtype.startswith("F"):
            # from_pretrained(torch_dtype=bfloat16) casts floating weights;
            # checkpoint file size is not the resident size for FP32 files.
            total += math.prod(tensor["shape"]) * 2
        else:
            start, end = tensor["data_offsets"]
            total += end - start
    return total


def _check_bf16_gpu_capacity(model_path: Path, request: dict[str, Any], torch) -> None:
    """Reject provably impossible residency before loading multi-GB weights.

    This is a weights-only lower bound, not a promise that activations fit.
    Use total capacity so a resident pipeline's own allocation is not counted
    against it again when processing the next request.
    """
    if request["memory_mode"] != "gpu" or request["precision"] not in {"bf16", "turbo_bf16"}:
        return
    transformer = model_path / "transformer"
    if request["precision"] == "turbo_bf16":
        transformer = model_path.parent / "turbo" / "bf16" / "transformer"
    indexes = (
        transformer / "diffusion_pytorch_model.safetensors.index.json",
        model_path / "text_encoder" / "model.safetensors.index.json",
    )
    weight_bytes = 0
    for index in indexes:
        if index.is_file():
            metadata = json.loads(index.read_text(encoding="utf-8")).get("metadata", {})
            size = metadata.get("total_size", 0)
            if isinstance(size, int) and size > 0:
                weight_bytes += size
        else:
            single_file = index.with_name(index.name.removesuffix(".index.json"))
            if single_file.is_file():
                weight_bytes += _bf16_safetensors_bytes(single_file)
    capacity = torch.cuda.get_device_properties(0).total_memory
    if weight_bytes > capacity:
        raise RuntimeError(
            f"BF16 GPU常駐には重みだけで {weight_bytes / 2**30:.1f} GiB 以上が必要です。"
            f"GPU容量 {capacity / 2**30:.1f} GiB を超えるため読み込みを中止しました。"
            "CPUオフロード (offload) または量子化モデルを選んでください。"
        )


def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    job = Path(payload["job_dir"]).resolve()
    runtime = None
    try:
        job, model_path, request = _read_request(payload)
        _check_cancel(job)
        if request.get("operation") == "prepare":
            from modules_forge.qwen_image21.quantized_cache import saved_components

            _progress(job, "loading", "保存済みモデルを確認中", 0.05)
            try:
                disk_cache = saved_components(model_path, request["precision"], check_cancel=lambda: _check_cancel(job))
            except (FileNotFoundError, ValueError, KeyError, TypeError):
                # Re-enter the existing loader to create only missing/invalid
                # components; valid components never pass through quantization.
                clear_runtime()
                runtime, _ = _runtime_for_request(model_path, request, job)
                disk_cache = saved_components(model_path, request["precision"], check_cancel=lambda: _check_cancel(job))
            _check_cancel(job)
            result = {"operation": "prepare", "precision": request["precision"], "quantized_cache": disk_cache}
            _atomic_json(job / "result.json", result)
            _progress(job, "complete", "モデルを保存しました。次回からこのモデルを使います。", 1.0)
            return result
        import torch

        _check_bf16_gpu_capacity(model_path, request, torch)
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        rewrite = _rewrite_for_request(model_path, request, job)
        runtime, reused = _runtime_for_request(model_path, request, job)
        _check_cancel(job)

        # Qwen's VAE otherwise decodes the entire 2K latent in one CUDA pass.
        # Tiling is supported by the pinned Qwen VAE and reduces decode peaks.
        vae_tiling = request["width"] * request["height"] > 1280 * 1024
        if vae_tiling:
            # The VAE's 256/192 defaults leave visible grid artifacts in 2K
            # images. Larger tiles with 50% overlap passed same-latent QA.
            runtime["pipe"].vae.enable_tiling(
                tile_sample_min_height=512,
                tile_sample_min_width=512,
                tile_sample_stride_height=256,
                tile_sample_stride_width=256,
            )
        elif getattr(getattr(runtime["pipe"], "vae", None), "use_tiling", False):
            runtime["pipe"].vae.disable_tiling()

        images = _load_images(request["input_images"])
        prompt = rewrite["rewritten_prompt"] if rewrite["applied"] else request["prompt"].strip()
        if request.get("transparent", False):
            prompt = (
                f"This is an RGBA image with transparency. {prompt}. "
                "The image has alpha channel and the background is transparent."
            )
        generator = torch.Generator(device="cpu").manual_seed(request["seed"])

        def on_step(_pipe, step: int, _timestep, callback_kwargs: dict) -> dict:
            _check_cancel(job)
            if step + 1 == request["steps"]:
                _progress(job, "decoding", "画像へ変換中", 0.93)
            else:
                _progress(
                    job,
                    "sampling",
                    f"生成中 {step + 1}/{request['steps']}",
                    0.32 + 0.60 * (step + 1) / request["steps"],
                )
            return callback_kwargs

        _progress(job, "sampling", "プロンプトと参照画像を処理中", 0.30)
        sampling_started = time.monotonic()
        output = runtime["pipe"](
            prompt=prompt,
            image=images or None,
            width=request["width"],
            height=request["height"],
            num_inference_steps=request["steps"],
            true_cfg_scale=1.0,
            use_kv_cache=True,
            output_resolution=1024,
            generator=generator,
            num_images_per_prompt=1,
            output_type="pil",
            return_dict=True,
            callback_on_step_end=on_step,
            callback_on_step_end_tensor_inputs=[],
        )
        sampling_seconds = time.monotonic() - sampling_started
        _check_cancel(job)
        image = output.images[0]
        if image.size != (request["width"], request["height"]):
            raise RuntimeError(
                f"Unexpected output size {image.size}; requested {(request['width'], request['height'])}."
            )
        if image.mode != "RGBA":
            raise RuntimeError(f"Qwen-Image 2.1 must return native RGBA output, received {image.mode}.")
        _progress(job, "saving", "RGBA PNG を保存中", 0.96)
        output_path = job / "output.png"
        partial = output_path.with_name(output_path.name + ".part")
        image.save(partial, format="PNG")
        _check_cancel(job)
        os.replace(partial, output_path)
        from modules_forge.qwen_image21.annotations import save_preserved_output

        preservation = save_preserved_output(request, image, job)
        _check_cancel(job)
        memory = _idle_cuda_memory(torch, request["memory_mode"])
        metadata = {
            "model": PROFILES[request["precision"]][0] if request["precision"] in PROFILES else MODEL_ID,
            "model_path": str(model_path),
            "model_revision": PROFILES[request["precision"]][1]
            if request["precision"] in PROFILES
            else _model_revision(model_path),
            "diffusers_revision": DIFFUSERS_REVISION,
            "prompt": request["prompt"],
            "effective_prompt": prompt,
            "prompt_rewrite": rewrite,
            "preservation": preservation.get("preservation", {}),
            "width": image.width,
            "height": image.height,
            "steps": request["steps"],
            "seed": request["seed"],
            "transparent": request.get("transparent", False),
            "output_mode": image.mode,
            "precision": request["precision"],
            "memory_mode": request["memory_mode"],
            "compute_dtype": "bfloat16",
            "true_cfg_scale": 1.0,
            "use_kv_cache": True,
            "input_resolution": 1024,
            "input_image_count": len(images),
            "input_image_names": [Path(path).name for path in request["input_images"]],
            "int8_layers": runtime["int8_layers"],
            "w4a8": runtime.get("w4a8", {}),
            "quantized_cache": runtime.get("disk_cache", {}),
            "int8_skip_modules": list(INT8_SKIP_MODULES) if request["precision"] == "int8" else [],
            "versions": runtime["versions"],
            "reused_model": reused,
            "memory": memory,
            "vae_tiling": vae_tiling,
            "vae_tiling_config": {"tile_size": 512, "stride": 256} if vae_tiling else None,
            "timings": {
                "load_seconds": 0.0 if reused else round(runtime["load_seconds"], 3),
                "sampling_seconds": round(sampling_seconds, 3),
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        metadata_path = job / "metadata.json"
        _atomic_json(metadata_path, metadata)
        _check_cancel(job)
        result = {
            "output_path": str(output_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata,
            **preservation,
        }
        _atomic_json(job / "result.json", result)
        _progress(job, "complete", "生成が完了しました", 1.0)
        return result
    except BaseException as exc:
        runtime = None
        clear_runtime()
        # A late cancellation must never leave a success marker for the service.
        (job / "result.json").unlink(missing_ok=True)
        _progress(job, "cancelled" if isinstance(exc, GenerationCancelled) else "error", str(exc), 0.0)
        raise


def resident_run(payload: dict[str, Any]) -> dict[str, Any]:
    return run_request(payload)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Forge Neo Qwen-Image 2.1 worker")
    parser.add_argument("--request", required=True, help="Path to the worker payload JSON.")
    args = parser.parse_args()
    try:
        run_request(json.loads(Path(args.request).read_text(encoding="utf-8")))
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        clear_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
