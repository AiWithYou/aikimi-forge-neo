"""One-command Krea2 native -> 4K preflight -> 8K detail-rich generation."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from urllib import request as urlrequest

import cv2
import numpy as np
from PIL import Image, PngImagePlugin


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.krea2_highres import (
    KREA2_DETAIL_PROMPT_SUFFIX,
    KREA2_NATIVE_PROMPT_PROFILES,
    krea2_detail_prompt,
    krea2_native_detail_prompt,
    krea2_vram_canvas_profile,
)
from modules_forge.krea2_upscale import native_diffusion_long_edge

DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "krea2_smart_8k"
NATIVE_SIZE = (1024, 1448)
RAW_NATIVE_SIZE = (720, 1024)
PREFLIGHT_SIZE = (2896, 4096)
FINAL_SIZE = (5792, 8192)
KREA2_INFERENCE_PROFILES = {
    "turbo-fast": {
        "model_profile": "turbo",
        "native_size": NATIVE_SIZE,
        "steps": 4,
        "sampler": "DPM++ 2M SDE",
        "scheduler": "Simple",
        "cfg": 1.0,
        "shift": 1.15,
        "provenance": "measured-local-fast",
    },
    "turbo-official": {
        "model_profile": "turbo",
        "native_size": NATIVE_SIZE,
        "steps": 8,
        "sampler": "Euler",
        "scheduler": "Simple",
        "cfg": 1.0,
        "shift": 1.15,
        "provenance": "official-settings-forge-cfg-mapping",
    },
    "raw-official": {
        "model_profile": "raw",
        "native_size": RAW_NATIVE_SIZE,
        "steps": 52,
        "sampler": "Euler",
        "scheduler": "Simple",
        "cfg": 3.5,
        "shift": None,
        "provenance": "official-settings-resolution-derived-shift",
    },
}
STAGE_PROFILES = {
    "4k": krea2_vram_canvas_profile("dense_detail_4k"),
    "8k": krea2_vram_canvas_profile("dense_detail_8k"),
}

DETAIL_REFINEMENT_SUFFIX = KREA2_DETAIL_PROMPT_SUFFIX


def emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_seed(seed: int) -> int:
    if seed < -1 or seed >= 2**32:
        raise ValueError("--seed must be -1 or a non-negative 32-bit integer")
    return secrets.randbelow(2**32) if seed == -1 else seed


def detail_prompt(base_prompt: str) -> str:
    return krea2_detail_prompt(base_prompt)


def native_detail_prompt(base_prompt: str, profile: str = "generic") -> str:
    return krea2_native_detail_prompt(base_prompt, profile=profile)


def raw_resolution_shift(width: int, height: int) -> float:
    """Match Krea2 Raw's official resolution-derived flow timestep shift."""

    if width <= 0 or height <= 0:
        raise ValueError("native dimensions must be greater than 0")
    alignment = 16
    aligned_width = ((int(width) + alignment - 1) // alignment) * alignment
    aligned_height = ((int(height) + alignment - 1) // alignment) * alignment
    sequence_length = (aligned_width // alignment) * (aligned_height // alignment)
    x1 = (256 // alignment) ** 2
    x2 = (1280 // alignment) ** 2
    y1, y2 = 0.5, 1.15
    slope = (y2 - y1) / (x2 - x1)
    return slope * sequence_length + (y1 - slope * x1)


def resolve_inference_profile(
    *,
    profile: str,
    model_profile: str | None,
    native_width: int | None,
    native_height: int | None,
    native_steps: int | None,
    sampler: str | None,
    scheduler: str | None,
    cfg: float | None,
    distilled_cfg: float | None,
) -> dict[str, object]:
    """Resolve one truthful fixed profile, or require every custom sampling value."""

    custom_values = {
        "--model-profile": model_profile,
        "--native-width": native_width,
        "--native-height": native_height,
        "--native-steps": native_steps,
        "--sampler": sampler,
        "--scheduler": scheduler,
        "--cfg": cfg,
        "--distilled-cfg": distilled_cfg,
    }
    if profile == "custom":
        missing = [name for name, value in custom_values.items() if value is None]
        if missing:
            raise ValueError(
                "--inference-profile custom requires explicit " + ", ".join(missing)
            )
        resolved = {
            "model_profile": str(model_profile),
            "native_size": (int(native_width), int(native_height)),
            "steps": int(native_steps),
            "sampler": str(sampler),
            "scheduler": str(scheduler),
            "cfg": float(cfg),
            "shift": float(distilled_cfg),
            "provenance": "user-specified-custom",
        }
    else:
        try:
            fixed = KREA2_INFERENCE_PROFILES[profile]
        except KeyError as exc:
            choices = ", ".join((*sorted(KREA2_INFERENCE_PROFILES), "custom"))
            raise ValueError(
                f"unknown Krea2 inference profile {profile!r}; choose {choices}"
            ) from exc
        sampling_overrides = {
            name: value
            for name, value in custom_values.items()
            if name
            not in {
                "--native-width",
                "--native-height",
            }
            and value is not None
        }
        if sampling_overrides:
            names = ", ".join(sampling_overrides)
            raise ValueError(
                f"{profile} is a fixed inference profile; {names} would make its name "
                "misleading. Use --inference-profile custom."
            )
        if (native_width is None) != (native_height is None):
            raise ValueError("--native-width and --native-height must be passed together")
        resolved = dict(fixed)
        if native_width is not None:
            resolved["native_size"] = (int(native_width), int(native_height))

    width, height = resolved["native_size"]
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("native dimensions must be greater than 0")
    if width % 8 or height % 8:
        raise ValueError("native dimensions must be divisible by 8")
    resolved_model_profile = str(resolved["model_profile"])
    if resolved_model_profile not in {"raw", "turbo"}:
        raise ValueError("--model-profile must be raw or turbo")
    native_limit = native_diffusion_long_edge(resolved_model_profile)
    if max(width, height) > native_limit:
        raise ValueError(
            f"native size {width}x{height} exceeds the {resolved_model_profile} "
            f"profile's {native_limit}px long-edge limit"
        )
    steps = int(resolved["steps"])
    if steps <= 0:
        raise ValueError("--native-steps must be greater than 0")
    if not str(resolved["sampler"]).strip() or not str(resolved["scheduler"]).strip():
        raise ValueError("sampler and scheduler must not be empty")
    cfg_value = float(resolved["cfg"])
    if not np.isfinite(cfg_value) or cfg_value < 0:
        raise ValueError("--cfg must be finite and 0 or greater")
    if profile == "raw-official":
        resolved["shift"] = raw_resolution_shift(width, height)
    shift = float(resolved["shift"])
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError("--distilled-cfg (Krea2 shift) must be finite and greater than 0")
    resolved.update(
        {
            "profile": profile,
            "native_size": (width, height),
            "steps": steps,
            "cfg": cfg_value,
            "shift": shift,
        }
    )
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def image_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return {
            "path": str(path),
            "size": list(image.size),
            "mode": image.mode,
            "metadata_keys": sorted(image.info),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }


def save_qa_crops(path: Path, output_dir: Path, crop_edge: int = 1024) -> list[dict]:
    """Save deterministic native-resolution crops for zoom-level visual QA."""

    if crop_edge <= 0:
        raise ValueError("crop_edge must be greater than 0")
    anchors = (
        ("upper_left", 0.25, 0.25),
        ("upper_center", 0.50, 0.28),
        ("upper_right", 0.75, 0.25),
        ("center", 0.50, 0.50),
        ("lower_left", 0.25, 0.78),
        ("lower_center", 0.50, 0.78),
        ("lower_right", 0.75, 0.78),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        edge = min(int(crop_edge), image.width, image.height)
        for name, x_fraction, y_fraction in anchors:
            left = min(
                max(0, round(image.width * x_fraction - edge / 2)),
                image.width - edge,
            )
            top = min(
                max(0, round(image.height * y_fraction - edge / 2)),
                image.height - edge,
            )
            box = (left, top, left + edge, top + edge)
            crop_path = output_dir / f"{name}.png"
            image.crop(box).save(crop_path, format="PNG", compress_level=6)
            records.append(
                {
                    "name": name,
                    "source_box": list(box),
                    **image_record(crop_path),
                }
            )
    return records


def require_size(path: Path, expected: tuple[int, int]) -> None:
    with Image.open(path) as image:
        actual = image.size
    if actual != expected:
        raise RuntimeError(f"{path} is {actual}; expected {expected}")


def highres_resolution_plan(
    source_size: tuple[int, int], source_stage: str
) -> dict[str, tuple[int, int]]:
    """Preserve source aspect and make the 8K stage an exact 2x of approved 4K."""

    width, height = (int(source_size[0]), int(source_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be greater than 0")
    long_edge = max(width, height)
    if source_stage == "native":
        if long_edge > 2048:
            raise ValueError(
                f"native source {width}x{height} exceeds the supported 2048px long edge"
            )
        if width >= height:
            preflight = (4096, max(1, int(height * 4096 / width)))
        else:
            preflight = (max(1, int(width * 4096 / height)), 4096)
    elif source_stage == "4k":
        if not 3840 <= long_edge <= 4096:
            raise ValueError(
                "--source-stage 4k requires a source whose long edge is 3840..4096"
            )
        preflight = (width, height)
    else:
        raise ValueError("source stage must be native or 4k")
    final = (preflight[0] * 2, preflight[1] * 2)
    if max(final) > 8192:
        raise ValueError(f"8K target {final[0]}x{final[1]} exceeds the 8192px edge limit")
    return {"source": (width, height), "preflight_4k": preflight, "final_8k": final}


def post_json(api: str, endpoint: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(
        f"{api.rstrip('/')}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{endpoint} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def get_json(api: str, endpoint: str, timeout: int) -> dict:
    request = urlrequest.Request(f"{api.rstrip('/')}{endpoint}", method="GET")
    with urlrequest.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{endpoint} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def portable_basename(value: object) -> str:
    """Return a filename for either Windows or POSIX path text on any host OS."""

    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def validate_krea2_backend(
    options: dict,
    runtime_status: dict,
    expected_model_profile: str,
    *,
    verify_checkpoint_variant: bool = True,
) -> dict:
    """Require the loaded Krea2 engine, transformer, modules, and model variant."""

    if not runtime_status.get("loaded"):
        raise RuntimeError("Forge reports that no diffusion model is loaded")
    architecture = str(runtime_status.get("architecture") or "")
    if architecture != "backend.diffusion_engine.krea.Krea2":
        raise RuntimeError(
            "loaded Forge architecture is not Krea2: "
            f"{architecture or '<unknown>'}"
        )
    transformer = str(runtime_status.get("transformer") or "")
    if transformer != "backend.nn.krea.SingleStreamDiT":
        raise RuntimeError(
            "loaded Forge transformer is not Krea2 SingleStreamDiT: "
            f"{transformer or '<unknown>'}"
        )

    selected_checkpoint = str(options.get("sd_model_checkpoint") or "")
    loaded_checkpoint = str(runtime_status.get("checkpoint") or "")
    checkpoint = loaded_checkpoint or selected_checkpoint
    if not checkpoint:
        raise RuntimeError("Forge did not report a loaded checkpoint")
    if selected_checkpoint and loaded_checkpoint:
        selected_name = portable_basename(selected_checkpoint).casefold()
        loaded_name = portable_basename(loaded_checkpoint).casefold()
        if selected_name != loaded_name:
            raise RuntimeError(
                "selected and loaded Forge checkpoints differ: "
                f"{portable_basename(selected_checkpoint)} != {portable_basename(loaded_checkpoint)}"
            )
    if expected_model_profile not in {"raw", "turbo"}:
        raise ValueError("expected model profile must be raw or turbo")
    normalized_checkpoint = "".join(
        character for character in portable_basename(checkpoint).lower() if character.isalnum()
    )
    if verify_checkpoint_variant and expected_model_profile not in normalized_checkpoint:
        raise RuntimeError(
            f"{expected_model_profile} inference profile requires a checkpoint whose "
            f"filename identifies that variant: {portable_basename(checkpoint)}"
        )

    additional = runtime_status.get("additional_modules") or options.get(
        "forge_additional_modules"
    ) or []
    if isinstance(additional, str):
        additional = [additional]
    module_names = [portable_basename(value) for value in additional]
    normalized_modules = [
        "".join(character for character in name.lower() if character.isalnum())
        for name in module_names
    ]
    if not any("qwenimagevae" in name for name in normalized_modules):
        raise RuntimeError("Krea2 requires qwen_image_vae in Forge additional modules")
    if not any("qwen3vl" in name for name in normalized_modules):
        raise RuntimeError("Krea2 requires a qwen3vl text encoder in Forge additional modules")
    return {
        "checkpoint": portable_basename(checkpoint),
        "checkpoint_hash": str(
            runtime_status.get("checkpoint_sha256")
            or options.get("sd_checkpoint_hash")
            or ""
        ),
        "model_profile": expected_model_profile,
        "architecture": architecture,
        "model_config": str(runtime_status.get("configuration") or ""),
        "transformer": transformer,
        "additional_modules": module_names,
        "quantization": runtime_status.get("quantization") or {},
    }


def decode_image(value: str) -> Image.Image:
    payload = value.split(",", 1)[-1]
    with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
        return image.convert("RGB")


def generate_native(
    *,
    api: str,
    output: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    cfg: float,
    distilled_cfg: float,
    size: tuple[int, int],
    timeout: int,
) -> dict:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "steps": steps,
        "cfg_scale": cfg,
        "distilled_cfg_scale": distilled_cfg,
        "width": size[0],
        "height": size[1],
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": True,
        "save_images": False,
    }
    started = time.perf_counter()
    data = post_json(api, "/sdapi/v1/txt2img", payload, timeout)
    elapsed = time.perf_counter() - started
    images = data.get("images") or []
    if len(images) != 1:
        raise RuntimeError(f"native txt2img returned {len(images)} images")
    image = decode_image(images[0])
    if image.size != size:
        raise RuntimeError(f"native txt2img returned {image.size}; expected {size}")
    try:
        info = json.loads(data.get("info") or "{}")
    except json.JSONDecodeError:
        info = {}
    infotexts = info.get("infotexts") or []
    parameters = str(infotexts[0]) if infotexts else ""
    pnginfo = PngImagePlugin.PngInfo()
    if parameters:
        pnginfo.add_text("parameters", parameters)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", pnginfo=pnginfo)
    return {
        **image_record(output),
        "duration_seconds": elapsed,
        "seed": seed,
        "steps": steps,
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": cfg,
        "distilled_cfg": distilled_cfg,
    }


def analysis_metrics(path: Path, long_edge: int = 1536) -> dict[str, object]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    scale = long_edge / max(image.size)
    analysis_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    if image.size != analysis_size:
        image = image.resize(analysis_size, Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.hypot(gx, gy)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    high = np.abs(gray - low)
    metrics: dict[str, object] = {
        "analysis_size": [image.width, image.height],
        "laplacian_variance": float(np.var(laplacian)),
        "gradient_p95": float(np.percentile(gradient, 95)),
        "highpass_abs_mean": float(np.mean(high)),
        "highpass_abs_p95": float(np.percentile(high, 95)),
    }
    normalized_long_edge = 4096
    with Image.open(path) as opened:
        normalized = opened.convert("L")
    normalized_scale = normalized_long_edge / max(normalized.size)
    normalized_size = (
        max(1, round(normalized.width * normalized_scale)),
        max(1, round(normalized.height * normalized_scale)),
    )
    if normalized.size != normalized_size:
        normalized = normalized.resize(normalized_size, Image.Resampling.LANCZOS)
    normalized_gray = np.asarray(normalized, dtype=np.float32)
    bands: dict[str, dict[str, float]] = {}
    previous = normalized_gray
    for inner_sigma, outer_sigma in ((0, 1), (1, 2), (2, 4), (4, 8)):
        blurred = cv2.GaussianBlur(
            normalized_gray,
            (0, 0),
            sigmaX=float(outer_sigma),
            sigmaY=float(outer_sigma),
        )
        band = np.abs(previous - blurred)
        bands[f"sigma_{inner_sigma}_{outer_sigma}"] = {
            "abs_mean": float(np.mean(band)),
            "abs_p95": float(np.percentile(band, 95)),
            "abs_p99": float(np.percentile(band, 99)),
            "active_percent_ge_1": float(np.mean(band >= 1.0) * 100.0),
        }
        previous = blurred
    metrics["detail_analysis_size"] = [normalized.width, normalized.height]
    metrics["normalized_multiband"] = bands
    return metrics


def detail_retention_gate(
    source: dict[str, object],
    candidate: dict[str, object],
    *,
    minimum_ratio: float,
    maximum_ratio: float = 1.8,
) -> dict[str, float]:
    if not np.isfinite(minimum_ratio) or minimum_ratio <= 0.0 or minimum_ratio > 1.0:
        raise ValueError("minimum detail-retention ratio must be in (0, 1]")
    if (
        not np.isfinite(maximum_ratio)
        or maximum_ratio <= 1.0
        or maximum_ratio <= minimum_ratio
    ):
        raise ValueError("maximum detail-retention ratio must be greater than 1")
    ratios = {
        key: float(candidate[key]) / max(float(source[key]), 1e-6)
        for key in ("gradient_p95", "highpass_abs_mean", "highpass_abs_p95")
    }
    if ratios["gradient_p95"] < minimum_ratio:
        raise RuntimeError(
            "detail-retention gate failed: "
            f"gradient_p95 ratio {ratios['gradient_p95']:.3f} < {minimum_ratio:.3f}"
        )
    if ratios["highpass_abs_p95"] < minimum_ratio:
        raise RuntimeError(
            "detail-retention gate failed: "
            f"highpass_abs_p95 ratio {ratios['highpass_abs_p95']:.3f} < {minimum_ratio:.3f}"
        )
    for key, ratio in ratios.items():
        if ratio > maximum_ratio:
            raise RuntimeError(
                "detail-retention gate failed: "
                f"{key} ratio {ratio:.3f} > {maximum_ratio:.3f}; "
                "possible noise or oversharpening"
            )
    return ratios


def redacted_command(command: list[str]) -> str:
    hidden_after = {"--prompt", "--negative-prompt"}
    values: list[str] = []
    hide_next = False
    for value in command:
        if hide_next:
            values.append("<redacted>")
            hide_next = False
            continue
        values.append(value)
        hide_next = value in hidden_after
    return subprocess.list2cmdline(values)


def run_command(
    command: list[str],
    label: str,
    timeout: int,
    *,
    interrupt_api: str | None = None,
    telemetry_path: Path | None = None,
    telemetry_interval: float = 1.0,
) -> list[str]:
    if timeout <= 0:
        raise ValueError("subprocess timeout must be greater than 0")
    emit(f"{label}_COMMAND={redacted_command(command)}")
    started = time.perf_counter()
    started_utc = utc_now()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                output_queue.put(raw_line.rstrip("\r\n"))
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    telemetry_stop = threading.Event()
    telemetry_samples: list[dict[str, float | int]] = []

    def sample_gpu() -> None:
        while not telemetry_stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    ),
                    check=False,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    fields = completed.stdout.splitlines()[0].split(",")
                    if len(fields) >= 3:
                        telemetry_samples.append(
                            {
                                "elapsed_seconds": round(
                                    time.perf_counter() - started, 3
                                ),
                                "memory_used_mib": int(fields[0].strip()),
                                "utilization_percent": int(fields[1].strip()),
                                "temperature_c": int(fields[2].strip()),
                            }
                        )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            telemetry_stop.wait(max(float(telemetry_interval), 0.25))

    telemetry_thread = None
    if telemetry_path is not None:
        telemetry_thread = threading.Thread(target=sample_gpu, daemon=True)
        telemetry_thread.start()
    lines: list[str] = []
    output_closed = False
    failure: BaseException | None = None
    exit_code: int | None = None
    try:
        while not output_closed or process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError(f"{label} exceeded the {timeout}s hard timeout")
            try:
                item = output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                output_closed = True
                continue
            lines.append(item)
            emit(f"{label}: {item}")
        exit_code = process.wait(timeout=10)
    except BaseException as exc:
        if isinstance(exc, TimeoutError) and interrupt_api:
            try:
                post_json(interrupt_api, "/sdapi/v1/interrupt", {}, 5)
            except Exception as interrupt_error:
                emit(f"{label}_INTERRUPT_ERROR={type(interrupt_error).__name__}")
        if process.poll() is None:
            process.kill()
        exit_code = process.wait(timeout=10)
        failure = exc
    finally:
        telemetry_stop.set()
        if telemetry_thread is not None:
            telemetry_thread.join(timeout=6)
        reader.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
    if exit_code is None:
        exit_code = process.returncode
    if exit_code is None:
        raise RuntimeError(f"{label} ended without a subprocess exit code")
    elapsed = time.perf_counter() - started
    emit(f"{label}_EXIT={exit_code} DURATION={elapsed:.3f}")
    if telemetry_path is not None:
        telemetry = {
            "format_version": 1,
            "label": label,
            "command": redacted_command(command),
            "started_at_utc": started_utc,
            "completed_at_utc": utc_now(),
            "duration_seconds": elapsed,
            "subprocess_exit_code": exit_code,
            "error": (
                {"type": type(failure).__name__, "message": str(failure)}
                if failure is not None
                else None
            ),
            "sample_interval_seconds": max(float(telemetry_interval), 0.25),
            "sample_count": len(telemetry_samples),
            "max_memory_used_mib": max(
                (item["memory_used_mib"] for item in telemetry_samples),
                default=None,
            ),
            "max_utilization_percent": max(
                (item["utilization_percent"] for item in telemetry_samples),
                default=None,
            ),
            "max_temperature_c": max(
                (item["temperature_c"] for item in telemetry_samples),
                default=None,
            ),
            "samples": telemetry_samples,
        }
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text(
            json.dumps(telemetry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines.append(f"{label}_TELEMETRY={telemetry_path}")
        emit(lines[-1])
    if failure is not None:
        raise failure
    if exit_code != 0:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(f"{label} failed with exit code {exit_code}\n{tail}")
    return lines


def emitted_path(lines: list[str], key: str) -> Path:
    prefix = f"{key}="
    for line in reversed(lines):
        if line.startswith(prefix):
            path = Path(line[len(prefix) :].strip())
            return path if path.is_absolute() else ROOT / path
    raise RuntimeError(f"subprocess output did not contain {prefix}")


def vram_canvas_command(
    args: argparse.Namespace,
    *,
    source: Path,
    output_root: Path,
    prompt: str,
    target: tuple[int, int],
    stage: str,
) -> list[str]:
    if stage not in STAGE_PROFILES:
        raise ValueError(f"unsupported stage: {stage}")
    profile = STAGE_PROFILES[stage]
    command = [
        sys.executable,
        str(ROOT / "tools" / "vram_canvas_highres.py"),
        "--input",
        str(source),
        "--api",
        args.api,
        "--output-root",
        str(output_root),
        "--prompt",
        prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--width",
        str(target[0]),
        "--height",
        str(target[1]),
        "--seed",
        str(args.seed),
        "--phase-count",
        str(args.phase_count),
        "--vram-budget-gib",
        str(args.vram_budget_gib),
        "--steps",
        str(profile["maximum_steps"]),
        "--minimum-steps",
        str(profile["minimum_steps"]),
        "--detail-knee",
        str(profile["detail_knee"]),
        "--coarse-denoise",
        str(profile["coarse_denoise"]),
        "--denoise",
        str(profile["denoise"]),
        "--low-pass-radius",
        str(profile["low_pass_radius"]),
        "--detail-gain",
        str(profile["detail_gain"]),
        "--max-detail-delta",
        str(profile["max_detail_delta"]),
        "--structure-sigma",
        str(profile["structure_sigma"]),
        "--base-detail-sigma",
        str(profile["base_detail_sigma"]),
        "--consensus-sigma",
        str(profile["consensus_sigma"]),
        "--novel-detail-gain",
        str(profile["novel_detail_gain"]),
        "--novel-detail-max-delta",
        str(profile["novel_detail_max_delta"]),
        "--timeout",
        str(args.timeout),
        "--progress-interval",
        str(args.progress_interval),
        "--no-progress-timeout",
        str(args.no_progress_timeout),
    ]
    if args.tile_size > 0:
        command.extend(["--tile-size", str(args.tile_size)])
    return command


def finish_command(
    args: argparse.Namespace,
    *,
    source: Path,
    output: Path,
    report: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "krea2_smart_finish.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
        "--color-strength",
        str(args.color_strength),
        "--analysis-long-edge",
        str(args.analysis_long_edge),
        "--detail-guard",
        "--detail-strength",
        str(args.detail_strength),
        "--detail-radius",
        str(args.detail_radius),
        "--detail-threshold",
        str(args.detail_threshold),
        "--max-detail-delta",
        str(args.max_finish_detail_delta),
    ]


def validate_vram_manifest(path: Path, target: tuple[int, int]) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("target_size") != list(target):
        raise RuntimeError(f"{path} target does not match {target}")
    reports = manifest.get("stage_reports") or []
    if not reports:
        raise RuntimeError(f"{path} has no completed stage reports")
    planned = sum(int(stage["tile_count"]) for stage in reports)
    processed = sum(int(stage["processed_tile_count"]) for stage in reports)
    skipped = sum(int(stage["skipped_tile_count"]) for stage in reports)
    if processed != planned or skipped != 0:
        raise RuntimeError(
            f"{path} tile gate failed: processed={processed}, planned={planned}, skipped={skipped}"
        )
    return {
        "path": str(path),
        "tile_count": planned,
        "processed_tile_count": processed,
        "skipped_tile_count": skipped,
        "estimated_spatial_activation_reduction": manifest.get(
            "estimated_spatial_activation_reduction"
        ),
        "stage_reports": reports,
    }


def validate_finish_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    detail = report.get("detail_guard") or {}
    if not detail.get("accepted"):
        raise RuntimeError(f"{path} Detail Guard candidate was rejected")
    if detail.get("applied"):
        if float(detail.get("detail_energy_ratio", 1.0)) < 1.002:
            raise RuntimeError(
                f"{path} did not produce a measurable coherent-detail gain"
            )
    elif int(detail.get("changed_pixels", 0)) != 0:
        raise RuntimeError(f"{path} reported a non-applied detail candidate with changes")
    if int(detail.get("flat_region_changed_pixels", 0)) != 0:
        raise RuntimeError(f"{path} changed flat-region pixels")
    if float(detail.get("clipped_channel_fraction", 0.0)) > 0.0005:
        raise RuntimeError(f"{path} exceeded the clipping guard")
    return report


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Krea2 native image, validate a 4K preflight, produce 8K with "
            "VRAM-Canvas, and finish with coherent source-detail protection."
        )
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--source", default=None)
    parser.add_argument("--source-stage", choices=("native", "4k"), default="native")
    parser.add_argument("--stop-after", choices=("native", "4k", "8k"), default="8k")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--inference-profile",
        choices=(*KREA2_INFERENCE_PROFILES, "custom"),
        default="turbo-fast",
        help=(
            "Fixed Krea2 sampling profile. turbo-official maps official CFG 0 to "
            "Forge CFG 1 (the Forge no-CFG value); custom requires every setting."
        ),
    )
    parser.add_argument(
        "--model-profile",
        choices=("raw", "turbo"),
        default=None,
        help="Required only by --inference-profile custom.",
    )
    parser.add_argument(
        "--native-prompt-profile",
        choices=tuple(KREA2_NATIVE_PROMPT_PROFILES),
        default="generic",
    )
    parser.add_argument("--native-width", type=int, default=None)
    parser.add_argument("--native-height", type=int, default=None)
    parser.add_argument("--native-steps", type=int, default=None)
    parser.add_argument("--sampler", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument(
        "--distilled-cfg",
        type=float,
        default=None,
        help="Krea2 flow timestep shift (sent through Forge's API field of this name).",
    )
    parser.add_argument("--vram-budget-gib", type=float, default=0.0)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--phase-count", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--color-strength",
        type=float,
        default=0.0,
        help="Adaptive chroma correction strength. Dense-detail workflow defaults to 0 to preserve intentional colors.",
    )
    parser.add_argument("--analysis-long-edge", type=int, default=1536)
    parser.add_argument("--detail-strength", type=float, default=0.75)
    parser.add_argument("--detail-radius", type=float, default=1.0)
    parser.add_argument("--detail-threshold", type=float, default=0.6)
    parser.add_argument("--max-finish-detail-delta", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--progress-interval", type=float, default=20.0)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.seed = resolve_seed(args.seed)
    inference = resolve_inference_profile(
        profile=args.inference_profile,
        model_profile=args.model_profile,
        native_width=args.native_width,
        native_height=args.native_height,
        native_steps=args.native_steps,
        sampler=args.sampler,
        scheduler=args.scheduler,
        cfg=args.cfg,
        distilled_cfg=args.distilled_cfg,
    )
    args.model_profile = inference["model_profile"]
    args.native_width, args.native_height = inference["native_size"]
    args.native_steps = inference["steps"]
    args.sampler = inference["sampler"]
    args.scheduler = inference["scheduler"]
    args.cfg = inference["cfg"]
    args.distilled_cfg = inference["shift"]
    args.inference_provenance = inference["provenance"]
    if args.vram_budget_gib < 0:
        raise ValueError("--vram-budget-gib must be 0 or greater")
    if args.tile_size < 0:
        raise ValueError("--tile-size must be 0 or greater")
    if args.source_stage == "4k" and not args.source:
        raise ValueError("--source-stage 4k requires --source")
    if args.source_stage == "4k" and args.stop_after in {"native", "4k"}:
        raise ValueError("a 4K source can only continue to --stop-after 8k")
    for value, name in (
        (args.cfg, "--cfg"),
        (args.distilled_cfg, "--distilled-cfg"),
        (args.color_strength, "--color-strength"),
        (args.detail_strength, "--detail-strength"),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if args.cfg < 0 or args.distilled_cfg < 0:
        raise ValueError("CFG values must be 0 or greater")
    if not 0 <= args.color_strength <= 1 or not 0 <= args.detail_strength <= 1:
        raise ValueError("finish strengths must be between 0 and 1")
    for value, name in (
        (args.detail_radius, "--detail-radius"),
        (args.detail_threshold, "--detail-threshold"),
        (args.max_finish_detail_delta, "--max-finish-detail-delta"),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than 0")
    if args.analysis_long_edge <= 0:
        raise ValueError("--analysis-long-edge must be greater than 0")
    if args.timeout <= 0 or args.progress_interval <= 0 or args.no_progress_timeout <= 0:
        raise ValueError("timeouts and progress interval must be greater than 0")
    return args


def main() -> int:
    args = parse_args()
    refined_prompt = detail_prompt(args.prompt)
    native_prompt = native_detail_prompt(args.prompt, args.native_prompt_profile)
    native_size = (args.native_width, args.native_height)
    source_path: Path | None = None
    if args.source:
        source_path = Path(args.source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as opened:
            source_size = opened.size
    else:
        source_size = native_size
    resolution_plan = highres_resolution_plan(source_size, args.source_stage)
    preflight_size = resolution_plan["preflight_4k"]
    final_size = resolution_plan["final_8k"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"smart8k_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "smart8k_manifest.json"
    manifest: dict = {
        "format_version": 2,
        "algorithm": "Krea2 Smart 8K",
        "created_at_utc": utc_now(),
        "status": "planned" if args.dry_run else "running",
        "base_prompt": args.prompt,
        "native_prompt_profile": args.native_prompt_profile,
        "native_prompt": native_prompt,
        "refinement_prompt": refined_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "resolution_plan": {
            "source_stage": args.source_stage,
            "source": list(resolution_plan["source"]),
            "native_generation": list(native_size) if source_path is None else None,
            "preflight_4k": list(preflight_size),
            "final_8k": list(final_size),
        },
        "settings": {
            "phase_count": args.phase_count,
            "vram_budget_gib": args.vram_budget_gib,
            "tile_size": args.tile_size,
            "inference_profile": args.inference_profile,
            "inference_provenance": args.inference_provenance,
            "model_profile": args.model_profile,
            "native_steps": args.native_steps,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "cfg": args.cfg,
            "distilled_cfg": args.distilled_cfg,
            "refinement_profiles": STAGE_PROFILES,
            "detail_guard": {
                "strength": args.detail_strength,
                "radius": args.detail_radius,
                "threshold": args.detail_threshold,
                "max_delta": args.max_finish_detail_delta,
            },
        },
        "steps": {},
    }
    write_manifest(manifest_path, manifest)
    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"MANIFEST={manifest_path}")
    emit(
        "PLAN="
        f"{source_size[0]}x{source_size[1]} -> "
        f"{preflight_size[0]}x{preflight_size[1]} -> "
        f"{final_size[0]}x{final_size[1]}"
    )
    if args.dry_run:
        emit("DRY_RUN=1")
        return 0

    try:
        if source_path is None or args.stop_after != "native":
            options = get_json(args.api, "/sdapi/v1/options", min(args.timeout, 30))
            runtime_status = post_json(
                args.api,
                "/sdapi/v1/forge-model-status/ensure-loaded",
                {},
                args.timeout,
            )
            manifest["backend"] = validate_krea2_backend(
                options,
                runtime_status,
                args.model_profile,
                verify_checkpoint_variant=args.inference_profile != "custom",
            )
            write_manifest(manifest_path, manifest)
        if source_path is not None:
            current = source_path
            current_metrics = analysis_metrics(current)
            source_step = {
                "kind": args.source_stage,
                **image_record(current),
                "analysis_metrics": current_metrics,
            }
            if args.source_stage == "4k":
                source_step["qa_crops_100pct"] = save_qa_crops(
                    current, output_dir / "qa_source_4k_100pct"
                )
            manifest["steps"]["source"] = source_step
            write_manifest(manifest_path, manifest)
        else:
            current = output_dir / "native_seed.png"
            native = generate_native(
                api=args.api,
                output=current,
                prompt=native_prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                steps=args.native_steps,
                sampler=args.sampler,
                scheduler=args.scheduler,
                cfg=args.cfg,
                distilled_cfg=args.distilled_cfg,
                size=native_size,
                timeout=args.timeout,
            )
            native["analysis_metrics"] = analysis_metrics(current)
            current_metrics = native["analysis_metrics"]
            manifest["steps"]["native"] = native
            write_manifest(manifest_path, manifest)
            emit(f"NATIVE={current}")
        if args.stop_after == "native":
            manifest["status"] = "complete_native"
            manifest["completed_at_utc"] = utc_now()
            write_manifest(manifest_path, manifest)
            return 0

        if args.source_stage == "native":
            telemetry_4k = output_dir / "telemetry_4k.json"
            lines = run_command(
                vram_canvas_command(
                    args,
                    source=current,
                    output_root=output_dir / "vram_4k",
                    prompt=refined_prompt,
                    target=preflight_size,
                    stage="4k",
                ),
                "PREFLIGHT_4K",
                args.timeout,
                interrupt_api=args.api,
                telemetry_path=telemetry_4k,
            )
            raw_4k = emitted_path(lines, "IMAGE")
            vram_4k_manifest = emitted_path(lines, "MANIFEST")
            require_size(raw_4k, preflight_size)
            finish_4k = output_dir / "smart4k_preflight.png"
            finish_4k_report = output_dir / "smart4k_preflight.quality.json"
            finish_lines = run_command(
                finish_command(
                    args,
                    source=raw_4k,
                    output=finish_4k,
                    report=finish_4k_report,
                ),
                "FINISH_4K",
                args.timeout,
                interrupt_api=args.api,
            )
            emitted_path(finish_lines, "OUTPUT")
            require_size(finish_4k, preflight_size)
            preflight_metrics = analysis_metrics(finish_4k)
            retention = detail_retention_gate(
                current_metrics, preflight_metrics, minimum_ratio=0.70
            )
            manifest["steps"]["preflight_4k"] = {
                "raw": image_record(raw_4k),
                "final": image_record(finish_4k),
                "vram_canvas": validate_vram_manifest(
                    vram_4k_manifest, preflight_size
                ),
                "finish_report": validate_finish_report(finish_4k_report),
                "analysis_metrics": preflight_metrics,
                "detail_retention_vs_source": retention,
                "qa_crops_100pct": save_qa_crops(
                    finish_4k, output_dir / "qa_4k_100pct"
                ),
                "telemetry": json.loads(
                    telemetry_4k.read_text(encoding="utf-8")
                ),
            }
            current = finish_4k
            current_metrics = preflight_metrics
            write_manifest(manifest_path, manifest)
            emit(f"PREFLIGHT_4K_FINAL={finish_4k}")
        if args.stop_after == "4k":
            manifest["status"] = "complete_4k"
            manifest["completed_at_utc"] = utc_now()
            write_manifest(manifest_path, manifest)
            return 0

        telemetry_8k = output_dir / "telemetry_8k.json"
        lines = run_command(
            vram_canvas_command(
                args,
                source=current,
                output_root=output_dir / "vram_8k",
                prompt=refined_prompt,
                target=final_size,
                stage="8k",
            ),
            "FINAL_8K",
            args.timeout,
            interrupt_api=args.api,
            telemetry_path=telemetry_8k,
        )
        raw_8k = emitted_path(lines, "IMAGE")
        vram_8k_manifest = emitted_path(lines, "MANIFEST")
        require_size(raw_8k, final_size)
        final_8k = output_dir / "smart8k_final.png"
        final_8k_report = output_dir / "smart8k_final.quality.json"
        finish_lines = run_command(
            finish_command(
                args,
                source=raw_8k,
                output=final_8k,
                report=final_8k_report,
            ),
            "FINISH_8K",
            args.timeout,
            interrupt_api=args.api,
        )
        emitted_path(finish_lines, "OUTPUT")
        require_size(final_8k, final_size)
        final_metrics = analysis_metrics(final_8k)
        retention = detail_retention_gate(
            current_metrics, final_metrics, minimum_ratio=0.88
        )
        manifest["steps"]["final_8k"] = {
            "raw": image_record(raw_8k),
            "final": image_record(final_8k),
            "vram_canvas": validate_vram_manifest(vram_8k_manifest, final_size),
            "finish_report": validate_finish_report(final_8k_report),
            "analysis_metrics": final_metrics,
            "detail_retention_vs_source": retention,
            "qa_crops_100pct": save_qa_crops(
                final_8k, output_dir / "qa_8k_100pct"
            ),
            "telemetry": json.loads(
                telemetry_8k.read_text(encoding="utf-8")
            ),
        }
        manifest["status"] = "complete_8k"
        manifest["completed_at_utc"] = utc_now()
        write_manifest(manifest_path, manifest)
        emit(f"FINAL_8K_IMAGE={final_8k}")
        return 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
