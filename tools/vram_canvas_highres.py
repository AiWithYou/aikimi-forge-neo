"""VRAM-bounded progressive 4K/8K refinement for Forge's img2img API.

VRAM-Canvas never sends the delivery canvas to the diffusion model.  It progressively
upsamples a globally coherent base, refines haloed native-size crops, and merges only
structure-consistent high-frequency differences that agree across overlapping tiles.
GPU spatial memory is therefore bounded by the selected tile payload, while large CPU
moment accumulators are backed by disk memmaps.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime
import io
import json
import math
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from modules.krea2_quality import adaptive_detail_guard
from modules_forge.krea2_highres import (
    EXACT_IMG2IMG_STEPS,
    EXACT_IMG2IMG_STEPS_SCOPE,
    KREA2_VRAM_CANVAS_PROFILES,
    KREA2_PHASEWEAVE_PRODUCT_NAME,
    KREA2_PHASEWEAVE_PROFILE_KEY,
    krea2_detail_prompt,
    krea2_vram_canvas_profile,
)
from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_CFG,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_DEFAULT_SHIFT,
    replace_infotext_size,
    target_size,
)
from modules_forge.vram_canvas import (
    CONSENSUS_MERGE_MODE,
    DEFAULT_ACTIVATION_BYTES_PER_PIXEL,
    DEFAULT_BASE_DETAIL_SIGMA,
    DEFAULT_MAX_TILE_SIZE,
    DEFAULT_MIN_TILE_SIZE,
    DEFAULT_MODEL_RESERVE_GIB,
    DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA,
    DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH,
    DEFAULT_NOVEL_DETAIL_INNER_RADIUS,
    DEFAULT_NOVEL_DETAIL_OUTER_RADIUS,
    DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA,
    DEFAULT_VRAM_USE_FRACTION,
    PHASE_WEAVE_DETAIL_FLOOR,
    PHASE_WEAVE_CONTEXT_RADIUS,
    PHASE_WEAVE_FEATHER_RADIUS,
    PHASE_WEAVE_MERGE_MODE,
    PHASE_WEAVE_QUALITY_RADIUS,
    PHASE_WEAVE_SELECTION_MARGIN,
    PHASE_WEAVE_SUPPORT_MIX,
    adaptive_step_count,
    balanced_virtual_axis_origin,
    consensus_gated_residual,
    coordinate_seed,
    detail_score,
    extract_tile_context,
    frequency_detail_delta,
    novel_detail_delta,
    phase_normalized_tile_weight,
    phase_weave_configuration,
    phase_weave_residual,
    phase_weight_normalizers,
    plan_tiles,
    progressive_stage_sizes,
    resolve_core_overlap,
    resolve_halo,
    resolve_tile_size,
    replace_infotext_seed,
    spatial_activation_reduction,
    vram_canvas_work_bytes_per_pixel,
)

DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output/vram_canvas"
DEFAULT_FALLBACK_VRAM_GIB = 8.0
DEFAULT_MAX_OUTPUT_PIXELS = 70_000_000
DEFAULT_LOW_PASS_RADIUS = 12
DEFAULT_MAX_DETAIL_DELTA = 32.0
DEFAULT_STRUCTURE_SIGMA = 18.0
DEFAULT_CONSENSUS_SIGMA = 8.0
DEFAULT_DIFFUSION_ALIGNMENT = 16

KREA2_PROFILE_CLI_DESTINATIONS = {
    "merge_mode": "merge_mode",
    "phase_count": "phase_count",
    "minimum_steps": "minimum_steps",
    "maximum_steps": "steps",
    "detail_knee": "detail_knee",
    "coarse_denoise": "coarse_denoise",
    "denoise": "denoise",
    "low_pass_radius": "low_pass_radius",
    "detail_gain": "detail_gain",
    "max_detail_delta": "max_detail_delta",
    "structure_sigma": "structure_sigma",
    "base_detail_sigma": "base_detail_sigma",
    "consensus_sigma": "consensus_sigma",
    "novel_detail_gain": "novel_detail_gain",
    "novel_detail_max_delta": "novel_detail_max_delta",
    "novel_detail_inner_radius": "novel_detail_inner_radius",
    "novel_detail_outer_radius": "novel_detail_outer_radius",
    "novel_detail_structure_sigma": "novel_detail_structure_sigma",
    "novel_detail_consensus_sigma": "novel_detail_consensus_sigma",
    "novel_detail_consensus_strength": "novel_detail_consensus_strength",
    "finish_detail_strength": "finish_detail_strength",
    "finish_detail_radius": "finish_detail_radius",
    "finish_detail_threshold": "finish_detail_threshold",
    "finish_max_detail_delta": "finish_max_detail_delta",
}


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def krea2_profile_cli_defaults(profile_name: str) -> dict[str, float | int]:
    """Translate a shared Krea2 profile into argparse destination defaults."""

    profile = krea2_vram_canvas_profile(profile_name)
    return {
        destination: profile[key]
        for key, destination in KREA2_PROFILE_CLI_DESTINATIONS.items()
    }


def resolve_seed(seed: int) -> int:
    if seed < -1:
        raise ValueError("--seed must be -1 or a non-negative integer.")
    return secrets.randbelow(2**32) if seed == -1 else seed


def flatten_source_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and flatten transparency onto a white matte."""

    oriented = ImageOps.exif_transpose(image)
    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    if not has_alpha:
        return oriented.convert("RGB")
    rgba = oriented.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def image_to_b64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_b64_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        return image.convert("RGB")


def prompt_from_png(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        parameters = str(image.info.get("parameters", ""))
    if not parameters:
        return "", ""
    negative = ""
    prompt_block = parameters
    if "\nNegative prompt:" in parameters:
        prompt_block, rest = parameters.split("\nNegative prompt:", 1)
        negative = rest.split("\nSteps:", 1)[0].strip()
    elif "\nSteps:" in parameters:
        prompt_block = parameters.split("\nSteps:", 1)[0]
    return prompt_block.strip(), negative


def replace_infotext_prompts(
    infotext: str,
    prompt: str,
    negative_prompt: str,
) -> str:
    """Write the effective CLI prompts while preserving the source settings line."""

    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")

    settings = ""
    lines = infotext.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].lstrip().startswith("Steps:"):
            settings = "\n".join(lines[index:]).strip()
            break

    blocks = [prompt]
    if negative_prompt:
        blocks.append(f"Negative prompt: {negative_prompt}")
    if settings:
        blocks.append(settings)
    return "\n".join(blocks)


def pad_tile_for_diffusion(tile: Image.Image, alignment: int = DEFAULT_DIFFUSION_ALIGNMENT) -> Image.Image:
    if alignment <= 0:
        raise ValueError("diffusion alignment must be > 0.")
    padded_width = max(alignment, ((tile.width + alignment - 1) // alignment) * alignment)
    padded_height = max(alignment, ((tile.height + alignment - 1) // alignment) * alignment)
    if (padded_width, padded_height) == tile.size:
        return tile.copy()
    values = np.asarray(tile.convert("RGB"), dtype=np.uint8)
    padded = np.pad(
        values,
        (
            (0, padded_height - tile.height),
            (0, padded_width - tile.width),
            (0, 0),
        ),
        mode="edge",
    )
    return Image.fromarray(padded, mode="RGB")


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urlrequest.Request(url, data=body, headers=headers, method=method)
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urlerror.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body[:4000]}") from exc
    parsed = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(parsed, dict):
        raise RuntimeError("Forge returned a non-object JSON response.")
    return status, parsed


def poll_progress(args, stage_name: str, stop_event: threading.Event):
    last_error = ""
    last_progress_key = None
    last_progress_at = time.monotonic()
    while not stop_event.wait(args.progress_interval):
        try:
            query = urlparse.urlencode({"skip_current_image": "true"})
            _, data = _http_json(f"{args.api.rstrip('/')}/sdapi/v1/progress?{query}", timeout=5)
        except (RuntimeError, urlerror.URLError, ValueError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if error != last_error:
                emit(f"{stage_name}_PROGRESS_ERROR={error}")
                last_error = error
            continue

        state = data.get("state") or {}
        progress = data.get("progress")
        progress_key = (
            progress,
            state.get("sampling_step"),
            state.get("sampling_steps"),
            state.get("job_timestamp"),
        )
        if progress_key != last_progress_key:
            last_progress_key = progress_key
            last_progress_at = time.monotonic()
        elif args.no_progress_timeout > 0 and time.monotonic() - last_progress_at >= args.no_progress_timeout:
            emit(f"{stage_name}_NO_PROGRESS_TIMEOUT={args.no_progress_timeout}")
            try:
                _http_json(
                    f"{args.api.rstrip('/')}/sdapi/v1/interrupt",
                    method="POST",
                    payload={},
                    timeout=5,
                )
            except (RuntimeError, urlerror.URLError, ValueError, OSError) as exc:
                emit(f"{stage_name}_INTERRUPT_ERROR={type(exc).__name__}: {exc}")
            stop_event.set()
            return

        progress_text = f"{progress:.4f}" if isinstance(progress, (int, float)) else str(progress)
        emit(f"{stage_name}_PROGRESS={progress_text} " f"STEP={state.get('sampling_step')}/{state.get('sampling_steps')} " f"JOB={state.get('job')} ETA={data.get('eta_relative')}")


def post_img2img(args, payload: dict, stage_name: str) -> dict:
    stop_event = None
    progress_thread = None
    if args.progress_interval > 0:
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=poll_progress, args=(args, stage_name, stop_event), daemon=True)
        progress_thread.start()
    try:
        status, data = _http_json(
            f"{args.api.rstrip('/')}/sdapi/v1/img2img",
            method="POST",
            payload=payload,
            timeout=args.timeout,
        )
    finally:
        if stop_event is not None:
            stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1)
    emit(f"{stage_name}_HTTP={status}")
    return data


def _finite_nonnegative(value: float, name: str):
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and >= 0.")


def validate_args(args):
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be passed together.")
    if args.width is None and args.long_edge <= 0:
        raise ValueError("--long-edge must be > 0 when width/height are omitted.")
    for value, name in (
        (args.width, "--width"),
        (args.height, "--height"),
        (args.minimum_steps, "--minimum-steps"),
        (args.steps, "--steps"),
        (args.timeout, "--timeout"),
        (args.max_output_pixels, "--max-output-pixels"),
        (args.activation_bytes_per_pixel, "--activation-bytes-per-pixel"),
        (args.minimum_tile_size, "--minimum-tile-size"),
        (args.maximum_tile_size, "--maximum-tile-size"),
        (args.phase_count, "--phase-count"),
        (args.low_pass_radius, "--low-pass-radius"),
        (args.finalize_stripe_height, "--finalize-stripe-height"),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be > 0.")
    if args.minimum_steps > args.steps:
        raise ValueError("--minimum-steps must be <= --steps.")
    if args.phase_count > 2:
        raise ValueError("--phase-count must be 1 or 2.")
    if args.merge_mode not in (CONSENSUS_MERGE_MODE, PHASE_WEAVE_MERGE_MODE):
        raise ValueError(f"unknown --merge-mode: {args.merge_mode}")
    if args.merge_mode == PHASE_WEAVE_MERGE_MODE and args.phase_count != 2:
        raise ValueError("--merge-mode phase_weave requires --phase-count 2.")
    if args.save_phase_candidates and args.merge_mode != PHASE_WEAVE_MERGE_MODE:
        raise ValueError(
            "--save-phase-candidates requires --merge-mode phase_weave."
        )
    if args.tile_size < 0 or args.halo < 0 or args.core_overlap < 0:
        raise ValueError("tile size, halo, and core overlap must be >= 0.")
    if args.minimum_tile_size > args.maximum_tile_size:
        raise ValueError("minimum tile size must be <= maximum tile size.")
    if args.vram_budget_gib < 0:
        raise ValueError("--vram-budget-gib must be >= 0.")
    if args.fallback_vram_gib <= 0:
        raise ValueError("--fallback-vram-gib must be > 0.")
    if not 0 < args.vram_use_fraction <= 1:
        raise ValueError("--vram-use-fraction must satisfy 0 < value <= 1.")
    _finite_nonnegative(args.model_reserve_gib, "--model-reserve-gib")
    if args.max_stage_scale <= 1:
        raise ValueError("--max-stage-scale must be > 1.")
    for value, name in (
        (args.coarse_denoise, "--coarse-denoise"),
        (args.denoise, "--denoise"),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if args.cfg < 0 or args.distilled_cfg < 0:
        raise ValueError("CFG values must be >= 0.")
    for value, name in (
        (args.detail_knee, "--detail-knee"),
        (args.detail_gain, "--detail-gain"),
        (args.max_detail_delta, "--max-detail-delta"),
        (args.structure_sigma, "--structure-sigma"),
    ):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0.")
    _finite_nonnegative(args.base_detail_sigma, "--base-detail-sigma")
    _finite_nonnegative(args.consensus_sigma, "--consensus-sigma")
    _finite_nonnegative(args.novel_detail_gain, "--novel-detail-gain")
    for value, name in (
        (args.novel_detail_max_delta, "--novel-detail-max-delta"),
        (args.novel_detail_structure_sigma, "--novel-detail-structure-sigma"),
        (args.novel_detail_consensus_sigma, "--novel-detail-consensus-sigma"),
        (args.novel_detail_consensus_strength, "--novel-detail-consensus-strength"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and > 0.")
    if args.novel_detail_inner_radius <= 0 or args.novel_detail_outer_radius <= 0:
        raise ValueError("novel detail radii must be > 0.")
    if args.novel_detail_inner_radius >= args.novel_detail_outer_radius:
        raise ValueError("novel detail inner radius must be smaller than outer radius.")
    if args.novel_detail_gain > 0 and args.phase_count < 2:
        raise ValueError("novel detail requires --phase-count 2 for independent evidence.")
    if args.novel_detail_gain > 0 and args.base_detail_sigma <= 0:
        raise ValueError("novel detail requires --base-detail-sigma > 0.")
    if not math.isfinite(float(args.finish_detail_strength)) or not 0 <= float(
        args.finish_detail_strength
    ) <= 1:
        raise ValueError("--finish-detail-strength must be finite and between 0 and 1.")
    for value, name in (
        (args.finish_detail_radius, "--finish-detail-radius"),
        (args.finish_detail_threshold, "--finish-detail-threshold"),
        (args.finish_max_detail_delta, "--finish-max-detail-delta"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and > 0.")
    _finite_nonnegative(args.skip_flat_below, "--skip-flat-below")
    _finite_nonnegative(args.progress_interval, "--progress-interval")
    _finite_nonnegative(args.no_progress_timeout, "--no-progress-timeout")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")


def _find_numeric_total_bytes(value) -> int | None:
    """Find the CUDA system total field used by A1111/Forge memory endpoints."""

    if not isinstance(value, dict):
        return None
    cuda = value.get("cuda")
    if isinstance(cuda, dict):
        system = cuda.get("system")
        if isinstance(system, dict):
            total = system.get("total")
            if isinstance(total, (int, float)) and total > 0:
                return int(total)
    return None


def query_total_vram_gib(api: str, timeout: float = 5.0) -> float | None:
    try:
        _, data = _http_json(f"{api.rstrip('/')}/sdapi/v1/memory", timeout=timeout)
        total = _find_numeric_total_bytes(data)
    except (RuntimeError, urlerror.URLError, ValueError, TypeError, OSError):
        return None
    return None if total is None else total / float(1024**3)


def stage_denoise(stage_index: int, stage_count: int, coarse_denoise: float, final_denoise: float) -> float:
    if stage_count <= 1:
        return float(final_denoise)
    fraction = stage_index / (stage_count - 1)
    return float(coarse_denoise * (1.0 - fraction) + final_denoise * fraction)


def build_tile_payload(
    args,
    tile: Image.Image,
    prompt: str,
    negative_prompt: str,
    *,
    seed: int,
    steps: int,
    denoise: float,
) -> tuple[dict, tuple[int, int]]:
    diffusion_tile = pad_tile_for_diffusion(tile)
    payload = {
        "init_images": [image_to_b64_png(diffusion_tile)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": diffusion_tile.width,
        "height": diffusion_tile.height,
        "resize_mode": 0,
        "denoising_strength": denoise,
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": True,
        "save_images": False,
        "include_init_images": False,
        "override_settings": {"img2img_fix_steps": True},
        "override_settings_restore_afterwards": True,
    }
    return payload, diffusion_tile.size


def _manifest_tile(tile, *, seed: int, steps: int, score: float, skipped: bool):
    record = asdict(tile)
    record.update(
        {
            "seed": seed,
            "steps": steps,
            "detail_score": round(float(score), 8),
            "skipped": bool(skipped),
        }
    )
    return record


def _close_memmap(value: np.memmap) -> None:
    value.flush()
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _save_stage_result(
    base: Image.Image,
    accumulators: list[np.memmap],
    weight_sums: list[np.memmap],
    energy_sums: list[np.memmap],
    novel_accumulators: list[np.memmap],
    novel_energy_sums: list[np.memmap],
    path: Path,
    result_work_path: Path,
    *,
    stripe_height: int,
    consensus_sigma: float,
    novel_consensus_sigma: float,
    novel_consensus_strength: float,
    merge_mode: str,
    phase_candidate_paths: tuple[Path, Path] | None = None,
    phase_selection_path: Path | None = None,
) -> tuple[Image.Image, dict[str, float | str | bool]]:
    width, height = base.size
    expected_sets = 2 if merge_mode == PHASE_WEAVE_MERGE_MODE else 1
    if not (
        len(accumulators)
        == len(weight_sums)
        == len(energy_sums)
        == expected_sets
    ):
        raise ValueError("VRAM-Canvas stage moment set count does not match merge mode.")
    if novel_accumulators and not (
        len(novel_accumulators)
        == len(novel_energy_sums)
        == expected_sets
    ):
        raise ValueError("VRAM-Canvas novel moment set count does not match merge mode.")
    if phase_candidate_paths is not None and merge_mode != PHASE_WEAVE_MERGE_MODE:
        raise ValueError("phase candidate export requires phase_weave merge mode.")
    if phase_selection_path is not None and merge_mode != PHASE_WEAVE_MERGE_MODE:
        raise ValueError("phase selection export requires phase_weave merge mode.")
    result = np.memmap(result_work_path, dtype=np.uint8, mode="w+", shape=(height, width, 3))
    phase_candidate_work_paths: list[Path] = []
    phase_candidate_results: list[np.memmap] = []
    if phase_candidate_paths is not None:
        for phase_index in range(2):
            candidate_work_path = result_work_path.with_name(
                f"{result_work_path.stem}_candidate_{phase_index}.uint8"
            )
            phase_candidate_work_paths.append(candidate_work_path)
            phase_candidate_results.append(
                np.memmap(
                    candidate_work_path,
                    dtype=np.uint8,
                    mode="w+",
                    shape=(height, width, 3),
                )
            )
    phase_selection_work_path: Path | None = None
    phase_selection_result: np.memmap | None = None
    if phase_selection_path is not None:
        phase_selection_work_path = result_work_path.with_name(
            f"{result_work_path.stem}_selection.uint8"
        )
        phase_selection_result = np.memmap(
            phase_selection_work_path,
            dtype=np.uint8,
            mode="w+",
            shape=(height, width, 3),
        )
    covered_pixels = 0
    confidence_total = 0.0
    disagreement_total = 0.0
    phase0_pixels = 0
    phase1_pixels = 0
    input_rejected_pixels = 0
    both_unfaithful_pixels = 0
    uncertain_fused_pixels = 0
    boundary_pixels = 0
    confidence_gain_total = 0.0
    selected_fidelity_total = 0.0
    input_mix_total = 0.0
    support_weight_total = 0.0
    low_frequency_luma_gain_total = 0.0
    novel_evidence_pixels = 0
    novel_confidence_total = 0.0
    novel_disagreement_total = 0.0
    novel_abs_total = 0.0
    for y0 in range(0, height, stripe_height):
        y1 = min(height, y0 + stripe_height)
        base_stripe = np.asarray(base.crop((0, y0, width, y1)), dtype=np.float32)
        if merge_mode == PHASE_WEAVE_MERGE_MODE:
            padding = PHASE_WEAVE_CONTEXT_RADIUS
            read_y0 = max(0, y0 - padding)
            read_y1 = min(height, y1 + padding)
            base_read = np.asarray(
                base.crop((0, read_y0, width, read_y1)),
                dtype=np.float32,
            )
            normalized_read, diagnostics_read = phase_weave_residual(
                np.asarray(accumulators[0][read_y0:read_y1], dtype=np.float32),
                np.asarray(weight_sums[0][read_y0:read_y1], dtype=np.float32),
                np.asarray(energy_sums[0][read_y0:read_y1], dtype=np.float32),
                np.asarray(accumulators[1][read_y0:read_y1], dtype=np.float32),
                np.asarray(weight_sums[1][read_y0:read_y1], dtype=np.float32),
                np.asarray(energy_sums[1][read_y0:read_y1], dtype=np.float32),
                base_rgb=base_read,
                sigma=consensus_sigma,
            )
            local_y0 = y0 - read_y0
            local_y1 = local_y0 + (y1 - y0)
            normalized = normalized_read[local_y0:local_y1]
            diagnostics = {
                key: value[local_y0:local_y1]
                for key, value in diagnostics_read.items()
            }
            phase0_candidate_residual = diagnostics["phase0_residual"].copy()
            phase1_candidate_residual = diagnostics["phase1_residual"].copy()
            covered = diagnostics["covered"]
            confidence = diagnostics["support_confidence"]
            disagreement = diagnostics["cross_disagreement"]
            phase0_pixels += int(
                np.count_nonzero(diagnostics["selected_phase"] == 0)
            )
            phase1_pixels += int(
                np.count_nonzero(diagnostics["selected_phase"] == 1)
            )
            input_rejected_pixels += int(
                np.count_nonzero(diagnostics["input_rejected"])
            )
            both_unfaithful_pixels += int(
                np.count_nonzero(diagnostics["both_unfaithful"])
            )
            uncertain_fused_pixels += int(
                np.count_nonzero(diagnostics["uncertain_fused"])
            )
            boundary_pixels += int(np.count_nonzero(diagnostics["boundary"]))
            confidence_gain_total += float(
                np.sum(diagnostics["confidence_gain"][covered], dtype=np.float64)
            )
            selected_fidelity_total += float(
                np.sum(
                    diagnostics["selected_fidelity"][covered],
                    dtype=np.float64,
                )
            )
            input_mix_total += float(
                np.sum(diagnostics["input_mix"][covered], dtype=np.float64)
            )
            support_weight_total += float(
                np.sum(diagnostics["support_weight"][covered], dtype=np.float64)
            )
            low_frequency_luma_gain_total += float(
                np.sum(
                    diagnostics["low_frequency_luma_gain"][covered],
                    dtype=np.float64,
                )
            )
            if phase_selection_result is not None:
                labels = diagnostics["selected_phase"]
                selection_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
                selection_rgb[labels == 0] = (194, 104, 54)
                selection_rgb[labels == 1] = (49, 123, 157)
                selection_rgb[labels == 2] = (96, 99, 105)
                selection_rgb[labels == 3] = (126, 72, 145)
                phase_selection_result[y0:y1] = selection_rgb
        else:
            weights = np.asarray(weight_sums[0][y0:y1], dtype=np.float32)
            normalized, confidence, disagreement = consensus_gated_residual(
                np.asarray(accumulators[0][y0:y1], dtype=np.float32),
                weights,
                np.asarray(energy_sums[0][y0:y1], dtype=np.float32),
                sigma=consensus_sigma,
            )
            covered = weights > 1e-8
            phase0_candidate_residual = normalized
            phase1_candidate_residual = normalized
        novel_normalized = np.zeros_like(normalized)
        phase0_novel_candidate = np.zeros_like(normalized)
        phase1_novel_candidate = np.zeros_like(normalized)
        if novel_accumulators:
            if merge_mode == PHASE_WEAVE_MERGE_MODE:
                novel_read, novel_diagnostics_read = phase_weave_residual(
                    np.asarray(
                        novel_accumulators[0][read_y0:read_y1], dtype=np.float32
                    ),
                    np.asarray(weight_sums[0][read_y0:read_y1], dtype=np.float32),
                    np.asarray(
                        novel_energy_sums[0][read_y0:read_y1], dtype=np.float32
                    ),
                    np.asarray(
                        novel_accumulators[1][read_y0:read_y1], dtype=np.float32
                    ),
                    np.asarray(weight_sums[1][read_y0:read_y1], dtype=np.float32),
                    np.asarray(
                        novel_energy_sums[1][read_y0:read_y1], dtype=np.float32
                    ),
                    base_rgb=base_read,
                    sigma=novel_consensus_sigma,
                    strength=novel_consensus_strength,
                )
                novel_normalized = novel_read[local_y0:local_y1]
                novel_diagnostics = {
                    key: value[local_y0:local_y1]
                    for key, value in novel_diagnostics_read.items()
                }
                phase0_novel_candidate = novel_diagnostics[
                    "phase0_residual"
                ].copy()
                phase1_novel_candidate = novel_diagnostics[
                    "phase1_residual"
                ].copy()
                independent_evidence = novel_diagnostics["both_covered"]
                novel_confidence = novel_diagnostics["support_confidence"]
                novel_disagreement = novel_diagnostics["cross_disagreement"]
            else:
                novel_normalized, novel_confidence, novel_disagreement = (
                    consensus_gated_residual(
                        np.asarray(
                            novel_accumulators[0][y0:y1], dtype=np.float32
                        ),
                        weights,
                        np.asarray(
                            novel_energy_sums[0][y0:y1], dtype=np.float32
                        ),
                        sigma=novel_consensus_sigma,
                        strength=novel_consensus_strength,
                    )
                )
                independent_evidence = weights >= 1.5
            novel_normalized[~independent_evidence] = 0.0
            novel_evidence_pixels += int(np.count_nonzero(independent_evidence))
            novel_confidence_total += float(
                np.sum(
                    novel_confidence[independent_evidence], dtype=np.float64
                )
            )
            novel_disagreement_total += float(
                np.sum(
                    novel_disagreement[independent_evidence], dtype=np.float64
                )
            )
            novel_abs_total += float(
                np.sum(np.abs(novel_normalized), dtype=np.float64) / 3.0
            )
        result[y0:y1] = np.clip(
            np.rint(base_stripe + normalized + novel_normalized), 0, 255
        ).astype(np.uint8)
        if phase_candidate_results:
            phase_candidate_results[0][y0:y1] = np.clip(
                np.rint(
                    base_stripe
                    + phase0_candidate_residual
                    + phase0_novel_candidate
                ),
                0,
                255,
            ).astype(np.uint8)
            phase_candidate_results[1][y0:y1] = np.clip(
                np.rint(
                    base_stripe
                    + phase1_candidate_residual
                    + phase1_novel_candidate
                ),
                0,
                255,
            ).astype(np.uint8)
        covered_pixels += int(np.count_nonzero(covered))
        confidence_total += float(np.sum(confidence[covered], dtype=np.float64))
        disagreement_total += float(np.sum(disagreement[covered], dtype=np.float64))
    result.flush()
    image = Image.fromarray(np.asarray(result), mode="RGB")
    image.save(path, format="PNG")
    image.close()
    del result
    with Image.open(path) as saved:
        image = saved.copy()
    if phase_candidate_paths is not None:
        for candidate_result, candidate_path in zip(
            phase_candidate_results,
            phase_candidate_paths,
        ):
            candidate_result.flush()
            candidate_image = Image.fromarray(
                np.asarray(candidate_result),
                mode="RGB",
            )
            candidate_image.save(candidate_path, format="PNG")
            candidate_image.close()
            _close_memmap(candidate_result)
        for candidate_work_path in phase_candidate_work_paths:
            candidate_work_path.unlink(missing_ok=True)
    if phase_selection_result is not None and phase_selection_path is not None:
        phase_selection_result.flush()
        selection_image = Image.fromarray(
            np.asarray(phase_selection_result),
            mode="RGB",
        )
        selection_image.save(phase_selection_path, format="PNG")
        selection_image.close()
        _close_memmap(phase_selection_result)
        if phase_selection_work_path is not None:
            phase_selection_work_path.unlink(missing_ok=True)
    divisor = covered_pixels or 1
    novel_divisor = novel_evidence_pixels or 1
    stats = {
        "merge_mode": merge_mode,
        "mean_consensus_gate": confidence_total / divisor,
        "mean_consensus_disagreement": disagreement_total / divisor,
        "phaseweave_enabled": merge_mode == PHASE_WEAVE_MERGE_MODE,
        "phaseweave_phase0_selected_percent": phase0_pixels * 100.0 / divisor,
        "phaseweave_phase1_selected_percent": phase1_pixels * 100.0 / divisor,
        "phaseweave_input_rejected_percent": (
            input_rejected_pixels * 100.0 / divisor
        ),
        "phaseweave_both_unfaithful_percent": (
            both_unfaithful_pixels * 100.0 / divisor
        ),
        "phaseweave_uncertain_fused_percent": (
            uncertain_fused_pixels * 100.0 / divisor
        ),
        "phaseweave_boundary_percent": boundary_pixels * 100.0 / divisor,
        "phaseweave_mean_detail_gain": (
            confidence_gain_total / divisor
            if merge_mode == PHASE_WEAVE_MERGE_MODE
            else 0.0
        ),
        "phaseweave_mean_selected_fidelity": (
            selected_fidelity_total / divisor
            if merge_mode == PHASE_WEAVE_MERGE_MODE
            else 0.0
        ),
        "phaseweave_mean_input_mix": (
            input_mix_total / divisor
            if merge_mode == PHASE_WEAVE_MERGE_MODE
            else 0.0
        ),
        "phaseweave_mean_support_weight": (
            support_weight_total / divisor
            if merge_mode == PHASE_WEAVE_MERGE_MODE
            else 0.0
        ),
        "phaseweave_mean_low_frequency_luma_gain": (
            low_frequency_luma_gain_total / divisor
            if merge_mode == PHASE_WEAVE_MERGE_MODE
            else 0.0
        ),
        "novel_detail_enabled": bool(novel_accumulators),
        "novel_evidence_percent": novel_evidence_pixels * 100.0 / (width * height),
        "mean_novel_consensus_gate": novel_confidence_total / novel_divisor,
        "mean_novel_consensus_disagreement": novel_disagreement_total
        / novel_divisor,
        "mean_abs_novel_residual": novel_abs_total / novel_divisor,
    }
    if phase_candidate_paths is not None:
        stats["phaseweave_phase0_candidate"] = str(phase_candidate_paths[0])
        stats["phaseweave_phase1_candidate"] = str(phase_candidate_paths[1])
    if phase_selection_path is not None:
        stats["phaseweave_selection_map"] = str(phase_selection_path)
    return image, stats


def refine_stage(
    base: Image.Image,
    args,
    prompt: str,
    negative_prompt: str,
    output_dir: Path,
    *,
    stage_index: int,
    stage_count: int,
    tile_size: int,
    halo: int,
    core_overlap: int,
) -> tuple[Image.Image, dict]:
    width, height = base.size
    plans = plan_tiles(
        width,
        height,
        tile_size=tile_size,
        halo=halo,
        core_overlap=core_overlap,
        phase_count=args.phase_count,
        virtual_padding=args.merge_mode == PHASE_WEAVE_MERGE_MODE,
    )
    grid_origin = (
        [
            balanced_virtual_axis_origin(
                width,
                tile_size - 2 * halo,
                core_overlap,
                phase_count=args.phase_count,
            ),
            balanced_virtual_axis_origin(
                height,
                tile_size - 2 * halo,
                core_overlap,
                phase_count=args.phase_count,
            ),
        ]
        if args.merge_mode == PHASE_WEAVE_MERGE_MODE
        else [0, 0]
    )
    phase_normalizers = phase_weight_normalizers(plans, width, height)
    denoise = stage_denoise(stage_index, stage_count, args.coarse_denoise, args.denoise)
    stage_number = stage_index + 1
    stage_prefix = f"stage_{stage_number:02d}"
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    result_work_path = work_dir / f"{stage_prefix}_result.uint8"
    moment_set_count = 2 if args.merge_mode == PHASE_WEAVE_MERGE_MODE else 1
    work_paths: list[Path] = []
    accumulators: list[np.memmap] = []
    weight_sums: list[np.memmap] = []
    energy_sums: list[np.memmap] = []
    novel_accumulators: list[np.memmap] = []
    novel_energy_sums: list[np.memmap] = []
    for phase_slot in range(moment_set_count):
        suffix = f"_phase{phase_slot}" if moment_set_count > 1 else ""
        accumulator_path = work_dir / f"{stage_prefix}{suffix}_delta.float32"
        weight_path = work_dir / f"{stage_prefix}{suffix}_weight.float32"
        energy_path = work_dir / f"{stage_prefix}{suffix}_energy.float32"
        work_paths.extend((accumulator_path, weight_path, energy_path))
        accumulators.append(
            np.memmap(
                accumulator_path,
                dtype=np.float32,
                mode="w+",
                shape=(height, width, 3),
            )
        )
        weight_sums.append(
            np.memmap(
                weight_path,
                dtype=np.float32,
                mode="w+",
                shape=(height, width),
            )
        )
        energy_sums.append(
            np.memmap(
                energy_path,
                dtype=np.float32,
                mode="w+",
                shape=(height, width),
            )
        )
        if args.novel_detail_gain > 0:
            novel_accumulator_path = (
                work_dir / f"{stage_prefix}{suffix}_novel_delta.float32"
            )
            novel_energy_path = (
                work_dir / f"{stage_prefix}{suffix}_novel_energy.float32"
            )
            work_paths.extend((novel_accumulator_path, novel_energy_path))
            novel_accumulators.append(
                np.memmap(
                    novel_accumulator_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(height, width, 3),
                )
            )
            novel_energy_sums.append(
                np.memmap(
                    novel_energy_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(height, width),
                )
            )
    for value in (
        accumulators
        + weight_sums
        + energy_sums
        + novel_accumulators
        + novel_energy_sums
    ):
        value[:] = 0

    records = []
    processed = 0
    skipped = 0
    stats_sum = {
        "mean_gate": 0.0,
        "mean_structure_gate": 0.0,
        "mean_base_detail_gate": 0.0,
        "mean_abs_delta": 0.0,
        "clipped_fraction": 0.0,
    }
    novel_stats_sum = {
        "mean_gate": 0.0,
        "mean_structure_gate": 0.0,
        "mean_novelty_gate": 0.0,
        "mean_abs_delta": 0.0,
        "clipped_fraction": 0.0,
    }
    for sequence, tile in enumerate(plans, start=1):
        context = extract_tile_context(base, tile)
        local_core = context.crop(tile.local_core_box)
        score = detail_score(np.asarray(local_core, dtype=np.uint8))
        steps = adaptive_step_count(score, args.minimum_steps, args.steps, knee=args.detail_knee)
        seed = coordinate_seed(
            args.seed,
            tile.phase + stage_number * args.phase_count,
            tile.grid_core_x0,
            tile.grid_core_y0,
        )
        is_skipped = args.skip_flat_below > 0 and score < args.skip_flat_below
        records.append(_manifest_tile(tile, seed=seed, steps=steps, score=score, skipped=is_skipped))
        if is_skipped:
            skipped += 1
            emit(f"STAGE={stage_number}/{stage_count} TILE={sequence}/{len(plans)} " f"SKIP_FLAT SCORE={score:.6f}")
            continue

        emit(f"STAGE={stage_number}/{stage_count} TILE={sequence}/{len(plans)} " f"PHASE={tile.phase + 1}/{args.phase_count} GRID_CORE=" f"{tile.grid_core_x0},{tile.grid_core_y0},{tile.grid_core_width}x{tile.grid_core_height} " f"CANVAS_CORE={tile.core_x0},{tile.core_y0},{tile.core_width}x{tile.core_height} " f"CONTEXT={context.width}x{context.height} STEPS={steps} " f"DENOISE={denoise:.3f} SCORE={score:.6f} SEED={seed}")
        payload, diffusion_size = build_tile_payload(
            args,
            context,
            prompt,
            negative_prompt,
            seed=seed,
            steps=steps,
            denoise=denoise,
        )
        data = post_img2img(args, payload, f"VRAM_CANVAS_S{stage_number:02d}_T{sequence:04d}")
        returned = data.get("images") or []
        if not returned:
            raise RuntimeError(f"Stage {stage_number} tile {sequence} returned no image.")
        refined = decode_b64_image(returned[0])
        if refined.size != diffusion_size:
            raise RuntimeError(f"Stage {stage_number} tile {sequence} returned {refined.size}; " f"expected {diffusion_size}.")
        refined = refined.crop((0, 0, context.width, context.height))
        delta, delta_stats = frequency_detail_delta(
            np.asarray(refined, dtype=np.uint8),
            np.asarray(context, dtype=np.uint8),
            radius=args.low_pass_radius,
            gain=args.detail_gain,
            max_delta=args.max_detail_delta,
            structure_sigma=args.structure_sigma,
            base_detail_sigma=args.base_detail_sigma,
        )
        local_x0, local_y0, local_x1, local_y1 = tile.local_core_box
        core_delta = delta[local_y0:local_y1, local_x0:local_x1]
        mask = phase_normalized_tile_weight(tile, phase_normalizers)
        canvas_slice = np.s_[tile.core_y0 : tile.core_y1, tile.core_x0 : tile.core_x1]
        phase_slot = tile.phase if args.merge_mode == PHASE_WEAVE_MERGE_MODE else 0
        accumulators[phase_slot][canvas_slice] += core_delta * mask[..., None]
        weight_sums[phase_slot][canvas_slice] += mask
        energy_sums[phase_slot][canvas_slice] += (
            np.mean(np.square(core_delta), axis=2, dtype=np.float32) * mask
        )
        if novel_accumulators:
            novel_delta, novel_stats = novel_detail_delta(
                np.asarray(refined, dtype=np.uint8),
                np.asarray(context, dtype=np.uint8),
                inner_radius=args.novel_detail_inner_radius,
                outer_radius=args.novel_detail_outer_radius,
                gain=args.novel_detail_gain,
                max_delta=args.novel_detail_max_delta,
                structure_sigma=args.novel_detail_structure_sigma,
                base_detail_sigma=args.base_detail_sigma,
            )
            novel_core_delta = novel_delta[
                local_y0:local_y1, local_x0:local_x1
            ]
            novel_accumulators[phase_slot][canvas_slice] += (
                novel_core_delta * mask[..., None]
            )
            novel_energy_sums[phase_slot][canvas_slice] += (
                np.mean(
                    np.square(novel_core_delta),
                    axis=2,
                    dtype=np.float32,
                )
                * mask
            )
            for name in novel_stats_sum:
                novel_stats_sum[name] += novel_stats[name]
        processed += 1
        for name in stats_sum:
            stats_sum[name] += delta_stats[name]

        if args.save_tiles:
            tile_path = output_dir / f"{stage_prefix}_tile_{sequence:04d}.png"
            refined.save(tile_path, format="PNG")

    for value in (
        accumulators
        + weight_sums
        + energy_sums
        + novel_accumulators
        + novel_energy_sums
    ):
        value.flush()
    stage_path = output_dir / f"{stage_prefix}_{width}x{height}.png"
    phase_candidate_paths = (
        (
            output_dir / f"{stage_prefix}_phase_a_{width}x{height}.png",
            output_dir / f"{stage_prefix}_phase_b_{width}x{height}.png",
        )
        if args.save_phase_candidates
        else None
    )
    phase_selection_path = (
        output_dir / f"{stage_prefix}_phase_selection_{width}x{height}.png"
        if args.save_phase_candidates
        else None
    )
    result_image, consensus_stats = _save_stage_result(
        base,
        accumulators,
        weight_sums,
        energy_sums,
        novel_accumulators,
        novel_energy_sums,
        stage_path,
        result_work_path,
        stripe_height=args.finalize_stripe_height,
        consensus_sigma=args.consensus_sigma,
        novel_consensus_sigma=args.novel_detail_consensus_sigma,
        novel_consensus_strength=args.novel_detail_consensus_strength,
        merge_mode=args.merge_mode,
        phase_candidate_paths=phase_candidate_paths,
        phase_selection_path=phase_selection_path,
    )
    for value in (
        accumulators
        + weight_sums
        + energy_sums
        + novel_accumulators
        + novel_energy_sums
    ):
        _close_memmap(value)
    if not args.keep_work:
        for work_path in work_paths:
            work_path.unlink(missing_ok=True)
        result_work_path.unlink(missing_ok=True)

    averaged_stats = {name: (value / processed if processed else 0.0) for name, value in stats_sum.items()}
    averaged_novel_stats = {
        name: (value / processed if processed else 0.0)
        for name, value in novel_stats_sum.items()
    }
    report = {
        "stage": stage_number,
        "size": [width, height],
        "tile_count": len(plans),
        "processed_tile_count": processed,
        "skipped_tile_count": skipped,
        "denoise": denoise,
        "grid_origin": grid_origin,
        "output": str(stage_path),
        "delta_stats": averaged_stats,
        "novel_delta_stats": averaged_novel_stats,
        "consensus_stats": consensus_stats,
        "tiles": records,
    }
    return result_image, report


def save_final_png(
    path: Path,
    image: Image.Image,
    *,
    parameters: str,
    prompt: str,
    negative_prompt: str,
    source_size: tuple[int, int],
    report: dict,
):
    pnginfo = PngImagePlugin.PngInfo()
    width, height = image.size
    parameters = replace_infotext_prompts(parameters, prompt, negative_prompt)
    if "\nSteps:" not in parameters:
        parameters += (
            "\nSteps: adaptive, VRAM-Canvas: frequency-separated progressive refinement, "
            f"Size: {width}x{height}, Seed: {report['seed']}"
        )
    parameters = replace_infotext_size(
        parameters,
        source_size[0],
        source_size[1],
        width,
        height,
    )
    parameters = replace_infotext_seed(parameters, report["seed"])
    pnginfo.add_text("parameters", parameters)
    pnginfo.add_text(
        "vram_canvas",
        json.dumps(report, ensure_ascii=False, separators=(",", ":")),
    )
    if report.get("merge_mode") == PHASE_WEAVE_MERGE_MODE:
        pnginfo.add_text(
            "krea2_phaseweave",
            json.dumps(
                {
                    "format_version": 4,
                    "product_name": KREA2_PHASEWEAVE_PRODUCT_NAME,
                    "profile_key": KREA2_PHASEWEAVE_PROFILE_KEY,
                    "merge_mode": PHASE_WEAVE_MERGE_MODE,
                    "target_size": report["target_size"],
                    "phase_count": report["phase_count"],
                    "grid_layout": report["grid_layout"],
                    "grid_stride": report["grid_stride"],
                    "grid_phase_offset": report["grid_phase_offset"],
                    "grid_padding_mode": report["grid_padding_mode"],
                    "grid_origin": report["grid_origin"],
                    **phase_weave_configuration(),
                    "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
                    "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
                    "stage_reports": report["stage_reports"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    image.save(path, format="PNG", pnginfo=pnginfo)


def main() -> int:
    parser = argparse.ArgumentParser(description=("VRAM-Canvas: progressive halo-tile img2img with frequency-separated " "residual blending and disk-backed accumulation."))
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--api", default=DEFAULT_API, help="Forge API base URL.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--krea2-profile",
        choices=tuple(sorted(KREA2_VRAM_CANVAS_PROFILES)),
        default=None,
        help="Apply shared Krea2 defaults; explicitly supplied flags still win.",
    )
    parser.add_argument(
        "--append-krea2-detail-prompt",
        action="store_true",
        help="Append geometry-preserving Krea2 dense-detail guidance once.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--long-edge", type=int, default=4096)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--max-stage-scale", type=float, default=2.0)

    parser.add_argument(
        "--vram-budget-gib",
        type=float,
        default=0.0,
        help="Total VRAM budget. 0 queries Forge's memory endpoint.",
    )
    parser.add_argument("--fallback-vram-gib", type=float, default=DEFAULT_FALLBACK_VRAM_GIB)
    parser.add_argument("--vram-use-fraction", type=float, default=DEFAULT_VRAM_USE_FRACTION)
    parser.add_argument("--model-reserve-gib", type=float, default=DEFAULT_MODEL_RESERVE_GIB)
    parser.add_argument(
        "--activation-bytes-per-pixel",
        type=int,
        default=DEFAULT_ACTIVATION_BYTES_PER_PIXEL,
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Diffusion payload edge. 0 uses the VRAM budget planner.",
    )
    parser.add_argument("--minimum-tile-size", type=int, default=DEFAULT_MIN_TILE_SIZE)
    parser.add_argument("--maximum-tile-size", type=int, default=DEFAULT_MAX_TILE_SIZE)
    parser.add_argument("--halo", type=int, default=0, help="0 selects tile/8.")
    parser.add_argument("--core-overlap", type=int, default=0, help="0 selects halo/2.")
    parser.add_argument(
        "--phase-count",
        type=int,
        choices=(1, 2),
        default=1,
        help="2 adds a half-stride shifted refinement pass.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=(CONSENSUS_MERGE_MODE, PHASE_WEAVE_MERGE_MODE),
        default=CONSENSUS_MERGE_MODE,
        help="consensus averages phase moments; phase_weave selects a local representative.",
    )

    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--minimum-steps", type=int, default=2)
    parser.add_argument("--detail-knee", type=float, default=0.035)
    parser.add_argument(
        "--skip-flat-below",
        type=float,
        default=0.0,
        help="0 refines every tile; a positive detail score may skip flat tiles.",
    )
    parser.add_argument("--coarse-denoise", type=float, default=0.12)
    parser.add_argument("--denoise", type=float, default=0.08)
    parser.add_argument("--sampler", default=KREA2_DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=KREA2_DEFAULT_SCHEDULER)
    parser.add_argument("--cfg", type=float, default=KREA2_DEFAULT_CFG)
    parser.add_argument("--distilled-cfg", type=float, default=KREA2_DEFAULT_SHIFT)
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--low-pass-radius", type=int, default=DEFAULT_LOW_PASS_RADIUS)
    parser.add_argument("--detail-gain", type=float, default=1.0)
    parser.add_argument("--max-detail-delta", type=float, default=DEFAULT_MAX_DETAIL_DELTA)
    parser.add_argument("--structure-sigma", type=float, default=DEFAULT_STRUCTURE_SIGMA)
    parser.add_argument(
        "--base-detail-sigma",
        type=float,
        default=DEFAULT_BASE_DETAIL_SIGMA,
        help="Flat-region protection strength in base high-frequency pixel levels; 0 disables the gate.",
    )
    parser.add_argument(
        "--consensus-sigma",
        type=float,
        default=DEFAULT_CONSENSUS_SIGMA,
        help="Relative-consensus noise floor in output pixel levels; 0 disables the gate.",
    )
    parser.add_argument("--novel-detail-gain", type=float, default=0.0)
    parser.add_argument("--novel-detail-max-delta", type=float, default=8.0)
    parser.add_argument(
        "--novel-detail-inner-radius",
        type=int,
        default=DEFAULT_NOVEL_DETAIL_INNER_RADIUS,
    )
    parser.add_argument(
        "--novel-detail-outer-radius",
        type=int,
        default=DEFAULT_NOVEL_DETAIL_OUTER_RADIUS,
    )
    parser.add_argument(
        "--novel-detail-structure-sigma",
        type=float,
        default=DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA,
    )
    parser.add_argument(
        "--novel-detail-consensus-sigma",
        type=float,
        default=DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA,
    )
    parser.add_argument(
        "--novel-detail-consensus-strength",
        type=float,
        default=DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH,
    )
    parser.add_argument(
        "--finish-detail-strength",
        type=float,
        default=0.0,
        help="Final coherent-detail amplification; 0 disables the finish.",
    )
    parser.add_argument("--finish-detail-radius", type=float, default=1.0)
    parser.add_argument("--finish-detail-threshold", type=float, default=0.6)
    parser.add_argument("--finish-max-detail-delta", type=float, default=5.0)
    parser.add_argument("--finalize-stripe-height", type=int, default=128)
    parser.add_argument("--max-output-pixels", type=int, default=DEFAULT_MAX_OUTPUT_PIXELS)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--progress-interval", type=float, default=20.0)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument("--save-tiles", action="store_true")
    parser.add_argument(
        "--save-phase-candidates",
        action="store_true",
        help=(
            "With phase_weave, also save the independently completed phase A "
            "and phase B images used by the selector."
        ),
    )
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    profile_probe = argparse.ArgumentParser(add_help=False)
    profile_probe.add_argument(
        "--krea2-profile",
        choices=tuple(sorted(KREA2_VRAM_CANVAS_PROFILES)),
        default=None,
    )
    selected_profile, _ = profile_probe.parse_known_args()
    if selected_profile.krea2_profile:
        parser.set_defaults(
            **krea2_profile_cli_defaults(selected_profile.krea2_profile)
        )
    args = parser.parse_args()
    validate_args(args)
    args.seed = resolve_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    with Image.open(input_path) as image:
        parameters = str(image.info.get("parameters", ""))
        source = flatten_source_image(image)
    source_size = source.size
    target_width, target_height = target_size(
        source.width,
        source.height,
        args.long_edge,
        args.width,
        args.height,
    )
    target_pixels = target_width * target_height
    if target_pixels > args.max_output_pixels:
        raise ValueError(f"target {target_width}x{target_height} has {target_pixels:,} pixels, " "exceeding --max-output-pixels.")

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else png_negative
    if not prompt:
        raise ValueError("Prompt is empty. Pass --prompt or use a PNG with infotext.")
    if args.append_krea2_detail_prompt:
        prompt = krea2_detail_prompt(prompt)

    detected_vram_gib = None
    if args.vram_budget_gib > 0:
        vram_budget_gib = args.vram_budget_gib
        vram_source = "explicit"
    else:
        detected_vram_gib = query_total_vram_gib(args.api)
        vram_budget_gib = detected_vram_gib or args.fallback_vram_gib
        vram_source = "forge-memory-endpoint" if detected_vram_gib else "fallback"
    tile_size = resolve_tile_size(
        vram_budget_gib,
        requested_tile_size=args.tile_size,
        use_fraction=args.vram_use_fraction,
        model_reserve_gib=args.model_reserve_gib,
        activation_bytes_per_pixel=args.activation_bytes_per_pixel,
        minimum=args.minimum_tile_size,
        maximum=args.maximum_tile_size,
    )
    halo = resolve_halo(tile_size, args.halo)
    core_size = tile_size - 2 * halo
    core_overlap = resolve_core_overlap(core_size, halo, args.core_overlap)
    stages = progressive_stage_sizes(
        source.width,
        source.height,
        target_width,
        target_height,
        max_stage_scale=args.max_stage_scale,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"vram_canvas_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    work_bytes_per_pixel = vram_canvas_work_bytes_per_pixel(
        phase_count=args.phase_count,
        merge_mode=args.merge_mode,
        novel_detail=args.novel_detail_gain > 0,
    )
    # Three RGB uint8 work maps plus conservative room for their encoded PNGs.
    phase_candidate_bytes_per_pixel = 18 if args.save_phase_candidates else 0
    required_work_bytes = sum(
        width
        * height
        * (work_bytes_per_pixel + phase_candidate_bytes_per_pixel)
        for width, height in stages
    )
    free_disk_bytes = shutil.disk_usage(output_dir).free
    if free_disk_bytes < required_work_bytes:
        raise RuntimeError(f"VRAM-Canvas needs about {required_work_bytes / 1024**3:.2f} GiB " f"of work space, but only {free_disk_bytes / 1024**3:.2f} GiB is free.")

    reduction = spatial_activation_reduction(target_width, target_height, tile_size)
    manifest = {
        "format_version": 5,
        "algorithm": (
            KREA2_PHASEWEAVE_PRODUCT_NAME
            if args.merge_mode == PHASE_WEAVE_MERGE_MODE
            else "VRAM-Canvas"
        ),
        "input": str(input_path),
        "source_size": list(source.size),
        "target_size": [target_width, target_height],
        "stages": [list(size) for size in stages],
        "seed": args.seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "krea2_profile": args.krea2_profile,
        "krea2_detail_prompt_appended": bool(args.append_krea2_detail_prompt),
        "vram_budget_gib": vram_budget_gib,
        "vram_budget_source": vram_source,
        "detected_vram_gib": detected_vram_gib,
        "vram_use_fraction": args.vram_use_fraction,
        "model_reserve_gib": args.model_reserve_gib,
        "activation_bytes_per_pixel": args.activation_bytes_per_pixel,
        "tile_size": tile_size,
        "halo": halo,
        "core_size": core_size,
        "core_overlap": core_overlap,
        "phase_count": args.phase_count,
        "merge_mode": args.merge_mode,
        "save_phase_candidates": bool(args.save_phase_candidates),
        "grid_layout": (
            "uniform_virtual_edge_balanced"
            if args.merge_mode == PHASE_WEAVE_MERGE_MODE
            else "legacy_edge_anchored"
        ),
        "grid_stride": core_size - core_overlap,
        "grid_phase_offset": (
            (core_size - core_overlap) // args.phase_count
            if args.phase_count > 1
            else 0
        ),
        "grid_padding_mode": (
            "edge" if args.merge_mode == PHASE_WEAVE_MERGE_MODE else "none"
        ),
        "grid_origin": (
            [
                balanced_virtual_axis_origin(
                    target_width,
                    core_size,
                    core_overlap,
                    phase_count=args.phase_count,
                ),
                balanced_virtual_axis_origin(
                    target_height,
                    core_size,
                    core_overlap,
                    phase_count=args.phase_count,
                ),
            ]
            if args.merge_mode == PHASE_WEAVE_MERGE_MODE
            else [0, 0]
        ),
        "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
        "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
        "phaseweave": {
            "enabled": args.merge_mode == PHASE_WEAVE_MERGE_MODE,
            "product_name": KREA2_PHASEWEAVE_PRODUCT_NAME,
            "profile_key": KREA2_PHASEWEAVE_PROFILE_KEY,
            "grid_layout": (
                "uniform_virtual_edge_balanced"
                if args.merge_mode == PHASE_WEAVE_MERGE_MODE
                else "legacy_edge_anchored"
            ),
            **phase_weave_configuration(),
        },
        "estimated_spatial_activation_reduction": reduction,
        "frequency_merge": {
            "low_pass": "separable box",
            "radius": args.low_pass_radius,
            "detail_gain": args.detail_gain,
            "max_detail_delta": args.max_detail_delta,
            "structure_sigma": args.structure_sigma,
            "base_detail_sigma": args.base_detail_sigma,
            "consensus_sigma": args.consensus_sigma,
            "novel_detail_gain": args.novel_detail_gain,
            "novel_detail_max_delta": args.novel_detail_max_delta,
            "novel_detail_inner_radius": args.novel_detail_inner_radius,
            "novel_detail_outer_radius": args.novel_detail_outer_radius,
            "novel_detail_structure_sigma": args.novel_detail_structure_sigma,
            "novel_detail_consensus_sigma": args.novel_detail_consensus_sigma,
            "novel_detail_consensus_strength": args.novel_detail_consensus_strength,
        },
        "texture_finish": {
            "enabled": args.finish_detail_strength > 0,
            "detail_strength": args.finish_detail_strength,
            "detail_radius": args.finish_detail_radius,
            "detail_threshold": args.finish_detail_threshold,
            "max_detail_delta": args.finish_max_detail_delta,
            "report": None,
        },
        "stage_reports": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"SOURCE={source.width}x{source.height}")
    emit(f"TARGET={target_width}x{target_height} ({target_pixels / 1_000_000:.1f} MP)")
    emit(f"KREA2_PROFILE={args.krea2_profile or 'custom'}")
    emit(f"STAGES={','.join(f'{w}x{h}' for w, h in stages)}")
    emit(f"VRAM_BUDGET_GIB={vram_budget_gib:.2f} SOURCE={vram_source}")
    emit(f"TILE={tile_size} HALO={halo} CORE={core_size} OVERLAP={core_overlap} " f"PHASES={args.phase_count} MERGE={args.merge_mode}")
    emit(f"GRID_LAYOUT={manifest['grid_layout']} STRIDE={manifest['grid_stride']} " f"PHASE_OFFSET={manifest['grid_phase_offset']} ORIGIN={manifest['grid_origin'][0]},{manifest['grid_origin'][1]} " f"PADDING={manifest['grid_padding_mode']}")
    emit(f"SPATIAL_ACTIVATION_REDUCTION_ESTIMATE={reduction:.2f}x")
    emit(f"MANIFEST={manifest_path}")

    current = source
    if args.dry_run:
        for stage_index, (stage_width, stage_height) in enumerate(stages, start=1):
            current = current.resize((stage_width, stage_height), Image.Resampling.LANCZOS)
            preview_path = output_dir / f"stage_{stage_index:02d}_base.png"
            current.save(preview_path, format="PNG")
            plan = plan_tiles(
                stage_width,
                stage_height,
                tile_size=tile_size,
                halo=halo,
                core_overlap=core_overlap,
                phase_count=args.phase_count,
                virtual_padding=args.merge_mode == PHASE_WEAVE_MERGE_MODE,
            )
            manifest["stage_reports"].append(
                {
                    "stage": stage_index,
                    "size": [stage_width, stage_height],
                    "tile_count": len(plan),
                    "base_preview": str(preview_path),
                    "grid_origin": (
                        [
                            balanced_virtual_axis_origin(
                                stage_width,
                                core_size,
                                core_overlap,
                                phase_count=args.phase_count,
                            ),
                            balanced_virtual_axis_origin(
                                stage_height,
                                core_size,
                                core_overlap,
                                phase_count=args.phase_count,
                            ),
                        ]
                        if args.merge_mode == PHASE_WEAVE_MERGE_MODE
                        else [0, 0]
                    ),
                    "dry_run": True,
                }
            )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        emit("DRY_RUN=1")
        return 0

    for stage_index, (stage_width, stage_height) in enumerate(stages):
        base = current.resize((stage_width, stage_height), Image.Resampling.LANCZOS)
        current, stage_report = refine_stage(
            base,
            args,
            prompt,
            negative_prompt,
            output_dir,
            stage_index=stage_index,
            stage_count=len(stages),
            tile_size=tile_size,
            halo=halo,
            core_overlap=core_overlap,
        )
        manifest["stage_reports"].append(stage_report)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.finish_detail_strength > 0:
        emit(
            "TEXTURE_FINISH="
            f"strength={args.finish_detail_strength:.2f} "
            f"radius={args.finish_detail_radius:.2f} "
            f"threshold={args.finish_detail_threshold:.2f} "
            f"max_delta={args.finish_max_detail_delta:.2f}"
        )
        current, finish_report = adaptive_detail_guard(
            current,
            strength=float(args.finish_detail_strength),
            radius=float(args.finish_detail_radius),
            detail_threshold=float(args.finish_detail_threshold),
            max_detail_delta=float(args.finish_max_detail_delta),
        )
        manifest["texture_finish"]["report"] = finish_report
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    final_path = output_dir / "vram_canvas_highres.png"
    save_final_png(
        final_path,
        current,
        parameters=parameters,
        prompt=prompt,
        negative_prompt=negative_prompt,
        source_size=source_size,
        report=manifest,
    )
    emit(f"IMAGE={final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
