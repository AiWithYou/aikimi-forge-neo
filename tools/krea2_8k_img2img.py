import argparse
import base64
import io
import json
import secrets
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from modules.krea2_quality import smart_finish_image, smart_finish_summary
from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_CFG,
    KREA2_DEFAULT_DENOISE,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_DEFAULT_SHIFT,
    KREA2_I2I_STEPS,
    KREA2_STAGE1_DENOISE,
    capped_diffusion_size,
    native_diffusion_long_edge,
    require_native_diffusion_size,
    require_positive_int,
    require_safe_diffusion_size,
    replace_infotext_size,
    target_size,
    two_stage_sizes,
    validate_tile_geometry,
)


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output"
MAX_SAFE_DELIVERY_LONG_EDGE = 8192
MAX_SAFE_DELIVERY_PIXELS = 20_000_000


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def validate_args(args):
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be passed together.")

    for value, name in (
        (args.long_edge, "--long-edge"),
        (args.width, "--width"),
        (args.height, "--height"),
        (args.steps, "--steps"),
        (args.tile_width, "--tile-width"),
        (args.tile_height, "--tile-height"),
        (args.tile_batch_size, "--tile-batch-size"),
        (args.timeout, "--timeout"),
    ):
        require_positive_int(value, name)

    if args.diffusion_long_edge_cap is not None and args.diffusion_long_edge_cap < 0:
        raise ValueError("--diffusion-long-edge-cap must be >= 0.")
    if args.first_pass_long_edge < 0:
        raise ValueError("--first-pass-long-edge must be >= 0.")
    if args.tile_overlap < 0:
        raise ValueError("--tile-overlap must be >= 0.")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be >= 0.")
    if args.no_progress_timeout < 0:
        raise ValueError("--no-progress-timeout must be >= 0.")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")
    if not 0 <= args.denoise <= 1:
        raise ValueError("--denoise must be between 0 and 1.")
    if args.cfg < 0:
        raise ValueError("--cfg must be >= 0.")
    if args.distilled_cfg < 0:
        raise ValueError("--distilled-cfg must be >= 0.")
    if not 0 <= args.first_pass_denoise <= 1:
        raise ValueError("--first-pass-denoise must be between 0 and 1.")
    if not 0 <= args.smart_color_strength <= 1:
        raise ValueError("--smart-color-strength must be between 0 and 1.")
    require_positive_int(args.smart_analysis_long_edge, "--smart-analysis-long-edge")
    if args.smart_max_speckle_percent <= 0:
        raise ValueError("--smart-max-speckle-percent must be > 0.")
    native_diffusion_long_edge(args.model_profile)
    validate_tile_geometry(
        args.tile_width,
        args.tile_height,
        args.tile_overlap,
        args.tile_batch_size,
    )


def resolve_seed(seed: int) -> int:
    if seed < -1:
        raise ValueError("--seed must be -1 or a non-negative integer.")
    return secrets.randbelow(2**32) if seed == -1 else seed


def validate_delivery_size(
    width: int, height: int, *, allow_unsafe_large_delivery: bool = False
):
    require_positive_int(width, "delivery width")
    require_positive_int(height, "delivery height")
    if allow_unsafe_large_delivery:
        return
    pixels = width * height
    if max(width, height) > MAX_SAFE_DELIVERY_LONG_EDGE:
        raise ValueError(
            f"delivery long edge exceeds {MAX_SAFE_DELIVERY_LONG_EDGE}; pass "
            "--allow-unsafe-large-delivery only after checking host RAM."
        )
    if pixels > MAX_SAFE_DELIVERY_PIXELS:
        raise ValueError(
            f"delivery image has {pixels:,} pixels, exceeding the safe limit "
            f"{MAX_SAFE_DELIVERY_PIXELS:,}; pass --allow-unsafe-large-delivery "
            "only after checking host RAM."
        )


def flatten_source_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and flatten transparency onto a defined white matte."""
    oriented = ImageOps.exif_transpose(image)
    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    if not has_alpha:
        return oriented.convert("RGB")
    rgba = oriented.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def resolve_upscale_mode(
    requested_mode: str,
    source_width: int,
    source_height: int,
    diffusion_width: int,
    diffusion_height: int,
) -> str:
    if requested_mode in {"single-stage", "two-stage"}:
        return requested_mode
    if requested_mode != "auto":
        raise ValueError("upscale mode must be auto, single-stage, or two-stage.")

    source_long_edge = max(source_width, source_height)
    diffusion_long_edge = max(diffusion_width, diffusion_height)
    if diffusion_long_edge >= source_long_edge + 128:
        return "two-stage"
    return "single-stage"


def validate_generated_image(
    image: Image.Image, expected_size: tuple[int, int], stage_name: str
) -> dict:
    if image.size != expected_size:
        raise RuntimeError(
            f"{stage_name} returned {image.size}, expected {expected_size}."
        )
    preview = image.convert("RGB")
    preview.thumbnail((256, 256), Image.Resampling.BOX)
    values = np.asarray(preview, dtype=np.float32)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 0.25 and (mean < 1.0 or mean > 254.0):
        raise RuntimeError(
            f"{stage_name} returned a near-empty image (mean={mean:.3f}, std={std:.3f})."
        )
    return {"mean": mean, "std": std}


def response_infotext(data: dict) -> str:
    raw_info = data.get("info", "")
    if not isinstance(raw_info, str):
        return str(raw_info)
    try:
        parsed = json.loads(raw_info)
    except (TypeError, ValueError):
        return raw_info
    infotexts = parsed.get("infotexts") if isinstance(parsed, dict) else None
    if isinstance(infotexts, list) and infotexts:
        return str(infotexts[0])
    return raw_info


def save_png(
    path: Path,
    image: Image.Image,
    *,
    parameters: str,
    quality_report: dict | None = None,
):
    pnginfo = PngImagePlugin.PngInfo()
    if parameters:
        pnginfo.add_text("parameters", parameters)
    if quality_report is not None:
        pnginfo.add_text(
            "krea2_smart_finish",
            json.dumps(quality_report, ensure_ascii=False, separators=(",", ":")),
        )
    image.save(path, format="PNG", pnginfo=pnginfo)


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_b64_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        return image.convert("RGB")


def prompt_from_png(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        params = image.info.get("parameters", "")
    if not params:
        return "", ""
    negative = ""
    prompt_block = params
    if "\nNegative prompt:" in params:
        prompt_block, rest = params.split("\nNegative prompt:", 1)
        negative = rest.split("\nSteps:", 1)[0].strip()
    elif "\nSteps:" in params:
        prompt_block = params.split("\nSteps:", 1)[0]
    return prompt_block.strip(), negative


def build_img2img_payload(
    args,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    denoise: float,
    send_images: bool,
    save_images: bool,
) -> dict:
    alwayson_scripts = {
        "multidiffusion integrated": {
            "args": [
                True,
                args.method,
                args.tile_width,
                args.tile_height,
                args.tile_overlap,
                args.tile_batch_size,
            ]
        }
    }
    if args.always_tiled_vae:
        alwayson_scripts["never oom integrated"] = {
            "args": [False, True]
        }

    return {
        "init_images": [image_to_b64_png(image)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": width,
        "height": height,
        "resize_mode": 0,
        "denoising_strength": denoise,
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": send_images,
        "save_images": save_images,
        "include_init_images": False,
        "override_settings": {"img2img_fix_steps": True},
        "override_settings_restore_afterwards": True,
        "alwayson_scripts": alwayson_scripts,
    }


def write_payload_preview(
    output_dir: Path,
    name: str,
    payload: dict,
    init_image_description: str,
    extra: dict | None = None,
) -> Path:
    preview = dict(payload)
    preview["init_images"] = [init_image_description]
    if extra:
        preview.update(extra)
    preview_path = output_dir / name
    preview_path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return preview_path


def poll_progress(args, stage_name: str, stop_event: threading.Event):
    last_error = ""
    last_progress_key = None
    last_progress_at = time.monotonic()
    while not stop_event.wait(args.progress_interval):
        try:
            response = requests.get(
                f"{args.api}/sdapi/v1/progress",
                params={"skip_current_image": "true"},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
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
        elif (
            args.no_progress_timeout > 0
            and time.monotonic() - last_progress_at >= args.no_progress_timeout
        ):
            emit(f"{stage_name}_NO_PROGRESS_TIMEOUT={args.no_progress_timeout}")
            try:
                requests.post(f"{args.api}/sdapi/v1/interrupt", timeout=5)
            except requests.RequestException as exc:
                emit(f"{stage_name}_INTERRUPT_ERROR={type(exc).__name__}: {exc}")
            stop_event.set()
            return

        if isinstance(progress, (int, float)):
            progress_text = f"{progress:.4f}"
        else:
            progress_text = str(progress)
        emit(
            f"{stage_name}_PROGRESS={progress_text} "
            f"STEP={state.get('sampling_step')}/{state.get('sampling_steps')} "
            f"JOB={state.get('job')} ETA={data.get('eta_relative')}"
        )


def post_img2img(args, payload: dict, stage_name: str) -> dict:
    stop_event = None
    progress_thread = None
    if args.progress_interval > 0:
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=poll_progress, args=(args, stage_name, stop_event), daemon=True
        )
        progress_thread.start()
    try:
        response = requests.post(
            f"{args.api}/sdapi/v1/img2img", json=payload, timeout=args.timeout
        )
    finally:
        if stop_event is not None:
            stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1)
    emit(f"{stage_name}_HTTP={response.status_code}")
    if response.status_code != 200:
        emit(response.text[:4000])
        raise RuntimeError(f"{stage_name} img2img failed: HTTP {response.status_code}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Krea2 profile-guarded img2img followed by safe high-resolution delivery."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--api", default=DEFAULT_API, help="Forge API base URL.")
    parser.add_argument(
        "--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory."
    )
    parser.add_argument(
        "--prompt", default=None, help="Prompt. Defaults to PNG infotext prompt."
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Defaults to PNG infotext negative prompt.",
    )
    parser.add_argument(
        "--upscale-mode",
        default="auto",
        choices=["auto", "single-stage", "two-stage"],
        help="Upscale flow. Auto skips the intermediate pass when the source is already near the diffusion proxy size.",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=4096,
        help="Target long edge when width/height are not set.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Explicit target width. Must be passed with --height.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Explicit target height. Must be passed with --width.",
    )
    parser.add_argument(
        "--first-pass-long-edge",
        type=int,
        default=0,
        help="Intermediate long edge for --upscale-mode two-stage. 0 selects an automatic size.",
    )
    parser.add_argument(
        "--model-profile",
        choices=["raw", "turbo", "custom"],
        default="custom",
        help="Krea2 resolution profile. Raw is limited to 1K; Turbo and unknown custom checkpoints default to a conservative 2K proxy.",
    )
    parser.add_argument(
        "--diffusion-long-edge-cap",
        type=int,
        default=None,
        help="Maximum long edge used for diffusion. Defaults to the selected model profile's resolution guard.",
    )
    parser.add_argument(
        "--allow-non-native-diffusion",
        action="store_true",
        help="Explicitly allow diffusion above the selected Krea2 profile's resolution guard.",
    )
    parser.add_argument(
        "--allow-unsafe-large-diffusion",
        action="store_true",
        help="Allow a final diffusion pass above the safe long-edge limit. This can crash the GPU driver on 24GB RTX 3090-class systems.",
    )
    parser.add_argument(
        "--allow-unsafe-large-delivery",
        action="store_true",
        help="Allow a local delivery image above 8192px or 20MP. This can exhaust host RAM during resize/finishing.",
    )
    parser.add_argument("--steps", type=int, default=KREA2_I2I_STEPS)
    parser.add_argument("--sampler", default=KREA2_DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=KREA2_DEFAULT_SCHEDULER)
    parser.add_argument("--cfg", type=float, default=KREA2_DEFAULT_CFG)
    parser.add_argument("--distilled-cfg", type=float, default=KREA2_DEFAULT_SHIFT)
    parser.add_argument(
        "--first-pass-denoise", type=float, default=KREA2_STAGE1_DENOISE
    )
    parser.add_argument("--denoise", type=float, default=KREA2_DEFAULT_DENOISE)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--tile-width", type=int, default=768)
    parser.add_argument("--tile-height", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument(
        "--method",
        default="Mixture of Diffusers",
        choices=["MultiDiffusion", "Mixture of Diffusers"],
    )
    parser.add_argument(
        "--always-tiled-vae",
        action="store_true",
        help=(
            "Enable Never OOM's tiled VAE for this API request instead of "
            "first attempting a full-canvas VAE pass."
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=43200, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between Forge progress polling messages during API calls. 0 disables progress polling.",
    )
    parser.add_argument(
        "--no-progress-timeout",
        type=float,
        default=600.0,
        help="Interrupt Forge if progress does not change for this many seconds. 0 disables this watchdog.",
    )
    parser.add_argument(
        "--return-image",
        action="store_true",
        help="Retained for command compatibility. High-resolution mode always returns and validates the final proxy image.",
    )
    parser.add_argument(
        "--smart-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply adaptive chroma cleanup after final resize, accepting it only when measured chroma mura improves.",
    )
    parser.add_argument(
        "--smart-despeckle",
        action="store_true",
        help="Repair isolated bright/dark speckles. Leave off for snow, stars, freckles, or deliberate grain.",
    )
    parser.add_argument("--smart-color-strength", type=float, default=0.80)
    parser.add_argument("--smart-analysis-long-edge", type=int, default=1536)
    parser.add_argument("--smart-max-speckle-percent", type=float, default=0.35)
    parser.add_argument(
        "--quality-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the machine-readable quality report and embed it in the PNG.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write resized input and request preview, but do not call Forge.",
    )
    args = parser.parse_args()
    validate_args(args)
    if args.diffusion_long_edge_cap is None:
        args.diffusion_long_edge_cap = native_diffusion_long_edge(args.model_profile)
    args.seed = resolve_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as image:
        source = flatten_source_image(image)
    target_w, target_h = target_size(
        source.width, source.height, args.long_edge, args.width, args.height
    )
    validate_delivery_size(
        target_w,
        target_h,
        allow_unsafe_large_delivery=args.allow_unsafe_large_delivery,
    )
    diffusion_w, diffusion_h = capped_diffusion_size(
        source.width, source.height, target_w, target_h, args.diffusion_long_edge_cap
    )
    require_safe_diffusion_size(
        diffusion_w, diffusion_h, args.allow_unsafe_large_diffusion
    )
    require_native_diffusion_size(
        diffusion_w,
        diffusion_h,
        args.model_profile,
        args.allow_non_native_diffusion,
    )
    needs_final_resize = (diffusion_w, diffusion_h) != (target_w, target_h)
    upscale_mode = resolve_upscale_mode(
        args.upscale_mode,
        source.width,
        source.height,
        diffusion_w,
        diffusion_h,
    )

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = (
        args.negative_prompt if args.negative_prompt is not None else png_negative
    )
    if not prompt:
        raise ValueError(
            "Prompt is empty. Pass --prompt or use a PNG with Forge infotext."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"krea2_highres_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    run_started_at = time.time()
    stage1_stats = None

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"MODEL_PROFILE={args.model_profile}")
    emit(f"SEED={args.seed}")
    emit(f"UPSCALE_MODE_REQUESTED={args.upscale_mode}")
    emit(f"UPSCALE_MODE={upscale_mode}")
    emit(f"ALWAYS_TILED_VAE={int(args.always_tiled_vae)}")
    emit(f"TARGET={target_w}x{target_h} ({target_w * target_h / 1_000_000:.1f} MP)")
    if needs_final_resize:
        emit(
            f"DIFFUSION_TARGET={diffusion_w}x{diffusion_h} ({diffusion_w * diffusion_h / 1_000_000:.1f} MP)"
        )
        emit(f"FINAL_RESIZE_TARGET={target_w}x{target_h}")

    if upscale_mode == "single-stage":
        resized = source.resize((diffusion_w, diffusion_h), Image.Resampling.LANCZOS)
        resized_path = output_dir / "resized_input.png"
        resized.save(resized_path)

        payload = build_img2img_payload(
            args,
            resized,
            prompt,
            negative_prompt,
            diffusion_w,
            diffusion_h,
            args.denoise,
            True,
            False,
        )
        preview_path = write_payload_preview(
            output_dir,
            "request_preview.json",
            payload,
            f"<base64 PNG omitted: {diffusion_w}x{diffusion_h}>",
            {
                "input_path": str(input_path),
                "resized_input": str(resized_path),
                "upscale_mode": upscale_mode,
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        emit(f"RESIZED_INPUT={resized_path}")
        emit(f"REQUEST_PREVIEW={preview_path}")

        if args.dry_run:
            emit("DRY_RUN=1")
            return 0

        data = post_img2img(args, payload, "FINAL")
    else:
        (stage1_w, stage1_h), _ = two_stage_sizes(
            source.width,
            source.height,
            diffusion_w,
            diffusion_h,
            args.first_pass_long_edge,
        )
        emit(
            f"STAGE1_TARGET={stage1_w}x{stage1_h} ({stage1_w * stage1_h / 1_000_000:.1f} MP)"
        )
        if args.first_pass_long_edge == 0:
            emit(f"STAGE1_LONG_EDGE=auto ({max(stage1_w, stage1_h)})")

        stage1_input = source.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
        stage1_input_path = output_dir / "stage1_resized_input.png"
        stage1_input.save(stage1_input_path)

        stage1_payload = build_img2img_payload(
            args,
            stage1_input,
            prompt,
            negative_prompt,
            stage1_w,
            stage1_h,
            args.first_pass_denoise,
            True,
            False,
        )
        stage1_preview_path = write_payload_preview(
            output_dir,
            "stage1_request_preview.json",
            stage1_payload,
            f"<base64 PNG omitted: {stage1_w}x{stage1_h}>",
            {
                "input_path": str(input_path),
                "stage1_resized_input": str(stage1_input_path),
                "upscale_mode": upscale_mode,
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        stage2_preview_payload = build_img2img_payload(
            args,
            Image.new("RGB", (64, 64)),
            prompt,
            negative_prompt,
            diffusion_w,
            diffusion_h,
            args.denoise,
            True,
            False,
        )
        stage2_preview_path = write_payload_preview(
            output_dir,
            "stage2_request_preview.json",
            stage2_preview_payload,
            f"<stage1 result resized to {diffusion_w}x{diffusion_h}>",
            {
                "upscale_mode": upscale_mode,
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        emit(f"STAGE1_RESIZED_INPUT={stage1_input_path}")
        emit(f"STAGE1_REQUEST_PREVIEW={stage1_preview_path}")
        emit(f"STAGE2_REQUEST_PREVIEW={stage2_preview_path}")

        if args.dry_run:
            emit("DRY_RUN=1")
            return 0

        stage1_data = post_img2img(args, stage1_payload, "STAGE1")
        if not stage1_data.get("images"):
            raise RuntimeError("STAGE1 img2img returned no image.")

        stage1_image = decode_b64_image(stage1_data["images"][0])
        stage1_stats = validate_generated_image(
            stage1_image, (stage1_w, stage1_h), "STAGE1"
        )
        stage1_output_path = output_dir / "stage1_img2img.png"
        save_png(
            stage1_output_path,
            stage1_image,
            parameters=response_infotext(stage1_data),
        )
        emit(f"STAGE1_IMAGE={stage1_output_path}")

        stage2_input = stage1_image.resize(
            (diffusion_w, diffusion_h), Image.Resampling.LANCZOS
        )
        stage2_input_path = output_dir / "stage2_resized_input.png"
        stage2_input.save(stage2_input_path)
        emit(f"STAGE2_RESIZED_INPUT={stage2_input_path}")

        stage2_payload = build_img2img_payload(
            args,
            stage2_input,
            prompt,
            negative_prompt,
            diffusion_w,
            diffusion_h,
            args.denoise,
            True,
            False,
        )
        write_payload_preview(
            output_dir,
            "stage2_request_preview.json",
            stage2_payload,
            f"<base64 PNG omitted: {diffusion_w}x{diffusion_h}>",
            {
                "upscale_mode": upscale_mode,
                "stage2_resized_input": str(stage2_input_path),
                "target_output": f"{target_w}x{target_h}",
                "diffusion_target": f"{diffusion_w}x{diffusion_h}",
                "final_resize": needs_final_resize,
            },
        )

        data = post_img2img(args, stage2_payload, "FINAL")

    raw_info = str(data.get("info", ""))
    info_path = output_dir / "response_info.txt"
    info_path.write_text(raw_info, encoding="utf-8")
    emit(f"INFO={info_path}")

    returned_images = data.get("images") or []
    if not returned_images:
        raise RuntimeError(
            "FINAL img2img returned no image for validation and local save."
        )

    diffusion_image = decode_b64_image(returned_images[0])
    diffusion_stats = validate_generated_image(
        diffusion_image, (diffusion_w, diffusion_h), "FINAL"
    )
    parameters = response_infotext(data)
    diffusion_output_path = output_dir / "final_diffusion_img2img.png"
    save_png(diffusion_output_path, diffusion_image, parameters=parameters)
    emit(f"FINAL_DIFFUSION_IMAGE={diffusion_output_path}")

    if needs_final_resize:
        delivery_image = diffusion_image.resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )
    else:
        delivery_image = diffusion_image.copy()
    parameters = replace_infotext_size(
        parameters,
        diffusion_w,
        diffusion_h,
        target_w,
        target_h,
    )

    if args.smart_finish:
        delivery_image, quality_report = smart_finish_image(
            delivery_image,
            color_strength=args.smart_color_strength,
            analysis_long_edge=args.smart_analysis_long_edge,
            despeckle=args.smart_despeckle,
            max_speckle_percent=args.smart_max_speckle_percent,
        )
        quality_summary = smart_finish_summary(quality_report)
        parameters = (
            f"{parameters}, Krea2 Smart Finish: {quality_summary}"
            if parameters
            else f"Krea2 Smart Finish: {quality_summary}"
        )
        emit(f"SMART_FINISH={quality_summary}")
    else:
        quality_report = {
            "version": 1,
            "input_size": [delivery_image.width, delivery_image.height],
            "output_size": [delivery_image.width, delivery_image.height],
            "disabled": True,
        }
        emit("SMART_FINISH=disabled")

    validate_generated_image(delivery_image, (target_w, target_h), "DELIVERY")
    output_path = output_dir / "krea2_highres.png"
    save_png(
        output_path,
        delivery_image,
        parameters=parameters,
        quality_report=quality_report if args.quality_report else None,
    )
    emit(f"IMAGE={output_path}")

    if args.quality_report:
        quality_path = output_dir / "quality_report.json"
        quality_path.write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit(f"QUALITY_REPORT={quality_path}")
    else:
        quality_path = None

    run_manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "model_profile": args.model_profile,
        "seed": args.seed,
        "upscale_mode_requested": args.upscale_mode,
        "upscale_mode": upscale_mode,
        "source_size": [source.width, source.height],
        "target_size": [target_w, target_h],
        "diffusion_size": [diffusion_w, diffusion_h],
        "diffusion_cap": args.diffusion_long_edge_cap,
        "steps": args.steps,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "first_pass_denoise": args.first_pass_denoise,
        "final_denoise": args.denoise,
        "tile": {
            "width": args.tile_width,
            "height": args.tile_height,
            "overlap": args.tile_overlap,
            "batch_size": args.tile_batch_size,
            "method": args.method,
        },
        "always_tiled_vae": bool(args.always_tiled_vae),
        "diffusion_validation": diffusion_stats,
        "stage1_validation": stage1_stats,
        "smart_finish": quality_report,
        "elapsed_seconds": round(time.time() - run_started_at, 3),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit(f"RUN_MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
