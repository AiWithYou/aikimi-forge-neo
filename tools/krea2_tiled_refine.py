import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_CFG,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_DEFAULT_SHIFT,
    KREA2_DIFFUSION_ALIGNMENT,
    KREA2_LOCAL_REFINE_STEPS,
    KREA2_STAGE1_DENOISE,
    require_positive_int,
    target_size,
)
from tools.krea2_8k_img2img import (
    decode_b64_image,
    flatten_source_image,
    image_to_b64_png,
    post_img2img,
    prompt_from_png,
    resolve_seed,
)


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output/krea2_tiled_refine"
DEFAULT_MAX_OUTPUT_PIXELS = 24_000_000
DEFAULT_MAX_TILE_PIXELS = 1_638_400
DEFAULT_COLOR_MATCH_RADIUS = 32
DEFAULT_COLOR_MATCH_MAX_SHIFT = 4.0


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def require_positive_float(value: float, name: str):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be > 0.")


def validate_args(args):
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be passed together.")

    for value, name in (
        (args.long_edge, "--long-edge"),
        (args.width, "--width"),
        (args.height, "--height"),
        (args.tile_size, "--tile-size"),
        (args.steps, "--steps"),
        (args.timeout, "--timeout"),
        (args.max_output_pixels, "--max-output-pixels"),
        (args.max_tile_pixels, "--max-tile-pixels"),
    ):
        require_positive_int(value, name)

    require_positive_float(args.scale, "--scale")
    if args.overlap < 0:
        raise ValueError("--overlap must be >= 0.")
    if args.overlap >= args.tile_size:
        raise ValueError("--overlap must be < --tile-size.")
    padded_tile_size = (
        (args.tile_size + KREA2_DIFFUSION_ALIGNMENT - 1)
        // KREA2_DIFFUSION_ALIGNMENT
        * KREA2_DIFFUSION_ALIGNMENT
    )
    if padded_tile_size * padded_tile_size > args.max_tile_pixels:
        raise ValueError("--tile-size exceeds --max-tile-pixels.")
    if not 0 <= args.denoise <= 1:
        raise ValueError("--denoise must be between 0 and 1.")
    if args.cfg < 0:
        raise ValueError("--cfg must be >= 0.")
    if args.distilled_cfg < 0:
        raise ValueError("--distilled-cfg must be >= 0.")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be >= 0.")
    if args.no_progress_timeout < 0:
        raise ValueError("--no-progress-timeout must be >= 0.")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")
    require_positive_int(args.color_match_radius, "--color-match-radius")
    if args.color_match_radius > args.tile_size:
        raise ValueError("--color-match-radius must be <= --tile-size.")
    require_positive_float(args.color_match_max_shift, "--color-match-max-shift")


def target_size_from_args(width: int, height: int, args) -> tuple[int, int]:
    if args.width is not None and args.height is not None:
        target_w, target_h = target_size(
            width, height, args.long_edge, args.width, args.height
        )
    elif args.long_edge is not None:
        target_w, target_h = target_size(width, height, args.long_edge, None, None)
    else:
        target_w = max(1, round(width * args.scale / 64) * 64)
        target_h = max(1, round(height * args.scale / 64) * 64)

    pixels = target_w * target_h
    if pixels > args.max_output_pixels:
        raise ValueError(
            f"target size {target_w}x{target_h} is {pixels:,} pixels, exceeding --max-output-pixels."
        )
    return target_w, target_h


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    require_positive_int(length, "length")
    require_positive_int(tile_size, "tile size")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= tile_size:
        raise ValueError("overlap must be < tile size.")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    end_position = length - tile_size
    if positions[-1] != end_position:
        positions.append(end_position)
    return positions


def axis_weights(
    tile_length: int, previous_overlap: int, next_overlap: int
) -> np.ndarray:
    require_positive_int(tile_length, "tile length")
    weights = np.ones(tile_length, dtype=np.float32)

    previous_overlap = min(max(0, previous_overlap), tile_length)
    next_overlap = min(max(0, next_overlap), tile_length)
    if previous_overlap > 0:
        ramp = np.linspace(0.0, 1.0, previous_overlap, endpoint=True, dtype=np.float32)
        weights[:previous_overlap] = np.minimum(
            weights[:previous_overlap],
            ramp * ramp * (3.0 - 2.0 * ramp),
        )
    if next_overlap > 0:
        ramp = np.linspace(1.0, 0.0, next_overlap, endpoint=True, dtype=np.float32)
        weights[-next_overlap:] = np.minimum(
            weights[-next_overlap:],
            ramp * ramp * (3.0 - 2.0 * ramp),
        )
    return weights


def tile_weight_mask(
    x_positions: list[int],
    y_positions: list[int],
    x_index: int,
    y_index: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    x = x_positions[x_index]
    y = y_positions[y_index]
    previous_x_overlap = (
        0 if x_index == 0 else max(0, x_positions[x_index - 1] + tile_w - x)
    )
    next_x_overlap = (
        0
        if x_index == len(x_positions) - 1
        else max(0, x + tile_w - x_positions[x_index + 1])
    )
    previous_y_overlap = (
        0 if y_index == 0 else max(0, y_positions[y_index - 1] + tile_h - y)
    )
    next_y_overlap = (
        0
        if y_index == len(y_positions) - 1
        else max(0, y + tile_h - y_positions[y_index + 1])
    )

    xw = axis_weights(tile_w, previous_x_overlap, next_x_overlap)
    yw = axis_weights(tile_h, previous_y_overlap, next_y_overlap)
    return np.outer(yw, xw).astype(np.float32)


def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to CIE Lab (D65) without external dependencies."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_to_lab_float expects an RGB uint8 image")

    srgb = rgb.astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    x = (
        linear[..., 0] * 0.4124564
        + linear[..., 1] * 0.3575761
        + linear[..., 2] * 0.1804375
    ) / 0.95047
    y = (
        linear[..., 0] * 0.2126729
        + linear[..., 1] * 0.7151522
        + linear[..., 2] * 0.0721750
    )
    z = (
        linear[..., 0] * 0.0193339
        + linear[..., 1] * 0.1191920
        + linear[..., 2] * 0.9503041
    ) / 1.08883

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def lab_curve(values: np.ndarray) -> np.ndarray:
        return np.where(
            values > epsilon,
            np.cbrt(values),
            (kappa * values + 16.0) / 116.0,
        )

    fx = lab_curve(x)
    fy = lab_curve(y)
    fz = lab_curve(z)
    return np.stack(
        (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)),
        axis=2,
    ).astype(np.float32)


def lab_float_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert a CIE Lab (D65) float image to RGB uint8."""
    lab = np.asarray(lab, dtype=np.float32)
    if lab.ndim != 3 or lab.shape[2] != 3 or not np.all(np.isfinite(lab)):
        raise ValueError("lab_float_to_rgb expects a finite HxWx3 Lab image")

    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def inverse_lab_curve(values: np.ndarray) -> np.ndarray:
        cubes = values**3
        return np.where(cubes > epsilon, cubes, (116.0 * values - 16.0) / kappa)

    x = inverse_lab_curve(fx) * 0.95047
    y = inverse_lab_curve(fy)
    z = inverse_lab_curve(fz) * 1.08883
    linear = np.stack(
        (
            x * 3.2404542 + y * -1.5371385 + z * -0.4985314,
            x * -0.9692660 + y * 1.8760108 + z * 0.0415560,
            x * 0.0556434 + y * -0.2040259 + z * 1.0572252,
        ),
        axis=2,
    )
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def _moving_average(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        raise ValueError("low-frequency radius must be > 0")
    window = radius * 2 + 1
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate(
        (np.zeros(zero_shape, dtype=np.float64), cumulative), axis=axis
    )
    high = [slice(None)] * values.ndim
    low = [slice(None)] * values.ndim
    high[axis] = slice(window, None)
    low[axis] = slice(None, -window)
    return ((cumulative[tuple(high)] - cumulative[tuple(low)]) / window).astype(
        np.float32
    )


def low_frequency_box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    """Return a dependency-free low-pass view with edge-replicated boundaries."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim not in (2, 3) or not np.all(np.isfinite(values)):
        raise ValueError("low_frequency_box_blur expects a finite 2D/3D array")
    return _moving_average(_moving_average(values, radius, 1), radius, 0)


def match_low_frequency_lab_chroma(
    refined_lab: np.ndarray,
    base_lab: np.ndarray,
    *,
    radius: int = DEFAULT_COLOR_MATCH_RADIUS,
    max_chroma_shift: float = DEFAULT_COLOR_MATCH_MAX_SHIFT,
) -> np.ndarray:
    """Match only low-frequency Lab a/b while preserving the refined L channel."""
    refined_lab = np.asarray(refined_lab, dtype=np.float32)
    base_lab = np.asarray(base_lab, dtype=np.float32)
    if (
        refined_lab.shape != base_lab.shape
        or refined_lab.ndim != 3
        or refined_lab.shape[2] != 3
    ):
        raise ValueError("refined and base Lab images must have the same HxWx3 shape")
    if not np.all(np.isfinite(refined_lab)) or not np.all(np.isfinite(base_lab)):
        raise ValueError("refined and base Lab images must be finite")
    require_positive_int(radius, "low-frequency radius")
    require_positive_float(max_chroma_shift, "maximum chroma shift")

    refined_low_ab = low_frequency_box_blur(refined_lab[..., 1:3], radius)
    base_low_ab = low_frequency_box_blur(base_lab[..., 1:3], radius)
    correction = base_low_ab - refined_low_ab
    correction_norm = np.linalg.norm(correction, axis=2)
    limiter = np.minimum(
        1.0, float(max_chroma_shift) / np.maximum(correction_norm, 1e-6)
    )
    correction *= limiter[..., None]

    matched = refined_lab.copy()
    matched[..., 1:3] += correction
    return matched


def match_low_frequency_chroma(
    refined_rgb: np.ndarray,
    base_rgb: np.ndarray,
    *,
    radius: int = DEFAULT_COLOR_MATCH_RADIUS,
    max_chroma_shift: float = DEFAULT_COLOR_MATCH_MAX_SHIFT,
) -> np.ndarray:
    """Conservatively align a refined tile's low-frequency color to its base crop."""
    refined_lab = rgb_to_lab_float(refined_rgb)
    base_lab = rgb_to_lab_float(base_rgb)
    matched_lab = match_low_frequency_lab_chroma(
        refined_lab,
        base_lab,
        radius=radius,
        max_chroma_shift=max_chroma_shift,
    )
    return lab_float_to_rgb(matched_lab)


def pad_tile_for_diffusion(
    tile: Image.Image, alignment: int = KREA2_DIFFUSION_ALIGNMENT
) -> Image.Image:
    require_positive_int(alignment, "diffusion alignment")
    padded_width = max(
        alignment, ((tile.width + alignment - 1) // alignment) * alignment
    )
    padded_height = max(
        alignment, ((tile.height + alignment - 1) // alignment) * alignment
    )
    if (padded_width, padded_height) == tile.size:
        return tile.copy()

    rgb = np.asarray(tile.convert("RGB"), dtype=np.uint8)
    padded = np.pad(
        rgb,
        ((0, padded_height - tile.height), (0, padded_width - tile.width), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded, mode="RGB")


def build_tile_payload(
    args, tile: Image.Image, prompt: str, negative_prompt: str
) -> dict:
    diffusion_tile = pad_tile_for_diffusion(tile)
    return {
        "init_images": [image_to_b64_png(diffusion_tile)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": diffusion_tile.width,
        "height": diffusion_tile.height,
        "resize_mode": 0,
        "denoising_strength": args.denoise,
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": True,
        "save_images": False,
        "include_init_images": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Krea2 tiled img2img refine with feather blending."
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
        "--scale",
        type=float,
        default=1.5,
        help="Target scale when --long-edge/--width/--height are not set.",
    )
    parser.add_argument("--long-edge", type=int, default=None, help="Target long edge.")
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
        "--tile-size",
        type=int,
        default=1024,
        help="Maximum tile width/height sent to img2img.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Tile overlap used for feather blending.",
    )
    parser.add_argument("--steps", type=int, default=KREA2_LOCAL_REFINE_STEPS)
    parser.add_argument("--sampler", default=KREA2_DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=KREA2_DEFAULT_SCHEDULER)
    parser.add_argument("--cfg", type=float, default=KREA2_DEFAULT_CFG)
    parser.add_argument("--distilled-cfg", type=float, default=KREA2_DEFAULT_SHIFT)
    parser.add_argument("--denoise", type=float, default=KREA2_STAGE1_DENOISE)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--color-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match each refined tile's low-frequency Lab chroma to its base crop before blending.",
    )
    parser.add_argument(
        "--color-match-radius",
        type=int,
        default=DEFAULT_COLOR_MATCH_RADIUS,
        help="Low-frequency box-filter radius in pixels for tile color matching.",
    )
    parser.add_argument(
        "--color-match-max-shift",
        type=float,
        default=DEFAULT_COLOR_MATCH_MAX_SHIFT,
        help="Maximum Lab chroma correction applied per pixel.",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-tile HTTP timeout in seconds."
    )
    parser.add_argument("--progress-interval", type=float, default=20.0)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-output-pixels", type=int, default=DEFAULT_MAX_OUTPUT_PIXELS
    )
    parser.add_argument("--max-tile-pixels", type=int, default=DEFAULT_MAX_TILE_PIXELS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare resized base and manifest without calling Forge.",
    )
    args = parser.parse_args()
    validate_args(args)
    args.seed = resolve_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as image:
        source = flatten_source_image(image)

    target_w, target_h = target_size_from_args(source.width, source.height, args)
    base = source.resize((target_w, target_h), Image.Resampling.LANCZOS)

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = (
        args.negative_prompt if args.negative_prompt is not None else png_negative
    )
    if not prompt:
        raise ValueError(
            "Prompt is empty. Pass --prompt or use a PNG with Forge infotext."
        )

    x_positions = tile_positions(target_w, args.tile_size, args.overlap)
    y_positions = tile_positions(target_h, args.tile_size, args.overlap)
    tile_count = len(x_positions) * len(y_positions)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"krea2_tiled_refine_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    base_path = output_dir / "base_lanczos.png"
    base.save(base_path)

    manifest = {
        "input": str(input_path),
        "source_size": f"{source.width}x{source.height}",
        "target_size": f"{target_w}x{target_h}",
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "feather_curve": "smoothstep",
        "tile_count": tile_count,
        "x_positions": x_positions,
        "y_positions": y_positions,
        "steps": args.steps,
        "denoise": args.denoise,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "cfg": args.cfg,
        "distilled_cfg": args.distilled_cfg,
        "seed": args.seed,
        "color_match": args.color_match,
        "color_match_radius": args.color_match_radius,
        "color_match_max_shift": args.color_match_max_shift,
        "base_lanczos": str(base_path),
    }
    manifest_path = output_dir / "tile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"SOURCE={source.width}x{source.height}")
    emit(f"TARGET={target_w}x{target_h} ({target_w * target_h / 1_000_000:.1f} MP)")
    emit(f"TILE_SIZE={args.tile_size}")
    emit(f"OVERLAP={args.overlap}")
    emit("FEATHER_CURVE=smoothstep")
    emit(f"TILE_COUNT={tile_count}")
    emit(f"SEED={args.seed}")
    emit(
        f"COLOR_MATCH={int(args.color_match)} RADIUS={args.color_match_radius} "
        f"MAX_SHIFT={args.color_match_max_shift}"
    )
    emit(f"BASE_LANCZOS={base_path}")
    emit(f"MANIFEST={manifest_path}")

    if args.dry_run:
        emit("DRY_RUN=1")
        return 0

    accumulator = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_sum = np.zeros((target_h, target_w), dtype=np.float32)

    tile_index = 0
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            tile_index += 1
            tile = base.crop(
                (
                    x,
                    y,
                    min(x + args.tile_size, target_w),
                    min(y + args.tile_size, target_h),
                )
            )
            emit(
                f"TILE={tile_index}/{tile_count} X={x} Y={y} SIZE={tile.width}x{tile.height}"
            )
            payload = build_tile_payload(args, tile, prompt, negative_prompt)
            preview_payload = dict(payload)
            preview_payload["init_images"] = [
                f"<base64 PNG omitted: {payload['width']}x{payload['height']}; "
                f"crop {tile.width}x{tile.height}>"
            ]
            (output_dir / f"tile_{tile_index:04d}_request_preview.json").write_text(
                json.dumps(preview_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            data = post_img2img(args, payload, f"TILE_{tile_index:04d}")
            images = data.get("images") or []
            if not images:
                raise RuntimeError(f"Tile {tile_index} returned no image.")

            refined_tile = decode_b64_image(images[0])
            expected_diffusion_size = (payload["width"], payload["height"])
            if refined_tile.size != expected_diffusion_size:
                raise RuntimeError(
                    f"Tile {tile_index} returned {refined_tile.size}, "
                    f"expected {expected_diffusion_size}."
                )
            if refined_tile.size != tile.size:
                refined_tile = refined_tile.crop((0, 0, tile.width, tile.height))
            tile_array = np.asarray(refined_tile, dtype=np.uint8)
            if args.color_match:
                tile_array = match_low_frequency_chroma(
                    tile_array,
                    np.asarray(tile, dtype=np.uint8),
                    radius=args.color_match_radius,
                    max_chroma_shift=args.color_match_max_shift,
                )
                refined_tile = Image.fromarray(tile_array, mode="RGB")
            tile_path = output_dir / f"tile_{tile_index:04d}.png"
            refined_tile.save(tile_path)

            tile_array = tile_array.astype(np.float32)
            mask = tile_weight_mask(
                x_positions,
                y_positions,
                x_index,
                y_index,
                refined_tile.width,
                refined_tile.height,
            )
            accumulator[y : y + refined_tile.height, x : x + refined_tile.width, :] += (
                tile_array * mask[:, :, None]
            )
            weight_sum[y : y + refined_tile.height, x : x + refined_tile.width] += mask

    if np.any(weight_sum <= 0):
        raise RuntimeError("Tile blending left uncovered pixels.")

    result = np.clip(accumulator / weight_sum[:, :, None], 0, 255).astype(np.uint8)
    result_image = Image.fromarray(result, mode="RGB")
    output_path = output_dir / "krea2_tiled_refine.png"
    result_image.save(output_path)
    emit(f"IMAGE={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
