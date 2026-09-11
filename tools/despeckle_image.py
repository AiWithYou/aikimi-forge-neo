import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_API = "http://127.0.0.1:7861"


def emit(message: str):
    sys.stdout.write(f"{message}\n")


def require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python is required for despeckle detection and local inpaint.")


def ensure_odd_kernel(value: int, name: str) -> int:
    if value < 3:
        raise ValueError(f"{name} must be >= 3.")
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd.")
    return value


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


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def save_image(path: Path, image: Image.Image, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def median_blur_rgb(rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    require_cv2()
    return cv2.medianBlur(rgb, kernel_size)


def luma(rgb: np.ndarray) -> np.ndarray:
    data = rgb.astype(np.float32)
    return (data[:, :, 0] * 0.299 + data[:, :, 1] * 0.587 + data[:, :, 2] * 0.114).astype(np.int16)


def component_filter(mask: np.ndarray, min_area: int, max_area: int, max_span: int) -> tuple[np.ndarray, dict]:
    require_cv2()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)

    kept = 0
    rejected_area = 0
    rejected_span = 0
    candidate_components = max(0, num_labels - 1)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if area < min_area or area > max_area:
            rejected_area += 1
            continue

        if width > max_span or height > max_span:
            rejected_span += 1
            continue

        filtered[labels == label] = 255
        kept += 1

    stats_out = {
        "candidate_components": candidate_components,
        "kept_components": kept,
        "rejected_by_area": rejected_area,
        "rejected_by_span": rejected_span,
    }
    return filtered, stats_out


def build_speckle_mask(
    image: Image.Image,
    *,
    threshold: int,
    median_size: int,
    polarity: str,
    min_area: int,
    max_area: int,
    max_span: int,
    dilate: int,
) -> tuple[Image.Image, dict]:
    require_cv2()
    median_size = ensure_odd_kernel(median_size, "median-size")

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    median = median_blur_rgb(rgb, median_size)

    rgb_diff = np.max(np.abs(rgb.astype(np.int16) - median.astype(np.int16)), axis=2)
    luma_delta = luma(rgb) - luma(median)

    candidate = rgb_diff >= threshold
    luma_floor = max(1, int(round(threshold * 0.35)))

    if polarity == "bright":
        candidate &= luma_delta >= luma_floor
    elif polarity == "dark":
        candidate &= luma_delta <= -luma_floor
    elif polarity == "both":
        candidate &= np.abs(luma_delta) >= luma_floor
    elif polarity != "all":
        raise ValueError(f"Unsupported polarity: {polarity}")

    raw_mask = candidate.astype(np.uint8) * 255
    filtered, component_stats = component_filter(raw_mask, min_area, max_area, max_span)

    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filtered = cv2.dilate(filtered, kernel, iterations=1)

    masked_pixels = int(np.count_nonzero(filtered))
    stats = {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "threshold": int(threshold),
        "median_size": int(median_size),
        "polarity": polarity,
        "min_area": int(min_area),
        "max_area": int(max_area),
        "max_span": int(max_span),
        "dilate": int(dilate),
        "raw_candidate_pixels": int(np.count_nonzero(raw_mask)),
        "masked_pixels": masked_pixels,
        "masked_percent": round(masked_pixels * 100.0 / (rgb.shape[0] * rgb.shape[1]), 6),
    }
    stats.update(component_stats)

    return Image.fromarray(filtered, mode="L"), stats


def local_inpaint(image: Image.Image, mask: Image.Image, radius: float, method: str) -> Image.Image:
    require_cv2()
    flags = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
    }[method]
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    mask_array = np.array(mask.convert("L"), dtype=np.uint8)
    repaired = cv2.inpaint(rgb, mask_array, radius, flags)
    return Image.fromarray(repaired, mode="RGB")


def median_fill(image: Image.Image, mask: Image.Image, median_size: int) -> Image.Image:
    median_size = ensure_odd_kernel(median_size, "fill-median-size")
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    median = median_blur_rgb(rgb, median_size)
    mask_array = np.array(mask.convert("L"), dtype=np.uint8) > 0
    repaired = rgb.copy()
    repaired[mask_array] = median[mask_array]
    return Image.fromarray(repaired, mode="RGB")


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    mask_array = np.array(mask.convert("L"), dtype=np.uint8) > 0
    overlay = rgb.copy()
    overlay[mask_array] = (overlay[mask_array].astype(np.float32) * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


def forge_inpaint(image: Image.Image, mask: Image.Image, input_path: Path, args) -> Image.Image:
    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else png_negative

    payload = {
        "init_images": [image_to_b64_png(image.convert("RGB"))],
        "mask": image_to_b64_png(mask.convert("L")),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": image.width,
        "height": image.height,
        "resize_mode": 0,
        "denoising_strength": args.denoise,
        "mask_blur": args.mask_blur,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": args.inpaint_padding,
        "inpainting_mask_invert": 0,
        "inpainting_fill": args.inpainting_fill,
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": True,
        "save_images": False,
        "include_init_images": False,
    }

    response = requests.post(f"{args.api}/sdapi/v1/img2img", json=payload, timeout=args.timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Forge img2img failed: HTTP {response.status_code}\n{response.text[:4000]}")

    data = response.json()
    images = data.get("images") or []
    if not images:
        raise RuntimeError("Forge img2img returned no image.")
    return decode_b64_image(images[0])


def write_report(path: Path, stats: dict, output_path: Path | None, mask_path: Path | None, preview_path: Path | None, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(stats)
    report["output"] = str(output_path) if output_path else None
    report["mask"] = str(mask_path) if mask_path else None
    report["preview"] = str(preview_path) if preview_path else None
    report["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and repair small speckle noise in generated images.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to '<input>_despeckled.<ext>'.")
    parser.add_argument("--mode", choices=["local-inpaint", "median", "forge-inpaint", "mask"], default="local-inpaint")
    parser.add_argument("--threshold", type=int, default=30, help="Local residual threshold, 0-255.")
    parser.add_argument("--median-size", type=int, default=5, help="Odd kernel size used for speckle detection.")
    parser.add_argument("--polarity", choices=["bright", "dark", "both", "all"], default="bright")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=48)
    parser.add_argument("--max-span", type=int, default=14)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--fill-median-size", type=int, default=7)
    parser.add_argument("--mask-out", default=None, help="Optional mask image output path.")
    parser.add_argument("--preview-out", default=None, help="Optional red overlay preview output path.")
    parser.add_argument("--report-out", default=None, help="Optional JSON report output path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files.")

    parser.add_argument("--api", default=DEFAULT_API, help="Forge API base URL for --mode forge-inpaint.")
    parser.add_argument("--prompt", default=None, help="Prompt for --mode forge-inpaint. Defaults to PNG infotext prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt for --mode forge-inpaint.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sampler", default="DPM++ SDE")
    parser.add_argument("--scheduler", default="Simple")
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--distilled-cfg", type=float, default=1.15)
    parser.add_argument("--denoise", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--mask-blur", type=int, default=4)
    parser.add_argument("--inpaint-padding", type=int, default=32)
    parser.add_argument("--inpainting-fill", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as source:
        image = source.convert("RGB")
    mask, stats = build_speckle_mask(
        image,
        threshold=args.threshold,
        median_size=args.median_size,
        polarity=args.polarity,
        min_area=args.min_area,
        max_area=args.max_area,
        max_span=args.max_span,
        dilate=args.dilate,
    )

    mask_path = Path(args.mask_out) if args.mask_out else None
    preview_path = Path(args.preview_out) if args.preview_out else None
    output_path = None if args.mode == "mask" else Path(args.output) if args.output else default_output_path(input_path, "_despeckled")

    if mask_path is not None:
        save_image(mask_path, mask, args.overwrite)

    if preview_path is not None:
        save_image(preview_path, overlay_mask(image, mask), args.overwrite)

    if args.mode == "local-inpaint":
        output = local_inpaint(image, mask, args.inpaint_radius, args.inpaint_method)
        save_image(output_path, output, args.overwrite)
    elif args.mode == "median":
        output = median_fill(image, mask, args.fill_median_size)
        save_image(output_path, output, args.overwrite)
    elif args.mode == "forge-inpaint":
        output = forge_inpaint(image, mask, input_path, args)
        save_image(output_path, output, args.overwrite)
    elif args.mode != "mask":
        raise ValueError(f"Unsupported mode: {args.mode}")

    report_path = Path(args.report_out) if args.report_out else None
    if report_path is not None:
        write_report(report_path, stats, output_path, mask_path, preview_path, args.overwrite)

    emit(f"INPUT={input_path}")
    emit(f"MODE={args.mode}")
    emit(f"MASKED_PIXELS={stats['masked_pixels']} ({stats['masked_percent']}%)")
    emit(f"KEPT_COMPONENTS={stats['kept_components']} / {stats['candidate_components']}")
    if output_path is not None:
        emit(f"OUTPUT={output_path}")
    if mask_path is not None:
        emit(f"MASK={mask_path}")
    if preview_path is not None:
        emit(f"PREVIEW={preview_path}")
    if report_path is not None:
        emit(f"REPORT={report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
