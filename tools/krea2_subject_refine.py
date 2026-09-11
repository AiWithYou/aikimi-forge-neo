import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFilter

from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_CFG,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_DEFAULT_SHIFT,
    KREA2_LOCAL_REFINE_STEPS,
    KREA2_STAGE1_DENOISE,
    require_positive_int,
    size_from_long_edge,
)
from tools.krea2_8k_img2img import (
    decode_b64_image,
    image_to_b64_png,
    post_img2img,
    prompt_from_png,
)


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = "output/krea2_subject_refine"
DEFAULT_PROCESS_LONG_EDGE = 1536
DEFAULT_MAX_PROCESS_PIXELS = DEFAULT_PROCESS_LONG_EDGE * DEFAULT_PROCESS_LONG_EDGE


@dataclass(frozen=True)
class RegionBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def as_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def require_non_negative_int(value: int, name: str):
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def require_non_negative_float(value: float, name: str):
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def parse_four_values(value: str, name: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(f"{name} must be four comma-separated values.")
    return parts


def validate_box(box: RegionBox, image_width: int, image_height: int, name: str):
    require_positive_int(image_width, "image width")
    require_positive_int(image_height, "image height")
    if box.left < 0 or box.top < 0:
        raise ValueError(f"{name} must not start outside the image.")
    if box.right > image_width or box.bottom > image_height:
        raise ValueError(f"{name} must fit inside the image.")
    if box.left >= box.right or box.top >= box.bottom:
        raise ValueError(f"{name} must have left < right and top < bottom.")


def parse_pixel_box(value: str, image_width: int, image_height: int) -> RegionBox:
    parts = parse_four_values(value, "--box")
    try:
        left, top, right, bottom = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--box values must be integers.") from exc
    box = RegionBox(left, top, right, bottom)
    validate_box(box, image_width, image_height, "--box")
    return box


def parse_normalized_box(value: str, image_width: int, image_height: int) -> RegionBox:
    parts = parse_four_values(value, "--box-normalized")
    try:
        left, top, right, bottom = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--box-normalized values must be numbers.") from exc
    for number in (left, top, right, bottom):
        if number < 0 or number > 1:
            raise ValueError("--box-normalized values must be between 0 and 1.")
    box = RegionBox(
        round(left * image_width),
        round(top * image_height),
        round(right * image_width),
        round(bottom * image_height),
    )
    validate_box(box, image_width, image_height, "--box-normalized")
    return box


def expand_box(
    box: RegionBox, padding: int, image_width: int, image_height: int
) -> RegionBox:
    require_non_negative_int(padding, "--padding")
    expanded = RegionBox(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(image_width, box.right + padding),
        min(image_height, box.bottom + padding),
    )
    validate_box(expanded, image_width, image_height, "expanded box")
    return expanded


def resolve_boxes(args, image_width: int, image_height: int) -> list[RegionBox]:
    if args.box and args.box_normalized:
        raise ValueError("Use either --box or --box-normalized, not both.")
    if args.box:
        return [parse_pixel_box(value, image_width, image_height) for value in args.box]
    if args.box_normalized:
        return [
            parse_normalized_box(value, image_width, image_height)
            for value in args.box_normalized
        ]
    raise ValueError("Pass at least one --box or --box-normalized region.")


def process_size_for_crop(
    crop_width: int, crop_height: int, process_long_edge: int, max_process_pixels: int
) -> tuple[int, int]:
    require_positive_int(crop_width, "crop width")
    require_positive_int(crop_height, "crop height")
    require_positive_int(process_long_edge, "--process-long-edge")
    require_positive_int(max_process_pixels, "--max-process-pixels")
    process_width, process_height = size_from_long_edge(
        crop_width, crop_height, process_long_edge
    )
    process_pixels = process_width * process_height
    if process_pixels > max_process_pixels:
        raise ValueError(
            f"process size {process_width}x{process_height} is {process_pixels:,} pixels, exceeding --max-process-pixels."
        )
    return process_width, process_height


def build_feather_mask(
    size: tuple[int, int], feather: int, mask_shape: str
) -> Image.Image:
    width, height = size
    require_positive_int(width, "mask width")
    require_positive_int(height, "mask height")
    require_non_negative_int(feather, "--feather")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    if mask_shape == "rectangle":
        inset = min(feather, max(0, min(width, height) // 2 - 1))
        if inset <= 0:
            draw.rectangle((0, 0, width, height), fill=255)
        else:
            draw.rectangle(
                (inset, inset, width - inset - 1, height - inset - 1), fill=255
            )
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))
    elif mask_shape == "ellipse":
        draw.ellipse((0, 0, width - 1, height - 1), fill=255)
        if feather > 0:
            edge = Image.new("L", size, 0)
            edge_draw = ImageDraw.Draw(edge)
            inset = min(feather, max(0, min(width, height) // 2 - 1))
            edge_draw.ellipse(
                (inset, inset, width - inset - 1, height - inset - 1), fill=255
            )
            mask = edge.filter(ImageFilter.GaussianBlur(radius=feather / 2))
    else:
        raise ValueError("--mask-shape must be rectangle or ellipse.")
    return mask


def build_subject_payload(
    args, process_image: Image.Image, prompt: str, negative_prompt: str
) -> dict:
    return {
        "init_images": [image_to_b64_png(process_image)],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": process_image.width,
        "height": process_image.height,
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


def write_payload_preview(
    output_dir: Path,
    name: str,
    payload: dict,
    init_image_description: str,
    extra: dict,
) -> Path:
    preview = dict(payload)
    preview["init_images"] = [init_image_description]
    preview.update(extra)
    preview_path = output_dir / name
    preview_path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return preview_path


def validate_args(args):
    if args.box and args.box_normalized:
        raise ValueError("Use either --box or --box-normalized, not both.")
    if not args.box and not args.box_normalized:
        raise ValueError("Pass at least one --box or --box-normalized region.")
    for value, name in (
        (args.process_long_edge, "--process-long-edge"),
        (args.max_process_pixels, "--max-process-pixels"),
        (args.steps, "--steps"),
        (args.timeout, "--timeout"),
    ):
        require_positive_int(value, name)
    for value, name in (
        (args.padding, "--padding"),
        (args.feather, "--feather"),
    ):
        require_non_negative_int(value, name)
    if not 0 <= args.denoise <= 1:
        raise ValueError("--denoise must be between 0 and 1.")
    require_non_negative_float(args.cfg, "--cfg")
    require_non_negative_float(args.distilled_cfg, "--distilled-cfg")
    require_non_negative_float(args.progress_interval, "--progress-interval")
    require_non_negative_float(args.no_progress_timeout, "--no-progress-timeout")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Krea2 img2img refine for selected face/body regions."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--api", default=DEFAULT_API, help="Forge API base URL.")
    parser.add_argument(
        "--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory."
    )
    parser.add_argument(
        "--box",
        action="append",
        help="Pixel region as left,top,right,bottom. May be passed more than once.",
    )
    parser.add_argument(
        "--box-normalized",
        action="append",
        help="Normalized region as left,top,right,bottom in 0..1. May be passed more than once.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=96,
        help="Pixels added around each selected region before img2img.",
    )
    parser.add_argument(
        "--process-long-edge",
        type=int,
        default=DEFAULT_PROCESS_LONG_EDGE,
        help="Long edge sent to img2img for each extracted region.",
    )
    parser.add_argument(
        "--max-process-pixels",
        type=int,
        default=DEFAULT_MAX_PROCESS_PIXELS,
        help="Maximum pixels sent to one img2img request.",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=96,
        help="Composite feather width in output pixels.",
    )
    parser.add_argument(
        "--mask-shape",
        choices=["rectangle", "ellipse"],
        default="rectangle",
        help="Mask shape used when compositing the refined crop.",
    )
    parser.add_argument(
        "--prompt", default=None, help="Prompt. Defaults to PNG infotext prompt."
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Defaults to PNG infotext negative prompt.",
    )
    parser.add_argument("--steps", type=int, default=KREA2_LOCAL_REFINE_STEPS)
    parser.add_argument("--sampler", default=KREA2_DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=KREA2_DEFAULT_SCHEDULER)
    parser.add_argument("--cfg", type=float, default=KREA2_DEFAULT_CFG)
    parser.add_argument("--distilled-cfg", type=float, default=KREA2_DEFAULT_SHIFT)
    parser.add_argument("--denoise", type=float, default=KREA2_STAGE1_DENOISE)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-region HTTP timeout in seconds."
    )
    parser.add_argument("--progress-interval", type=float, default=20.0)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare crop images and request previews without calling Forge.",
    )
    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as image:
        base = image.convert("RGB")

    boxes = resolve_boxes(args, base.width, base.height)

    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    negative_prompt = (
        args.negative_prompt if args.negative_prompt is not None else png_negative
    )
    if not prompt:
        raise ValueError(
            "Prompt is empty. Pass --prompt or use a PNG with Forge infotext."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"krea2_subject_refine_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = base.copy()
    manifest = {
        "input": str(input_path),
        "source_size": f"{base.width}x{base.height}",
        "padding": args.padding,
        "process_long_edge": args.process_long_edge,
        "max_process_pixels": args.max_process_pixels,
        "feather": args.feather,
        "mask_shape": args.mask_shape,
        "steps": args.steps,
        "denoise": args.denoise,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "cfg": args.cfg,
        "distilled_cfg": args.distilled_cfg,
        "regions": [],
    }

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"SOURCE={base.width}x{base.height}")
    emit(f"REGION_COUNT={len(boxes)}")

    for index, box in enumerate(boxes, start=1):
        expanded = expand_box(box, args.padding, base.width, base.height)
        source_crop = result.crop(expanded.as_tuple())
        process_w, process_h = process_size_for_crop(
            source_crop.width,
            source_crop.height,
            args.process_long_edge,
            args.max_process_pixels,
        )
        process_input = source_crop.resize(
            (process_w, process_h), Image.Resampling.LANCZOS
        )
        region_prefix = f"region_{index:03d}"
        source_crop_path = output_dir / f"{region_prefix}_source_crop.png"
        process_input_path = output_dir / f"{region_prefix}_process_input.png"
        source_crop.save(source_crop_path)
        process_input.save(process_input_path)

        payload = build_subject_payload(args, process_input, prompt, negative_prompt)
        request_preview_path = write_payload_preview(
            output_dir,
            f"{region_prefix}_request_preview.json",
            payload,
            f"<base64 PNG omitted: {process_w}x{process_h}>",
            {
                "input_path": str(input_path),
                "box": box.as_dict(),
                "expanded_box": expanded.as_dict(),
                "source_crop": str(source_crop_path),
                "process_input": str(process_input_path),
            },
        )

        region_manifest = {
            "box": box.as_dict(),
            "expanded_box": expanded.as_dict(),
            "source_crop": str(source_crop_path),
            "process_input": str(process_input_path),
            "process_size": f"{process_w}x{process_h}",
            "request_preview": str(request_preview_path),
        }
        manifest["regions"].append(region_manifest)

        emit(
            f"REGION={index}/{len(boxes)} BOX={box.left},{box.top},{box.right},{box.bottom} "
            f"EXPANDED={expanded.left},{expanded.top},{expanded.right},{expanded.bottom} "
            f"PROCESS={process_w}x{process_h}"
        )
        emit(f"{region_prefix.upper()}_SOURCE_CROP={source_crop_path}")
        emit(f"{region_prefix.upper()}_PROCESS_INPUT={process_input_path}")
        emit(f"{region_prefix.upper()}_REQUEST_PREVIEW={request_preview_path}")

        if args.dry_run:
            continue

        data = post_img2img(args, payload, f"REGION_{index:03d}")
        images = data.get("images") or []
        if not images:
            raise RuntimeError(f"Region {index} returned no image.")
        refined_process = decode_b64_image(images[0])
        if refined_process.size != process_input.size:
            raise RuntimeError(
                f"Region {index} returned {refined_process.size}, expected {process_input.size}."
            )
        refined_process_path = output_dir / f"{region_prefix}_refined_process.png"
        refined_crop_path = output_dir / f"{region_prefix}_refined_crop.png"
        mask_path = output_dir / f"{region_prefix}_mask.png"
        refined_process.save(refined_process_path)
        refined_crop = refined_process.resize(
            source_crop.size, Image.Resampling.LANCZOS
        )
        refined_crop.save(refined_crop_path)
        mask = build_feather_mask(source_crop.size, args.feather, args.mask_shape)
        mask.save(mask_path)
        result.paste(refined_crop, (expanded.left, expanded.top), mask)
        region_manifest.update(
            {
                "refined_process": str(refined_process_path),
                "refined_crop": str(refined_crop_path),
                "mask": str(mask_path),
            }
        )
        emit(f"{region_prefix.upper()}_REFINED_PROCESS={refined_process_path}")
        emit(f"{region_prefix.upper()}_REFINED_CROP={refined_crop_path}")
        emit(f"{region_prefix.upper()}_MASK={mask_path}")

    manifest_path = output_dir / "subject_refine_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit(f"MANIFEST={manifest_path}")

    if args.dry_run:
        emit("DRY_RUN=1")
        return 0

    output_path = output_dir / "krea2_subject_refine.png"
    result.save(output_path)
    emit(f"IMAGE={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
