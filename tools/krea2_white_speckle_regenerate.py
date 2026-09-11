import argparse
import json
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, PngImagePlugin

try:
    import cv2
except ImportError:
    cv2 = None

from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_CFG,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_DEFAULT_SHIFT,
    KREA2_LOCAL_REFINE_STEPS,
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
DEFAULT_OUTPUT_ROOT = "output/krea2_white_speckle_regenerate"
DEFAULT_REPAIR_PROMPT = "seamless continuation of the surrounding image, matching the local color, " "lighting, texture, linework, and material"
TARGETED_NEGATIVE_PROMPT = "white speck, white dot, bright pixel artifact, isolated white blemish, " "sparkle artifact, watermark"


@dataclass(frozen=True)
class Box:
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

    def as_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class SpeckleComponent:
    index: int
    box: Box
    area: int
    centroid_x: float
    centroid_y: float
    peak_luma: float
    mean_local_contrast: float
    mean_chroma: float
    fill_ratio: float
    points: tuple[tuple[int, int], ...] = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "box": self.box.as_dict(),
            "area": self.area,
            "centroid": [self.centroid_x, self.centroid_y],
            "peak_luma": self.peak_luma,
            "mean_local_contrast": self.mean_local_contrast,
            "mean_chroma": self.mean_chroma,
            "fill_ratio": self.fill_ratio,
        }


@dataclass(frozen=True)
class RegionPlan:
    index: int
    component: SpeckleComponent
    components: tuple[SpeckleComponent, ...]
    mask_box: Box
    roi_box: Box
    micro_mask: Image.Image = field(repr=False, compare=False)

    @property
    def masked_pixels(self) -> int:
        return int(np.count_nonzero(np.asarray(self.micro_mask, dtype=np.uint8)))

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "component_indices": [item.index for item in self.components],
            "components": [item.as_dict() for item in self.components],
            "mask_box": self.mask_box.as_dict(),
            "roi_box": self.roi_box.as_dict(),
            "masked_pixels": self.masked_pixels,
        }


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    manifest_path: Path
    output_path: Path | None
    region_count: int


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python is required for white-speck detection and mask building.")


def ensure_odd_kernel(value: int, name: str) -> int:
    if value < 3:
        raise ValueError(f"{name} must be >= 3.")
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd.")
    return value


def require_non_negative_int(value: int, name: str):
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def require_non_negative_float(value: float, name: str):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite value >= 0.")


def validate_box(box: Box, image_width: int, image_height: int, name: str):
    if box.left < 0 or box.top < 0:
        raise ValueError(f"{name} must not start outside the image.")
    if box.right > image_width or box.bottom > image_height:
        raise ValueError(f"{name} must fit inside the image.")
    if box.left >= box.right or box.top >= box.bottom:
        raise ValueError(f"{name} must have left < right and top < bottom.")


def parse_box_argument(value: str) -> Box:
    try:
        coordinates = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("box must contain four comma-separated integers.") from exc
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("box must be LEFT,TOP,RIGHT,BOTTOM.")
    box = Box(*coordinates)
    if box.left < 0 or box.top < 0:
        raise argparse.ArgumentTypeError("box LEFT and TOP must be non-negative.")
    if box.left >= box.right or box.top >= box.bottom:
        raise argparse.ArgumentTypeError("box must have LEFT < RIGHT and TOP < BOTTOM.")
    return box


def build_protection_mask(
    image_size: tuple[int, int],
    source_mask: Image.Image | None,
    exclude_boxes: list[Box] | tuple[Box, ...],
) -> Image.Image | None:
    if source_mask is None and not exclude_boxes:
        return None
    image_width, image_height = image_size
    protection = Image.new("L", image_size, 0)
    if source_mask is not None:
        if source_mask.size != image_size:
            raise ValueError("protection mask size " f"{source_mask.width}x{source_mask.height} does not match " f"input size {image_width}x{image_height}.")
        source = np.asarray(source_mask.convert("L"), dtype=np.uint8)
        binary = np.where(source >= 128, 255, 0).astype(np.uint8)
        protection = Image.fromarray(binary, mode="L")
    if exclude_boxes:
        draw = ImageDraw.Draw(protection)
        for index, box in enumerate(exclude_boxes, start=1):
            validate_box(
                box,
                image_width,
                image_height,
                f"--exclude-box #{index}",
            )
            draw.rectangle(
                (box.left, box.top, box.right - 1, box.bottom - 1),
                fill=255,
            )
    return protection


def expand_box(box: Box, padding: int, image_width: int, image_height: int) -> Box:
    require_non_negative_int(padding, "padding")
    expanded = Box(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(image_width, box.right + padding),
        min(image_height, box.bottom + padding),
    )
    validate_box(expanded, image_width, image_height, "expanded box")
    return expanded


def union_box(boxes: list[Box] | tuple[Box, ...]) -> Box:
    if not boxes:
        raise ValueError("at least one box is required.")
    return Box(
        min(box.left for box in boxes),
        min(box.top for box in boxes),
        max(box.right for box in boxes),
        max(box.bottom for box in boxes),
    )


def box_distance(first: Box, second: Box) -> int:
    horizontal = max(0, first.left - second.right, second.left - first.right)
    vertical = max(0, first.top - second.bottom, second.top - first.bottom)
    return max(horizontal, vertical)


def fit_minimum_box(box: Box, minimum_edge: int, image_width: int, image_height: int) -> Box:
    require_positive_int(minimum_edge, "minimum edge")
    validate_box(box, image_width, image_height, "box")

    target_width = min(image_width, max(box.width, minimum_edge))
    target_height = min(image_height, max(box.height, minimum_edge))
    center_x = (box.left + box.right) / 2.0
    center_y = (box.top + box.bottom) / 2.0
    left = int(round(center_x - target_width / 2.0))
    top = int(round(center_y - target_height / 2.0))
    left = min(max(0, left), image_width - target_width)
    top = min(max(0, top), image_height - target_height)
    fitted = Box(left, top, left + target_width, top + target_height)
    validate_box(fitted, image_width, image_height, "minimum box")
    if fitted.left > box.left or fitted.top > box.top or fitted.right < box.right or fitted.bottom < box.bottom:
        raise RuntimeError("minimum box failed to contain the requested region.")
    return fitted


def _luma(rgb: np.ndarray) -> np.ndarray:
    data = rgb.astype(np.float32)
    return data[..., 0] * 0.299 + data[..., 1] * 0.587 + data[..., 2] * 0.114


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    require_cv2()
    source = np.ascontiguousarray(gray.astype(np.float32))
    gradient_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return cv2.magnitude(gradient_x, gradient_y)


def detect_white_speckles(
    image: Image.Image,
    *,
    white_luma_min: int,
    local_contrast_min: int,
    max_chroma: int,
    median_size: int,
    min_area: int,
    max_area: int,
    max_span: int,
    min_fill_ratio: float,
    max_background_gradient: float,
    max_components: int,
    max_masked_percent: float,
    protection_mask: Image.Image | None = None,
) -> tuple[Image.Image, list[SpeckleComponent], dict]:
    require_cv2()
    median_size = ensure_odd_kernel(median_size, "--median-size")
    for value, name in (
        (white_luma_min, "--white-luma-min"),
        (local_contrast_min, "--local-contrast-min"),
        (max_chroma, "--max-chroma"),
    ):
        if value < 0 or value > 255:
            raise ValueError(f"{name} must be between 0 and 255.")
    for value, name in (
        (min_area, "--min-area"),
        (max_area, "--max-area"),
        (max_span, "--max-span"),
        (max_components, "--max-components"),
    ):
        require_positive_int(value, name)
    if min_area > max_area:
        raise ValueError("--min-area must be <= --max-area.")
    if not 0 < min_fill_ratio <= 1:
        raise ValueError("--min-fill-ratio must be in (0, 1].")
    require_non_negative_float(max_background_gradient, "--max-background-gradient")
    if not np.isfinite(max_masked_percent) or max_masked_percent <= 0:
        raise ValueError("--max-masked-percent must be a finite value > 0.")

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[..., :3])
    alpha_valid = rgba[..., 3] > 8
    if protection_mask is not None and protection_mask.size != image.size:
        raise ValueError("protection mask size " f"{protection_mask.width}x{protection_mask.height} does not match " f"input size {image.width}x{image.height}.")
    protected = np.asarray(protection_mask.convert("L"), dtype=np.uint8) >= 128 if protection_mask is not None else np.zeros(alpha_valid.shape, dtype=bool)
    source_luma = _luma(rgb)
    median_rgb = cv2.medianBlur(rgb, median_size)
    median_luma = _luma(median_rgb)
    local_contrast = source_luma - median_luma
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    background_guide = cv2.GaussianBlur(median_luma.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    background_gradient = _gradient_magnitude(background_guide)

    raw_candidate = (source_luma >= white_luma_min) & (local_contrast >= local_contrast_min) & (chroma <= max_chroma) & (background_gradient <= max_background_gradient) & alpha_valid
    candidate = raw_candidate & ~protected
    candidate_mask = candidate.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

    rejected_area = 0
    rejected_span = 0
    rejected_fill = 0
    kept: list[SpeckleComponent] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            rejected_area += 1
            continue
        if width > max_span or height > max_span:
            rejected_span += 1
            continue
        fill_ratio = area / float(width * height)
        if fill_ratio < min_fill_ratio:
            rejected_fill += 1
            continue

        ys, xs = np.where(labels == label)
        points = tuple(zip(xs.tolist(), ys.tolist()))
        kept.append(
            SpeckleComponent(
                index=0,
                box=Box(left, top, left + width, top + height),
                area=area,
                centroid_x=float(centroids[label, 0]),
                centroid_y=float(centroids[label, 1]),
                peak_luma=float(np.max(source_luma[ys, xs])),
                mean_local_contrast=float(np.mean(local_contrast[ys, xs])),
                mean_chroma=float(np.mean(chroma[ys, xs])),
                fill_ratio=fill_ratio,
                points=points,
            )
        )

    kept.sort(key=lambda item: (item.box.top, item.box.left))
    components = [
        SpeckleComponent(
            index=index,
            box=item.box,
            area=item.area,
            centroid_x=item.centroid_x,
            centroid_y=item.centroid_y,
            peak_luma=item.peak_luma,
            mean_local_contrast=item.mean_local_contrast,
            mean_chroma=item.mean_chroma,
            fill_ratio=item.fill_ratio,
            points=item.points,
        )
        for index, item in enumerate(kept, start=1)
    ]
    if len(components) > max_components:
        raise ValueError(f"detected {len(components)} white-speck components, exceeding " f"--max-components {max_components}; tighten detection or explicitly " "raise the component limit for a bounded dry-run.")

    detection = np.zeros(candidate.shape, dtype=np.uint8)
    for component in components:
        for x, y in component.points:
            detection[y, x] = 255
    masked_pixels = int(np.count_nonzero(detection))
    visible_pixels = int(np.count_nonzero(alpha_valid))
    masked_percent = masked_pixels * 100.0 / visible_pixels if visible_pixels else 0.0
    if masked_percent > max_masked_percent:
        raise ValueError(f"detected mask covers {masked_percent:.6f}% of visible pixels, exceeding " f"--max-masked-percent {max_masked_percent}; inspect a dry-run mask or " "tighten detection.")

    report = {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "visible_pixels": visible_pixels,
        "white_luma_min": int(white_luma_min),
        "local_contrast_min": int(local_contrast_min),
        "max_chroma": int(max_chroma),
        "median_size": int(median_size),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "max_span": int(max_span),
        "min_fill_ratio": float(min_fill_ratio),
        "max_background_gradient": float(max_background_gradient),
        "raw_candidate_pixels": int(np.count_nonzero(raw_candidate)),
        "protected_pixels": int(np.count_nonzero(protected)),
        "protected_candidate_pixels": int(np.count_nonzero(raw_candidate & protected)),
        "unprotected_candidate_pixels": int(np.count_nonzero(candidate_mask)),
        "candidate_components": max(0, num_labels - 1),
        "kept_components": len(components),
        "rejected_by_area": rejected_area,
        "rejected_by_span": rejected_span,
        "rejected_by_fill": rejected_fill,
        "masked_pixels": masked_pixels,
        "masked_percent": masked_percent,
        "max_components": int(max_components),
        "max_masked_percent": float(max_masked_percent),
    }
    return Image.fromarray(detection, mode="L"), components, report


def build_group_micro_mask(
    components: list[SpeckleComponent] | tuple[SpeckleComponent, ...],
    *,
    image_width: int,
    image_height: int,
    radius: int,
) -> tuple[Box, Image.Image]:
    require_cv2()
    require_non_negative_int(radius, "--mask-radius")
    if not components:
        raise ValueError("at least one component is required.")
    component_box = union_box([item.box for item in components])
    mask_box = expand_box(component_box, radius, image_width, image_height)
    local = np.zeros((mask_box.height, mask_box.width), dtype=np.uint8)
    for component in components:
        for x, y in component.points:
            local[y - mask_box.top, x - mask_box.left] = 255
    if radius > 0:
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        local = cv2.dilate(local, kernel, iterations=1)
    return mask_box, Image.fromarray(local, mode="L")


def group_speckle_components(
    components: list[SpeckleComponent],
    *,
    image_width: int,
    image_height: int,
    mask_radius: int,
    roi_padding: int,
    minimum_roi_edge: int,
    merge_distance: int,
    max_group_edge: int,
) -> list[list[SpeckleComponent]]:
    require_non_negative_int(merge_distance, "--merge-distance")
    require_positive_int(max_group_edge, "--max-group-edge")
    groups: list[list[SpeckleComponent]] = []
    for component in components:
        best_index = None
        best_growth = None
        for group_index, group in enumerate(groups):
            current_box = union_box([item.box for item in group])
            if box_distance(current_box, component.box) > merge_distance:
                continue
            combined = [*group, component]
            combined_box = union_box([item.box for item in combined])
            mask_box = expand_box(combined_box, mask_radius, image_width, image_height)
            roi_box = fit_minimum_box(
                expand_box(mask_box, roi_padding, image_width, image_height),
                minimum_roi_edge,
                image_width,
                image_height,
            )
            if roi_box.width > max_group_edge or roi_box.height > max_group_edge:
                continue
            growth = combined_box.width * combined_box.height - (current_box.width * current_box.height)
            if best_growth is None or growth < best_growth:
                best_index = group_index
                best_growth = growth
        if best_index is None:
            groups.append([component])
        else:
            groups[best_index].append(component)
    groups.sort(
        key=lambda group: (
            min(item.box.top for item in group),
            min(item.box.left for item in group),
        )
    )
    return groups


def build_region_plans(
    components: list[SpeckleComponent],
    *,
    image_width: int,
    image_height: int,
    mask_radius: int,
    roi_padding: int,
    minimum_roi_edge: int,
    merge_distance: int = 0,
    max_group_edge: int = 512,
) -> list[RegionPlan]:
    require_non_negative_int(mask_radius, "--mask-radius")
    require_non_negative_int(roi_padding, "--roi-padding")
    require_positive_int(minimum_roi_edge, "--minimum-roi-edge")
    groups = group_speckle_components(
        components,
        image_width=image_width,
        image_height=image_height,
        mask_radius=mask_radius,
        roi_padding=roi_padding,
        minimum_roi_edge=minimum_roi_edge,
        merge_distance=merge_distance,
        max_group_edge=max_group_edge,
    )
    plans = []
    for index, group in enumerate(groups, start=1):
        mask_box, micro_mask = build_group_micro_mask(
            group,
            image_width=image_width,
            image_height=image_height,
            radius=mask_radius,
        )
        roi_box = expand_box(mask_box, roi_padding, image_width, image_height)
        roi_box = fit_minimum_box(roi_box, minimum_roi_edge, image_width, image_height)
        plans.append(
            RegionPlan(
                index=index,
                component=group[0],
                components=tuple(group),
                mask_box=mask_box,
                roi_box=roi_box,
                micro_mask=micro_mask,
            )
        )
    return plans


def region_mask(plan: RegionPlan) -> Image.Image:
    mask = Image.new("L", (plan.roi_box.width, plan.roi_box.height), 0)
    offset = (
        plan.mask_box.left - plan.roi_box.left,
        plan.mask_box.top - plan.roi_box.top,
    )
    mask.paste(plan.micro_mask, offset)
    return mask


def build_union_micro_mask(plans: list[RegionPlan], image_size: tuple[int, int]) -> Image.Image:
    union = Image.new("L", image_size, 0)
    for plan in plans:
        layer = Image.new("L", image_size, 0)
        layer.paste(plan.micro_mask, (plan.mask_box.left, plan.mask_box.top))
        union = ImageChops.lighter(union, layer)
    return union


def build_detection_preview(image: Image.Image, micro_mask: Image.Image, plans: list[RegionPlan]) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected = np.asarray(micro_mask.convert("L"), dtype=np.uint8) > 0
    overlay = rgb.copy()
    overlay[selected] = (overlay[selected].astype(np.float32) * 0.25 + np.array([255, 0, 0], dtype=np.float32) * 0.75).astype(np.uint8)
    preview = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(preview)
    line_width = max(1, round(max(image.size) / 1536))
    for plan in plans:
        draw.rectangle(
            (
                plan.roi_box.left,
                plan.roi_box.top,
                plan.roi_box.right - 1,
                plan.roi_box.bottom - 1,
            ),
            outline=(255, 210, 0),
            width=line_width,
        )
        draw.text(
            (plan.roi_box.left + line_width, plan.roi_box.top + line_width),
            str(plan.index),
            fill=(255, 210, 0),
        )
    return preview


def process_size_for_roi(
    roi_width: int,
    roi_height: int,
    process_long_edge: int,
    max_process_pixels: int,
) -> tuple[int, int]:
    require_positive_int(roi_width, "ROI width")
    require_positive_int(roi_height, "ROI height")
    require_positive_int(process_long_edge, "--process-long-edge")
    require_positive_int(max_process_pixels, "--max-process-pixels")
    process_width, process_height = size_from_long_edge(roi_width, roi_height, process_long_edge, alignment=64)
    if process_width * process_height > max_process_pixels:
        raise ValueError(f"process size {process_width}x{process_height} exceeds " f"--max-process-pixels {max_process_pixels:,}.")
    return process_width, process_height


def scaled_mask_blur(
    source_blur: int,
    roi_size: tuple[int, int],
    process_size: tuple[int, int],
) -> int:
    require_non_negative_int(source_blur, "--mask-blur")
    if source_blur == 0:
        return 0
    scale = max(
        process_size[0] / roi_size[0],
        process_size[1] / roi_size[1],
    )
    return min(32, max(1, int(round(source_blur * scale))))


def build_inpaint_payload(
    args,
    process_image: Image.Image,
    process_mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    mask_blur: int,
) -> dict:
    return {
        "init_images": [image_to_b64_png(process_image.convert("RGB"))],
        "mask": image_to_b64_png(process_mask.convert("L")),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "distilled_cfg_scale": args.distilled_cfg,
        "width": process_image.width,
        "height": process_image.height,
        "resize_mode": 0,
        "denoising_strength": args.denoise,
        "mask_blur": mask_blur,
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 0,
        "inpainting_mask_invert": 0,
        "inpainting_fill": args.inpainting_fill,
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


def payload_preview(payload: dict, input_description: str, mask_description: str):
    preview = dict(payload)
    preview["init_images"] = [input_description]
    preview["mask"] = mask_description
    return preview


def composite_refined_region(
    source_crop: Image.Image,
    refined_crop: Image.Image,
    mask: Image.Image,
    feather: float,
) -> Image.Image:
    require_non_negative_float(feather, "--composite-feather")
    if source_crop.size != refined_crop.size or source_crop.size != mask.size:
        raise ValueError("source crop, refined crop, and mask must have equal sizes.")
    blend_mask = mask.convert("L")
    if feather > 0:
        blend_mask = blend_mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return Image.composite(refined_crop.convert("RGB"), source_crop.convert("RGB"), blend_mask)


def resolve_prompts(args, input_path: Path) -> tuple[str, str]:
    png_prompt, png_negative = prompt_from_png(input_path)
    prompt = args.prompt if args.prompt is not None else png_prompt
    if not prompt:
        prompt = DEFAULT_REPAIR_PROMPT
    negative = args.negative_prompt if args.negative_prompt is not None else png_negative
    if negative:
        negative = f"{negative}, {TARGETED_NEGATIVE_PROMPT}"
    else:
        negative = TARGETED_NEGATIVE_PROMPT
    return prompt, negative


def resolve_seed(seed: int) -> int:
    if seed < -1:
        raise ValueError("--seed must be -1 or a non-negative integer.")
    return secrets.randbelow(2**32) if seed == -1 else seed


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_result_png(path: Path, image: Image.Image, source_info: dict, manifest: dict):
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in source_info.items():
        if isinstance(key, str) and isinstance(value, str):
            pnginfo.add_text(key, value)
    pnginfo.add_text(
        "krea2_white_speckle_regenerate",
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
    )
    image.save(path, format="PNG", pnginfo=pnginfo)


def validate_args(args):
    for value, name in (
        (args.white_luma_min, "--white-luma-min"),
        (args.local_contrast_min, "--local-contrast-min"),
        (args.max_chroma, "--max-chroma"),
    ):
        if value < 0 or value > 255:
            raise ValueError(f"{name} must be between 0 and 255.")
    ensure_odd_kernel(args.median_size, "--median-size")
    for value, name in (
        (args.min_area, "--min-area"),
        (args.max_area, "--max-area"),
        (args.max_span, "--max-span"),
        (args.max_components, "--max-components"),
        (args.max_regions, "--max-regions"),
        (args.max_group_edge, "--max-group-edge"),
        (args.minimum_roi_edge, "--minimum-roi-edge"),
        (args.process_long_edge, "--process-long-edge"),
        (args.max_process_pixels, "--max-process-pixels"),
        (args.steps, "--steps"),
        (args.timeout, "--timeout"),
    ):
        require_positive_int(value, name)
    if args.min_area > args.max_area:
        raise ValueError("--min-area must be <= --max-area.")
    if not 0 < args.min_fill_ratio <= 1:
        raise ValueError("--min-fill-ratio must be in (0, 1].")
    if not np.isfinite(args.max_masked_percent) or args.max_masked_percent <= 0:
        raise ValueError("--max-masked-percent must be a finite value > 0.")
    for value, name in (
        (args.mask_radius, "--mask-radius"),
        (args.roi_padding, "--roi-padding"),
        (args.merge_distance, "--merge-distance"),
        (args.mask_blur, "--mask-blur"),
        (args.limit_regions, "--limit-regions"),
    ):
        require_non_negative_int(value, name)
    for value, name in (
        (args.max_background_gradient, "--max-background-gradient"),
        (args.composite_feather, "--composite-feather"),
        (args.cfg, "--cfg"),
        (args.distilled_cfg, "--distilled-cfg"),
        (args.progress_interval, "--progress-interval"),
        (args.no_progress_timeout, "--no-progress-timeout"),
    ):
        require_non_negative_float(value, name)
    if not 0 <= args.denoise <= 1:
        raise ValueError("--denoise must be between 0 and 1.")
    if args.no_progress_timeout > 0 and args.progress_interval <= 0:
        raise ValueError("--no-progress-timeout requires --progress-interval > 0.")
    if args.inpainting_fill not in {0, 1, 2, 3}:
        raise ValueError("--inpainting-fill must be 0, 1, 2, or 3.")
    if args.seed < -1:
        raise ValueError("--seed must be -1 or a non-negative integer.")
    for index, box in enumerate(args.exclude_box, start=1):
        if not isinstance(box, Box):
            raise ValueError(f"--exclude-box #{index} was not parsed as a box.")
        if box.left < 0 or box.top < 0:
            raise ValueError(f"--exclude-box #{index} must start at non-negative coordinates.")
        if box.left >= box.right or box.top >= box.bottom:
            raise ValueError(f"--exclude-box #{index} must have LEFT < RIGHT and TOP < BOTTOM.")


def run(args) -> RunArtifacts:
    validate_args(args)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as opened:
        source_info = dict(opened.info)
        base = ImageOps.exif_transpose(opened).convert("RGB")

    source_protection = None
    protection_input_path = None
    if args.protect_mask is not None:
        protection_input_path = Path(args.protect_mask)
        if not protection_input_path.exists():
            raise FileNotFoundError(protection_input_path)
        with Image.open(protection_input_path) as opened:
            source_protection = ImageOps.exif_transpose(opened).convert("L")
    protection_mask = build_protection_mask(
        base.size,
        source_protection,
        args.exclude_box,
    )

    detection_mask, components, detection_report = detect_white_speckles(
        base,
        white_luma_min=args.white_luma_min,
        local_contrast_min=args.local_contrast_min,
        max_chroma=args.max_chroma,
        median_size=args.median_size,
        min_area=args.min_area,
        max_area=args.max_area,
        max_span=args.max_span,
        min_fill_ratio=args.min_fill_ratio,
        max_background_gradient=args.max_background_gradient,
        max_components=args.max_components,
        max_masked_percent=args.max_masked_percent,
        protection_mask=protection_mask,
    )
    detected_count = len(components)
    all_plans = build_region_plans(
        components,
        image_width=base.width,
        image_height=base.height,
        mask_radius=args.mask_radius,
        roi_padding=args.roi_padding,
        minimum_roi_edge=args.minimum_roi_edge,
        merge_distance=args.merge_distance,
        max_group_edge=args.max_group_edge,
    )
    plans = all_plans[: args.limit_regions] if args.limit_regions > 0 else all_plans
    if not args.dry_run and len(plans) > args.max_regions:
        raise ValueError(f"grouped plan has {len(plans)} API regions, exceeding " f"--max-regions {args.max_regions}; inspect a dry-run preview, " "increase --merge-distance, or use --limit-regions intentionally.")
    prompt, negative_prompt = resolve_prompts(args, input_path)
    base_seed = resolve_seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"krea2_white_speckle_regenerate_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    exact_mask_path = output_dir / "detected_white_speckles.png"
    micro_mask_path = output_dir / "micro_mask.png"
    preview_path = output_dir / "detection_preview.png"
    protection_path = output_dir / "protection_mask.png" if protection_mask is not None else None
    detection_mask.save(exact_mask_path)
    union_mask = build_union_micro_mask(all_plans, base.size)
    union_mask.save(micro_mask_path)
    build_detection_preview(base, union_mask, all_plans).save(preview_path)
    if protection_path is not None:
        protection_mask.save(protection_path)

    manifest = {
        "version": 1,
        "input": str(input_path),
        "source_size": [base.width, base.height],
        "dry_run": bool(args.dry_run),
        "detected_component_count": detected_count,
        "grouped_region_count": len(all_plans),
        "selected_region_count": len(plans),
        "limit_regions": int(args.limit_regions),
        "base_seed": base_seed,
        "protection": {
            "input_mask": (str(protection_input_path) if protection_input_path is not None else None),
            "exclude_boxes": [box.as_dict() for box in args.exclude_box],
            "protected_pixels": detection_report["protected_pixels"],
        },
        "detection": detection_report,
        "settings": {
            "mask_radius": args.mask_radius,
            "roi_padding": args.roi_padding,
            "minimum_roi_edge": args.minimum_roi_edge,
            "merge_distance": args.merge_distance,
            "max_group_edge": args.max_group_edge,
            "max_regions": args.max_regions,
            "process_long_edge": args.process_long_edge,
            "max_process_pixels": args.max_process_pixels,
            "composite_feather": args.composite_feather,
            "steps": args.steps,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "cfg": args.cfg,
            "distilled_cfg": args.distilled_cfg,
            "denoise": args.denoise,
            "mask_blur": args.mask_blur,
            "inpainting_fill": args.inpainting_fill,
        },
        "artifacts": {
            "detected_mask": str(exact_mask_path),
            "micro_mask": str(micro_mask_path),
            "preview": str(preview_path),
            "protection_mask": (str(protection_path) if protection_path is not None else None),
        },
        "regions": [],
    }
    result = base.copy()

    emit(f"OUTPUT_DIR={output_dir}")
    emit(f"INPUT={input_path}")
    emit(f"SOURCE={base.width}x{base.height}")
    emit(f"DETECTED_COMPONENTS={detected_count}")
    emit(f"GROUPED_REGIONS={len(all_plans)}")
    emit(f"SELECTED_REGIONS={len(plans)}")
    emit(f"EXACT_MASK={exact_mask_path}")
    emit(f"MICRO_MASK={micro_mask_path}")
    emit(f"PREVIEW={preview_path}")
    if protection_path is not None:
        emit(f"PROTECTION_MASK={protection_path}")

    progress_args = SimpleNamespace(
        api=args.api,
        timeout=args.timeout,
        progress_interval=args.progress_interval,
        no_progress_timeout=args.no_progress_timeout,
    )
    for plan in plans:
        prefix = f"region_{plan.index:03d}"
        roi_mask = region_mask(plan)
        source_crop = result.crop(plan.roi_box.as_tuple())
        process_width, process_height = process_size_for_roi(
            source_crop.width,
            source_crop.height,
            args.process_long_edge,
            args.max_process_pixels,
        )
        process_input = source_crop.resize((process_width, process_height), Image.Resampling.LANCZOS)
        process_mask = roi_mask.resize((process_width, process_height), Image.Resampling.NEAREST)
        request_mask_blur = scaled_mask_blur(
            args.mask_blur,
            source_crop.size,
            process_input.size,
        )
        region_seed = (base_seed + plan.index - 1) % (2**32)
        payload = build_inpaint_payload(
            args,
            process_input,
            process_mask,
            prompt,
            negative_prompt,
            region_seed,
            request_mask_blur,
        )

        source_crop_path = output_dir / f"{prefix}_source_crop.png"
        roi_mask_path = output_dir / f"{prefix}_roi_mask.png"
        process_input_path = output_dir / f"{prefix}_process_input.png"
        process_mask_path = output_dir / f"{prefix}_process_mask.png"
        request_preview_path = output_dir / f"{prefix}_request_preview.json"
        source_crop.save(source_crop_path)
        roi_mask.save(roi_mask_path)
        process_input.save(process_input_path)
        process_mask.save(process_mask_path)
        write_json(
            request_preview_path,
            {
                **payload_preview(
                    payload,
                    f"<base64 PNG omitted: {process_width}x{process_height}>",
                    f"<base64 mask omitted: {process_width}x{process_height}>",
                ),
                "input_path": str(input_path),
                "plan": plan.as_dict(),
                "source_crop": str(source_crop_path),
                "roi_mask": str(roi_mask_path),
                "process_input": str(process_input_path),
                "process_mask": str(process_mask_path),
            },
        )
        region_manifest = {
            **plan.as_dict(),
            "seed": region_seed,
            "process_size": [process_width, process_height],
            "request_mask_blur": request_mask_blur,
            "source_crop": str(source_crop_path),
            "roi_mask": str(roi_mask_path),
            "process_input": str(process_input_path),
            "process_mask": str(process_mask_path),
            "request_preview": str(request_preview_path),
            "status": "dry-run" if args.dry_run else "pending",
        }
        manifest["regions"].append(region_manifest)
        emit(f"REGION={plan.index}/{len(plans)} " f"COMPONENTS={len(plan.components)} " f"DOT_BOX={plan.mask_box.left},{plan.mask_box.top}," f"{plan.mask_box.right},{plan.mask_box.bottom} " f"ROI={plan.roi_box.left},{plan.roi_box.top}," f"{plan.roi_box.right},{plan.roi_box.bottom} " f"PROCESS={process_width}x{process_height} " f"MASKED_PIXELS={plan.masked_pixels}")
        if args.dry_run:
            continue

        data = post_img2img(progress_args, payload, prefix.upper())
        returned_images = data.get("images") or []
        if not returned_images:
            raise RuntimeError(f"{prefix} returned no image.")
        refined_process = decode_b64_image(returned_images[0])
        if refined_process.size != process_input.size:
            raise RuntimeError(f"{prefix} returned {refined_process.size}, " f"expected {process_input.size}.")
        refined_process_path = output_dir / f"{prefix}_refined_process.png"
        refined_crop_path = output_dir / f"{prefix}_refined_crop.png"
        composite_crop_path = output_dir / f"{prefix}_composite_crop.png"
        refined_process.save(refined_process_path)
        refined_crop = refined_process.resize(source_crop.size, Image.Resampling.LANCZOS)
        refined_crop.save(refined_crop_path)
        composite_crop = composite_refined_region(
            source_crop,
            refined_crop,
            roi_mask,
            args.composite_feather,
        )
        composite_crop.save(composite_crop_path)
        result.paste(composite_crop, (plan.roi_box.left, plan.roi_box.top))
        region_manifest.update(
            {
                "refined_process": str(refined_process_path),
                "refined_crop": str(refined_crop_path),
                "composite_crop": str(composite_crop_path),
                "status": "complete",
            }
        )

    output_path = None
    if not args.dry_run:
        output_path = output_dir / "white_speckle_regenerated.png"
        manifest["output"] = str(output_path)
        save_result_png(output_path, result, source_info, manifest)
        emit(f"IMAGE={output_path}")
    else:
        manifest["output"] = None
        emit("DRY_RUN=1")

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    emit(f"MANIFEST={manifest_path}")
    return RunArtifacts(
        output_dir=output_dir,
        manifest_path=manifest_path,
        output_path=output_path,
        region_count=len(plans),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Detect isolated white specks, create one micro-mask per connected " "component, and regenerate only a small surrounding ROI through Forge."))
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the unique run directory.",
    )
    parser.add_argument("--white-luma-min", type=int, default=250)
    parser.add_argument("--local-contrast-min", type=int, default=80)
    parser.add_argument("--max-chroma", type=int, default=18)
    parser.add_argument("--median-size", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=20)
    parser.add_argument("--max-span", type=int, default=6)
    parser.add_argument("--min-fill-ratio", type=float, default=0.35)
    parser.add_argument("--max-background-gradient", type=float, default=12.0)
    parser.add_argument(
        "--max-components",
        type=int,
        default=256,
        help="Abort detection when more raw connected components are found.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=32,
        help="Abort live generation when more grouped API regions are selected.",
    )
    parser.add_argument(
        "--max-masked-percent",
        type=float,
        default=0.05,
        help="Abort when exact detected pixels exceed this image percentage.",
    )
    parser.add_argument(
        "--limit-regions",
        type=int,
        default=0,
        help="Process only the first N detected regions; 0 processes all.",
    )
    parser.add_argument(
        "--protect-mask",
        default=None,
        help=("Optional same-size mask whose white pixels protect legitimate " "highlights from detection."),
    )
    parser.add_argument(
        "--exclude-box",
        action="append",
        type=parse_box_argument,
        default=[],
        metavar="LEFT,TOP,RIGHT,BOTTOM",
        help=("Protect a rectangular area from detection. Repeat for multiple " "areas."),
    )
    parser.add_argument(
        "--mask-radius",
        type=int,
        default=2,
        help="Output-pixel dilation around each exact detected component.",
    )
    parser.add_argument(
        "--roi-padding",
        type=int,
        default=48,
        help="Context pixels around each micro-mask; these are processed but not pasted.",
    )
    parser.add_argument("--minimum-roi-edge", type=int, default=96)
    parser.add_argument(
        "--merge-distance",
        type=int,
        default=256,
        help=("Merge nearby components into one context ROI while preserving " "separate micro-mask islands."),
    )
    parser.add_argument(
        "--max-group-edge",
        type=int,
        default=640,
        help="Maximum source-pixel edge of a merged context ROI.",
    )
    parser.add_argument("--process-long-edge", type=int, default=512)
    parser.add_argument("--max-process-pixels", type=int, default=512 * 512)
    parser.add_argument(
        "--composite-feather",
        type=float,
        default=1.0,
        help="Output-pixel blur applied only to the micro-mask composite edge.",
    )
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument(
        "--prompt",
        default=None,
        help=("Prompt for local regeneration. Defaults to PNG infotext, then to a " "documented seamless-texture repair prompt."),
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. The targeted white-speck terms are always appended.",
    )
    parser.add_argument("--steps", type=int, default=KREA2_LOCAL_REFINE_STEPS)
    parser.add_argument("--sampler", default=KREA2_DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=KREA2_DEFAULT_SCHEDULER)
    parser.add_argument("--cfg", type=float, default=KREA2_DEFAULT_CFG)
    parser.add_argument("--distilled-cfg", type=float, default=KREA2_DEFAULT_SHIFT)
    parser.add_argument("--denoise", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--mask-blur",
        type=int,
        default=1,
        help="Output-pixel mask blur, scaled to the process ROI and capped at 32.",
    )
    parser.add_argument(
        "--inpainting-fill",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--progress-interval", type=float, default=20.0)
    parser.add_argument("--no-progress-timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write masks, ROI crops, previews, and payloads without calling Forge.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
