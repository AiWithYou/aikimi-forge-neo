"""Pure image-domain primitives for Krea2 Local Supersample Detail.

This module intentionally has no Torch, Gradio, or WebUI imports.  The standard
path enlarges fixed-size source payloads and composites guarded ``C1 - C0``
detail residuals.  The focused rewrite path instead enlarges one complete ROI
with surrounding context, reduces one coherent candidate, and writes the full
round-trip-compensated rewrite only inside the feathered target ROI.

The GUI exposes luma/chroma caps in familiar 8-bit-equivalent units.  Internally
all residual filtering, gating, clipping, and compositing use normalized linear
RGB.  A cap of 8 therefore means ``8 / 255`` in linear-light units; it is not an
sRGB code-value delta.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from modules_forge.vram_canvas import (
    axis_blend_weights,
    axis_positions,
    coordinate_seed,
)

MODE_FULL_IMAGE_GRID = "Full Image Grid"
MODE_ROI_BOXES = "ROI Boxes"
MODE_FOCUSED_ROI_REWRITE = "Focused ROI Rewrite"
MODES = (MODE_FULL_IMAGE_GRID, MODE_ROI_BOXES, MODE_FOCUSED_ROI_REWRITE)

PROFILE_SAFE_1536 = "Safe 1536"
PROFILE_ULTRA_1536 = "Ultra Detail 1536"
PROFILE_ROI_ULTRA_2048 = "ROI Ultra 2048"
PROFILE_FOCUSED_FACE_1536 = "Focused Face Rewrite 1536"

LOCAL_SUPERSAMPLE_PROFILES: dict[str, dict[str, int | float]] = {
    PROFILE_SAFE_1536: {
        "payload": 512,
        "core": 384,
        "overlap": 64,
        "process_edge": 1536,
        "steps": 4,
        "denoise": 0.10,
        "candidates": 1,
        "luma_cap": 8.0,
        "chroma_cap": 2.0,
        "low_frequency_reject_radius": 12.0,
        "context_scale": 2.0,
        "rewrite_feather": 20.0,
    },
    PROFILE_ULTRA_1536: {
        "payload": 512,
        "core": 384,
        "overlap": 64,
        "process_edge": 1536,
        "steps": 5,
        "denoise": 0.15,
        "candidates": 2,
        "luma_cap": 12.0,
        "chroma_cap": 3.0,
        "low_frequency_reject_radius": 12.0,
        "context_scale": 2.0,
        "rewrite_feather": 20.0,
    },
    PROFILE_ROI_ULTRA_2048: {
        "payload": 512,
        "core": 384,
        "overlap": 64,
        "process_edge": 2048,
        "steps": 5,
        "denoise": 0.14,
        "candidates": 2,
        "luma_cap": 12.0,
        "chroma_cap": 3.0,
        "low_frequency_reject_radius": 12.0,
        "context_scale": 2.0,
        "rewrite_feather": 20.0,
    },
    PROFILE_FOCUSED_FACE_1536: {
        "payload": 512,
        "core": 384,
        "overlap": 64,
        "process_edge": 1536,
        "steps": 6,
        "denoise": 0.38,
        "candidates": 2,
        "luma_cap": 12.0,
        "chroma_cap": 3.0,
        "low_frequency_reject_radius": 12.0,
        "context_scale": 2.0,
        "rewrite_feather": 20.0,
    },
}

LOCAL_DETAIL_PROMPT_SUFFIX = "This input is an enlarged local crop, not a complete image. Preserve the " "exact subject identity, face, expression, anatomy, object count, silhouette, " "composition, local geometry, and crop boundaries. Add only coherent, " "material-specific fine detail already implied by the source, such as natural " "hair strands, iris fibers, eyelashes, seams, embroidery, lace, and restrained " "surface detail where those features are already present. Do not add or remove " "people, limbs, hands, fingers, eyes, objects, text, logos, silhouettes, " "outlines, or repeated patterns. Do not introduce random grain, fake noise, " "oversharpening halos, double contours, changed crop edges, or tile seams."
FOCUSED_FACE_PROMPT_SUFFIX = "This is a deliberately magnified context crop centered on one existing face. Re-render that same face at native high resolution with clearly defined symmetrical eyes, visible irises and eyelashes, a coherent nose and mouth, clean facial contours, and fine individual hair strands. Preserve the same identity, expression, head angle, illustration style, complexion, lighting, and surrounding hair. Keep exactly one face and two eyes. Do not add facial features, people, text, blur, or sharpening halos."

LINEAR_LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
DEFAULT_LOW_FREQUENCY_REJECT_RADIUS = 12.0
DEFAULT_INNER_DETAIL_SIGMA = 0.65
DEFAULT_FOCUSED_CONTEXT_SCALE = 2.0
DEFAULT_FOCUSED_REWRITE_FEATHER = 20.0
TEMP_BYTES_PER_PIXEL = 16  # float32 RGB residual sum + float32 weight sum


def get_profile(name: str) -> dict[str, int | float]:
    """Return an isolated profile copy so UI changes cannot mutate defaults."""

    try:
        return dict(LOCAL_SUPERSAMPLE_PROFILES[str(name)])
    except KeyError as exc:
        choices = ", ".join(LOCAL_SUPERSAMPLE_PROFILES)
        raise ValueError(f"unknown local supersample profile {name!r}; choose {choices}") from exc


def append_local_detail_guidance(base_prompt: str) -> str:
    """Append general local-crop guidance once while preserving the prompt prefix."""

    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("local supersample detail requires one non-empty text prompt")
    if LOCAL_DETAIL_PROMPT_SUFFIX in base_prompt:
        return base_prompt
    if base_prompt[-1].isspace():
        separator = ""
    else:
        separator = " " if base_prompt.rstrip().endswith((".", ",", ";", "!", "?")) else ". "
    return f"{base_prompt}{separator}{LOCAL_DETAIL_PROMPT_SUFFIX}"


def append_focused_face_guidance(base_prompt: str) -> str:
    """Append focused face-rewrite guidance once while preserving the prefix."""

    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("focused ROI rewrite requires one non-empty text prompt")
    if FOCUSED_FACE_PROMPT_SUFFIX in base_prompt:
        return base_prompt
    if base_prompt[-1].isspace():
        separator = ""
    else:
        separator = " " if base_prompt.rstrip().endswith((".", ",", ";", "!", "?")) else ". "
    return f"{base_prompt}{separator}{FOCUSED_FACE_PROMPT_SUFFIX}"


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be > 0")
    return int(value)


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


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
        return self.left, self.top, self.right, self.bottom

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]

    def intersects(self, other: "Box") -> bool:
        return self.left < other.right and self.right > other.left and self.top < other.bottom and self.bottom > other.top


def _validate_box(box: Box, width: int, height: int, name: str) -> Box:
    width = _positive_int(width, "image width")
    height = _positive_int(height, "image height")
    if box.left < 0 or box.top < 0 or box.right > width or box.bottom > height:
        raise ValueError(f"{name} must fit inside the source image")
    if box.left >= box.right or box.top >= box.bottom:
        raise ValueError(f"{name} must satisfy left < right and top < bottom")
    return box


def parse_roi_boxes(text: str, width: int, height: int) -> list[Box]:
    """Parse semicolon-separated source-pixel ``left,top,right,bottom`` boxes."""

    width = _positive_int(width, "image width")
    height = _positive_int(height, "image height")
    if text is None or not str(text).strip():
        return []
    boxes: list[Box] = []
    for index, value in enumerate(str(text).split(";"), start=1):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4 or any(not part for part in parts):
            raise ValueError(f"ROI box {index} must be left,top,right,bottom; separate boxes with semicolons")
        try:
            box = Box(*(int(part) for part in parts))
        except ValueError as exc:
            raise ValueError(f"ROI box {index} values must be integers") from exc
        boxes.append(_validate_box(box, width, height, f"ROI box {index}"))
    return boxes


def validate_request(
    *,
    mode: str,
    profile: str,
    roi_boxes: Sequence[Box],
    payload: int,
    core: int,
    overlap: int,
    process_edge: int,
    steps: int,
    denoise: float,
    candidate_count: int,
    luma_cap: float,
    chroma_cap: float,
    low_frequency_reject_radius: float,
    focused_context_scale: float,
    focused_rewrite_feather: float,
    allow_expensive_2048_full_grid: bool,
    maximum_tile_count: int,
) -> None:
    """Fail closed before any long-running model work begins."""

    if mode not in MODES:
        raise ValueError(f"Mode must be one of: {', '.join(MODES)}")
    get_profile(profile)
    payload = _positive_int(payload, "Crop Payload")
    core = _positive_int(core, "Core Size")
    maximum_tile_count = _positive_int(maximum_tile_count, "Maximum Tile Count")
    _positive_int(steps, "Steps")
    if core > payload:
        raise ValueError("Core Size must be <= Crop Payload")
    if (payload - core) % 2:
        raise ValueError("Crop Payload minus Core Size must be even so the core stays centered")
    if int(overlap) < 0 or int(overlap) >= core:
        raise ValueError("Core Overlap must satisfy 0 <= overlap < Core Size")
    if int(process_edge) not in (1536, 2048):
        raise ValueError("Process Edge must be 1536 or 2048")
    if int(candidate_count) not in (1, 2):
        raise ValueError("Candidate Count must be 1 or 2")
    denoise = float(denoise)
    if not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("Denoising Strength must be finite and between 0 and 1")
    _finite_positive(luma_cap, "Luma Residual Cap")
    _finite_positive(chroma_cap, "Chroma Residual Cap")
    _finite_positive(low_frequency_reject_radius, "Low-frequency Reject Radius")
    focused_context_scale = _finite_positive(
        focused_context_scale,
        "Focused Context Scale",
    )
    if focused_context_scale < 1.0 or focused_context_scale > 8.0:
        raise ValueError("Focused Context Scale must be between 1.0 and 8.0")
    focused_rewrite_feather = float(focused_rewrite_feather)
    if not math.isfinite(focused_rewrite_feather) or focused_rewrite_feather < 0.0:
        raise ValueError("Focused Rewrite Feather must be finite and >= 0")
    if profile == PROFILE_ROI_ULTRA_2048:
        if not roi_boxes:
            raise ValueError("ROI Ultra 2048 requires ROI Boxes before processing starts")
        if mode != MODE_ROI_BOXES:
            raise ValueError("ROI Ultra 2048 requires ROI Boxes mode")
    if mode == MODE_ROI_BOXES and not roi_boxes:
        raise ValueError("ROI Boxes mode requires at least one ROI box")
    if mode == MODE_FOCUSED_ROI_REWRITE:
        if not roi_boxes:
            raise ValueError("Focused ROI Rewrite mode requires at least one tight target ROI")
        if profile != PROFILE_FOCUSED_FACE_1536:
            raise ValueError(
                "Focused ROI Rewrite requires the Focused Face Rewrite 1536 profile"
            )
    if profile == PROFILE_FOCUSED_FACE_1536 and mode != MODE_FOCUSED_ROI_REWRITE:
        raise ValueError(
            "Focused Face Rewrite 1536 requires Focused ROI Rewrite mode"
        )
    if mode == MODE_FULL_IMAGE_GRID and int(process_edge) == 2048 and not bool(allow_expensive_2048_full_grid):
        raise ValueError("2048 Full Image Grid is disabled. Enable Allow expensive 2048 full-grid explicitly or use 1536.")


@dataclass(frozen=True)
class LocalTilePlan:
    phase: int
    index: int
    core_x0: int
    core_y0: int
    core_x1: int
    core_y1: int
    payload_x0: int
    payload_y0: int
    payload_x1: int
    payload_y1: int
    source_x0: int
    source_y0: int
    source_x1: int
    source_y1: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    local_core_x0: int
    local_core_y0: int
    local_core_x1: int
    local_core_y1: int
    previous_x_overlap: int
    next_x_overlap: int
    previous_y_overlap: int
    next_y_overlap: int

    @property
    def core_width(self) -> int:
        return self.core_x1 - self.core_x0

    @property
    def core_height(self) -> int:
        return self.core_y1 - self.core_y0

    @property
    def local_core_box(self) -> tuple[int, int, int, int]:
        return (
            self.local_core_x0,
            self.local_core_y0,
            self.local_core_x1,
            self.local_core_y1,
        )

    @property
    def core_box(self) -> Box:
        return Box(self.core_x0, self.core_y0, self.core_x1, self.core_y1)

    @property
    def padding(self) -> tuple[int, int, int, int]:
        return self.pad_left, self.pad_top, self.pad_right, self.pad_bottom


def _axis_tile_values(length: int, payload: int, core: int, overlap: int) -> list[dict[str, int]]:
    positions = axis_positions(length, core, overlap)
    result: list[dict[str, int]] = []
    for index, start in enumerate(positions):
        end = min(length, start + core)
        actual_core = end - start
        context_before = (payload - actual_core) // 2
        payload_start = start - context_before
        payload_end = payload_start + payload
        source_start = max(0, payload_start)
        source_end = min(length, payload_end)
        previous = 0 if index == 0 else max(0, min(actual_core, positions[index - 1] + core - start))
        following = 0 if index == len(positions) - 1 else max(0, min(actual_core, start + actual_core - positions[index + 1]))
        result.append(
            {
                "core0": start,
                "core1": end,
                "payload0": payload_start,
                "payload1": payload_end,
                "source0": source_start,
                "source1": source_end,
                "pad_before": source_start - payload_start,
                "pad_after": payload_end - source_end,
                "local_core0": start - payload_start,
                "local_core1": end - payload_start,
                "previous": previous,
                "following": following,
            }
        )
    return result


def plan_local_tiles(
    width: int,
    height: int,
    *,
    payload: int = 512,
    core: int = 384,
    overlap: int = 64,
) -> list[LocalTilePlan]:
    """Plan a complete fixed-payload core grid, including forced end tiles."""

    width = _positive_int(width, "canvas width")
    height = _positive_int(height, "canvas height")
    payload = _positive_int(payload, "payload")
    core = _positive_int(core, "core")
    if core > payload or (payload - core) % 2:
        raise ValueError("payload/core geometry must keep a centered core")
    if int(overlap) < 0 or int(overlap) >= core:
        raise ValueError("overlap must satisfy 0 <= overlap < core")
    xs = _axis_tile_values(width, payload, core, int(overlap))
    ys = _axis_tile_values(height, payload, core, int(overlap))
    plans: list[LocalTilePlan] = []
    for y in ys:
        for x in xs:
            plans.append(
                LocalTilePlan(
                    phase=0,
                    index=len(plans) + 1,
                    core_x0=x["core0"],
                    core_y0=y["core0"],
                    core_x1=x["core1"],
                    core_y1=y["core1"],
                    payload_x0=x["payload0"],
                    payload_y0=y["payload0"],
                    payload_x1=x["payload1"],
                    payload_y1=y["payload1"],
                    source_x0=x["source0"],
                    source_y0=y["source0"],
                    source_x1=x["source1"],
                    source_y1=y["source1"],
                    pad_left=x["pad_before"],
                    pad_top=y["pad_before"],
                    pad_right=x["pad_after"],
                    pad_bottom=y["pad_after"],
                    local_core_x0=x["local_core0"],
                    local_core_y0=y["local_core0"],
                    local_core_x1=x["local_core1"],
                    local_core_y1=y["local_core1"],
                    previous_x_overlap=x["previous"],
                    next_x_overlap=x["following"],
                    previous_y_overlap=y["previous"],
                    next_y_overlap=y["following"],
                )
            )
    return plans


def plan_focused_rois(
    width: int,
    height: int,
    roi_boxes: Sequence[Box],
    *,
    context_scale: float = DEFAULT_FOCUSED_CONTEXT_SCALE,
) -> list[LocalTilePlan]:
    """Plan one square context payload per disjoint target ROI.

    The ROI is the exact writable core.  The larger square payload is only
    generation context, so one face is never split across independently sampled
    tiles.
    """

    width = _positive_int(width, "canvas width")
    height = _positive_int(height, "canvas height")
    context_scale = _finite_positive(context_scale, "focused context scale")
    if context_scale < 1.0 or context_scale > 8.0:
        raise ValueError("focused context scale must be between 1.0 and 8.0")
    if not roi_boxes:
        raise ValueError("focused ROI planning requires at least one target ROI")

    validated: list[Box] = []
    for index, roi in enumerate(roi_boxes, start=1):
        current = _validate_box(roi, width, height, f"focused ROI {index}")
        if any(current.intersects(previous) for previous in validated):
            raise ValueError(
                "Focused ROI Rewrite target boxes must not overlap; use one box per face"
            )
        validated.append(current)

    plans: list[LocalTilePlan] = []
    for index, roi in enumerate(validated, start=1):
        side = int(math.ceil(max(roi.width, roi.height) * context_scale))
        side = max(side, roi.width, roi.height)
        center_x2 = roi.left + roi.right
        center_y2 = roi.top + roi.bottom
        payload_x0 = (center_x2 - side) // 2
        payload_y0 = (center_y2 - side) // 2
        payload_x1 = payload_x0 + side
        payload_y1 = payload_y0 + side
        source_x0 = max(0, payload_x0)
        source_y0 = max(0, payload_y0)
        source_x1 = min(width, payload_x1)
        source_y1 = min(height, payload_y1)
        plans.append(
            LocalTilePlan(
                phase=1,
                index=index,
                core_x0=roi.left,
                core_y0=roi.top,
                core_x1=roi.right,
                core_y1=roi.bottom,
                payload_x0=payload_x0,
                payload_y0=payload_y0,
                payload_x1=payload_x1,
                payload_y1=payload_y1,
                source_x0=source_x0,
                source_y0=source_y0,
                source_x1=source_x1,
                source_y1=source_y1,
                pad_left=source_x0 - payload_x0,
                pad_top=source_y0 - payload_y0,
                pad_right=payload_x1 - source_x1,
                pad_bottom=payload_y1 - source_y1,
                local_core_x0=roi.left - payload_x0,
                local_core_y0=roi.top - payload_y0,
                local_core_x1=roi.right - payload_x0,
                local_core_y1=roi.bottom - payload_y0,
                previous_x_overlap=0,
                next_x_overlap=0,
                previous_y_overlap=0,
                next_y_overlap=0,
            )
        )
    return plans


def select_tiles_for_rois(plans: Sequence[LocalTilePlan], roi_boxes: Sequence[Box]) -> list[LocalTilePlan]:
    if not roi_boxes:
        return list(plans)
    return [tile for tile in plans if any(tile.core_box.intersects(roi) for roi in roi_boxes)]


def enforce_maximum_tile_count(plans: Sequence[LocalTilePlan], maximum_tile_count: int) -> None:
    maximum = _positive_int(maximum_tile_count, "Maximum Tile Count")
    if len(plans) > maximum:
        raise ValueError(f"local supersample plan has {len(plans)} tiles, exceeding Maximum Tile Count {maximum}")


def _as_rgb_uint8(image: Image.Image | np.ndarray, name: str = "image") -> np.ndarray:
    if isinstance(image, Image.Image):
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        values = np.asarray(image)
        if values.dtype != np.uint8:
            raise ValueError(f"{name} must use uint8 RGB values")
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"{name} must have an HxWx3 RGB shape")
    return np.ascontiguousarray(values)


def extract_padded_payload(
    source_rgb: Image.Image | np.ndarray,
    tile: LocalTilePlan,
    *,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Extract one payload without moving its exact local/global core mapping."""

    source = _as_rgb_uint8(source_rgb, "source")
    if tile.source_x1 > source.shape[1] or tile.source_y1 > source.shape[0]:
        raise ValueError("tile source box lies outside the supplied image")
    crop = source[tile.source_y0 : tile.source_y1, tile.source_x0 : tile.source_x1]
    if pad_mode not in ("edge", "reflect"):
        raise ValueError("pad mode must be edge or reflect")
    mode = pad_mode
    if mode == "reflect" and (crop.shape[0] < 2 or crop.shape[1] < 2):
        mode = "edge"
    padded = np.pad(
        crop,
        (
            (tile.pad_top, tile.pad_bottom),
            (tile.pad_left, tile.pad_right),
            (0, 0),
        ),
        mode=mode,
    )
    expected_shape = (
        tile.payload_y1 - tile.payload_y0,
        tile.payload_x1 - tile.payload_x0,
        3,
    )
    if padded.shape != expected_shape:
        raise AssertionError(f"fixed payload extraction produced {padded.shape}, expected {expected_shape}")
    return np.ascontiguousarray(padded)


def build_axis_normalizers(plans: Sequence[LocalTilePlan], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return O(W + H) raw smoothstep sums for forced-end normalization."""

    width = _positive_int(width, "canvas width")
    height = _positive_int(height, "canvas height")
    if not plans:
        raise ValueError("at least one tile plan is required")
    x_sum = np.zeros(width, dtype=np.float32)
    y_sum = np.zeros(height, dtype=np.float32)
    x_segments = {(tile.core_x0, tile.core_x1, tile.previous_x_overlap, tile.next_x_overlap) for tile in plans}
    y_segments = {(tile.core_y0, tile.core_y1, tile.previous_y_overlap, tile.next_y_overlap) for tile in plans}
    for x0, x1, previous, following in x_segments:
        x_sum[x0:x1] += axis_blend_weights(x1 - x0, previous, following)
    for y0, y1, previous, following in y_segments:
        y_sum[y0:y1] += axis_blend_weights(y1 - y0, previous, following)
    if not np.all(np.isfinite(x_sum)) or not np.all(np.isfinite(y_sum)):
        raise ValueError("tile weight normalizers must be finite")
    if np.any(x_sum <= 0) or np.any(y_sum <= 0):
        raise ValueError("tile plan does not cover the complete canvas")
    return x_sum, y_sum


def normalized_tile_weight(tile: LocalTilePlan, normalizers: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    x_sum, y_sum = normalizers
    if tile.core_x1 > x_sum.size or tile.core_y1 > y_sum.size:
        raise ValueError("tile lies outside its weight normalizer")
    x_raw = axis_blend_weights(tile.core_width, tile.previous_x_overlap, tile.next_x_overlap)
    y_raw = axis_blend_weights(tile.core_height, tile.previous_y_overlap, tile.next_y_overlap)
    denominator = np.outer(
        y_sum[tile.core_y0 : tile.core_y1],
        x_sum[tile.core_x0 : tile.core_x1],
    )
    raw = np.outer(y_raw, x_raw).astype(np.float32)
    result = np.divide(
        raw,
        denominator,
        out=np.zeros_like(raw),
        where=denominator > 0,
    ).astype(np.float32)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("normalized tile weights must be finite and non-negative")
    return result


def roi_core_mask(
    tile: LocalTilePlan,
    roi_boxes: Sequence[Box],
    *,
    feather: float = 0.0,
) -> np.ndarray:
    """Return a core-local ROI union mask with optional inward feathering."""

    if not roi_boxes:
        return np.ones((tile.core_height, tile.core_width), dtype=np.float32)
    feather = float(feather)
    if not math.isfinite(feather) or feather < 0.0:
        raise ValueError("ROI feather must be finite and >= 0")
    mask = np.zeros((tile.core_height, tile.core_width), dtype=np.float32)
    for roi in roi_boxes:
        left = max(tile.core_x0, roi.left)
        top = max(tile.core_y0, roi.top)
        right = min(tile.core_x1, roi.right)
        bottom = min(tile.core_y1, roi.bottom)
        if left < right and top < bottom:
            local_slice = np.s_[
                top - tile.core_y0 : bottom - tile.core_y0,
                left - tile.core_x0 : right - tile.core_x0,
            ]
            if feather <= 0.0:
                mask[local_slice] = 1.0
                continue
            xs = np.arange(left, right, dtype=np.float32) + 0.5
            ys = np.arange(top, bottom, dtype=np.float32) + 0.5
            x_distance = np.minimum(xs - roi.left, roi.right - xs)
            y_distance = np.minimum(ys - roi.top, roi.bottom - ys)
            distance = np.minimum(y_distance[:, None], x_distance[None, :])
            weight = np.clip(distance / feather, 0.0, 1.0)
            weight = (weight * weight * (3.0 - 2.0 * weight)).astype(np.float32)
            mask[local_slice] = np.maximum(mask[local_slice], weight)
    return mask


def tile_composition_weights(
    tile: LocalTilePlan,
    normalizers: tuple[np.ndarray, np.ndarray],
    roi_boxes: Sequence[Box],
    *,
    feather: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual and normalization weights without cancelling ROI feather."""

    tile_weight = normalized_tile_weight(tile, normalizers)
    if not roi_boxes:
        return tile_weight, tile_weight
    roi_weight = roi_core_mask(tile, roi_boxes, feather=feather)
    residual_weight = (tile_weight * roi_weight).astype(np.float32)
    normalization_weight = (
        tile_weight * (roi_weight > 0.0).astype(np.float32)
    ).astype(np.float32)
    return residual_weight, normalization_weight


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("sRGB values must be finite and within 0..1")
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("linear RGB values must be finite")
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def linear_to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(linear_to_srgb(values) * 255.0), 0, 255).astype(np.uint8)


def uint8_to_linear(values: np.ndarray) -> np.ndarray:
    return srgb_to_linear(_as_rgb_uint8(values).astype(np.float32) / 255.0)


def lanczos_upscale(payload_rgb: Image.Image | np.ndarray, process_edge: int) -> np.ndarray:
    """Create the square sRGB process input with Lanczos."""

    process_edge = _positive_int(process_edge, "process edge")
    if process_edge not in (1536, 2048):
        raise ValueError("process edge must be 1536 or 2048")
    payload = _as_rgb_uint8(payload_rgb, "payload")
    if payload.shape[0] != payload.shape[1]:
        raise ValueError("local supersample payload must be square")
    image = Image.fromarray(payload, mode="RGB")
    return np.asarray(
        image.resize((process_edge, process_edge), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def linear_area_downsample(
    process_rgb: Image.Image | np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Decode sRGB, INTER_AREA downsample, and return normalized linear RGB."""

    process = _as_rgb_uint8(process_rgb, "process image")
    width = _positive_int(output_size[0], "output width")
    height = _positive_int(output_size[1], "output height")
    linear = srgb_to_linear(process.astype(np.float32) / 255.0)
    resized = cv2.resize(linear, (width, height), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[..., None]
    resized = np.asarray(resized, dtype=np.float32)
    if resized.shape != (height, width, 3) or not np.all(np.isfinite(resized)):
        raise ValueError("linear area downsample produced an invalid RGB image")
    return resized


def round_trip_residual(
    process_input: Image.Image | np.ndarray,
    refined_process: Image.Image | np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return linear-light ``C0``, ``C1``, and exactly ``C1 - C0``."""

    process = _as_rgb_uint8(process_input, "process input")
    refined = _as_rgb_uint8(refined_process, "refined process image")
    if refined.shape != process.shape:
        raise ValueError("process input and refined process image must share one shape")
    c0 = linear_area_downsample(process, output_size)
    c1 = linear_area_downsample(refined, output_size)
    delta = (c1 - c0).astype(np.float32)
    if not np.all(np.isfinite(delta)):
        raise ValueError("round-trip residual must be finite")
    return c0, c1, delta


def _validate_linear_rgb(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or result.shape[2] != 3:
        raise ValueError(f"{name} must have an HxWx3 shape")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    sigma = _finite_positive(sigma, "Gaussian sigma")
    return cv2.GaussianBlur(
        np.asarray(values, dtype=np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)


def low_frequency_reject(
    delta_linear: np.ndarray,
    *,
    radius: float = DEFAULT_LOW_FREQUENCY_REJECT_RADIUS,
    inner_sigma: float = DEFAULT_INNER_DETAIL_SIGMA,
) -> np.ndarray:
    """Keep an approximately 1--12 px detail band in normalized linear RGB."""

    delta = _validate_linear_rgb(delta_linear, "linear residual")
    radius = _finite_positive(radius, "low-frequency reject radius")
    inner_sigma = _finite_positive(inner_sigma, "inner detail sigma")
    if inner_sigma >= radius:
        raise ValueError("inner detail sigma must be smaller than the reject radius")
    return (_gaussian(delta, inner_sigma) - _gaussian(delta, radius)).astype(np.float32)


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("smoothstep bounds must be finite with high > low")
    t = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def split_luma_chroma(delta_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = _validate_linear_rgb(delta_linear, "linear residual")
    luma = np.tensordot(delta, LINEAR_LUMA_WEIGHTS, axes=([2], [0])).astype(np.float32)
    chroma = (delta - luma[..., None]).astype(np.float32)
    return luma, chroma


def cap_luma_chroma(
    delta_linear: np.ndarray,
    *,
    luma_cap: float,
    chroma_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap separate components using 8-bit-equivalent linear-light units."""

    luma_cap_linear = _finite_positive(luma_cap, "luma cap") / 255.0
    chroma_cap_linear = _finite_positive(chroma_cap, "chroma cap") / 255.0
    luma, chroma = split_luma_chroma(delta_linear)
    luma = np.clip(luma, -luma_cap_linear, luma_cap_linear).astype(np.float32)
    max_chroma = np.max(np.abs(chroma), axis=2)
    chroma_scale = np.minimum(
        1.0,
        chroma_cap_linear / np.maximum(max_chroma, 1e-12),
    ).astype(np.float32)
    chroma = (chroma * chroma_scale[..., None]).astype(np.float32)
    combined = (luma[..., None] + chroma).astype(np.float32)
    return combined, luma, chroma


def strong_edge_guard(base_linear: np.ndarray) -> np.ndarray:
    """Attenuate, but never fully erase, residuals on source silhouettes."""

    base = _validate_linear_rgb(base_linear, "base linear RGB")
    luma = np.tensordot(base, LINEAR_LUMA_WEIGHTS, axes=([2], [0])).astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(gx, gy).astype(np.float32)
    edge = _smoothstep(0.025, 0.12, magnitude)
    return (1.0 - 0.75 * edge).astype(np.float32)


def _boundary_mean(values: np.ndarray) -> float:
    height, width = values.shape[:2]
    border = max(1, min(24, min(height, width) // 16))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border] = True
    mask[-border:] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return float(np.mean(np.abs(values[mask]), dtype=np.float64)) if np.any(mask) else 0.0


@dataclass
class CandidateEvaluation:
    residual: np.ndarray
    c0_linear: np.ndarray
    c1_linear: np.ndarray
    stats: dict[str, float | bool | str]
    accepted: bool
    rejection_reason: str | None


def filter_local_residual(
    base_payload_rgb: Image.Image | np.ndarray,
    c0_linear: np.ndarray,
    c1_linear: np.ndarray,
    *,
    low_frequency_reject_radius: float = DEFAULT_LOW_FREQUENCY_REJECT_RADIUS,
    luma_cap: float = 8.0,
    chroma_cap: float = 2.0,
    protect_strong_edges: bool = True,
) -> CandidateEvaluation:
    """Build one bounded local-detail candidate entirely in linear RGB."""

    base_u8 = _as_rgb_uint8(base_payload_rgb, "base payload")
    base = uint8_to_linear(base_u8)
    c0 = _validate_linear_rgb(c0_linear, "C0")
    c1 = _validate_linear_rgb(c1_linear, "C1")
    if c0.shape != c1.shape or c0.shape != base.shape:
        raise ValueError("base payload, C0, and C1 must share one HxWx3 shape")
    radius = _finite_positive(low_frequency_reject_radius, "low-frequency reject radius")
    luma_cap = _finite_positive(luma_cap, "luma cap")
    chroma_cap = _finite_positive(chroma_cap, "chroma cap")

    raw_delta = (c1 - c0).astype(np.float32)
    raw_luma, _ = split_luma_chroma(raw_delta)
    low_luma = _gaussian(raw_luma, radius)
    band = low_frequency_reject(raw_delta, radius=radius)

    structure_scale = max(luma_cap / 255.0, 1.0 / 255.0)
    structure = np.exp(-np.abs(low_luma) / structure_scale).astype(np.float32)
    edge = strong_edge_guard(base) if protect_strong_edges else np.ones(base.shape[:2], dtype=np.float32)
    gated = (band * structure[..., None] * edge[..., None]).astype(np.float32)
    bounded, bounded_luma, bounded_chroma = cap_luma_chroma(
        gated,
        luma_cap=luma_cap,
        chroma_cap=chroma_cap,
    )

    proposed = base + bounded
    clipping_fraction = float(np.mean((proposed < 0.0) | (proposed > 1.0)))
    # Scale the complete RGB residual vector per pixel instead of clipping its
    # channels independently.  This preserves the already enforced luma/chroma
    # component caps while keeping the actual addition inside source headroom.
    channel_scale = np.ones_like(bounded, dtype=np.float32)
    positive = bounded > 0
    negative = bounded < 0
    np.divide(
        1.0 - base,
        bounded,
        out=channel_scale,
        where=positive,
    )
    negative_scale = np.ones_like(bounded, dtype=np.float32)
    np.divide(
        -base,
        bounded,
        out=negative_scale,
        where=negative,
    )
    channel_scale[negative] = negative_scale[negative]
    pixel_scale = np.minimum(1.0, np.min(channel_scale, axis=2)).astype(np.float32)
    bounded = (bounded * pixel_scale[..., None]).astype(np.float32)

    c0_band_luma = _gaussian(
        np.tensordot(c0, LINEAR_LUMA_WEIGHTS, axes=([2], [0])).astype(np.float32),
        DEFAULT_INNER_DETAIL_SIGMA,
    ) - _gaussian(
        np.tensordot(c0, LINEAR_LUMA_WEIGHTS, axes=([2], [0])).astype(np.float32),
        radius,
    )
    c1_band_luma = _gaussian(
        np.tensordot(c1, LINEAR_LUMA_WEIGHTS, axes=([2], [0])).astype(np.float32),
        DEFAULT_INNER_DETAIL_SIGMA,
    ) - _gaussian(
        np.tensordot(c1, LINEAR_LUMA_WEIGHTS, axes=([2], [0])).astype(np.float32),
        radius,
    )
    detail_increase = float((np.mean(np.abs(c1_band_luma), dtype=np.float64) - np.mean(np.abs(c0_band_luma), dtype=np.float64)) * 255.0)
    residual_scalar = np.mean(np.abs(bounded), axis=2)
    low_abs = np.abs(low_luma) * 255.0
    residual_abs = residual_scalar * 255.0
    boundary_residual = _boundary_mean(bounded) * 255.0
    mean_residual = float(np.mean(residual_abs, dtype=np.float64))
    p95_residual = float(np.percentile(residual_abs, 95))
    mean_drift = float(np.mean(low_abs, dtype=np.float64))
    p95_drift = float(np.percentile(low_abs, 95))

    rejection_reason: str | None = None
    if mean_residual <= 1e-4:
        rejection_reason = "insufficient_detail_residual"
    elif detail_increase <= 0.0:
        rejection_reason = "detail_energy_did_not_increase"
    elif p95_drift > max(luma_cap * 2.0, 18.0):
        rejection_reason = "excessive_low_frequency_drift"
    elif clipping_fraction > 0.01:
        rejection_reason = "excessive_rgb_clipping"
    elif boundary_residual > max(mean_residual * 4.0, luma_cap):
        rejection_reason = "excessive_payload_boundary_residual"

    quality_score = mean_drift * 2.0 + p95_drift * 0.5 + clipping_fraction * 1000.0 + boundary_residual * 0.25 - min(max(detail_increase, 0.0), luma_cap) * 0.2
    stats: dict[str, float | bool | str] = {
        "mean_low_frequency_drift": mean_drift,
        "p95_low_frequency_drift": p95_drift,
        "detail_increase": detail_increase,
        "mean_residual": mean_residual,
        "p95_residual": p95_residual,
        "boundary_residual": boundary_residual,
        "clipping_fraction": clipping_fraction,
        "mean_structure_gate": float(np.mean(structure, dtype=np.float64)),
        "mean_strong_edge_gate": float(np.mean(edge, dtype=np.float64)),
        "maximum_abs_luma_component": float(np.max(np.abs(bounded_luma)) * 255.0),
        "maximum_abs_chroma_component": float(np.max(np.abs(bounded_chroma)) * 255.0),
        "quality_score": float(quality_score),
        "accepted": rejection_reason is None,
        "rejection_reason": rejection_reason or "",
    }
    return CandidateEvaluation(
        residual=bounded,
        c0_linear=c0,
        c1_linear=c1,
        stats=stats,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def evaluate_highres_candidate(
    base_payload_rgb: Image.Image | np.ndarray,
    process_input: Image.Image | np.ndarray,
    refined_process: Image.Image | np.ndarray,
    *,
    low_frequency_reject_radius: float = DEFAULT_LOW_FREQUENCY_REJECT_RADIUS,
    luma_cap: float = 8.0,
    chroma_cap: float = 2.0,
    protect_strong_edges: bool = True,
) -> CandidateEvaluation:
    base = _as_rgb_uint8(base_payload_rgb, "base payload")
    c0, c1, _ = round_trip_residual(
        process_input,
        refined_process,
        (base.shape[1], base.shape[0]),
    )
    return filter_local_residual(
        base,
        c0,
        c1,
        low_frequency_reject_radius=low_frequency_reject_radius,
        luma_cap=luma_cap,
        chroma_cap=chroma_cap,
        protect_strong_edges=protect_strong_edges,
    )


def agreement_mask(
    representative_residual: np.ndarray,
    support_residual: np.ndarray,
) -> np.ndarray:
    """Measure same-place, same-sign, locally correlated residual support."""

    representative = _validate_linear_rgb(representative_residual, "representative residual")
    support = _validate_linear_rgb(support_residual, "support residual")
    if representative.shape != support.shape:
        raise ValueError("candidate residuals must share one shape")
    a_luma, _ = split_luma_chroma(representative)
    b_luma, _ = split_luma_chroma(support)
    cross = _gaussian(a_luma * b_luma, 1.0)
    a_energy = _gaussian(a_luma * a_luma, 1.0)
    b_energy = _gaussian(b_luma * b_luma, 1.0)
    correlation = cross / np.sqrt(np.maximum(a_energy * b_energy, 1e-16))
    correlation_gate = _smoothstep(0.05, 0.75, correlation)

    dot = np.sum(representative * support, axis=2)
    a_norm = np.linalg.norm(representative, axis=2)
    b_norm = np.linalg.norm(support, axis=2)
    direction = dot / np.maximum(a_norm * b_norm, 1e-12)
    direction_gate = _smoothstep(0.0, 0.8, direction)
    magnitude_support = np.minimum(a_norm, b_norm) / np.maximum(np.maximum(a_norm, b_norm), 1e-12)
    sign_gate = (a_luma * b_luma > 0).astype(np.float32)
    active = (a_norm > 0.02 / 255.0) & (b_norm > 0.02 / 255.0)
    mask = correlation_gate * direction_gate * np.sqrt(np.clip(magnitude_support, 0.0, 1.0)) * sign_gate * active.astype(np.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


@dataclass
class CandidateSelection:
    residual: np.ndarray
    selected_index: int | None
    agreement: np.ndarray
    agreement_coverage: float
    rejection_reason: str | None
    quality_gate_override_reason: str | None = None


def select_candidate(
    evaluations: Sequence[CandidateEvaluation],
    *,
    apply_full_rewrite: bool = False,
) -> CandidateSelection:
    """Select a guarded representative or its full downsampled rewrite."""

    if len(evaluations) not in (1, 2):
        raise ValueError("candidate selection requires one or two evaluations")
    shape = evaluations[0].residual.shape
    for evaluation in evaluations:
        residual = _validate_linear_rgb(evaluation.residual, "candidate residual")
        if residual.shape != shape:
            raise ValueError("all candidate residuals must share one shape")
    zero = np.zeros(shape, dtype=np.float32)
    zero_mask = np.zeros(shape[:2], dtype=np.float32)

    if bool(apply_full_rewrite):
        accepted_indices = [
            index for index, evaluation in enumerate(evaluations) if evaluation.accepted
        ]
        selectable_indices = accepted_indices or list(range(len(evaluations)))
        selected_index = min(
            selectable_indices,
            key=lambda index: (
                float(evaluations[index].stats.get("quality_score", math.inf)),
                index,
            ),
        )
        candidate = evaluations[selected_index]
        c0 = _validate_linear_rgb(candidate.c0_linear, "full rewrite C0")
        c1 = _validate_linear_rgb(candidate.c1_linear, "full rewrite C1")
        if c0.shape != shape or c1.shape != shape:
            raise ValueError("full rewrite C0/C1 must match the candidate residual shape")
        return CandidateSelection(
            (c1 - c0).astype(np.float32),
            selected_index,
            np.ones(shape[:2], dtype=np.float32),
            1.0,
            None,
            None if candidate.accepted else candidate.rejection_reason,
        )

    if len(evaluations) == 1:
        candidate = evaluations[0]
        if not candidate.accepted:
            return CandidateSelection(
                zero,
                None,
                zero_mask,
                0.0,
                candidate.rejection_reason or "candidate_failed_quality_gate",
            )
        return CandidateSelection(
            candidate.residual.copy(),
            0,
            np.ones(shape[:2], dtype=np.float32),
            1.0,
            None,
        )

    accepted_indices = [index for index, candidate in enumerate(evaluations) if candidate.accepted]
    if not accepted_indices:
        reasons = [candidate.rejection_reason or "quality_gate_failed" for candidate in evaluations]
        return CandidateSelection(
            zero,
            None,
            zero_mask,
            0.0,
            "candidate_quality_gate: " + "; ".join(reasons),
        )

    selected_index = min(
        accepted_indices,
        key=lambda index: (
            float(evaluations[index].stats.get("quality_score", math.inf)),
            index,
        ),
    )
    support_index = 1 - selected_index
    representative = evaluations[selected_index].residual
    support = evaluations[support_index].residual
    mask = agreement_mask(representative, support)
    active = np.linalg.norm(representative, axis=2) > 0.02 / 255.0
    coverage = float(np.mean(mask[active], dtype=np.float64)) if np.any(active) else 0.0
    if coverage < 0.05:
        return CandidateSelection(
            zero,
            None,
            mask,
            coverage,
            "candidate_agreement_too_low",
        )
    return CandidateSelection(
        (representative * mask[..., None]).astype(np.float32),
        selected_index,
        mask,
        coverage,
        None,
    )


def candidate_seed(
    global_seed: int,
    tile_x: int,
    tile_y: int,
    candidate_index: int,
) -> int:
    if int(candidate_index) not in (0, 1):
        raise ValueError("candidate index must be 0 or 1")
    return coordinate_seed(int(global_seed), int(candidate_index) + 1, int(tile_x), int(tile_y))


def apply_canvas_residual(
    source_rgb: Image.Image | np.ndarray,
    residual_sum: np.ndarray,
    weight_sum: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize CPU moments and add them to B; zero residual is bit-identical."""

    source = _as_rgb_uint8(source_rgb, "source")
    return apply_canvas_residual_striped(
        source,
        residual_sum,
        weight_sum,
        stripe_height=source.shape[0],
    )


def apply_canvas_residual_striped(
    source_rgb: Image.Image | np.ndarray,
    residual_sum: np.ndarray,
    weight_sum: np.ndarray,
    *,
    stripe_height: int = 256,
) -> tuple[np.ndarray, dict[str, float]]:
    """Stripe-wise final composition suitable for disk-backed 4K accumulators."""

    source = _as_rgb_uint8(source_rgb, "source")
    residuals = np.asarray(residual_sum, dtype=np.float32)
    weights = np.asarray(weight_sum, dtype=np.float32)
    stripe_height = _positive_int(stripe_height, "stripe height")
    if residuals.shape != source.shape:
        raise ValueError("residual sum must match the source HxWx3 shape")
    if weights.shape != source.shape[:2]:
        raise ValueError("weight sum must match the source spatial shape")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(weights)):
        raise ValueError("canvas residual and weight sums must be finite")
    if np.any(weights < 0):
        raise ValueError("canvas weights must be non-negative")
    result = source.copy()
    covered_pixels = 0
    clipped_channels = 0
    changed_residual = False
    for y0 in range(0, source.shape[0], stripe_height):
        y1 = min(source.shape[0], y0 + stripe_height)
        stripe_weights = weights[y0:y1]
        covered = stripe_weights > 1e-8
        covered_pixels += int(np.count_nonzero(covered))
        normalized = np.divide(
            residuals[y0:y1],
            stripe_weights[..., None],
            out=np.zeros_like(residuals[y0:y1]),
            where=covered[..., None],
        ).astype(np.float32)
        if not np.any(normalized):
            continue
        changed_residual = True
        base = uint8_to_linear(source[y0:y1])
        proposed = base + normalized
        clipped_channels += int(np.count_nonzero((proposed < 0.0) | (proposed > 1.0)))
        encoded = linear_to_uint8(np.clip(proposed, 0.0, 1.0))
        encoded[~covered] = source[y0:y1][~covered]
        result[y0:y1] = encoded
    if not changed_residual:
        return source.copy(), {
            "clipping_fraction": 0.0,
            "covered_fraction": covered_pixels / float(source.shape[0] * source.shape[1]),
        }
    return result, {
        "clipping_fraction": clipped_channels / float(source.size),
        "covered_fraction": covered_pixels / float(source.shape[0] * source.shape[1]),
    }


def estimate_temporary_bytes(width: int, height: int) -> int:
    return _positive_int(width, "canvas width") * _positive_int(height, "canvas height") * TEMP_BYTES_PER_PIXEL


def rgb_sha256(image: Image.Image | np.ndarray) -> str:
    """Hash decoded RGB dimensions and pixels (avoids self-referential PNG hashes)."""

    values = _as_rgb_uint8(image)
    digest = hashlib.sha256()
    digest.update(f"RGB:{values.shape[1]}x{values.shape[0]}\0".encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def focused_roi_difference_metrics(
    source_rgb: Image.Image | np.ndarray,
    output_rgb: Image.Image | np.ndarray,
    roi_boxes: Sequence[Box],
) -> dict[str, int | float]:
    """Measure focused changes inside the ROI union and its exact exterior."""

    source = _as_rgb_uint8(source_rgb, "focused source")
    output = _as_rgb_uint8(output_rgb, "focused output")
    if source.shape != output.shape:
        raise ValueError("focused source and output must share one HxWx3 shape")
    if not roi_boxes:
        raise ValueError("focused difference metrics require at least one ROI")
    height, width = source.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for index, roi in enumerate(roi_boxes, start=1):
        valid = _validate_box(roi, width, height, f"focused metric ROI {index}")
        mask[valid.top : valid.bottom, valid.left : valid.right] = True
    changed = np.any(source != output, axis=2)
    absolute = np.abs(output.astype(np.int16) - source.astype(np.int16))
    inside_absolute = absolute[mask]
    inside_total = int(np.count_nonzero(mask))
    return {
        "target_pixel_count": inside_total,
        "changed_pixels_inside_target": int(np.count_nonzero(changed & mask)),
        "changed_percent_inside_target": float(np.mean(changed[mask]) * 100.0),
        "changed_pixels_outside_target": int(np.count_nonzero(changed & ~mask)),
        "mean_abs_rgb_delta_inside_target": float(np.mean(inside_absolute)),
        "p95_abs_rgb_delta_inside_target": float(np.percentile(inside_absolute, 95)),
        "max_abs_rgb_delta_inside_target": int(np.max(inside_absolute)),
    }


def validate_krea2_module_names(
    checkpoint: str,
    additional_modules: Iterable[str],
) -> dict[str, str | list[str]]:
    """Validate the configured Krea2/Qwen filenames before tile generation."""

    checkpoint_text = str(checkpoint or "")
    normalized_checkpoint = "".join(character for character in checkpoint_text.lower() if character.isalnum())
    if "krea2" not in normalized_checkpoint:
        raise ValueError("selected checkpoint is not identifiable as Krea2; load Krea2 before local supersampling")
    if isinstance(additional_modules, str):
        module_values = [additional_modules]
    else:
        module_values = [str(value) for value in (additional_modules or [])]
    names = [Path(value).name for value in module_values]
    normalized = ["".join(character for character in name.lower() if character.isalnum()) for name in names]
    if not any("qwenimagevae" in name for name in normalized):
        raise ValueError("Krea2 Local Supersample requires Qwen Image VAE in Forge additional modules")
    if not any("qwen3vl" in name for name in normalized):
        raise ValueError("Krea2 Local Supersample requires Qwen3-VL text encoder in Forge additional modules")
    return {"checkpoint": Path(checkpoint_text).name, "additional_modules": names}
