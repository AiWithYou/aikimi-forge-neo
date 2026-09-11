"""Face/head, hair-flow, and material-region helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analysis import FaceDetection
from .color import luminance
from .frequency import gaussian_blur, robust_normalize


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class RegionSpec:
    region_id: int
    kind: str
    source_box: tuple[float, float, float, float]
    stage_box: Box
    original_face_size: tuple[float, float]
    processing_size: int
    feather: int


@dataclass(frozen=True)
class FaceRegionMasks:
    evaluation: np.ndarray
    write: np.ndarray
    core: np.ndarray
    other_core: np.ndarray


def face_core_source_size(detection: FaceDetection) -> tuple[float, float]:
    if detection.mask is not None:
        mask = np.asarray(detection.mask, dtype=np.float32)
        if mask.ndim == 2 and np.isfinite(mask).all():
            ys, xs = np.nonzero(mask > 0.05)
            if xs.size:
                return (
                    float(int(xs.max()) - int(xs.min()) + 1),
                    float(int(ys.max()) - int(ys.min()) + 1),
                )
    return detection.original_bbox_size


def classify_face_size(detection: FaceDetection) -> str:
    short = min(face_core_source_size(detection))
    if short < 24:
        return "micro"
    if short < 64:
        return "tiny"
    if short < 128:
        return "small"
    if short < 192:
        return "medium"
    return "large"


def default_face_candidate_count(detection: FaceDetection) -> int:
    short = min(face_core_source_size(detection))
    if short < 64:
        return 8
    if short < 128:
        return 6
    if short < 192:
        return 4
    return 3


def expand_face_roi(
    detection: FaceDetection,
    *,
    scale_x: float,
    scale_y: float,
    stage_size: tuple[int, int],
    region_id: int,
) -> RegionSpec:
    x0, y0, x1, y1 = detection.bbox
    width = x1 - x0
    height = y1 - y0
    source_box = (
        x0 - 1.15 * width,
        y0 - 1.45 * height,
        x1 + 1.15 * width,
        y1 + 1.00 * height,
    )
    stage_box = (
        max(0, int(np.floor(source_box[0] * scale_x))),
        max(0, int(np.floor(source_box[1] * scale_y))),
        min(stage_size[0], int(np.ceil(source_box[2] * scale_x))),
        min(stage_size[1], int(np.ceil(source_box[3] * scale_y))),
    )
    width_stage = max(1, stage_box[2] - stage_box[0])
    height_stage = max(1, stage_box[3] - stage_box[1])
    longest = max(width_stage, height_stage)
    processing_size = 768 if longest <= 640 else 896 if longest <= 800 else 1024
    feather = int(np.clip(round(min(width_stage, height_stage) * 0.12), 32, 96))
    return RegionSpec(
        region_id=region_id,
        kind="face",
        source_box=source_box,
        stage_box=stage_box,
        original_face_size=detection.original_bbox_size,
        processing_size=processing_size,
        feather=feather,
    )


def _intersection_over_min(box_a: Box, box_b: Box) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(1, box_a[2] - box_a[0]) * max(1, box_a[3] - box_a[1])
    area_b = max(1, box_b[2] - box_b[0]) * max(1, box_b[3] - box_b[1])
    return intersection / min(area_a, area_b)


def merge_overlapping_regions(
    regions: list[RegionSpec], threshold: float = 0.55
) -> list[RegionSpec]:
    pending = list(regions)
    result: list[RegionSpec] = []
    while pending:
        current = pending.pop(0)
        merged_indices: list[int] = []
        box = current.stage_box
        for index, other in enumerate(pending):
            if _intersection_over_min(box, other.stage_box) >= threshold:
                box = (
                    min(box[0], other.stage_box[0]),
                    min(box[1], other.stage_box[1]),
                    max(box[2], other.stage_box[2]),
                    max(box[3], other.stage_box[3]),
                )
                merged_indices.append(index)
        if merged_indices:
            for index in reversed(merged_indices):
                pending.pop(index)
            current = RegionSpec(
                region_id=current.region_id,
                kind=current.kind,
                source_box=current.source_box,
                stage_box=box,
                original_face_size=current.original_face_size,
                processing_size=current.processing_size,
                feather=current.feather,
            )
        result.append(current)
    return result


def feathered_region_mask(
    canvas_size: tuple[int, int], box: Box, feather: int
) -> np.ndarray:
    width, height = canvas_size
    mask = np.zeros((height, width), dtype=np.float32)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 1.0
    if feather > 0:
        mask = gaussian_blur(mask, max(1.0, feather / 3.0))
        maximum = float(np.max(mask))
        if maximum > 0:
            mask /= maximum
    return np.clip(mask, 0.0, 1.0)


def _stage_face_box(
    detection: FaceDetection,
    *,
    scale_x: float,
    scale_y: float,
    stage_size: tuple[int, int],
) -> Box:
    x0, y0, x1, y1 = detection.bbox
    return (
        max(0, int(np.floor(x0 * scale_x))),
        max(0, int(np.floor(y0 * scale_y))),
        min(stage_size[0], int(np.ceil(x1 * scale_x))),
        min(stage_size[1], int(np.ceil(y1 * scale_y))),
    )


def face_core_mask(
    detection: FaceDetection,
    *,
    scale_x: float,
    scale_y: float,
    stage_size: tuple[int, int],
    output_box: Box | None = None,
) -> np.ndarray:
    """Render a retained component mask (or detector fallback ellipse) locally."""

    output = output_box or (0, 0, stage_size[0], stage_size[1])
    ox0, oy0, ox1, oy1 = output
    result = np.zeros((max(0, oy1 - oy0), max(0, ox1 - ox0)), np.float32)
    face_box = _stage_face_box(
        detection,
        scale_x=scale_x,
        scale_y=scale_y,
        stage_size=stage_size,
    )
    fx0, fy0, fx1, fy1 = face_box
    if fx1 <= fx0 or fy1 <= fy0 or result.size == 0:
        return result

    width = fx1 - fx0
    height = fy1 - fy0
    if detection.mask is not None:
        source_mask = np.asarray(detection.mask, dtype=np.float32)
        expected = (
            int(detection.source_resolution[1]),
            int(detection.source_resolution[0]),
        )
        if source_mask.ndim != 2 or not np.isfinite(source_mask).all():
            source_mask = np.zeros(expected, dtype=np.float32)
        elif source_mask.shape != expected:
            source_mask = cv2.resize(
                source_mask,
                detection.source_resolution,
                interpolation=cv2.INTER_AREA,
            )
        sx0 = max(0, int(np.floor(detection.bbox[0])))
        sy0 = max(0, int(np.floor(detection.bbox[1])))
        sx1 = min(source_mask.shape[1], int(np.ceil(detection.bbox[2])))
        sy1 = min(source_mask.shape[0], int(np.ceil(detection.bbox[3])))
        source_crop = source_mask[sy0:sy1, sx0:sx1]
        if source_crop.size and float(np.max(source_crop)) > 0.0:
            face = cv2.resize(
                source_crop,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            face = np.clip(face, 0.0, 1.0).astype(np.float32)
        else:
            face = np.zeros((height, width), np.float32)
    else:
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        nx = (xx + 0.5 - width * 0.5) / max(width * 0.48, 1.0)
        ny = (yy + 0.5 - height * 0.5) / max(height * 0.49, 1.0)
        distance = np.sqrt(nx * nx + ny * ny)
        face = np.clip((1.08 - distance) / 0.16, 0.0, 1.0).astype(np.float32)

    ix0 = max(ox0, fx0)
    iy0 = max(oy0, fy0)
    ix1 = min(ox1, fx1)
    iy1 = min(oy1, fy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return result
    result[iy0 - oy0 : iy1 - oy0, ix0 - ox0 : ix1 - ox0] = face[
        iy0 - fy0 : iy1 - fy0,
        ix0 - fx0 : ix1 - fx0,
    ]
    return np.clip(result, 0.0, 1.0)


def face_core_union_mask(
    detections: list[FaceDetection],
    *,
    scale_x: float,
    scale_y: float,
    stage_size: tuple[int, int],
) -> np.ndarray:
    union = np.zeros((stage_size[1], stage_size[0]), dtype=np.float32)
    for detection in detections:
        box = _stage_face_box(
            detection,
            scale_x=scale_x,
            scale_y=scale_y,
            stage_size=stage_size,
        )
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        local = face_core_mask(
            detection,
            scale_x=scale_x,
            scale_y=scale_y,
            stage_size=stage_size,
            output_box=box,
        )
        union[y0:y1, x0:x1] = np.maximum(union[y0:y1, x0:x1], local)
    return np.clip(union, 0.0, 1.0)


def face_region_masks(
    detections: list[FaceDetection],
    region_id: int,
    *,
    scale_x: float,
    scale_y: float,
    stage_size: tuple[int, int],
    context_box: Box,
) -> FaceRegionMasks:
    """Build person-owned evaluation and write masks inside one context crop."""

    if not 0 <= region_id < len(detections):
        raise IndexError("Face region id is outside the detection list.")
    x0, y0, x1, y1 = context_box
    height, width = y1 - y0, x1 - x0
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    cores: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for detection in detections:
        core = face_core_mask(
            detection,
            scale_x=scale_x,
            scale_y=scale_y,
            stage_size=stage_size,
            output_box=context_box,
        )
        cores.append(core)
        fx0, fy0, fx1, fy1 = _stage_face_box(
            detection,
            scale_x=scale_x,
            scale_y=scale_y,
            stage_size=stage_size,
        )
        face_width = max(float(fx1 - fx0), 1.0)
        face_height = max(float(fy1 - fy0), 1.0)
        center_x = 0.5 * (fx0 + fx1)
        center_y = 0.5 * (fy0 + fy1)
        distances.append(
            np.square((xx - center_x) / (0.70 * face_width))
            + np.square((yy - center_y) / (0.70 * face_height))
        )

    distance_stack = np.stack(distances, axis=0)
    hard_owner = np.argmin(distance_stack, axis=0) == region_id
    shifted = distance_stack - np.min(distance_stack, axis=0, keepdims=True)
    ownership = np.exp(-2.0 * np.minimum(shifted, 40.0))
    ownership /= np.maximum(np.sum(ownership, axis=0, keepdims=True), 1e-6)
    target_core = cores[region_id]
    other_core = np.zeros((height, width), dtype=np.float32)
    for index, core in enumerate(cores):
        if index != region_id:
            other_core = np.maximum(other_core, core)
    evaluation = target_core * hard_owner.astype(np.float32)
    if float(np.sum(evaluation, dtype=np.float64)) <= 1e-6:
        evaluation = target_core.copy()

    face_box = _stage_face_box(
        detections[region_id],
        scale_x=scale_x,
        scale_y=scale_y,
        stage_size=stage_size,
    )
    margin = int(
        np.clip(
            round(
                min(
                    face_box[2] - face_box[0],
                    face_box[3] - face_box[1],
                )
                * 0.10
            ),
            2,
            64,
        )
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
    )
    expanded = cv2.dilate(target_core, kernel)
    expanded = gaussian_blur(expanded, max(1.0, margin / 2.5))
    maximum = float(np.max(expanded))
    if maximum > 0:
        expanded /= maximum
    write = (
        expanded
        * ownership[region_id]
        * (1.0 - np.clip(other_core * 1.25, 0.0, 1.0))
    )
    return FaceRegionMasks(
        evaluation=np.clip(evaluation, 0.0, 1.0).astype(np.float32),
        write=np.clip(write, 0.0, 1.0).astype(np.float32),
        core=np.clip(target_core, 0.0, 1.0).astype(np.float32),
        other_core=np.clip(other_core, 0.0, 1.0).astype(np.float32),
    )


def local_feather_mask(size: tuple[int, int], feather: int) -> np.ndarray:
    width, height = size
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    wx = np.clip(x / max(feather, 1), 0.0, 1.0)
    wy = np.clip(y / max(feather, 1), 0.0, 1.0)
    return (wx[None, :] * wy[:, None]).astype(np.float32)


def hair_region_mask(
    canvas_size: tuple[int, int],
    face_box: Box,
    *,
    expansion: float = 1.15,
    output_box: Box | None = None,
) -> np.ndarray:
    width, height = canvas_size
    x0, y0, x1, y1 = face_box
    face_w = x1 - x0
    face_h = y1 - y0
    head_box = (
        max(0, round(x0 - expansion * face_w)),
        max(0, round(y0 - 1.6 * face_h)),
        min(width, round(x1 + expansion * face_w)),
        min(height, round(y1 + 0.85 * face_h)),
    )
    if output_box is None:
        head = feathered_region_mask(
            canvas_size, head_box, max(16, round(face_w * 0.2))
        )
        face_core = feathered_region_mask(
            canvas_size, face_box, max(8, round(face_w * 0.08))
        )
    else:
        ox0, oy0, ox1, oy1 = output_box

        def local_box_mask(box: Box, feather: int) -> np.ndarray:
            local = (
                min(ox1 - ox0, max(0, box[0] - ox0)),
                min(oy1 - oy0, max(0, box[1] - oy0)),
                max(0, min(ox1 - ox0, box[2] - ox0)),
                max(0, min(oy1 - oy0, box[3] - oy0)),
            )
            return feathered_region_mask(
                (ox1 - ox0, oy1 - oy0), local, feather
            )

        head = local_box_mask(head_box, max(16, round(face_w * 0.2)))
        face_core = local_box_mask(face_box, max(8, round(face_w * 0.08)))
    # Protect face center while retaining the fringe boundary.
    return np.clip(head * (1.0 - 0.85 * face_core), 0.0, 1.0)


@dataclass(frozen=True)
class HairFlowScore:
    orientation_alignment: float
    strand_continuity: float
    crossing_penalty: float
    silhouette_change: float
    total: float


def hair_flow_score(
    anchor: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> HairFlowScore:
    anchor_y = luminance(anchor)
    candidate_y = luminance(candidate)
    ax = cv2.Sobel(anchor_y, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(anchor_y, cv2.CV_32F, 0, 1, ksize=3)
    cx = cv2.Sobel(candidate_y, cv2.CV_32F, 1, 0, ksize=3)
    cy = cv2.Sobel(candidate_y, cv2.CV_32F, 0, 1, ksize=3)
    an = cv2.magnitude(ax, ay)
    cn = cv2.magnitude(cx, cy)
    cosine = (ax * cx + ay * cy) / (an * cn + 1e-6)
    active = (mask > 0.2) & (
        an
        > (
            np.percentile(an[mask > 0.1], 40)
            if np.any(mask > 0.1)
            else np.inf
        )
    )
    if not np.any(active):
        alignment = 1.0
    else:
        alignment = float(np.mean(np.clip(np.abs(cosine[active]), 0.0, 1.0)))

    # Added-strand orientation is measured with a sign-invariant double-angle
    # tensor against the dominant source flow. This also sees new crossing lines
    # away from an existing source edge.
    residual_y = candidate_y - anchor_y
    rx = cv2.Sobel(residual_y, cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(residual_y, cv2.CV_32F, 0, 1, ksize=3)
    rn = cv2.magnitude(rx, ry)
    masked_anchor = mask * an
    source_x = float(np.sum((ax * ax - ay * ay) * mask))
    source_y = float(np.sum((2.0 * ax * ay) * mask))
    source_norm = max((source_x * source_x + source_y * source_y) ** 0.5, 1e-6)
    source_x /= source_norm
    source_y /= source_norm
    residual_norm2 = rx * rx + ry * ry + 1e-6
    residual_x = (rx * rx - ry * ry) / residual_norm2
    residual_y2 = (2.0 * rx * ry) / residual_norm2
    added = (mask > 0.2) & (
        rn
        > (
            np.percentile(rn[mask > 0.1], 65)
            if np.any(mask > 0.1)
            else np.inf
        )
    )
    if np.any(added) and float(np.sum(masked_anchor)) > 1e-6:
        added_alignment = np.clip(
            (residual_x * source_x + residual_y2 * source_y + 1.0) * 0.5,
            0.0,
            1.0,
        )
        crossing = float(np.mean(added_alignment[added] < 0.35))
        alignment = 0.55 * alignment + 0.45 * float(
            np.mean(added_alignment[added])
        )
    else:
        crossing = 0.0
    candidate_coherence = gaussian_blur(cx * cx - cy * cy, 1.5)
    continuity = float(
        np.mean(np.abs(candidate_coherence) * mask)
        / max(float(np.mean(cn * mask)), 1e-6)
    )
    continuity = float(np.clip(continuity, 0.0, 1.0))
    anchor_silhouette = cv2.morphologyEx(
        ((an > np.percentile(an, 85)) & (mask > 0.05)).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    candidate_silhouette = cv2.morphologyEx(
        ((cn > np.percentile(cn, 85)) & (mask > 0.05)).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    silhouette = float(np.mean(anchor_silhouette != candidate_silhouette))
    total = alignment + 0.7 * continuity - 1.4 * crossing - 1.5 * silhouette
    return HairFlowScore(alignment, continuity, crossing, silhouette, total)


def detail_potential_map(
    linear_rgb: np.ndarray,
    *,
    flat_region_detail: float,
    face_protection: np.ndarray | None = None,
    manual_boost: np.ndarray | None = None,
) -> np.ndarray:
    y = luminance(linear_rgb)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    edges = robust_normalize(cv2.magnitude(gx, gy), 45.0, 97.0)
    variance = np.maximum(
        0.0, gaussian_blur(y * y, 2.0) - gaussian_blur(y, 2.0) ** 2
    )
    texture = robust_normalize(variance, 45.0, 97.0)
    potential = np.clip(
        0.45 * edges + 0.45 * texture + float(flat_region_detail) * (1.0 - texture),
        0.0,
        1.0,
    )
    if face_protection is not None:
        potential *= 1.0 - np.clip(
            1.5 * np.asarray(face_protection, dtype=np.float32), 0.0, 1.0
        )
    if manual_boost is not None:
        potential *= 1.0 + 0.75 * np.clip(manual_boost, 0.0, 1.0)
    return np.clip(potential, 0.0, 1.0).astype(np.float32)
