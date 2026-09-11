"""Pure planning and image-domain primitives for VRAM-Canvas.

The module deliberately has no Torch, Gradio, or WebUI imports.  The high-resolution
driver can therefore validate geometry and blending on machines without a GPU, while
the actual diffusion work is delegated to Forge's img2img API one bounded tile at a
time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from PIL import Image

GIB = 1024**3
DEFAULT_VRAM_USE_FRACTION = 0.85
DEFAULT_MODEL_RESERVE_GIB = 5.5
DEFAULT_ACTIVATION_BYTES_PER_PIXEL = 4096
DEFAULT_MIN_TILE_SIZE = 384
DEFAULT_MAX_TILE_SIZE = 1280
DEFAULT_TILE_ALIGNMENT = 64
DEFAULT_DIFFUSION_ALIGNMENT = 16
DEFAULT_BASE_DETAIL_SIGMA = 6.0
DEFAULT_NOVEL_DETAIL_INNER_RADIUS = 1
DEFAULT_NOVEL_DETAIL_OUTER_RADIUS = 4
DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA = 6.0
DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA = 0.75
DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH = 8.0
CONSENSUS_MERGE_MODE = "consensus"
PHASE_WEAVE_MERGE_MODE = "phase_weave"
PHASE_WEAVE_DETAIL_FLOOR = 0.90
PHASE_WEAVE_SUPPORT_MIX = 0.10
PHASE_WEAVE_SUPPORT_CONFIDENCE_POWER = 2.0
PHASE_WEAVE_QUALITY_RADIUS = 16
PHASE_WEAVE_FEATHER_RADIUS = 5
PHASE_WEAVE_SELECTION_CONFIDENCE_FLOOR = 0.25
PHASE_WEAVE_SELECTION_MARGIN = 0.03
PHASE_WEAVE_GUIDED_EPSILON = 1e-3
PHASE_WEAVE_PROPAGATION_RADIUS = 24
PHASE_WEAVE_PROPAGATION_CONFIDENCE = 0.15
PHASE_WEAVE_ISLAND_MIN_AREA = 3000
PHASE_WEAVE_INPUT_ISLAND_MIN_AREA = 512
PHASE_WEAVE_STRONG_REJECTION_RATIO = 1.0
PHASE_WEAVE_FIDELITY_RADIUS = 8
PHASE_WEAVE_LOW_FREQUENCY_SIGMA = 12.0
PHASE_WEAVE_LOW_FREQUENCY_LUMA_GAIN = 0.32
PHASE_WEAVE_LOW_FREQUENCY_CHROMA_GAIN = 0.18
PHASE_WEAVE_HIGHLIGHT_THRESHOLD = 0.85
PHASE_WEAVE_HIGHLIGHT_LOW_FREQUENCY_SCALE = 0.50
PHASE_WEAVE_FIDELITY_REJECT_THRESHOLD = 0.42
PHASE_WEAVE_EDGE_FIDELITY_WEIGHT = 1.25
PHASE_WEAVE_LOW_FIDELITY_WEIGHT = 1.00
PHASE_WEAVE_CHROMA_FIDELITY_WEIGHT = 0.75
PHASE_WEAVE_CLOSE_RMS = 3.0
PHASE_WEAVE_CLOSE_SSIM = 0.96
PHASE_WEAVE_SUPPORT_SSIM = 0.90
PHASE_WEAVE_ALIGNMENT_RADIUS = 1
PHASE_WEAVE_CONTEXT_RADIUS = max(
    2 * PHASE_WEAVE_QUALITY_RADIUS + PHASE_WEAVE_PROPAGATION_RADIUS,
    int(math.ceil(2.0 * math.sqrt(PHASE_WEAVE_ISLAND_MIN_AREA))),
    int(math.ceil(3.0 * PHASE_WEAVE_LOW_FREQUENCY_SIGMA))
    + PHASE_WEAVE_PROPAGATION_RADIUS
    + PHASE_WEAVE_FEATHER_RADIUS,
)
BASE_WORK_BYTES_PER_PIXEL = 32
MOMENT_WORK_BYTES_PER_PIXEL = 20
NOVEL_MOMENT_WORK_BYTES_PER_PIXEL = 16


def phase_weave_configuration() -> dict[str, float | int | bool | str]:
    """Return the fixed experimental defaults recorded in manifests and PNGs."""

    return {
        "selection_mode": "ternary_input_fallback",
        "input_fallback": True,
        "selection_margin": PHASE_WEAVE_SELECTION_MARGIN,
        "guided_filter_radius": PHASE_WEAVE_QUALITY_RADIUS,
        "guided_filter_epsilon": PHASE_WEAVE_GUIDED_EPSILON,
        "propagation_radius": PHASE_WEAVE_PROPAGATION_RADIUS,
        "propagation_confidence": PHASE_WEAVE_PROPAGATION_CONFIDENCE,
        "island_min_area": PHASE_WEAVE_ISLAND_MIN_AREA,
        "input_island_min_area": PHASE_WEAVE_INPUT_ISLAND_MIN_AREA,
        "strong_rejection_ratio": PHASE_WEAVE_STRONG_REJECTION_RATIO,
        "fidelity_guided_radius": PHASE_WEAVE_FIDELITY_RADIUS,
        "feather_radius": PHASE_WEAVE_FEATHER_RADIUS,
        "low_frequency_sigma": PHASE_WEAVE_LOW_FREQUENCY_SIGMA,
        "low_frequency_luma_gain": PHASE_WEAVE_LOW_FREQUENCY_LUMA_GAIN,
        "low_frequency_chroma_gain": PHASE_WEAVE_LOW_FREQUENCY_CHROMA_GAIN,
        "highlight_threshold": PHASE_WEAVE_HIGHLIGHT_THRESHOLD,
        "highlight_low_frequency_scale": (
            PHASE_WEAVE_HIGHLIGHT_LOW_FREQUENCY_SCALE
        ),
        "quality_measure": "high_frequency_fidelity",
        "fidelity_reject_threshold": PHASE_WEAVE_FIDELITY_REJECT_THRESHOLD,
        "edge_fidelity_weight": PHASE_WEAVE_EDGE_FIDELITY_WEIGHT,
        "low_frequency_fidelity_weight": PHASE_WEAVE_LOW_FIDELITY_WEIGHT,
        "chroma_fidelity_weight": PHASE_WEAVE_CHROMA_FIDELITY_WEIGHT,
        "close_candidate_rms": PHASE_WEAVE_CLOSE_RMS,
        "close_candidate_ssim": PHASE_WEAVE_CLOSE_SSIM,
        "detail_floor": PHASE_WEAVE_DETAIL_FLOOR,
        "support_mix": PHASE_WEAVE_SUPPORT_MIX,
        "support_confidence_power": PHASE_WEAVE_SUPPORT_CONFIDENCE_POWER,
        "support_ssim_threshold": PHASE_WEAVE_SUPPORT_SSIM,
        "support_alignment_radius": PHASE_WEAVE_ALIGNMENT_RADIUS,
        "processing_context_radius": PHASE_WEAVE_CONTEXT_RADIUS,
    }


def _positive(value: float, name: str) -> float:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and > 0.")
    return float(value)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be > 0.")
    return int(value)


def floor_aligned(value: float, alignment: int) -> int:
    alignment = _positive_int(alignment, "alignment")
    return max(alignment, int(math.floor(float(value) / alignment)) * alignment)


def nearest_aligned(value: float, alignment: int) -> int:
    alignment = _positive_int(alignment, "alignment")
    return max(alignment, int(round(float(value) / alignment)) * alignment)


def resolve_tile_size(
    total_vram_gib: float,
    *,
    requested_tile_size: int = 0,
    use_fraction: float = DEFAULT_VRAM_USE_FRACTION,
    model_reserve_gib: float = DEFAULT_MODEL_RESERVE_GIB,
    activation_bytes_per_pixel: int = DEFAULT_ACTIVATION_BYTES_PER_PIXEL,
    minimum: int = DEFAULT_MIN_TILE_SIZE,
    maximum: int = DEFAULT_MAX_TILE_SIZE,
    alignment: int = DEFAULT_TILE_ALIGNMENT,
) -> int:
    """Resolve a conservative square diffusion payload from a VRAM budget.

    ``activation_bytes_per_pixel`` is intentionally exposed: model families and
    attention implementations differ.  The default is a conservative engineering
    prior, not a promise that every checkpoint will fit.
    """

    minimum = _positive_int(minimum, "minimum tile size")
    maximum = _positive_int(maximum, "maximum tile size")
    alignment = _positive_int(alignment, "tile alignment")
    if minimum > maximum:
        raise ValueError("minimum tile size must be <= maximum tile size.")

    aligned_minimum = int(math.ceil(minimum / alignment)) * alignment
    aligned_maximum = floor_aligned(maximum, alignment)
    if aligned_minimum > aligned_maximum:
        raise ValueError("tile bounds contain no aligned size.")

    total_vram_gib = _positive(total_vram_gib, "total VRAM GiB")
    use_fraction = _positive(use_fraction, "VRAM use fraction")
    if use_fraction > 1:
        raise ValueError("VRAM use fraction must be <= 1.")
    if not math.isfinite(float(model_reserve_gib)) or model_reserve_gib < 0:
        raise ValueError("model reserve GiB must be finite and >= 0.")
    activation_bytes_per_pixel = _positive_int(activation_bytes_per_pixel, "activation bytes per pixel")

    available_bytes = (
        total_vram_gib * use_fraction - float(model_reserve_gib)
    ) * GIB
    minimum_required_bytes = (
        aligned_minimum * aligned_minimum * activation_bytes_per_pixel
    )
    if available_bytes < minimum_required_bytes:
        raise ValueError(
            "VRAM budget cannot fit the minimum aligned diffusion tile after the "
            "model/runtime reserve. Lower the reserve only with measured evidence or "
            "use a smaller model."
        )
    estimated_edge = math.sqrt(available_bytes / activation_bytes_per_pixel)
    feasible_edge = min(aligned_maximum, floor_aligned(estimated_edge, alignment))

    if requested_tile_size:
        if requested_tile_size < minimum or requested_tile_size > maximum:
            raise ValueError("requested tile size is outside the configured bounds.")
        requested_edge = min(
            aligned_maximum,
            max(aligned_minimum, floor_aligned(requested_tile_size, alignment)),
        )
        if requested_edge > feasible_edge:
            raise ValueError(
                f"requested tile {requested_edge} exceeds the estimated feasible edge "
                f"{feasible_edge} for the declared VRAM budget."
            )
        return requested_edge

    return max(aligned_minimum, feasible_edge)


def resolve_halo(tile_size: int, requested_halo: int = 0) -> int:
    tile_size = _positive_int(tile_size, "tile size")
    if requested_halo < 0:
        raise ValueError("halo must be >= 0.")
    halo = requested_halo or nearest_aligned(max(32, tile_size / 8), 16)
    if halo * 2 >= tile_size:
        raise ValueError("halo must be smaller than half the tile size.")
    return int(halo)


def resolve_core_overlap(core_size: int, halo: int, requested_overlap: int = 0) -> int:
    core_size = _positive_int(core_size, "core size")
    if halo < 0 or requested_overlap < 0:
        raise ValueError("halo and core overlap must be >= 0.")
    overlap = requested_overlap or nearest_aligned(max(16, halo / 2), 16)
    if overlap >= core_size:
        raise ValueError("core overlap must be smaller than the core size.")
    return int(overlap)


def progressive_stage_sizes(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    max_stage_scale: float = 2.0,
    intermediate_alignment: int = DEFAULT_DIFFUSION_ALIGNMENT,
) -> list[tuple[int, int]]:
    """Return monotonically progressive sizes ending at the exact delivery size."""

    source_width = _positive_int(source_width, "source width")
    source_height = _positive_int(source_height, "source height")
    target_width = _positive_int(target_width, "target width")
    target_height = _positive_int(target_height, "target height")
    max_stage_scale = _positive(max_stage_scale, "maximum stage scale")
    intermediate_alignment = _positive_int(intermediate_alignment, "intermediate alignment")
    if max_stage_scale <= 1:
        raise ValueError("maximum stage scale must be > 1.")

    width_ratio = target_width / source_width
    height_ratio = target_height / source_height
    largest_ratio = max(width_ratio, height_ratio)
    if largest_ratio <= 1:
        return [(target_width, target_height)]

    stage_count = max(1, int(math.ceil(math.log(largest_ratio, max_stage_scale))))
    sizes: list[tuple[int, int]] = []
    previous = (source_width, source_height)
    for index in range(1, stage_count + 1):
        if index == stage_count:
            candidate = (target_width, target_height)
        else:
            fraction = index / stage_count
            candidate = (
                nearest_aligned(source_width * (width_ratio**fraction), intermediate_alignment),
                nearest_aligned(source_height * (height_ratio**fraction), intermediate_alignment),
            )
        if candidate != previous and (not sizes or candidate != sizes[-1]):
            sizes.append(candidate)
        previous = candidate

    if not sizes or sizes[-1] != (target_width, target_height):
        sizes.append((target_width, target_height))
    return sizes


def axis_positions(
    length: int,
    core_size: int,
    overlap: int,
    *,
    phase: int = 0,
    phase_count: int = 1,
    virtual_padding: bool = False,
) -> list[int]:
    """Return edge-anchored positions or an edge-balanced virtual grid.

    The virtual layout keeps one exact stride for every phase and moves the shared
    grid origin so that none of the four phase/edge intersections becomes an
    avoidably thin strip.  Tiles remain full-size model inputs; only their canvas
    intersections are accumulated.
    """

    length = _positive_int(length, "axis length")
    core_size = _positive_int(core_size, "core size")
    if overlap < 0 or overlap >= core_size:
        raise ValueError("overlap must satisfy 0 <= overlap < core size.")
    phase_count = _positive_int(phase_count, "phase count")
    if phase < 0 or phase >= phase_count:
        raise ValueError("phase index is outside the phase count.")
    if length <= core_size and not virtual_padding:
        return [0]

    stride = core_size - overlap
    offset = int(round(stride * phase / phase_count)) if phase else 0
    if virtual_padding:
        origin = balanced_virtual_axis_origin(
            length,
            core_size,
            overlap,
            phase_count=phase_count,
        )
        residue = (origin + offset) % stride
        return _regular_virtual_axis_positions(
            length,
            core_size,
            stride,
            residue,
        )

    end = length - core_size
    positions = {0, end}
    positions.update(range(offset, end + 1, stride))
    return sorted(positions)


def _regular_virtual_axis_positions(
    length: int,
    core_size: int,
    stride: int,
    residue: int,
) -> list[int]:
    """Return one regular grid residue that covers the complete finite axis."""

    residue %= stride
    first = residue - stride if residue > 0 else 0
    positions = [first]
    while positions[-1] + core_size < length:
        positions.append(positions[-1] + stride)
    return positions


def balanced_virtual_axis_origin(
    length: int,
    core_size: int,
    overlap: int,
    *,
    phase_count: int = 2,
) -> int:
    """Choose a shared virtual-grid origin with balanced canvas-edge coverage.

    A fixed half-stride shift can leave a final tile contributing only a very thin
    strip when the canvas length is not a multiple of the stride.  We examine the
    finite set of integer origins modulo one stride and maximize the smallest edge
    intersection across all phases.  Ties prefer a smaller coverage spread, then
    more total edge evidence, and finally stronger edge evidence in phase zero.
    """

    length = _positive_int(length, "axis length")
    core_size = _positive_int(core_size, "core size")
    if overlap < 0 or overlap >= core_size:
        raise ValueError("overlap must satisfy 0 <= overlap < core size.")
    phase_count = _positive_int(phase_count, "phase count")
    stride = core_size - overlap
    best_origin = 0
    best_score: tuple[int, int, int, int, int] | None = None
    for origin in range(stride):
        edge_coverages: list[int] = []
        phase0_edge_coverage = 0
        for phase in range(phase_count):
            offset = int(round(stride * phase / phase_count)) if phase else 0
            positions = _regular_virtual_axis_positions(
                length,
                core_size,
                stride,
                (origin + offset) % stride,
            )
            phase_edges = [
                min(length, position + core_size) - max(0, position)
                for position in (positions[0], positions[-1])
            ]
            edge_coverages.extend(phase_edges)
            if phase == 0:
                phase0_edge_coverage = sum(phase_edges)
        minimum = min(edge_coverages)
        spread = max(edge_coverages) - minimum
        score = (
            minimum,
            -spread,
            sum(edge_coverages),
            phase0_edge_coverage,
            -origin,
        )
        if best_score is None or score > best_score:
            best_origin = origin
            best_score = score
    return best_origin


@dataclass(frozen=True)
class TilePlan:
    phase: int
    index: int
    grid_core_x0: int
    grid_core_y0: int
    grid_core_x1: int
    grid_core_y1: int
    core_x0: int
    core_y0: int
    core_x1: int
    core_y1: int
    context_x0: int
    context_y0: int
    context_x1: int
    context_y1: int
    source_x0: int
    source_y0: int
    source_x1: int
    source_y1: int
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
    def context_width(self) -> int:
        return self.context_x1 - self.context_x0

    @property
    def context_height(self) -> int:
        return self.context_y1 - self.context_y0

    @property
    def grid_core_width(self) -> int:
        return self.grid_core_x1 - self.grid_core_x0

    @property
    def grid_core_height(self) -> int:
        return self.grid_core_y1 - self.grid_core_y0

    @property
    def source_width(self) -> int:
        return self.source_x1 - self.source_x0

    @property
    def source_height(self) -> int:
        return self.source_y1 - self.source_y0

    @property
    def context_padding(self) -> tuple[int, int, int, int]:
        return (
            self.source_x0 - self.context_x0,
            self.source_y0 - self.context_y0,
            self.context_x1 - self.source_x1,
            self.context_y1 - self.source_y1,
        )

    @property
    def local_core_box(self) -> tuple[int, int, int, int]:
        x0 = self.core_x0 - self.context_x0
        y0 = self.core_y0 - self.context_y0
        return x0, y0, x0 + self.core_width, y0 + self.core_height


def plan_tiles(
    width: int,
    height: int,
    *,
    tile_size: int,
    halo: int,
    core_overlap: int,
    phase_count: int = 1,
    virtual_padding: bool = False,
) -> list[TilePlan]:
    width = _positive_int(width, "canvas width")
    height = _positive_int(height, "canvas height")
    tile_size = _positive_int(tile_size, "tile size")
    phase_count = _positive_int(phase_count, "phase count")
    if halo < 0 or halo * 2 >= tile_size:
        raise ValueError("halo must satisfy 0 <= 2 * halo < tile size.")
    core_size = tile_size - 2 * halo
    if core_overlap < 0 or core_overlap >= core_size:
        raise ValueError("core overlap must satisfy 0 <= overlap < core size.")

    result: list[TilePlan] = []
    tile_index = 0
    for phase in range(phase_count):
        xs = axis_positions(
            width,
            core_size,
            core_overlap,
            phase=phase,
            phase_count=phase_count,
            virtual_padding=virtual_padding,
        )
        ys = axis_positions(
            height,
            core_size,
            core_overlap,
            phase=phase,
            phase_count=phase_count,
            virtual_padding=virtual_padding,
        )
        x_spans = [(max(0, x), min(width, x + core_size)) for x in xs]
        y_spans = [(max(0, y), min(height, y + core_size)) for y in ys]
        for y_index, y in enumerate(ys):
            core_y0, core_y1 = y_spans[y_index]
            previous_y_overlap = (
                0
                if y_index == 0
                else max(
                    0,
                    min(core_y1, y_spans[y_index - 1][1])
                    - max(core_y0, y_spans[y_index - 1][0]),
                )
            )
            next_y_overlap = (
                0
                if y_index == len(ys) - 1
                else max(
                    0,
                    min(core_y1, y_spans[y_index + 1][1])
                    - max(core_y0, y_spans[y_index + 1][0]),
                )
            )
            for x_index, x in enumerate(xs):
                tile_index += 1
                core_x0, core_x1 = x_spans[x_index]
                previous_x_overlap = (
                    0
                    if x_index == 0
                    else max(
                        0,
                        min(core_x1, x_spans[x_index - 1][1])
                        - max(core_x0, x_spans[x_index - 1][0]),
                    )
                )
                next_x_overlap = (
                    0
                    if x_index == len(xs) - 1
                    else max(
                        0,
                        min(core_x1, x_spans[x_index + 1][1])
                        - max(core_x0, x_spans[x_index + 1][0]),
                    )
                )
                if virtual_padding:
                    context_x0 = x - halo
                    context_y0 = y - halo
                    context_x1 = x + core_size + halo
                    context_y1 = y + core_size + halo
                else:
                    context_x0 = max(0, x - halo)
                    context_y0 = max(0, y - halo)
                    context_x1 = min(width, core_x1 + halo)
                    context_y1 = min(height, core_y1 + halo)
                tile = TilePlan(
                    phase=phase,
                    index=tile_index,
                    grid_core_x0=x,
                    grid_core_y0=y,
                    grid_core_x1=x + core_size,
                    grid_core_y1=y + core_size,
                    core_x0=core_x0,
                    core_y0=core_y0,
                    core_x1=core_x1,
                    core_y1=core_y1,
                    context_x0=context_x0,
                    context_y0=context_y0,
                    context_x1=context_x1,
                    context_y1=context_y1,
                    source_x0=max(0, context_x0),
                    source_y0=max(0, context_y0),
                    source_x1=min(width, context_x1),
                    source_y1=min(height, context_y1),
                    previous_x_overlap=min(previous_x_overlap, core_x1 - core_x0),
                    next_x_overlap=min(next_x_overlap, core_x1 - core_x0),
                    previous_y_overlap=min(previous_y_overlap, core_y1 - core_y0),
                    next_y_overlap=min(next_y_overlap, core_y1 - core_y0),
                )
                if tile.core_width <= 0 or tile.core_height <= 0:
                    raise AssertionError("planned core does not intersect the canvas.")
                if tile.source_width <= 0 or tile.source_height <= 0:
                    raise AssertionError("planned context does not intersect the canvas.")
                if tile.context_width > tile_size or tile.context_height > tile_size:
                    raise AssertionError("planned context exceeds the tile payload.")
                if virtual_padding and (
                    tile.context_width != tile_size or tile.context_height != tile_size
                ):
                    raise AssertionError("virtual grid context must equal the tile payload.")
                result.append(tile)
    return result


def extract_tile_context(image: Image.Image, tile: TilePlan) -> Image.Image:
    """Extract one planned context and edge-pad any virtual area outside the canvas."""

    if image.width < tile.source_x1 or image.height < tile.source_y1:
        raise ValueError("tile source rectangle exceeds the supplied image.")
    source = image.crop(
        (tile.source_x0, tile.source_y0, tile.source_x1, tile.source_y1)
    ).convert("RGB")
    left, top, right, bottom = tile.context_padding
    if min(left, top, right, bottom) < 0:
        raise ValueError("tile context padding must be nonnegative.")
    values = np.asarray(source, dtype=np.uint8)
    if left or top or right or bottom:
        values = np.pad(
            values,
            ((top, bottom), (left, right), (0, 0)),
            mode="edge",
        )
    expected = (tile.context_height, tile.context_width, 3)
    if values.shape != expected:
        raise AssertionError(
            f"extracted tile context has shape {values.shape}; expected {expected}."
        )
    return Image.fromarray(values, mode="RGB")


def axis_blend_weights(length: int, previous_overlap: int, next_overlap: int) -> np.ndarray:
    length = _positive_int(length, "blend length")
    weights = np.ones(length, dtype=np.float32)
    previous_overlap = min(max(0, int(previous_overlap)), length)
    next_overlap = min(max(0, int(next_overlap)), length)

    if previous_overlap:
        t = np.linspace(0.0, 1.0, previous_overlap, dtype=np.float32)
        weights[:previous_overlap] = np.minimum(weights[:previous_overlap], t * t * (3.0 - 2.0 * t))
    if next_overlap:
        t = np.linspace(1.0, 0.0, next_overlap, dtype=np.float32)
        weights[-next_overlap:] = np.minimum(weights[-next_overlap:], t * t * (3.0 - 2.0 * t))
    return weights


def tile_weight_mask(tile: TilePlan) -> np.ndarray:
    x_weights = axis_blend_weights(tile.core_width, tile.previous_x_overlap, tile.next_x_overlap)
    y_weights = axis_blend_weights(tile.core_height, tile.previous_y_overlap, tile.next_y_overlap)
    return np.outer(y_weights, x_weights).astype(np.float32)


def phase_weight_normalizers(
    plans: list[TilePlan],
    width: int,
    height: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return separable weight sums that normalize every phase to unit coverage.

    Legacy shifted phases can contain forced end tiles, while virtual grids contain
    clipped edge contributions.  Normalizing each phase before cross-phase fusion
    keeps every covered pixel at unit influence in either layout.  Only O(W + H)
    values are required because tile masks are separable.
    """

    width = _positive_int(width, "canvas width")
    height = _positive_int(height, "canvas height")
    if not plans:
        raise ValueError("at least one tile plan is required.")
    phases = sorted({tile.phase for tile in plans})
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for phase in phases:
        phase_tiles = [tile for tile in plans if tile.phase == phase]
        x_sum = np.zeros(width, dtype=np.float32)
        y_sum = np.zeros(height, dtype=np.float32)
        x_segments = {
            (
                tile.core_x0,
                tile.core_x1,
                tile.previous_x_overlap,
                tile.next_x_overlap,
            )
            for tile in phase_tiles
        }
        y_segments = {
            (
                tile.core_y0,
                tile.core_y1,
                tile.previous_y_overlap,
                tile.next_y_overlap,
            )
            for tile in phase_tiles
        }
        for x0, x1, previous, following in x_segments:
            x_sum[x0:x1] += axis_blend_weights(x1 - x0, previous, following)
        for y0, y1, previous, following in y_segments:
            y_sum[y0:y1] += axis_blend_weights(y1 - y0, previous, following)
        if np.any(x_sum <= 0) or np.any(y_sum <= 0):
            raise ValueError(f"phase {phase} does not cover the complete canvas.")
        result[phase] = (x_sum, y_sum)
    return result


def phase_normalized_tile_weight(
    tile: TilePlan,
    normalizers: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Return a tile mask whose masks sum to one within its grid phase."""

    if tile.phase not in normalizers:
        raise ValueError(f"missing weight normalizer for phase {tile.phase}.")
    x_sum, y_sum = normalizers[tile.phase]
    if tile.core_x1 > x_sum.size or tile.core_y1 > y_sum.size:
        raise ValueError("tile lies outside its phase weight normalizer.")
    denominator = np.outer(
        y_sum[tile.core_y0 : tile.core_y1],
        x_sum[tile.core_x0 : tile.core_x1],
    )
    raw = tile_weight_mask(tile)
    return np.divide(raw, denominator, out=np.zeros_like(raw), where=denominator > 0).astype(np.float32)


def detail_score(rgb: np.ndarray, *, max_analysis_edge: int = 256) -> float:
    """Scale-stable local detail proxy based on absolute luminance Laplacian."""

    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("detail score expects an HxWx3 image.")
    max_analysis_edge = _positive_int(max_analysis_edge, "analysis edge")
    stride = max(1, int(math.ceil(max(values.shape[:2]) / max_analysis_edge)))
    sampled = values[::stride, ::stride].astype(np.float32) / 255.0
    if sampled.shape[0] < 3 or sampled.shape[1] < 3:
        return 0.0
    gray = sampled[..., 0] * 0.2126 + sampled[..., 1] * 0.7152 + sampled[..., 2] * 0.0722
    laplacian = 4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(np.mean(np.abs(laplacian), dtype=np.float64))


def adaptive_step_count(score: float, minimum_steps: int, maximum_steps: int, *, knee: float = 0.035) -> int:
    minimum_steps = _positive_int(minimum_steps, "minimum steps")
    maximum_steps = _positive_int(maximum_steps, "maximum steps")
    knee = _positive(knee, "detail knee")
    if minimum_steps > maximum_steps:
        raise ValueError("minimum steps must be <= maximum steps.")
    if not math.isfinite(float(score)) or score < 0:
        raise ValueError("detail score must be finite and >= 0.")
    importance = float(score) / (float(score) + knee)
    steps = minimum_steps + round((maximum_steps - minimum_steps) * importance)
    return min(maximum_steps, max(minimum_steps, int(steps)))


def _moving_average(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    window = radius * 2 + 1
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate((np.zeros(zero_shape, dtype=np.float64), cumulative), axis=axis)
    high = [slice(None)] * values.ndim
    low = [slice(None)] * values.ndim
    high[axis] = slice(window, None)
    low[axis] = slice(None, -window)
    return ((cumulative[tuple(high)] - cumulative[tuple(low)]) / window).astype(np.float32)


def box_lowpass(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    radius = _positive_int(radius, "low-pass radius")
    if values.ndim not in (2, 3) or not np.all(np.isfinite(values)):
        raise ValueError("box low-pass expects a finite 2D or 3D array.")
    return _moving_average(_moving_average(values, radius, 1), radius, 0)


def gaussian_lowpass(values: np.ndarray, sigma: float) -> np.ndarray:
    """Return a fast three-box approximation of a Gaussian low-pass.

    Three identical box passes have variance ``r(r+1)``.  Choosing ``r`` from the
    requested sigma keeps the implementation dependency-light and stripe friendly
    while avoiding the hard rectangular response of a single box filter.
    """

    data = np.asarray(values, dtype=np.float32)
    if data.ndim not in (2, 3) or not np.all(np.isfinite(data)):
        raise ValueError("Gaussian low-pass expects a finite 2D or 3D array.")
    if not math.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("Gaussian sigma must be finite and >= 0.")
    if float(sigma) == 0:
        return data.copy()
    radius = max(
        1,
        int(round((math.sqrt(1.0 + 4.0 * float(sigma) ** 2) - 1.0) / 2.0)),
    )
    result = data
    for _ in range(3):
        result = box_lowpass(result, radius)
    return result.astype(np.float32, copy=False)


def guided_filter(
    guide: np.ndarray,
    values: np.ndarray,
    *,
    radius: int,
    regularization: float,
) -> np.ndarray:
    """Edge-aware scalar smoothing using the grayscale guided-filter model."""

    guidance = np.asarray(guide, dtype=np.float32)
    source = np.asarray(values, dtype=np.float32)
    if guidance.ndim != 2 or source.shape != guidance.shape:
        raise ValueError("guided filter expects matching 2D guide and value arrays.")
    if not np.all(np.isfinite(guidance)) or not np.all(np.isfinite(source)):
        raise ValueError("guided filter inputs must contain finite values.")
    if isinstance(radius, bool) or int(radius) < 0:
        raise ValueError("guided-filter radius must be >= 0.")
    if not math.isfinite(float(regularization)) or float(regularization) <= 0:
        raise ValueError("guided-filter regularization must be finite and > 0.")
    if int(radius) == 0:
        return source.copy()

    mean_guide = box_lowpass(guidance, int(radius))
    mean_source = box_lowpass(source, int(radius))
    correlation_guide = box_lowpass(guidance * guidance, int(radius))
    correlation_cross = box_lowpass(guidance * source, int(radius))
    variance_guide = np.maximum(
        correlation_guide - mean_guide * mean_guide,
        0.0,
    )
    covariance = correlation_cross - mean_guide * mean_source
    coefficient = covariance / (variance_guide + float(regularization))
    intercept = mean_source - coefficient * mean_guide
    return (
        box_lowpass(coefficient, int(radius)) * guidance
        + box_lowpass(intercept, int(radius))
    ).astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("luminance expects an HxWx3 array.")
    return (
        values[..., 0] * 0.2126
        + values[..., 1] * 0.7152
        + values[..., 2] * 0.0722
    ).astype(np.float32)


def _sobel_gradients(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("Sobel gradients expect a finite 2D array.")
    padded = np.pad(source, ((1, 1), (1, 1)), mode="edge")
    gx = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    ) * 0.25
    gy = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    ) * 0.25
    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return gx.astype(np.float32), gy.astype(np.float32), magnitude


def _local_ssim(first: np.ndarray, second: np.ndarray, radius: int = 3) -> np.ndarray:
    first32 = np.asarray(first, dtype=np.float32)
    second32 = np.asarray(second, dtype=np.float32)
    if first32.ndim != 2 or second32.shape != first32.shape:
        raise ValueError("local SSIM expects matching 2D arrays.")
    mean_first = box_lowpass(first32, int(radius))
    mean_second = box_lowpass(second32, int(radius))
    variance_first = np.maximum(
        box_lowpass(first32 * first32, int(radius)) - mean_first * mean_first,
        0.0,
    )
    variance_second = np.maximum(
        box_lowpass(second32 * second32, int(radius)) - mean_second * mean_second,
        0.0,
    )
    covariance = (
        box_lowpass(first32 * second32, int(radius))
        - mean_first * mean_second
    )
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mean_first * mean_second + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        mean_first * mean_first + mean_second * mean_second + c1
    ) * (variance_first + variance_second + c2)
    return np.clip(
        numerator / np.maximum(denominator, 1e-8),
        -1.0,
        1.0,
    ).astype(np.float32)


def _shift_edge(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    pad_y = abs(int(dy))
    pad_x = abs(int(dx))
    padded = np.pad(
        source,
        ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
        mode="edge",
    )
    start_y = pad_y + int(dy)
    start_x = pad_x + int(dx)
    return padded[
        start_y : start_y + source.shape[0],
        start_x : start_x + source.shape[1],
    ]


def _locally_align_support(
    reference: np.ndarray,
    support: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Align a support residual to a representative with bounded block matching."""

    reference32 = np.asarray(reference, dtype=np.float32)
    support32 = np.asarray(support, dtype=np.float32)
    if reference32.shape != support32.shape or reference32.ndim != 3:
        raise ValueError("local alignment expects matching HxWx3 residual arrays.")
    if isinstance(radius, bool) or int(radius) < 0:
        raise ValueError("local alignment radius must be >= 0.")
    if int(radius) == 0:
        return support32.copy()

    aligned = support32.copy()
    best_cost = np.full(reference32.shape[:2], np.inf, dtype=np.float32)
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            shifted = _shift_edge(support32, dy, dx)
            cost = np.mean(np.abs(reference32 - shifted), axis=2, dtype=np.float32)
            cost = box_lowpass(cost, 2)
            better = cost < best_cost
            best_cost[better] = cost[better]
            aligned[better] = shifted[better]
    return aligned


def _remove_small_label_islands(
    labels: np.ndarray,
    *,
    minimum_area: int,
) -> np.ndarray:
    """Turn compact A/B islands into undecided pixels before propagation.

    Components touching the read stripe border are preserved.  Production callers
    read substantially more context than the largest compact component covered by
    ``minimum_area`` and discard that context afterwards, so this protects only
    components whose extent cannot be determined safely inside the current stripe.
    """

    result = np.asarray(labels, dtype=np.int8).copy()
    if isinstance(minimum_area, bool) or int(minimum_area) < 0:
        raise ValueError("selection island area must be >= 0.")
    if int(minimum_area) == 0:
        return result
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - Forge installs OpenCV.
        raise RuntimeError("PhaseWeave island cleanup requires OpenCV.") from error

    height, width = result.shape
    for phase_label in (0, 1):
        mask = (result == phase_label).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        for component in range(1, int(count)):
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            component_width = int(stats[component, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            area = int(stats[component, cv2.CC_STAT_AREA])
            touches_border = (
                x == 0
                or y == 0
                or x + component_width == width
                or y + component_height == height
            )
            if area < int(minimum_area) and not touches_border:
                result[components == component] = 2
    return result


def _propagate_phase_labels(
    labels: np.ndarray,
    fidelity0: np.ndarray,
    fidelity1: np.ndarray,
    *,
    radius: int,
    confidence: float,
    fidelity_floor: float,
) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int8).copy()
    if int(radius) <= 0:
        return result
    for _ in range(2):
        support0 = box_lowpass((result == 0).astype(np.float32), int(radius))
        support1 = box_lowpass((result == 1).astype(np.float32), int(radius))
        total = support0 + support1
        dominance = np.abs(support1 - support0) / np.maximum(total, 1e-8)
        prefer1 = support1 > support0
        propagated_fidelity = np.where(prefer1, fidelity1, fidelity0)
        fill = (
            (result == 2)
            & (total > 0.02)
            & (dominance >= float(confidence))
            & (propagated_fidelity >= float(fidelity_floor))
        )
        if not np.any(fill):
            break
        result[fill & ~prefer1] = 0
        result[fill & prefer1] = 1
    return result


def _resolve_small_input_islands(
    labels: np.ndarray,
    fidelity0: np.ndarray,
    fidelity1: np.ndarray,
    *,
    minimum_area: int,
    strong_fidelity: float,
) -> np.ndarray:
    """Fill weak compact input holes from their surrounding confident phase.

    A compact rejection is retained when both candidates are strongly unfaithful.
    Otherwise its one-pixel ring votes A or B.  Components touching the expanded
    stripe border are left unchanged because their full extent is not visible.
    """

    result = np.asarray(labels, dtype=np.int8).copy()
    if isinstance(minimum_area, bool) or int(minimum_area) < 0:
        raise ValueError("input-island area must be >= 0.")
    if int(minimum_area) == 0:
        return result
    if not math.isfinite(float(strong_fidelity)) or not 0 <= float(
        strong_fidelity
    ) <= 1:
        raise ValueError("strong rejection fidelity must be between 0 and 1.")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - Forge installs OpenCV.
        raise RuntimeError("PhaseWeave island cleanup requires OpenCV.") from error

    mask = (result == 2).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    height, width = result.shape
    maximum_fidelity = np.maximum(
        np.asarray(fidelity0, dtype=np.float32),
        np.asarray(fidelity1, dtype=np.float32),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    for component in range(1, int(count)):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if area >= int(minimum_area) or touches_border:
            continue
        y0 = max(0, y - 1)
        y1 = min(height, y + component_height + 1)
        x0 = max(0, x - 1)
        x1 = min(width, x + component_width + 1)
        local_components = components[y0:y1, x0:x1]
        component_mask = local_components == component
        local_fidelity = maximum_fidelity[y0:y1, x0:x1]
        if float(np.mean(local_fidelity[component_mask], dtype=np.float64)) < float(
            strong_fidelity
        ):
            continue
        ring = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1).astype(
            bool
        )
        ring &= ~component_mask
        local_labels = result[y0:y1, x0:x1]
        support0 = int(np.count_nonzero(ring & (local_labels == 0)))
        support1 = int(np.count_nonzero(ring & (local_labels == 1)))
        if support0 + support1 == 0:
            continue
        local_labels[component_mask] = 0 if support0 >= support1 else 1
    return result


def frequency_detail_delta(
    refined_rgb: np.ndarray,
    base_rgb: np.ndarray,
    *,
    radius: int = 12,
    gain: float = 1.0,
    max_delta: float = 32.0,
    structure_sigma: float = 18.0,
    base_detail_sigma: float = DEFAULT_BASE_DETAIL_SIGMA,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a gated high-frequency delta instead of an absolute tile image.

    Low frequencies stay anchored to the globally coherent base.  A tile may only
    contribute the high-frequency difference it learned, attenuated where its local
    low-frequency structure diverges from the base image and where the base has no
    corresponding detail energy.  Set ``base_detail_sigma`` to zero to disable the
    flat-region protection.
    """

    refined = np.asarray(refined_rgb, dtype=np.float32)
    base = np.asarray(base_rgb, dtype=np.float32)
    if refined.shape != base.shape or refined.ndim != 3 or refined.shape[2] != 3:
        raise ValueError("refined and base tiles must share an HxWx3 shape.")
    if not np.all(np.isfinite(refined)) or not np.all(np.isfinite(base)):
        raise ValueError("refined and base tiles must contain finite values.")
    radius = _positive_int(radius, "detail radius")
    gain = _positive(gain, "detail gain")
    max_delta = _positive(max_delta, "maximum detail delta")
    structure_sigma = _positive(structure_sigma, "structure sigma")
    if not math.isfinite(float(base_detail_sigma)) or float(base_detail_sigma) < 0:
        raise ValueError("base detail sigma must be finite and >= 0.")
    base_detail_sigma = float(base_detail_sigma)

    refined_low = box_lowpass(refined, radius)
    base_low = box_lowpass(base, radius)
    refined_high = refined - refined_low
    base_high = base - base_low

    refined_luma = refined_low[..., 0] * 0.2126 + refined_low[..., 1] * 0.7152 + refined_low[..., 2] * 0.0722
    base_luma = base_low[..., 0] * 0.2126 + base_low[..., 1] * 0.7152 + base_low[..., 2] * 0.0722
    structure_gate = np.exp(-np.abs(refined_luma - base_luma) / structure_sigma).astype(np.float32)
    if base_detail_sigma > 0:
        base_detail_energy = np.sqrt(np.mean(np.square(base_high), axis=2, dtype=np.float32))
        base_detail_gate = base_detail_energy / (base_detail_energy + base_detail_sigma)
    else:
        base_detail_gate = np.ones(base.shape[:2], dtype=np.float32)
    gate = structure_gate * base_detail_gate
    delta = (refined_high - base_high) * gate[..., None] * gain
    delta = np.clip(delta, -max_delta, max_delta).astype(np.float32)
    stats = {
        "mean_gate": float(np.mean(gate, dtype=np.float64)),
        "mean_structure_gate": float(np.mean(structure_gate, dtype=np.float64)),
        "mean_base_detail_gate": float(np.mean(base_detail_gate, dtype=np.float64)),
        "mean_abs_delta": float(np.mean(np.abs(delta), dtype=np.float64)),
        "clipped_fraction": float(np.mean(np.abs(delta) >= max_delta - 1e-6)),
    }
    return delta, stats


def novel_detail_delta(
    refined_rgb: np.ndarray,
    base_rgb: np.ndarray,
    *,
    inner_radius: int = DEFAULT_NOVEL_DETAIL_INNER_RADIUS,
    outer_radius: int = DEFAULT_NOVEL_DETAIL_OUTER_RADIUS,
    gain: float = 1.0,
    max_delta: float = 8.0,
    structure_sigma: float = DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA,
    base_detail_sigma: float = DEFAULT_BASE_DETAIL_SIGMA,
) -> tuple[np.ndarray, dict[str, float]]:
    """Propose bounded luminance microdetail where the source is locally sparse.

    This is the complementary branch to :func:`frequency_detail_delta`.  It does not
    make an acceptance decision by itself: callers must accumulate at least two
    independently shifted phases and run a strict second-moment consensus gate.  The
    source-detail complement keeps this branch focused on otherwise smooth material
    regions, while the low-frequency structure gate rejects geometric or tonal drift.
    Only a 2-8 px luminance band is returned so the branch cannot directly repaint
    color, silhouette, or large facial structure.
    """

    refined = np.asarray(refined_rgb, dtype=np.float32)
    base = np.asarray(base_rgb, dtype=np.float32)
    if refined.shape != base.shape or refined.ndim != 3 or refined.shape[2] != 3:
        raise ValueError("refined and base tiles must share an HxWx3 shape.")
    if not np.all(np.isfinite(refined)) or not np.all(np.isfinite(base)):
        raise ValueError("refined and base tiles must contain finite values.")
    inner_radius = _positive_int(inner_radius, "novel detail inner radius")
    outer_radius = _positive_int(outer_radius, "novel detail outer radius")
    if inner_radius >= outer_radius:
        raise ValueError("novel detail inner radius must be smaller than outer radius.")
    gain = _positive(gain, "novel detail gain")
    max_delta = _positive(max_delta, "novel detail maximum delta")
    structure_sigma = _positive(
        structure_sigma, "novel detail structure sigma"
    )
    base_detail_sigma = _positive(
        base_detail_sigma, "novel detail base sigma"
    )

    refined_inner = box_lowpass(refined, inner_radius)
    refined_outer = box_lowpass(refined, outer_radius)
    base_inner = box_lowpass(base, inner_radius)
    base_outer = box_lowpass(base, outer_radius)
    refined_band = refined_inner - refined_outer
    base_band = base_inner - base_outer

    refined_luma = (
        refined_outer[..., 0] * 0.2126
        + refined_outer[..., 1] * 0.7152
        + refined_outer[..., 2] * 0.0722
    )
    base_luma = (
        base_outer[..., 0] * 0.2126
        + base_outer[..., 1] * 0.7152
        + base_outer[..., 2] * 0.0722
    )
    structure_gate = np.exp(
        -np.abs(refined_luma - base_luma) / structure_sigma
    ).astype(np.float32)
    base_detail_energy = np.sqrt(
        np.mean(np.square(base_band), axis=2, dtype=np.float32)
    )
    novelty_gate = base_detail_sigma / (
        base_detail_energy + base_detail_sigma
    )

    band_difference = refined_band - base_band
    luminance_difference = (
        band_difference[..., 0] * 0.2126
        + band_difference[..., 1] * 0.7152
        + band_difference[..., 2] * 0.0722
    )
    gate = structure_gate * novelty_gate
    luminance_delta = np.clip(
        luminance_difference * gate * gain,
        -max_delta,
        max_delta,
    ).astype(np.float32)
    delta = np.repeat(luminance_delta[..., None], 3, axis=2)
    stats = {
        "mean_gate": float(np.mean(gate, dtype=np.float64)),
        "mean_structure_gate": float(
            np.mean(structure_gate, dtype=np.float64)
        ),
        "mean_novelty_gate": float(np.mean(novelty_gate, dtype=np.float64)),
        "mean_abs_delta": float(
            np.mean(np.abs(luminance_delta), dtype=np.float64)
        ),
        "clipped_fraction": float(
            np.mean(np.abs(luminance_delta) >= max_delta - 1e-6)
        ),
    }
    return delta, stats


def consensus_gated_residual(
    delta_sum: np.ndarray,
    weight_sum: np.ndarray,
    energy_sum: np.ndarray,
    *,
    sigma: float = 8.0,
    strength: float = 4.0,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize overlapping residuals and suppress tile disagreement.

    ``delta_sum`` stores the weighted first moment ``sum(w_i * delta_i)``.
    ``energy_sum`` stores the weighted scalar second moment
    ``sum(w_i * mean(delta_i ** 2, channels))``.  Their centered difference is
    the population variance across overlapping tile predictions.  The returned
    residual uses a scale-aware gate
    ``exp(-strength * variance / (mean_energy + sigma ** 2))``.  Strong but
    mutually consistent detail is preserved, while directional disagreement is
    attenuated.  The result is independent of tile accumulation order, apart from
    floating-point summation noise.

    A pixel covered by only one tile has zero disagreement and passes unchanged.
    Setting ``sigma`` to zero disables the consensus attenuation while retaining
    the weighted normalization.
    """

    deltas = np.asarray(delta_sum, dtype=np.float32)
    weights = np.asarray(weight_sum, dtype=np.float32)
    energies = np.asarray(energy_sum, dtype=np.float32)
    if deltas.ndim != 3 or deltas.shape[2] != 3:
        raise ValueError("delta sum must have an HxWx3 shape.")
    if weights.shape != deltas.shape[:2] or energies.shape != deltas.shape[:2]:
        raise ValueError("weight and energy sums must match the delta spatial shape.")
    if not np.all(np.isfinite(deltas)) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(energies)):
        raise ValueError("consensus moments must contain finite values.")
    if np.any(weights < 0) or np.any(energies < 0):
        raise ValueError("consensus weights and energies must be non-negative.")
    if not math.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("consensus sigma must be finite and >= 0.")
    if not math.isfinite(float(strength)) or float(strength) < 0:
        raise ValueError("consensus strength must be finite and >= 0.")
    epsilon = _positive(epsilon, "consensus epsilon")

    covered = weights > epsilon
    deltas64 = deltas.astype(np.float64)
    weights64 = weights.astype(np.float64)
    energies64 = energies.astype(np.float64)
    mean64 = np.divide(
        deltas64,
        weights64[..., None],
        out=np.zeros_like(deltas64),
        where=covered[..., None],
    )
    mean_energy64 = np.divide(
        energies64,
        weights64,
        out=np.zeros_like(energies64),
        where=covered,
    )
    squared_mean64 = np.mean(np.square(mean64), axis=2)
    variance64 = np.maximum(mean_energy64 - squared_mean64, 0.0)
    standard_deviation = np.sqrt(variance64).astype(np.float32)
    if float(sigma) == 0 or float(strength) == 0:
        confidence = covered.astype(np.float32)
    else:
        relative_disagreement = variance64 / (mean_energy64 + float(sigma) ** 2)
        confidence = np.exp(-float(strength) * relative_disagreement).astype(np.float32)
        confidence[~covered] = 0.0
    return (mean64 * confidence[..., None]).astype(np.float32), confidence, standard_deviation


def _normalized_residual_moments(
    delta_sum: np.ndarray,
    weight_sum: np.ndarray,
    energy_sum: np.ndarray,
    *,
    sigma: float,
    strength: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return an unattenuated residual plus coverage, confidence, and deviation."""

    deltas = np.asarray(delta_sum, dtype=np.float32)
    weights = np.asarray(weight_sum, dtype=np.float32)
    energies = np.asarray(energy_sum, dtype=np.float32)
    if deltas.ndim != 3 or deltas.shape[2] != 3:
        raise ValueError("phase delta sum must have an HxWx3 shape.")
    if weights.shape != deltas.shape[:2] or energies.shape != deltas.shape[:2]:
        raise ValueError("phase weight and energy sums must match the delta shape.")
    if (
        not np.all(np.isfinite(deltas))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(energies))
    ):
        raise ValueError("phase moments must contain finite values.")
    if np.any(weights < 0) or np.any(energies < 0):
        raise ValueError("phase weights and energies must be non-negative.")

    covered = weights > epsilon
    deltas64 = deltas.astype(np.float64)
    weights64 = weights.astype(np.float64)
    energies64 = energies.astype(np.float64)
    mean64 = np.divide(
        deltas64,
        weights64[..., None],
        out=np.zeros_like(deltas64),
        where=covered[..., None],
    )
    mean_energy64 = np.divide(
        energies64,
        weights64,
        out=np.zeros_like(energies64),
        where=covered,
    )
    variance64 = np.maximum(
        mean_energy64 - np.mean(np.square(mean64), axis=2),
        0.0,
    )
    deviation = np.sqrt(variance64).astype(np.float32)
    if float(sigma) == 0 or float(strength) == 0:
        confidence = covered.astype(np.float32)
    else:
        relative_disagreement = variance64 / (
            mean_energy64 + float(sigma) ** 2
        )
        confidence = np.exp(
            -float(strength) * relative_disagreement
        ).astype(np.float32)
        confidence[~covered] = 0.0
    return mean64.astype(np.float32), covered, confidence, deviation


def phase_weave_residual(
    phase0_delta_sum: np.ndarray,
    phase0_weight_sum: np.ndarray,
    phase0_energy_sum: np.ndarray,
    phase1_delta_sum: np.ndarray,
    phase1_weight_sum: np.ndarray,
    phase1_energy_sum: np.ndarray,
    *,
    base_rgb: np.ndarray,
    sigma: float = 8.0,
    strength: float = 4.0,
    detail_floor: float = PHASE_WEAVE_DETAIL_FLOOR,
    support_mix: float = PHASE_WEAVE_SUPPORT_MIX,
    support_confidence_power: float = PHASE_WEAVE_SUPPORT_CONFIDENCE_POWER,
    quality_radius: int = PHASE_WEAVE_QUALITY_RADIUS,
    feather_radius: int = PHASE_WEAVE_FEATHER_RADIUS,
    selection_confidence_floor: float = PHASE_WEAVE_SELECTION_CONFIDENCE_FLOOR,
    selection_margin: float = PHASE_WEAVE_SELECTION_MARGIN,
    guided_epsilon: float = PHASE_WEAVE_GUIDED_EPSILON,
    propagation_radius: int = PHASE_WEAVE_PROPAGATION_RADIUS,
    propagation_confidence: float = PHASE_WEAVE_PROPAGATION_CONFIDENCE,
    island_min_area: int = PHASE_WEAVE_ISLAND_MIN_AREA,
    input_island_min_area: int = PHASE_WEAVE_INPUT_ISLAND_MIN_AREA,
    strong_rejection_ratio: float = PHASE_WEAVE_STRONG_REJECTION_RATIO,
    fidelity_radius: int = PHASE_WEAVE_FIDELITY_RADIUS,
    low_frequency_sigma: float = PHASE_WEAVE_LOW_FREQUENCY_SIGMA,
    low_frequency_luma_gain: float = PHASE_WEAVE_LOW_FREQUENCY_LUMA_GAIN,
    low_frequency_chroma_gain: float = PHASE_WEAVE_LOW_FREQUENCY_CHROMA_GAIN,
    highlight_threshold: float = PHASE_WEAVE_HIGHLIGHT_THRESHOLD,
    highlight_low_frequency_scale: float = PHASE_WEAVE_HIGHLIGHT_LOW_FREQUENCY_SCALE,
    fidelity_reject_threshold: float = PHASE_WEAVE_FIDELITY_REJECT_THRESHOLD,
    edge_fidelity_weight: float = PHASE_WEAVE_EDGE_FIDELITY_WEIGHT,
    low_fidelity_weight: float = PHASE_WEAVE_LOW_FIDELITY_WEIGHT,
    chroma_fidelity_weight: float = PHASE_WEAVE_CHROMA_FIDELITY_WEIGHT,
    close_rms: float = PHASE_WEAVE_CLOSE_RMS,
    close_ssim: float = PHASE_WEAVE_CLOSE_SSIM,
    support_ssim: float = PHASE_WEAVE_SUPPORT_SSIM,
    alignment_radius: int = PHASE_WEAVE_ALIGNMENT_RADIUS,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select A, B, or the unchanged input from two independently completed phases.

    Selection is deliberately conservative.  Residual low frequencies are attenuated,
    quality measures high-frequency evidence multiplied by source fidelity, and an
    edge-guided score creates confident A/B seeds with a symmetric dead band.  Labels
    are cleaned and propagated from confident neighbours.  Remaining low-fidelity
    pixels reject both candidates and retain the input; sufficiently close candidates
    receive a locally aligned, weak support blend.  Only cleaned label boundaries are
    feathered.

    ``selected_phase`` uses 0 for A, 1 for B, 2 for input rejection, 3 for an
    aligned near-tie, and -1 for uncovered pixels.
    """

    if not math.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("PhaseWeave sigma must be finite and >= 0.")
    if not math.isfinite(float(strength)) or float(strength) < 0:
        raise ValueError("PhaseWeave strength must be finite and >= 0.")
    if not math.isfinite(float(detail_floor)) or not 0 < float(detail_floor) <= 1:
        raise ValueError("PhaseWeave detail floor must satisfy 0 < value <= 1.")
    if not math.isfinite(float(support_mix)) or not 0 <= float(support_mix) <= 1:
        raise ValueError("PhaseWeave support mix must be between 0 and 1.")
    if (
        not math.isfinite(float(support_confidence_power))
        or float(support_confidence_power) < 1
    ):
        raise ValueError("PhaseWeave support confidence power must be >= 1.")
    if (
        not math.isfinite(float(selection_confidence_floor))
        or not 0 <= float(selection_confidence_floor) <= 1
    ):
        raise ValueError(
            "PhaseWeave selection confidence floor must be between 0 and 1."
        )
    if (
        not math.isfinite(float(selection_margin))
        or not 0 <= float(selection_margin) < 1
    ):
        raise ValueError("PhaseWeave selection margin must satisfy 0 <= value < 1.")
    if isinstance(quality_radius, bool) or int(quality_radius) < 0:
        raise ValueError("PhaseWeave quality radius must be >= 0.")
    if isinstance(feather_radius, bool) or int(feather_radius) < 0:
        raise ValueError("PhaseWeave feather radius must be >= 0.")
    if isinstance(propagation_radius, bool) or int(propagation_radius) < 0:
        raise ValueError("PhaseWeave propagation radius must be >= 0.")
    if isinstance(island_min_area, bool) or int(island_min_area) < 0:
        raise ValueError("PhaseWeave island area must be >= 0.")
    if isinstance(input_island_min_area, bool) or int(input_island_min_area) < 0:
        raise ValueError("PhaseWeave input-island area must be >= 0.")
    if isinstance(fidelity_radius, bool) or int(fidelity_radius) < 0:
        raise ValueError("PhaseWeave fidelity radius must be >= 0.")
    if isinstance(alignment_radius, bool) or int(alignment_radius) < 0:
        raise ValueError("PhaseWeave alignment radius must be >= 0.")
    bounded_values = (
        (guided_epsilon, "guided epsilon", False),
        (propagation_confidence, "propagation confidence", True),
        (low_frequency_luma_gain, "low-frequency luma gain", True),
        (low_frequency_chroma_gain, "low-frequency chroma gain", True),
        (highlight_threshold, "highlight threshold", True),
        (highlight_low_frequency_scale, "highlight low-frequency scale", True),
        (fidelity_reject_threshold, "fidelity rejection threshold", True),
        (close_ssim, "close SSIM", True),
        (support_ssim, "support SSIM", True),
        (strong_rejection_ratio, "strong rejection ratio", True),
    )
    for value, name, unit_interval in bounded_values:
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"PhaseWeave {name} must be finite and >= 0.")
        if unit_interval and float(value) > 1:
            raise ValueError(f"PhaseWeave {name} must be <= 1.")
    if float(guided_epsilon) <= 0:
        raise ValueError("PhaseWeave guided epsilon must be > 0.")
    for value, name in (
        (low_frequency_sigma, "low-frequency sigma"),
        (edge_fidelity_weight, "edge fidelity weight"),
        (low_fidelity_weight, "low-frequency fidelity weight"),
        (chroma_fidelity_weight, "chroma fidelity weight"),
        (close_rms, "close RMS"),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"PhaseWeave {name} must be finite and >= 0.")
    epsilon = _positive(epsilon, "PhaseWeave epsilon")

    phase0, covered0, confidence0, deviation0 = _normalized_residual_moments(
        phase0_delta_sum,
        phase0_weight_sum,
        phase0_energy_sum,
        sigma=float(sigma),
        strength=float(strength),
        epsilon=epsilon,
    )
    phase1, covered1, confidence1, deviation1 = _normalized_residual_moments(
        phase1_delta_sum,
        phase1_weight_sum,
        phase1_energy_sum,
        sigma=float(sigma),
        strength=float(strength),
        epsilon=epsilon,
    )
    if phase0.shape != phase1.shape:
        raise ValueError("PhaseWeave phase moments must have identical shapes.")
    base = np.asarray(base_rgb, dtype=np.float32)
    if base.shape != phase0.shape or not np.all(np.isfinite(base)):
        raise ValueError(
            "PhaseWeave base image must be finite and match the residual shape."
        )
    base_luma = _luminance(base)

    def anchored_residual(
        residual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        low = gaussian_lowpass(residual, float(low_frequency_sigma))
        high = residual - low
        low_luma = _luminance(low)
        low_chroma = low - low_luma[..., None]
        luma_gain = np.full(
            low_luma.shape,
            float(low_frequency_luma_gain),
            dtype=np.float32,
        )
        luma_gain[base_luma >= float(highlight_threshold) * 255.0] *= float(
            highlight_low_frequency_scale
        )
        anchored = (
            high
            + low_luma[..., None] * luma_gain[..., None]
            + low_chroma * float(low_frequency_chroma_gain)
        )
        candidate = np.clip(base + anchored, 0.0, 255.0).astype(np.float32)
        return (
            anchored.astype(np.float32),
            high.astype(np.float32),
            low_luma.astype(np.float32),
            low_chroma.astype(np.float32),
            candidate,
        )

    phase0, high0, low_luma0, low_chroma0, candidate0 = anchored_residual(
        phase0
    )
    phase1, high1, low_luma1, low_chroma1, candidate1 = anchored_residual(
        phase1
    )
    candidate_luma0 = _luminance(candidate0)
    candidate_luma1 = _luminance(candidate1)
    base_gx, base_gy, base_gradient = _sobel_gradients(base_luma)

    def fidelity_and_detail(
        candidate_luma: np.ndarray,
        high: np.ndarray,
        low_luma: np.ndarray,
        low_chroma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_gx, candidate_gy, candidate_gradient = _sobel_gradients(
            candidate_luma
        )
        direction = (
            base_gx * candidate_gx + base_gy * candidate_gy
        ) / np.maximum(base_gradient * candidate_gradient, 1e-6)
        direction_penalty = 0.5 * (1.0 - np.clip(direction, -1.0, 1.0))
        base_edge_weight = base_gradient / (base_gradient + 8.0)
        invented_edge = np.clip(
            (candidate_gradient - base_gradient)
            / (candidate_gradient + base_gradient + 8.0),
            0.0,
            1.0,
        )
        edge_distance = (
            base_edge_weight * direction_penalty
            + (1.0 - base_edge_weight) * invented_edge
        )
        low_distance = np.abs(low_luma) / 12.0
        chroma_distance = np.sqrt(
            np.mean(np.square(low_chroma), axis=2, dtype=np.float32)
        ) / 10.0
        fidelity = np.exp(
            -float(edge_fidelity_weight) * edge_distance
            -float(low_fidelity_weight) * low_distance
            -float(chroma_fidelity_weight) * chroma_distance
        ).astype(np.float32)
        high_detail = np.sqrt(
            np.mean(np.square(high), axis=2, dtype=np.float32)
        ).astype(np.float32)
        return fidelity, high_detail

    fidelity0, magnitude0 = fidelity_and_detail(
        candidate_luma0,
        high0,
        low_luma0,
        low_chroma0,
    )
    fidelity1, magnitude1 = fidelity_and_detail(
        candidate_luma1,
        high1,
        low_luma1,
        low_chroma1,
    )
    if int(fidelity_radius) > 0:
        fidelity_guide = np.clip(base_luma / 255.0, 0.0, 1.0)
        fidelity0 = np.clip(
            guided_filter(
                fidelity_guide,
                fidelity0,
                radius=int(fidelity_radius),
                regularization=float(guided_epsilon),
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        fidelity1 = np.clip(
            guided_filter(
                fidelity_guide,
                fidelity1,
                radius=int(fidelity_radius),
                regularization=float(guided_epsilon),
            ),
            0.0,
            1.0,
        ).astype(np.float32)
    confidence_floor = float(selection_confidence_floor)
    quality0 = magnitude0 * (
        confidence_floor + (1.0 - confidence_floor) * confidence0
    ) * fidelity0
    quality1 = magnitude1 * (
        confidence_floor + (1.0 - confidence_floor) * confidence1
    ) * fidelity1
    if int(quality_radius) > 0:
        quality0 = box_lowpass(quality0, int(quality_radius))
        quality1 = box_lowpass(quality1, int(quality_radius))

    selection_score = (quality1 - quality0) / (
        quality0 + quality1 + float(epsilon)
    )
    covered = covered0 | covered1
    both_covered = covered0 & covered1
    if int(quality_radius) > 0:
        selection_score = guided_filter(
            np.clip(base_luma / 255.0, 0.0, 1.0),
            selection_score,
            radius=int(quality_radius),
            regularization=float(guided_epsilon),
        )

    selected_phase = np.full(covered.shape, 2, dtype=np.int8)
    selected_phase[
        both_covered
        & (selection_score < -float(selection_margin))
        & (fidelity0 >= float(fidelity_reject_threshold))
    ] = 0
    selected_phase[
        both_covered
        & (selection_score > float(selection_margin))
        & (fidelity1 >= float(fidelity_reject_threshold))
    ] = 1
    selected_phase[covered0 & ~covered1] = 0
    selected_phase[covered1 & ~covered0] = 1
    selected_phase[~covered] = -1

    selected_phase = _remove_small_label_islands(
        selected_phase,
        minimum_area=int(island_min_area),
    )
    selected_phase = _propagate_phase_labels(
        selected_phase,
        fidelity0,
        fidelity1,
        radius=int(propagation_radius),
        confidence=float(propagation_confidence),
        fidelity_floor=float(fidelity_reject_threshold),
    )
    selected_phase = _remove_small_label_islands(
        selected_phase,
        minimum_area=int(island_min_area),
    )
    selected_phase = _propagate_phase_labels(
        selected_phase,
        fidelity0,
        fidelity1,
        radius=max(1, int(propagation_radius) // 2)
        if int(propagation_radius) > 0
        else 0,
        confidence=float(propagation_confidence),
        fidelity_floor=float(fidelity_reject_threshold),
    )

    near_tie = selected_phase == 2
    candidate_ssim = _local_ssim(candidate_luma0, candidate_luma1, radius=3)
    residual_difference_rms = np.sqrt(
        box_lowpass(
            np.mean(np.square(phase0 - phase1), axis=2, dtype=np.float32),
            4,
        )
    ).astype(np.float32)
    close_candidates = (
        near_tie
        & (np.maximum(fidelity0, fidelity1) >= float(fidelity_reject_threshold))
        & (residual_difference_rms <= float(close_rms))
        & (candidate_ssim >= float(close_ssim))
    )
    selected_phase[close_candidates] = 3

    near_tie = selected_phase == 2
    decisively_faithful = (
        near_tie
        & (np.maximum(fidelity0, fidelity1) >= max(
            float(fidelity_reject_threshold),
            0.55,
        ))
        & (np.abs(fidelity1 - fidelity0) >= 0.08)
    )
    selected_phase[decisively_faithful & (fidelity0 >= fidelity1)] = 0
    selected_phase[decisively_faithful & (fidelity1 > fidelity0)] = 1
    selected_phase = _resolve_small_input_islands(
        selected_phase,
        fidelity0,
        fidelity1,
        minimum_area=int(input_island_min_area),
        strong_fidelity=float(fidelity_reject_threshold)
        * float(strong_rejection_ratio),
    )
    selected_phase = _remove_small_label_islands(
        selected_phase,
        minimum_area=int(island_min_area),
    )
    selected_phase = _propagate_phase_labels(
        selected_phase,
        fidelity0,
        fidelity1,
        radius=max(1, int(propagation_radius) // 2)
        if int(propagation_radius) > 0
        else 0,
        confidence=float(propagation_confidence),
        fidelity_floor=float(fidelity_reject_threshold),
    )
    selected_phase = _resolve_small_input_islands(
        selected_phase,
        fidelity0,
        fidelity1,
        minimum_area=int(input_island_min_area),
        strong_fidelity=float(fidelity_reject_threshold)
        * float(strong_rejection_ratio),
    )
    selected_phase[~covered] = -1

    fusion_prefers1 = quality1 > quality0
    effective_phase0 = (selected_phase == 0) | (
        (selected_phase == 3) & ~fusion_prefers1
    )
    effective_phase1 = (selected_phase == 1) | (
        (selected_phase == 3) & fusion_prefers1
    )
    input_rejected = selected_phase == 2
    phase0_weight = effective_phase0.astype(np.float32)
    phase1_weight = effective_phase1.astype(np.float32)
    input_weight = input_rejected.astype(np.float32)
    if int(feather_radius) > 0:
        phase0_weight = box_lowpass(phase0_weight, int(feather_radius))
        phase1_weight = box_lowpass(phase1_weight, int(feather_radius))
        input_weight = box_lowpass(input_weight, int(feather_radius))
        phase0_weight = phase0_weight * phase0_weight * (
            3.0 - 2.0 * phase0_weight
        )
        phase1_weight = phase1_weight * phase1_weight * (
            3.0 - 2.0 * phase1_weight
        )
        input_weight = input_weight * input_weight * (3.0 - 2.0 * input_weight)
    total_label_weight = phase0_weight + phase1_weight + input_weight
    phase0_weight = np.divide(
        phase0_weight,
        total_label_weight,
        out=np.zeros_like(phase0_weight),
        where=total_label_weight > epsilon,
    )
    phase1_weight = np.divide(
        phase1_weight,
        total_label_weight,
        out=np.zeros_like(phase1_weight),
        where=total_label_weight > epsilon,
    )
    input_weight = np.divide(
        input_weight,
        total_label_weight,
        out=np.zeros_like(input_weight),
        where=total_label_weight > epsilon,
    )
    phase0_weight[covered0 & ~covered1] = 1.0
    phase1_weight[covered0 & ~covered1] = 0.0
    input_weight[covered0 & ~covered1] = 0.0
    phase0_weight[covered1 & ~covered0] = 0.0
    phase1_weight[covered1 & ~covered0] = 1.0
    input_weight[covered1 & ~covered0] = 0.0
    phase0_weight[~covered] = 0.0
    phase1_weight[~covered] = 0.0
    input_weight[~covered] = 0.0

    representative_weight = phase0_weight + phase1_weight
    phase1_mix = np.divide(
        phase1_weight,
        representative_weight,
        out=np.zeros_like(phase1_weight),
        where=representative_weight > epsilon,
    )
    mix = phase1_mix[..., None]
    representative = phase0 * (1.0 - mix) + phase1 * mix

    aligned1_to0 = _locally_align_support(
        phase0,
        phase1,
        radius=int(alignment_radius),
    )
    aligned0_to1 = _locally_align_support(
        phase1,
        phase0,
        radius=int(alignment_radius),
    )
    other = aligned1_to0 * (1.0 - mix) + aligned0_to1 * mix
    difference0 = phase0 - aligned1_to0
    difference1 = phase1 - aligned0_to1
    difference_energy = (
        np.mean(np.square(difference0), axis=2, dtype=np.float32)
        * (1.0 - phase1_mix)
        + np.mean(np.square(difference1), axis=2, dtype=np.float32)
        * phase1_mix
    )
    reference_energy = (
        0.5
        * (
            np.mean(np.square(phase0), axis=2, dtype=np.float32)
            + np.mean(np.square(aligned1_to0), axis=2, dtype=np.float32)
        )
        * (1.0 - phase1_mix)
        + 0.5
        * (
            np.mean(np.square(phase1), axis=2, dtype=np.float32)
            + np.mean(np.square(aligned0_to1), axis=2, dtype=np.float32)
        )
        * phase1_mix
    )
    if float(sigma) == 0 or float(strength) == 0:
        support_confidence = both_covered.astype(np.float32)
    else:
        relative_disagreement = difference_energy / (
            reference_energy + float(sigma) ** 2
        )
        support_confidence = np.exp(
            -float(strength) * relative_disagreement
        ).astype(np.float32)
        support_confidence[~both_covered] = 0.0

    aligned_candidate1_luma = _luminance(np.clip(base + aligned1_to0, 0.0, 255.0))
    aligned_candidate0_luma = _luminance(np.clip(base + aligned0_to1, 0.0, 255.0))
    ssim0 = _local_ssim(candidate_luma0, aligned_candidate1_luma, radius=3)
    ssim1 = _local_ssim(candidate_luma1, aligned_candidate0_luma, radius=3)

    def gradient_agreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_gx, first_gy, first_magnitude = _sobel_gradients(first)
        second_gx, second_gy, second_magnitude = _sobel_gradients(second)
        return np.clip(
            (first_gx * second_gx + first_gy * second_gy)
            / np.maximum(first_magnitude * second_magnitude, 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)

    structure0 = (
        np.clip(
            (ssim0 - float(support_ssim))
            / max(1.0 - float(support_ssim), 1e-6),
            0.0,
            1.0,
        )
        * gradient_agreement(candidate_luma0, aligned_candidate1_luma)
    )
    structure1 = (
        np.clip(
            (ssim1 - float(support_ssim))
            / max(1.0 - float(support_ssim), 1e-6),
            0.0,
            1.0,
        )
        * gradient_agreement(candidate_luma1, aligned_candidate0_luma)
    )
    support_structure = (
        structure0 * (1.0 - phase1_mix) + structure1 * phase1_mix
    ).astype(np.float32)
    support_confidence *= support_structure
    support_confidence[~both_covered] = 0.0
    support_weight = float(support_mix) * np.power(
        support_confidence,
        float(support_confidence_power),
    )
    supported = representative + support_weight[..., None] * (
        other - representative
    )
    confidence_gain = (
        float(detail_floor)
        + (1.0 - float(detail_floor)) * support_confidence
    )
    confidence_gain[covered & ~both_covered] = 1.0
    confidence_gain[~covered] = 0.0
    woven = (
        supported
        * confidence_gain[..., None]
        * representative_weight[..., None]
    )
    dominant_label_weight = np.maximum.reduce(
        (phase0_weight, phase1_weight, input_weight)
    )
    boundary = covered & (dominant_label_weight < 1.0 - 1e-4)
    selected_confidence = confidence0 * (1.0 - phase1_mix) + confidence1 * phase1_mix
    selected_deviation = deviation0 * (1.0 - phase1_mix) + deviation1 * phase1_mix
    selected_fidelity = fidelity0 * (1.0 - phase1_mix) + fidelity1 * phase1_mix
    return woven.astype(np.float32), {
        "covered": covered,
        "both_covered": both_covered,
        "selected_phase": selected_phase,
        "selection_score": selection_score.astype(np.float32),
        "phase1_mix": phase1_mix.astype(np.float32),
        "input_mix": input_weight.astype(np.float32),
        "input_rejected": input_rejected,
        "uncertain_fused": selected_phase == 3,
        "boundary": boundary,
        "support_confidence": support_confidence,
        "support_structure": support_structure,
        "support_weight": support_weight.astype(np.float32),
        "cross_disagreement": np.sqrt(difference_energy).astype(np.float32),
        "selected_intra_confidence": selected_confidence.astype(np.float32),
        "selected_intra_disagreement": selected_deviation.astype(np.float32),
        "confidence_gain": confidence_gain.astype(np.float32),
        "fidelity0": fidelity0,
        "fidelity1": fidelity1,
        "both_unfaithful": both_covered
        & (np.maximum(fidelity0, fidelity1) < float(fidelity_reject_threshold)),
        "selected_fidelity": selected_fidelity.astype(np.float32),
        "low_frequency_luma_gain": np.where(
            base_luma >= float(highlight_threshold) * 255.0,
            float(low_frequency_luma_gain)
            * float(highlight_low_frequency_scale),
            float(low_frequency_luma_gain),
        ).astype(np.float32),
        "candidate_difference_rms": residual_difference_rms,
        "phase0_residual": phase0,
        "phase1_residual": phase1,
    }


def vram_canvas_work_bytes_per_pixel(
    *,
    phase_count: int,
    merge_mode: str,
    novel_detail: bool,
) -> int:
    """Conservative disk preflight cost for all stage-backed accumulators."""

    phase_count = _positive_int(phase_count, "phase count")
    merge_mode = str(merge_mode)
    if merge_mode not in (CONSENSUS_MERGE_MODE, PHASE_WEAVE_MERGE_MODE):
        raise ValueError(f"unknown VRAM-Canvas merge mode: {merge_mode}")
    if merge_mode == PHASE_WEAVE_MERGE_MODE and phase_count != 2:
        raise ValueError("PhaseWeave requires exactly two grid phases.")
    moment_sets = phase_count if merge_mode == PHASE_WEAVE_MERGE_MODE else 1
    base_bytes = BASE_WORK_BYTES_PER_PIXEL + (
        moment_sets - 1
    ) * MOMENT_WORK_BYTES_PER_PIXEL
    novel_bytes = (
        moment_sets * NOVEL_MOMENT_WORK_BYTES_PER_PIXEL if novel_detail else 0
    )
    return int(base_bytes + novel_bytes)


def coordinate_seed(global_seed: int, phase: int, x: int, y: int) -> int:
    """Derive a reproducible uint32 seed without repeating one texture per tile."""

    for value, name in ((global_seed, "global seed"), (phase, "phase")):
        if int(value) < 0:
            raise ValueError(f"{name} must be >= 0.")
    mask = (1 << 64) - 1
    value = (int(global_seed) ^ (int(phase) * 0x9E3779B97F4A7C15) ^ (int(x) * 0xBF58476D1CE4E5B9) ^ (int(y) * 0x94D049BB133111EB)) & mask
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return int(value & 0xFFFFFFFF)


def spatial_activation_reduction(target_width: int, target_height: int, tile_size: int) -> float:
    target_width = _positive_int(target_width, "target width")
    target_height = _positive_int(target_height, "target height")
    tile_size = _positive_int(tile_size, "tile size")
    return (target_width * target_height) / float(tile_size * tile_size)


def replace_infotext_seed(infotext: str, seed: int) -> str:
    """Replace only the settings-line seed, leaving prompt literals untouched."""

    if not infotext:
        return infotext
    if int(seed) < 0:
        raise ValueError("seed must be >= 0.")
    lines = infotext.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].lstrip().startswith("Steps:"):
            continue
        lines[index], _ = re.subn(
            r"(?<!\w)Seed:\s*-?\d+(?=,|$)",
            f"Seed: {int(seed)}",
            lines[index],
            count=1,
        )
        break
    return "\n".join(lines)
