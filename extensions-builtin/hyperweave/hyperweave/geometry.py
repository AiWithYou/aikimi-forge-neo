"""Stage and overlap-tile planning with exact coverage guarantees."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np

from .config import HyperWeaveConfig


Size = tuple[int, int]
Box = tuple[int, int, int, int]


def align_up(value: int, alignment: int) -> int:
    return max(alignment, int(ceil(value / alignment) * alignment))


@dataclass(frozen=True)
class StageSpec:
    index: int
    source_size: Size
    target_size: Size
    processing_size: Size
    scale_x: float
    scale_y: float
    anchor_strength: float
    overdraw_strength: float
    face_strength: float
    hair_strength: float
    material_strength: float
    micro_strength: float
    frequency_gains: dict[str, float]
    run_face_pass: bool
    run_hair_pass: bool
    run_material_pass: bool
    source_to_stage: tuple[float, float]


def _next_aspect_size(current: Size, final: Size, max_scale: float) -> Size:
    width, height = current
    final_w, final_h = final
    remaining = max(final_w / width, final_h / height)
    if remaining <= max_scale + 1e-9:
        return final
    scale = max_scale
    return min(final_w, round(width * scale)), min(final_h, round(height * scale))


def plan_upscale_stages(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    max_scale_per_stage: float = 2.0,
    alignment: int = 8,
    config: HyperWeaveConfig | None = None,
) -> list[StageSpec]:
    if min(source_width, source_height, target_width, target_height) < 1:
        raise ValueError("Stage dimensions must be positive.")
    if target_width <= source_width and target_height <= source_height:
        raise ValueError("Target must be larger than source.")
    if not 1.0 < max_scale_per_stage <= 2.0:
        raise ValueError("Maximum stage scale must be greater than 1 and at most 2.")

    cfg = config or HyperWeaveConfig()
    exact_sizes: list[Size] = []
    current = (source_width, source_height)
    final = (target_width, target_height)
    while current != final:
        next_size = _next_aspect_size(current, final, max_scale_per_stage)
        if next_size == current:
            raise RuntimeError("Stage planner made no progress.")
        exact_sizes.append(next_size)
        current = next_size

    count = len(exact_sizes)
    stages: list[StageSpec] = []
    previous = (source_width, source_height)
    for zero_index, target in enumerate(exact_sizes):
        late_scale = 0.82 if zero_index == count - 1 and count > 1 else 1.0
        run_roi = zero_index == count - 1 or (
            count >= 3 and zero_index >= count - 2
        )
        if cfg.roi_stages == "Final stage only":
            run_roi = zero_index == count - 1
        elif cfg.roi_stages == "Every stage":
            run_roi = True
        gains = cfg.frequency_gains
        mid_low_factor = 0.82 if zero_index == count - 1 and count > 1 else 1.0
        process_size = (
            align_up(target[0], alignment),
            align_up(target[1], alignment),
        )
        stages.append(
            StageSpec(
                index=zero_index,
                source_size=previous,
                target_size=target,
                processing_size=process_size,
                scale_x=target[0] / previous[0],
                scale_y=target[1] / previous[1],
                anchor_strength=cfg.anchor_strength * late_scale,
                overdraw_strength=cfg.global_overdraw_strength * late_scale,
                face_strength=cfg.face_strength * late_scale,
                hair_strength=cfg.hair_strength * late_scale,
                material_strength=cfg.material_strength * late_scale,
                micro_strength=cfg.micro_strength * late_scale,
                frequency_gains={
                    "high_0": gains.high_0,
                    "high_1": gains.high_1,
                    "mid_high": gains.mid_high,
                    "mid": gains.mid,
                    "mid_low": gains.mid_low * mid_low_factor,
                    "low": gains.low,
                    "chroma_ratio": gains.chroma_ratio,
                },
                run_face_pass=run_roi and cfg.enable_face_redraw,
                run_hair_pass=run_roi and cfg.enable_hair_redraw,
                run_material_pass=run_roi and cfg.enable_material_redraw,
                source_to_stage=(
                    target[0] / source_width,
                    target[1] / source_height,
                ),
            )
        )
        previous = target
    return stages


@dataclass(frozen=True)
class TileSpec:
    index: int
    row: int
    column: int
    canvas_size: Size
    grid_core_box: Box
    core_box: Box
    input_box: Box
    local_core_box: Box
    padding: tuple[int, int, int, int]
    touches_left: bool
    touches_top: bool
    touches_right: bool
    touches_bottom: bool

    @property
    def core_width(self) -> int:
        return self.core_box[2] - self.core_box[0]

    @property
    def core_height(self) -> int:
        return self.core_box[3] - self.core_box[1]


def _axis_positions(length: int, core: int, stride: int, alignment: int) -> list[int]:
    if length <= core:
        return [0]
    last = length - core
    if last % alignment:
        raise ValueError(
            f"Aligned processing length {length} leaves an unaligned final tile origin."
        )
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


class TilePlanner:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        tile_input_size: int = 1280,
        core_size: int = 960,
        context_size: int = 160,
        stride: int = 768,
        alignment: int = 8,
    ):
        if width < 1 or height < 1:
            raise ValueError("Canvas dimensions must be positive.")
        if tile_input_size != core_size + 2 * context_size:
            raise ValueError("tile_input_size must equal core_size + 2*context_size.")
        if stride > core_size:
            raise ValueError("stride cannot exceed core_size.")
        if any(value % alignment for value in (core_size, context_size, stride)):
            raise ValueError("Tile geometry must be latent-aligned.")
        if width % alignment or height % alignment:
            raise ValueError("Processing canvas must be latent-aligned.")
        self.width = width
        self.height = height
        self.tile_input_size = tile_input_size
        self.core_size = core_size
        self.context_size = context_size
        self.stride = stride
        self.alignment = alignment

    def plan(self) -> list[TileSpec]:
        xs = _axis_positions(
            self.width, self.core_size, self.stride, self.alignment
        )
        ys = _axis_positions(
            self.height, self.core_size, self.stride, self.alignment
        )
        result: list[TileSpec] = []
        index = 0
        for row, grid_y0 in enumerate(ys):
            for column, grid_x0 in enumerate(xs):
                grid_x1 = grid_x0 + self.core_size
                grid_y1 = grid_y0 + self.core_size
                core_x0 = max(0, grid_x0)
                core_y0 = max(0, grid_y0)
                core_x1 = min(self.width, grid_x1)
                core_y1 = min(self.height, grid_y1)
                input_box = (
                    grid_x0 - self.context_size,
                    grid_y0 - self.context_size,
                    grid_x1 + self.context_size,
                    grid_y1 + self.context_size,
                )
                pad_left = max(0, -input_box[0])
                pad_top = max(0, -input_box[1])
                pad_right = max(0, input_box[2] - self.width)
                pad_bottom = max(0, input_box[3] - self.height)
                local_x0 = self.context_size + (core_x0 - grid_x0)
                local_y0 = self.context_size + (core_y0 - grid_y0)
                result.append(
                    TileSpec(
                        index=index,
                        row=row,
                        column=column,
                        canvas_size=(self.width, self.height),
                        grid_core_box=(grid_x0, grid_y0, grid_x1, grid_y1),
                        core_box=(core_x0, core_y0, core_x1, core_y1),
                        input_box=input_box,
                        local_core_box=(
                            local_x0,
                            local_y0,
                            local_x0 + core_x1 - core_x0,
                            local_y0 + core_y1 - core_y0,
                        ),
                        padding=(pad_left, pad_top, pad_right, pad_bottom),
                        touches_left=core_x0 == 0,
                        touches_top=core_y0 == 0,
                        touches_right=core_x1 == self.width,
                        touches_bottom=core_y1 == self.height,
                    )
                )
                index += 1
        return result

    def coverage(self, tiles: Iterable[TileSpec] | None = None) -> np.ndarray:
        coverage = np.zeros((self.height, self.width), dtype=np.int32)
        for tile in tiles or self.plan():
            x0, y0, x1, y1 = tile.core_box
            coverage[y0:y1, x0:x1] += 1
        return coverage

    @staticmethod
    def weight_window(tile: TileSpec, taper: float = 0.5) -> np.ndarray:
        height, width = tile.core_height, tile.core_width

        def axis_window(
            length: int, starts_at_edge: bool, ends_at_edge: bool
        ) -> np.ndarray:
            if length <= 1:
                return np.ones(length, dtype=np.float32)
            edge = max(1, round(length * taper * 0.5))
            values = np.ones(length, dtype=np.float32)
            ramp = 0.5 - 0.5 * np.cos(
                np.linspace(0.0, np.pi, edge, dtype=np.float32)
            )
            if not starts_at_edge:
                values[:edge] = np.maximum(ramp, 1e-4)
            if not ends_at_edge:
                values[-edge:] = np.maximum(ramp[::-1], 1e-4)
            return values

        wx = axis_window(width, tile.touches_left, tile.touches_right)
        wy = axis_window(height, tile.touches_top, tile.touches_bottom)
        return wy[:, None] * wx[None, :]
