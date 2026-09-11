"""Deterministic image-coordinate latent noise for overlapping diffusion tiles."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

import numpy as np

from .geometry import TileSpec


def resolve_seed(seed: int) -> int:
    if int(seed) == -1:
        return secrets.randbits(32)
    return int(seed) & ((1 << 63) - 1)


def derive_seed(
    base_seed: int,
    stage_index: int,
    pass_name: str,
    candidate_index: int,
    roi_id: int = -1,
) -> int:
    payload = (
        f"hyperweave-noise-v1\0{int(base_seed)}\0{int(stage_index)}\0"
        f"{pass_name}\0{int(candidate_index)}\0{int(roi_id)}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _safe_pad_mode(height: int, width: int) -> str:
    return "reflect" if height > 1 and width > 1 else "edge"


@dataclass
class CoordinateNoiseProvider:
    base_seed: int
    _resolved_seed: int = field(init=False)
    _canvases: dict[tuple[object, ...], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._resolved_seed = resolve_seed(self.base_seed)

    @property
    def resolved_seed(self) -> int:
        return self._resolved_seed

    def clear(self) -> None:
        self._canvases.clear()

    def canvas(
        self,
        *,
        stage_index: int,
        pass_name: str,
        candidate_index: int,
        latent_width: int,
        latent_height: int,
        latent_channels: int,
        dtype: np.dtype | type[np.floating] = np.float32,
        roi_id: int = -1,
        family_pass_name: str | None = None,
    ) -> np.ndarray:
        if min(latent_width, latent_height, latent_channels) < 1:
            raise ValueError("Noise canvas dimensions must be positive.")
        dtype = np.dtype(dtype)
        family = family_pass_name or pass_name
        key = (
            stage_index,
            family,
            candidate_index,
            latent_width,
            latent_height,
            latent_channels,
            dtype.str,
            roi_id,
        )
        if key not in self._canvases:
            seed = derive_seed(
                self._resolved_seed,
                stage_index,
                family,
                candidate_index,
                roi_id,
            )
            generator = np.random.Generator(np.random.PCG64(seed))
            values = generator.standard_normal(
                (latent_channels, latent_height, latent_width), dtype=np.float32
            )
            self._canvases[key] = values.astype(dtype, copy=False)
        return self._canvases[key]

    def crop_for_tile(
        self,
        tile: TileSpec,
        *,
        stage_index: int,
        pass_name: str,
        candidate_index: int,
        latent_channels: int,
        latent_scale: int = 8,
        dtype: np.dtype | type[np.floating] = np.float32,
        roi_id: int = -1,
        family_pass_name: str | None = None,
    ) -> np.ndarray:
        width, height = tile.canvas_size
        if width % latent_scale or height % latent_scale:
            raise ValueError("Noise canvas must be aligned to latent scale.")
        if any(value % latent_scale for value in tile.input_box):
            raise ValueError("Tile input coordinates must be latent-aligned.")
        full = self.canvas(
            stage_index=stage_index,
            pass_name=pass_name,
            candidate_index=candidate_index,
            latent_width=width // latent_scale,
            latent_height=height // latent_scale,
            latent_channels=latent_channels,
            dtype=dtype,
            roi_id=roi_id,
            family_pass_name=family_pass_name,
        )
        x0, y0, x1, y1 = (value // latent_scale for value in tile.input_box)
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - full.shape[2])
        pad_bottom = max(0, y1 - full.shape[1])
        padded = np.pad(
            full,
            (
                (0, 0),
                (pad_top, pad_bottom),
                (pad_left, pad_right),
            ),
            mode=_safe_pad_mode(full.shape[1], full.shape[2]),
        )
        crop_x0 = x0 + pad_left
        crop_y0 = y0 + pad_top
        result = padded[
            :,
            crop_y0 : crop_y0 + (y1 - y0),
            crop_x0 : crop_x0 + (x1 - x0),
        ]
        expected = (
            latent_channels,
            (tile.input_box[3] - tile.input_box[1]) // latent_scale,
            (tile.input_box[2] - tile.input_box[0]) // latent_scale,
        )
        if result.shape != expected:
            raise RuntimeError(
                f"Coordinate noise crop is {result.shape}; expected {expected}."
            )
        return np.ascontiguousarray(result)


def coordinate_noise_crop(
    provider: CoordinateNoiseProvider,
    *,
    stage_index: int,
    pass_name: str,
    candidate_index: int,
    latent_channels: int,
    latent_scale: int,
    canvas_size: tuple[int, int],
    input_box: tuple[int, int, int, int],
    roi_id: int = -1,
) -> np.ndarray:
    """Convenience helper for non-tile ROI requests."""
    x0, y0, x1, y1 = input_box
    tile = TileSpec(
        index=0,
        row=0,
        column=0,
        canvas_size=canvas_size,
        grid_core_box=input_box,
        core_box=(
            max(0, x0),
            max(0, y0),
            min(canvas_size[0], x1),
            min(canvas_size[1], y1),
        ),
        input_box=input_box,
        local_core_box=(0, 0, x1 - x0, y1 - y0),
        padding=(0, 0, 0, 0),
        touches_left=x0 <= 0,
        touches_top=y0 <= 0,
        touches_right=x1 >= canvas_size[0],
        touches_bottom=y1 >= canvas_size[1],
    )
    return provider.crop_for_tile(
        tile,
        stage_index=stage_index,
        pass_name=pass_name,
        candidate_index=candidate_index,
        latent_channels=latent_channels,
        latent_scale=latent_scale,
        roi_id=roi_id,
    )
