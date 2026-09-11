"""Configuration, presets, prompts, and validation for HyperWeave."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image


HYPERWEAVE_VERSION = "1.2.0"
LATENT_ALIGNMENT = 8


class HyperWeavePreset(StrEnum):
    STRUCTURE_SAFE = "Structure Safe"
    OVERDRAW = "Overdraw"
    MAX_OVERDRAW = "Max Overdraw"
    CUSTOM = "Custom"


class TargetMode(StrEnum):
    X2 = "x2"
    X4 = "x4"
    LONG_EDGE_4K = "4K long edge"
    LONG_EDGE_8K = "8K long edge"
    CUSTOM_LONG_EDGE = "Custom long edge"
    CUSTOM_SIZE = "Custom width and height"


class ContentProfile(StrEnum):
    AUTO = "Auto"
    ILLUSTRATION = "Illustration / Anime"
    PHOTO = "Photo"
    RENDER = "3D / Render"


class AccumulatorMode(StrEnum):
    AUTO = "Auto"
    MEMORY = "RAM"
    MEMMAP = "Disk-backed memmap"


@dataclass(frozen=True)
class FrequencyGains:
    high_0: float
    high_1: float
    mid_high: float
    mid: float
    mid_low: float
    low: float
    chroma_ratio: float = 0.35

    def scaled(self, amount: float) -> "FrequencyGains":
        return FrequencyGains(
            high_0=self.high_0 * amount,
            high_1=self.high_1 * amount,
            mid_high=self.mid_high * amount,
            mid=self.mid * amount,
            mid_low=self.mid_low * amount,
            low=self.low * amount,
            chroma_ratio=self.chroma_ratio,
        )


@dataclass(frozen=True)
class PresetValues:
    anchor_strength: float
    global_overdraw_strength: float
    face_strength: float
    hair_strength: float
    material_strength: float
    micro_strength: float
    structural_lock: float
    low_frequency_lock: float
    overdraw_amount: float
    global_candidates: int
    face_candidates: int
    hair_candidates: int
    material_candidates: int
    flat_region_detail: float
    frequency_gains: FrequencyGains


PRESETS: dict[HyperWeavePreset, PresetValues] = {
    HyperWeavePreset.STRUCTURE_SAFE: PresetValues(
        anchor_strength=0.12,
        global_overdraw_strength=0.24,
        face_strength=0.22,
        hair_strength=0.30,
        material_strength=0.30,
        micro_strength=0.14,
        structural_lock=0.90,
        low_frequency_lock=1.00,
        overdraw_amount=0.75,
        global_candidates=1,
        face_candidates=4,
        hair_candidates=3,
        material_candidates=1,
        flat_region_detail=0.15,
        frequency_gains=FrequencyGains(0.70, 0.80, 0.80, 0.55, 0.25, 0.00),
    ),
    HyperWeavePreset.OVERDRAW: PresetValues(
        anchor_strength=0.15,
        global_overdraw_strength=0.34,
        face_strength=0.30,
        hair_strength=0.40,
        material_strength=0.42,
        micro_strength=0.20,
        structural_lock=0.78,
        low_frequency_lock=0.96,
        overdraw_amount=1.00,
        global_candidates=2,
        face_candidates=6,
        hair_candidates=4,
        material_candidates=2,
        flat_region_detail=0.35,
        frequency_gains=FrequencyGains(0.90, 1.05, 1.15, 0.95, 0.65, 0.00),
    ),
    HyperWeavePreset.MAX_OVERDRAW: PresetValues(
        anchor_strength=0.18,
        global_overdraw_strength=0.42,
        face_strength=0.36,
        hair_strength=0.48,
        material_strength=0.50,
        micro_strength=0.24,
        structural_lock=0.68,
        low_frequency_lock=0.92,
        overdraw_amount=1.30,
        global_candidates=2,
        face_candidates=8,
        hair_candidates=6,
        material_candidates=2,
        flat_region_detail=0.55,
        frequency_gains=FrequencyGains(1.00, 1.15, 1.30, 1.15, 0.90, 0.05),
    ),
}


COMMON_STRUCTURE_SUFFIX = (
    "preserve the exact composition, pose, character identity, facial expression, "
    "gaze direction, face angle, hairstyle silhouette, clothing design, object "
    "placement, color palette and original art style"
)
ANCHOR_SUFFIX = (
    "coherent high-resolution redraw, clean consistent forms, stable anatomy, stable "
    "facial placement, stable hairstyle silhouette, refined but restrained details"
)
GLOBAL_OVERDRAW_SUFFIX = (
    "extremely dense coherent hand-drawn detail, rich mid-frequency structure, fine "
    "linework, layered material detail, subtle secondary forms, refined shading, "
    "high-resolution illustration detail, preserve all original semantic structure"
)
ILLUSTRATION_FACE_SUFFIX = (
    "same character, same expression, same gaze, same head angle, refined eyelids, "
    "delicate eyelashes, coherent iris structure, natural eye highlights, precise "
    "mouth and nose shading, clean high-resolution anime facial drawing, preserve the "
    "original facial proportions and hairstyle boundary"
)
PHOTO_FACE_SUFFIX = (
    "same person, same expression, same gaze, same head angle, natural skin detail, "
    "eyelashes, iris detail, subtle facial microcontrast, preserve identity, facial "
    "proportions and lighting"
)
HAIR_SUFFIX = (
    "preserve the exact hairstyle silhouette, parting and major hair clumps, add "
    "layered coherent hair strands, internal strand groups, fine flyaway hairs and "
    "directional highlights following the existing hair flow, no random crossing strands"
)
MATERIAL_SUFFIX = (
    "rich style-consistent fabric weave, stitching, folds, fine ornament, metal "
    "reflections, glass reflections, paper fibers, wood and stone microstructure, "
    "preserve the exact object shapes and original material identity"
)
MICRO_SUFFIX = (
    "crisp natural micro-detail, refined line edges, subtle high-resolution texture, "
    "fine reflections and clean anti-aliasing, no noise, no oversharpening"
)
NEGATIVE_SUFFIX = (
    "changed composition, changed identity, changed expression, changed gaze, changed "
    "face angle, changed hairstyle silhouette, different clothing, new objects, "
    "duplicated objects, extra eyes, duplicated facial features, double mouth, "
    "malformed face, extra fingers, broken anatomy, crossing random hair, wire-like "
    "hair, duplicated outlines, tile seams, halo, ringing, checkerboard, random "
    "texture noise, oversharpening, color drift"
)


PASS_SUFFIXES = {
    "anchor": ANCHOR_SUFFIX,
    "global": GLOBAL_OVERDRAW_SUFFIX,
    "face_illustration": ILLUSTRATION_FACE_SUFFIX,
    "face_photo": PHOTO_FACE_SUFFIX,
    "hair": HAIR_SUFFIX,
    "material": MATERIAL_SUFFIX,
    "micro": MICRO_SUFFIX,
}


@dataclass
class HyperWeaveConfig:
    enabled: bool = True
    target_mode: TargetMode = TargetMode.LONG_EDGE_4K
    custom_long_edge: int = 4096
    custom_width: int = 0
    custom_height: int = 0
    preset: HyperWeavePreset = HyperWeavePreset.OVERDRAW
    content_profile: ContentProfile = ContentProfile.AUTO
    seed: int = -1
    exact_steps: int = 6

    overdraw_amount: float = 1.00
    structural_lock: float = 0.78
    low_frequency_lock: float = 0.96
    anchor_strength: float = 0.15
    global_overdraw_strength: float = 0.34
    face_strength: float = 0.30
    hair_strength: float = 0.40
    material_strength: float = 0.42
    micro_strength: float = 0.20

    tile_input_size: int = 1280
    core_size: int = 960
    context_size: int = 160
    stride: int = 768
    latent_alignment: int = LATENT_ALIGNMENT
    accumulator_mode: AccumulatorMode = AccumulatorMode.AUTO
    temp_directory: str = ""
    maximum_ram_gib: float = 8.0

    global_candidates: int = 2
    face_candidates: int = 6
    hair_candidates: int = 4
    material_candidates: int = 2
    roi_final_pass_count: int = 1

    enable_face_redraw: bool = True
    enable_hair_redraw: bool = True
    enable_material_redraw: bool = True
    enable_micro_pass: bool = True
    detector_provider: str = "Auto (local only)"
    detector_model_path: str = ""
    minimum_face_size: int = 12
    maximum_face_count: int = 12
    identity_reference: Image.Image | None = field(default=None, repr=False)
    structure_conditioner: str = "None"
    protection_mask: Image.Image | None = field(default=None, repr=False)
    boost_mask: Image.Image | None = field(default=None, repr=False)
    manual_face_mask: Image.Image | None = field(default=None, repr=False)
    mask_channel: str = "Luminance"
    boost_strength: float = 0.75

    flat_region_detail: float = 0.35
    face_structure_tolerance: float = 0.20
    hair_flow_tolerance: float = 0.30
    new_edge_tolerance: float = 0.20
    color_drift_tolerance: float = 0.08
    candidate_rejection_strictness: float = 0.70
    candidate_score_margin: float = 0.02
    enable_spatial_rescue: bool = True
    spatial_decision_size: int = 480
    spatial_transition_width: int = 48
    spatial_score_margin: float = 0.05
    spatial_fragmentation_limit: float = 0.45
    spatial_minimum_component_cells: int = 2
    roi_stages: str = "Last two stages"
    back_projection_iterations: int = 2
    back_projection_beta: float = 0.70

    append_prompt_suffixes: bool = True
    common_suffix: str = COMMON_STRUCTURE_SUFFIX
    anchor_suffix: str = ANCHOR_SUFFIX
    global_suffix: str = GLOBAL_OVERDRAW_SUFFIX
    face_illustration_suffix: str = ILLUSTRATION_FACE_SUFFIX
    face_photo_suffix: str = PHOTO_FACE_SUFFIX
    hair_suffix: str = HAIR_SUFFIX
    material_suffix: str = MATERIAL_SUFFIX
    micro_suffix: str = MICRO_SUFFIX
    negative_suffix: str = NEGATIVE_SUFFIX

    save_debug_images: bool = False
    save_all_candidates: bool = False
    save_maps: bool = False
    save_roi_crops: bool = False
    save_metrics_json: bool = True
    save_metrics_csv: bool = False
    debug_output_directory: str = ""

    model_background: str = "Auto edge color"
    share_anchor_noise_family: bool = True
    oom_retry_smaller_tile: bool = True

    @classmethod
    def from_preset(
        cls,
        preset: HyperWeavePreset | str = HyperWeavePreset.OVERDRAW,
        **overrides: Any,
    ) -> "HyperWeaveConfig":
        selected = HyperWeavePreset(preset)
        base = PRESETS.get(selected, PRESETS[HyperWeavePreset.OVERDRAW])
        values: dict[str, Any] = {
            "preset": selected,
            "anchor_strength": base.anchor_strength,
            "global_overdraw_strength": base.global_overdraw_strength,
            "face_strength": base.face_strength,
            "hair_strength": base.hair_strength,
            "material_strength": base.material_strength,
            "micro_strength": base.micro_strength,
            "structural_lock": base.structural_lock,
            "low_frequency_lock": base.low_frequency_lock,
            "overdraw_amount": base.overdraw_amount,
            "global_candidates": base.global_candidates,
            "face_candidates": base.face_candidates,
            "hair_candidates": base.hair_candidates,
            "material_candidates": base.material_candidates,
            "flat_region_detail": base.flat_region_detail,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def frequency_gains(self) -> FrequencyGains:
        preset = self.preset
        if preset == HyperWeavePreset.CUSTOM:
            preset = HyperWeavePreset.OVERDRAW
        return PRESETS[preset].frequency_gains.scaled(self.overdraw_amount)

    def validate(self, source_size: tuple[int, int] | None = None) -> None:
        if self.exact_steps < 1:
            raise ValueError("Exact Steps must be at least 1.")
        if self.tile_input_size != self.core_size + 2 * self.context_size:
            raise ValueError(
                "Tile input size must equal core size + 2 × context size."
            )
        if self.stride > self.core_size:
            raise ValueError("Tile stride cannot exceed core size.")
        if self.stride < 1 or self.core_size < 1 or self.context_size < 0:
            raise ValueError("Tile dimensions must be positive.")
        for name in ("tile_input_size", "core_size", "context_size", "stride"):
            if getattr(self, name) % self.latent_alignment:
                raise ValueError(
                    f"{name.replace('_', ' ').title()} must be divisible by "
                    f"latent alignment {self.latent_alignment}."
                )
        if not 0.0 <= self.structural_lock <= 1.0:
            raise ValueError("Structural Lock must be between 0 and 1.")
        if not 0.0 <= self.low_frequency_lock <= 1.0:
            raise ValueError("Low Frequency Lock must be between 0 and 1.")
        if not 0.0 <= self.overdraw_amount <= 2.0:
            raise ValueError("Overdraw Amount must be between 0 and 2.")
        if min(
            self.global_candidates,
            self.face_candidates,
            self.hair_candidates,
            self.material_candidates,
        ) < 1:
            raise ValueError("Candidate counts must be at least 1.")
        if self.spatial_decision_size < 32:
            raise ValueError("Spatial decision size must be at least 32 pixels.")
        if self.spatial_transition_width < 1:
            raise ValueError("Spatial transition width must be positive.")
        if self.spatial_transition_width * 2 > self.spatial_decision_size:
            raise ValueError(
                "Spatial transition width cannot exceed half the decision size."
            )
        if self.spatial_score_margin < 0:
            raise ValueError("Spatial score margin cannot be negative.")
        if self.candidate_score_margin < 0:
            raise ValueError("Candidate score margin cannot be negative.")
        if not 0.0 <= self.spatial_fragmentation_limit <= 1.0:
            raise ValueError("Spatial fragmentation limit must be between 0 and 1.")
        if self.spatial_minimum_component_cells < 1:
            raise ValueError(
                "Spatial minimum component cells must be at least 1."
            )
        if source_size is not None:
            target = resolve_target_size(source_size, self)
            if target[0] <= source_size[0] and target[1] <= source_size[1]:
                raise ValueError(
                    f"HyperWeave target {target[0]}x{target[1]} must be larger "
                    f"than input {source_size[0]}x{source_size[1]}."
                )
        if self.temp_directory:
            root = Path(self.temp_directory).expanduser()
            if root.exists() and not root.is_dir():
                raise ValueError("Temp directory points to a file.")

    def metadata_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "identity_reference",
            "protection_mask",
            "boost_mask",
            "manual_face_mask",
        ):
            image = getattr(self, key)
            data[key] = None if image is None else {
                "size": list(image.size),
                "mode": image.mode,
            }
        data["frequency_gains"] = asdict(self.frequency_gains)
        data["version"] = HYPERWEAVE_VERSION
        return data


def resolve_target_size(
    source_size: tuple[int, int], config: HyperWeaveConfig
) -> tuple[int, int]:
    width, height = source_size
    if width < 1 or height < 1:
        raise ValueError("Input image dimensions must be positive.")

    mode = TargetMode(config.target_mode)
    if mode == TargetMode.X2:
        return width * 2, height * 2
    if mode == TargetMode.X4:
        return width * 4, height * 4
    if mode == TargetMode.CUSTOM_SIZE:
        if config.custom_width < 1 or config.custom_height < 1:
            raise ValueError("Custom width and height must both be positive.")
        return int(config.custom_width), int(config.custom_height)

    long_edge = {
        TargetMode.LONG_EDGE_4K: 4096,
        TargetMode.LONG_EDGE_8K: 8192,
        TargetMode.CUSTOM_LONG_EDGE: int(config.custom_long_edge),
    }[mode]
    if long_edge < 1:
        raise ValueError("Custom long edge must be positive.")
    scale = long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def pass_prompt_suffix(config: HyperWeaveConfig, pass_name: str) -> str:
    if not config.append_prompt_suffixes:
        return ""
    suffixes = {
        "anchor": config.anchor_suffix,
        "global": config.global_suffix,
        "face_illustration": config.face_illustration_suffix,
        "face_photo": config.face_photo_suffix,
        "hair": config.hair_suffix,
        "material": config.material_suffix,
        "micro": config.micro_suffix,
    }
    parts = [config.common_suffix, suffixes.get(pass_name, "")]
    return ", ".join(part.strip(" ,") for part in parts if part.strip(" ,"))
