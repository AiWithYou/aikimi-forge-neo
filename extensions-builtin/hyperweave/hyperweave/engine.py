"""HyperWeave staged generative-redraw engine."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import psutil
from PIL import Image

from .accumulator import (
    AccumulatorBackend,
    InMemoryAccumulator,
    MemmapAccumulator,
    accumulator_bytes,
)
from .analysis import SourceAnalyzer
from .color import (
    image_to_linear_rgb,
    linear_rgb_to_image,
)
from .config import (
    AccumulatorMode,
    ContentProfile,
    HYPERWEAVE_VERSION,
    FrequencyGains,
    HyperWeaveConfig,
    pass_prompt_suffix,
    resolve_target_size,
)
from .debug import DebugWriter
from .frequency import (
    ComposeMaps,
    FrequencyAwareComposer,
    StructureMapBuilder,
    gaussian_blur,
    new_edge_confidence,
    orientation_confidence,
    robust_normalize,
)
from .generator import GenerationRequest, GeneratorAdapter
from .geometry import StageSpec, TilePlanner, plan_upscale_stages
from .noise import CoordinateNoiseProvider, derive_seed
from .quality import (
    LowFrequencyBackProjection,
    SeamAnalyzer,
    roundtrip_metrics,
    resize_float,
)
from .regions import (
    default_face_candidate_count,
    detail_potential_map,
    expand_face_roi,
    face_core_mask,
    face_core_union_mask,
    face_region_masks,
    hair_flow_score,
    hair_region_mask,
    local_feather_mask,
)
from .scoring import CandidateScore, CandidateScorer
from .scoring_features import SCORING_VERSION
from .spatial_selection import SpatialResidualSelector


logger = logging.getLogger("hyperweave")
logger.setLevel(logging.INFO)


class HyperWeaveInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class ProgressEvent:
    stage_index: int
    stage_count: int
    phase: str
    current: int = 0
    total: int = 0
    message: str = ""


@dataclass
class HyperWeaveResult:
    image: Image.Image
    resolved_seed: int
    metadata: dict[str, object]
    metrics: dict[str, object]
    messages: list[str]
    debug_files: list[Path] = field(default_factory=list)
    last_processed: object | None = None


def _extract_reflect(
    array: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = box
    height, width = array.shape[:2]
    clipped = array[
        max(0, y0) : min(height, y1),
        max(0, x0) : min(width, x1),
    ]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)
    mode = "reflect" if clipped.shape[0] > 1 and clipped.shape[1] > 1 else "edge"
    padding = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
        *(([(0, 0)] if clipped.ndim == 3 else [])),
    )
    result = np.pad(clipped, padding, mode=mode)
    expected = (y1 - y0, x1 - x0)
    if result.shape[:2] != expected:
        raise RuntimeError(f"Padded tile is {result.shape[:2]}; expected {expected}.")
    return np.ascontiguousarray(result)


def _candidate_gains(stage: StageSpec) -> FrequencyGains:
    values = stage.frequency_gains
    return FrequencyGains(
        high_0=values["high_0"],
        high_1=values["high_1"],
        mid_high=values["mid_high"],
        mid=values["mid"],
        mid_low=values["mid_low"],
        low=values["low"],
        chroma_ratio=values["chroma_ratio"],
    )


def _resize_with_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    size: tuple[int, int],
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    if alpha is None:
        return resize_float(rgb, size), None
    resized_alpha = np.clip(resize_float(alpha, size), 0.0, 1.0)
    resized_premultiplied = resize_float(rgb * alpha[..., None], size)
    safe_alpha = np.maximum(resized_alpha[..., None], 1e-6)
    resized_rgb = np.where(
        resized_alpha[..., None] > 1e-5,
        resized_premultiplied / safe_alpha,
        background[None, None, :],
    )
    return np.clip(resized_rgb, 0.0, 1.0), resized_alpha


def _flatten_score(score: CandidateScore) -> dict[str, object]:
    result = score.to_dict()
    roundtrip = result.pop("roundtrip")
    result.update({f"roundtrip_{key}": value for key, value in roundtrip.items()})
    result["rejection_reasons"] = "; ".join(score.rejection_reasons)
    return result


def _frequency_debug_maps(
    anchor: np.ndarray, candidate: np.ndarray
) -> dict[str, np.ndarray]:
    bands = FrequencyAwareComposer().decomposer.decompose(candidate - anchor)

    def magnitude(*names: str) -> np.ndarray:
        combined = np.sum([bands[name] for name in names], axis=0)
        values = np.mean(np.abs(combined), axis=2)
        return robust_normalize(values, 20.0, 99.0).astype(np.float32)

    return {
        "high": magnitude("high_0", "high_1"),
        "mid": magnitude("mid_high", "mid"),
        "midlow": magnitude("mid_low", "low"),
    }


def _roundtrip_confidence_map(
    reference: np.ndarray,
    candidate: np.ndarray,
    scalar_confidence: float,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if (
        reference.ndim != 3
        or candidate.ndim != 3
        or reference.shape[2] != 3
        or candidate.shape[2] != 3
    ):
        raise ValueError("Round-trip confidence inputs must be RGB arrays.")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Round-trip confidence input contains NaN or Inf.")
    if not np.isfinite(scalar_confidence):
        raise ValueError("Round-trip scalar confidence is NaN or Inf.")
    reference_size = (reference.shape[1], reference.shape[0])
    downsampled = resize_float(candidate, reference_size)
    reference_low = gaussian_blur(reference, 2.0)
    candidate_low = gaussian_blur(downsampled, 2.0)
    local_error = np.mean(np.abs(reference_low - candidate_low), axis=2)
    local_confidence = np.exp(-12.0 * local_error).astype(np.float32)
    resized = resize_float(
        local_confidence, (candidate.shape[1], candidate.shape[0])
    )
    result = resized * float(np.clip(scalar_confidence, 0.0, 1.0))
    if result.shape != candidate.shape[:2] or not np.isfinite(result).all():
        raise RuntimeError("Local round-trip confidence map is invalid.")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _seam_map(
    size: tuple[int, int],
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
) -> np.ndarray:
    width, height = size
    result = np.zeros((height, width), dtype=np.float32)
    for x in vertical_boundaries:
        if 0 <= x < width:
            result[:, max(0, x - 1) : min(width, x + 2)] = 1.0
    for y in horizontal_boundaries:
        if 0 <= y < height:
            result[max(0, y - 1) : min(height, y + 2), :] = 1.0
    return np.clip(gaussian_blur(result, 1.5), 0.0, 1.0)


class HyperWeaveEngine:
    def __init__(
        self,
        config: HyperWeaveConfig,
        generator: GeneratorAdapter,
        *,
        progress: Callable[[ProgressEvent], None] | None = None,
        interrupted: Callable[[], bool] | None = None,
    ):
        self.config = config
        self.generator = generator
        self.progress = progress or (lambda event: None)
        self.interrupted = interrupted or (lambda: False)
        self.composer = FrequencyAwareComposer()
        self.back_projection = LowFrequencyBackProjection()
        self.seam_analyzer = SeamAnalyzer()
        self._work_directory: Path | None = None
        self._debug: DebugWriter | None = None
        self._noise: CoordinateNoiseProvider | None = None
        self._candidate_rows: list[dict[str, object]] = []

    def _check_interrupted(self) -> None:
        if self.interrupted():
            raise HyperWeaveInterrupted(
                "HyperWeave was interrupted; incomplete candidates were discarded."
            )

    def _emit(
        self,
        stage: StageSpec,
        stage_count: int,
        phase: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
    ) -> None:
        self._check_interrupted()
        self.progress(
            ProgressEvent(
                stage_index=stage.index,
                stage_count=stage_count,
                phase=phase,
                current=current,
                total=total,
                message=message,
            )
        )

    def _planner(self, stage: StageSpec) -> TilePlanner:
        return TilePlanner(
            *stage.processing_size,
            tile_input_size=self.config.tile_input_size,
            core_size=self.config.core_size,
            context_size=self.config.context_size,
            stride=self.config.stride,
            alignment=self.config.latent_alignment,
        )

    def _use_memmap(self, width: int, height: int) -> bool:
        mode = AccumulatorMode(self.config.accumulator_mode)
        if mode == AccumulatorMode.MEMMAP:
            return True
        if mode == AccumulatorMode.MEMORY:
            return False
        estimate = accumulator_bytes(width, height)
        cap = int(self.config.maximum_ram_gib * 1024**3)
        return max(width, height) >= 8192 or estimate > cap

    def _accumulator(
        self, base: np.ndarray, stem: str
    ) -> AccumulatorBackend:
        if self._work_directory is None:
            raise RuntimeError("Work directory is not initialized.")
        if self._use_memmap(base.shape[1], base.shape[0]):
            return MemmapAccumulator(base, self._work_directory, stem)
        return InMemoryAccumulator(base)

    def _make_request(
        self,
        *,
        stage: StageSpec,
        pass_name: str,
        candidate_index: int,
        roi_id: int,
        strength: float,
        noise: np.ndarray,
        input_box: tuple[int, int, int, int],
        canvas_size: tuple[int, int],
    ) -> GenerationRequest:
        profile = "face_photo" if (
            pass_name == "face"
            and getattr(self, "_analysis").content_profile == ContentProfile.PHOTO
        ) else "face_illustration" if pass_name == "face" else pass_name
        return GenerationRequest(
            stage_index=stage.index,
            pass_name=pass_name,
            candidate_index=candidate_index,
            roi_id=roi_id,
            strength=float(strength),
            steps=int(self.config.exact_steps),
            seed=derive_seed(
                self._noise.resolved_seed,
                stage.index,
                pass_name,
                candidate_index,
                roi_id,
            ),
            prompt_suffix=pass_prompt_suffix(self.config, profile),
            negative_suffix=(
                self.config.negative_suffix
                if self.config.append_prompt_suffixes
                else ""
            ),
            coordinate_noise=noise,
            absolute_input_box=input_box,
            canvas_size=canvas_size,
        )

    def _generate_tiled_candidate(
        self,
        base: np.ndarray,
        stage: StageSpec,
        *,
        pass_name: str,
        candidate_index: int,
        strength: float,
        stage_count: int,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
        planner = self._planner(stage)
        tiles = planner.plan()
        accumulator = self._accumulator(
            base, f"stage{stage.index + 1:02d}_{pass_name}_c{candidate_index:02d}"
        )
        try:
            for tile_number, tile in enumerate(tiles, start=1):
                self._emit(
                    stage,
                    stage_count,
                    f"{pass_name} tiles",
                    tile_number,
                    len(tiles),
                    f"candidate {candidate_index + 1}",
                )
                tile_linear = _extract_reflect(base, tile.input_box)
                tile_image = linear_rgb_to_image(tile_linear)
                family = (
                    "anchor_global"
                    if self.config.share_anchor_noise_family
                    and pass_name in ("anchor", "global")
                    else None
                )
                noise = self._noise.crop_for_tile(
                    tile,
                    stage_index=stage.index,
                    pass_name=pass_name,
                    candidate_index=candidate_index,
                    latent_channels=self.generator.latent_channels,
                    latent_scale=self.generator.latent_scale,
                    roi_id=-1,
                    family_pass_name=family,
                )
                request = self._make_request(
                    stage=stage,
                    pass_name=pass_name,
                    candidate_index=candidate_index,
                    roi_id=-1,
                    strength=strength,
                    noise=noise,
                    input_box=tile.input_box,
                    canvas_size=stage.processing_size,
                )
                generated = self.generator.generate(tile_image, request)
                if generated.size != tile_image.size:
                    raise RuntimeError(
                        f"Generator returned {generated.size}; expected {tile_image.size}."
                    )
                generated_linear, _ = image_to_linear_rgb(generated.convert("RGB"))
                lx0, ly0, lx1, ly1 = tile.local_core_box
                cx0, cy0, cx1, cy1 = tile.core_box
                generated_core = generated_linear[ly0:ly1, lx0:lx1]
                base_core = base[cy0:cy1, cx0:cx1]
                accumulator.add(
                    tile.core_box,
                    generated_core,
                    base_core,
                    planner.weight_window(tile),
                )
            result = accumulator.finalize()
            candidate = np.asarray(result.candidate, dtype=np.float32).copy()
            confidence = np.asarray(
                result.tile_confidence, dtype=np.float32
            ).copy()
        finally:
            if isinstance(accumulator, MemmapAccumulator):
                accumulator.cleanup()
            else:
                accumulator.close()
            self._noise.clear()
        vertical = sorted(
            {
                tile.core_box[0]
                for tile in tiles
                if 0 < tile.core_box[0] < stage.processing_size[0]
            }
        )
        horizontal = sorted(
            {
                tile.core_box[1]
                for tile in tiles
                if 0 < tile.core_box[1] < stage.processing_size[1]
            }
        )
        return candidate, confidence, vertical, horizontal

    def _score_candidate(
        self,
        anchor: np.ndarray,
        candidate: np.ndarray,
        reference: np.ndarray,
        *,
        stage_index: int,
        pass_name: str,
        candidate_index: int,
        vertical: list[int],
        horizontal: list[int],
        evaluation_mask: np.ndarray | None = None,
        require_anchor_improvement: bool = True,
        anchor_baseline: CandidateScore | None = None,
    ) -> CandidateScore:
        # Evaluate the generated residual rather than scene edges that legitimately
        # happen to cross a planned boundary.
        seam = self.seam_analyzer.analyze(
            candidate - anchor, vertical, horizontal
        )
        scorer = CandidateScorer(
            strictness=self.config.candidate_rejection_strictness,
            color_drift_tolerance=self.config.color_drift_tolerance,
            new_edge_tolerance=self.config.new_edge_tolerance,
        )
        score = scorer.score(
            anchor,
            candidate,
            reference,
            candidate_index=candidate_index,
            boundary_error=seam.ratio,
            evaluation_mask=evaluation_mask,
        )
        baseline_total: float | None = None
        if require_anchor_improvement:
            baseline = anchor_baseline
            if baseline is None:
                baseline = scorer.score(
                    anchor,
                    anchor,
                    reference,
                    candidate_index=-1,
                    boundary_error=0.0,
                    evaluation_mask=evaluation_mask,
                )
            baseline_total = baseline.total
            scorer.enforce_anchor_baseline(
                score,
                baseline,
                margin=self.config.candidate_score_margin,
            )
        row = _flatten_score(score)
        row.update(
            {
                "stage": stage_index + 1,
                "pass": pass_name,
                "anchor_baseline_total": baseline_total,
                "candidate_score_margin": (
                    self.config.candidate_score_margin
                    if require_anchor_improvement
                    else None
                ),
            }
        )
        self._candidate_rows.append(row)
        logger.info(
            "HyperWeave stage=%d pass=%s candidate=%d accepted=%s "
            "score=%.6f ssim=%.6f low_mse=%.8f seam=%.6f reasons=%s",
            stage_index + 1,
            pass_name,
            candidate_index,
            score.accepted,
            score.total,
            score.roundtrip.ssim,
            score.roundtrip.low_frequency_mse,
            score.boundary_error,
            "; ".join(score.rejection_reasons) or "none",
        )
        return score

    def _select_global(
        self,
        anchor: np.ndarray,
        reference: np.ndarray,
        stage: StageSpec,
        *,
        pass_name: str,
        candidate_count: int,
        strength: float,
        stage_count: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        CandidateScore | None,
        list[int],
        list[int],
        dict[str, object],
    ]:
        best: np.ndarray | None = None
        best_confidence: np.ndarray | None = None
        best_score: CandidateScore | None = None
        boundaries: tuple[list[int], list[int]] = ([], [])
        spatial_selector: SpatialResidualSelector | None = None
        baseline_scorer = CandidateScorer(
            strictness=self.config.candidate_rejection_strictness,
            color_drift_tolerance=self.config.color_drift_tolerance,
            new_edge_tolerance=self.config.new_edge_tolerance,
        )
        anchor_baseline = baseline_scorer.score(
            anchor,
            anchor,
            reference,
            candidate_index=-1,
        )
        for candidate_index in range(candidate_count):
            candidate, confidence, vertical, horizontal = (
                self._generate_tiled_candidate(
                    anchor,
                    stage,
                    pass_name=pass_name,
                    candidate_index=candidate_index,
                    strength=strength,
                    stage_count=stage_count,
                )
            )
            # Every full-canvas candidate uses the same planned boundaries.
            # Keep them even when all candidates fail hard constraints so the
            # anchor fallback still receives seam measurement/harmonization.
            boundaries = (vertical, horizontal)
            self._emit(
                stage,
                stage_count,
                "Candidate scoring",
                candidate_index + 1,
                candidate_count,
                pass_name,
            )
            score = self._score_candidate(
                anchor,
                candidate,
                reference,
                stage_index=stage.index,
                pass_name=pass_name,
                candidate_index=candidate_index,
                vertical=vertical,
                horizontal=horizontal,
                anchor_baseline=anchor_baseline,
            )
            if self.config.save_debug_images and (
                self.config.save_all_candidates or score.accepted
            ):
                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_{pass_name}_candidate"
                    f"{candidate_index:02d}.png",
                    candidate,
                )
            if score.accepted and (
                best_score is None or score.total > best_score.total
            ):
                best = candidate
                best_confidence = confidence
                best_score = score
                spatial_selector = None
            elif (
                self.config.enable_spatial_rescue
                and best_score is None
                and not score.accepted
            ):
                if spatial_selector is None:
                    spatial_selector = SpatialResidualSelector(
                        anchor,
                        reference,
                        decision_size=self.config.spatial_decision_size,
                        transition_width=self.config.spatial_transition_width,
                        score_margin=self.config.spatial_score_margin,
                        fragmentation_limit=(
                            self.config.spatial_fragmentation_limit
                        ),
                        minimum_component_cells=(
                            self.config.spatial_minimum_component_cells
                        ),
                        strictness=self.config.candidate_rejection_strictness,
                        color_drift_tolerance=self.config.color_drift_tolerance,
                        new_edge_tolerance=self.config.new_edge_tolerance,
                        vertical_boundaries=vertical,
                        horizontal_boundaries=horizontal,
                    )
                spatial_selector.consider(
                    candidate,
                    confidence,
                    candidate_index=candidate_index,
                    global_score=score,
                )
            del candidate, confidence
        if best is None:
            spatial_failure: dict[str, object] | None = None
            if spatial_selector is not None:
                spatial = spatial_selector.finalize()
                if spatial is not None:
                    vertical = list(boundaries[0])
                    horizontal = list(boundaries[1])
                    final_score = self._score_candidate(
                        anchor,
                        spatial.candidate,
                        reference,
                        stage_index=stage.index,
                        pass_name=f"{pass_name}_spatial_rescue",
                        candidate_index=-1,
                        vertical=vertical,
                        horizontal=horizontal,
                        anchor_baseline=anchor_baseline,
                    )
                    selection_report = dict(spatial.report)
                    selection_report["final_validation"] = (
                        final_score.to_dict()
                    )
                    if final_score.accepted:
                        logger.info(
                            "HyperWeave stage=%d pass=%s used spatial "
                            "residual rescue cells=%d/%d",
                            stage.index + 1,
                            pass_name,
                            spatial.report["selected_cells"],
                            spatial.report["total_cells"],
                        )
                        return (
                            spatial.candidate,
                            spatial.confidence,
                            final_score,
                            vertical,
                            horizontal,
                            selection_report,
                        )
                    selection_report["failure_reason"] = (
                        "final whole-canvas validation rejected the rescue"
                    )
                    spatial_failure = selection_report
                    logger.warning(
                        "HyperWeave stage=%d pass=%s spatial rescue failed "
                        "final validation: %s",
                        stage.index + 1,
                        pass_name,
                        "; ".join(final_score.rejection_reasons),
                    )
                else:
                    spatial_failure = spatial_selector.diagnostic_report()
            logger.warning(
                "HyperWeave %s stage %d: all candidates rejected; using anchor.",
                pass_name,
                stage.index + 1,
            )
            return (
                anchor.copy(),
                np.ones(anchor.shape[:2], dtype=np.float32),
                None,
                boundaries[0],
                boundaries[1],
                {
                    "mode": "anchor",
                    "reason": "all whole-canvas and spatial candidates rejected",
                    "spatial_rescue_enabled": (
                        self.config.enable_spatial_rescue
                    ),
                    "spatial_rescue": spatial_failure,
                },
            )
        logger.info(
            "HyperWeave stage=%d pass=%s selected_candidate=%d score=%.6f",
            stage.index + 1,
            pass_name,
            best_score.candidate_index,
            best_score.total,
        )
        return (
            best,
            best_confidence,
            best_score,
            boundaries[0],
            boundaries[1],
            {
                "mode": "whole_canvas",
                "candidate_index": best_score.candidate_index,
            },
        )

    def _stage_maps(
        self,
        anchor: np.ndarray,
        candidate: np.ndarray,
        tile_confidence: np.ndarray,
        stage: StageSpec,
        *,
        region: np.ndarray | float = 1.0,
    ) -> ComposeMaps:
        manual_protection = resize_float(
            self._analysis.manual_protection, stage.processing_size
        )
        manual_boost = resize_float(
            self._analysis.manual_boost, stage.processing_size
        )
        structure = StructureMapBuilder().build(
            anchor, manual_protection
        ).protection
        return ComposeMaps(
            structure=structure,
            orientation=orientation_confidence(anchor, candidate),
            new_edge=new_edge_confidence(anchor, candidate),
            tile=tile_confidence,
            roundtrip=1.0,
            region=region,
            manual_boost=manual_boost,
        )

    def _compose_selected(
        self,
        anchor: np.ndarray,
        candidate: np.ndarray,
        reference: np.ndarray,
        confidence: np.ndarray,
        stage: StageSpec,
        *,
        score: CandidateScore | None,
        region: np.ndarray | float = 1.0,
        gains: FrequencyGains | None = None,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        if score is None:
            return (
                anchor.copy(),
                {
                    name: 0.0
                    for name in (
                        "high_0",
                        "high_1",
                        "mid_high",
                        "mid",
                        "mid_low",
                        "low",
                    )
                },
                np.ones(anchor.shape[:2], dtype=np.float32),
            )
        maps = self._stage_maps(
            anchor, candidate, confidence, stage, region=region
        )
        local_roundtrip = _roundtrip_confidence_map(
            reference,
            candidate,
            score.roundtrip.confidence,
        )
        maps = ComposeMaps(
            structure=maps.structure,
            orientation=maps.orientation,
            new_edge=maps.new_edge,
            tile=maps.tile,
            roundtrip=local_roundtrip,
            region=maps.region,
            manual_boost=maps.manual_boost,
        )
        composed, energy = self.composer.compose(
            anchor,
            candidate,
            gains=gains or _candidate_gains(stage),
            maps=maps,
            structural_lock=self.config.structural_lock,
            low_frequency_lock=self.config.low_frequency_lock,
            boost_strength=self.config.boost_strength,
        )
        return composed, energy, local_roundtrip

    def _prepare_roi_payload(
        self, crop: np.ndarray, size: int
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = crop.shape[:2]
        scale = min(size / width, size / height)
        resized_size = (
            max(8, round(width * scale / 8) * 8),
            max(8, round(height * scale / 8) * 8),
        )
        resized = resize_float(crop, resized_size)
        left = (size - resized_size[0]) // 2
        top = (size - resized_size[1]) // 2
        right = size - resized_size[0] - left
        bottom = size - resized_size[1] - top
        mode = "reflect" if min(resized.shape[:2]) > 1 else "edge"
        payload = np.pad(
            resized,
            ((top, bottom), (left, right), (0, 0)),
            mode=mode,
        )
        return payload, (left, top, left + resized_size[0], top + resized_size[1])

    def _generate_roi_candidates(
        self,
        canvas: np.ndarray,
        stage: StageSpec,
        box: tuple[int, int, int, int],
        *,
        pass_name: str,
        roi_id: int,
        candidate_count: int,
        strength: float,
        processing_size: int,
        stage_count: int,
        evaluation_mask: np.ndarray | None = None,
        write_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, CandidateScore | None]:
        x0, y0, x1, y1 = box
        anchor_crop = canvas[y0:y1, x0:x1].copy()
        payload, content_box = self._prepare_roi_payload(
            anchor_crop, processing_size
        )
        best_crop: np.ndarray | None = None
        best_score: CandidateScore | None = None
        roi_strictness = self.config.candidate_rejection_strictness
        if pass_name == "face":
            roi_strictness = max(
                roi_strictness,
                1.0 - self.config.face_structure_tolerance,
            )
        scorer = CandidateScorer(
            strictness=roi_strictness,
            color_drift_tolerance=self.config.color_drift_tolerance,
            new_edge_tolerance=self.config.new_edge_tolerance,
        )
        baseline = scorer.score(
            anchor_crop,
            anchor_crop,
            anchor_crop,
            candidate_index=-1,
            evaluation_mask=evaluation_mask,
        )
        if pass_name == "hair" and evaluation_mask is not None:
            baseline.total += hair_flow_score(
                anchor_crop, anchor_crop, evaluation_mask
            ).total
        if self.config.save_debug_images and self.config.save_roi_crops:
            self._debug.save_image(
                f"_hw_{pass_name}_{roi_id:03d}_anchor.png", anchor_crop
            )
        for candidate_index in range(candidate_count):
            self._emit(
                stage,
                stage_count,
                f"{pass_name.title()} ROI {roi_id + 1}",
                candidate_index + 1,
                candidate_count,
            )
            noise = self._noise.canvas(
                stage_index=stage.index,
                pass_name=pass_name,
                candidate_index=candidate_index,
                latent_width=processing_size // self.generator.latent_scale,
                latent_height=processing_size // self.generator.latent_scale,
                latent_channels=self.generator.latent_channels,
                roi_id=roi_id,
            )
            request = self._make_request(
                stage=stage,
                pass_name=pass_name,
                candidate_index=candidate_index,
                roi_id=roi_id,
                strength=strength,
                noise=noise,
                input_box=box,
                canvas_size=stage.processing_size,
            )
            generated = self.generator.generate(
                linear_rgb_to_image(payload), request
            )
            self._noise.clear()
            generated_linear, _ = image_to_linear_rgb(generated.convert("RGB"))
            px0, py0, px1, py1 = content_box
            generated_crop = resize_float(
                generated_linear[py0:py1, px0:px1],
                (anchor_crop.shape[1], anchor_crop.shape[0]),
            )
            score = scorer.score(
                anchor_crop,
                generated_crop,
                anchor_crop,
                candidate_index=candidate_index,
                evaluation_mask=evaluation_mask,
            )
            if pass_name == "hair" and evaluation_mask is not None:
                flow = hair_flow_score(
                    anchor_crop, generated_crop, evaluation_mask
                )
                score.total += flow.total
                if flow.crossing_penalty > self.config.hair_flow_tolerance:
                    score.accepted = False
                    score.rejection_reasons.append(
                        f"hair crossing penalty {flow.crossing_penalty:.4f}"
                    )
            scorer.enforce_anchor_baseline(
                score,
                baseline,
                margin=self.config.candidate_score_margin,
            )
            row = _flatten_score(score)
            row.update(
                {
                    "stage": stage.index + 1,
                    "pass": pass_name,
                    "roi_id": roi_id,
                    "anchor_baseline_total": baseline.total,
                    "candidate_score_margin": self.config.candidate_score_margin,
                }
            )
            self._candidate_rows.append(row)
            logger.info(
                "HyperWeave stage=%d pass=%s roi=%d candidate=%d "
                "accepted=%s score=%.6f ssim=%.6f reasons=%s",
                stage.index + 1,
                pass_name,
                roi_id,
                candidate_index,
                score.accepted,
                score.total,
                score.roundtrip.ssim,
                "; ".join(score.rejection_reasons) or "none",
            )
            if self.config.save_debug_images and self.config.save_roi_crops:
                self._debug.save_image(
                    f"_hw_{pass_name}_{roi_id:03d}_candidate"
                    f"{candidate_index:02d}.png",
                    generated_crop,
                )
            if score.accepted and (
                best_score is None or score.total > best_score.total
            ):
                best_crop = generated_crop
                best_score = score
        if best_crop is None:
            logger.warning(
                "HyperWeave stage=%d pass=%s roi=%d: all candidates "
                "rejected; keeping the current ROI.",
                stage.index + 1,
                pass_name,
                roi_id,
            )
            return canvas, None
        local_region = (
            local_feather_mask(
                (anchor_crop.shape[1], anchor_crop.shape[0]),
                max(24, min(anchor_crop.shape[:2]) // 10),
            )
            if write_mask is None
            else np.asarray(write_mask, dtype=np.float32)
        )
        if local_region.shape != anchor_crop.shape[:2]:
            local_region = resize_float(
                local_region, (anchor_crop.shape[1], anchor_crop.shape[0])
            )
        if not np.isfinite(local_region).all():
            raise ValueError("ROI write mask contains NaN or Inf.")
        local_region = np.clip(local_region, 0.0, 1.0)
        local_roundtrip = _roundtrip_confidence_map(
            anchor_crop,
            best_crop,
            best_score.roundtrip.confidence,
        )
        maps = ComposeMaps(
            structure=StructureMapBuilder().build(anchor_crop).protection,
            orientation=orientation_confidence(anchor_crop, best_crop),
            new_edge=new_edge_confidence(anchor_crop, best_crop),
            tile=np.ones(anchor_crop.shape[:2], dtype=np.float32),
            roundtrip=local_roundtrip,
            region=local_region,
            manual_boost=0.0,
        )
        composed, _ = self.composer.compose(
            anchor_crop,
            best_crop,
            gains=_candidate_gains(stage),
            maps=maps,
            structural_lock=self.config.structural_lock,
            low_frequency_lock=1.0,
            boost_strength=0.0,
        )
        result = canvas.copy()
        result[y0:y1, x0:x1] = composed
        if self.config.save_debug_images and self.config.save_roi_crops:
            self._debug.save_image(
                f"_hw_{pass_name}_{roi_id:03d}_selected.png", composed
            )
            self._debug.save_image(
                f"_hw_{pass_name}_{roi_id:03d}_mask.png", local_region
            )
            self._debug.save_image(
                f"_hw_{pass_name}_{roi_id:03d}_roundtrip_confidence.png",
                local_roundtrip,
            )
            self._debug.save_json(
                f"_hw_{pass_name}_{roi_id:03d}_score.json",
                best_score.to_dict(),
            )
        logger.info(
            "HyperWeave stage=%d pass=%s roi=%d selected_candidate=%d "
            "score=%.6f",
            stage.index + 1,
            pass_name,
            roi_id,
            best_score.candidate_index,
            best_score.total,
        )
        return result, best_score

    def _run_face_and_hair(
        self,
        canvas: np.ndarray,
        stage: StageSpec,
        stage_count: int,
        face_core_union: np.ndarray | None,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        reports: list[dict[str, object]] = []
        scale_x = stage.processing_size[0] / self._analysis.source_size[0]
        scale_y = stage.processing_size[1] / self._analysis.source_size[1]
        regions = [
            expand_face_roi(
                detection,
                scale_x=scale_x,
                scale_y=scale_y,
                stage_size=stage.processing_size,
                region_id=index,
            )
            for index, detection in enumerate(self._analysis.face_detections)
        ]
        logger.info(
            "HyperWeave stage=%d semantic_regions=%d face_pass=%s hair_pass=%s",
            stage.index + 1,
            len(regions),
            stage.run_face_pass,
            stage.run_hair_pass,
        )

        # Hair for every person precedes every Face pass.  Context crops may
        # overlap, but identities and masks are never merged.
        if stage.run_hair_pass:
            for region in regions:
                detection = self._analysis.face_detections[region.region_id]
                x0, y0, x1, y1 = region.stage_box
                face_box_source = detection.bbox
                face_box_stage = (
                    max(x0, round(face_box_source[0] * scale_x)),
                    max(y0, round(face_box_source[1] * scale_y)),
                    min(x1, round(face_box_source[2] * scale_x)),
                    min(y1, round(face_box_source[3] * scale_y)),
                )
                local_hair = hair_region_mask(
                    stage.processing_size,
                    face_box_stage,
                    output_box=region.stage_box,
                )
                if face_core_union is None:
                    raise RuntimeError("Face Core union is unavailable for Hair pass.")
                local_face_union = face_core_union[y0:y1, x0:x1]
                local_hair *= 1.0 - np.clip(
                    local_face_union * 1.5, 0.0, 1.0
                )
                canvas, score = self._generate_roi_candidates(
                    canvas,
                    stage,
                    region.stage_box,
                    pass_name="hair",
                    roi_id=region.region_id,
                    candidate_count=self.config.hair_candidates,
                    strength=stage.hair_strength,
                    processing_size=region.processing_size,
                    stage_count=stage_count,
                    evaluation_mask=local_hair,
                    write_mask=local_hair,
                )
                reports.append(
                    {
                        "kind": "hair",
                        "roi_id": region.region_id,
                        "box": list(region.stage_box),
                        "processing_size": region.processing_size,
                        "candidate_count": self.config.hair_candidates,
                        "evaluation_fraction": (
                            float(np.mean(local_hair))
                            if score is None
                            else score.evaluation_fraction
                        ),
                        "selected": (
                            None if score is None else score.candidate_index
                        ),
                        "selected_score": (
                            None if score is None else score.to_dict()
                        ),
                    }
                )

        if stage.run_face_pass:
            for region in regions:
                detection = self._analysis.face_detections[region.region_id]
                masks = face_region_masks(
                    self._analysis.face_detections,
                    region.region_id,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    stage_size=stage.processing_size,
                    context_box=region.stage_box,
                )
                count = min(
                    self.config.face_candidates,
                    default_face_candidate_count(detection),
                )
                canvas, score = self._generate_roi_candidates(
                    canvas,
                    stage,
                    region.stage_box,
                    pass_name="face",
                    roi_id=region.region_id,
                    candidate_count=count,
                    strength=stage.face_strength,
                    processing_size=region.processing_size,
                    stage_count=stage_count,
                    evaluation_mask=masks.evaluation,
                    write_mask=masks.write,
                )
                reports.append(
                    {
                        "kind": "face",
                        "roi_id": region.region_id,
                        "box": list(region.stage_box),
                        "source_face_size": list(region.original_face_size),
                        "processing_size": region.processing_size,
                        "candidate_count": count,
                        "evaluation_fraction": (
                            float(np.mean(masks.evaluation))
                            if score is None
                            else score.evaluation_fraction
                        ),
                        "selected": (
                            None if score is None else score.candidate_index
                        ),
                        "selected_score": (
                            None if score is None else score.to_dict()
                        ),
                    }
                )
        return canvas, reports

    def _final_face_metrics(
        self,
        final_canvas: np.ndarray,
        stage: StageSpec,
    ) -> tuple[list[dict[str, object]], dict[str, float | None]]:
        source_width, source_height = self._analysis.source_size
        target_width, target_height = stage.target_size
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        reports: list[dict[str, object]] = []

        def clipped_box(
            box: tuple[float, float, float, float],
            size: tuple[int, int],
            *,
            scale: tuple[float, float] = (1.0, 1.0),
        ) -> tuple[int, int, int, int]:
            x0 = max(0, int(np.floor(box[0] * scale[0])))
            y0 = max(0, int(np.floor(box[1] * scale[1])))
            x1 = min(size[0], int(np.ceil(box[2] * scale[0])))
            y1 = min(size[1], int(np.ceil(box[3] * scale[1])))
            if x1 - x0 < 3:
                center = (x0 + x1) // 2
                x0 = max(0, center - 2)
                x1 = min(size[0], x0 + 4)
                x0 = max(0, x1 - 4)
            if y1 - y0 < 3:
                center = (y0 + y1) // 2
                y0 = max(0, center - 2)
                y1 = min(size[1], y0 + 4)
                y0 = max(0, y1 - 4)
            return x0, y0, x1, y1

        for face_id, detection in enumerate(self._analysis.face_detections):
            source_box = clipped_box(
                detection.bbox, (source_width, source_height)
            )
            target_box = clipped_box(
                detection.bbox,
                (target_width, target_height),
                scale=(scale_x, scale_y),
            )
            sx0, sy0, sx1, sy1 = source_box
            tx0, ty0, tx1, ty1 = target_box
            if sx1 <= sx0 or sy1 <= sy0 or tx1 <= tx0 or ty1 <= ty0:
                continue
            source_crop = self._analysis.source_linear_rgb[sy0:sy1, sx0:sx1]
            target_crop = final_canvas[ty0:ty1, tx0:tx1]
            # Metrics describe the delivered 8-bit image, not only the
            # pre-encoding float canvas.  Quantize just the small face crop so
            # 4K/8K peak memory does not grow by another full RGB canvas.
            target_crop, _ = image_to_linear_rgb(
                linear_rgb_to_image(target_crop)
            )
            source_mask = face_core_mask(
                detection,
                scale_x=1.0,
                scale_y=1.0,
                stage_size=self._analysis.source_size,
                output_box=source_box,
            )
            metrics = roundtrip_metrics(
                source_crop,
                target_crop,
                evaluation_mask=source_mask,
            )
            reports.append(
                {
                    "face_id": face_id,
                    "source_bbox": list(source_box),
                    "target_bbox": list(target_box),
                    "roundtrip_ssim": metrics.ssim,
                    "edge_displacement": metrics.edge_displacement,
                    "edge_displacement_forward": (
                        metrics.edge_displacement_forward
                    ),
                    "edge_displacement_reverse": (
                        metrics.edge_displacement_reverse
                    ),
                    "edge_precision": metrics.edge_precision,
                    "edge_recall": metrics.edge_recall,
                    "edge_f1": metrics.edge_f1,
                    "low_frequency_mse": metrics.low_frequency_mse,
                    "color_drift": metrics.color_drift,
                    "evaluation_fraction": metrics.evaluation_fraction,
                }
            )

        ssim_values = [
            float(item["roundtrip_ssim"]) for item in reports
        ]
        edge_values = [float(item["edge_f1"]) for item in reports]
        aggregates: dict[str, float | None] = {
            "final_face_roundtrip_ssim_mean": (
                float(np.mean(ssim_values)) if ssim_values else None
            ),
            "final_face_edge_f1_mean": (
                float(np.mean(edge_values)) if edge_values else None
            ),
            "final_face_edge_f1_min": (
                float(np.min(edge_values)) if edge_values else None
            ),
        }
        return reports, aggregates

    def _preflight(
        self, stages: list[StageSpec], work_root: Path
    ) -> dict[str, object]:
        largest = max(stages, key=lambda item: np.prod(item.processing_size))
        width, height = largest.processing_size
        accumulator_estimate = accumulator_bytes(width, height)
        working_estimate = width * height * 48 + accumulator_estimate
        use_memmap = self._use_memmap(width, height)
        disk_estimate = accumulator_estimate * (2 if use_memmap else 0) + 256 * 1024**2
        free_disk = shutil.disk_usage(work_root).free
        if free_disk < disk_estimate:
            raise RuntimeError(
                f"HyperWeave needs about {disk_estimate / 1024**3:.2f} GiB "
                f"temporary disk; only {free_disk / 1024**3:.2f} GiB is free."
            )
        available_ram = psutil.virtual_memory().available
        if not use_memmap and working_estimate > available_ram * 0.85:
            raise RuntimeError(
                f"HyperWeave estimates {working_estimate / 1024**3:.2f} GiB "
                f"working RAM; only {available_ram / 1024**3:.2f} GiB is available. "
                "Select disk-backed memmap."
            )
        return {
            "largest_processing_size": list(largest.processing_size),
            "accumulator_bytes": accumulator_estimate,
            "working_ram_estimate_bytes": working_estimate,
            "disk_estimate_bytes": disk_estimate,
            "available_ram_bytes": available_ram,
            "free_disk_bytes": free_disk,
            "memmap": use_memmap,
        }

    def _estimated_generation_calls(self, stages: list[StageSpec]) -> int:
        total = 0
        for stage in stages:
            tile_count = len(self._planner(stage).plan())
            full_passes = 1 + self.config.global_candidates
            if stage.run_material_pass:
                full_passes += self.config.material_candidates
            if self.config.enable_micro_pass and stage.index == len(stages) - 1:
                full_passes += 1
            total += tile_count * full_passes

            if not (stage.run_face_pass or stage.run_hair_pass):
                continue
            scale_x = stage.processing_size[0] / self._analysis.source_size[0]
            scale_y = stage.processing_size[1] / self._analysis.source_size[1]
            regions = [
                expand_face_roi(
                    detection,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    stage_size=stage.processing_size,
                    region_id=index,
                )
                for index, detection in enumerate(
                    self._analysis.face_detections
                )
            ]
            repeats = (
                self.config.roi_final_pass_count
                if stage.index == len(stages) - 1
                else 1
            )
            for region in regions:
                detection = self._analysis.face_detections[
                    min(region.region_id, len(self._analysis.face_detections) - 1)
                ]
                if stage.run_face_pass:
                    total += repeats * min(
                        self.config.face_candidates,
                        default_face_candidate_count(detection),
                    )
                if stage.run_hair_pass:
                    total += repeats * self.config.hair_candidates
        return max(1, total)

    def run(
        self,
        source: Image.Image,
        *,
        debug_stem: str = "hyperweave",
        debug_destination: str | Path | None = None,
    ) -> HyperWeaveResult:
        started = time.perf_counter()
        self._candidate_rows.clear()
        self.config.validate(source.size)
        target = resolve_target_size(source.size, self.config)
        stages = plan_upscale_stages(
            source.width,
            source.height,
            target[0],
            target[1],
            config=self.config,
        )
        self._analysis = SourceAnalyzer().analyze(source, self.config)
        self._noise = CoordinateNoiseProvider(self.config.seed)
        resolved_seed = self._noise.resolved_seed
        model_metadata = self.generator.model_metadata()
        logger.info(
            "HyperWeave start input=%dx%d target=%dx%d stages=%d preset=%s "
            "profile=%s seed=%d",
            source.width,
            source.height,
            target[0],
            target[1],
            len(stages),
            self.config.preset,
            self._analysis.content_profile,
            resolved_seed,
        )
        logger.info(
            "HyperWeave model=%s vae=%s sampler=%s scheduler=%s detector=%s "
            "faces=%d source_face_sizes=%s",
            model_metadata.get("model"),
            model_metadata.get("vae"),
            model_metadata.get("sampler"),
            model_metadata.get("scheduler"),
            self._analysis.detector_provider,
            len(self._analysis.face_detections),
            [
                [round(item.width), round(item.height)]
                for item in self._analysis.face_detections
            ],
        )
        logger.info(
            "HyperWeave stage_plan=%s strengths=%s candidates=%s",
            [
                {
                    "source": stage.source_size,
                    "target": stage.target_size,
                    "processing": stage.processing_size,
                }
                for stage in stages
            ],
            {
                "anchor": self.config.anchor_strength,
                "global": self.config.global_overdraw_strength,
                "face": self.config.face_strength,
                "hair": self.config.hair_strength,
                "material": self.config.material_strength,
                "micro": self.config.micro_strength,
            },
            {
                "global": self.config.global_candidates,
                "face": self.config.face_candidates,
                "hair": self.config.hair_candidates,
                "material": self.config.material_candidates,
            },
        )
        self.progress(
            ProgressEvent(
                stage_index=0,
                stage_count=len(stages),
                phase="Plan",
                total=self._estimated_generation_calls(stages),
                message=(
                    f"{source.width}x{source.height} -> "
                    f"{target[0]}x{target[1]}"
                ),
            )
        )
        temp_parent = (
            Path(self.config.temp_directory).expanduser()
            if self.config.temp_directory
            else None
        )
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        messages = list(self._analysis.messages)
        stage_reports: list[dict[str, object]] = []
        debug_files: list[Path] = []
        previous_linear = self._analysis.source_linear_rgb
        previous_alpha = self._analysis.source_alpha
        if self.config.model_background == "White":
            model_background = np.ones(3, dtype=np.float32)
        elif self.config.model_background == "Black":
            model_background = np.zeros(3, dtype=np.float32)
        else:
            border = np.concatenate(
                [
                    previous_linear[0, :, :],
                    previous_linear[-1, :, :],
                    previous_linear[:, 0, :],
                    previous_linear[:, -1, :],
                ],
                axis=0,
            )
            if previous_alpha is not None:
                alpha_border = np.concatenate(
                    [
                        previous_alpha[0, :],
                        previous_alpha[-1, :],
                        previous_alpha[:, 0],
                        previous_alpha[:, -1],
                    ]
                )
                visible_border = border[alpha_border > 0.05]
                if len(visible_border):
                    border = visible_border
            model_background = np.median(border, axis=0).astype(np.float32)
        current_linear = previous_linear
        current_alpha = previous_alpha
        memory_report: dict[str, object]

        with tempfile.TemporaryDirectory(
            prefix="hyperweave_", dir=temp_parent
        ) as temporary:
            self._work_directory = Path(temporary)
            self._debug = DebugWriter(
                self._work_directory,
                stem=debug_stem,
                enabled=self.config.save_debug_images,
            )
            memory_report = self._preflight(stages, self._work_directory)
            logger.info(
                "HyperWeave preflight ram_estimate=%d accumulator=%d "
                "disk_estimate=%d memmap=%s temp=%s",
                memory_report["working_ram_estimate_bytes"],
                memory_report["accumulator_bytes"],
                memory_report["disk_estimate_bytes"],
                memory_report["memmap"],
                self._work_directory,
            )
            for stage in stages:
                tile_count = len(self._planner(stage).plan())
                logger.info(
                    "HyperWeave stage=%d/%d source=%s target=%s processing=%s "
                    "tiles=%d tile=%d core=%d context=%d stride=%d",
                    stage.index + 1,
                    len(stages),
                    stage.source_size,
                    stage.target_size,
                    stage.processing_size,
                    tile_count,
                    self.config.tile_input_size,
                    self.config.core_size,
                    self.config.context_size,
                    self.config.stride,
                )
                self._emit(
                    stage, len(stages), "Base resize", message=str(stage.processing_size)
                )
                base, alpha = _resize_with_alpha(
                    previous_linear,
                    previous_alpha,
                    stage.processing_size,
                    model_background,
                )
                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_base.png", base
                )

                anchor, anchor_confidence, vertical, horizontal = (
                    self._generate_tiled_candidate(
                        base,
                        stage,
                        pass_name="anchor",
                        candidate_index=0,
                        strength=stage.anchor_strength,
                        stage_count=len(stages),
                    )
                )
                anchor_score = self._score_candidate(
                    base,
                    anchor,
                    previous_linear,
                    stage_index=stage.index,
                    pass_name="anchor",
                    candidate_index=0,
                    vertical=vertical,
                    horizontal=horizontal,
                    require_anchor_improvement=False,
                )
                if not anchor_score.accepted:
                    messages.append(
                        f"Stage {stage.index + 1} anchor rejected: "
                        + "; ".join(anchor_score.rejection_reasons)
                    )
                    logger.warning(
                        "HyperWeave stage=%d anchor rejected; using base resize: %s",
                        stage.index + 1,
                        "; ".join(anchor_score.rejection_reasons),
                    )
                    anchor = base.copy()
                    anchor_confidence[:] = 1.0
                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_anchor.png", anchor
                )
                self.generator.pass_cleanup()

                (
                    selected,
                    tile_confidence,
                    selected_score,
                    vertical,
                    horizontal,
                    global_selection,
                ) = self._select_global(
                    anchor,
                    previous_linear,
                    stage,
                    pass_name="global",
                    candidate_count=self.config.global_candidates,
                    strength=stage.overdraw_strength,
                    stage_count=len(stages),
                )
                self.generator.pass_cleanup()
                selection_reports = [
                    {"pass": "global", **global_selection}
                ]
                composed, band_energy, roundtrip_confidence = self._compose_selected(
                    anchor,
                    selected,
                    previous_linear,
                    tile_confidence,
                    stage,
                    score=selected_score,
                )
                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_global_selected.png",
                    selected,
                )

                structure_maps = StructureMapBuilder().build(composed)
                if self.config.save_maps:
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_structure_protect.png",
                        structure_maps.protection,
                    )
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_orientation_confidence.png",
                        orientation_confidence(anchor, selected),
                    )
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_new_edge_confidence.png",
                        new_edge_confidence(anchor, selected),
                    )
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_tile_confidence.png",
                        tile_confidence,
                    )
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_roundtrip_confidence.png",
                        roundtrip_confidence,
                    )
                    for frequency_name, frequency_map in _frequency_debug_maps(
                        anchor, selected
                    ).items():
                        self._debug.save_image(
                            f"_hw_stage{stage.index + 1:02d}_frequency_"
                            f"{frequency_name}.png",
                            frequency_map,
                        )
                combined_confidence = tile_confidence * roundtrip_confidence
                logger.info(
                    "HyperWeave stage=%d structure_p50=%.6f "
                    "structure_p95=%.6f confidence_p50=%.6f "
                    "confidence_p95=%.6f band_energy=%s",
                    stage.index + 1,
                    float(np.percentile(structure_maps.protection, 50.0)),
                    float(np.percentile(structure_maps.protection, 95.0)),
                    float(np.percentile(combined_confidence, 50.0)),
                    float(np.percentile(combined_confidence, 95.0)),
                    band_energy,
                )

                face_scale_x = (
                    stage.processing_size[0] / self._analysis.source_size[0]
                )
                face_scale_y = (
                    stage.processing_size[1] / self._analysis.source_size[1]
                )
                face_core_union = (
                    face_core_union_mask(
                        self._analysis.face_detections,
                        scale_x=face_scale_x,
                        scale_y=face_scale_y,
                        stage_size=stage.processing_size,
                    )
                    if self._analysis.face_detections
                    else None
                )

                if stage.run_material_pass:
                    potential = detail_potential_map(
                        composed,
                        flat_region_detail=self.config.flat_region_detail,
                        face_protection=face_core_union,
                        manual_boost=resize_float(
                            self._analysis.manual_boost,
                            stage.processing_size,
                        ),
                    )
                    (
                        material,
                        confidence,
                        material_score,
                        mv,
                        mh,
                        material_selection,
                    ) = self._select_global(
                        composed,
                        previous_linear,
                        stage,
                        pass_name="material",
                        candidate_count=self.config.material_candidates,
                        strength=stage.material_strength,
                        stage_count=len(stages),
                    )
                    selection_reports.append(
                        {"pass": "material", **material_selection}
                    )
                    (
                        composed,
                        material_energy,
                        material_roundtrip,
                    ) = self._compose_selected(
                        composed,
                        material,
                        previous_linear,
                        confidence,
                        stage,
                        score=material_score,
                        region=potential,
                    )
                    if self.config.save_maps:
                        self._debug.save_image(
                            f"_hw_stage{stage.index + 1:02d}_material_"
                            "roundtrip_confidence.png",
                            material_roundtrip,
                        )
                    band_energy.update(
                        {f"material_{key}": value for key, value in material_energy.items()}
                    )
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_detail_potential.png",
                        potential,
                    )
                    vertical = sorted(set(vertical + mv))
                    horizontal = sorted(set(horizontal + mh))
                    self.generator.pass_cleanup()

                if (
                    self.config.enable_micro_pass
                    and stage.index == len(stages) - 1
                ):
                    (
                        micro,
                        confidence,
                        micro_score,
                        mv,
                        mh,
                        micro_selection,
                    ) = self._select_global(
                        composed,
                        previous_linear,
                        stage,
                        pass_name="micro",
                        candidate_count=1,
                        strength=stage.micro_strength,
                        stage_count=len(stages),
                    )
                    selection_reports.append(
                        {"pass": "micro", **micro_selection}
                    )
                    micro_gains = FrequencyGains(
                        high_0=0.90,
                        high_1=1.00,
                        mid_high=0.35,
                        mid=0.0,
                        mid_low=0.0,
                        low=0.0,
                        chroma_ratio=0.25,
                    )
                    micro_region = (
                        1.0
                        if face_core_union is None
                        else 1.0 - np.clip(
                            face_core_union * 1.5, 0.0, 1.0
                        )
                    )
                    composed, micro_energy, micro_roundtrip = self._compose_selected(
                        composed,
                        micro,
                        previous_linear,
                        confidence,
                        stage,
                        score=micro_score,
                        region=micro_region,
                        gains=micro_gains,
                    )
                    if self.config.save_maps:
                        self._debug.save_image(
                            f"_hw_stage{stage.index + 1:02d}_micro_"
                            "roundtrip_confidence.png",
                            micro_roundtrip,
                        )
                    band_energy.update(
                        {f"micro_{key}": value for key, value in micro_energy.items()}
                    )
                    vertical = sorted(set(vertical + mv))
                    horizontal = sorted(set(horizontal + mh))
                    self.generator.pass_cleanup()

                roi_reports: list[dict[str, object]] = []
                if stage.run_face_pass or stage.run_hair_pass:
                    repeats = (
                        self.config.roi_final_pass_count
                        if stage.index == len(stages) - 1
                        else 1
                    )
                    for roi_pass_index in range(repeats):
                        decay = 0.85**roi_pass_index
                        roi_stage = replace(
                            stage,
                            face_strength=stage.face_strength * decay,
                            hair_strength=stage.hair_strength * decay,
                        )
                        composed, pass_reports = self._run_face_and_hair(
                            composed,
                            roi_stage,
                            len(stages),
                            face_core_union,
                        )
                        for report in pass_reports:
                            report["final_pass"] = roi_pass_index + 1
                        roi_reports.extend(pass_reports)
                    self.generator.pass_cleanup()
                del face_core_union

                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_composed.png", composed
                )
                if self.config.save_maps:
                    self._debug.save_image(
                        f"_hw_stage{stage.index + 1:02d}_seam_map.png",
                        _seam_map(
                            stage.processing_size, vertical, horizontal
                        ),
                    )
                composed, seam_report = self.seam_analyzer.harmonize(
                    composed, vertical, horizontal
                )
                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_before_backprojection.png",
                    composed,
                )
                self._emit(
                    stage,
                    len(stages),
                    "Back projection",
                    1,
                    self.config.back_projection_iterations,
                )
                composed, back_report = self.back_projection.apply(
                    composed,
                    previous_linear,
                    iterations=self.config.back_projection_iterations,
                    beta=self.config.back_projection_beta,
                )
                exact_linear = (
                    composed
                    if stage.processing_size == stage.target_size
                    else resize_float(composed, stage.target_size)
                )
                exact_alpha = (
                    None
                    if alpha is None
                    else (
                        alpha
                        if stage.processing_size == stage.target_size
                        else np.clip(
                            resize_float(alpha, stage.target_size),
                            0.0,
                            1.0,
                        )
                    )
                )
                self._debug.save_image(
                    f"_hw_stage{stage.index + 1:02d}_final.png", exact_linear
                )
                final_face_metrics, final_face_aggregates = (
                    self._final_face_metrics(exact_linear, stage)
                )
                selected_face_by_id: dict[int, float] = {}
                for roi_report in roi_reports:
                    if roi_report.get("kind") != "face":
                        continue
                    selected_roi_score = roi_report.get("selected_score")
                    selected_roundtrip = (
                        selected_roi_score.get("roundtrip")
                        if isinstance(selected_roi_score, dict)
                        else None
                    )
                    if (
                        isinstance(selected_roundtrip, dict)
                        and isinstance(
                            selected_roundtrip.get("ssim"), (int, float)
                        )
                    ):
                        selected_face_by_id[int(roi_report["roi_id"])] = float(
                            selected_roundtrip["ssim"]
                        )
                selected_face_candidate_ssim = (
                    float(np.mean(list(selected_face_by_id.values())))
                    if selected_face_by_id
                    else None
                )
                stage_reports.append(
                    {
                        "stage": stage.index + 1,
                        "source_size": list(stage.source_size),
                        "target_size": list(stage.target_size),
                        "processing_size": list(stage.processing_size),
                        "scale": [stage.scale_x, stage.scale_y],
                        "selected_global_candidate": (
                            None
                            if (
                                selected_score is None
                                or global_selection["mode"] != "whole_canvas"
                            )
                            else selected_score.candidate_index
                        ),
                        "selection_reports": selection_reports,
                        "anchor_score": anchor_score.to_dict(),
                        "selected_score": (
                            None
                            if selected_score is None
                            else selected_score.to_dict()
                        ),
                        "frequency_band_energy": band_energy,
                        "seam": asdict(seam_report),
                        "back_projection": asdict(back_report),
                        "rois": roi_reports,
                        "final_face_metrics": final_face_metrics,
                        "selected_face_candidate_roundtrip_ssim": (
                            selected_face_candidate_ssim
                        ),
                        "final_face_roundtrip_ssim": final_face_aggregates[
                            "final_face_roundtrip_ssim_mean"
                        ],
                        "final_face_edge_f1": final_face_aggregates[
                            "final_face_edge_f1_mean"
                        ],
                        **final_face_aggregates,
                    }
                )
                logger.info(
                    "HyperWeave stage=%d complete selected=%s seam_ratio=%.6f "
                    "roundtrip_ssim=%s low_frequency_error=%s rois=%d",
                    stage.index + 1,
                    (
                        None
                        if selected_score is None
                        else selected_score.candidate_index
                    ),
                    seam_report.ratio,
                    (
                        "n/a"
                        if selected_score is None
                        else f"{selected_score.roundtrip.ssim:.6f}"
                    ),
                    (
                        "n/a"
                        if selected_score is None
                        else f"{selected_score.roundtrip.low_frequency_mse:.8f}"
                    ),
                    len(roi_reports),
                )
                previous_linear = exact_linear
                previous_alpha = exact_alpha
                current_linear = exact_linear
                current_alpha = exact_alpha

            processing_time = time.perf_counter() - started
            runtime_metrics = self.generator.runtime_metrics()
            metrics: dict[str, object] = {
                "processing_time_seconds": processing_time,
                "stage_reports": stage_reports,
                "candidate_scores": self._candidate_rows,
                "memory": memory_report,
                "runtime": runtime_metrics,
            }
            metadata: dict[str, object] = {
                "format_version": 1,
                "mode_id": "hyper_weave",
                "version": HYPERWEAVE_VERSION,
                "scoring_version": SCORING_VERSION,
                "candidate_score_margin": self.config.candidate_score_margin,
                "local_roundtrip_gate_enabled": True,
                "semantic_face_ownership_enabled": True,
                "face_protection_enabled": True,
                "symmetric_edge_metrics_enabled": True,
                "pass_order": [
                    "anchor",
                    "global",
                    "material",
                    "micro",
                    "hair",
                    "face",
                    "seam_analysis",
                    "low_frequency_back_projection",
                    "exact_resize",
                    "final_face_evaluation",
                ],
                "preset": str(self.config.preset),
                "content_profile": str(self._analysis.content_profile),
                "source_size": list(source.size),
                "target_size": list(target),
                "stage_plan": [
                    {
                        "source": list(stage.source_size),
                        "target": list(stage.target_size),
                        "processing": list(stage.processing_size),
                    }
                    for stage in stages
                ],
                "seed": resolved_seed,
                "exact_steps": self.config.exact_steps,
                "strengths": {
                    "anchor": self.config.anchor_strength,
                    "global": self.config.global_overdraw_strength,
                    "face": self.config.face_strength,
                    "hair": self.config.hair_strength,
                    "material": self.config.material_strength,
                    "micro": self.config.micro_strength,
                },
                "candidate_counts": {
                    "global": self.config.global_candidates,
                    "face": self.config.face_candidates,
                    "hair": self.config.hair_candidates,
                    "material": self.config.material_candidates,
                },
                "tile": {
                    "input": self.config.tile_input_size,
                    "core": self.config.core_size,
                    "context": self.config.context_size,
                    "stride": self.config.stride,
                },
                "frequency_gains": asdict(self.config.frequency_gains),
                "detector_provider": self._analysis.detector_provider,
                "detected_faces": len(self._analysis.face_detections),
                "identity_provider": "None",
                "structure_conditioner": self.config.structure_conditioner,
                "back_projection": {
                    "iterations": self.config.back_projection_iterations,
                    "beta": self.config.back_projection_beta,
                },
                "coordinate_noise": {
                    "provider": "BLAKE2b + PCG64 global latent canvas",
                    "scope": "initial latent noise",
                    "shared_anchor_family": self.config.share_anchor_noise_family,
                },
                "debug_enabled": self.config.save_debug_images,
                "processing_time_seconds": processing_time,
                "memmap_usage": memory_report["memmap"],
                "memory": memory_report,
                "runtime": runtime_metrics,
                "final_face_metrics": stage_reports[-1][
                    "final_face_metrics"
                ],
                "selected_face_candidate_roundtrip_ssim": stage_reports[-1][
                    "selected_face_candidate_roundtrip_ssim"
                ],
                "final_face_roundtrip_ssim": stage_reports[-1][
                    "final_face_roundtrip_ssim"
                ],
                "final_face_edge_f1": stage_reports[-1][
                    "final_face_edge_f1"
                ],
                "final_face_roundtrip_ssim_mean": stage_reports[-1][
                    "final_face_roundtrip_ssim_mean"
                ],
                "final_face_edge_f1_mean": stage_reports[-1][
                    "final_face_edge_f1_mean"
                ],
                "final_face_edge_f1_min": stage_reports[-1][
                    "final_face_edge_f1_min"
                ],
                "quality": {
                    "stage_reports": stage_reports,
                    "final_face_metrics": stage_reports[-1][
                        "final_face_metrics"
                    ],
                },
                "model": self.generator.model_metadata(),
                "messages": messages,
            }
            final = linear_rgb_to_image(current_linear, current_alpha)
            metadata_json = json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
                allow_nan=False,
            )
            final.info["hyperweave"] = metadata_json
            if self.config.save_debug_images:
                if self.config.save_metrics_json:
                    self._debug.save_json("_hw_metrics.json", metrics)
                    self._debug.save_json(
                        "_hw_settings.json", self.config.metadata_dict()
                    )
                if self.config.save_metrics_csv:
                    self._debug.save_csv(
                        "_hw_candidates.csv", self._candidate_rows
                    )
                destination = (
                    Path(debug_destination)
                    if debug_destination is not None
                    else Path(self.config.debug_output_directory)
                    if self.config.debug_output_directory
                    else Path.cwd() / "outputs" / "hyperweave_debug"
                )
                debug_files = self._debug.publish(destination)

        logger.info(
            "HyperWeave complete target=%dx%d time=%.3fs peak_allocated=%s "
            "peak_reserved=%s temp_cleanup=%s",
            target[0],
            target[1],
            processing_time,
            runtime_metrics.get("peak_allocated_bytes", "unavailable"),
            runtime_metrics.get("peak_reserved_bytes", "unavailable"),
            not self._work_directory.exists(),
        )
        self._noise.clear()
        self._work_directory = None
        return HyperWeaveResult(
            image=final,
            resolved_seed=resolved_seed,
            metadata=metadata,
            metrics=metrics,
            messages=messages,
            debug_files=debug_files,
            last_processed=self.generator.last_processed,
        )
