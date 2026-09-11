"""Reusable comparison metrics and visual artifacts for upscale methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import image_to_linear_rgb, linear_rgb_to_image, luminance
from .frequency import FrequencyDecomposer, StructureMapBuilder, gaussian_blur
from .geometry import TilePlanner
from .quality import SeamAnalyzer, roundtrip_metrics
from .regions import hair_flow_score
from .scoring_features import spectral_flatness


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _label(image: Image.Image, text: str, height: int = 28) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), text, fill="black", font=ImageFont.load_default())
    return canvas


def _hyperweave_manifest(image: Image.Image) -> dict[str, object] | None:
    raw = image.info.get("hyperweave")
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_metrics(manifest: dict[str, object] | None) -> dict[str, object]:
    result: dict[str, object] = {
        "face_structure_score": None,
        "selected_face_candidate_roundtrip_ssim": None,
        "final_face_roundtrip_ssim": None,
        "final_face_edge_f1": None,
        "seam_ratio": None,
        "processing_time_seconds": None,
        "peak_vram_bytes": None,
        "peak_allocated_vram_bytes": None,
        "peak_reserved_vram_bytes": None,
        "ram_estimate_bytes": None,
        "disk_estimate_bytes": None,
        "memmap_usage": None,
    }
    if manifest is None:
        return result
    result["processing_time_seconds"] = manifest.get("processing_time_seconds")
    result["memmap_usage"] = manifest.get("memmap_usage")
    memory = manifest.get("memory")
    if isinstance(memory, dict):
        result["ram_estimate_bytes"] = memory.get("working_ram_estimate_bytes")
        result["disk_estimate_bytes"] = memory.get("disk_estimate_bytes")
    runtime = manifest.get("runtime")
    if isinstance(runtime, dict):
        allocated = runtime.get("peak_allocated_bytes")
        reserved = runtime.get("peak_reserved_bytes")
        result["peak_allocated_vram_bytes"] = allocated
        result["peak_reserved_vram_bytes"] = reserved
        numeric = [
            int(value)
            for value in (allocated, reserved)
            if isinstance(value, (int, float))
        ]
        result["peak_vram_bytes"] = max(numeric) if numeric else None

    quality = manifest.get("quality")
    stage_reports = (
        quality.get("stage_reports")
        if isinstance(quality, dict)
        else None
    )
    if isinstance(stage_reports, list):
        seam_values: list[float] = []
        selected_face_values: list[float] = []
        final_face_values: list[float] = []
        final_edge_values: list[float] = []
        for report in stage_reports:
            if not isinstance(report, dict):
                continue
            seam = report.get("seam")
            if isinstance(seam, dict) and isinstance(
                seam.get("ratio"), (int, float)
            ):
                seam_values.append(float(seam["ratio"]))
            rois = report.get("rois")
            if isinstance(rois, list):
                for roi in rois:
                    if not isinstance(roi, dict) or roi.get("kind") != "face":
                        continue
                    score = roi.get("selected_score")
                    roundtrip = (
                        score.get("roundtrip")
                        if isinstance(score, dict)
                        else None
                    )
                    if isinstance(roundtrip, dict) and isinstance(
                        roundtrip.get("ssim"), (int, float)
                    ):
                        selected_face_values.append(float(roundtrip["ssim"]))
            final_faces = report.get("final_face_metrics")
            if isinstance(final_faces, list) and final_faces:
                stage_ssim = [
                    float(item["roundtrip_ssim"])
                    for item in final_faces
                    if isinstance(item, dict)
                    and isinstance(item.get("roundtrip_ssim"), (int, float))
                ]
                stage_edge = [
                    float(item["edge_f1"])
                    for item in final_faces
                    if isinstance(item, dict)
                    and isinstance(item.get("edge_f1"), (int, float))
                ]
                if stage_ssim:
                    final_face_values = stage_ssim
                if stage_edge:
                    final_edge_values = stage_edge
        result["seam_ratio"] = max(seam_values) if seam_values else None
        result["selected_face_candidate_roundtrip_ssim"] = (
            float(np.mean(selected_face_values))
            if selected_face_values
            else None
        )
        result["final_face_roundtrip_ssim"] = (
            float(np.mean(final_face_values)) if final_face_values else None
        )
        result["final_face_edge_f1"] = (
            float(np.mean(final_edge_values)) if final_edge_values else None
        )
        result["face_structure_score"] = (
            result["final_face_roundtrip_ssim"]
            if result["final_face_roundtrip_ssim"] is not None
            else result["selected_face_candidate_roundtrip_ssim"]
        )
        for report in reversed(stage_reports):
            if not isinstance(report, dict):
                continue
            if (
                result["selected_face_candidate_roundtrip_ssim"] is None
                and isinstance(
                    report.get("selected_face_candidate_roundtrip_ssim"),
                    (int, float),
                )
            ):
                result["selected_face_candidate_roundtrip_ssim"] = float(
                    report["selected_face_candidate_roundtrip_ssim"]
                )
            if (
                result["final_face_roundtrip_ssim"] is None
                and isinstance(
                    report.get("final_face_roundtrip_ssim"), (int, float)
                )
            ):
                result["final_face_roundtrip_ssim"] = float(
                    report["final_face_roundtrip_ssim"]
                )
            if (
                result["final_face_edge_f1"] is None
                and isinstance(report.get("final_face_edge_f1"), (int, float))
            ):
                result["final_face_edge_f1"] = float(
                    report["final_face_edge_f1"]
                )
            if (
                result["selected_face_candidate_roundtrip_ssim"] is not None
                and result["final_face_roundtrip_ssim"] is not None
                and result["final_face_edge_f1"] is not None
            ):
                break
        result["face_structure_score"] = (
            result["final_face_roundtrip_ssim"]
            if result["final_face_roundtrip_ssim"] is not None
            else result["selected_face_candidate_roundtrip_ssim"]
        )
    if result["final_face_roundtrip_ssim"] is None:
        top_level_faces = manifest.get("final_face_metrics")
        if isinstance(top_level_faces, list):
            ssim_values = [
                float(item["roundtrip_ssim"])
                for item in top_level_faces
                if isinstance(item, dict)
                and isinstance(item.get("roundtrip_ssim"), (int, float))
            ]
            edge_values = [
                float(item["edge_f1"])
                for item in top_level_faces
                if isinstance(item, dict)
                and isinstance(item.get("edge_f1"), (int, float))
            ]
            if ssim_values:
                result["final_face_roundtrip_ssim"] = float(
                    np.mean(ssim_values)
                )
                result["face_structure_score"] = result[
                    "final_face_roundtrip_ssim"
                ]
            if edge_values:
                result["final_face_edge_f1"] = float(np.mean(edge_values))
    if result["selected_face_candidate_roundtrip_ssim"] is None and isinstance(
        manifest.get("selected_face_candidate_roundtrip_ssim"), (int, float)
    ):
        result["selected_face_candidate_roundtrip_ssim"] = float(
            manifest["selected_face_candidate_roundtrip_ssim"]
        )
    if result["final_face_roundtrip_ssim"] is None and isinstance(
        manifest.get("final_face_roundtrip_ssim"), (int, float)
    ):
        result["final_face_roundtrip_ssim"] = float(
            manifest["final_face_roundtrip_ssim"]
        )
    if result["final_face_edge_f1"] is None and isinstance(
        manifest.get("final_face_edge_f1"), (int, float)
    ):
        result["final_face_edge_f1"] = float(
            manifest["final_face_edge_f1"]
        )
    result["face_structure_score"] = (
        result["final_face_roundtrip_ssim"]
        if result["final_face_roundtrip_ssim"] is not None
        else result["selected_face_candidate_roundtrip_ssim"]
    )
    return result


def comparison_metrics(
    source: Image.Image, candidate: Image.Image
) -> dict[str, object]:
    source_linear, _ = image_to_linear_rgb(source.convert("RGB"))
    candidate_linear, _ = image_to_linear_rgb(candidate.convert("RGB"))
    roundtrip = roundtrip_metrics(source_linear, candidate_linear)
    resized_source = cv2.resize(
        source_linear,
        (candidate.width, candidate.height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    residual = candidate_linear - resized_source
    bands = FrequencyDecomposer().decompose(luminance(residual))
    energy = FrequencyDecomposer.energy(bands)
    orientation = hair_flow_score(
        resized_source,
        candidate_linear,
        np.ones(candidate_linear.shape[:2], dtype=np.float32),
    )
    residual_y = luminance(residual)
    noise_flatness = spectral_flatness(residual_y)
    metrics: dict[str, object] = {
        "roundtrip_ssim": roundtrip.ssim,
        "roundtrip_psnr": roundtrip.psnr,
        "low_frequency_luminance_error": roundtrip.low_frequency_mse,
        "color_drift": roundtrip.color_drift,
        "edge_displacement": roundtrip.edge_displacement,
        "edge_precision": roundtrip.edge_precision,
        "edge_recall": roundtrip.edge_recall,
        "edge_f1": roundtrip.edge_f1,
        "mid_energy": energy["mid"],
        "mid_high_energy": energy["mid_high"],
        "high_energy": energy["high_0"] + energy["high_1"],
        "coherent_line_score": orientation.orientation_alignment,
        "hair_flow_score": orientation.total,
        "noise_penalty": noise_flatness,
    }
    metrics.update(_manifest_metrics(_hyperweave_manifest(candidate)))
    return metrics


def _roundtrip_confidence_visual(
    source: Image.Image, candidate: Image.Image
) -> Image.Image:
    source_linear, _ = image_to_linear_rgb(source.convert("RGB"))
    candidate_linear, _ = image_to_linear_rgb(candidate.convert("RGB"))
    downsampled = cv2.resize(
        candidate_linear,
        source.size,
        interpolation=cv2.INTER_AREA,
    )
    error = np.mean(
        np.abs(
            gaussian_blur(source_linear, 2.0)
            - gaussian_blur(downsampled, 2.0)
        ),
        axis=2,
    )
    confidence = np.exp(-12.0 * error).astype(np.float32)
    confidence = cv2.resize(
        confidence, candidate.size, interpolation=cv2.INTER_LANCZOS4
    )
    return Image.fromarray(
        np.clip(np.rint(confidence * 255), 0, 255).astype(np.uint8),
        mode="L",
    )


def _manifest_boundaries(
    candidate: Image.Image,
) -> tuple[list[int], list[int]]:
    manifest = _hyperweave_manifest(candidate)
    if manifest is None:
        return [], []
    stage_plan = manifest.get("stage_plan")
    tile = manifest.get("tile")
    if not isinstance(stage_plan, list) or not stage_plan or not isinstance(tile, dict):
        return [], []
    final_stage = stage_plan[-1]
    processing = (
        final_stage.get("processing")
        if isinstance(final_stage, dict)
        else None
    )
    if (
        not isinstance(processing, list)
        or len(processing) != 2
        or not all(isinstance(value, int) for value in processing)
    ):
        return [], []
    try:
        planner = TilePlanner(
            int(processing[0]),
            int(processing[1]),
            tile_input_size=int(tile["input"]),
            core_size=int(tile["core"]),
            context_size=int(tile["context"]),
            stride=int(tile["stride"]),
        )
        tiles = planner.plan()
    except (KeyError, TypeError, ValueError):
        return [], []
    vertical = sorted(
        {
            round(item.core_box[0] * candidate.width / int(processing[0]))
            for item in tiles
            if 0 < item.core_box[0] < int(processing[0])
        }
    )
    horizontal = sorted(
        {
            round(item.core_box[1] * candidate.height / int(processing[1]))
            for item in tiles
            if 0 < item.core_box[1] < int(processing[1])
        }
    )
    return vertical, horizontal


def _seam_visual(
    candidate_linear: np.ndarray,
    baseline_linear: np.ndarray,
    vertical: list[int],
    horizontal: list[int],
) -> Image.Image:
    residual = luminance(candidate_linear - baseline_linear)
    result = np.zeros(residual.shape, dtype=np.float32)
    height, width = residual.shape
    for x in vertical:
        if 1 <= x < width:
            discontinuity = np.abs(residual[:, x] - residual[:, x - 1])
            result[:, max(0, x - 2) : min(width, x + 3)] = np.maximum(
                result[:, max(0, x - 2) : min(width, x + 3)],
                discontinuity[:, None],
            )
    for y in horizontal:
        if 1 <= y < height:
            discontinuity = np.abs(residual[y, :] - residual[y - 1, :])
            result[max(0, y - 2) : min(height, y + 3), :] = np.maximum(
                result[max(0, y - 2) : min(height, y + 3), :],
                discontinuity[None, :],
            )
    nonzero = result[result > 0]
    scale = max(
        float(np.percentile(nonzero, 99.0)) if nonzero.size else 0.0,
        1e-6,
    )
    result = np.clip(gaussian_blur(result, 1.0) / scale, 0.0, 1.0)
    return Image.fromarray(
        np.clip(np.rint(result * 255), 0, 255).astype(np.uint8),
        mode="L",
    )


def _band_visual(band: np.ndarray) -> Image.Image:
    scale = max(float(np.percentile(np.abs(band), 99.0)), 1e-6)
    normalized = np.clip(0.5 + band / (2 * scale), 0.0, 1.0)
    return Image.fromarray(
        np.clip(np.rint(normalized * 255), 0, 255).astype(np.uint8), mode="L"
    )


def build_comparison_artifacts(
    source: Image.Image,
    candidates: Mapping[str, Image.Image],
    output_directory: str | Path,
    *,
    crop_boxes: Mapping[str, tuple[int, int, int, int]] | None = None,
) -> dict[str, object]:
    if not candidates:
        raise ValueError("At least one comparison candidate is required.")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    target_size = next(iter(candidates.values())).size
    lanczos = source.resize(target_size, Image.Resampling.LANCZOS)
    methods: dict[str, Image.Image] = {"Lanczos": lanczos, **candidates}
    report = {
        name: comparison_metrics(source, image)
        for name, image in methods.items()
    }
    if crop_boxes:
        face_boxes = [
            box
            for crop_name, box in crop_boxes.items()
            if "face" in crop_name.casefold()
            or "head" in crop_name.casefold()
            or "顔" in crop_name
        ]
        for name, image in methods.items():
            face_scores: list[float] = []
            for box in face_boxes:
                source_box = (
                    round(box[0] * source.width / image.width),
                    round(box[1] * source.height / image.height),
                    round(box[2] * source.width / image.width),
                    round(box[3] * source.height / image.height),
                )
                source_crop = source.crop(source_box).convert("RGB")
                candidate_crop = image.crop(box).convert("RGB")
                if min(source_crop.size) >= 8 and min(candidate_crop.size) >= 8:
                    source_linear, _ = image_to_linear_rgb(source_crop)
                    candidate_linear, _ = image_to_linear_rgb(candidate_crop)
                    face_scores.append(
                        roundtrip_metrics(
                            source_linear, candidate_linear
                        ).ssim
                    )
            if (
                face_scores
                and report[name].get("final_face_roundtrip_ssim") is None
            ):
                report[name]["face_structure_score"] = float(
                    np.mean(face_scores)
                )
    cells = [
        _label(_fit(source, (360, 360)), f"Source {source.width}x{source.height}")
    ]
    cells.extend(
        _label(_fit(image, (360, 360)), f"{name} {image.width}x{image.height}")
        for name, image in methods.items()
    )
    columns = min(3, len(cells))
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 360, rows * 388), (235, 235, 235))
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 360, (index // columns) * 388))
    sheet.save(root / "contact_sheet.png")

    target_source_linear, _ = image_to_linear_rgb(lanczos)
    for name, image in methods.items():
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        candidate_linear, _ = image_to_linear_rgb(image.convert("RGB"))
        residual = candidate_linear - target_source_linear
        residual_y = luminance(residual)
        bands = FrequencyDecomposer().decompose(residual_y)
        difference = np.clip(np.abs(residual) * 5.0, 0.0, 1.0)
        linear_rgb_to_image(difference).save(root / f"{safe_name}_difference.png")
        _band_visual(bands["high_0"] + bands["high_1"]).save(
            root / f"{safe_name}_frequency_high.png"
        )
        _band_visual(bands["mid_high"] + bands["mid"]).save(
            root / f"{safe_name}_frequency_mid.png"
        )
        _band_visual(bands["mid_low"] + bands["low"]).save(
            root / f"{safe_name}_frequency_midlow.png"
        )
        structure = StructureMapBuilder().build(candidate_linear).protection
        Image.fromarray(
            np.clip(np.rint(structure * 255), 0, 255).astype(np.uint8), mode="L"
        ).save(root / f"{safe_name}_structure_map.png")
        _roundtrip_confidence_visual(source, image).save(
            root / f"{safe_name}_confidence_map.png"
        )
        vertical, horizontal = _manifest_boundaries(image)
        _seam_visual(
            candidate_linear,
            target_source_linear,
            vertical,
            horizontal,
        ).save(root / f"{safe_name}_seam_map.png")
        if (
            report[name]["seam_ratio"] is None
            and (vertical or horizontal)
        ):
            report[name]["seam_ratio"] = SeamAnalyzer().analyze(
                residual, vertical, horizontal
            ).ratio

    # Seam ratios derived from old manifests without quality summaries are
    # populated while rendering the maps, so persist the completed report.
    (root / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    if crop_boxes:
        for crop_name, box in crop_boxes.items():
            strip_cells = [
                _label(image.crop(box).resize((320, 320), Image.Resampling.LANCZOS), name)
                for name, image in methods.items()
            ]
            strip = Image.new("RGB", (320 * len(strip_cells), 348), "white")
            for index, cell in enumerate(strip_cells):
                strip.paste(cell, (index * 320, 0))
            safe_crop = "".join(
                ch if ch.isalnum() else "_" for ch in crop_name
            ).strip("_")
            strip.save(root / f"crop_{safe_crop}.png")
    return {
        "output_directory": str(root),
        "metrics": report,
        "methods": list(methods),
    }
