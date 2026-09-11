"""Evaluate one Krea2 PhaseWeave 4K result against a resize-only reference.

The resize-only image is not a quality baseline.  It preserves the source pixels at
the target dimensions so that reviewers can distinguish newly reconstructed detail
from interpolation.  The script also checks the recorded tile run and writes
same-coordinate visual comparisons for manual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_krea2_4k import (
    box_lowpass,
    candidate_metrics,
    final_stage,
    fit,
    load_font,
    load_rgb,
    local_luma_ssim,
    luminance,
    read_json,
    sha256,
)
from modules_forge.vram_canvas import axis_positions, balanced_virtual_axis_origin


CANONICAL_REGION_SIZE = (2897, 4096)
REGIONS = {
    "A": ((420, 820, 1370, 1770), "青い髪・目・細い輪郭"),
    "B": ((1570, 850, 2670, 1900), "白い髪・目・衣装"),
    "C": ((1090, 1510, 2390, 2670), "本・指・布の縫い目"),
    "D": ((850, 2690, 2240, 4050), "スライム・反射・敷物"),
    "E": ((0, 0, 1420, 1120), "本棚・本・木目"),
}


def regions_for_size(
    size: tuple[int, int],
) -> dict[str, tuple[tuple[int, int, int, int], str]]:
    """Scale the review regions from the original 4K sheet to any delivery size."""

    width, height = size
    reference_width, reference_height = CANONICAL_REGION_SIZE
    scale_x = width / reference_width
    scale_y = height / reference_height
    scaled = {}
    for name, (box, label) in REGIONS.items():
        x0, y0, x1, y1 = box
        scaled[name] = (
            (
                max(0, min(width - 1, round(x0 * scale_x))),
                max(0, min(height - 1, round(y0 * scale_y))),
                max(1, min(width, round(x1 * scale_x))),
                max(1, min(height, round(y1 * scale_y))),
            ),
            label,
        )
    return scaled


def read_png_metadata(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        raw = image.info.get("krea2_phaseweave")
        return {
            "keys": sorted(image.info),
            "krea2_phaseweave": json.loads(raw) if raw else None,
        }


def validate_run(manifest: dict, result_size: tuple[int, int], metadata: dict) -> dict:
    if manifest.get("merge_mode") != "phase_weave":
        raise ValueError("manifest is not a PhaseWeave run")
    if manifest.get("krea2_profile") != "phaseweave_4k":
        raise ValueError("manifest profile is not phaseweave_4k")
    if manifest.get("exact_img2img_steps") is not True:
        raise ValueError("manifest does not confirm exact img2img steps")
    if manifest.get("exact_img2img_steps_scope") != "internal_tiles_only":
        raise ValueError("exact-step scope is not internal_tiles_only")
    core_size = int(manifest.get("core_size", 0))
    core_overlap = int(manifest.get("core_overlap", -1))
    phase_count = int(manifest.get("phase_count", 0))
    tile_size = int(manifest.get("tile_size", 0))
    if core_size <= 0 or not 0 <= core_overlap < core_size:
        raise ValueError("manifest core size or overlap is invalid")
    if phase_count != 2:
        raise ValueError(f"manifest phase_count={phase_count!r}; expected 2")
    if tile_size <= core_size:
        raise ValueError("manifest tile size must be larger than the core size")
    stride = core_size - core_overlap
    phase_offset = int(round(stride / phase_count))
    expected_origin = [
        balanced_virtual_axis_origin(
            result_size[0],
            core_size,
            core_overlap,
            phase_count=phase_count,
        ),
        balanced_virtual_axis_origin(
            result_size[1],
            core_size,
            core_overlap,
            phase_count=phase_count,
        ),
    ]
    expected_grid = {
        "grid_layout": "uniform_virtual_edge_balanced",
        "grid_stride": stride,
        "grid_phase_offset": phase_offset,
        "grid_padding_mode": "edge",
        "grid_origin": expected_origin,
    }
    for key, value in expected_grid.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"manifest {key}={manifest.get(key)!r}; expected {value!r}"
            )

    stage = final_stage(manifest, result_size)
    if stage.get("processed_tile_count") != stage.get("tile_count"):
        raise ValueError("not every tile was processed")
    if stage.get("skipped_tile_count") != 0:
        raise ValueError("the final stage contains skipped tiles")
    if stage.get("grid_origin") != expected_origin:
        raise ValueError("final-stage grid origin is not edge-balanced")

    tiles = stage.get("tiles") or []
    expected_axes = {}
    for phase in range(phase_count):
        expected_axes[phase] = {
            "x": axis_positions(
                result_size[0],
                core_size,
                core_overlap,
                phase=phase,
                phase_count=phase_count,
                virtual_padding=True,
            ),
            "y": axis_positions(
                result_size[1],
                core_size,
                core_overlap,
                phase=phase,
                phase_count=phase_count,
                virtual_padding=True,
            ),
        }
    for phase, axes in expected_axes.items():
        phase_tiles = [tile for tile in tiles if int(tile["phase"]) == phase]
        actual_x = sorted({int(tile["grid_core_x0"]) for tile in phase_tiles})
        actual_y = sorted({int(tile["grid_core_y0"]) for tile in phase_tiles})
        if actual_x != axes["x"] or actual_y != axes["y"]:
            raise ValueError(f"phase {phase} grid coordinates are not reproducible")
    context_sizes = {
        (
            int(tile["context_x1"]) - int(tile["context_x0"]),
            int(tile["context_y1"]) - int(tile["context_y0"]),
        )
        for tile in tiles
    }
    if context_sizes != {(tile_size, tile_size)}:
        raise ValueError(f"tile payload sizes are not uniform: {context_sizes}")
    minimum_width = min(int(tile["core_x1"]) - int(tile["core_x0"]) for tile in tiles)
    minimum_height = min(int(tile["core_y1"]) - int(tile["core_y0"]) for tile in tiles)
    if minimum_width <= 0 or minimum_height <= 0:
        raise ValueError("a planned tile does not intersect the output canvas")

    identity = metadata.get("krea2_phaseweave")
    if not identity:
        raise ValueError("PNG metadata krea2_phaseweave is missing")
    expected = {
        "product_name": "Krea2 PhaseWeave 4K",
        "profile_key": "phaseweave_4k",
        "merge_mode": "phase_weave",
        "exact_img2img_steps": True,
        "exact_img2img_steps_scope": "internal_tiles_only",
        **expected_grid,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(
                f"PNG metadata {key}={identity.get(key)!r}; expected {value!r}"
            )
    return stage


def region_metrics(
    reference: Image.Image,
    result: Image.Image,
    regions: dict[str, tuple[tuple[int, int, int, int], str]],
) -> dict:
    metrics = {}
    for name, (box, label) in regions.items():
        before_y = luminance(np.asarray(reference.crop(box)))
        after_y = luminance(np.asarray(result.crop(box)))
        before_hp = np.abs(before_y - box_lowpass(before_y, 2))
        after_hp = np.abs(after_y - box_lowpass(after_y, 2))
        before_mean = float(np.mean(before_hp, dtype=np.float64))
        after_mean = float(np.mean(after_hp, dtype=np.float64))
        metrics[name] = {
            "label": label,
            "box": list(box),
            "resize_only_highpass_mean": before_mean,
            "phaseweave_highpass_mean": after_mean,
            "phaseweave_to_resize_only_highpass_ratio": after_mean
            / max(before_mean, 1e-9),
            "luma_ssim_to_resize_only": local_luma_ssim(before_y, after_y),
        }
    return metrics


def draw_overview(reference: Image.Image, result: Image.Image, output: Path) -> None:
    canvas = Image.new("RGB", (1500, 1200), "#F7F5F1")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (750, 32),
        "同じ入力からの高解像度比較",
        font=load_font(42, bold=True),
        fill="#27212E",
        anchor="ma",
    )
    panels = (
        ("入力を同寸法に拡大しただけ", "新しい描き込みなし", reference, "#68717C"),
        ("PhaseWeave 4K", "細部を再構成", result, "#7A3D87"),
    )
    for index, (title, note, image, color) in enumerate(panels):
        x = 90 + index * 720
        panel = fit(image, (600, 980))
        canvas.paste(panel, (x, 105))
        draw.rectangle((x, 105, x + 600, 1085), outline=color, width=3)
        draw.text(
            (x + 300, 1118),
            title,
            font=load_font(26, bold=True),
            fill=color,
            anchor="ma",
        )
        draw.text(
            (x + 300, 1160),
            note,
            font=load_font(20),
            fill="#60636B",
            anchor="ma",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def draw_contact_sheet(
    reference: Image.Image,
    result: Image.Image,
    regions: dict[str, tuple[tuple[int, int, int, int], str]],
    output: Path,
) -> None:
    panel_width = 610
    panel_height = 410
    row_height = 460
    canvas = Image.new(
        "RGB", (1580, 150 + len(regions) * row_height), "#F7F5F1"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (790, 24),
        "同じ場所を拡大して比較",
        font=load_font(38, bold=True),
        fill="#27212E",
        anchor="ma",
    )
    for x, label, color in (
        (560, "入力を同寸法に拡大しただけ", "#68717C"),
        (1210, "PhaseWeave 4K", "#7A3D87"),
    ):
        draw.text(
            (x, 80), label, font=load_font(23, bold=True), fill=color, anchor="ma"
        )

    for row, (name, (box, label)) in enumerate(regions.items()):
        y = 120 + row * row_height
        draw.multiline_text(
            (18, y + 175),
            f"{name}\n{label}",
            font=load_font(21, bold=True),
            fill="#353039",
            anchor="lm",
            spacing=8,
        )
        for column, image in enumerate((reference, result)):
            x = 255 + column * 650
            panel = fit(image.crop(box), (panel_width, panel_height))
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + panel_width, y + panel_height),
                outline="#90919A",
                width=2,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def draw_crop_pair(
    name: str,
    label: str,
    box: tuple[int, int, int, int],
    reference: Image.Image,
    result: Image.Image,
    output: Path,
) -> None:
    crops = (reference.crop(box), result.crop(box))
    crop_width, crop_height = crops[0].size
    gap = 24
    header = 118
    canvas = Image.new(
        "RGB",
        (crop_width * 2 + gap * 3, crop_height + header + 20),
        "#F7F5F1",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 18),
        f"{name}  {label}",
        font=load_font(29, bold=True),
        fill="#27212E",
        anchor="la",
    )
    for index, (crop, panel_label, color) in enumerate(
        zip(
            crops,
            ("入力を同寸法に拡大しただけ", "PhaseWeave 4K"),
            ("#68717C", "#7A3D87"),
        )
    ):
        x = gap + index * (crop_width + gap)
        canvas.paste(crop, (x, header))
        draw.rectangle(
            (x, header, x + crop_width, header + crop_height),
            outline=color,
            width=4,
        )
        draw.text(
            (x + crop_width // 2, 82),
            panel_label,
            font=load_font(21, bold=True),
            fill=color,
            anchor="mm",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = load_rgb(args.source)
    result = load_rgb(args.result)
    manifest = read_json(args.manifest)
    metadata = read_png_metadata(args.result)
    stage = validate_run(manifest, result.size, metadata)
    reference = source.resize(result.size, Image.Resampling.LANCZOS)
    regions = regions_for_size(result.size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resize_only_path = (
        args.output_dir / f"resize_only_{result.width}x{result.height}.png"
    )
    overview_path = args.output_dir / "overview_resize_only_vs_phaseweave.png"
    contact_path = args.output_dir / "crop_contact_sheet.png"
    reference.save(resize_only_path, format="PNG", optimize=True)
    draw_overview(reference, result, overview_path)
    draw_contact_sheet(reference, result, regions, contact_path)
    for name, (box, label) in regions.items():
        draw_crop_pair(
            name,
            label,
            box,
            reference,
            result,
            args.output_dir / f"crop_{name}_pair.png",
        )

    measurements = candidate_metrics(reference, result, stage)
    region_measurements = region_metrics(reference, result, regions)
    stats = stage.get("consensus_stats") or {}
    phase_one_percent = float(stats.get("phaseweave_phase1_selected_percent", -1))
    boundary_percent = float(stats.get("phaseweave_boundary_percent", 100))
    tiles = stage.get("tiles") or []
    minimum_canvas_intersection = [
        min(int(tile["core_x1"]) - int(tile["core_x0"]) for tile in tiles),
        min(int(tile["core_y1"]) - int(tile["core_y0"]) for tile in tiles),
    ]
    checks = {
        "target_matches_manifest": list(result.size) == manifest.get("target_size"),
        "phaseweave_identity_metadata_matches": True,
        "exact_steps_internal_only_recorded": True,
        "all_tiles_processed": stage.get("processed_tile_count")
        == stage.get("tile_count"),
        "no_tiles_skipped": stage.get("skipped_tile_count") == 0,
        "uniform_edge_balanced_grid": True,
        "all_model_inputs_match_recorded_tile_size": True,
        "all_canvas_intersections_are_nonempty": min(minimum_canvas_intersection)
        > 0,
        "both_shifted_divisions_selected": 5 < phase_one_percent < 95,
        "transition_area_below_40_percent": boundary_percent < 40,
        "single_representative_area_above_60_percent": 100.0 - boundary_percent > 60,
        "planned_boundary_jump_ratio_below_1_5": measurements[
            "residual_boundary_to_global_p95_ratio"
        ]
        < 1.5,
        "manual_visual_review_required": True,
    }
    report = {
        "format_version": 1,
        "method": {
            "comparison": "same source resized with Lanczos versus the PhaseWeave result",
            "resize_only_role": "same-size pixel reference, not a semantic quality baseline",
            "highpass": "absolute luminance residual from a separable box low-pass",
            "seam": "p95 generated-residual jump at recorded core boundaries divided by the global residual-jump p95",
            "caution": "high-frequency energy includes useful drawing, sharpening, and noise; visual review decides quality",
        },
        "artifacts": {
            "source": {
                "path": str(args.source),
                "size": list(source.size),
                "sha256": sha256(args.source),
            },
            "result": {
                "path": str(args.result),
                "size": list(result.size),
                "sha256": sha256(args.result),
            },
            "manifest": str(args.manifest),
            "resize_only": str(resize_only_path),
            "overview": str(overview_path),
            "contact_sheet": str(contact_path),
        },
        "run": {
            "tile_count": stage.get("tile_count"),
            "processed_tile_count": stage.get("processed_tile_count"),
            "skipped_tile_count": stage.get("skipped_tile_count"),
            "phase_one_selected_percent": phase_one_percent,
            "transition_area_percent": boundary_percent,
            "mean_selected_detail_retained": stats.get(
                "phaseweave_mean_detail_gain"
            ),
            "grid_layout": manifest.get("grid_layout"),
            "grid_origin": manifest.get("grid_origin"),
            "grid_stride": manifest.get("grid_stride"),
            "grid_phase_offset": manifest.get("grid_phase_offset"),
            "minimum_canvas_intersection": minimum_canvas_intersection,
        },
        "png_metadata": metadata,
        "measurements": measurements,
        "regions": region_measurements,
        "checks": checks,
    }
    report_path = args.output_dir / "phaseweave_result_qa.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "whole_image_highpass_ratio": measurements["highpass_mean_ratio"],
                "luma_ssim_to_resize_only": measurements[
                    "luma_ssim_half_resolution"
                ],
                "planned_boundary_jump_ratio": measurements[
                    "residual_boundary_to_global_p95_ratio"
                ],
                "regions": {
                    name: round(
                        value["phaseweave_to_resize_only_highpass_ratio"], 4
                    )
                    for name, value in region_measurements.items()
                },
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
