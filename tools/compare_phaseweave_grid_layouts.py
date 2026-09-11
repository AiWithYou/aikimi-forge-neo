"""Compare the previous edge-anchored PhaseWeave grid with the balanced grid."""

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
    full_rgb_error,
    load_font,
    load_rgb,
    luminance,
    make_difference_heatmap,
    read_json,
    sha256,
)
from tools.evaluate_krea2_phaseweave_result import REGIONS


def validate_stage(manifest: dict, size: tuple[int, int]) -> dict:
    if manifest.get("merge_mode") != "phase_weave":
        raise ValueError("comparison manifest is not a PhaseWeave run")
    stage = final_stage(manifest, size)
    if stage.get("processed_tile_count") != stage.get("tile_count"):
        raise ValueError("comparison run did not process every tile")
    if stage.get("skipped_tile_count") != 0:
        raise ValueError("comparison run contains skipped tiles")
    return stage


def maximum_neighbor_overlap(stage: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for phase in (0, 1):
        phase_tiles = [tile for tile in stage["tiles"] if int(tile["phase"]) == phase]
        for axis in ("x", "y"):
            spans = sorted(
                {
                    (
                        int(tile[f"core_{axis}0"]),
                        int(tile[f"core_{axis}1"]),
                    )
                    for tile in phase_tiles
                }
            )
            overlaps = [
                max(0, min(left[1], right[1]) - max(left[0], right[0]))
                for left, right in zip(spans, spans[1:])
            ]
            result[f"phase_{phase}_{axis}"] = max(overlaps, default=0)
    result["maximum"] = max(result.values(), default=0)
    return result


def regional_comparison(
    reference: Image.Image,
    previous: Image.Image,
    improved: Image.Image,
) -> dict:
    result = {}
    for name, (box, label) in REGIONS.items():
        arrays = [luminance(np.asarray(image.crop(box))) for image in (reference, previous, improved)]
        highpass = [np.abs(values - box_lowpass(values, 2)) for values in arrays]
        means = [float(np.mean(values, dtype=np.float64)) for values in highpass]
        result[name] = {
            "label": label,
            "box": list(box),
            "resize_only_highpass_mean": means[0],
            "previous_highpass_mean": means[1],
            "improved_highpass_mean": means[2],
            "improved_to_previous_highpass_ratio": means[2] / max(means[1], 1e-9),
        }
    return result


def draw_overview(
    reference: Image.Image,
    previous: Image.Image,
    improved: Image.Image,
    output: Path,
) -> None:
    canvas = Image.new("RGB", (2180, 1240), "#F7F5F1")
    draw = ImageDraw.Draw(canvas)
    draw.text((1090, 30), "格子配置を直す前後の4K比較", font=load_font(42, bold=True), fill="#27212E", anchor="ma")
    panels = (
        ("入力補間", "生成なし", reference, "#68717C"),
        ("修正前", "端へ強制配置", previous, "#9A6947"),
        ("改善版", "端を均等化", improved, "#6E3D83"),
    )
    for index, (title, note, image, color) in enumerate(panels):
        x = 70 + index * 710
        panel = fit(image, (610, 990))
        canvas.paste(panel, (x, 105))
        draw.rectangle((x, 105, x + 610, 1095), outline=color, width=3)
        draw.text((x + 305, 1130), title, font=load_font(27, bold=True), fill=color, anchor="ma")
        draw.text((x + 305, 1175), note, font=load_font(20), fill="#60636B", anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def draw_contact_sheet(
    reference: Image.Image,
    previous: Image.Image,
    improved: Image.Image,
    output: Path,
) -> None:
    panel_width, panel_height = 500, 390
    row_height = 445
    canvas = Image.new("RGB", (1900, 145 + len(REGIONS) * row_height), "#F7F5F1")
    draw = ImageDraw.Draw(canvas)
    draw.text((950, 22), "同一座標の細部比較", font=load_font(38, bold=True), fill="#27212E", anchor="ma")
    for x, label, color in (
        (505, "入力補間", "#68717C"),
        (1050, "修正前", "#9A6947"),
        (1595, "改善版", "#6E3D83"),
    ):
        draw.text((x, 78), label, font=load_font(23, bold=True), fill=color, anchor="ma")
    for row, (name, (box, label)) in enumerate(REGIONS.items()):
        y = 115 + row * row_height
        draw.multiline_text((18, y + 165), f"{name}\n{label}", font=load_font(20, bold=True), fill="#353039", anchor="lm", spacing=8)
        for column, image in enumerate((reference, previous, improved)):
            x = 255 + column * 545
            panel = fit(image.crop(box), (panel_width, panel_height))
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + panel_width, y + panel_height), outline="#90919A", width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--improved", required=True, type=Path)
    parser.add_argument("--previous-manifest", required=True, type=Path)
    parser.add_argument("--improved-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = load_rgb(args.source)
    previous = load_rgb(args.previous)
    improved = load_rgb(args.improved)
    if previous.size != improved.size:
        raise ValueError("previous and improved results have different sizes")
    previous_manifest = read_json(args.previous_manifest)
    improved_manifest = read_json(args.improved_manifest)
    previous_stage = validate_stage(previous_manifest, previous.size)
    improved_stage = validate_stage(improved_manifest, improved.size)
    reference = source.resize(improved.size, Image.Resampling.LANCZOS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = args.output_dir / "grid_layout_before_after_overview.png"
    contact = args.output_dir / "grid_layout_before_after_crops.png"
    heatmap = args.output_dir / "grid_layout_before_after_heatmap.png"
    draw_overview(reference, previous, improved, overview)
    draw_contact_sheet(reference, previous, improved, contact)
    heatmap_scale = make_difference_heatmap(previous, improved, heatmap)

    previous_metrics = candidate_metrics(reference, previous, previous_stage)
    improved_metrics = candidate_metrics(reference, improved, improved_stage)
    report = {
        "format_version": 1,
        "comparison_scope": "same source, profile, target, prompt, and global seed; grid coordinates and therefore coordinate-derived tile seeds differ",
        "artifacts": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "previous": {"path": str(args.previous), "sha256": sha256(args.previous)},
            "improved": {"path": str(args.improved), "sha256": sha256(args.improved)},
            "overview": str(overview),
            "contact_sheet": str(contact),
            "difference_heatmap": str(heatmap),
        },
        "geometry": {
            "previous_layout": previous_manifest.get("grid_layout", "legacy_edge_anchored"),
            "improved_layout": improved_manifest.get("grid_layout"),
            "improved_origin": improved_manifest.get("grid_origin"),
            "previous_neighbor_overlap": maximum_neighbor_overlap(previous_stage),
            "improved_neighbor_overlap": maximum_neighbor_overlap(improved_stage),
        },
        "previous_metrics": previous_metrics,
        "improved_metrics": improved_metrics,
        "improved_vs_previous": {
            **full_rgb_error(previous, improved),
            "boundary_ratio_change": improved_metrics["residual_boundary_to_global_p95_ratio"] - previous_metrics["residual_boundary_to_global_p95_ratio"],
            "highpass_ratio_change": improved_metrics["highpass_mean_ratio"] - previous_metrics["highpass_mean_ratio"],
            "difference_heatmap_p99_scale": heatmap_scale,
        },
        "regions": regional_comparison(reference, previous, improved),
    }
    report_path = args.output_dir / "grid_layout_before_after_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "geometry": report["geometry"], "improved_vs_previous": report["improved_vs_previous"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
