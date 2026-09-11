"""Build an auditable Consensus-versus-PhaseWeave 4K comparison.

The generated metrics separate high-frequency energy from low-frequency drift and
measure jumps at the actual VRAM-Canvas core boundaries.  High-frequency energy is
not treated as a semantic quality score; the original-pixel crops remain the visual
review gate.
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

from tools.compare_krea2_4k import (  # noqa: E402
    box_lowpass,
    candidate_metrics,
    final_stage,
    fit,
    full_rgb_error,
    load_font,
    load_rgb,
    local_luma_ssim,
    luminance,
    read_json,
    sha256,
)


REGIONS = {
    "A": ((500, 900, 1300, 1700), "青髪・睫毛・虹彩"),
    "B": ((1750, 900, 2550, 1700), "白髪・睫毛・衣装輪郭"),
    "C": ((1150, 1550, 2150, 2450), "本・指・レース・縫い目"),
    "D": ((1050, 2850, 2050, 3850), "スライム・反射・透明感"),
    "E": ((0, 0, 1000, 1000), "本棚・木目・小物"),
}

SCHEDULE_KEYS = (
    "phase",
    "index",
    "core_x0",
    "core_y0",
    "core_x1",
    "core_y1",
    "context_x0",
    "context_y0",
    "context_x1",
    "context_y1",
    "seed",
    "steps",
    "skipped",
)


def png_metadata(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        result = {"keys": sorted(image.info)}
        for key in ("vram_canvas", "krea2_phaseweave"):
            raw = image.info.get(key)
            if raw is not None:
                result[key] = json.loads(raw)
        return result


def validate_manifest(manifest: dict, expected_merge: str, size: tuple[int, int]) -> dict:
    if manifest.get("merge_mode") != expected_merge:
        raise ValueError(
            f"expected merge mode {expected_merge}, got {manifest.get('merge_mode')}"
        )
    if manifest.get("phase_count") != 2:
        raise ValueError("comparison requires exactly two grid phases")
    if manifest.get("exact_img2img_steps") is not True:
        raise ValueError("manifest does not confirm Exact Steps")
    if manifest.get("exact_img2img_steps_scope") != "internal_tiles_only":
        raise ValueError("unexpected Exact Steps scope")
    stage = final_stage(manifest, size)
    if stage.get("processed_tile_count") != stage.get("tile_count"):
        raise ValueError("not every final-stage tile was processed")
    if stage.get("skipped_tile_count") != 0:
        raise ValueError("final stage contains skipped tiles")
    return stage


def schedule_signature(manifest: dict) -> list[list[dict]]:
    return [
        [
            {key: tile.get(key) for key in SCHEDULE_KEYS}
            for tile in stage.get("tiles") or []
        ]
        for stage in manifest.get("stage_reports") or []
    ]


def crop_metrics(
    lanczos: Image.Image,
    consensus: Image.Image,
    phaseweave: Image.Image,
) -> dict:
    result = {}
    for name, (box, label) in REGIONS.items():
        base_y = luminance(np.asarray(lanczos.crop(box)))
        consensus_y = luminance(np.asarray(consensus.crop(box)))
        phaseweave_y = luminance(np.asarray(phaseweave.crop(box)))
        consensus_hp = np.abs(consensus_y - box_lowpass(consensus_y, 2))
        phaseweave_hp = np.abs(phaseweave_y - box_lowpass(phaseweave_y, 2))
        result[name] = {
            "label": label,
            "box": list(box),
            "consensus_highpass_mean": float(
                np.mean(consensus_hp, dtype=np.float64)
            ),
            "phaseweave_highpass_mean": float(
                np.mean(phaseweave_hp, dtype=np.float64)
            ),
            "phaseweave_to_consensus_highpass_ratio": float(
                np.mean(phaseweave_hp, dtype=np.float64)
                / max(np.mean(consensus_hp, dtype=np.float64), 1e-9)
            ),
            "consensus_luma_ssim_to_lanczos": local_luma_ssim(
                base_y, consensus_y
            ),
            "phaseweave_luma_ssim_to_lanczos": local_luma_ssim(
                base_y, phaseweave_y
            ),
            "consensus_to_phaseweave_luma_ssim": local_luma_ssim(
                consensus_y, phaseweave_y
            ),
        }
    return result


def make_overview(
    lanczos: Image.Image,
    consensus: Image.Image,
    phaseweave: Image.Image,
    output: Path,
) -> None:
    canvas = Image.new("RGB", (2100, 1240), "#F4F6F8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1050, 32),
        "Krea2 PhaseWeave 4K 実画像比較",
        font=load_font(42, bold=True),
        fill="#17202A",
        anchor="ma",
    )
    panels = (
        ("Lanczos 4K基準", "生成なし・補間のみ", lanczos),
        ("Consensus 4K", "2 phase平均 + consensus gate", consensus),
        ("PhaseWeave 4K", "局所代表phase + 境界weave", phaseweave),
    )
    colors = ("#59646F", "#376C82", "#A34D1F")
    for index, ((label, note, image), color) in enumerate(zip(panels, colors)):
        x = 60 + index * 680
        panel = fit(image, (600, 980))
        canvas.paste(panel, (x, 110))
        draw.rectangle((x, 110, x + 600, 1090), outline=color, width=3)
        draw.text(
            (x + 300, 1125), label, font=load_font(28, bold=True), fill=color, anchor="ma"
        )
        draw.text(
            (x + 300, 1170), note, font=load_font(21), fill="#59646F", anchor="ma"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_contact_sheet(
    lanczos: Image.Image,
    consensus: Image.Image,
    phaseweave: Image.Image,
    output: Path,
) -> None:
    panel_width = 420
    panel_height = 390
    row_height = 430
    canvas = Image.new("RGB", (1680, 165 + len(REGIONS) * row_height), "#F4F6F8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (840, 28),
        "同一4K座標の局所detail比較",
        font=load_font(38, bold=True),
        fill="#17202A",
        anchor="ma",
    )
    for index, label in enumerate(("Lanczos基準", "Consensus", "PhaseWeave")):
        draw.text(
            (460 + index * 480, 86),
            label,
            font=load_font(24, bold=True),
            fill="#176B8A",
            anchor="ma",
        )
    for row, (name, (box, label)) in enumerate(REGIONS.items()):
        y = 130 + row * row_height
        draw.multiline_text(
            (18, y + 170),
            f"{name}\n{label}",
            font=load_font(21, bold=True),
            fill="#27323C",
            anchor="lm",
            spacing=8,
        )
        for col, image in enumerate((lanczos, consensus, phaseweave)):
            x = 250 + col * 480
            panel = fit(image.crop(box), (panel_width, panel_height))
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + panel_width, y + panel_height),
                outline="#818B94",
                width=2,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_original_crop_strip(
    name: str,
    label: str,
    box: tuple[int, int, int, int],
    lanczos: Image.Image,
    consensus: Image.Image,
    phaseweave: Image.Image,
    output: Path,
) -> None:
    crops = [image.crop(box) for image in (lanczos, consensus, phaseweave)]
    width, height = crops[0].size
    gap = 20
    header = 120
    canvas = Image.new("RGB", (width * 3 + gap * 4, height + header + 20), "#F4F6F8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 16),
        f"{name}  {label} — 4K原寸crop",
        font=load_font(30, bold=True),
        fill="#17202A",
        anchor="la",
    )
    labels = ("Lanczos基準", "Consensus", "PhaseWeave")
    colors = ("#59646F", "#376C82", "#A34D1F")
    for index, (crop, panel_label, color) in enumerate(zip(crops, labels, colors)):
        x = gap + index * (width + gap)
        canvas.paste(crop, (x, header))
        draw.rectangle((x, header, x + width, header + height), outline=color, width=4)
        draw.text(
            (x + width // 2, 82),
            panel_label,
            font=load_font(22, bold=True),
            fill=color,
            anchor="mm",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_difference_heatmap(
    consensus: Image.Image,
    phaseweave: Image.Image,
    output: Path,
) -> float:
    size = (consensus.width // 2, consensus.height // 2)
    before = np.asarray(consensus.resize(size, Image.Resampling.LANCZOS), dtype=np.int16)
    after = np.asarray(phaseweave.resize(size, Image.Resampling.LANCZOS), dtype=np.int16)
    magnitude = np.mean(np.abs(after - before), axis=2, dtype=np.float32)
    scale = float(np.percentile(magnitude, 99.0)) or 1.0
    t = np.clip(magnitude / scale, 0.0, 1.0)
    rgb = np.stack(
        (
            np.clip(2.2 * t, 0.0, 1.0),
            np.clip(2.0 * (t - 0.22), 0.0, 1.0),
            np.clip(1.3 - 2.0 * t, 0.0, 1.0),
        ),
        axis=2,
    )
    heatmap = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")
    canvas = Image.new("RGB", (1200, 1850), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (600, 30),
        "Consensus → PhaseWeave 差分強度",
        font=load_font(31, bold=True),
        fill="#17202A",
        anchor="ma",
    )
    panel = fit(heatmap, (1080, 1620))
    canvas.paste(panel, (60, 90))
    draw.rectangle((60, 90, 1140, 1710), outline="#818B94", width=2)
    draw.text(
        (600, 1765),
        f"青=小、黄〜赤=大（半解像度RGB差分、p99={scale:.2f} code）",
        font=load_font(22),
        fill="#59646F",
        anchor="mm",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--consensus", required=True, type=Path)
    parser.add_argument("--phaseweave", required=True, type=Path)
    parser.add_argument("--consensus-manifest", required=True, type=Path)
    parser.add_argument("--phaseweave-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = load_rgb(args.source)
    consensus = load_rgb(args.consensus)
    phaseweave = load_rgb(args.phaseweave)
    if consensus.size != phaseweave.size:
        raise ValueError("Consensus and PhaseWeave output dimensions differ")
    lanczos = source.resize(phaseweave.size, Image.Resampling.LANCZOS)

    consensus_manifest = read_json(args.consensus_manifest)
    phaseweave_manifest = read_json(args.phaseweave_manifest)
    consensus_stage = validate_manifest(
        consensus_manifest, "consensus", consensus.size
    )
    phaseweave_stage = validate_manifest(
        phaseweave_manifest, "phase_weave", phaseweave.size
    )
    schedules_match = schedule_signature(consensus_manifest) == schedule_signature(
        phaseweave_manifest
    )
    if not schedules_match:
        raise ValueError("tile coordinates, seeds, or Exact Step counts differ")

    consensus_png = png_metadata(args.consensus)
    phaseweave_png = png_metadata(args.phaseweave)
    if "krea2_phaseweave" not in phaseweave_png:
        raise ValueError("PhaseWeave PNG metadata is missing krea2_phaseweave")
    phaseweave_identity = phaseweave_png["krea2_phaseweave"]
    expected_identity = {
        "product_name": "Krea2 PhaseWeave 4K",
        "profile_key": "phaseweave_4k",
        "merge_mode": "phase_weave",
        "exact_img2img_steps": True,
        "exact_img2img_steps_scope": "internal_tiles_only",
    }
    for key, expected in expected_identity.items():
        if phaseweave_identity.get(key) != expected:
            raise ValueError(
                f"PhaseWeave PNG metadata {key} is "
                f"{phaseweave_identity.get(key)!r}, expected {expected!r}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = args.output_dir / "overview_lanczos_consensus_phaseweave.png"
    contact = args.output_dir / "crop_contact_sheet.png"
    heatmap = args.output_dir / "consensus_vs_phaseweave_difference_heatmap.png"
    make_overview(lanczos, consensus, phaseweave, overview)
    make_contact_sheet(lanczos, consensus, phaseweave, contact)
    for name, (box, label) in REGIONS.items():
        make_original_crop_strip(
            name,
            label,
            box,
            lanczos,
            consensus,
            phaseweave,
            args.output_dir / f"crop_{name}_original_pixels.png",
        )
    heatmap_scale = make_difference_heatmap(consensus, phaseweave, heatmap)

    consensus_metrics = candidate_metrics(lanczos, consensus, consensus_stage)
    phaseweave_metrics = candidate_metrics(lanczos, phaseweave, phaseweave_stage)
    pairwise = full_rgb_error(consensus, phaseweave)
    eval_size = (consensus.width // 2, consensus.height // 2)
    consensus_y = luminance(
        np.asarray(consensus.resize(eval_size, Image.Resampling.LANCZOS))
    )
    phaseweave_y = luminance(
        np.asarray(phaseweave.resize(eval_size, Image.Resampling.LANCZOS))
    )
    pairwise["luma_ssim_half_resolution"] = local_luma_ssim(
        consensus_y, phaseweave_y
    )
    regions = crop_metrics(lanczos, consensus, phaseweave)
    highpass_ratio = phaseweave_metrics["candidate_highpass_mean"] / max(
        consensus_metrics["candidate_highpass_mean"], 1e-9
    )
    structured_ratio = phaseweave_metrics["structured_region_highpass_mean"] / max(
        consensus_metrics["structured_region_highpass_mean"], 1e-9
    )
    flat_ratio = phaseweave_metrics["flat_region_highpass_mean"] / max(
        consensus_metrics["flat_region_highpass_mean"], 1e-9
    )
    phase_stats = phaseweave_stage.get("consensus_stats") or {}
    checks = {
        "same_tile_seed_step_schedule": schedules_match,
        "exact_steps_recorded_for_both": True,
        "phaseweave_png_metadata_present": "krea2_phaseweave" in phaseweave_png,
        "phaseweave_identity_metadata_matches": True,
        "whole_image_highpass_increased": highpass_ratio > 1.0,
        "structured_region_highpass_increased": structured_ratio > 1.0,
        "all_review_regions_highpass_increased": all(
            value["phaseweave_to_consensus_highpass_ratio"] > 1.0
            for value in regions.values()
        ),
        "phaseweave_seam_ratio_below_1_5": (
            phaseweave_metrics["residual_boundary_to_global_p95_ratio"] < 1.5
        ),
        "phaseweave_boundary_feather_below_35_percent": (
            phase_stats.get("phaseweave_boundary_percent", 100.0) < 35.0
        ),
        "both_phases_selected": (
            5.0
            < phase_stats.get("phaseweave_phase1_selected_percent", -1.0)
            < 95.0
        ),
        "visual_review_required": True,
    }
    report = {
        "format_version": 1,
        "method": {
            "baseline": "same-size sRGB Lanczos upscale of the exact source",
            "controlled_variables": "same source, target, profile numerics, tile geometry, phases, coordinate seeds, Exact Steps, and prompt; only merge_mode differs",
            "ssim": "11x11 uniform-window luminance SSIM at half resolution",
            "highpass": "absolute luminance residual from a separable box low-pass",
            "seam": "p95 generated-residual jump at planned core boundaries divided by global residual-jump p95",
            "interpretation": "High-frequency energy includes coherent drawing, sharpening, and noise; original-pixel crops are the semantic visual gate.",
        },
        "artifacts": {
            "source": {
                "path": str(args.source),
                "size": list(source.size),
                "sha256": sha256(args.source),
            },
            "consensus": {
                "path": str(args.consensus),
                "size": list(consensus.size),
                "sha256": sha256(args.consensus),
            },
            "phaseweave": {
                "path": str(args.phaseweave),
                "size": list(phaseweave.size),
                "sha256": sha256(args.phaseweave),
            },
            "consensus_manifest": str(args.consensus_manifest),
            "phaseweave_manifest": str(args.phaseweave_manifest),
            "overview": str(overview),
            "contact_sheet": str(contact),
            "difference_heatmap": str(heatmap),
        },
        "png_metadata": {
            "consensus": consensus_png,
            "phaseweave": phaseweave_png,
        },
        "consensus_4k": consensus_metrics,
        "phaseweave_4k": phaseweave_metrics,
        "consensus_vs_phaseweave": pairwise,
        "phaseweave_to_consensus_ratios": {
            "whole_image_highpass_mean": highpass_ratio,
            "structured_region_highpass_mean": structured_ratio,
            "flat_region_highpass_mean": flat_ratio,
            "low_frequency_drift_mean": phaseweave_metrics[
                "low_frequency_luma_drift_mean"
            ]
            / max(consensus_metrics["low_frequency_luma_drift_mean"], 1e-9),
        },
        "phaseweave_selection": phase_stats,
        "regions": regions,
        "difference_heatmap_p99_code_value": heatmap_scale,
        "checks": checks,
    }
    report_path = args.output_dir / "phaseweave_4k_qa_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        json.dumps(
            {
                "checks": checks,
                "phaseweave_to_consensus_ratios": report[
                    "phaseweave_to_consensus_ratios"
                ],
                "consensus_seam_ratio": consensus_metrics[
                    "residual_boundary_to_global_p95_ratio"
                ],
                "phaseweave_seam_ratio": phaseweave_metrics[
                    "residual_boundary_to_global_p95_ratio"
                ],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
