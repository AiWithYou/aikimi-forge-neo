"""Compare PhaseWeave's independent A/B candidates with the selected result."""

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
    read_json,
    sha256,
)
from tools.evaluate_krea2_phaseweave_result import REGIONS


def _validate_sizes(images: dict[str, Image.Image]) -> tuple[int, int]:
    sizes = {image.size for image in images.values()}
    if len(sizes) != 1:
        raise ValueError(f"comparison images have different sizes: {sorted(sizes)}")
    return next(iter(sizes))


def _draw_overview(images: list[tuple[str, str, Image.Image, str]], output: Path) -> None:
    panel_width, panel_height = 590, 995
    gap = 45
    left = 55
    canvas_width = left * 2 + len(images) * panel_width + (len(images) - 1) * gap
    canvas = Image.new("RGB", (canvas_width, 1235), "#F7F5F1")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (canvas_width // 2, 26),
        "同一タイル生成から得た4K比較",
        font=load_font(42, bold=True),
        fill="#27212E",
        anchor="ma",
    )
    for index, (title, note, image, color) in enumerate(images):
        x = left + index * (panel_width + gap)
        panel = fit(image, (panel_width, panel_height))
        canvas.paste(panel, (x, 100))
        draw.rectangle(
            (x, 100, x + panel_width, 100 + panel_height),
            outline=color,
            width=3,
        )
        draw.text(
            (x + panel_width // 2, 1130),
            title,
            font=load_font(27, bold=True),
            fill=color,
            anchor="ma",
        )
        draw.text(
            (x + panel_width // 2, 1175),
            note,
            font=load_font(20),
            fill="#60636B",
            anchor="ma",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _draw_crops(images: list[tuple[str, Image.Image, str]], output: Path) -> None:
    panel_width, panel_height = 420, 330
    gap = 28
    label_width = 225
    top = 120
    row_height = 385
    canvas_width = label_width + gap + len(images) * (panel_width + gap)
    canvas = Image.new(
        "RGB",
        (canvas_width, top + len(REGIONS) * row_height + 25),
        "#F7F5F1",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (canvas_width // 2, 20),
        "同一座標の細部比較",
        font=load_font(38, bold=True),
        fill="#27212E",
        anchor="ma",
    )
    for column, (title, _image, color) in enumerate(images):
        x = label_width + gap + column * (panel_width + gap)
        draw.text(
            (x + panel_width // 2, 76),
            title,
            font=load_font(22, bold=True),
            fill=color,
            anchor="ma",
        )
    for row, (name, (box, label)) in enumerate(REGIONS.items()):
        y = top + row * row_height
        draw.multiline_text(
            (16, y + panel_height // 2),
            f"{name}\n{label}",
            font=load_font(20, bold=True),
            fill="#353039",
            anchor="lm",
            spacing=8,
        )
        for column, (_title, image, color) in enumerate(images):
            x = label_width + gap + column * (panel_width + gap)
            panel = fit(image.crop(box), (panel_width, panel_height))
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + panel_width, y + panel_height),
                outline=color,
                width=2,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _draw_exact_crop(
    name: str,
    label: str,
    box: tuple[int, int, int, int],
    images: list[tuple[str, Image.Image, str]],
    output: Path,
) -> None:
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    gap = 24
    header = 116
    canvas = Image.new(
        "RGB",
        (
            len(images) * crop_width + (len(images) + 1) * gap,
            crop_height + header + gap,
        ),
        "#F7F5F1",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 18),
        f"{name}  {label}（原寸）",
        font=load_font(28, bold=True),
        fill="#27212E",
        anchor="la",
    )
    for index, (title, image, color) in enumerate(images):
        x = gap + index * (crop_width + gap)
        canvas.paste(image.crop(box), (x, header))
        draw.rectangle(
            (x, header, x + crop_width, header + crop_height),
            outline=color,
            width=4,
        )
        draw.text(
            (x + crop_width // 2, 79),
            title,
            font=load_font(21, bold=True),
            fill=color,
            anchor="mm",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _regional_metrics(images: dict[str, Image.Image]) -> dict:
    metrics: dict[str, dict] = {}
    for name, (box, label) in REGIONS.items():
        values: dict[str, float | str | list[int]] = {
            "label": label,
            "box": list(box),
        }
        for image_name, image in images.items():
            luma = luminance(np.asarray(image.crop(box)))
            highpass = np.abs(luma - box_lowpass(luma, 2))
            values[f"{image_name}_highpass_mean"] = float(
                np.mean(highpass, dtype=np.float64)
            )
        metrics[name] = values
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--phase-a", required=True, type=Path)
    parser.add_argument("--phase-b", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = load_rgb(args.source)
    phase_a = load_rgb(args.phase_a)
    phase_b = load_rgb(args.phase_b)
    selected = load_rgb(args.selected)
    size = _validate_sizes(
        {"phase_a": phase_a, "phase_b": phase_b, "selected": selected}
    )
    reference = source.resize(size, Image.Resampling.LANCZOS)
    manifest = read_json(args.manifest)
    stage = final_stage(manifest, size)
    if stage.get("processed_tile_count") != stage.get("tile_count"):
        raise ValueError("not every PhaseWeave tile was processed")
    if manifest.get("phaseweave", {}).get("selection_mode") != "ternary_input_fallback":
        raise ValueError("manifest does not record the ternary input fallback")

    previous = load_rgb(args.previous) if args.previous else None
    if previous is not None and previous.size != size:
        raise ValueError("previous result has a different size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = args.output_dir / "phase_a_b_selected_overview.png"
    crops_path = args.output_dir / "phase_a_b_selected_crops.png"
    _draw_overview(
        [
            ("入力補間", "生成なし", reference, "#68717C"),
            ("配置A単独", "Aを全面採用", phase_a, "#9A6947"),
            ("配置B単独", "Bを全面採用", phase_b, "#3D7187"),
            ("改良版", "A・B・入力を局所選択", selected, "#713D83"),
        ],
        overview_path,
    )
    _draw_crops(
        [
            ("入力補間", reference, "#68717C"),
            ("配置A単独", phase_a, "#9A6947"),
            ("配置B単独", phase_b, "#3D7187"),
            ("改良版", selected, "#713D83"),
        ],
        crops_path,
    )
    exact_crop_paths = {}
    exact_images = [
        ("入力補間", reference, "#68717C"),
        ("配置A単独", phase_a, "#9A6947"),
        ("配置B単独", phase_b, "#3D7187"),
        ("改良版", selected, "#713D83"),
    ]
    for name, (box, label) in REGIONS.items():
        exact_path = args.output_dir / f"exact_crop_{name}_input_a_b_selected.png"
        _draw_exact_crop(name, label, box, exact_images, exact_path)
        exact_crop_paths[name] = str(exact_path)

    revision_overview = None
    revision_crops = None
    if previous is not None:
        revision_overview = args.output_dir / "previous_vs_ternary_overview.png"
        revision_crops = args.output_dir / "previous_vs_ternary_crops.png"
        _draw_overview(
            [
                ("入力補間", "生成なし", reference, "#68717C"),
                ("旧版", "A/B二値選択", previous, "#9A6947"),
                ("改良版", "A/B/入力の三値選択", selected, "#713D83"),
            ],
            revision_overview,
        )
        _draw_crops(
            [
                ("入力補間", reference, "#68717C"),
                ("旧版", previous, "#9A6947"),
                ("改良版", selected, "#713D83"),
            ],
            revision_crops,
        )

    metrics = {
        "phase_a": candidate_metrics(reference, phase_a, stage),
        "phase_b": candidate_metrics(reference, phase_b, stage),
        "selected": candidate_metrics(reference, selected, stage),
    }
    if previous is not None:
        metrics["previous"] = candidate_metrics(reference, previous, stage)
    report = {
        "format_version": 1,
        "comparison_scope": (
            "phase A, phase B, and the ternary result share one tile-generation run; "
            "the optional previous result is a separate deterministic run with the same source, prompt, profile, target, and coordinate seeds"
        ),
        "artifacts": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "phase_a": {"path": str(args.phase_a), "sha256": sha256(args.phase_a)},
            "phase_b": {"path": str(args.phase_b), "sha256": sha256(args.phase_b)},
            "selected": {"path": str(args.selected), "sha256": sha256(args.selected)},
            "previous": (
                {"path": str(args.previous), "sha256": sha256(args.previous)}
                if args.previous
                else None
            ),
            "overview": str(overview_path),
            "crops": str(crops_path),
            "revision_overview": str(revision_overview) if revision_overview else None,
            "revision_crops": str(revision_crops) if revision_crops else None,
            "exact_crops": exact_crop_paths,
        },
        "selection": stage.get("consensus_stats") or {},
        "metrics_to_resize_only": metrics,
        "pairwise": {
            "phase_a_vs_phase_b": full_rgb_error(phase_a, phase_b),
            "phase_a_vs_selected": full_rgb_error(phase_a, selected),
            "phase_b_vs_selected": full_rgb_error(phase_b, selected),
            "previous_vs_selected": (
                full_rgb_error(previous, selected) if previous is not None else None
            ),
        },
        "regions": _regional_metrics(
            {
                "resize_only": reference,
                "phase_a": phase_a,
                "phase_b": phase_b,
                "selected": selected,
                **({"previous": previous} if previous is not None else {}),
            }
        ),
    }
    report_path = args.output_dir / "phase_a_b_selected_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "report": str(report_path),
        "selection": {
            key: report["selection"].get(key)
            for key in (
                "phaseweave_phase0_selected_percent",
                "phaseweave_phase1_selected_percent",
                "phaseweave_input_rejected_percent",
                "phaseweave_uncertain_fused_percent",
                "phaseweave_boundary_percent",
                "phaseweave_mean_support_weight",
            )
        },
        "highpass_ratio": {
            name: values["highpass_mean_ratio"] for name, values in metrics.items()
        },
        "low_frequency_luma_drift_mean": {
            name: values["low_frequency_luma_drift_mean"]
            for name, values in metrics.items()
        },
        "boundary_jump_ratio": {
            name: values["residual_boundary_to_global_p95_ratio"]
            for name, values in metrics.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
