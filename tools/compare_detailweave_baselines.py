"""Compare Lanczos, MultiDiffusion, and DetailWeave at one delivery size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_krea2_4k import (
    candidate_metrics,
    fit,
    full_rgb_error,
    load_font,
    load_rgb,
    read_json,
    sha256,
)
from tools.evaluate_krea2_phaseweave_result import REGIONS


def draw_overview(
    images: list[tuple[str, str, Image.Image, str]], output: Path
) -> None:
    panel_width, panel_height = 650, 1010
    gap = 48
    margin = 55
    width = margin * 2 + len(images) * panel_width + (len(images) - 1) * gap
    canvas = Image.new("RGB", (width, 1240), "#F7F5F1")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, 24),
        "Lanczos・MultiDiffusion・DetailWeave 4K",
        font=load_font(40, bold=True),
        fill="#241F2B",
        anchor="ma",
    )
    for index, (title, note, image, color) in enumerate(images):
        x = margin + index * (panel_width + gap)
        panel = fit(image, (panel_width, panel_height))
        canvas.paste(panel, (x, 96))
        draw.rectangle(
            (x, 96, x + panel_width, 96 + panel_height),
            outline=color,
            width=4,
        )
        draw.text(
            (x + panel_width // 2, 1140),
            title,
            font=load_font(29, bold=True),
            fill=color,
            anchor="ma",
        )
        draw.text(
            (x + panel_width // 2, 1190),
            note,
            font=load_font(20),
            fill="#5A5D64",
            anchor="ma",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def draw_crops(
    images: list[tuple[str, Image.Image, str]], output: Path
) -> dict[str, str]:
    panel_width, panel_height = 500, 340
    gap = 30
    label_width = 225
    top = 120
    row_height = 400
    width = label_width + gap + len(images) * (panel_width + gap)
    canvas = Image.new(
        "RGB", (width, top + len(REGIONS) * row_height + 25), "#F7F5F1"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, 20),
        "同一4K座標の局所比較",
        font=load_font(38, bold=True),
        fill="#241F2B",
        anchor="ma",
    )
    for column, (title, _image, color) in enumerate(images):
        x = label_width + gap + column * (panel_width + gap)
        draw.text(
            (x + panel_width // 2, 76),
            title,
            font=load_font(23, bold=True),
            fill=color,
            anchor="ma",
        )
    exact_paths: dict[str, str] = {}
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
                width=3,
            )

        crop_width = box[2] - box[0]
        crop_height = box[3] - box[1]
        exact_gap = 24
        header = 110
        exact = Image.new(
            "RGB",
            (
                len(images) * crop_width + (len(images) + 1) * exact_gap,
                crop_height + header + exact_gap,
            ),
            "#F7F5F1",
        )
        exact_draw = ImageDraw.Draw(exact)
        exact_draw.text(
            (exact_gap, 14),
            f"{name}  {label}（原寸）",
            font=load_font(27, bold=True),
            fill="#241F2B",
            anchor="la",
        )
        for column, (title, image, color) in enumerate(images):
            x = exact_gap + column * (crop_width + exact_gap)
            exact.paste(image.crop(box), (x, header))
            exact_draw.rectangle(
                (x, header, x + crop_width, header + crop_height),
                outline=color,
                width=4,
            )
            exact_draw.text(
                (x + crop_width // 2, 73),
                title,
                font=load_font(21, bold=True),
                fill=color,
                anchor="mm",
            )
        exact_path = output.parent / f"exact_crop_{name}_three_way.png"
        exact.save(exact_path, format="PNG", optimize=True)
        exact_paths[name] = str(exact_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return exact_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--multidiffusion", type=Path, required=True)
    parser.add_argument("--detailweave", type=Path, required=True)
    parser.add_argument("--detailweave-manifest", type=Path, required=True)
    parser.add_argument("--multidiffusion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    detailweave = load_rgb(args.detailweave)
    multidiffusion = load_rgb(args.multidiffusion)
    source = load_rgb(args.source)
    if multidiffusion.size != detailweave.size:
        raise ValueError(
            f"MultiDiffusion {multidiffusion.size} differs from DetailWeave {detailweave.size}"
        )
    lanczos = source.resize(detailweave.size, Image.Resampling.LANCZOS)
    detail_manifest = read_json(args.detailweave_manifest)
    multi_manifest = read_json(args.multidiffusion_manifest)
    stage = (detail_manifest.get("stage_reports") or [])[-1]
    if tuple(stage.get("size") or ()) != detailweave.size:
        raise ValueError("DetailWeave manifest does not describe the compared image")
    if multi_manifest.get("target_size") != list(detailweave.size):
        raise ValueError("MultiDiffusion manifest does not describe the compared image")
    if (multi_manifest.get("tile") or {}).get("method") != "MultiDiffusion":
        raise ValueError("comparison manifest is not a MultiDiffusion run")

    images = [
        ("Lanczos補間", "生成なし", lanczos, "#606B75"),
        ("MultiDiffusion", "単一分割拡大", multidiffusion, "#2E718D"),
        ("DetailWeave 4K", "二候補＋入力維持", detailweave, "#7A3D87"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = args.output_dir / "three_way_overview.png"
    crops = args.output_dir / "three_way_crops.png"
    draw_overview(images, overview)
    exact = draw_crops(
        [(title, image, color) for title, _note, image, color in images], crops
    )

    metrics = {
        "multidiffusion": candidate_metrics(lanczos, multidiffusion, stage),
        "detailweave": candidate_metrics(lanczos, detailweave, stage),
    }
    report = {
        "format_version": 1,
        "comparison": ["lanczos", "multidiffusion", "detailweave"],
        "target_size": list(detailweave.size),
        "artifacts": {
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "multidiffusion": str(args.multidiffusion),
            "multidiffusion_sha256": sha256(args.multidiffusion),
            "detailweave": str(args.detailweave),
            "detailweave_sha256": sha256(args.detailweave),
            "overview": str(overview),
            "crops": str(crops),
            "exact_crops": exact,
        },
        "metrics_to_lanczos": metrics,
        "pairwise": {
            "lanczos_vs_multidiffusion": full_rgb_error(lanczos, multidiffusion),
            "lanczos_vs_detailweave": full_rgb_error(lanczos, detailweave),
            "multidiffusion_vs_detailweave": full_rgb_error(
                multidiffusion, detailweave
            ),
        },
        "notes": {
            "boundary_metrics": "Both candidates are sampled at DetailWeave planned core boundaries; use only as a common diagnostic, not as each method's native seam measure.",
            "lanczos": "Resize-only reference, not ground truth.",
        },
    }
    report_path = args.output_dir / "three_way_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "overview": str(overview),
                "crops": str(crops),
                "highpass_ratio": {
                    name: value["highpass_mean_ratio"]
                    for name, value in metrics.items()
                },
                "low_frequency_luma_drift_mean": {
                    name: value["low_frequency_luma_drift_mean"]
                    for name, value in metrics.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
