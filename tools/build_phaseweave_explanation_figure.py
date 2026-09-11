"""Draw the two complete shifted layouts and their measured selection map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_krea2_4k import fit, load_font, load_rgb


PANEL_SIZE = (690, 975)
PANEL_TOP = 130
PANEL_LEFTS = (70, 855, 1640)


def _panel_image(image: Image.Image) -> Image.Image:
    return fit(image, PANEL_SIZE)


def _draw_layout_panel(
    canvas: Image.Image,
    source: Image.Image,
    tiles: list[dict],
    phase: int,
    x: int,
) -> None:
    panel = _panel_image(source)
    dim = Image.new("RGBA", PANEL_SIZE, (18, 20, 24, 80))
    panel = Image.alpha_composite(panel.convert("RGBA"), dim)
    overlay = Image.new("RGBA", PANEL_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = min(PANEL_SIZE[0] / source.width, PANEL_SIZE[1] / source.height)
    offset_x = (PANEL_SIZE[0] - source.width * scale) / 2.0
    offset_y = (PANEL_SIZE[1] - source.height * scale) / 2.0
    phase_tiles = [tile for tile in tiles if int(tile["phase"]) == phase]
    palette = (
        (234, 153, 80, 55),
        (248, 211, 121, 45),
    ) if phase == 0 else (
        (70, 163, 194, 55),
        (114, 204, 219, 45),
    )
    outline = (255, 218, 154, 235) if phase == 0 else (168, 231, 244, 235)
    for index, tile in enumerate(phase_tiles):
        x0 = offset_x + float(tile["grid_core_x0"]) * scale
        y0 = offset_y + float(tile["grid_core_y0"]) * scale
        x1 = offset_x + float(tile["grid_core_x1"]) * scale
        y1 = offset_y + float(tile["grid_core_y1"]) * scale
        draw.rectangle((x0, y0, x1, y1), fill=palette[index % 2], outline=outline, width=3)
        clipped_x0 = max(0, x0)
        clipped_y0 = max(0, y0)
        clipped_x1 = min(PANEL_SIZE[0], x1)
        clipped_y1 = min(PANEL_SIZE[1], y1)
        if clipped_x1 - clipped_x0 > 36 and clipped_y1 - clipped_y0 > 36:
            draw.text(
                ((clipped_x0 + clipped_x1) / 2, (clipped_y0 + clipped_y1) / 2),
                str(index + 1),
                font=load_font(17, bold=True),
                fill=(255, 255, 255, 225),
                stroke_width=2,
                stroke_fill=(20, 25, 30, 180),
                anchor="mm",
            )
    panel = Image.alpha_composite(panel, overlay).convert("RGB")
    canvas.paste(panel, (x, PANEL_TOP))


def _draw_selection_panel(
    canvas: Image.Image,
    result: Image.Image,
    selection: Image.Image,
    x: int,
) -> None:
    panel = _panel_image(result).convert("RGBA")
    selection_panel = _panel_image(selection).convert("RGBA")
    selection_panel.putalpha(112)
    panel = Image.alpha_composite(panel, selection_panel).convert("RGB")
    canvas.paste(panel, (x, PANEL_TOP))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--selection-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    stage = manifest["stage_reports"][-1]
    tiles = stage["tiles"]
    source = load_rgb(Path(manifest["input"])).resize(
        tuple(map(int, manifest["target_size"])),
        Image.Resampling.LANCZOS,
    )
    result = load_rgb(args.image)
    selection = load_rgb(args.selection_map)
    if source.size != result.size or result.size != selection.size:
        raise ValueError("source, result, and selection map sizes differ")

    canvas = Image.new("RGB", (2400, 1310), "#F7F5F1")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1200, 30),
        "2つの配置は、どちらも画像全体を処理する",
        font=load_font(43, bold=True),
        fill="#27212E",
        anchor="ma",
    )
    draw.text(
        (1200, 84),
        "二分割ではない。格子を半分ずらして、別の4K候補をもう一度つくる。",
        font=load_font(23),
        fill="#555A62",
        anchor="ma",
    )

    _draw_layout_panel(canvas, source, tiles, 0, PANEL_LEFTS[0])
    _draw_layout_panel(canvas, source, tiles, 1, PANEL_LEFTS[1])
    _draw_selection_panel(canvas, result, selection, PANEL_LEFTS[2])

    headings = (
        ("配置A：画像全体", "20領域（4列×5行）", "#A9572E"),
        ("配置B：画像全体", "半格子ずらした24領域（4列×6行）", "#2F718C"),
        ("最終選択", "橙=A、青=B、灰=入力、紫=弱い融合", "#713D83"),
    )
    for x, (title, note, color) in zip(PANEL_LEFTS, headings):
        draw.rectangle(
            (x, PANEL_TOP, x + PANEL_SIZE[0], PANEL_TOP + PANEL_SIZE[1]),
            outline=color,
            width=4,
        )
        draw.text(
            (x + PANEL_SIZE[0] // 2, 1140),
            title,
            font=load_font(27, bold=True),
            fill=color,
            anchor="ma",
        )
        draw.text(
            (x + PANEL_SIZE[0] // 2, 1182),
            note,
            font=load_font(19),
            fill="#555A62",
            anchor="ma",
        )
    draw.rounded_rectangle(
        (255, 1230, 2145, 1290),
        radius=18,
        fill="#ECE8EF",
        outline="#C7BFCC",
        width=2,
    )
    draw.text(
        (1200, 1260),
        "各番号の採用領域は960 px。生成時は周囲160 pxも含む1280 pxを見せ、端の影響を採用領域へ入れない。",
        font=load_font(20),
        fill="#403946",
        anchor="mm",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
