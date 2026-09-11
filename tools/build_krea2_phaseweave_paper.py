"""Build a two-page Japanese B5 academic paper for DetailWeave 4K."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import re

import pdfplumber
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import portrait
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    Image as PlatypusImage,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "output/detailweave_paper_image_cropped_20260722"
PAPER_INPUT_ROOT = RUN_ROOT / "paper_inputs"
DEFAULT_SOURCE = PAPER_INPUT_ROOT / "source.png"
DEFAULT_RESULT = PAPER_INPUT_ROOT / "detailweave_4k.png"
DEFAULT_MANIFEST = PAPER_INPUT_ROOT / "detailweave_manifest.json"
DEFAULT_QA = PAPER_INPUT_ROOT / "detailweave_qa.json"
DEFAULT_SELECTION_MAP = PAPER_INPUT_ROOT / "selection_map.png"
DEFAULT_SINGLE_GRID = PAPER_INPUT_ROOT / "phase_a.png"
DEFAULT_PHASE_B = PAPER_INPUT_ROOT / "phase_b.png"
DEFAULT_COMPARISON = (
    RUN_ROOT / "single_grid_comparison/phase_a_b_selected_metrics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/pdf/detailweave_4k_b5_ja.pdf"
DEFAULT_ASSET_DIR = (
    RUN_ROOT / "paper_assets"
)

JIS_B5 = portrait((182 * mm, 257 * mm))
PAGE_WIDTH, PAGE_HEIGHT = JIS_B5
MARGIN_X = 10.5 * mm
BOTTOM = 11.5 * mm
TOP_RULE_Y = PAGE_HEIGHT - 10.5 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
COLUMN_GAP = 5.2 * mm
COLUMN_WIDTH = (CONTENT_WIDTH - COLUMN_GAP) / 2

INK = colors.HexColor("#111111")
MID = colors.HexColor("#454545")
LIGHT = colors.HexColor("#B8B8B8")
VERY_LIGHT = colors.HexColor("#E8E8E8")
WHITE = colors.white

JP_REGULAR = "PhaseWeaveAcademicMincho-Regular"
JP_BOLD = "PhaseWeaveAcademicMincho-Bold"

REGIONS = {
    "A": ((420, 820, 1370, 1770), "青い髪・目・角"),
    "B": ((1570, 850, 2670, 1900), "白い髪・目・衣装"),
    "C": ((1090, 1510, 2390, 2670), "本・指・縫い目"),
    "D": ((850, 2690, 2240, 4050), "透明体・反射・敷物"),
}

DETAIL_EXAMPLES = (
    {
        "label": "例1　青い髪",
        "note": "主にBを採用",
        "box": (600, 1500, 720, 1590),
        "focus": (0.73, 0.43),
        "problem_column": 0,
        "choice": "B",
        "color": "#D5533A",
    },
    {
        "label": "例2　本の頁",
        "note": "主にAを採用",
        "box": (1570, 1910, 1690, 2000),
        "focus": (0.28, 0.37),
        "problem_column": 1,
        "choice": "A",
        "color": "#2F7C9D",
    },
)


def find_font(explicit: str | None, *, bold: bool) -> Path:
    candidates = [
        explicit,
        os.environ.get(
            "PHASEWEAVE_PAPER_JP_BOLD_FONT"
            if bold
            else "PHASEWEAVE_PAPER_JP_FONT"
        ),
        "C:/Windows/Fonts/yumindb.ttf" if bold else "C:/Windows/Fonts/yumin.ttf",
        "C:/Windows/Fonts/YuMincho-Demibold.ttc"
        if bold
        else "C:/Windows/Fonts/YuMincho.ttc",
        "C:/Windows/Fonts/msmincho.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
        if bold
        else "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Japanese Mincho font not found")


def register_fonts(regular: Path, bold: Path) -> None:
    regular_options = {"subfontIndex": 0} if regular.suffix.lower() == ".ttc" else {}
    bold_options = {"subfontIndex": 0} if bold.suffix.lower() == ".ttc" else {}
    pdfmetrics.registerFont(TTFont(JP_REGULAR, str(regular), **regular_options))
    pdfmetrics.registerFont(TTFont(JP_BOLD, str(bold), **bold_options))
    pdfmetrics.registerFontFamily(
        JP_REGULAR,
        normal=JP_REGULAR,
        bold=JP_BOLD,
        italic=JP_REGULAR,
        boldItalic=JP_BOLD,
    )


def paragraph_style(
    name: str,
    *,
    size: float,
    leading: float,
    bold: bool = False,
    alignment=TA_JUSTIFY,
    first_indent: float = 0,
    left_indent: float = 0,
    right_indent: float = 0,
    space_before: float = 0,
    space_after: float = 0,
    color=INK,
    keep_with_next: bool = False,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName=JP_BOLD if bold else JP_REGULAR,
        fontSize=size,
        leading=leading,
        alignment=alignment,
        firstLineIndent=first_indent,
        leftIndent=left_indent,
        rightIndent=right_indent,
        spaceBefore=space_before,
        spaceAfter=space_after,
        textColor=color,
        wordWrap="CJK",
        allowWidows=0,
        allowOrphans=0,
        keepWithNext=keep_with_next,
    )


STYLES = {
    "title": paragraph_style(
        "academic-title",
        size=13.4,
        leading=16.2,
        bold=True,
        alignment=TA_CENTER,
        space_after=2.0,
    ),
    "english_title": paragraph_style(
        "academic-english-title",
        size=6.2,
        leading=7.4,
        alignment=TA_CENTER,
        color=MID,
        space_after=2.4,
    ),
    "author": paragraph_style(
        "academic-author",
        size=7.7,
        leading=9.0,
        bold=True,
        alignment=TA_CENTER,
        space_after=3.2,
    ),
    "abstract": paragraph_style(
        "academic-abstract",
        size=6.25,
        leading=8.15,
        alignment=TA_JUSTIFY,
    ),
    "keywords": paragraph_style(
        "academic-keywords",
        size=5.9,
        leading=7.2,
        alignment=TA_LEFT,
        space_before=2.0,
    ),
    "section": paragraph_style(
        "academic-section",
        size=9.2,
        leading=11.0,
        bold=True,
        alignment=TA_LEFT,
        space_before=4.0,
        space_after=1.6,
        keep_with_next=True,
    ),
    "subsection": paragraph_style(
        "academic-subsection",
        size=7.45,
        leading=9.1,
        bold=True,
        alignment=TA_LEFT,
        space_before=2.5,
        space_after=0.7,
        keep_with_next=True,
    ),
    "body": paragraph_style(
        "academic-body",
        size=6.75,
        leading=8.85,
        alignment=TA_JUSTIFY,
        first_indent=6.75,
        space_after=1.4,
    ),
    "body_no_indent": paragraph_style(
        "academic-body-no-indent",
        size=6.75,
        leading=8.85,
        alignment=TA_JUSTIFY,
        space_after=1.4,
    ),
    "caption": paragraph_style(
        "academic-caption",
        size=5.65,
        leading=7.0,
        alignment=TA_CENTER,
        space_before=1.2,
        space_after=2.0,
    ),
    "table_caption": paragraph_style(
        "academic-table-caption",
        size=5.75,
        leading=7.1,
        alignment=TA_LEFT,
        space_before=2.2,
        space_after=1.5,
        keep_with_next=True,
    ),
    "table": paragraph_style(
        "academic-table",
        size=5.55,
        leading=6.7,
        alignment=TA_LEFT,
    ),
    "table_center": paragraph_style(
        "academic-table-center",
        size=5.55,
        leading=6.7,
        alignment=TA_CENTER,
    ),
    "equation": paragraph_style(
        "academic-equation",
        size=6.2,
        leading=7.6,
        alignment=TA_CENTER,
    ),
    "reference": paragraph_style(
        "academic-reference",
        size=5.45,
        leading=6.65,
        alignment=TA_LEFT,
        first_indent=-9.0,
        left_indent=9.0,
        space_after=0.7,
    ),
    "reference_compact": paragraph_style(
        "academic-reference-compact",
        size=4.65,
        leading=5.55,
        alignment=TA_LEFT,
        space_after=0.5,
    ),
}


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def section(number: str, title: str) -> Paragraph:
    return para(f"{number}. {title}", "section")


def subsection(number: str, title: str) -> Paragraph:
    return para(f"{number} {title}", "subsection")


def equation(text: str, number: int) -> Table:
    row = [para(escape(text), "equation"), para(f"({number})", "table_center")]
    table = Table([row], colWidths=[COLUMN_WIDTH - 9 * mm, 7 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    return table


def academic_table(
    rows: list[list[str]],
    widths: list[float],
    *,
    centered_columns: tuple[int, ...] = (),
) -> Table:
    rendered: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        rendered.append(
            [
                para(
                    value,
                    "table_center"
                    if row_index == 0 or column_index in centered_columns
                    else "table",
                )
                for column_index, value in enumerate(row)
            ]
        )
    table = Table(rendered, colWidths=widths, hAlign="LEFT", repeatRows=1)
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), JP_BOLD),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.25),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.25),
    ]
    table.setStyle(TableStyle(commands))
    return table


def compact_reference_table(references: list[str]) -> Table:
    midpoint = (len(references) + 1) // 2
    left = references[:midpoint]
    right = references[midpoint:]
    row_count = max(len(left), len(right))
    rows: list[list[Paragraph]] = []
    for row in range(row_count):
        rows.append(
            [
                para(left[row], "reference_compact")
                if row < len(left)
                else para("", "reference_compact"),
                para(right[row], "reference_compact")
                if row < len(right)
                else para("", "reference_compact"),
            ]
        )
    table = Table(
        rows,
        colWidths=[COLUMN_WIDTH * 0.5 - 1.2 * mm, COLUMN_WIDTH * 0.5 - 1.2 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 2.4 * mm),
                ("LEFTPADDING", (1, 0), (1, -1), 0),
                ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8),
            ]
        )
    )
    return table


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def pil_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/yumindb.ttf" if bold else "C:/Windows/Fonts/yumin.ttf"),
        Path("C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothR.ttc"),
        Path("C:/Windows/Fonts/meiryob.ttc" if bold else "C:/Windows/Fonts/meiryo.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size, index=0)
    raise FileNotFoundError("Japanese font for figure not found")


def dashed_line(
    draw: ImageDraw.ImageDraw,
    points: tuple[tuple[int, int], tuple[int, int]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: int = 12,
    gap: int = 8,
) -> None:
    (x0, y0), (x1, y1) = points
    if x0 == x1:
        direction = 1 if y1 >= y0 else -1
        for start in range(y0, y1, direction * (dash + gap)):
            end = start + direction * dash
            if direction > 0:
                end = min(end, y1)
            else:
                end = max(end, y1)
            draw.line((x0, start, x1, end), fill=fill, width=width)
    else:
        direction = 1 if x1 >= x0 else -1
        for start in range(x0, x1, direction * (dash + gap)):
            end = start + direction * dash
            if direction > 0:
                end = min(end, x1)
            else:
                end = max(end, x1)
            draw.line((start, y0, end, y1), fill=fill, width=width)


def make_grid_figure(
    manifest: dict,
    source: Image.Image,
    selection: Image.Image,
    output: Path,
) -> None:
    stage = manifest["stage_reports"][-1]
    tiles = stage["tiles"]
    target = tuple(map(int, manifest["target_size"]))
    source = source.resize(target, Image.Resampling.LANCZOS)
    if selection.size != target:
        raise ValueError("selection map size does not match the paper result")

    canvas = Image.new("RGB", (1200, 680), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(32, bold=True)
    label_font = pil_font(25, bold=True)
    body_font = pil_font(20)
    small_font = pil_font(18)
    draw.text(
        (600, 20),
        "AとBは、どちらも画像全体を処理する",
        font=title_font,
        fill="#111111",
        anchor="ma",
    )
    draw.text(
        (600, 57),
        "左右分割ではなく、格子を半分ずらした二つの完成候補を作る",
        font=body_font,
        fill="#555555",
        anchor="ma",
    )

    panel_size = (330, 467)
    panel_y = 100
    panel_x = (38, 435, 832)
    colors_by_phase = ("#C66834", "#317B9D")

    def grid_panel(phase: int, x: int) -> None:
        base = fit_image(source, panel_size).convert("RGBA")
        base = Image.alpha_composite(
            base,
            Image.new("RGBA", panel_size, (12, 16, 20, 68)),
        )
        overlay = Image.new("RGBA", panel_size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        scale = min(panel_size[0] / target[0], panel_size[1] / target[1])
        offset_x = (panel_size[0] - target[0] * scale) / 2
        offset_y = (panel_size[1] - target[1] * scale) / 2
        rgba = (237, 164, 89, 48) if phase == 0 else (76, 173, 204, 48)
        outline = (255, 222, 166, 235) if phase == 0 else (171, 235, 247, 235)
        for tile in tiles:
            if int(tile["phase"]) != phase:
                continue
            box = (
                offset_x + float(tile["grid_core_x0"]) * scale,
                offset_y + float(tile["grid_core_y0"]) * scale,
                offset_x + float(tile["grid_core_x1"]) * scale,
                offset_y + float(tile["grid_core_y1"]) * scale,
            )
            overlay_draw.rectangle(box, fill=rgba, outline=outline, width=3)
        panel = Image.alpha_composite(base, overlay).convert("RGB")
        canvas.paste(panel, (x, panel_y))
        draw.rectangle(
            (x, panel_y, x + panel_size[0], panel_y + panel_size[1]),
            outline=colors_by_phase[phase],
            width=4,
        )

    grid_panel(0, panel_x[0])
    grid_panel(1, panel_x[1])
    selection_panel = fit_image(selection, panel_size)
    canvas.paste(selection_panel, (panel_x[2], panel_y))
    draw.rectangle(
        (
            panel_x[2],
            panel_y,
            panel_x[2] + panel_size[0],
            panel_y + panel_size[1],
        ),
        outline="#713D83",
        width=4,
    )

    labels = (
        ("配置A：全体", "20領域（4列×5行）", "#A9572E"),
        ("配置B：全体", "半格子ずらした24領域", "#2F718C"),
        ("最終選択", "橙=A、青=B、灰=入力、紫=弱い融合", "#713D83"),
    )
    for x, (title, note, color) in zip(panel_x, labels):
        draw.text(
            (x + panel_size[0] // 2, 592),
            title,
            font=label_font,
            fill=color,
            anchor="ma",
        )
        draw.text(
            (x + panel_size[0] // 2, 624),
            note,
            font=small_font,
            fill="#555555",
            anchor="ma",
        )
    draw.text(
        (600, 659),
        "各領域は中央960 pxを採用し、生成時は周囲160 pxを含む1280 pxを見る。隣とは80 px重なる。",
        font=small_font,
        fill="#333333",
        anchor="mm",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def crop_cover(image: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    crop = image.crop(box)
    scale = max(size[0] / crop.width, size[1] / crop.height)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def fit_image(image: Image.Image, size: tuple[int, int], background="white") -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    result = Image.new("RGB", size, background)
    result.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return result


def make_comparison_figure(
    source: Image.Image,
    single_grid: Image.Image,
    phase_b: Image.Image,
    result: Image.Image,
    selection_map: Image.Image,
    output: Path,
) -> None:
    reference = source.resize(result.size, Image.Resampling.LANCZOS)
    if any(image.size != result.size for image in (single_grid, phase_b, selection_map)):
        raise ValueError("paper comparison image sizes differ from DetailWeave")

    canvas = Image.new("RGB", (2770, 1220), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(33, bold=True)
    label_font = pil_font(24, bold=True)
    small_font = pil_font(20)
    tiny_font = pil_font(17)
    draw.text(
        (1385, 20),
        "全体比較と、候補選択が見える400%拡大クロップ",
        font=title_font,
        fill="#111111",
        anchor="ma",
    )

    overview_panels = (
        (reference, "Lanczos", "#68717C"),
        (single_grid, "候補A＝単一格子", "#C26836"),
        (phase_b, "候補B", "#317B9D"),
        (result, "DetailWeave", "#713D83"),
    )
    thumb_size = (145, 205)
    thumb_y = 82
    for index, (image, label, color) in enumerate(overview_panels):
        x = 35 + index * 178
        panel = fit_image(image, thumb_size)
        canvas.paste(panel, (x, thumb_y))
        draw.rectangle(
            (x, thumb_y, x + thumb_size[0], thumb_y + thumb_size[1]),
            outline=color,
            width=3,
        )
        draw.text(
            (x + thumb_size[0] // 2, 65),
            label,
            font=tiny_font,
            fill=color,
            anchor="mm",
        )
        if index > 0:
            fitted_scale = min(
                thumb_size[0] / image.width,
                thumb_size[1] / image.height,
            )
            fitted_width = round(image.width * fitted_scale)
            fitted_height = round(image.height * fitted_scale)
            fitted_x = x + (thumb_size[0] - fitted_width) // 2
            fitted_y = thumb_y + (thumb_size[1] - fitted_height) // 2
            for example in DETAIL_EXAMPLES:
                x0, y0, x1, y1 = example["box"]
                draw.rectangle(
                    (
                        fitted_x + round(x0 * fitted_scale),
                        fitted_y + round(y0 * fitted_scale),
                        fitted_x + round(x1 * fitted_scale),
                        fitted_y + round(y1 * fitted_scale),
                    ),
                    outline=example["color"],
                    width=2,
                )

    def rounded_label(
        xy: tuple[int, int, int, int],
        text: str,
        *,
        outline: str,
        fill: str = "white",
    ) -> None:
        draw.rounded_rectangle(xy, radius=13, fill=fill, outline=outline, width=3)
        draw.text(
            ((xy[0] + xy[2]) // 2, (xy[1] + xy[3]) // 2),
            text,
            font=small_font,
            fill=outline,
            anchor="mm",
        )

    rounded_label((820, 94, 1150, 150), "候補A（単一格子）", outline="#C26836")
    rounded_label((820, 196, 1150, 252), "候補B（ずらした格子）", outline="#317B9D")
    rounded_label((1300, 145, 1570, 205), "局所的に選択", outline="#713D83")
    rounded_label((1740, 145, 2035, 205), "DetailWeave", outline="#713D83")
    draw.line((1150, 122, 1290, 163), fill="#713D83", width=5)
    draw.line((1150, 224, 1290, 187), fill="#713D83", width=5)
    draw.line((1570, 175, 1730, 175), fill="#713D83", width=5)
    draw.polygon(((1730, 175), (1708, 163), (1708, 187)), fill="#713D83")
    draw.text(
        (2210, 90),
        "選択マスク",
        font=label_font,
        fill="#111111",
        anchor="ma",
    )
    legend_items = (
        ("#C26836", "A"),
        ("#317B9D", "B"),
        ("#606369", "入力維持"),
        ("#7E4891", "弱い融合"),
    )
    for index, (color, label) in enumerate(legend_items):
        x = 2100 + (index % 2) * 300
        y = 137 + (index // 2) * 70
        draw.rounded_rectangle((x, y, x + 42, y + 34), radius=6, fill=color)
        draw.text((x + 54, y + 17), label, font=small_font, fill="#333333", anchor="lm")

    column_labels = (
        ("単一格子 A", "#C26836"),
        ("候補B", "#317B9D"),
        ("DetailWeave", "#713D83"),
        ("| A − DetailWeave |", "#B24435"),
        ("選択マスク重ね表示", "#4A4A4A"),
    )
    panel_width = 480
    panel_height = 360
    panel_x0 = 190
    panel_gap = 25
    label_y = 340
    for column, (label, color) in enumerate(column_labels):
        x = panel_x0 + column * (panel_width + panel_gap)
        draw.text(
            (x + panel_width // 2, label_y),
            label,
            font=small_font,
            fill=color,
            anchor="mm",
        )

    def draw_focus_arrow(
        panel_x: int,
        panel_y: int,
        relative: tuple[float, float],
        color: str,
    ) -> None:
        focus_x = panel_x + round(relative[0] * panel_width)
        focus_y = panel_y + round(relative[1] * panel_height)
        radius = 35
        draw.ellipse(
            (focus_x - radius, focus_y - radius, focus_x + radius, focus_y + radius),
            outline=color,
            width=6,
        )
        start = (panel_x + 35, panel_y + 38)
        end = (focus_x - radius + 3, focus_y - radius + 3)
        draw.line((*start, *end), fill=color, width=6)
        draw.polygon(
            (
                end,
                (end[0] - 8, end[1] - 23),
                (end[0] - 23, end[1] - 8),
            ),
            fill=color,
        )

    row_ys = (385, 790)
    for row, example in enumerate(DETAIL_EXAMPLES):
        box = example["box"]
        row_y = row_ys[row]
        draw.multiline_text(
            (18, row_y + 38),
            f"{example['label']}\n{example['note']}",
            font=tiny_font,
            fill="#222222",
            spacing=8,
        )
        crops = (
            single_grid.crop(box),
            phase_b.crop(box),
            result.crop(box),
        )
        rendered = [
            crop.resize((panel_width, panel_height), Image.Resampling.NEAREST)
            for crop in crops
        ]
        difference = ImageChops.difference(crops[0], crops[2])
        red, green, blue = difference.split()
        maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
        amplified = maximum.point(tuple(min(255, value * 6) for value in range(256)))
        heat = ImageOps.colorize(
            amplified,
            black="#130C2B",
            mid="#DC493B",
            white="#FFF174",
        ).resize((panel_width, panel_height), Image.Resampling.NEAREST)
        mask_overlay = Image.blend(
            rendered[2],
            selection_map.crop(box).resize(
                (panel_width, panel_height),
                Image.Resampling.NEAREST,
            ),
            0.43,
        )
        row_panels = (*rendered, heat, mask_overlay)
        border_colors = (
            "#C26836",
            "#317B9D",
            "#713D83",
            "#B24435",
            "#555555",
        )
        for column, (panel, border_color) in enumerate(
            zip(row_panels, border_colors)
        ):
            x = panel_x0 + column * (panel_width + panel_gap)
            canvas.paste(panel, (x, row_y))
            draw.rectangle(
                (x, row_y, x + panel_width, row_y + panel_height),
                outline=border_color,
                width=4,
            )
        problem_x = panel_x0 + example["problem_column"] * (
            panel_width + panel_gap
        )
        final_x = panel_x0 + 2 * (panel_width + panel_gap)
        draw_focus_arrow(
            problem_x,
            row_y,
            example["focus"],
            example["color"],
        )
        draw_focus_arrow(
            final_x,
            row_y,
            example["focus"],
            example["color"],
        )
        draw.rounded_rectangle(
            (final_x + 12, row_y + panel_height - 48, final_x + 120, row_y + panel_height - 12),
            radius=8,
            fill="white",
            outline=example["color"],
            width=3,
        )
        draw.text(
            (final_x + 66, row_y + panel_height - 30),
            f"{example['choice']}採用",
            font=tiny_font,
            fill=example["color"],
            anchor="mm",
        )

    draw.text(
        (panel_x0 + 3 * (panel_width + panel_gap) + panel_width // 2, 1176),
        "暗色＝同じ、黄＝差が大きい（両例で同じ強調倍率）",
        font=tiny_font,
        fill="#444444",
        anchor="mm",
    )
    draw.text(
        (panel_x0 + 4 * (panel_width + panel_gap) + panel_width // 2, 1176),
        "橙=A、青=B、灰=入力、紫=弱い融合",
        font=tiny_font,
        fill="#444444",
        anchor="mm",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


class AcademicDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, pagesize=JIS_B5, **kwargs)
        self.title_frame = Frame(
            MARGIN_X,
            PAGE_HEIGHT - 64 * mm,
            CONTENT_WIDTH,
            49 * mm,
            id="title",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        page1_body_top = PAGE_HEIGHT - 68 * mm
        page1_height = page1_body_top - BOTTOM
        self.page1_left = Frame(
            MARGIN_X,
            BOTTOM,
            COLUMN_WIDTH,
            page1_height,
            id="page1-left",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        self.page1_right = Frame(
            MARGIN_X + COLUMN_WIDTH + COLUMN_GAP,
            BOTTOM,
            COLUMN_WIDTH,
            page1_height,
            id="page1-right",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )

        self.page2_figure = Frame(
            MARGIN_X,
            PAGE_HEIGHT - 97 * mm,
            CONTENT_WIDTH,
            82 * mm,
            id="page2-figure",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        page2_body_top = PAGE_HEIGHT - 101 * mm
        page2_height = page2_body_top - BOTTOM
        self.page2_left = Frame(
            MARGIN_X,
            BOTTOM,
            COLUMN_WIDTH,
            page2_height,
            id="page2-left",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        self.page2_right = Frame(
            MARGIN_X + COLUMN_WIDTH + COLUMN_GAP,
            BOTTOM,
            COLUMN_WIDTH,
            page2_height,
            id="page2-right",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(
                    id="page1",
                    frames=[self.title_frame, self.page1_left, self.page1_right],
                    onPage=draw_header_footer,
                ),
                PageTemplate(
                    id="page2",
                    frames=[self.page2_figure, self.page2_left, self.page2_right],
                    onPage=draw_header_footer,
                ),
            ]
        )


def draw_header_footer(c, document) -> None:
    c.saveState()
    c.setStrokeColor(INK)
    c.setLineWidth(0.55)
    c.line(MARGIN_X, TOP_RULE_Y, PAGE_WIDTH - MARGIN_X, TOP_RULE_Y)
    c.setFont(JP_REGULAR, 6.2)
    c.setFillColor(INK)
    c.drawCentredString(PAGE_WIDTH / 2, TOP_RULE_Y + 2.4 * mm, "DETAILWEAVE 4K 2026")
    c.setFont(JP_REGULAR, 5.6)
    c.drawString(MARGIN_X, 7 * mm, "二つの分割候補から細部を選ぶ高精細拡大")
    c.drawRightString(PAGE_WIDTH - MARGIN_X, 7 * mm, str(document.page))
    c.restoreState()


def image_flowable(path: Path, width: float) -> PlatypusImage:
    with Image.open(path) as image:
        ratio = image.height / image.width
    return PlatypusImage(str(path), width=width, height=width * ratio)


def paper_facts(
    qa: dict,
    manifest: dict,
    comparison: dict,
) -> dict[str, float]:
    run = qa["run"]
    stats = manifest["stage_reports"][-1]["consensus_stats"]
    comparison_metrics = comparison["metrics_to_resize_only"]
    single_grid = comparison_metrics["phase_a"]
    detailweave = comparison_metrics["selected"]
    return {
        "phase_a": float(stats["phaseweave_phase0_selected_percent"]),
        "phase_b": float(stats["phaseweave_phase1_selected_percent"]),
        "input_rejected": float(stats["phaseweave_input_rejected_percent"]),
        "uncertain_fused": float(stats["phaseweave_uncertain_fused_percent"]),
        "boundary": float(run["transition_area_percent"]),
        "support_weight": float(stats["phaseweave_mean_support_weight"]),
        "sg_mae": float(single_grid["mae_8bit"]),
        "dw_mae": float(detailweave["mae_8bit"]),
        "sg_ssim": float(single_grid["luma_ssim_half_resolution"]),
        "dw_ssim": float(detailweave["luma_ssim_half_resolution"]),
        "sg_highpass": float(single_grid["highpass_mean_ratio"]),
        "dw_highpass": float(detailweave["highpass_mean_ratio"]),
        "sg_low_drift": float(single_grid["low_frequency_luma_drift_mean"]),
        "dw_low_drift": float(detailweave["low_frequency_luma_drift_mean"]),
        "sg_boundary": float(
            single_grid["residual_boundary_to_global_p95_ratio"]
        ),
        "dw_boundary": float(
            detailweave["residual_boundary_to_global_p95_ratio"]
        ),
    }


def title_story(facts: dict[str, float]) -> list[Flowable]:
    return [
        para("DetailWeave 4K：二つの分割候補から<br/>細部を選ぶ高精細拡大", "title"),
        para(
            "DetailWeave 4K: Input-Preserving Local Selection of Two Tiled Candidates",
            "english_title",
        ),
        para("あいきみ", "author"),
        para(
            "<b>要旨：</b> 大画像処理では、画像と潜在画像の変換を小領域へ分ける分割VAEと、一つの格子で画像から画像への局所生成を行う分割拡大が広く使われる。前者は変換の省メモリー化であり、新しい細部を生成しない。後者はGPU作業領域を固定できる一方、格子位置によって細線が変わる。本稿は、半格子ずらした二つの完成候補からA、B、入力維持を局所選択するDetailWeave 4Kを示す。Lanczos補間、単一格子分割、提案法を2897×4096で比較した。本手法は特定の画像モデルへ依存しない。",
            "abstract",
        ),
        para(
            "<b>キーワード：</b> 4Kアップスケール、分割VAE、分割生成、局所選択、入力維持",
            "keywords",
        ),
    ]


def page1_left_story(grid_path: Path) -> list[Flowable]:
    grid = image_flowable(grid_path, COLUMN_WIDTH)
    return [
        section("1", "はじめに"),
        para(
            "画像を大きくする最も単純な方法は、周囲の画素から新しい画素を計算する補間である。Lanczos補間は構図と色を保つが、元画像にない髪の細線や布目は生成しない。生成モデルを使えば細部を描き足せる一方、大画像を一度に扱うには多くのメモリーが必要になる。",
        ),
        para(
            "現在はVAEの変換を重なり付き領域へ分ける分割VAEが一般化している。しかし、これは入口と出口の変換を省メモリー化する機構であり、拡散生成自体を分けない。局所生成まで分ける単一格子法は大画像を扱えるが、領域位置が変わると同じ毛束や文字状の線が数ピクセルずれる。",
        ),
        section("2", "比較する三つの拡大法"),
        subsection("2.1", "Lanczos補間"),
        para(
            "入力画素だけから拡大する。生成による形の変化はないが、新しい意味的細部も増えない。本稿では入力由来の形を確認する基準とする。",
        ),
        subsection("2.2", "分割VAEと単一格子分割"),
        para(
            "潜在拡散モデルではVAEが画像と潜在画像を相互変換する [1]。分割VAEはこの符号化と復号だけを重なり付き領域へ分け、境界を重み付き平均する。指示文による生成や細部選択は行わない。実測比較には、画像から画像への局所生成と候補整形を共有し、一つの格子だけを採用する単一格子版を用いる。",
        ),
        subsection("2.3", "DetailWeave 4K"),
        para(
            "位置をずらした二配置を別々に最後まで処理し、同寸法の候補A、Bを作る。完成後に場所ごとにA、B、入力維持を選ぶ。単一格子法と同じ省メモリー性を保ちながら、格子位置への依存と不適切な描き足しを抑える。",
        ),
        section("3", "提案手法"),
        subsection("3.1", "二つの格子配置"),
        para(
            "一領域では1280×1280をモデルへ見せ、中央960×960を候補へ戻す。周囲160ピクセルは文脈として使う。次の中央部は880ピクセル先なので、隣同士は80ピクセル重なる。",
        ),
        para(
            "出力2897×4096に対し、配置Aは4列×5行の20領域、配置Bは格子を縦横へ440ピクセルずらした4列×6行の24領域である。AだけでもBだけでも画像全体を完成させる。左右分割や市松分担ではない。端では画像外へ端画素を延長し、入力寸法を保つ。",
        ),
        KeepTogether(
            [
                grid,
                para(
                    "図1　二つの配置と実測選択図。A、Bはいずれも画像全体を完成させる。完成後に橙=A、青=B、灰=入力、紫=弱い融合として選ぶ。",
                    "caption",
                ),
            ]
        ),
    ]


def page1_right_story() -> list[Flowable]:
    parameter_rows = [
        ["項目", "値"],
        ["配置A / B", "20 / 24領域"],
        ["モデル入力 / 採用部", "1280 / 960 px"],
        ["間隔 / 重なり", "880 / 80 px"],
        ["得点 / 忠実度の案内半径", "16 / 8 px"],
        ["選択しきい値 τ", "0.03"],
        ["忠実度の最低値", "0.42"],
        ["島除去 / 境界接続", "3000 / 5 px"],
        ["低周波係数（輝度 / 色差）", "0.32 / 0.18"],
        ["補助上限 / 保持下限", "0.10 / 0.90"],
    ]
    return [
        subsection("3.2", "配置ごとの独立再構成"),
        para(
            "入力補間画像に対する、配置 g、領域 t の変化を Δ<sub>gt</sub>、窓重みを w<sub>gt</sub> とする。配置ごとに重み付き和 D<sub>g</sub>、重み和 W<sub>g</sub>、二乗量 E<sub>g</sub> を蓄積し、他方を混ぜず候補変化 R<sub>g</sub> を求める。",
        ),
        equation("Dg = Σt wgt Δgt,   Wg = Σt wgt,   Eg = Σt wgt mean(Δgt²)", 1),
        equation("Rg = Dg / Wg", 2),
        para(
            "同じ配置内で重なる領域のばらつきから、結果がどれだけ揃うかを表す安定度 C<sub>g</sub> を求める。ここまではAとBを混ぜない。",
        ),
        subsection("3.3", "低周波を入力へ寄せる"),
        para(
            "候補差分を標準偏差12ピクセル相当の低周波 R<sup>L</sup><sub>g</sub> と高周波 R<sup>H</sup><sub>g</sub> に分ける。高周波は保ち、低周波は輝度0.32、色差0.18へ弱める。入力輝度0.85以上では輝度係数をさらに半減する。",
        ),
        equation("RgL = G12(Rg),   RgH = Rg - RgL", 3),
        equation("R̂g = RgH + ηY RgL,Y + ηC RgL,C", 4),
        subsection("3.4", "細部量と入力忠実度"),
        para(
            "変化の大きさだけを品質とはみなさない。高周波量 H<sub>g</sub> に、配置内安定度と入力忠実度 F<sub>g</sub> を掛ける。d<sub>edge</sub> は輪郭方向の不一致、d<sub>low</sub> は低周波輝度差、d<sub>chroma</sub> は低周波色差である。",
        ),
        equation("Qg = Hg (0.25 + 0.75 Cg) Fg", 5),
        equation("Fg = exp(-1.25 dedge - 1.00 dlow - 0.75 dchroma)", 6),
        subsection("3.5", "A、B、入力維持の三値選択"),
        para(
            "入力輝度を案内画像として得点を半径16、忠実度を半径8で整理し、S=(Q<sub>B</sub>-Q<sub>A</sub>)/(Q<sub>A</sub>+Q<sub>B</sub>+ε) とする。S&gt;0.03ならB、S&lt;-0.03ならA、それ以外は未確定とする。ただし候補の忠実度が0.42未満なら確定させない。",
        ),
        para(
            "未確定部には周囲の確かな選択を伝える。3000画素未満のA/B島と穴を整理し、両候補が不適切なら入力補間へ戻す。二候補が十分近い場合だけ±1ピクセルで位置を合わせて弱く融合し、整理後の境界だけを5ピクセルで接続する。",
        ),
        subsection("3.6", "補助候補を弱く使う"),
        para(
            "選択差分をR*、位置合わせした補助差分をR<sup>-</sup><sub>align</sub>、近さをaとする。局所構造類似度が0.90以上で輪郭方向も揃う場合だけ補助する。aを二乗するため、かなり近いときに限って最大10%が働く。",
        ),
        equation("Rsup = R* + 0.10 a² (R-align - R*)", 7),
        equation("Rout = (0.90 + 0.10 a) Rsup", 8),
        para("<b>表1　提案法の主要設定</b>", "table_caption"),
        academic_table(parameter_rows, [COLUMN_WIDTH * 0.63, COLUMN_WIDTH * 0.37], centered_columns=(1,)),
    ]


def page2_figure_story(comparison_path: Path) -> list[Flowable]:
    figure = image_flowable(comparison_path, CONTENT_WIDTH)
    return [
        figure,
        para(
            "図2　上：Lanczos、候補A（単一格子）、候補B、最終像の全体比較。下：同一座標の400%拡大。矢印は候補差、差分像は |A−最終|、右端は最終像へ選択マスクを重ねたもの。髪ではB、本の頁ではAが主に採用された。",
            "caption",
        ),
    ]


def page2_left_story(facts: dict[str, float]) -> list[Flowable]:
    condition_rows = [
        ["方法", "処理"],
        ["Lanczos", "補間のみ"],
        ["単一格子分割", "配置Aの20領域を全面採用"],
        ["DetailWeave", "A 20領域 + B 24領域"],
        ["生成条件", "各領域6回（Exact Steps）、変化強度0.16"],
    ]
    metric_rows = [
        ["評価量", "Lanczos", "単一格子", "提案法"],
        ["細かな明暗変化", "1.000", f"{facts['sg_highpass']:.3f}", f"{facts['dw_highpass']:.3f}"],
        ["低周波明暗ずれ", "0.000", f"{facts['sg_low_drift']:.3f}", f"{facts['dw_low_drift']:.3f}"],
        ["輝度SSIM", "1.000", f"{facts['sg_ssim']:.3f}", f"{facts['dw_ssim']:.3f}"],
        ["RGB平均差", "0.000", f"{facts['sg_mae']:.3f}", f"{facts['dw_mae']:.3f}"],
        ["境界ジャンプ比 J↓", "-", f"{facts['sg_boundary']:.3f}", f"{facts['dw_boundary']:.3f}"],
    ]
    return [
        section("4", "事例検証"),
        subsection("4.1", "条件"),
        para(
            "同じ入力を2897×4096へ拡大した。単一格子像はDetailWeaveの配置Aを全面採用したもので、二方式は同じ局所生成を共有する。生成には<b>独自に調整したKrea2 Turboモデル</b>、同じ指示文、各領域6回（Exact Steps）、変化強度0.16を用いた。これは実験条件であり、選択式には関与しない。",
        ),
        para("<b>表2　事例画像の処理条件</b>", "table_caption"),
        academic_table(condition_rows, [COLUMN_WIDTH * 0.48, COLUMN_WIDTH * 0.52], centered_columns=(1,)),
        subsection("4.2", "評価方法"),
        para(
            "Lanczos像を正解ではなく入力維持の基準とする。細かな明暗変化は、輝度から2ピクセル相当の平滑成分を引いた絶対値で測り、Lanczosを1とした。低周波明暗ずれは12ピクセル相当の平滑像との差、RGB平均差は0から255の画素値差である。",
        ),
        para(
            "SSIMは輝度構造の近さを表し、1に近いほどLanczos像へ近い。境界ジャンプ比Jは、A/B両配置の計画境界における残差ジャンプの95百分位を、画像全体の同値で割る。J=1は全体と同程度、J&gt;1は境界への集中、J&lt;1では小さいほど境界変化が相対的に弱い。いずれも細部の正しさを直接判定しないため、図2を併用する。",
        ),
        section("5", "結果"),
        para(
            "三方式とも2897×4096を出力した。単一格子像と提案法は一回の同じ領域生成から得ており、差は二配置の完成後に行う選択・拒否・接続にある。",
        ),
        para("<b>表3　Lanczos像を基準とした測定</b>", "table_caption"),
        academic_table(
            metric_rows,
            [COLUMN_WIDTH * 0.40, COLUMN_WIDTH * 0.20, COLUMN_WIDTH * 0.20, COLUMN_WIDTH * 0.20],
            centered_columns=(1, 2, 3),
        ),
        para(
            "単一格子の細かな明暗変化は{sg_highpass:.3f}倍、提案法は{dw_highpass:.3f}倍であった。提案法は変化量を最大化せず、輝度SSIMを{sg_ssim:.3f}から{dw_ssim:.3f}、RGB平均差を{sg_mae:.3f}から{dw_mae:.3f}へ改善した。".format(**facts),
        ),
        para(
            "境界ジャンプ比Jは単一格子の{sg_boundary:.3f}から{dw_boundary:.3f}へわずかに低下し、計画境界への変化集中は増えなかった。図2の髪ではB、本の頁ではAが主に選ばれ、全体でもA {phase_a:.2f}%、B {phase_b:.2f}%であった。".format(**facts),
        ),
    ]


def page2_right_story(facts: dict[str, float]) -> list[Flowable]:
    use_case_rows = [
        ["用途・重視する点", "向いている方式"],
        ["VAE変換だけを省メモリー化", "分割VAE"],
        ["計算量を抑えた局所生成", "単一格子"],
        ["元画像に忠実な高精細拡大", "DetailWeave"],
        ["髪や文字など細線を守る", "DetailWeave"],
        ["不適切な生成を入力へ戻す", "DetailWeave"],
        ["GPUメモリーと出力解像度を分離", "単一格子・DetailWeave"],
        ["32K・64Kへ段階的に拡大", "単一格子・DetailWeave"],
    ]
    references = [
        "[1] R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” <i>Proc. CVPR</i>, pp. 10684-10695, 2022.",
        "[2] O. Bar-Tal et al., “MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation,” <i>Proc. ICML</i>, PMLR 202, pp. 1737-1752, 2023.",
        "[3] K. He, J. Sun, and X. Tang, “Guided Image Filtering,” <i>Proc. ECCV</i>, pp. 1-14, 2010.",
        "[4] Z. Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” <i>IEEE Trans. Image Processing</i>, 13(4), pp. 600-612, 2004.",
    ]
    return [
        section("6", "標準的な分割処理との違い"),
        subsection("6.1", "分割VAEは入口と出口を分ける"),
        para(
            "VAEは画像を潜在画像へ変換し、最後に画像へ戻す。分割VAEはこの計算を小領域へ分けるため、VAEのGPU作業量を抑えられる。ただし一般的な実装は、全体潜在画像、全体RGB出力、加算・重みバッファをCPUメモリーに保持する。新しい細部を生成せず、ノイズ除去や指示文にも手を加えない。",
        ),
        para(
            "DetailWeaveが分けるのは画像から画像への局所生成の全経路である。二候補を生成して忠実度を評価し、A、Bを拒否して入力維持も選べる。分割VAEの代替ではなく、その上で動く生成・選択層であり、各領域には通常VAEと分割VAEのどちらも使える。MultiDiffusion [2] はさらに別で、各段階の予測を一つの共有潜在画像へ戻す。",
        ),
        subsection("6.2", "単一格子に加えるもの"),
        para(
            "単一格子分割とDetailWeaveは、一度に見る領域を固定するため、どちらも<b>GPUメモリー使用量を出力解像度から切り離せる</b>。単一格子は一つの完成像を採用する。DetailWeaveは半格子ずらした二候補を保ち、局所的にA、B、入力維持を選ぶ。",
        ),
        subsection("6.3", "用途による使い分け"),
        para("<b>表4　目的に応じた方式の選択</b>", "table_caption"),
        academic_table(
            use_case_rows,
            [COLUMN_WIDTH * 0.68, COLUMN_WIDTH * 0.32],
            centered_columns=(1,),
        ),
        para(
            "出力を2倍にすると領域数、処理時間、一時保存量はおおむね4倍になるが、一領域のGPU作業量は変わらない。この局所生成部は32K・64Kにも同じ処理を反復できる。最終復号と保存まで総メモリーを一定にする場合は、全体RGBを保持しない帯状の逐次復号・逐次保存を組み合わせる。",
        ),
        section("7", "結論"),
        para(
            "DetailWeaveは分割VAEを置き換えず、単一格子へ二候補と入力維持の選択を加える。実測では入力構造への近さを改善し、計画境界への変化集中を増やさなかった。",
        ),
        para("参考文献", "subsection"),
        compact_reference_table(references),
    ]


def validate_inputs(
    source: Path,
    single_grid: Path,
    phase_b: Path,
    result: Path,
    selection_map: Path,
    manifest: dict,
    qa: dict,
    comparison: dict,
) -> None:
    image_paths = (source, single_grid, phase_b, result, selection_map)
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"paper image inputs missing: {missing}")
    for path in (single_grid, phase_b, result, selection_map):
        with Image.open(path) as image:
            if image.size != (2897, 4096):
                raise ValueError(f"unexpected image dimensions for {path}: {image.size}")
    if manifest.get("target_size") != [2897, 4096]:
        raise ValueError("manifest target dimensions do not match")
    stage = manifest["stage_reports"][-1]
    phase_counts = {
        phase: sum(1 for tile in stage["tiles"] if int(tile["phase"]) == phase)
        for phase in (0, 1)
    }
    if phase_counts != {0: 20, 1: 24}:
        raise ValueError(f"unexpected phase tile counts: {phase_counts}")
    config = manifest.get("phaseweave") or {}
    expected_config = {
        "selection_mode": "ternary_input_fallback",
        "selection_margin": 0.03,
        "fidelity_reject_threshold": 0.42,
        "fidelity_guided_radius": 8,
        "island_min_area": 3000,
        "feather_radius": 5,
        "low_frequency_sigma": 12.0,
        "support_mix": 0.1,
        "support_confidence_power": 2.0,
        "support_alignment_radius": 1,
    }
    mismatched = {
        key: (config.get(key), expected)
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"paper algorithm configuration mismatch: {mismatched}")
    required_checks = (
        "target_is_2897x4096",
        "phaseweave_identity_metadata_matches",
        "all_tiles_processed",
        "no_tiles_skipped",
        "uniform_edge_balanced_grid",
        "all_model_inputs_are_1280_square",
        "minimum_canvas_intersection_is_388x328_or_larger",
        "both_shifted_divisions_selected",
        "transition_area_below_40_percent",
        "single_representative_area_above_60_percent",
        "planned_boundary_jump_ratio_below_1_5",
    )
    failures = [name for name in required_checks if qa["checks"].get(name) is not True]
    if failures:
        raise ValueError(f"required QA checks failed: {failures}")
    comparison_scope = str(comparison.get("comparison_scope") or "")
    if "phase A" not in comparison_scope or "ternary result" not in comparison_scope:
        raise ValueError("single-grid comparison scope does not match the paper")
    comparison_metrics = comparison.get("metrics_to_resize_only") or {}
    required_metrics = (
        "mae_8bit",
        "luma_ssim_half_resolution",
        "highpass_mean_ratio",
        "low_frequency_luma_drift_mean",
        "residual_boundary_to_global_p95_ratio",
    )
    missing_metrics = {
        method: [key for key in required_metrics if key not in comparison_metrics.get(method, {})]
        for method in ("phase_a", "selected")
    }
    missing_metrics = {method: keys for method, keys in missing_metrics.items() if keys}
    if missing_metrics:
        raise ValueError(f"comparison metrics missing: {missing_metrics}")


def japanese_fonts_embedded(reader: PdfReader) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for page in reader.pages:
        fonts = (page["/Resources"].get("/Font") or {}).get_object()
        for reference in fonts.values():
            font = reference.get_object()
            name = str(font.get("/BaseFont"))
            descriptor = font.get("/FontDescriptor")
            if not descriptor and font.get("/DescendantFonts"):
                descendant = font["/DescendantFonts"][0].get_object()
                descriptor = descendant.get("/FontDescriptor")
            descriptor = descriptor.get_object() if descriptor else None
            embedded = bool(
                descriptor
                and any(
                    descriptor.get(key)
                    for key in ("/FontFile", "/FontFile2", "/FontFile3")
                )
            )
            if "YuMincho" in name or "YuMin" in name or "Mincho" in name:
                result[name] = embedded
    return result


def validate_pdf(path: Path) -> dict:
    reader = PdfReader(str(path))
    if len(reader.pages) != 2:
        raise RuntimeError(f"expected 2 pages, got {len(reader.pages)}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    forbidden = (
        "2:3",
        "1:1.414",
        "縦横比",
        "上下だけ",
        "切り取り",
        "Consensus",
        "VRAM-Canvas",
        "PhaseWeave",
        "Krea2 PhaseWeave",
        "シード",
        "seed",
        "旧二値選択版",
        "配置A単独",
        "配置B単独",
        "限界",
    )
    found = [term for term in forbidden if re.search(re.escape(term), text, re.I)]
    if found:
        raise RuntimeError(f"forbidden terms in paper: {found}")
    required = (
        "あいきみ",
        "DetailWeave 4K",
        "局所選択",
        "分割VAE",
        "単一格子",
        "画像と潜在画像の変換",
        "新しい細部を生成しない",
        "配置A",
        "配置B",
        "20領域",
        "24領域",
        "MultiDiffusion",
        "入力維持",
        "0.03",
        "低周波",
        "独自に調整したKrea2 Turboモデル",
        "特定の画像モデルへ依存しない",
        "共有潜在画像",
        "A、Bを拒否して入力維持",
        "GPUメモリー使用量を出力解像度から切り離せる",
        "単一格子・DetailWeave",
        "32K・64K",
        "分割VAEを置き換えず",
    )
    compact_text = re.sub(r"\s+", "", text)
    missing = [term for term in required if re.sub(r"\s+", "", term) not in compact_text]
    if missing:
        raise RuntimeError(f"required terms missing from paper: {missing}")
    krea2_count = len(re.findall("Krea2", compact_text, re.I))
    if krea2_count != 1:
        raise RuntimeError(
            "Krea2 must appear exactly once, in the qualified experiment-model phrase"
        )
    page_sizes = []
    for page in reader.pages:
        width_mm = float(page.mediabox.width) / mm
        height_mm = float(page.mediabox.height) / mm
        page_sizes.append([round(width_mm, 3), round(height_mm, 3)])
        if abs(width_mm - 182) > 0.1 or abs(height_mm - 257) > 0.1:
            raise RuntimeError("paper is not JIS B5")
    out_of_bounds = []
    used_fonts = set()
    with pdfplumber.open(str(path)) as document:
        for page_number, page in enumerate(document.pages, 1):
            for char in page.chars:
                used_fonts.add(char["fontname"])
                if (
                    char["x0"] < -0.1
                    or char["x1"] > page.width + 0.1
                    or char["top"] < -0.1
                    or char["bottom"] > page.height + 0.1
                ):
                    out_of_bounds.append(
                        [page_number, char.get("text"), char["x0"], char["x1"], char["top"], char["bottom"]]
                    )
    if out_of_bounds:
        raise RuntimeError(f"text outside page: {out_of_bounds[:5]}")
    embedded = japanese_fonts_embedded(reader)
    if not embedded or not all(embedded.values()):
        raise RuntimeError(f"Japanese font embedding failed: {embedded}")
    return {
        "pages": len(reader.pages),
        "page_sizes_mm": page_sizes,
        "author": reader.metadata.author,
        "text_characters": len(text),
        "forbidden_terms": found,
        "out_of_bounds_characters": len(out_of_bounds),
        "used_fonts": sorted(used_fonts),
        "embedded_japanese_fonts": embedded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--single-grid", type=Path, default=DEFAULT_SINGLE_GRID
    )
    parser.add_argument("--phase-b", type=Path, default=DEFAULT_PHASE_B)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--selection-map", type=Path, default=DEFAULT_SELECTION_MAP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--font")
    parser.add_argument("--bold-font")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    validate_inputs(
        args.source,
        args.single_grid,
        args.phase_b,
        args.result,
        args.selection_map,
        manifest,
        qa,
        comparison,
    )
    facts = paper_facts(qa, manifest, comparison)
    regular = find_font(args.font, bold=False)
    bold = find_font(args.bold_font, bold=True)
    register_fonts(regular, bold)

    source = load_rgb(args.source)
    single_grid = load_rgb(args.single_grid)
    phase_b = load_rgb(args.phase_b)
    result = load_rgb(args.result)
    selection_map = load_rgb(args.selection_map)
    grid_path = args.asset_dir / "phaseweave_grid_layout.png"
    comparison_path = args.asset_dir / "phaseweave_paper_comparison.png"
    make_grid_figure(manifest, source, selection_map, grid_path)
    make_comparison_figure(
        source,
        single_grid,
        phase_b,
        result,
        selection_map,
        comparison_path,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = AcademicDocTemplate(
        str(args.output),
        pageCompression=1,
        title="DetailWeave 4K：二つの分割候補から細部を選ぶ高精細拡大",
        author="あいきみ",
        subject="DetailWeave 4K technical paper",
    )
    story: list[Flowable] = []
    story.extend(title_story(facts))
    story.append(FrameBreak())
    story.extend(page1_left_story(grid_path))
    story.append(FrameBreak())
    story.extend(page1_right_story())
    story.append(NextPageTemplate("page2"))
    story.append(PageBreak())
    story.extend(page2_figure_story(comparison_path))
    story.append(FrameBreak())
    story.extend(page2_left_story(facts))
    story.append(FrameBreak())
    story.extend(page2_right_story(facts))
    document.build(story)

    report = validate_pdf(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "grid_figure": str(grid_path),
                "comparison_figure": str(comparison_path),
                "font": str(regular),
                "bold_font": str(bold),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
