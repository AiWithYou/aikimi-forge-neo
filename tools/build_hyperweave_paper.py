"""Build and validate a two-page Japanese B5 HyperWeave technical paper."""

from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
import os
from pathlib import Path
import re

import pdfplumber
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("H:/tmp/image-cropped.png")
DEFAULT_RESULT = (
    REPO_ROOT
    / "outputs/hyperweave_image_cropped_full4k_structure_safe_s1_20260724_112523.png"
)
DEFAULT_COMPARISON_DIR = (
    REPO_ROOT / "outputs/hyperweave_comparison_full4k_20260724_113800"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/pdf/hyperweave_4k_b5_ja.pdf"
DEFAULT_ASSET_DIR = REPO_ROOT / "output/hyperweave_paper_20260724/assets"

EXPECTED_SOURCE_SHA256 = (
    "4dbf27fca20acdddbc19ec55f2a6a27c396daaeab9f3cc4a287d609efb1148e0"
)
EXPECTED_RESULT_SHA256 = (
    "3467bc81351921127ade757a34ebb937a52f88ff38cd4bd60021ff7cbf7317ca"
)

JIS_B5 = portrait((182 * mm, 257 * mm))
PAGE_WIDTH, PAGE_HEIGHT = JIS_B5
MARGIN_X = 10.5 * mm
BOTTOM = 12.0 * mm
TOP = 10.5 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
COLUMN_GAP = 5.0 * mm
COLUMN_WIDTH = (CONTENT_WIDTH - COLUMN_GAP) / 2

INK = colors.HexColor("#16181B")
MID = colors.HexColor("#515861")
LIGHT = colors.HexColor("#BBC2CA")
VERY_LIGHT = colors.HexColor("#EEF1F4")
BLUE = colors.HexColor("#2A5D7B")
TEAL = colors.HexColor("#2B7876")
ORANGE = colors.HexColor("#B95F39")
WHITE = colors.white

JP_REGULAR = "HyperWeaveAcademicMincho-Regular"
JP_BOLD = "HyperWeaveAcademicMincho-Bold"


def find_font(explicit: str | None, *, bold: bool) -> Path:
    env_name = "HYPERWEAVE_PAPER_JP_BOLD_FONT" if bold else "HYPERWEAVE_PAPER_JP_FONT"
    candidates = [
        explicit,
        os.environ.get(env_name),
        "C:/Windows/Fonts/yumindb.ttf" if bold else "C:/Windows/Fonts/yumin.ttf",
        (
            "C:/Windows/Fonts/YuMincho-Demibold.ttc"
            if bold
            else "C:/Windows/Fonts/YuMincho.ttc"
        ),
        "C:/Windows/Fonts/msmincho.ttc",
        (
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
            if bold
            else "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
        ),
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
        "title",
        size=13.2,
        leading=15.6,
        bold=True,
        alignment=TA_CENTER,
        space_after=1.5,
    ),
    "english": paragraph_style(
        "english-title",
        size=5.9,
        leading=7.0,
        alignment=TA_CENTER,
        color=MID,
        space_after=1.8,
    ),
    "author": paragraph_style(
        "author",
        size=7.4,
        leading=8.6,
        bold=True,
        alignment=TA_CENTER,
        space_after=2.3,
    ),
    "abstract": paragraph_style(
        "abstract",
        size=5.9,
        leading=7.35,
        alignment=TA_JUSTIFY,
    ),
    "keywords": paragraph_style(
        "keywords",
        size=5.55,
        leading=6.7,
        alignment=TA_LEFT,
        space_before=1.4,
        space_after=1.8,
    ),
    "section": paragraph_style(
        "section",
        size=8.65,
        leading=10.35,
        bold=True,
        alignment=TA_LEFT,
        space_before=3.2,
        space_after=1.2,
        keep_with_next=True,
    ),
    "subsection": paragraph_style(
        "subsection",
        size=7.3,
        leading=8.8,
        bold=True,
        alignment=TA_LEFT,
        space_before=2.0,
        space_after=0.7,
        keep_with_next=True,
    ),
    "body": paragraph_style(
        "body",
        size=6.55,
        leading=8.3,
        alignment=TA_JUSTIFY,
        first_indent=6.55,
        space_after=1.1,
    ),
    "body_no_indent": paragraph_style(
        "body-no-indent",
        size=6.55,
        leading=8.3,
        alignment=TA_JUSTIFY,
        space_after=1.1,
    ),
    "compact": paragraph_style(
        "compact",
        size=5.9,
        leading=7.35,
        alignment=TA_JUSTIFY,
        space_after=0.8,
    ),
    "formula": paragraph_style(
        "formula",
        size=5.7,
        leading=7.0,
        alignment=TA_CENTER,
        space_before=0.6,
        space_after=1.1,
    ),
    "caption": paragraph_style(
        "caption",
        size=5.25,
        leading=6.4,
        alignment=TA_CENTER,
        space_before=0.8,
        space_after=1.4,
    ),
    "table": paragraph_style(
        "table-cell",
        size=5.15,
        leading=6.25,
        alignment=TA_LEFT,
    ),
    "table_right": paragraph_style(
        "table-cell-right",
        size=5.15,
        leading=6.25,
        alignment=TA_LEFT,
    ),
    "reference": paragraph_style(
        "reference",
        size=4.7,
        leading=5.8,
        alignment=TA_LEFT,
        left_indent=7.0,
        first_indent=-7.0,
        space_after=0.6,
        color=MID,
    ),
    "note": paragraph_style(
        "note",
        size=5.15,
        leading=6.45,
        alignment=TA_JUSTIFY,
        left_indent=3.0,
        right_indent=3.0,
        color=MID,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with Image.open(path) as image:
        raw = image.info.get("hyperweave")
        if not raw:
            raise ValueError("result PNG does not contain hyperweave metadata")
        return json.loads(raw)


def validate_inputs(
    source_path: Path,
    result_path: Path,
    comparison_dir: Path,
) -> tuple[dict, dict]:
    required = [
        source_path,
        result_path,
        comparison_dir / "metrics.json",
        comparison_dir / "crop_left_face.png",
        comparison_dir / "crop_right_face.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"paper inputs missing: {missing}")
    if sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source image SHA-256 does not match the measured run")
    if sha256(result_path) != EXPECTED_RESULT_SHA256:
        raise ValueError("result image SHA-256 does not match the measured run")

    with Image.open(source_path) as source:
        if source.size != (1664, 2353) or source.mode != "RGBA":
            raise ValueError(
                f"unexpected source image: size={source.size} mode={source.mode}"
            )
        if source.getextrema()[3] != (255, 255):
            raise ValueError("source alpha is not fully opaque")
    with Image.open(result_path) as result:
        if result.size != (2897, 4096) or result.mode != "RGBA":
            raise ValueError(
                f"unexpected result image: size={result.size} mode={result.mode}"
            )
        if result.getextrema()[3] != (255, 255):
            raise ValueError("result alpha is not fully opaque")

    manifest = load_manifest(result_path)
    expected_manifest = {
        "version": "1.0.0",
        "preset": "Structure Safe",
        "content_profile": "Illustration / Anime",
        "source_size": [1664, 2353],
        "target_size": [2897, 4096],
        "seed": 976834651,
        "exact_steps": 1,
        "detected_faces": 2,
        "memmap_usage": False,
    }
    mismatched = {
        key: (manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"result manifest mismatch: {mismatched}")

    stages = manifest["quality"]["stage_reports"]
    if len(stages) != 1:
        raise ValueError("paper run must contain exactly one upscale stage")
    stage = stages[0]
    stage_expected = {
        "processing_size": [2904, 4096],
        "selected_global_candidate": None,
    }
    stage_mismatched = {
        key: (stage.get(key), expected)
        for key, expected in stage_expected.items()
        if stage.get(key) != expected
    }
    if stage_mismatched:
        raise ValueError(f"stage report mismatch: {stage_mismatched}")
    if len(stage.get("rois", [])) != 1:
        raise ValueError("paper run must contain one merged coherent face ROI")
    if manifest["model"].get("internal_calls") != 49:
        raise ValueError("paper run must contain 49 internal model calls")
    if stage["seam"].get("boundary_count") != 8:
        raise ValueError("paper run must contain eight planned boundaries")

    metrics = json.loads((comparison_dir / "metrics.json").read_text(encoding="utf-8"))
    if set(metrics) != {"Lanczos", "HyperWeave-full4K"}:
        raise ValueError(f"unexpected comparison methods: {sorted(metrics)}")
    expected_metrics = {
        ("Lanczos", "roundtrip_ssim"): 0.9943106770515442,
        ("HyperWeave-full4K", "roundtrip_ssim"): 0.8737668395042419,
        ("HyperWeave-full4K", "seam_ratio"): 0.8104278695246762,
        ("HyperWeave-full4K", "processing_time_seconds"): 662.117528200004,
        ("HyperWeave-full4K", "peak_reserved_vram_bytes"): 21720203264,
    }
    metric_mismatches = {}
    for (method, key), expected in expected_metrics.items():
        actual = metrics[method].get(key)
        if actual is None or abs(float(actual) - expected) > 1e-10:
            metric_mismatches[f"{method}.{key}"] = (actual, expected)
    if metric_mismatches:
        raise ValueError(f"comparison metrics mismatch: {metric_mismatches}")
    return manifest, metrics


def pil_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    *,
    spacing: int = 4,
) -> None:
    bounds = draw.multiline_textbbox(
        (0, 0), text, font=font, align="center", spacing=spacing
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - width) / 2
    y = y0 + (y1 - y0 - height) / 2 - bounds[1]
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=spacing,
    )


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    radius: int = 22,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=3,
    )


def make_pipeline_figure(
    output: Path,
    regular_font: Path,
    bold_font: Path,
) -> None:
    width, height = 2200, 350
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = pil_font(bold_font, 28)
    box_font = pil_font(bold_font, 25)
    note_font = pil_font(regular_font, 20)
    draw.text((52, 22), "図1  HyperWeave の処理フロー", font=title_font, fill="#20262C")

    labels = [
        "入力解析\n段階計画",
        "重複タイル\n座標 noise",
        "Global\nAnchor",
        "全画面候補\nhard reject",
        "六周波数帯\n残差合成",
        "Face/Head\nROI 再描画",
        "seam 評価\n低周波 BP",
    ]
    fills = [
        "#EAF1F7",
        "#E9F4F3",
        "#EDF0F8",
        "#F8ECE6",
        "#F4F0E3",
        "#EFEAF6",
        "#E8F1EC",
    ]
    outlines = [
        "#2A5D7B",
        "#2B7876",
        "#5B6590",
        "#B95F39",
        "#8E762E",
        "#76538B",
        "#3D7957",
    ]
    margin = 50
    arrow_width = 42
    box_width = (width - 2 * margin - 6 * arrow_width) // 7
    y0, y1 = 82, 252
    for index, (label, fill, outline) in enumerate(
        zip(labels, fills, outlines, strict=True)
    ):
        x0 = margin + index * (box_width + arrow_width)
        x1 = x0 + box_width
        rounded_box(draw, (x0, y0, x1, y1), fill, outline)
        draw_centered_text(
            draw,
            (x0 + 8, y0 + 6, x1 - 8, y1 - 6),
            label,
            box_font,
            "#1B2228",
            spacing=7,
        )
        if index < len(labels) - 1:
            arrow_y = (y0 + y1) // 2
            draw.line(
                (x1 + 7, arrow_y, x1 + arrow_width - 8, arrow_y),
                fill="#66717B",
                width=5,
            )
            draw.polygon(
                [
                    (x1 + arrow_width - 8, arrow_y),
                    (x1 + arrow_width - 21, arrow_y - 10),
                    (x1 + arrow_width - 21, arrow_y + 10),
                ],
                fill="#66717B",
            )
    note = (
        "候補を全面採用せず、制約通過帯域だけを採用。"
        "全 Global 候補が不合格なら Anchor へ戻る。"
    )
    draw_centered_text(
        draw,
        (280, 276, 1920, 334),
        note,
        note_font,
        "#46515B",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def contain(
    source: Image.Image,
    size: tuple[int, int],
    *,
    background: str = "white",
) -> Image.Image:
    rgb = source.convert("RGB")
    fitted = ImageOps.contain(rgb, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    return canvas


def make_comparison_figure(
    source_path: Path,
    result_path: Path,
    comparison_dir: Path,
    output: Path,
    regular_font: Path,
    bold_font: Path,
) -> None:
    width, height = 2200, 930
    canvas = Image.new("RGB", (width, height), "#F7F8FA")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(bold_font, 29)
    label_font = pil_font(bold_font, 24)
    note_font = pil_font(regular_font, 19)
    draw.text(
        (48, 24),
        "図2  入力、Lanczos、HyperWeave と同一座標の顔クロップ",
        font=title_font,
        fill="#20262C",
    )

    with Image.open(source_path) as source_image:
        source = source_image.convert("RGB")
    with Image.open(result_path) as result_image:
        result = result_image.convert("RGB")
    lanczos = source.resize(result.size, Image.Resampling.LANCZOS)
    top_images = [
        ("入力 1664×2353", source),
        ("Lanczos 2897×4096", lanczos),
        ("HyperWeave 2897×4096", result),
    ]
    panel_width, panel_height = 640, 410
    panel_y = 100
    gap = 52
    start_x = (width - (3 * panel_width + 2 * gap)) // 2
    for index, (label, panel_image) in enumerate(top_images):
        x = start_x + index * (panel_width + gap)
        draw.rounded_rectangle(
            (x, panel_y, x + panel_width, panel_y + panel_height),
            radius=12,
            fill="white",
            outline="#C8CED5",
            width=2,
        )
        fitted = contain(panel_image, (panel_width - 18, panel_height - 52))
        canvas.paste(fitted, (x + 9, panel_y + 42))
        draw_centered_text(
            draw,
            (x + 5, panel_y + 5, x + panel_width - 5, panel_y + 39),
            label,
            label_font,
            "#242B31",
        )

    crop_y = 550
    crop_width, crop_height = 1032, 315
    for index, (name, label) in enumerate(
        [
            ("crop_left_face.png", "左人物：Lanczos / HyperWeave"),
            ("crop_right_face.png", "右人物：Lanczos / HyperWeave"),
        ]
    ):
        x = 48 + index * (crop_width + 40)
        with Image.open(comparison_dir / name) as crop_image:
            crop = contain(crop_image, (crop_width - 12, crop_height - 42))
        draw.rounded_rectangle(
            (x, crop_y, x + crop_width, crop_y + crop_height),
            radius=12,
            fill="white",
            outline="#C8CED5",
            width=2,
        )
        draw_centered_text(
            draw,
            (x + 5, crop_y + 3, x + crop_width - 5, crop_y + 37),
            label,
            label_font,
            "#242B31",
        )
        canvas.paste(crop, (x + 6, crop_y + 38))

    note = (
        "クロップは同一座標。差は生成による再描画であり、"
        "観測されていない真の高解像度情報ではない。"
    )
    draw_centered_text(
        draw,
        (210, 874, 1990, 922),
        note,
        note_font,
        "#515A63",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def paragraph(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def section(title: str) -> Paragraph:
    return paragraph(title, "section")


def subsection(title: str) -> Paragraph:
    return paragraph(title, "subsection")


def formula(text: str) -> Table:
    content = Paragraph(escape(text), STYLES["formula"])
    table = Table([[content]], colWidths=[COLUMN_WIDTH - 4])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), VERY_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.3, LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.0),
                ("TOPPADDING", (0, 0), (-1, -1), 1.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
            ]
        )
    )
    return table


def compact_table(
    rows: list[list[str]],
    widths: list[float],
    *,
    header: bool = True,
    align_right_columns: tuple[int, ...] = (),
) -> Table:
    data: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        cells = []
        for value in row:
            if row_index == 0 and header:
                cells.append(Paragraph(f"<b>{escape(value)}</b>", STYLES["table"]))
            else:
                cells.append(Paragraph(escape(value), STYLES["table"]))
        data.append(cells)
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), JP_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 5.15),
        ("LEADING", (0, 0), (-1, -1), 6.25),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.1),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EDF2")),
    ]
    for column in align_right_columns:
        commands.append(("ALIGN", (column, 1), (column, -1), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return table


def title_story(pipeline_path: Path) -> list[Flowable]:
    abstract = (
        "生成型アップスケールは補間では得られない描写を加えられる一方、"
        "人物同一性、髪型、物体配置、色、タイル境界を変える危険がある。"
        "本稿は、重複タイル、画像座標潜在ノイズ、低強度Anchor、全画面候補"
        "のハード棄却、六周波数帯の残差採用、顔ROI、低周波Back Projection"
        "を統合するHyperWeaveを示す。1664×2353のイラストをRTX 3090上で"
        "2897×4096へ生成し、49回の内部生成を662秒で完了した。Global候補"
        "は棄却され、AnchorとFace/Head ROIを用いた安全側の出力となった。"
        "忠実度ではLanczosが優位であり、本結果はフル4K実行可能性と棄却"
        "機構の作動を示す予備的事例である。"
    )
    return [
        paragraph(
            "HyperWeave 4K/8K：構造制約付き周波数選択によるタイル型生成アップスケール",
            "title",
        ),
        paragraph(
            "HyperWeave 4K/8K: Tiled Generative Upscaling with Structural "
            "Constraints and Frequency-Selective Residual Fusion",
            "english",
        ),
        paragraph("あいきみ", "author"),
        Table(
            [
                [
                    Paragraph("<b>要旨</b>", STYLES["abstract"]),
                    Paragraph(abstract, STYLES["abstract"]),
                ]
            ],
            colWidths=[10 * mm, CONTENT_WIDTH - 10 * mm],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.35, LIGHT),
                ]
            ),
        ),
        paragraph(
            "<b>キーワード：</b> 生成型超解像、潜在拡散、タイル生成、"
            "構造制約、周波数合成、4K",
            "keywords",
        ),
        PlatypusImage(
            str(pipeline_path),
            width=CONTENT_WIDTH,
            height=CONTENT_WIDTH * 350 / 2200,
        ),
        paragraph(
            "図1　候補を全面採用せず、制約を通過した帯域だけを合成する。",
            "caption",
        ),
    ]


def page1_left_story() -> list[Flowable]:
    return [
        section("1. はじめに"),
        paragraph(
            "Lanczos補間は構図と色を強く維持するが、元画像にない睫毛、"
            "髪内部線、布目などは生成しない。拡散型超解像は知覚的細部を"
            "推定できる一方、単一入力に対応する高解像度像は一意でなく、"
            "生成細部は観測事実ではない［1-4］。"
        ),
        paragraph(
            "HyperWeaveは入力を構図、顔向き、表情、髪型、衣装、物体、"
            "代表色、画風の設計図として扱う生成型再描画である。目的は"
            "画素一致だけでなく、意味構造の制約を満たす候補内で一貫した"
            "描写を増やすことである。Forgeの独立Scriptとして、現在の"
            "checkpoint、VAE、sampler、CFG、offloadを再利用する。"
        ),
        paragraph(
            "本稿が主張するのは、既存モデルを再学習せず、24 GiB GPUで"
            "フル4K候補を生成・評価できること、および不合格候補を実際に"
            "捨てられることである。未知の高解像度正解がないため、生成線や"
            "材質を真の復元とは扱わず、入力忠実度と帯域変化を分けて報告する。"
        ),
        section("2. 提案手法"),
        subsection("2.1 段階計画と重複タイル"),
        paragraph(
            "各段階の拡大率を2.0以下とし、内部寸法を潜在倍率8へ切り上げ、"
            "最後に指定寸法へ戻す。既定タイルは1280角、書込coreは960角、"
            "contextは各辺160、strideは768である。最終行列を画像端へ"
            "揃え、外周窓の重みを0にしない。"
        ),
        paragraph(
            "StageSpecはsource、target、内部processing寸法、倍率、各pass"
            "強度、周波数gain、入力から段階への座標変換を保持する。4Kや8Kへ"
            "一度に飛ばず、1024→2048→4096のように計画する。今回の1.741倍"
            "処理は一段階で、2897幅だけを一時的に2904へ揃えた。"
        ),
        formula("Δ = Σt wt Δt / max(Σt wt, ε)"),
        paragraph(
            "タイル差分Δtをraised-cosine窓wtで蓄積し、差分二乗量から"
            "重複間のtile confidenceを得る。候補画像を直接貼らず、"
            "基準画像と生成変化を分離する。"
        ),
        paragraph(
            "accumulatorは差分和、重み和、差分二乗輝度をfloat32でCPU RAM"
            "またはdisk memmapへ逐次蓄積する。GPUには現在タイルとlatentだけ"
            "を置くため、GPU作業量は4K/8K canvasの総画素数ではなくタイル寸法"
            "で決まる。"
        ),
        subsection("2.2 Coordinate Noise"),
        paragraph(
            "stage、pass、candidate、ROIをBLAKE2bで名前空間化し、"
            "PCG64で全画面に対応するCPU潜在noise canvasを作る。タイルは"
            "絶対潜在座標のcropを使うため、同一候補の重複位置で初期noise"
            "がbit-exactに一致する。保証は初期latent noiseまでであり、"
            "SDE sampler内部の二次noiseは対象外である。"
        ),
        formula(
            "seed' = BLAKE2b(base, stage, pass, candidate, roi); "
            "Nt = crop(Nglobal(seed'))"
        ),
        paragraph(
            "Forgeの通常静止画latent (B,C,H,W) と、Qwen Image系のsingleton"
            " temporal latent (B,C,1,H,W) の双方へ形を適応する。時間長1超の"
            "video latentは暗黙broadcastせず停止する。先行callbackが加えた"
            "noise差分は保持し、ランダム基底だけを座標noiseへ置換する。"
        ),
    ]


def page1_right_story() -> list[Flowable]:
    return [
        subsection("2.3 Anchorと候補棄却"),
        paragraph(
            "低strengthのGlobal Anchor Aを作り、Aから候補indexごとの"
            "全画面Overdraw候補Ciを完成させる。各候補はsourceへ縮小した"
            "SSIM、低周波誤差、色、輪郭位置・方向、新規輪郭、二重線proxy、"
            "clipping、seamを先に検査する。既定strictness 0.70でSSIM"
            "下限は0.696である。全候補不合格ならAnchorへ戻す。"
        ),
        formula("SSIMmin = 0.50 + 0.28 × strictness = 0.696"),
        paragraph(
            "通過候補だけを中周波量、line continuity、material richness、"
            "style consistency、noise penaltyで順位付けする。ただしAnchor"
            "も生成像であり、fallbackは入力画素との同一性を保証しない。"
        ),
        formula("Q = 2.2Dcoherent + 1.4Emid + 0.55L + 0.8M + 0.9S + 0.8O - penalties"),
        paragraph(
            "penaltyはrandom noise、二重線、色drift、seam、structure error"
            "を含む。重要なのはQを計算して最良候補を選ぶ前にhard constraint"
            "を適用する点である。最大Qの不合格候補を救済しない。"
        ),
        subsection("2.4 六帯域の残差合成"),
        paragraph(
            "R=Ci-Aをlinear RGBで輝度と色差に分け、Gaussian blurから"
            "HIGH_0、HIGH_1、MID_HIGH、MID、MID_LOW、LOWへ分解する［7］。"
            "各帯域をMADと99.5百分位によるtanh soft clippingへ通す。"
        ),
        formula("O = A + Σb gb Mb softclip(Rb)"),
        paragraph(
            "Mbはstructure protection、輪郭方向、新規輪郭、tile、"
            "round-trip、ROI、manual maskの積である。中低周波ほど既存"
            "輪郭を強く保護し、LOWはLow Frequency Lockで原則採用しない。"
            "SSIMは入力構造への近さの一指標として用いる［8］。"
        ),
        paragraph(
            "structure mapはσ=1,2,4のSobel勾配、structure tensor、局所"
            "coherence、textureから作る。候補の強線がAnchor線から3 px超"
            "離れる連続成分ではMID/MID_LOWを拒否する一方、平坦部の孤立した"
            "微細textureまで一律に消さない。色差gainは輝度gainの0.35倍である。"
        ),
        subsection("2.5 ROI、seam、低周波BP"),
        paragraph(
            "顔検出は入力座標で一度行い、頭頂、前髪、首、肩、周囲背景を"
            "含むROIへ拡張する。ROI候補にも同じハード制約を適用する。"
            "計画境界の残差勾配比が1.65を超える場合だけ局所整合し、最後に"
            "前段へ縮小した低周波誤差だけをβ=0.70で戻す。誤差増加時は"
            "係数を半減し、再悪化ならrollbackする。"
        ),
        formula("Ok+1 = clip(Ok + β U(Gσ(P - D(Ok))))"),
        paragraph(
            "Manual Protectionは自動構造保護とmaxで統合し、Manual Boostは"
            "採用gainを局所的に増やす。RGBAではlinear RGBをpremultiplyして"
            "拡大し、alphaは入力由来のまま別管理する。モデルに透明度を生成"
            "させず、透明境界へhidden RGBの黒をにじませない。"
        ),
        paragraph(
            "学習済み空間条件を追加するControlNet［6］とは異なり、本手法"
            "は既存生成モデルを再学習せず、推論時の候補評価と帯域選択で"
            "構造逸脱を抑える。広画面を複数拡散経路で扱う点は"
            "MultiDiffusion［5］と関連するが、HyperWeaveは完成img2img"
            "候補の残差を選ぶ。"
        ),
    ]


def results_table(metrics: dict) -> Table:
    lanczos = metrics["Lanczos"]
    hyper = metrics["HyperWeave-full4K"]
    rows = [
        ["評価量", "Lanczos", "HyperWeave"],
        [
            "round-trip SSIM ↑",
            f"{lanczos['roundtrip_ssim']:.4f}",
            f"{hyper['roundtrip_ssim']:.4f}",
        ],
        [
            "PSNR ↑",
            f"{lanczos['roundtrip_psnr']:.2f} dB",
            f"{hyper['roundtrip_psnr']:.2f} dB",
        ],
        [
            "低周波誤差 ↓",
            f"{lanczos['low_frequency_luminance_error']:.2e}",
            f"{hyper['low_frequency_luminance_error']:.2e}",
        ],
        [
            "色 drift ↓",
            f"{lanczos['color_drift']:.2e}",
            f"{hyper['color_drift']:.2e}",
        ],
        [
            "edge displacement ↓",
            f"{lanczos['edge_displacement']:.4f}",
            f"{hyper['edge_displacement']:.4f}",
        ],
        [
            "face structure ↑",
            f"{lanczos['face_structure_score']:.4f}",
            f"{hyper['face_structure_score']:.4f}",
        ],
        [
            "MID energy",
            f"{lanczos['mid_energy']:.2e}",
            f"{hyper['mid_energy']:.2e}",
        ],
        [
            "MID_HIGH energy",
            f"{lanczos['mid_high_energy']:.2e}",
            f"{hyper['mid_high_energy']:.2e}",
        ],
        [
            "HIGH energy",
            f"{lanczos['high_energy']:.2e}",
            f"{hyper['high_energy']:.2e}",
        ],
        ["seam ratio", "-", f"{hyper['seam_ratio']:.4f}"],
    ]
    return compact_table(
        rows,
        [
            COLUMN_WIDTH * 0.46,
            COLUMN_WIDTH * 0.25,
            COLUMN_WIDTH * 0.29,
        ],
        align_right_columns=(1, 2),
    )


def page2_figure_story(comparison_path: Path) -> list[Flowable]:
    return [
        PlatypusImage(
            str(comparison_path),
            width=CONTENT_WIDTH,
            height=CONTENT_WIDTH * 930 / 2200,
        ),
        paragraph(
            "図2　上段は全体像、下段は同一座標の顔クロップ。"
            "HyperWeaveは目、髪内部線、陰影をわずかに再描画する。",
            "caption",
        ),
    ]


def page2_left_story(manifest: dict, metrics: dict) -> list[Flowable]:
    stage = manifest["quality"]["stage_reports"][0]
    runtime = manifest["runtime"]
    memory = manifest["memory"]
    conditions = [
        ["項目", "実測値"],
        ["入力 → 出力", "1664×2353 → 2897×4096"],
        ["内部canvas / 段階", "2904×4096 / 1"],
        ["tile / core / stride", "1280 / 960 / 768"],
        ["タイル / 内部生成", "24 / pass、計49回"],
        ["強度 A / G / Face", "0.12 / 0.24 / 0.22"],
        ["候補 Global / Face", "1 / 1"],
        ["Hair / Material / Micro", "無効 / 無効 / 無効"],
        ["Sampler / Steps", "DPM++ 2M SDE / 1"],
        ["Seed", "976834651"],
    ]
    resources = [
        ["評価量", "実測値"],
        ["HyperWeave処理", f"{manifest['processing_time_seconds']:.2f}秒"],
        [
            "peak allocated",
            f"{runtime['peak_allocated_bytes'] / 1024**3:.2f} GiB",
        ],
        [
            "peak reserved",
            f"{runtime['peak_reserved_bytes'] / 1024**3:.2f} GiB",
        ],
        [
            "CPU RAM見積り",
            f"{memory['working_ram_estimate_bytes'] / 1024**2:.0f} MiB",
        ],
        [
            "accumulator",
            f"{memory['accumulator_bytes'] / 1024**2:.0f} MiB",
        ],
        ["memmap / CUDA OOM", "未使用 / 0回"],
    ]
    return [
        section("3. フル4K事例"),
        paragraph(
            "不透明RGBAイラストをKrea2 int8 checkpoint、Structure Safe、"
            "DPM++ 2M SDE、Simple scheduler、CFG 1.0、Exact Steps 1で"
            "処理した。GPUはNVIDIA GeForce RTX 3090 24 GiBである。左右"
            "人物へManual ROIを与え、重なったcontextを一つの整合した"
            "Face/Head ROIへ統合した。",
            "body_no_indent",
        ),
        compact_table(
            conditions,
            [COLUMN_WIDTH * 0.47, COLUMN_WIDTH * 0.53],
        ),
        Spacer(1, 2.0),
        subsection("3.1 候補選択と資源"),
        paragraph(
            "Anchorはround-trip SSIM 0.8382で受理された。Global候補は"
            "ハード制約で棄却され、selected_global_candidate=nullとなった。"
            "統合Face/Head候補はSSIM 0.7575で受理された。したがって最終差"
            "はAnchor、Face ROI、seam処理、Back Projectionに由来する。"
        ),
        compact_table(
            resources,
            [COLUMN_WIDTH * 0.53, COLUMN_WIDTH * 0.47],
        ),
        Spacer(1, 2.0),
        paragraph(
            "計画境界8本のseam ratioは"
            f"{stage['seam']['ratio']:.3f}で、局所平滑化の閾値1.65を下回る。"
            "Back Projectionは低周波誤差を"
            f"{stage['back_projection']['initial_error']:.6f}から"
            f"{stage['back_projection']['final_error']:.6f}へ減らし、"
            "rollbackしなかった。",
            "compact",
        ),
        subsection("3.2 入力忠実度と帯域変化"),
        results_table(metrics),
    ]


def reference(number: int, text: str) -> Paragraph:
    return Paragraph(f"[{number}] {text}", STYLES["reference"])


def page2_right_story(metrics: dict) -> list[Flowable]:
    lanczos = metrics["Lanczos"]
    hyper = metrics["HyperWeave-full4K"]
    mid_ratio = hyper["mid_energy"] / lanczos["mid_energy"]
    mid_high_ratio = hyper["mid_high_energy"] / lanczos["mid_high_energy"]
    high_ratio = hyper["high_energy"] / lanczos["high_energy"]
    references = [
        reference(
            1,
            "J. Ho et al., “Denoising Diffusion Probabilistic Models,” "
            "<i>NeurIPS</i>, 2020.",
        ),
        reference(
            2,
            "R. Rombach et al., “High-Resolution Image Synthesis with "
            "Latent Diffusion Models,” <i>CVPR</i>, 2022.",
        ),
        reference(
            3,
            "C. Meng et al., “SDEdit: Guided Image Synthesis and Editing "
            "with Stochastic Differential Equations,” <i>ICLR</i>, 2022.",
        ),
        reference(
            4,
            "C. Saharia et al., “Image Super-Resolution via Iterative "
            "Refinement,” arXiv:2104.07636, 2021.",
        ),
        reference(
            5,
            "O. Bar-Tal et al., “MultiDiffusion: Fusing Diffusion Paths for "
            "Controlled Image Generation,” <i>ICML</i>, 2023.",
        ),
        reference(
            6,
            "L. Zhang et al., “Adding Conditional Control to Text-to-Image "
            "Diffusion Models,” <i>ICCV</i>, 2023.",
        ),
        reference(
            7,
            "P. J. Burt and E. H. Adelson, “The Laplacian Pyramid as a "
            "Compact Image Code,” <i>IEEE Trans. Commun.</i>, 1983.",
        ),
        reference(
            8,
            "Z. Wang et al., “Image Quality Assessment: From Error Visibility "
            "to Structural Similarity,” <i>IEEE TIP</i>, 2004.",
        ),
        reference(
            9,
            "X. Wang et al., “Real-ESRGAN: Training Real-World Blind "
            "Super-Resolution with Pure Synthetic Data,” <i>ICCVW</i>, 2021.",
        ),
    ]
    return [
        section("4. 結果と考察"),
        paragraph(
            "Lanczosは入力忠実度の基準であり、未知の正解高解像度像ではない。"
            "HyperWeaveの最終像はLanczosに対し、MID約"
            f"{mid_ratio:.0f}倍、MID_HIGH約{mid_high_ratio:.0f}倍、HIGH約"
            f"{high_ratio:.1f}倍の変化energyを持つ。ただしenergyは正しい"
            "意味的細部の量ではない。"
        ),
        paragraph(
            "round-trip SSIM、PSNR、face structure、hair flow、coherent line"
            "はLanczosが上回った。入力が既に1664×2353で拡大率1.741倍に"
            "とどまる点も補間に有利である。クロップでは顔、視線、髪型"
            "silhouetteは概ね保たれるが、目、髪内部線、陰影は再描画される。"
            "高解像度ground truthがないため、これを真の復元とは判定できない。"
        ),
        paragraph(
            "本事例の明確な成果はGlobal候補の棄却である。全面採用なら残る"
            "構造違反を捨て、Anchorと局所Face ROIに留めた。一方、Anchorの"
            "SSIMは0.838であり、Anchor fallbackは入力へのfallbackではない。"
            "より安全にするにはAnchor不合格時に補間baseへ戻す必要がある。"
        ),
        section("5. 制限と結論"),
        paragraph(
            "本実験は一画像、一モデル、一seed、Steps 1の予備的事例である。"
            "Hair、Material、Micro、複数候補、8Kを評価していない。SDEの二次"
            "noiseは座標固定せず、イラスト顔はManual ROIに依存する。今後は"
            "複数画像・seed・model、候補数とstepsのablation、入力fallback、"
            "人手ペア比較が必要である。"
        ),
        paragraph(
            "HyperWeaveは重複タイル、Coordinate Noise、候補棄却、周波数選択、"
            "ROI、低周波BPをForgeへ統合し、RTX 3090でフル4Kを完走した。"
            "これはLanczosより高品質であることの証明ではなく、単一GPUでの"
            "実行可能性とfail-safe作動の実測である。"
        ),
        section("参考文献"),
        *references,
    ]


class AcademicDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(
            filename,
            pagesize=JIS_B5,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            **kwargs,
        )
        page1_top_height = 78 * mm
        page1_top_y = PAGE_HEIGHT - TOP - page1_top_height
        page1_columns_top = page1_top_y - 3.0 * mm
        page1_column_height = page1_columns_top - BOTTOM
        page2_figure_height = 74 * mm
        page2_figure_y = PAGE_HEIGHT - TOP - page2_figure_height
        page2_columns_top = page2_figure_y - 3.0 * mm
        page2_column_height = page2_columns_top - BOTTOM

        page1_frames = [
            Frame(
                MARGIN_X,
                page1_top_y,
                CONTENT_WIDTH,
                page1_top_height,
                id="page1-top",
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            ),
            Frame(
                MARGIN_X,
                BOTTOM,
                COLUMN_WIDTH,
                page1_column_height,
                id="page1-left",
                leftPadding=0,
                rightPadding=1.2,
                topPadding=0,
                bottomPadding=0,
            ),
            Frame(
                MARGIN_X + COLUMN_WIDTH + COLUMN_GAP,
                BOTTOM,
                COLUMN_WIDTH,
                page1_column_height,
                id="page1-right",
                leftPadding=1.2,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            ),
        ]
        page2_frames = [
            Frame(
                MARGIN_X,
                page2_figure_y,
                CONTENT_WIDTH,
                page2_figure_height,
                id="page2-figure",
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            ),
            Frame(
                MARGIN_X,
                BOTTOM,
                COLUMN_WIDTH,
                page2_column_height,
                id="page2-left",
                leftPadding=0,
                rightPadding=1.2,
                topPadding=0,
                bottomPadding=0,
            ),
            Frame(
                MARGIN_X + COLUMN_WIDTH + COLUMN_GAP,
                BOTTOM,
                COLUMN_WIDTH,
                page2_column_height,
                id="page2-right",
                leftPadding=1.2,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            ),
        ]
        self.addPageTemplates(
            [
                PageTemplate(
                    id="page1",
                    frames=page1_frames,
                    onPage=draw_page,
                ),
                PageTemplate(
                    id="page2",
                    frames=page2_frames,
                    onPage=draw_page,
                ),
            ]
        )


def draw_page(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.35)
    canvas.line(
        MARGIN_X,
        PAGE_HEIGHT - 8.5 * mm,
        PAGE_WIDTH - MARGIN_X,
        PAGE_HEIGHT - 8.5 * mm,
    )
    canvas.setFillColor(MID)
    canvas.setFont(JP_REGULAR, 5.2)
    canvas.drawString(
        MARGIN_X,
        PAGE_HEIGHT - 7.0 * mm,
        "HyperWeave 4K/8K 技術報告",
    )
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_X,
        PAGE_HEIGHT - 7.0 * mm,
        "2026-07-24",
    )
    canvas.setStrokeColor(LIGHT)
    canvas.line(
        MARGIN_X,
        9.2 * mm,
        PAGE_WIDTH - MARGIN_X,
        9.2 * mm,
    )
    canvas.setFont(JP_REGULAR, 5.2)
    canvas.drawCentredString(
        PAGE_WIDTH / 2,
        6.5 * mm,
        f"{document.page}",
    )
    canvas.restoreState()


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
            if "Mincho" in name or "YuMin" in name:
                result[name] = embedded
    return result


def validate_pdf(path: Path) -> dict:
    reader = PdfReader(str(path))
    if len(reader.pages) != 2:
        raise RuntimeError(f"expected 2 pages, got {len(reader.pages)}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    compact_text = re.sub(r"\s+", "", text)
    required = (
        "あいきみ",
        "HyperWeave 4K/8K",
        "Coordinate Noise",
        "ハード制約",
        "0.696",
        "Global候補",
        "selected_global_candidate=null",
        "2897×4096",
        "49回",
        "662",
        "RTX 3090",
        "0.8738",
        "0.9943",
        "Lanczos",
        "fail-safe",
        "予備的事例",
        "参考文献",
    )
    missing = [
        term for term in required if re.sub(r"\s+", "", term) not in compact_text
    ]
    if missing:
        raise RuntimeError(f"required paper terms missing: {missing}")

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
            for character in page.chars:
                used_fonts.add(character["fontname"])
                if (
                    character["x0"] < -0.1
                    or character["x1"] > page.width + 0.1
                    or character["top"] < -0.1
                    or character["bottom"] > page.height + 0.1
                ):
                    out_of_bounds.append(
                        [
                            page_number,
                            character.get("text"),
                            character["x0"],
                            character["x1"],
                            character["top"],
                            character["bottom"],
                        ]
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
        "out_of_bounds_characters": len(out_of_bounds),
        "used_fonts": sorted(used_fonts),
        "embedded_japanese_fonts": embedded,
    }


def build_paper(
    source: Path,
    result: Path,
    comparison_dir: Path,
    output: Path,
    asset_dir: Path,
    regular: Path,
    bold: Path,
) -> dict:
    manifest, metrics = validate_inputs(source, result, comparison_dir)
    register_fonts(regular, bold)

    pipeline_path = asset_dir / "hyperweave_pipeline.png"
    comparison_path = asset_dir / "hyperweave_comparison.png"
    make_pipeline_figure(pipeline_path, regular, bold)
    make_comparison_figure(
        source,
        result,
        comparison_dir,
        comparison_path,
        regular,
        bold,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document = AcademicDocTemplate(
        str(output),
        pageCompression=1,
        title=(
            "HyperWeave 4K/8K：構造制約付き周波数選択によるタイル型生成アップスケール"
        ),
        author="あいきみ",
        subject="HyperWeave 4K/8K preliminary technical paper",
    )
    story: list[Flowable] = []
    story.extend(title_story(pipeline_path))
    story.append(FrameBreak())
    story.extend(page1_left_story())
    story.append(FrameBreak())
    story.extend(page1_right_story())
    story.append(NextPageTemplate("page2"))
    story.append(PageBreak())
    story.extend(page2_figure_story(comparison_path))
    story.append(FrameBreak())
    story.extend(page2_left_story(manifest, metrics))
    story.append(FrameBreak())
    story.extend(page2_right_story(metrics))
    document.build(story)

    report = validate_pdf(output)
    return {
        "output": str(output),
        "pipeline_figure": str(pipeline_path),
        "comparison_figure": str(comparison_path),
        "source_sha256": sha256(source),
        "result_sha256": sha256(result),
        "font": str(regular),
        "bold_font": str(bold),
        **report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--font")
    parser.add_argument("--bold-font")
    args = parser.parse_args()

    regular = find_font(args.font, bold=False)
    bold = find_font(args.bold_font, bold=True)
    report = build_paper(
        args.source,
        args.result,
        args.comparison_dir,
        args.output,
        args.asset_dir,
        regular,
        bold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
