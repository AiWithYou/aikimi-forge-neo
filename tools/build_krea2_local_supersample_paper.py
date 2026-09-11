"""Build the two-page Japanese B5 Krea2 local-supersample case-study paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFESTS = {
    "safe_face": REPO_ROOT
    / "output/krea2_local_supersample_case_study/local_supersample_20260713_200920_824565/experiment_manifest.json",
    "ultra_face": REPO_ROOT
    / "output/krea2_local_supersample_case_study/local_supersample_20260713_201527_322028/experiment_manifest.json",
    "safe_full": REPO_ROOT
    / "output/krea2_local_supersample_case_study/local_supersample_20260713_201930_238319/experiment_manifest.json",
    "ultra_eye_2048": REPO_ROOT
    / "output/krea2_local_supersample_case_study/local_supersample_20260713_204452_893039/experiment_manifest.json",
}

EXACT_PROMPT = (
    "light blue hair,long_wavy_hair,devil’s_horn,purple horn,purple_eyes,"
    "green_slime,jig eyes,smile,jitome,Expressionless,"
)
EXPECTED_PROMPT_SHA256 = "CFC5251C000146F95A841C77B4943BC3011968F89B426BFE275D364D98A5350A"
EXPECTED_INPUT_PIXEL_SHA256 = "241cba297e0549737cc644e5772fa696e64666b7d79fe7c08e6e3c00617132e1"
EXPECTED_FULL_OUTPUT_PIXEL_SHA256 = "5a5a3dc82e4609a853cde49ad6a24b9311d432927fc3d2c6952d6b0eb8b8b983"

JIS_B5 = (182 * mm, 257 * mm)
PAGE_WIDTH, PAGE_HEIGHT = JIS_B5
MARGIN_X = 10 * mm
GUTTER = 5 * mm
BOTTOM = 13 * mm
TOP = 10 * mm
HEADER_HEIGHT = 52 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
COLUMN_WIDTH = (CONTENT_WIDTH - GUTTER) / 2

INK = colors.HexColor("#111318")
MUTED = colors.HexColor("#555C66")
RULE = colors.HexColor("#737B86")
HAIRLINE = colors.HexColor("#C4C9D0")
TABLE_FILL = colors.HexColor("#EEF1F4")
BOX_FILL = colors.HexColor("#F7F8FA")
ACCENT = colors.HexColor("#B5134F")
ACCENT_PALE = colors.HexColor("#F9EAF0")
JP_REGULAR = "Krea2LocalPaperJP-Regular"
JP_BOLD = "Krea2LocalPaperJP-Bold"


def find_japanese_font(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("KREA2_LOCAL_PAPER_JP_FONT"),
        "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Japanese TrueType font not found; pass --font.")


def find_japanese_bold_font(regular: Path, explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("KREA2_LOCAL_PAPER_JP_BOLD_FONT"),
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
        str(regular),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Japanese bold TrueType font not found; pass --bold-font.")


def register_fonts(regular_path: Path, bold_path: Path) -> None:
    regular_kwargs = {"subfontIndex": 0} if regular_path.suffix.lower() == ".ttc" else {}
    bold_kwargs = {"subfontIndex": 0} if bold_path.suffix.lower() == ".ttc" else {}
    pdfmetrics.registerFont(TTFont(JP_REGULAR, str(regular_path), **regular_kwargs))
    pdfmetrics.registerFont(TTFont(JP_BOLD, str(bold_path), **bold_kwargs))
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
    font: str = JP_REGULAR,
    size: float = 6.9,
    leading: float = 9.3,
    alignment=TA_JUSTIFY,
    color=INK,
    before: float = 0,
    after: float = 0,
    first_indent: float = 0,
    keep_with_next: bool = False,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName=font,
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=alignment,
        wordWrap="CJK",
        spaceBefore=before,
        spaceAfter=after,
        firstLineIndent=first_indent,
        allowWidows=0,
        allowOrphans=0,
        keepWithNext=keep_with_next,
    )


STYLES = {
    "title": paragraph_style(
        "title", font=JP_BOLD, size=13.1, leading=15.7, alignment=TA_CENTER, after=1.8
    ),
    "subtitle": paragraph_style(
        "subtitle", size=6.15, leading=7.5, alignment=TA_CENTER, color=MUTED, after=2.0
    ),
    "author": paragraph_style(
        "author", size=6.35, leading=7.8, alignment=TA_CENTER, after=1.6
    ),
    "abstract": paragraph_style("abstract", size=6.75, leading=8.75),
    "keywords": paragraph_style(
        "keywords", size=5.9, leading=7.2, alignment=TA_LEFT, color=MUTED, before=1.5
    ),
    "section": paragraph_style(
        "section",
        font=JP_BOLD,
        size=8.9,
        leading=11.0,
        alignment=TA_LEFT,
        before=3.2,
        after=1.3,
        keep_with_next=True,
    ),
    "subsection": paragraph_style(
        "subsection",
        font=JP_BOLD,
        size=7.6,
        leading=9.4,
        alignment=TA_LEFT,
        before=2.2,
        after=0.8,
        keep_with_next=True,
    ),
    "body": paragraph_style("body", size=7.05, leading=9.55, first_indent=7.05),
    "body0": paragraph_style("body0", size=7.05, leading=9.55),
    "small": paragraph_style("small", size=6.2, leading=8.1, alignment=TA_LEFT),
    "caption": paragraph_style(
        "caption", size=5.55, leading=7.0, alignment=TA_LEFT, color=MUTED, before=1.0, after=1.8
    ),
    "table": paragraph_style("table", size=5.35, leading=6.75, alignment=TA_LEFT),
    "table_center": paragraph_style("table-center", size=5.35, leading=6.75, alignment=TA_CENTER),
    "table_head": paragraph_style(
        "table-head", font=JP_BOLD, size=5.35, leading=6.75, alignment=TA_CENTER
    ),
    "equation": paragraph_style("equation", size=6.25, leading=8.0, alignment=TA_CENTER),
    "algorithm": paragraph_style("algorithm", size=5.55, leading=6.9, alignment=TA_LEFT),
    "reference": paragraph_style("reference", size=5.3, leading=6.75, alignment=TA_LEFT),
    "callout": paragraph_style("callout", size=6.15, leading=8.05, alignment=TA_LEFT),
}


def p(text: str, name: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[name])


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rgb_sha256(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.shape[1]}x{rgb.shape[0]}\0".encode("ascii"))
    digest.update(rgb.tobytes(order="C"))
    return digest.hexdigest()


def flatten_candidate_metrics(run: dict) -> list[dict]:
    metadata = run["embedded_krea2_local_supersample"]
    return [metric for tile in metadata["tiles"] for metric in tile["candidate_metrics"]]


def load_case(paths: dict[str, Path]) -> dict:
    runs = {name: read_json(path) for name, path in paths.items()}
    for name, run in runs.items():
        if run["request"]["base_prompt"] != EXACT_PROMPT:
            raise RuntimeError(f"{name}: manifest does not contain the exact requested prompt")
        if run["request"]["base_prompt_sha256"].upper() != EXPECTED_PROMPT_SHA256:
            raise RuntimeError(f"{name}: prompt hash mismatch")
        if run["input"]["pixel_sha256"] != EXPECTED_INPUT_PIXEL_SHA256:
            raise RuntimeError(f"{name}: input pixel hash mismatch")
        if run["input"]["size"] != [2896, 4096] or run["output"]["size"] != [2896, 4096]:
            raise RuntimeError(f"{name}: expected a 2896x4096 input and output")
        metadata = run["embedded_krea2_local_supersample"]
        if metadata["processed_tile_count"] != metadata["tile_count"]:
            raise RuntimeError(f"{name}: run did not process every selected tile")

    no_op_names = ("safe_face", "ultra_face", "ultra_eye_2048")
    for name in no_op_names:
        run = runs[name]
        if run["output"]["pixel_sha256"] != run["input"]["pixel_sha256"]:
            raise RuntimeError(f"{name}: expected a bit-identical fail-closed result")
        if run["difference_metrics"]["changed_pixel_count"] != 0:
            raise RuntimeError(f"{name}: expected zero changed pixels")

    full = runs["safe_full"]
    metadata = full["embedded_krea2_local_supersample"]
    accepted_tiles = [tile for tile in metadata["tiles"] if tile["selected_candidate"] is not None]
    if metadata["tile_count"] != 117 or len(accepted_tiles) != 12:
        raise RuntimeError("safe_full: expected 12 accepted tiles out of 117")
    if full["output"]["pixel_sha256"] != EXPECTED_FULL_OUTPUT_PIXEL_SHA256:
        raise RuntimeError("safe_full: output pixel hash mismatch")
    if abs(full["difference_metrics"]["changed_pixel_percent"] - 3.72053114748791) > 1e-9:
        raise RuntimeError("safe_full: changed-pixel percentage mismatch")

    input_path = artifact_path(full["input"]["path"])
    output_path = artifact_path(full["output"]["path"])
    if sha256_file(input_path) != full["input"]["file_sha256"]:
        raise RuntimeError("safe_full: input file SHA-256 does not match the manifest")
    if sha256_file(output_path) != full["output"]["file_sha256"]:
        raise RuntimeError("safe_full: output file SHA-256 does not match the manifest")

    input_image = Image.open(input_path).convert("RGB")
    output_image = Image.open(output_path).convert("RGB")
    if rgb_sha256(input_image) != full["input"]["pixel_sha256"]:
        raise RuntimeError("safe_full: decoded input RGB hash mismatch")
    if rgb_sha256(output_image) != full["output"]["pixel_sha256"]:
        raise RuntimeError("safe_full: decoded output RGB hash mismatch")

    metrics = flatten_candidate_metrics(full)
    accepted_metrics = [metric for metric in metrics if metric["accepted"]]
    case = {
        "runs": runs,
        "full": full,
        "input_path": input_path,
        "output_path": output_path,
        "input_image": input_image,
        "output_image": output_image,
        "accepted_tiles": accepted_tiles,
        "candidate_detail_mean": float(np.mean([metric["detail_increase"] for metric in metrics])),
        "candidate_detail_min": float(np.min([metric["detail_increase"] for metric in metrics])),
        "candidate_detail_max": float(np.max([metric["detail_increase"] for metric in metrics])),
        "accepted_detail_mean": float(np.mean([metric["detail_increase"] for metric in accepted_metrics])),
    }
    return case


def pil_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=0)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    return result


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=font, fill=fill)


def make_overview(case: dict, output: Path, regular_font: Path, bold_font: Path) -> Path:
    before = case["input_image"]
    after = case["output_image"]
    canvas = Image.new("RGB", (1600, 860), "white")
    draw = ImageDraw.Draw(canvas)
    label = pil_font(bold_font, 38)
    small = pil_font(regular_font, 25)
    thumb_height = 710
    thumb_width = 600
    panels = ((before, 150, "Input / approved 4K"), (after, 850, "Local residual / Safe 1536"))
    for image, center_x, title in panels:
        thumb = fit_image(image, thumb_width, thumb_height)
        x = int(center_x + 300 - thumb.width / 2)
        y = 85
        canvas.paste(thumb, (x, y))
        draw.rectangle((x - 2, y - 2, x + thumb.width + 1, y + thumb.height + 1), outline=(20, 22, 26), width=2)
        draw_centered(draw, (center_x + 300, 22), title, label, (20, 22, 26))
    draw.line((790, 20, 790, 835), fill=(205, 208, 213), width=2)
    draw_centered(
        draw,
        (800, 810),
        "2896×4096 / 12 of 117 tiles accepted / changed pixels 3.7205% / RGB p99 Δ=1",
        small,
        (78, 84, 94),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return output


def make_tile_map(case: dict, output: Path, regular_font: Path, bold_font: Path) -> Path:
    source = ImageEnhance.Brightness(case["input_image"]).enhance(0.72)
    image_width, image_height = 820, 1160
    thumb = fit_image(source, image_width, image_height)
    canvas = Image.new("RGB", (980, 1390), (246, 247, 249))
    x0 = (canvas.width - thumb.width) // 2
    y0 = 105
    canvas.paste(thumb, (x0, y0))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    sx = thumb.width / case["input_image"].width
    sy = thumb.height / case["input_image"].height
    accepted_indices = {tile["tile_index"] for tile in case["accepted_tiles"]}
    for tile in case["full"]["embedded_krea2_local_supersample"]["tiles"]:
        left, top, right, bottom = tile["core"]
        box = (
            round(x0 + left * sx),
            round(y0 + top * sy),
            round(x0 + right * sx),
            round(y0 + bottom * sy),
        )
        if tile["tile_index"] in accepted_indices:
            draw.rectangle(box, outline=(255, 37, 112, 255), width=5)
        else:
            draw.rectangle(box, outline=(255, 255, 255, 62), width=1)
    face = (1100, 900, 1700, 1500)
    eye = (1344, 1024, 1600, 1280)
    for box, color, width in ((face, (35, 225, 238, 255), 4), (eye, (255, 214, 61, 255), 4)):
        left, top, right, bottom = box
        draw.rectangle(
            (
                round(x0 + left * sx),
                round(y0 + top * sy),
                round(x0 + right * sx),
                round(y0 + bottom * sy),
            ),
            outline=color,
            width=width,
        )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    title = pil_font(bold_font, 40)
    label = pil_font(regular_font, 28)
    draw_centered(draw, (canvas.width // 2, 24), "Tile acceptance map", title, (20, 22, 26))
    draw.rectangle((80, 1300, 115, 1325), fill=(255, 37, 112))
    draw.text((130, 1295), "accepted core: 12", font=label, fill=(32, 35, 41))
    draw.rectangle((410, 1300, 445, 1325), outline=(35, 225, 238), width=4)
    draw.text((460, 1295), "face ROI: no-op", font=label, fill=(32, 35, 41))
    draw.rectangle((705, 1300, 740, 1325), outline=(235, 184, 15), width=4)
    draw.text((755, 1295), "eye", font=label, fill=(32, 35, 41))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return output


def difference_heatmap(before: np.ndarray, after: np.ndarray, amplification: float = 32.0) -> Image.Image:
    delta = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=2).astype(np.float32)
    intensity = np.clip(delta * amplification, 0, 255).astype(np.uint8)
    heat = np.zeros((*intensity.shape, 3), dtype=np.uint8)
    heat[..., 0] = intensity
    heat[..., 1] = (intensity.astype(np.float32) * 0.58).astype(np.uint8)
    heat[..., 2] = (intensity.astype(np.float32) * 0.08).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB")


def make_crop_grid(case: dict, output: Path, regular_font: Path, bold_font: Path) -> Path:
    before_array = np.asarray(case["input_image"], dtype=np.uint8)
    after_array = np.asarray(case["output_image"], dtype=np.uint8)
    crops = [
        ("Face / rejected → exact no-op", (1000, 850, 1800, 1650)),
        ("Accepted left-edge region", (0, 1280, 800, 2080)),
        ("Accepted lower-clothing region", (1728, 3328, 2496, 4096)),
    ]
    canvas = Image.new("RGB", (1500, 1475), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    title = pil_font(bold_font, 34)
    label = pil_font(regular_font, 25)
    column_labels = ("Input 100% crop", "Output 100% crop", "|Δ| ×32 visualization")
    for column, text in enumerate(column_labels):
        draw_centered(draw, (270 + column * 480, 18), text, title, (22, 24, 29))
    for row, (row_name, box) in enumerate(crops):
        left, top, right, bottom = box
        before = before_array[top:bottom, left:right]
        after = after_array[top:bottom, left:right]
        panels = (
            Image.fromarray(before, mode="RGB"),
            Image.fromarray(after, mode="RGB"),
            difference_heatmap(before, after),
        )
        y = 70 + row * 465
        changed = float(np.mean(np.any(before != after, axis=2)) * 100.0)
        draw.text((30, y), f"{row_name}  /  changed {changed:.4f}%", font=label, fill=(65, 70, 79))
        for column, panel in enumerate(panels):
            thumb = panel.resize((420, 420), Image.Resampling.LANCZOS)
            x = 60 + column * 480
            canvas.paste(thumb, (x, y + 35))
            draw.rectangle((x - 1, y + 34, x + 420, y + 455), outline=(45, 48, 54), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return output


def create_assets(case: dict, assets_dir: Path, regular_font: Path, bold_font: Path) -> dict[str, Path]:
    return {
        "overview": make_overview(case, assets_dir / "overview_before_after.png", regular_font, bold_font),
        "tile_map": make_tile_map(case, assets_dir / "tile_acceptance_map.png", regular_font, bold_font),
        "crop_grid": make_crop_grid(case, assets_dir / "crop_comparison.png", regular_font, bold_font),
    }


def paper_table(
    rows: list[list[str]],
    widths: list[float],
    caption: str | None = None,
    *,
    centered_columns: tuple[int, ...] = (),
) -> KeepTogether:
    cells = []
    for row_index, row in enumerate(rows):
        cells.append(
            [
                p(
                    value,
                    "table_head"
                    if row_index == 0
                    else ("table_center" if column in centered_columns else "table"),
                )
                for column, value in enumerate(row)
            ]
        )
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_FILL),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
                ("LINEBELOW", (0, 0), (-1, 0), 0.45, INK),
                ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.25),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.25),
                ("TOPPADDING", (0, 0), (-1, -1), 1.45),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.45),
            ]
        )
    )
    content = [table, Spacer(1, 1.8)]
    if caption:
        content.insert(0, p(caption, "caption"))
    return KeepTogether(content)


def callout(text: str, *, accent: bool = False) -> Table:
    box = Table([[p(text, "callout")]], colWidths=[COLUMN_WIDTH], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT_PALE if accent else BOX_FILL),
                ("BOX", (0, 0), (-1, -1), 0.55, ACCENT if accent else HAIRLINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.0),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
            ]
        )
    )
    return box


def prompt_box() -> Table:
    lines = (
        "light blue hair,long_wavy_hair,devil’s_horn,purple horn,<br/>"
        "purple_eyes,green_slime,jig eyes,smile,jitome,Expressionless,"
    )
    table = Table([[p("<b>Exact base prompt</b><br/>" + lines, "small")]], colWidths=[COLUMN_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BOX_FILL),
                ("BOX", (0, 0), (-1, -1), 0.45, HAIRLINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


class ResidualPipelineFigure(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 30 * mm

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFont(JP_REGULAR, 5.35)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        nodes = [
            ("512 px payload B", 1.5, 21.0),
            ("Lanczos U(B)", 28.0, 21.0),
            ("Krea2 R(U(B))", 54.5, 21.0),
            ("same D", 54.5, 11.8),
            ("C1 − C0", 28.0, 11.8),
            ("band / gate / cap", 1.5, 11.8),
            ("core stitch; reject = Δ0", 18.5, 2.6),
        ]
        box_widths = (24.0, 24.0, 23.0, 23.0, 24.0, 24.0, 41.0)
        for (label, x_mm, y_mm), width_mm in zip(nodes, box_widths):
            x, y, width = x_mm * mm, y_mm * mm, width_mm * mm
            canvas.setFillColor(BOX_FILL)
            canvas.rect(x, y, width, 5.8 * mm, fill=1, stroke=1)
            canvas.setFillColor(INK)
            canvas.drawCentredString(x + width / 2, y + 2.0 * mm, label)
        arrows = [
            ((25.5, 23.9), (28.0, 23.9)),
            ((52.0, 23.9), (54.5, 23.9)),
            ((66.0, 21.0), (66.0, 17.6)),
            ((54.5, 14.7), (52.0, 14.7)),
            ((28.0, 14.7), (25.5, 14.7)),
            ((13.5, 11.8), (27.5, 8.4)),
        ]
        canvas.setStrokeColor(INK)
        for (x1, y1), (x2, y2) in arrows:
            canvas.line(x1 * mm, y1 * mm, x2 * mm, y2 * mm)
        canvas.setFont(JP_REGULAR, 4.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(31.8 * mm, 18.0 * mm, "C0 = D(U(B))")
        canvas.drawString(55.5 * mm, 8.5 * mm, "quality fail → zero")


def algorithm_block() -> KeepTogether:
    lines = [
        "<b>Algorithm 1</b>　Fail-closed local residual",
        "1: halo付きpayload Bと中央coreを計画",
        "2: U ← Lanczos(B, process edge)",
        "3: 候補ごとに R ← Krea2-img2img(U, seed)",
        "4: C0,C1 ← same linear-light area downsample",
        "5: Δ ← bandpass(C1−C0) × structure/edge gate",
        "6: luma/chromaを制限し、detail/drift/clip/boundaryを検査",
        "7: 2候補時は代表候補×agreement mask（平均しない）",
        "8: 不合格はΔ=0；合格だけ正規化重みでcoreへ合成",
    ]
    rows = [[p(line, "algorithm")] for line in lines]
    table = Table(rows, colWidths=[COLUMN_WIDTH], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_FILL),
                ("LINEABOVE", (0, 0), (-1, 0), 0.65, INK),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, INK),
                ("LINEBELOW", (0, -1), (-1, -1), 0.65, INK),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.0),
                ("TOPPADDING", (0, 0), (-1, -1), 0.65),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0.65),
            ]
        )
    )
    return KeepTogether([table, Spacer(1, 1.5)])


def scaled_image(path: Path, width: float) -> RLImage:
    with Image.open(path) as image:
        ratio = image.height / image.width
    return RLImage(str(path), width=width, height=width * ratio)


def result_visual(case: dict, assets: dict[str, Path]) -> Table:
    tile_map = scaled_image(assets["tile_map"], 31 * mm)
    full = case["full"]
    diff = full["difference_metrics"]
    telemetry = full["telemetry"]
    text = p(
        "<b>全画面Safe 1536</b><br/>"
        "採用 <b>12 / 117</b> tile<br/>"
        f"変更 {diff['changed_pixel_percent']:.4f}%<br/>"
        f"RGB |Δ| p99 / max: {diff['p99_abs_rgb_code_delta']:.0f} / {diff['max_abs_rgb_code_delta']:.0f}<br/>"
        f"合成clip: {case['full']['embedded_krea2_local_supersample']['clipping_fraction']:.0f}<br/>"
        f"peak VRAM: {telemetry['peak_memory_used_mib']:,} MiB<br/>"
        f"elapsed: {full['duration_seconds']:.1f} s<br/><br/>"
        "<font color='#B5134F'>赤</font>: 採用core<br/>"
        "水色: 顔ROI（no-op）<br/>黄: 目ROI（no-op）",
        "small",
    )
    table = Table([[tile_map, text]], colWidths=[34 * mm, COLUMN_WIDTH - 34 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 3.0),
                ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def draw_page_chrome(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.35)
    canvas.line(MARGIN_X, 9.2 * mm, PAGE_WIDTH - MARGIN_X, 9.2 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 5.0)
    canvas.drawString(MARGIN_X, 5.8 * mm, "Krea2 Local Supersample Detail / Single-Image Technical Short Paper")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 5.8 * mm, f"{doc.page} / 2")
    if doc.page > 1:
        canvas.setFont("Helvetica", 4.7)
        canvas.drawString(MARGIN_X, PAGE_HEIGHT - 6.4 * mm, "FAIL-CLOSED LOCAL RESIDUAL REFINEMENT ON APPROVED 4K")
        canvas.line(MARGIN_X, PAGE_HEIGHT - 8.0 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 8.0 * mm)
    canvas.restoreState()


def page_templates() -> list[PageTemplate]:
    header_bottom = PAGE_HEIGHT - TOP - HEADER_HEIGHT
    first_column_top = header_bottom - 3 * mm
    first_column_height = first_column_top - BOTTOM
    header = Frame(
        MARGIN_X,
        header_bottom,
        CONTENT_WIDTH,
        HEADER_HEIGHT,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first-header",
    )
    first_left = Frame(
        MARGIN_X,
        BOTTOM,
        COLUMN_WIDTH,
        first_column_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first-left",
    )
    first_right = Frame(
        MARGIN_X + COLUMN_WIDTH + GUTTER,
        BOTTOM,
        COLUMN_WIDTH,
        first_column_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first-right",
    )
    later_height = PAGE_HEIGHT - 11 * mm - BOTTOM
    later_left = Frame(
        MARGIN_X,
        BOTTOM,
        COLUMN_WIDTH,
        later_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="later-left",
    )
    later_right = Frame(
        MARGIN_X + COLUMN_WIDTH + GUTTER,
        BOTTOM,
        COLUMN_WIDTH,
        later_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="later-right",
    )
    return [
        PageTemplate(
            id="first",
            frames=[header, first_left, first_right],
            onPage=draw_page_chrome,
            autoNextPageTemplate="later",
        ),
        PageTemplate(id="later", frames=[later_left, later_right], onPage=draw_page_chrome),
    ]


def run_summary(run: dict) -> tuple[int, int, int]:
    metadata = run["embedded_krea2_local_supersample"]
    accepted = metadata["tile_count"] - metadata["rejected_noop_tile_count"]
    candidate_total = sum(len(tile["candidate_metrics"]) for tile in metadata["tiles"])
    return accepted, metadata["tile_count"], candidate_total


def build_story(case: dict, assets: dict[str, Path]) -> list:
    runs = case["runs"]
    full = case["full"]
    diff = full["difference_metrics"]
    full_meta = full["embedded_krea2_local_supersample"]

    title_block = [
        p("承認済み4K画像に対する局所超解像残差のFail-Closed評価", "title"),
        p("Fail-Closed Evaluation of Local Supersample Residual Refinement on an Approved 4K Krea2 Image", "subtitle"),
        p("AiWithYou　—　Technical Short Paper / 2026-07-13", "author"),
        HRFlowable(width="100%", thickness=0.45, color=RULE, spaceBefore=0.5, spaceAfter=1.8),
        p(
            "<b>要旨—</b> 生成済み画像の構図・顔・色を保持し、局所的な微細描写だけを加えるKrea2 Local Supersample Detailを、"
            "指定promptの承認済み2896×4096画像で評価した。512 px payloadを1536/2048へ拡大するが候補画像は貼らず、"
            "同一linear-light縮小経路のC1−C0残差だけを品質gate通過時に合成する。顔・目の3条件は全候補が不採用で入力と"
            "画素hashまで一致した。全画面Safe 1536は117 tile中12 tileを採用し、変更3.7205%、RGB差p99=1、max=8、"
            "合成clip=0、peak VRAM 22,478 MiB、1,387.2 sであった。本単一事例は普遍的な画質向上ではなく、悪化が疑われる"
            "候補をno-opへ戻すfail-closed特性を示す。",
            "abstract",
        ),
        p("Keywords: Krea2, local supersampling, residual refinement, fail-closed quality gate, 4K, RTX 3090", "keywords"),
        FrameBreak(),
    ]

    page1_left = [
        p("1. はじめに", "section"),
        p(
            "拡散モデルによる高解像度仕上げでは、長辺の一括処理はVRAM制約を受け、分割処理は顔同一性、低周波の色・明度、"
            "偽テクスチャ、継ぎ目を変え得る。Latent Diffusion [2] とMultiDiffusion [3] は大画像処理の基盤を与えるが、"
            "本研究の対象は再生成ではなく、承認後4Kへ保守的な局所残差だけを加える工程である。",
        ),
        p(
            "問いは「常に細部が増えるか」ではない。候補が元画像より細部を失う、または構図成分を変える場合、元画像を壊さず"
            "何もしない結果へ戻れるかを検証する。",
        ),
        p("2. 手法", "section"),
        ResidualPipelineFigure(COLUMN_WIDTH),
        p(
            "入力payloadをB、Lanczos拡大をU、Krea2 refineをR、linear-light area縮小をDとする。",
            "body0",
        ),
        p("C0 = D(U(B))　　C1 = D(R(U(B)))　　Δraw = C1 − C0", "equation"),
        p(
            "同じDを通した差により拡大縮小の往復誤差を相殺する。ΔrawからGaussian radius 12 pxのluma低周波を除き、"
            "構造・強エッジgateを掛ける。Safeはluma/chromaを8/2 code、Ultraは12/3 codeへ制限する。",
        ),
        algorithm_block(),
        p("2.1 Fail-closed gate", "subsection"),
        paper_table(
            [
                ["観測", "不採用条件 / 動作"],
                ["detail", "C1の局所energy ≤ C0 → Δ=0"],
                ["drift / clip", "低周波超過、RGB clip>1% → Δ=0"],
                ["boundary", "payload端の残差過大 → Δ=0"],
                ["2候補", "agreement coverage<5% → Δ=0"],
            ],
            [20 * mm, COLUMN_WIDTH - 20 * mm],
            centered_columns=(0,),
        ),
        p(
            "2候補は平均せず、品質scoreが良い一方を代表とし、もう一方は同位置・同符号・局所相関の支持証拠にだけ使う。"
            "agreement coverageが5%未満なら不採用である。",
        ),
        FrameBreak(),
    ]

    profile_rows = [
        ["ID", "範囲 / profile", "tile", "候補", "step / d"],
        ["A", "顔 / Safe 1536", "9", "9", "4 / .10"],
        ["B", "顔 / Ultra 1536", "9", "18", "5 / .15"],
        ["C", "目 / Ultra 2048", "1", "2", "5 / .14"],
        ["D", "全画面 / Safe 1536", "117", "117", "4 / .10"],
    ]
    page1_right = [
        p("3. 実験条件", "section"),
        p(
            "既存4K preflightを通過したRGB PNG（2896×4096）、seed 3883506083を入力とした。ユーザー指定文字列は"
            "綴り・大小文字・末尾commaを含め変更していない。prompt SHA-256はCFC5251C0001…である。",
        ),
        prompt_box(),
        Spacer(1, 2.0),
        p(
            "checkpointはturbo_gpt0630_krea2_final_forge_bnb_nf4、Qwen Image VAE、Qwen3-VL 4B bf16、GPUはRTX 3090 24 GB。"
            "VRAM等はnvidia-smiで1秒間隔に記録したGPU全体値である。",
        ),
        paper_table(
            profile_rows,
            [6 * mm, 30 * mm, 11 * mm, 11 * mm, 18.5 * mm],
            "表1　評価条件。顔ROI=(1100,900,1700,1500)、目ROI=(1344,1024,1600,1280)。",
            centered_columns=(0, 2, 3, 4),
        ),
        scaled_image(assets["overview"], COLUMN_WIDTH),
        p(
            "図1　入力と全画面Safe 1536出力の全体像。差は原寸でも微小であり、図2・3で採用位置と増幅差分を示す。",
            "caption",
        ),
        p("3.1 評価量", "subsection"),
        p(
            "tile採否、decoded RGB hash、変更画素率、RGB code差、low/high-frequency luma差、合成clipping、処理時間、peak VRAMを"
            "manifestへ保存した。no-opは見た目だけでなくpixel hash一致を必須とした。",
        ),
        p(
            "API応答PNGにはtile別の候補seed、quality統計、採否理由を埋め込み、実行器側manifestには入力・出力file hash、"
            "decoded pixel hash、GPU時系列を保存した。2048失敗時のsilent fallbackは設けていない。",
        ),
        callout(
            "<b>検証仮説:</b> 既に高密度な顔でKrea2候補のdetail energyが増えなければ、品質gateは変更を拒否し、"
            "元画像へbit-identicalに戻る。",
            accent=True,
        ),
        PageBreak(),
    ]

    result_rows = [
        ["ID", "採用", "変更", "時間", "peak VRAM"],
        ["A", "0 / 9", "0%", "310.1 s*", "21,479 MiB"],
        ["B", "0 / 9", "0%", "197.4 s", "21,617 MiB"],
        ["C", "0 / 1", "0%", "49.4 s", "22,600 MiB"],
        ["D", "12 / 117", "3.7205%", "1,387.2 s", "22,478 MiB"],
    ]
    page2_left = [
        p("4. 結果", "section"),
        paper_table(
            result_rows,
            [7 * mm, 14 * mm, 16 * mm, 18 * mm, 23.5 * mm],
            "表2　実測結果。*Aは初回model loadを含み、時間の単純比較対象外。",
            centered_columns=(0, 1, 2, 3, 4),
        ),
        p(
            "A–Cは全候補がdetail-energy gateを含む品質理由で拒否され、変更0画素、入力と同じpixel hash"
            "（241cba297e05…）になった。Cの2048処理自体はOOMなく完走したが、有用な残差は得られずpeak 22,600 MiBであった。",
        ),
        result_visual(case, assets),
        p(
            "図2　Dのtile採否。採用は上端・左端・下部衣装の一部へ偏り、顔中心では採用されなかった。",
            "caption",
        ),
        scaled_image(assets["crop_grid"], COLUMN_WIDTH),
        p(
            "図3　100% crop（紙面では縮小）と絶対差×32。顔cropは完全一致。採用域でも差は疎で、連続した境界帯を認めない。",
            "caption",
        ),
        FrameBreak(),
    ]

    page2_right = [
        p("4.1 全画面条件の数値", "subsection"),
        p(
            f"Dは{diff['changed_pixel_count']:,}画素（{diff['changed_pixel_percent']:.4f}%）を変更した。全RGB channelの平均絶対code差"
            f"{diff['mean_abs_rgb_code_delta']:.5f}、p95={diff['p95_abs_rgb_code_delta']:.0f}、p99={diff['p99_abs_rgb_code_delta']:.0f}、"
            f"最大{diff['max_abs_rgb_code_delta']:.0f}である。合成metadataのclipping fractionは{full_meta['clipping_fraction']:.0f}。"
            f"候補detail増分は平均{case['candidate_detail_mean']:+.4f} code、採用候補だけでは平均{case['accepted_detail_mean']:+.4f} codeであった。",
        ),
        paper_table(
            [
                ["ID", "候補", "detail Δ mean", "min … max"],
                ["A", "9", "−0.173", "−0.356 … −0.066"],
                ["B", "18", "−0.147", "−0.287 … −0.044"],
                ["C", "2", "−0.127", "−0.132 … −0.122"],
                ["D", "117", "−0.173", "−0.747 … +0.143"],
            ],
            [7 * mm, 12 * mm, 25 * mm, COLUMN_WIDTH - 44 * mm],
            "表3　C1−C0の局所detail-energy増分（RGB code相当）。",
            centered_columns=(0, 1, 2, 3),
        ),
        callout(
            "<b>主結果:</b> 顔・目は「改善なし」を検知してbit-identical no-op。全画面でも117 tile中12 tile、"
            "RGB差p99=1に留まった。これは改善率ではなく、変更を狭く抑えた安全性の結果である。",
            accent=True,
        ),
        p("5. 考察", "section"),
        p(
            "本事例の成果は「顔を改善した」ことではなく、改善の数値根拠がない顔・目候補を採用しなかった点にある。候補全体の"
            "detail増分が負であることから、既に密な4K入力に対する低denoise img2imgは局所帯域をわずかに平滑化したと解釈できる。"
            "正のdetail-energyとdrift/clip/boundary/agreementを別々に検査する設計は、その状況で安全側を選んだ。",
        ),
        p(
            "全画面は23分超を要した一方、変更は3.72%かつp99=1で顔には反映されなかった。本画像ではSafe 1536 ROIを先行し、"
            "採用tileと100% cropを確認する方が費用対効果に優れる。2048が1536より良いという証拠は得られない。",
        ),
        p("5.1 実務上の判断", "subsection"),
        callout(
            "<b>推奨順序</b><br/>1) Safe 1536を必要ROIだけに実行。<br/>"
            "2) no-opなら同一条件の反復ではなく、元4Kを採用。<br/>"
            "3) Ultraは2候補agreementと原寸QAを確認。<br/>"
            "4) 2048は24 GBで余裕が小さいため、単一ROIで実利を確認。",
        ),
        p("6. 限界", "section"),
        p(
            "画像1枚・prompt 1件・seed 1件のcase studyであり、ground truth、他方式との盲検比較、複数評価者の主観評価を含まない。"
            "detail-energyは知覚品質や意味的正しさを保証しない。telemetryはGPU全体値で、閾値の一般性、別画風・顔サイズ・素材・GPUでの"
            "再現性も未検証である。",
        ),
        p("7. 結論", "section"),
        p(
            "指定promptの承認済み4Kへ局所超解像残差を適用し、顔・目の候補を安全にno-opとし、全画面でも微小変更だけを採用した。"
            "一般的な高精細化効果の証明ではないが、悪化が疑われる候補を元画像へ戻すfail-closed実装の実証である。実運用はSafe 1536 ROI、"
            "100% crop確認、必要箇所だけの採用を基本とする。",
        ),
        p("再現性", "subsection"),
        p(
            "入力file: B303B89DEEB1…　出力file: 4C2BDBE5F8A4…<br/>"
            "入力pixel: 241cba297e05…　出力pixel: 5a5a3dc82e46…<br/>"
            "runner: tools/run_krea2_local_supersample_experiment.py",
            "reference",
        ),
        p("参考文献", "section"),
        p(
            "[1] S. Lee et al., “Krea 2 Technical Report,” Krea, 2026.<br/>"
            "[2] R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022.<br/>"
            "[3] O. Bar-Tal et al., “MultiDiffusion,” ICML, 2023.",
            "reference",
        ),
    ]
    return title_block + page1_left + page1_right + page2_left + page2_right


def build_pdf(
    output: Path,
    repo_copy: Path | None,
    regular_font: Path,
    bold_font: Path,
    case: dict,
    assets: dict[str, Path],
) -> None:
    register_fonts(regular_font, bold_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(output),
        pagesize=JIS_B5,
        leftMargin=0,
        rightMargin=0,
        topMargin=0,
        bottomMargin=0,
        pageTemplates=page_templates(),
        title="承認済み4K画像に対する局所超解像残差のFail-Closed評価",
        author="AiWithYou",
        subject="Krea2 Local Supersample Detailの単一画像実機評価",
    )
    doc.build(build_story(case, assets))
    reader = PdfReader(str(output))
    if len(reader.pages) != 2:
        raise RuntimeError(f"Expected exactly 2 JIS B5 pages, generated {len(reader.pages)}")
    expected = tuple(value / mm for value in JIS_B5)
    for index, page in enumerate(reader.pages, start=1):
        actual = (float(page.mediabox.width) / mm, float(page.mediabox.height) / mm)
        if any(abs(got - want) > 0.05 for got, want in zip(actual, expected)):
            raise RuntimeError(
                f"Page {index}: {actual[0]:.2f}x{actual[1]:.2f} mm; expected {expected[0]:.2f}x{expected[1]:.2f} mm"
            )
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = (
        "Fail-Closed",
        "light blue hair,long_wavy_hair",
        "12 / 117",
        "3.7205%",
        "22,478 MiB",
        "241cba297e05",
        "5a5a3dc82e46",
    )
    missing = [value for value in required if value not in extracted]
    if missing:
        raise RuntimeError("PDF text extraction is missing required evidence: " + ", ".join(missing))
    if repo_copy is not None:
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, repo_copy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-face-manifest", type=Path, default=DEFAULT_MANIFESTS["safe_face"])
    parser.add_argument("--ultra-face-manifest", type=Path, default=DEFAULT_MANIFESTS["ultra_face"])
    parser.add_argument("--safe-full-manifest", type=Path, default=DEFAULT_MANIFESTS["safe_full"])
    parser.add_argument("--ultra-eye-2048-manifest", type=Path, default=DEFAULT_MANIFESTS["ultra_eye_2048"])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output/pdf/krea2_local_supersample_b5_ja.pdf")
    parser.add_argument("--repo-copy", type=Path, default=REPO_ROOT / "docs/krea2_local_supersample_b5_ja.pdf")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=REPO_ROOT / "output/krea2_local_supersample_case_study/paper_assets",
    )
    parser.add_argument("--font", default=None)
    parser.add_argument("--bold-font", default=None)
    args = parser.parse_args()

    paths = {
        "safe_face": args.safe_face_manifest,
        "ultra_face": args.ultra_face_manifest,
        "safe_full": args.safe_full_manifest,
        "ultra_eye_2048": args.ultra_eye_2048_manifest,
    }
    regular_font = find_japanese_font(args.font)
    bold_font = find_japanese_bold_font(regular_font, args.bold_font)
    case = load_case(paths)
    assets = create_assets(case, args.assets_dir, regular_font, bold_font)
    build_pdf(args.output, args.repo_copy, regular_font, bold_font, case, assets)
    sys.stdout.write(f"PDF: {args.output}\n")
    sys.stdout.write(f"Repository copy: {args.repo_copy}\n")
    for name, path in assets.items():
        sys.stdout.write(f"Asset {name}: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
