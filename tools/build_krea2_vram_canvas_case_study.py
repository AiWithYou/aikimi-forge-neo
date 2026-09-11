"""Build the illustrated Japanese Krea2/VRAM-Canvas 8K case-study PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_ROOT = ROOT / "output" / "vram_canvas_case_study"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "vram_canvas_krea2_case_study_ja.pdf"
DEFAULT_ASSETS = ROOT / "output" / "pdf" / "vram_canvas_krea2_case_study_assets"

JP_REGULAR = "CaseStudyJP-Regular"
JP_BOLD = "CaseStudyJP-Bold"
INK = colors.HexColor("#182033")
MUTED = colors.HexColor("#59627A")
ACCENT = colors.HexColor("#6C4FD3")
ACCENT_2 = colors.HexColor("#239A72")
PALE = colors.HexColor("#F2F0FC")
PALE_GREEN = colors.HexColor("#EAF7F1")
RULE = colors.HexColor("#D8DCE8")
TABLE_HEAD = colors.HexColor("#E7E3F8")
WHITE = colors.white


@dataclass(frozen=True)
class CasePaths:
    root: Path
    native_dir: Path
    run_4k: Path
    run_8k: Path
    native_report: Path
    manifest_4k: Path
    manifest_8k: Path
    quality_4k: Path
    quality_8k: Path
    record: Path
    final_4k: Path
    final_8k: Path


def paths_for(case_root: Path) -> CasePaths:
    native_dir = case_root / "krea2_a_series_20260712_2044"
    run_4k = case_root / "runs" / "vram_canvas_20260712_210520_185513"
    run_8k = case_root / "runs" / "vram_canvas_20260712_212322_712827"
    return CasePaths(
        root=case_root,
        native_dir=native_dir,
        run_4k=run_4k,
        run_8k=run_8k,
        native_report=native_dir / "txt2img_report.json",
        manifest_4k=run_4k / "run_manifest.json",
        manifest_8k=run_8k / "run_manifest.json",
        quality_4k=run_4k / "vram_canvas_highres_4k_final.quality.json",
        quality_8k=run_8k / "vram_canvas_highres_8k_final.quality.json",
        record=case_root / "case_study_record.json",
        final_4k=run_4k / "krea2_vram_canvas_4k_final.png",
        final_8k=run_8k / "krea2_vram_canvas_8k_final.png",
    )


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def find_font(explicit: str | None, *, bold: bool) -> Path:
    environment_key = "VRAM_CANVAS_JP_BOLD_FONT" if bold else "VRAM_CANVAS_JP_FONT"
    candidates = [
        explicit,
        os.environ.get(environment_key),
        "C:/Windows/Fonts/BIZ-UDGothicB.ttc"
        if bold
        else "C:/Windows/Fonts/BIZ-UDGothicR.ttc",
        "C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothR.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    role = "bold" if bold else "regular"
    raise FileNotFoundError(
        f"Japanese {role} font was not found; pass an explicit font path."
    )


def register_fonts(regular: Path, bold: Path) -> None:
    pdfmetrics.registerFont(TTFont(JP_REGULAR, str(regular), subfontIndex=0))
    pdfmetrics.registerFont(TTFont(JP_BOLD, str(bold), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        JP_REGULAR,
        normal=JP_REGULAR,
        bold=JP_BOLD,
        italic=JP_REGULAR,
        boldItalic=JP_BOLD,
    )


def validate_case(paths: CasePaths, data: dict) -> None:
    expected = {
        "native": (1024, 1448),
        "4k": (2896, 4096),
        "8k": (5792, 8192),
    }
    candidate_path = paths.native_dir / "candidate_03_seed_20260714.png"
    for label, path, size in (
        ("native", candidate_path, expected["native"]),
        ("4k", paths.final_4k, expected["4k"]),
        ("8k", paths.final_8k, expected["8k"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        with PILImage.open(path) as image:
            if image.size != size:
                raise RuntimeError(f"{label} size is {image.size}, expected {size}")
            if label in {"4k", "8k"}:
                embedded = json.loads(str(image.info.get("vram_canvas", "{}")))
                if embedded.get("prompt") != data["record"]["highres_prompt"]:
                    raise RuntimeError(
                        f"{label} effective prompt metadata is missing or stale"
                    )

    native = data["native"]
    if native["candidates"][2]["seed"] != 20260714:
        raise RuntimeError("Selected native candidate seed is inconsistent")
    if data["m4"]["target_size"] != [2896, 4096]:
        raise RuntimeError("4K manifest target is inconsistent")
    if data["m8"]["target_size"] != [5792, 8192]:
        raise RuntimeError("8K manifest target is inconsistent")
    for manifest in (data["m4"], data["m8"]):
        for stage in manifest["stage_reports"]:
            if stage["processed_tile_count"] != stage["tile_count"]:
                raise RuntimeError("Not all planned tiles were processed")


def font_for_asset(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=0)


def panel(
    image: PILImage.Image,
    size: tuple[int, int],
    label: str,
    font: ImageFont.FreeTypeFont,
    *,
    selected: bool = False,
    resample: PILImage.Resampling = PILImage.Resampling.LANCZOS,
) -> PILImage.Image:
    width, height = size
    bar = 64
    canvas = PILImage.new("RGB", (width, height + bar), "#F7F8FC")
    fitted = ImageOps.fit(image.convert("RGB"), (width, height), method=resample)
    canvas.paste(fitted, (0, 0))
    draw = ImageDraw.Draw(canvas)
    if selected:
        draw.rectangle((3, 3, width - 4, height - 4), outline="#805EE8", width=10)
    draw.rectangle((0, height, width, height + bar), fill="#182033")
    draw.text(
        (width // 2, height + bar // 2), label, font=font, fill="white", anchor="mm"
    )
    return canvas


def save_jpeg(image: PILImage.Image, path: Path, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        path, format="JPEG", quality=quality, subsampling=0, optimize=True
    )


def generate_assets(
    paths: CasePaths, assets_dir: Path, asset_font: Path
) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    label_font = font_for_asset(asset_font, 34)
    small_font = font_for_asset(asset_font, 28)

    with PILImage.open(paths.final_8k) as opened:
        hero = opened.convert("RGB")
        hero.thumbnail((1600, 2264), PILImage.Resampling.LANCZOS)
    hero_path = assets_dir / "hero_8k.jpg"
    save_jpeg(hero, hero_path, quality=94)

    candidate_panels = []
    for index in range(1, 5):
        seed = 20260711 + index
        path = paths.native_dir / f"candidate_{index:02d}_seed_{seed}.png"
        with PILImage.open(path) as opened:
            candidate_panels.append(
                panel(
                    opened,
                    (500, 707),
                    f"Candidate {index} / seed {seed}",
                    small_font,
                    selected=index == 3,
                )
            )
    gap = 24
    candidate_grid = PILImage.new(
        "RGB",
        (4 * 500 + 3 * gap, candidate_panels[0].height),
        "white",
    )
    for index, item in enumerate(candidate_panels):
        candidate_grid.paste(item, (index * (500 + gap), 0))
    candidate_path = assets_dir / "native_candidates.jpg"
    save_jpeg(candidate_grid, candidate_path)

    pipeline_specs = [
        (
            paths.native_dir / "candidate_03_seed_20260714.png",
            "Native 1024 x 1448",
        ),
        (paths.final_4k, "4K 2896 x 4096 / 62 tiles"),
        (paths.final_8k, "8K 5792 x 8192 / 140 tiles"),
    ]
    pipeline_panels = []
    for source, label in pipeline_specs:
        with PILImage.open(source) as opened:
            pipeline_panels.append(panel(opened, (620, 877), label, label_font))
    pipeline = PILImage.new(
        "RGB", (3 * 620 + 2 * 40, pipeline_panels[0].height), "white"
    )
    pipeline_draw = ImageDraw.Draw(pipeline)
    for index, item in enumerate(pipeline_panels):
        x = index * (620 + 40)
        pipeline.paste(item, (x, 0))
        if index < 2:
            pipeline_draw.polygon(
                [(x + 630, 420), (x + 655, 445), (x + 630, 470)],
                fill="#6C4FD3",
            )
    pipeline_path = assets_dir / "resolution_pipeline.jpg"
    save_jpeg(pipeline, pipeline_path)

    detail_regions = {
        "Face and horns": (0.22, 0.02, 0.78, 0.34),
        "Slime and fabric": (0.10, 0.46, 0.90, 0.88),
    }
    with PILImage.open(paths.native_dir / "candidate_03_seed_20260714.png") as opened:
        native_image = opened.convert("RGB")
    with PILImage.open(paths.final_8k) as opened:
        final_image = opened.convert("RGB")
    detail_rows = []
    for name, normalized in detail_regions.items():
        row = []
        for image, source_label in (
            (native_image, "Native + Lanczos display"),
            (final_image, "VRAM-Canvas 8K final"),
        ):
            x0, y0, x1, y1 = normalized
            crop = image.crop(
                (
                    round(image.width * x0),
                    round(image.height * y0),
                    round(image.width * x1),
                    round(image.height * y1),
                )
            )
            row.append(panel(crop, (980, 520), f"{name} / {source_label}", small_font))
        detail_rows.append(row)
    detail = PILImage.new("RGB", (1984, 2 * 584 + 24), "white")
    for row_index, row in enumerate(detail_rows):
        for column_index, item in enumerate(row):
            detail.paste(item, (column_index * 1004, row_index * 608))
    detail_path = assets_dir / "detail_comparison.jpg"
    save_jpeg(detail, detail_path, quality=94)

    qa_regions = {
        "Face / horns": (0.22, 0.00, 0.78, 0.32),
        "Torso / sleeves": (0.12, 0.20, 0.88, 0.64),
        "Green slime": (0.03, 0.48, 0.97, 0.94),
        "Feet / floor": (0.20, 0.75, 0.80, 1.00),
        "Background L": (0.00, 0.12, 0.28, 0.78),
        "Background R": (0.72, 0.12, 1.00, 0.78),
    }
    qa_panels = []
    for name, normalized in qa_regions.items():
        x0, y0, x1, y1 = normalized
        crop = final_image.crop(
            (
                round(final_image.width * x0),
                round(final_image.height * y0),
                round(final_image.width * x1),
                round(final_image.height * y1),
            )
        )
        qa_panels.append(panel(crop, (650, 430), name, small_font))
    qa_grid = PILImage.new("RGB", (3 * 650 + 2 * 20, 2 * 494 + 20), "white")
    for index, item in enumerate(qa_panels):
        qa_grid.paste(item, ((index % 3) * 670, (index // 3) * 514))
    qa_path = assets_dir / "qa_regions.jpg"
    save_jpeg(qa_grid, qa_path, quality=94)

    return {
        "hero": hero_path,
        "candidates": candidate_path,
        "pipeline": pipeline_path,
        "detail": detail_path,
        "qa": qa_path,
    }


def paragraph_style(
    name: str,
    *,
    font: str = JP_REGULAR,
    size: float = 8.2,
    leading: float = 12.0,
    alignment: int = TA_JUSTIFY,
    color=INK,
    before: float = 0,
    after: float = 4,
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
        keepWithNext=keep_with_next,
        allowWidows=0,
        allowOrphans=0,
    )


def make_styles() -> dict[str, ParagraphStyle]:
    return {
        "title": paragraph_style(
            "title", font=JP_BOLD, size=18.5, leading=23, alignment=TA_CENTER, after=4
        ),
        "subtitle": paragraph_style(
            "subtitle", size=8.2, leading=11, alignment=TA_CENTER, color=MUTED, after=5
        ),
        "meta": paragraph_style(
            "meta", size=7.1, leading=9.5, alignment=TA_CENTER, color=MUTED, after=6
        ),
        "h1": paragraph_style(
            "h1",
            font=JP_BOLD,
            size=12.5,
            leading=16,
            alignment=TA_LEFT,
            color=ACCENT,
            before=2,
            after=6,
            keep_with_next=True,
        ),
        "h2": paragraph_style(
            "h2",
            font=JP_BOLD,
            size=9.5,
            leading=12.5,
            alignment=TA_LEFT,
            before=5,
            after=3,
            keep_with_next=True,
        ),
        "body": paragraph_style("body"),
        "body0": paragraph_style("body0", alignment=TA_LEFT),
        "small": paragraph_style(
            "small", size=7.1, leading=10.1, alignment=TA_LEFT, color=MUTED, after=3
        ),
        "caption": paragraph_style(
            "caption", size=6.8, leading=9.2, alignment=TA_LEFT, color=MUTED, after=5
        ),
        "table": paragraph_style(
            "table", size=6.8, leading=9.2, alignment=TA_LEFT, after=0
        ),
        "table_center": paragraph_style(
            "table-center", size=6.8, leading=9.2, alignment=TA_CENTER, after=0
        ),
        "table_head": paragraph_style(
            "table-head",
            font=JP_BOLD,
            size=6.8,
            leading=9.2,
            alignment=TA_CENTER,
            after=0,
        ),
        "code": paragraph_style(
            "code", size=6.5, leading=8.8, alignment=TA_LEFT, color=INK, after=0
        ),
        "equation": paragraph_style(
            "equation", size=8.0, leading=11.5, alignment=TA_CENTER, after=3
        ),
        "ref": paragraph_style(
            "ref", size=6.6, leading=9.0, alignment=TA_LEFT, color=MUTED, after=2
        ),
    }


def p(text: str, style_name: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles[style_name])


def box(flowables: list, *, fill=PALE, stroke=RULE, padding: float = 7) -> Table:
    table = Table([[flowables]], colWidths=[178 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.6, stroke),
                ("LEFTPADDING", (0, 0), (-1, -1), padding),
                ("RIGHTPADDING", (0, 0), (-1, -1), padding),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        )
    )
    return table


def paper_table(
    rows: list[list[str]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    *,
    centered: tuple[int, ...] = (),
) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        converted_row = []
        for column_index, value in enumerate(row):
            style_name = (
                "table_head"
                if row_index == 0
                else ("table_center" if column_index in centered else "table")
            )
            converted_row.append(p(str(value), style_name, styles))
        converted.append(converted_row)
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("GRID", (0, 0), (-1, -1), 0.35, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [WHITE, colors.HexColor("#FAFAFD")],
                ),
            ]
        )
    )
    return table


def image_for(path: Path, width: float) -> RLImage:
    with PILImage.open(path) as image:
        ratio = image.height / image.width
    return RLImage(str(path), width=width, height=width * ratio)


def seconds_text(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes}分{remainder:.1f}秒"


def percentage(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def build_story(
    data: dict, assets: dict[str, Path], styles: dict[str, ParagraphStyle]
) -> list:
    native = data["native"]
    m4 = data["m4"]
    m8 = data["m8"]
    q4 = data["q4"]
    q8 = data["q8"]
    record = data["record"]
    rt = record["runtime_observations"]
    selected = native["candidates"][2]
    prompt_hash = (
        hashlib.sha256(record["highres_prompt"].encode("utf-8")).hexdigest().upper()
    )
    native_ratio = native["height"] / native["width"]
    final_ratio = m8["target_size"][1] / m8["target_size"][0]
    sqrt2_error = abs(final_ratio - math.sqrt(2.0)) / math.sqrt(2.0) * 100.0
    s4_1, s4_2 = m4["stage_reports"]
    s8 = m8["stage_reports"][0]
    q8c = q8["chroma_mura"]

    story = [
        p(
            "Krea2 A-Series × VRAM-Canvasによる<br/>1:√2縦長イラストの8K生成",
            "title",
            styles,
        ),
        p(
            "A reproducible single-image case study of progressive high-resolution diffusion refinement",
            "subtitle",
            styles,
        ),
        p(
            "技術ケーススタディ / 2026-07-12 / AiWithYou / Forge neo-2.26",
            "meta",
            styles,
        ),
        HRFlowable(
            width="100%", thickness=1.1, color=ACCENT, spaceBefore=0, spaceAfter=8
        ),
    ]

    hero = image_for(assets["hero"], 74 * mm)
    abstract = [
        p("概要", "h2", styles),
        p(
            "ユーザー指定promptからKrea2 A-Seriesのネイティブ候補を4枚生成し、定性的選定した1枚を"
            "VRAM-Canvasで1024×1448から2896×4096、さらに5792×8192へ漸進的にrefineした。"
            "最終画像は47,448,064画素で、縦横比の√2に対する相対誤差は"
            f"{sqrt2_error:.4f}%である。4Kは62/62、8Kは140/140の局所API呼出しがHTTP成功し、"
            "OOMと非有限値は観測しなかった。",
            "body",
            styles,
        ),
        p("実測ハイライト", "h2", styles),
        paper_table(
            [
                ["項目", "結果"],
                ["Native", "1024×1448 / seed 20260714"],
                ["4K delivery", "2896×4096 / 11.86 MP"],
                ["8K delivery", "5792×8192 / 47.45 MP"],
                ["8K tile", "140/140成功 / 2 phases"],
                ["観測GPU", "最大21,656 MiB（間欠sampling）"],
                [
                    "色むらp95",
                    f"{q8c['before']['p95_chroma_delta']:.2f} → {q8c['after']['p95_chroma_delta']:.2f}",
                ],
            ],
            [34 * mm, 62 * mm],
            styles,
        ),
        Spacer(1, 5),
        p(
            "<b>証拠区分:</b> 数値はmanifest・quality report・ファイル時刻からの測定。"
            "継ぎ目や破綻の有無は、全体像と6領域を目視した単一例の観察である。",
            "small",
            styles,
        ),
    ]
    cover_table = Table(
        [[hero, abstract]], colWidths=[78 * mm, 100 * mm], hAlign="CENTER"
    )
    cover_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 6),
                ("LEFTPADDING", (1, 0), (1, 0), 7),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.extend(
        [
            cover_table,
            Spacer(1, 7),
            box(
                [
                    p("本稿の主張範囲", "h2", styles),
                    p(
                        "この結果は「単一RTX 3090 24 GiB環境で、指定例を47.45 MPまで一貫して処理できた」"
                        "ことを示すcase studyである。blind評価、他方式との同条件比較、一般的な知覚品質の優位性、"
                        "文字・手・顔の正しさの保証は含まない。",
                        "body0",
                        styles,
                    ),
                ],
                fill=PALE_GREEN,
                stroke=ACCENT_2,
            ),
            PageBreak(),
            p("1　実験設計とネイティブ候補", "h1", styles),
            p(
                "要求語彙の衝突を最小化するため、<i>smile</i>と<i>Expressionless</i>は「無表情を基調にした"
                "ごく小さな閉口微笑」、<i>jitome</i>と<i>jig eyes</i>は半眼の紫眼としてpositive prompt内で"
                "具体化した。人物は単独の成人として明示した。",
                "body",
                styles,
            ),
            p("1.1　ネイティブ生成prompt", "h2", styles),
            box(
                [
                    p(html.escape(native["prompt"]), "code", styles),
                ],
                fill=colors.HexColor("#F8F8FB"),
            ),
            Spacer(1, 6),
            paper_table(
                [
                    ["設定", "値", "設定", "値"],
                    ["Model", "Krea2 A-Series NF4", "Model hash", "47a2b78020"],
                    ["Native size", "1024×1448", "候補数", "4"],
                    ["Steps", "4", "Sampler", "DPM++ 2M SDE"],
                    ["Scheduler", "Simple", "CFG / distilled", "1.0 / 1.15"],
                    ["Seed", "20260712-20260715", "選定seed", "20260714"],
                ],
                [27 * mm, 62 * mm, 28 * mm, 61 * mm],
                styles,
            ),
            Spacer(1, 7),
            image_for(assets["candidates"], 178 * mm),
            p(
                "Fig. 1　同一promptで生成した4候補。紫枠のCandidate 3を、顔・角・半眼・髪・スライム・全身構図の"
                "均衡が最も良いものとして非blindの定性的レビューで選定した。",
                "caption",
                styles,
            ),
            box(
                [
                    p("CFG=1.0におけるnegative promptの扱い", "h2", styles),
                    p(
                        "Forge logは「Negative Prompts are Ignored when CFG = 1.0」と記録した。したがって本例では"
                        "negative promptを有効な制約として評価しない。重要な禁止条件はpositive promptにも"
                        "no text / no watermark / no tile seams等として重ねた。これは結果解釈上の制約である。",
                        "body0",
                        styles,
                    ),
                ],
                fill=PALE,
                stroke=ACCENT,
            ),
            PageBreak(),
            p("2　高解像度化方法", "h1", styles),
            image_for(assets["pipeline"], 174 * mm),
            p(
                "Fig. 2　漸進的pipeline。4K段は1728×2432と2896×4096の2段、8K段は4Kからの2倍1段。"
                "画像全体をdiffusion modelへ投入せず、halo付き1280px payloadだけを逐次処理した。",
                "caption",
                styles,
            ),
            p("2.1　VRAM-Canvasの残差選択", "h2", styles),
            p(
                "段階基準像をB、局所生成候補をR、低域通過をL、高域をH=I-Lとする。低周波構造が基準像と"
                "一致する候補だけをstructure gate aで通し、基準像自身に局所detailが乏しい領域では"
                "base-detail gate bで新規微細模様を抑える。",
                "body",
                styles,
            ),
            p(
                "a<sub>i</sub>=exp(-|Y(LR<sub>i</sub>)-Y(LB<sub>i</sub>)|/τ<sub>s</sub>)",
                "equation",
                styles,
            ),
            p(
                "e<sub>B</sub>=√mean<sub>c</sub>(H(B)<sub>c</sub>²),　b<sub>i</sub>=e<sub>B</sub>/(e<sub>B</sub>+τ<sub>b</sub>)",
                "equation",
                styles,
            ),
            p(
                "Δ<sub>i</sub>=clip(a<sub>i</sub>b<sub>i</sub>[H(R<sub>i</sub>)-H(B<sub>i</sub>)], -d, d)",
                "equation",
                styles,
            ),
            p(
                "重複候補の加重平均μと分散Vから、互いに競合するdetailをconsensus gateで減衰する。"
                "全候補が一致するとV=0となり、gateは1である。",
                "body",
                styles,
            ),
            p(
                "g<sub>con</sub>=exp(-λV/(E<sub>2</sub>+κ²)),　O=clip(B+g<sub>con</sub>μ,0,255)",
                "equation",
                styles,
            ),
            p("2.2　実行parameter", "h2", styles),
            paper_table(
                [
                    ["parameter", "値", "作用"],
                    [
                        "tile / halo / core",
                        "1280 / 160 / 960px",
                        "GPU payloadと境界文脈",
                    ],
                    ["core overlap / phases", "80px / 2", "重複合意と境界分散"],
                    ["denoise", "0.12 → 0.08 → 0.08", "後段ほど構造保持を強化"],
                    ["low-pass radius / delta", "12px / ±32", "高周波分離と摂動上限"],
                    ["τs / τb / κ", "18 / 6 / 8", "構造・平坦部・合意gate"],
                    ["VRAM budget", "24.0 GiB explicit", "tile planner入力"],
                ],
                [45 * mm, 48 * mm, 85 * mm],
                styles,
            ),
            PageBreak(),
            p("3　測定結果", "h1", styles),
            image_for(assets["detail"], 176 * mm),
            p(
                "Fig. 3　同じ正規化領域の表示比較。左列はnativeを通常のLanczos表示拡大した基準、右列は8K最終像。"
                "これは視覚例であり、perceptual scoreやblind選好を表さない。",
                "caption",
                styles,
            ),
            paper_table(
                [
                    [
                        "stage",
                        "size",
                        "tile",
                        "denoise",
                        "mean a",
                        "mean b",
                        "mean gcon",
                        "|Δ| mean",
                    ],
                    [
                        "4K-1",
                        "1728×2432",
                        "18/18",
                        f"{s4_1['denoise']:.2f}",
                        f"{s4_1['delta_stats']['mean_structure_gate']:.3f}",
                        f"{s4_1['delta_stats']['mean_base_detail_gate']:.3f}",
                        f"{s4_1['consensus_stats']['mean_consensus_gate']:.3f}",
                        f"{s4_1['delta_stats']['mean_abs_delta']:.3f}",
                    ],
                    [
                        "4K-2",
                        "2896×4096",
                        "44/44",
                        f"{s4_2['denoise']:.2f}",
                        f"{s4_2['delta_stats']['mean_structure_gate']:.3f}",
                        f"{s4_2['delta_stats']['mean_base_detail_gate']:.3f}",
                        f"{s4_2['consensus_stats']['mean_consensus_gate']:.3f}",
                        f"{s4_2['delta_stats']['mean_abs_delta']:.3f}",
                    ],
                    [
                        "8K",
                        "5792×8192",
                        "140/140",
                        f"{s8['denoise']:.2f}",
                        f"{s8['delta_stats']['mean_structure_gate']:.3f}",
                        f"{s8['delta_stats']['mean_base_detail_gate']:.3f}",
                        f"{s8['consensus_stats']['mean_consensus_gate']:.3f}",
                        f"{s8['delta_stats']['mean_abs_delta']:.3f}",
                    ],
                ],
                [
                    18 * mm,
                    30 * mm,
                    19 * mm,
                    18 * mm,
                    20 * mm,
                    20 * mm,
                    24 * mm,
                    24 * mm,
                ],
                styles,
                centered=(1, 2, 3, 4, 5, 6, 7),
            ),
            Spacer(1, 6),
            paper_table(
                [
                    ["測定項目", "4K", "8K", "解釈上の注意"],
                    [
                        "wall time",
                        seconds_text(rt["four_k_wall_seconds_from_file_timestamps"]),
                        seconds_text(rt["eight_k_wall_seconds_from_file_timestamps"]),
                        "ファイル時刻差。Smart Finishを含まない",
                    ],
                    [
                        "空間活性削減比",
                        f"{m4['estimated_spatial_activation_reduction']:.2f}×",
                        f"{m8['estimated_spatial_activation_reduction']:.2f}×",
                        "全画面/tile面積比の理論値",
                    ],
                    [
                        "clipped fraction",
                        percentage(s4_2["delta_stats"]["clipped_fraction"]),
                        percentage(s8["delta_stats"]["clipped_fraction"]),
                        "±32へ達した残差割合",
                    ],
                    [
                        "GPU memory",
                        "観測最大21,408 MiB",
                        "観測最大21,656 MiB",
                        "間欠nvidia-smi sampling",
                    ],
                    ["error scan", "0", "0", "OOM / NaN / HTTP failure pattern"],
                ],
                [35 * mm, 34 * mm, 35 * mm, 74 * mm],
                styles,
            ),
            Spacer(1, 6),
            box(
                [
                    p("解像度と比率", "h2", styles),
                    p(
                        f"Native比率は1:{native_ratio:.7f}、最終比率は1:{final_ratio:.7f}。"
                        f"最終値の√2に対する相対誤差は{sqrt2_error:.4f}%で、画素数はnativeの正確に32倍である。",
                        "body0",
                        styles,
                    ),
                ],
                fill=PALE_GREEN,
                stroke=ACCENT_2,
            ),
            PageBreak(),
            p("4　最終仕上げと視覚QA", "h1", styles),
            image_for(assets["qa"], 176 * mm),
            p(
                "Fig. 4　8K Smart Finish後に確認した6領域。表示用に縮小しているが、判定時は各cropを"
                "原寸相当で開き、全体像も別途確認した。",
                "caption",
                styles,
            ),
            p("4.1　Smart Finish", "h2", styles),
            paper_table(
                [
                    ["metric", "4K before", "4K after", "8K before", "8K after"],
                    [
                        "chroma p95",
                        f"{q4['chroma_mura']['before']['p95_chroma_delta']:.2f}",
                        f"{q4['chroma_mura']['after']['p95_chroma_delta']:.2f}",
                        f"{q8c['before']['p95_chroma_delta']:.2f}",
                        f"{q8c['after']['p95_chroma_delta']:.2f}",
                    ],
                    [
                        "area Δchroma > 5",
                        f"{q4['chroma_mura']['before']['area_chroma_delta_gt_5_pct']:.2f}%",
                        f"{q4['chroma_mura']['after']['area_chroma_delta_gt_5_pct']:.2f}%",
                        f"{q8c['before']['area_chroma_delta_gt_5_pct']:.2f}%",
                        f"{q8c['after']['area_chroma_delta_gt_5_pct']:.2f}%",
                    ],
                    [
                        "mean chroma shift",
                        "-",
                        f"{q4['chroma_mura']['mean_chroma_shift']:.3f}",
                        "-",
                        f"{q8c['mean_chroma_shift']:.3f}",
                    ],
                    ["despeckle", "off", "off", "off", "off"],
                ],
                [37 * mm, 35 * mm, 35 * mm, 35 * mm, 35 * mm],
                styles,
                centered=(1, 2, 3, 4),
            ),
            p(
                "chroma指標はLab a/bの局所平滑参照との差に基づく内部heuristicであり、標準化された知覚尺度ではない。"
                "雪・星・意図的粒子・髪の微細線を誤除去しないよう、despeckleは明示的に無効化した。",
                "small",
                styles,
            ),
            p("4.2　目視チェック", "h2", styles),
            paper_table(
                [
                    ["領域", "確認対象", "本例の観察"],
                    [
                        "顔・角",
                        "追加角、二重輪郭、眼の不一致",
                        "明瞭な該当artifactなし",
                    ],
                    [
                        "胴体・袖",
                        "人物重複、輪郭ghost、局所継ぎ目",
                        "明瞭な該当artifactなし",
                    ],
                    ["スライム", "透明面の破断、反復模様、色境界", "連続性を維持"],
                    ["足・床", "接地破綻、tile seam", "明瞭なseamなし"],
                    [
                        "左右背景",
                        "平坦部の偽微細模様、縦横境界",
                        "反復hallucinationなし",
                    ],
                ],
                [31 * mm, 75 * mm, 72 * mm],
                styles,
            ),
            Spacer(1, 6),
            box(
                [
                    p("判定の限界", "h2", styles),
                    p(
                        "「見つからなかった」は「存在しない」の証明ではない。6領域は事前定義した正規化cropであり、"
                        "評価者1名、単一画像、非blindである。手は衣装・スライムに隠れる構図のため、手指の正しさを"
                        "この例から検証したとは扱わない。",
                        "body0",
                        styles,
                    ),
                ],
                fill=PALE,
                stroke=ACCENT,
            ),
            PageBreak(),
            p("5　考察・再現性・限界", "h1", styles),
            p("5.1　この例から言えること", "h2", styles),
            p(
                "全画面8K diffusionを行わず、1280px payloadとdisk-backed accumulatorに分離することで、"
                "RTX 3090上で47.45 MPの納品canvasを処理できた。後段ほどbase-detail gate平均が"
                f"{s4_1['delta_stats']['mean_base_detail_gate']:.3f}から{s8['delta_stats']['mean_base_detail_gate']:.3f}へ下がり、"
                "平均残差も縮小した。これは高解像度段で基準像に既存detailが乏しい場所ほど新規detail注入を"
                "抑えたという実装統計であり、知覚品質向上の直接証拠ではない。",
                "body",
                styles,
            ),
            p("5.2　再現手順", "h2", styles),
            box(
                [
                    p(
                        "1. Krea2 A-Series NF4をForge APIで起動し、1024×1448、4 steps、seed 20260712-20260715で4候補生成。<br/>"
                        "2. seed 20260714を選び、VRAM-Canvasで--width 2896 --height 4096 --phase-count 2 --seed 20260714。<br/>"
                        "3. 4K raw出力を入力し、--width 5792 --height 8192 --phase-count 2 --seed 20260714。<br/>"
                        "4. Smart Finishをdespeckle無効で実行し、PNGのdimensions・metadata・quality JSONを再読込。",
                        "code",
                        styles,
                    ),
                ],
                fill=colors.HexColor("#F8F8FB"),
            ),
            Spacer(1, 5),
            paper_table(
                [
                    ["再現キー", "値"],
                    ["selected native SHA-256", selected["sha256"]],
                    ["highres prompt SHA-256", prompt_hash],
                    ["8K file SHA-256", file_sha256(data["paths"].final_8k)],
                    ["8K pixel dimensions", "5792×8192 / RGB PNG"],
                    [
                        "embedded metadata",
                        "parameters / vram_canvas / krea2_smart_finish",
                    ],
                ],
                [46 * mm, 132 * mm],
                styles,
            ),
            p("5.3　限界と今後の評価", "h2", styles),
            p(
                "(1) 単一prompt・単一選定画像であり、多様な構図や写実画像への一般化は未評価。"
                "(2) 候補選定とartifact判定は非blindで、観察者間一致を測っていない。"
                "(3) nvidia-smiは間欠samplingのため真の瞬間peakを保証しない。"
                "(4) wall timeはホスト負荷、checkpoint、attention backend、storageに依存する。"
                "(5) consensusは全候補が同じ誤りに合意した場合を検出できない。"
                "今後は複数seed・prompt・画像種で、blind pairwise preference、identity/structure距離、seam detector、"
                "連続GPU telemetry、ablation（τb=0を含む）を事前登録して比較する必要がある。",
                "body",
                styles,
            ),
            p("6　結論", "h1", styles),
            p(
                "Krea2 A-Seriesによる1024×1448の縦長イラストを、VRAM-Canvasの構造・基準detail・合意gateで"
                "段階的に5792×8192へ拡張し、Smart Finishまで完了した。全202局所呼出しは成功し、"
                "8K最終像のPNGメタデータには有効prompt、manifest、品質レポートを保持した。"
                "本成果は高解像度生成pipelineの実装可能性を示す再現可能な単一例であり、一般的優位性の結論は"
                "今後の対照実験に留保する。",
                "body",
                styles,
            ),
            p("参考文献", "h2", styles),
            p(
                "[1] R. Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR, 2022.",
                "ref",
                styles,
            ),
            p(
                "[2] O. Bar-Tal et al., MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation, ICML, 2023.",
                "ref",
                styles,
            ),
            p(
                "[3] R. Du et al., DemoFusion: Democratising High-Resolution Image Generation With No $$$, CVPR, 2024.",
                "ref",
                styles,
            ),
            p(
                "[4] Z. Lin et al., AccDiffusion: An Accurate Method for Higher-Resolution Image Generation, arXiv:2407.10738, 2024.",
                "ref",
                styles,
            ),
            p(
                "[5] T. Vontobel et al., HiWave: Training-Free High-Resolution Image Generation via Wavelet-Based Diffusion Sampling, arXiv:2506.20452, 2025.",
                "ref",
                styles,
            ),
        ]
    )
    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(16 * mm, 11 * mm, width - 16 * mm, 11 * mm)
    canvas.setFont(JP_REGULAR, 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 7 * mm, "Krea2 A-Series × VRAM-Canvas 8K Case Study")
    canvas.drawRightString(width - 16 * mm, 7 * mm, str(doc.page))
    canvas.restoreState()


def build_pdf(
    output: Path,
    data: dict,
    assets: dict[str, Path],
    regular_font: Path,
    bold_font: Path,
) -> int:
    register_fonts(regular_font, bold_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
        title="Krea2 A-Series × VRAM-Canvasによる1:√2縦長イラストの8K生成",
        author="AiWithYou",
        subject="RTX 3090上の漸進的高解像度拡散refinement単一画像ケーススタディ",
    )
    doc.build(
        build_story(data, assets, styles), onFirstPage=footer, onLaterPages=footer
    )
    page_count = int(doc.page)
    if page_count != 6:
        raise RuntimeError(f"Expected exactly 6 pages, generated {page_count}")
    return page_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", default=str(DEFAULT_CASE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS))
    parser.add_argument("--font-regular", default=None)
    parser.add_argument("--font-bold", default=None)
    args = parser.parse_args()

    paths = paths_for(Path(args.case_root).resolve())
    data = {
        "native": load_json(paths.native_report),
        "m4": load_json(paths.manifest_4k),
        "m8": load_json(paths.manifest_8k),
        "q4": load_json(paths.quality_4k),
        "q8": load_json(paths.quality_8k),
        "record": load_json(paths.record),
        "paths": paths,
    }
    validate_case(paths, data)
    regular = find_font(args.font_regular, bold=False)
    bold = find_font(args.font_bold, bold=True)
    assets = generate_assets(paths, Path(args.assets_dir).resolve(), bold)
    page_count = build_pdf(Path(args.output).resolve(), data, assets, regular, bold)
    sys.stdout.write(f"PDF={Path(args.output).resolve()}\n")
    sys.stdout.write(f"PAGES={page_count}\n")
    for name, path in assets.items():
        sys.stdout.write(f"ASSET_{name.upper()}={path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
