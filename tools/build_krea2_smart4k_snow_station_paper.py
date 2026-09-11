"""Build the measured Japanese Krea2 Smart 4K snow-station case-study PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
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
DEFAULT_SMART_MANIFEST = ROOT / (
    "output/krea2_smart4k_snow_station/smart8k_20260714_013427_407859/"
    "smart8k_manifest.json"
)
DEFAULT_LOCAL_MANIFEST = ROOT / (
    "output/krea2_smart4k_snow_station/face_refine_1536/"
    "local_supersample_20260714_015753_389398/experiment_manifest.json"
)
DEFAULT_QA_DIR = ROOT / (
    "output/img2img-images/krea2_local_supersample_qa/20260714_015905_497128"
)
DEFAULT_OUTPUT = ROOT / "output/pdf/krea2_smart4k_snow_station_case_study_ja.pdf"
DEFAULT_ASSETS = ROOT / "output/pdf/krea2_smart4k_snow_station_assets"

EXPECTED_SOURCE_SHA256 = (
    "B8563596CFEDE2055EF9BBA0D3258CDDA9EC3EBCB8A3C17DFD5E4637A4B441E9"
)
EXPECTED_FINAL_SHA256 = (
    "45CF42C6CEC5447D6ABEAD221956649429F0541E13D4927FCE7CA10730391BD3"
)
EXPECTED_PIXEL_SHA256 = (
    "78d02e584a4a8bf447a541ae5e3c34f7e0c5b0098fc26b7d0d40741db9634eb3"
)

JP = "Smart4KPaperJP"
JP_BOLD = "Smart4KPaperJPBold"
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#586574")
BLUE = colors.HexColor("#126A8A")
BLUE_PALE = colors.HexColor("#EAF5F8")
MAGENTA = colors.HexColor("#A52D6A")
MAGENTA_PALE = colors.HexColor("#F8EAF1")
RULE = colors.HexColor("#AAB4BE")
TABLE_HEAD = colors.HexColor("#E8EEF3")
BOX = colors.HexColor("#F6F8FA")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def find_font(explicit: str | None, *, bold: bool = False) -> Path:
    candidates = [
        explicit,
        os.environ.get(
            "KREA2_SMART4K_PAPER_JP_BOLD_FONT"
            if bold
            else "KREA2_SMART4K_PAPER_JP_FONT"
        ),
        "C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/meiryob.ttc" if bold else "C:/Windows/Fonts/meiryo.ttc",
        "/usr/share/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold
        else "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "Japanese TrueType font not found; pass --font/--bold-font."
    )


def register_fonts(regular: Path, bold: Path) -> None:
    regular_kwargs = {"subfontIndex": 0} if regular.suffix.lower() == ".ttc" else {}
    bold_kwargs = {"subfontIndex": 0} if bold.suffix.lower() == ".ttc" else {}
    pdfmetrics.registerFont(TTFont(JP, str(regular), **regular_kwargs))
    pdfmetrics.registerFont(TTFont(JP_BOLD, str(bold), **bold_kwargs))
    pdfmetrics.registerFontFamily(
        JP,
        normal=JP,
        bold=JP_BOLD,
        italic=JP,
        boldItalic=JP_BOLD,
    )


def pil_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=0)


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    )
    return canvas


def make_overview(
    source: Image.Image,
    final: Image.Image,
    output: Path,
    bold_font: Path,
) -> Path:
    width, height = 1800, 780
    panel_w, panel_h = 860, 640
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(bold_font, 34)
    label_font = pil_font(bold_font, 26)
    draw.text(
        (width // 2, 28),
        "入力と位相合意付きKrea2 Smart 4K出力",
        font=title_font,
        fill="#17212B",
        anchor="ma",
    )
    source_panel = fit(source.convert("RGB"), (panel_w, panel_h))
    final_panel = fit(final.convert("RGB"), (panel_w, panel_h))
    canvas.paste(source_panel, (25, 90))
    canvas.paste(final_panel, (915, 90))
    draw.rectangle((25, 90, 25 + panel_w, 90 + panel_h), outline="#7F8C99", width=2)
    draw.rectangle((915, 90, 915 + panel_w, 90 + panel_h), outline="#126A8A", width=3)
    draw.text(
        (455, 742), "Input 1915 x 821", font=label_font, fill="#33404D", anchor="ma"
    )
    draw.text(
        (1345, 742),
        "Smart 4K 4096 x 1756",
        font=label_font,
        fill="#126A8A",
        anchor="ma",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


def residual_heatmap(baseline: np.ndarray, final: np.ndarray) -> Image.Image:
    magnitude = np.mean(
        np.abs(final.astype(np.float32) - baseline.astype(np.float32)), axis=2
    )
    scale = float(np.percentile(magnitude, 99.0)) or 1.0
    normalized = np.clip(magnitude / scale, 0.0, 1.0)
    red = np.clip(normalized * 2.0, 0.0, 1.0)
    green = np.clip((normalized - 0.25) * 1.6, 0.0, 1.0)
    blue = np.clip((normalized - 0.75) * 4.0, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")


def make_face_comparison(
    baseline: Image.Image,
    final: Image.Image,
    output: Path,
    bold_font: Path,
) -> Path:
    box = (2520, 430, 3120, 1180)
    before = baseline.crop(box).convert("RGB")
    after = final.crop(box).convert("RGB")
    heat = residual_heatmap(np.asarray(before), np.asarray(after))
    panel_size = (600, 750)
    width, height = 1900, 850
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(bold_font, 30)
    label_font = pil_font(bold_font, 24)
    labels = ("Lanczos 4K基準", "VRAM-Canvas + Finish", "差分heatmap (p99正規化)")
    colors_ = ("#586574", "#126A8A", "#A52D6A")
    panels = (before, after, heat)
    draw.text(
        (width // 2, 22),
        "人物・顔周辺の同一座標100% crop",
        font=title_font,
        fill="#17212B",
        anchor="ma",
    )
    for index, (panel, label, color) in enumerate(zip(panels, labels, colors_)):
        x = 25 + index * 625
        canvas.paste(panel.resize(panel_size, Image.Resampling.LANCZOS), (x, 70))
        draw.rectangle((x, 70, x + 600, 820), outline=color, width=3)
        draw.text((x + 300, 835), label, font=label_font, fill=color, anchor="ms")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


def make_tile_map(
    final: Image.Image,
    stage: dict,
    output: Path,
    regular_font: Path,
    bold_font: Path,
) -> Path:
    preview_w = 1600
    preview_h = round(final.height * preview_w / final.width)
    top = 92
    canvas = Image.new("RGB", (preview_w, preview_h + top + 44), "white")
    preview = final.convert("RGB").resize(
        (preview_w, preview_h), Image.Resampling.LANCZOS
    )
    canvas.paste(preview, (0, top))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    sx = preview_w / final.width
    sy = preview_h / final.height
    phase_colors = ((0, 214, 255, 210), (255, 43, 155, 210))
    for tile in stage["tiles"]:
        color = phase_colors[int(tile["phase"])]
        x0 = round(tile["core_x0"] * sx)
        y0 = top + round(tile["core_y0"] * sy)
        x1 = round(tile["core_x1"] * sx)
        y1 = top + round(tile["core_y1"] * sy)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw_rgb = ImageDraw.Draw(canvas)
    title_font = pil_font(bold_font, 31)
    label_font = pil_font(regular_font, 21)
    draw_rgb.text(
        (preview_w // 2, 22),
        "最終段4096 x 1756: 半strideずらしの2位相core配置",
        font=title_font,
        fill="#17212B",
        anchor="ma",
    )
    draw_rgb.text((30, 66), "cyan: phase 1", font=label_font, fill="#0088A8")
    draw_rgb.text((230, 66), "magenta: phase 2", font=label_font, fill="#B21867")
    draw_rgb.text(
        (preview_w - 30, 66),
        f"{stage['tile_count']} tiles / overlap and halo omitted from lines",
        font=label_font,
        fill="#586574",
        anchor="ra",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


def make_local_qa(
    qa_dir: Path,
    output: Path,
    regular_font: Path,
    bold_font: Path,
) -> Path:
    names = (
        "source_payload.png",
        "downsampled_candidate_c1.png",
        "residual_visualization.png",
    )
    labels = ("4K source payload", "1536生成後を512へ縮小", "採用残差 (全ゼロ)")
    panels = []
    for name in names:
        path = qa_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            panels.append(image.convert("RGB"))
    width, height = 1640, 650
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(bold_font, 29)
    label_font = pil_font(regular_font, 21)
    draw.text(
        (width // 2, 20),
        "顔ROI Ultra Detail 1536: 候補を貼らず品質gateでno-op",
        font=title_font,
        fill="#17212B",
        anchor="ma",
    )
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = 20 + index * 540
        resized = panel.resize(
            (512, 512),
            Image.Resampling.NEAREST if index == 2 else Image.Resampling.LANCZOS,
        )
        canvas.paste(resized, (x, 72))
        draw.rectangle((x, 72, x + 512, 584), outline="#7F8C99", width=2)
        draw.text((x + 256, 615), label, font=label_font, fill="#33404D", anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


def grayscale(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32)
    return values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722


def filtered_gray(gray: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.clip(np.rint(gray), 0, 255).astype(np.uint8), mode="L")
    return np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32
    )


def p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95.0)) if values.size else 0.0


def compute_metrics(baseline: Image.Image, final: Image.Image, stage: dict) -> dict:
    before = np.asarray(baseline.convert("RGB"), dtype=np.uint8)
    after = np.asarray(final.convert("RGB"), dtype=np.uint8)
    if before.shape != after.shape:
        raise ValueError("baseline and final shapes differ")
    delta = after.astype(np.float32) - before.astype(np.float32)
    absolute = np.abs(delta)
    before_y = grayscale(before)
    after_y = grayscale(after)
    before_low = filtered_gray(before_y, 1.0)
    after_low = filtered_gray(after_y, 1.0)
    before_hp = np.abs(before_y - before_low)
    after_hp = np.abs(after_y - after_low)
    low_drift = np.abs(filtered_gray(after_y, 12.0) - filtered_gray(before_y, 12.0))

    residual_y = grayscale(delta)
    global_jumps = np.concatenate(
        (
            np.abs(np.diff(residual_y, axis=1)).ravel(),
            np.abs(np.diff(residual_y, axis=0)).ravel(),
        )
    )
    boundary_values = []
    x_boundaries = sorted(
        {
            int(value)
            for tile in stage["tiles"]
            for value in (tile["core_x0"], tile["core_x1"])
            if 0 < int(value) < residual_y.shape[1]
        }
    )
    y_boundaries = sorted(
        {
            int(value)
            for tile in stage["tiles"]
            for value in (tile["core_y0"], tile["core_y1"])
            if 0 < int(value) < residual_y.shape[0]
        }
    )
    for x in x_boundaries:
        boundary_values.append(np.abs(residual_y[:, x] - residual_y[:, x - 1]).ravel())
    for y in y_boundaries:
        boundary_values.append(np.abs(residual_y[y, :] - residual_y[y - 1, :]).ravel())
    boundaries = (
        np.concatenate(boundary_values)
        if boundary_values
        else np.empty(0, dtype=np.float32)
    )
    global_jump_p95 = p95(global_jumps)
    boundary_jump_p95 = p95(boundaries)
    changed = np.any(before != after, axis=2)
    return {
        "comparison": "Smart 4K versus same-size sRGB Lanczos baseline",
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_pixel_percent": float(np.mean(changed) * 100.0),
        "mean_abs_rgb_delta": float(np.mean(absolute, dtype=np.float64)),
        "p95_abs_rgb_delta": p95(absolute),
        "p99_abs_rgb_delta": float(np.percentile(absolute, 99.0)),
        "max_abs_rgb_delta": int(np.max(absolute)),
        "low_frequency_luma_drift_mean": float(np.mean(low_drift, dtype=np.float64)),
        "low_frequency_luma_drift_p95": p95(low_drift),
        "baseline_highpass_abs_mean": float(np.mean(before_hp, dtype=np.float64)),
        "final_highpass_abs_mean": float(np.mean(after_hp, dtype=np.float64)),
        "highpass_mean_ratio": float(
            np.mean(after_hp, dtype=np.float64)
            / max(np.mean(before_hp, dtype=np.float64), 1e-9)
        ),
        "baseline_highpass_abs_p95": p95(before_hp),
        "final_highpass_abs_p95": p95(after_hp),
        "highpass_p95_ratio": float(p95(after_hp) / max(p95(before_hp), 1e-9)),
        "residual_global_jump_p95": global_jump_p95,
        "residual_tile_boundary_jump_p95": boundary_jump_p95,
        "residual_boundary_to_global_p95_ratio": float(
            boundary_jump_p95 / max(global_jump_p95, 1e-9)
        ),
        "x_boundary_count": len(x_boundaries),
        "y_boundary_count": len(y_boundaries),
    }


def validate_case(smart: dict, local: dict, qa_dir: Path) -> dict:
    if smart.get("status") != "complete_4k":
        raise RuntimeError("Smart manifest is not complete_4k")
    if smart["resolution_plan"]["source"] != [1915, 821]:
        raise RuntimeError("unexpected source size")
    if smart["resolution_plan"]["preflight_4k"] != [4096, 1756]:
        raise RuntimeError("unexpected target size")
    source = artifact_path(smart["steps"]["source"]["path"])
    final = artifact_path(smart["steps"]["preflight_4k"]["final"]["path"])
    vram_manifest = artifact_path(smart["steps"]["preflight_4k"]["vram_canvas"]["path"])
    if file_sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("source file hash mismatch")
    if file_sha256(final) != EXPECTED_FINAL_SHA256:
        raise RuntimeError("final file hash mismatch")
    vram = read_json(vram_manifest)
    if sum(int(stage["processed_tile_count"]) for stage in vram["stage_reports"]) != 44:
        raise RuntimeError("expected 44 processed VRAM-Canvas tiles")
    if sum(int(stage["skipped_tile_count"]) for stage in vram["stage_reports"]) != 0:
        raise RuntimeError("VRAM-Canvas manifest contains skipped tiles")
    embedded = local["embedded_krea2_local_supersample"]
    if local["input"]["pixel_sha256"] != EXPECTED_PIXEL_SHA256:
        raise RuntimeError("local input pixel hash mismatch")
    if local["output"]["pixel_sha256"] != EXPECTED_PIXEL_SHA256:
        raise RuntimeError("local output is not pixel-identical")
    if (
        embedded["processed_tile_count"] != 4
        or embedded["rejected_noop_tile_count"] != 4
    ):
        raise RuntimeError("expected four fail-closed local no-op tiles")
    if local["difference_metrics"]["changed_pixel_count"] != 0:
        raise RuntimeError("local output unexpectedly changed pixels")
    if not qa_dir.is_dir():
        raise FileNotFoundError(qa_dir)
    return {"source": source, "final": final, "vram": vram}


class PipelineFigure(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 50 * mm

    def draw(self) -> None:
        c = self.canv
        c.setLineWidth(0.8)
        c.setStrokeColor(BLUE)
        nodes = [
            ("Input", 0.01, 0.68),
            ("Progressive\nLanczos", 0.20, 0.68),
            ("Halo tile\nKrea2 i2i", 0.40, 0.68),
            ("Band-limited\nresidual", 0.60, 0.68),
            ("2-phase\nconsensus", 0.80, 0.68),
            ("Smart Finish", 0.40, 0.20),
            ("Optional local\nsupersample", 0.67, 0.20),
        ]
        node_w = self.width * 0.16
        node_h = 15 * mm
        for label, xf, yf in nodes:
            x = self.width * xf
            y = self.height * yf
            c.setFillColor(BLUE_PALE if yf > 0.5 else MAGENTA_PALE)
            c.roundRect(x, y, node_w, node_h, 2.2 * mm, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 7.0)
            lines = label.split("\n")
            for index, line in enumerate(lines):
                c.drawCentredString(
                    x + node_w / 2,
                    y + node_h / 2 + (len(lines) / 2 - index - 0.7) * 8,
                    line,
                )
        c.setStrokeColor(INK)
        c.setFillColor(INK)
        top_indices = range(4)
        for index in top_indices:
            x1 = self.width * nodes[index][1] + node_w
            y = self.height * nodes[index][2] + node_h / 2
            x2 = self.width * nodes[index + 1][1]
            c.line(x1, y, x2, y)
            c.line(x2 - 4, y + 2, x2, y)
            c.line(x2 - 4, y - 2, x2, y)
        x = self.width * nodes[4][1] + node_w / 2
        y1 = self.height * nodes[4][2]
        y2 = self.height * nodes[5][2] + node_h
        c.line(x, y1, x, y2)
        c.line(x, y2, self.width * nodes[5][1] + node_w, y2)
        c.line(self.width * nodes[5][1] + node_w, y2, self.width * nodes[6][1], y2)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=JP_BOLD,
            fontSize=17.2,
            leading=21.5,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=JP,
            fontSize=9.4,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "author": ParagraphStyle(
            "author",
            parent=base["Normal"],
            fontName=JP,
            fontSize=8.2,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=JP_BOLD,
            fontSize=13.2,
            leading=17,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=JP_BOLD,
            fontSize=10.5,
            leading=13.5,
            textColor=BLUE,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=8.3,
            leading=12.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            wordWrap="CJK",
            firstLineIndent=8.3,
            spaceAfter=3,
        ),
        "body0": ParagraphStyle(
            "body0",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=8.3,
            leading=12.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            wordWrap="CJK",
            spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=7.2,
            leading=10.4,
            textColor=INK,
            wordWrap="CJK",
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=7.0,
            leading=9.8,
            textColor=MUTED,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceBefore=2,
            spaceAfter=5,
        ),
        "equation": ParagraphStyle(
            "equation",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=9.0,
            leading=13,
            textColor=INK,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=7.0,
            leading=9.5,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["BodyText"],
            fontName=JP_BOLD,
            fontSize=7.0,
            leading=9.5,
            textColor=INK,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "reference",
            parent=base["BodyText"],
            fontName=JP,
            fontSize=7.0,
            leading=9.8,
            textColor=INK,
            wordWrap="CJK",
            leftIndent=9,
            firstLineIndent=-9,
            spaceAfter=2,
        ),
    }


def paragraph(
    styles: dict[str, ParagraphStyle], text: str, style: str = "body"
) -> Paragraph:
    return Paragraph(text, styles[style])


def paper_table(
    styles: dict[str, ParagraphStyle], rows: list[list[str]], widths: list[float]
) -> Table:
    content = []
    for row_index, row in enumerate(rows):
        content.append(
            [
                paragraph(styles, value, "table_head" if row_index == 0 else "table")
                for value in row
            ]
        )
    table = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
                ("GRID", (0, 1), (-1, -2), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def callout(
    styles: dict[str, ParagraphStyle], text: str, *, accent: bool = False
) -> Table:
    table = Table([[paragraph(styles, text, "body0")]], colWidths=[167 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), MAGENTA_PALE if accent else BLUE_PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, MAGENTA if accent else BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def scaled_image(path: Path, width: float) -> RLImage:
    with Image.open(path) as image:
        ratio = image.height / image.width
    return RLImage(str(path), width=width, height=width * ratio)


def page_chrome(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.35)
    canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
    canvas.setFont("Helvetica", 6.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        20 * mm, 9.5 * mm, "Krea2 Smart 4K / Single-image measured technical report"
    )
    canvas.drawRightString(A4[0] - 20 * mm, 9.5 * mm, str(doc.page))
    if doc.page > 1:
        canvas.setFont("Helvetica-Bold", 6.2)
        canvas.drawString(
            20 * mm, A4[1] - 12 * mm, "PHASE-CONSENSUS BAND-LIMITED TILED DIFFUSION"
        )
        canvas.line(20 * mm, A4[1] - 14 * mm, A4[0] - 20 * mm, A4[1] - 14 * mm)
    canvas.restoreState()


def build_story(
    styles: dict[str, ParagraphStyle],
    smart: dict,
    local: dict,
    vram: dict,
    metrics: dict,
    assets: dict[str, Path],
) -> list:
    preflight = smart["steps"]["preflight_4k"]
    finish = preflight["finish_report"]["detail_guard"]
    telemetry = preflight["telemetry"]["samples"]
    peak_vram = max(int(sample["memory_used_mib"]) for sample in telemetry)
    peak_temp = max(int(sample["temperature_c"]) for sample in telemetry)
    stage2 = vram["stage_reports"][1]
    local_peak_vram = max(
        int(sample["memory_used_mib"]) for sample in local["telemetry_samples"]
    )
    local_peak_temp = max(
        int(sample["temperature_c"]) for sample in local["telemetry_samples"]
    )

    story = [
        Spacer(1, 4 * mm),
        paragraph(
            styles, "位相合意付き帯域制限タイル拡散による任意画像の4K化", "title"
        ),
        paragraph(
            styles, "Krea2 / Forge Neo / RTX 3090 による単一画像実測短報", "subtitle"
        ),
        paragraph(styles, "AiWithYou - Technical Report - 2026-07-14", "author"),
        HRFlowable(
            width="100%", thickness=0.6, color=RULE, spaceBefore=2, spaceAfter=7
        ),
        callout(
            styles,
            "<b>要旨 - </b>Krea2のnative域を超える4Kを一括生成せず、段階拡大、halo付き重複タイル、"
            "周波数分離残差、半strideずらしの2位相合意、disk-backed合成、fail-closed局所超標本化を組み合わせた。"
            "1915x821の添付画像を4096x1756へ変換し、44/44 tile成功、skip 0、拡散387.911秒、peak VRAM "
            f"{peak_vram:,} MiBで完走した。顔ROIの512->1536超標本化は4/4 tileを品質gateがno-opとし、"
            "改善根拠のない再描画を一画素も採用しなかった。",
        ),
        Spacer(1, 5),
        scaled_image(assets["overview"], 167 * mm),
        paragraph(
            styles,
            "Fig. 1. 入力と長辺4K出力。アスペクト比は保持し、16:9へのcropや引き延ばしを行っていない。",
            "caption",
        ),
        paragraph(styles, "1. 問題設定", "h1"),
        paragraph(
            styles,
            "Krea2公式実装はRawを最大約1K、Turboを約1K-2Kの生成域として案内する。Technical Reportもnative 2K/4Kを将来能力として挙げる。"
            "したがって4096幅をモデルへ一括投入せず、モデルが扱える局所contextを全体キャンバス上で統合する必要がある。",
        ),
        paragraph(
            styles,
            "本研究の焦点は単なるtile平均ではない。人物同一性、低周波の構図、平坦面、文字、反復模様を守りながら、"
            "複数の独立観測が支持した小振幅の高周波差分だけを採用する。生成細部は未知の真値復元ではなく、入力とpromptに整合する保守的推定である。",
        ),
        paragraph(styles, "主な貢献", "h2"),
        paper_table(
            styles,
            [
                ["要素", "役割"],
                [
                    "Progressive resize",
                    "各辺の段階倍率を2以下に制限し、全体構造を基準像へ固定",
                ],
                [
                    "Halo + overlap",
                    "tile境界をcontext外へ退避し、smoothstep重みで正規化",
                ],
                ["Band-limited residual", "低周波の色・明度・輪郭移動を直接合成しない"],
                [
                    "Shifted 2-phase consensus",
                    "一度だけ出たhallucinationを二次momentで減衰",
                ],
                [
                    "Local fail-closed",
                    "拡大生成後の候補が改善しなければbit-identical no-op",
                ],
            ],
            [42 * mm, 125 * mm],
        ),
        PageBreak(),
        paragraph(styles, "2. 提案手法", "h1"),
        PipelineFigure(167 * mm),
        paragraph(styles, "Fig. 2. VRAM-Canvasと局所超標本化の処理系列。", "caption"),
        paragraph(styles, "2.1 段階拡大とタイル幾何", "h2"),
        paragraph(
            styles,
            "入力I0から各辺の拡大率が2を超えないstage列を作る。今回のstageは2800x1200と4096x1756である。"
            "最終段はpayload T=1280、halo h=160、core c=960、overlap o=80。GPU側の空間活性量は全キャンバスではなくT x Tで上限を持つ。",
        ),
        paragraph(
            styles, "Bs = Resize(Is-1),   max(Ws/Ws-1, Hs/Hs-1) <= 2", "equation"
        ),
        paragraph(styles, "2.2 既存detailとnovel-detail", "h2"),
        paragraph(
            styles,
            "Krea2 tile Kiと基準tile Biの高域差へstructure gateとbase-detail gateを掛け、RGB差を最大+/-32 codeへ制限する。"
            "novel branchは2-8px帯域の輝度差だけを最大+/-8 codeで提案し、色や大きな顔構造を直接描き替えない。",
        ),
        paragraph(
            styles, "DeltaE = [HP(Ki) - HP(Bi)] * Gstructure * Gbase", "equation"
        ),
        paragraph(
            styles, "DeltaN = clip(Y[(L2-L8)(Ki) - (L2-L8)(Bi)] * G, -8, 8)", "equation"
        ),
        paragraph(styles, "2.3 2位相の一次・二次moment合意", "h2"),
        paragraph(
            styles,
            "通常gridと半strideずらしgridを別seedで処理し、画素ごとの重み付き一次momentと二次momentをdisk-backed float32配列へ蓄積する。"
            "mean residual muに対する分散vからconfidence gを求める。強くても位相間で方向が一致しない差分は抑制される。",
        ),
        paragraph(
            styles,
            "mu = sum(wi Deltai) / sum(wi),   v = E[Delta^2] - E[Delta]^2",
            "equation",
        ),
        paragraph(
            styles,
            "g = exp(-4v / (e + sigma^2)),   Is = Bs + g mu,   sigma = 8",
            "equation",
        ),
        paragraph(styles, "2.4 任意画像向けprompt境界", "h2"),
        paragraph(
            styles,
            "GUIは元promptをprefixとして保持し、人物の同一性・顔比率・年齢・表情・視線・手指数、camera、文字、物体数、scene geometryを固定する。"
            "髪、虹彩、布、木、石、植生、透明物、液体、線画など入力に実在する材質だけを精密化し、アニメやflat-colorへ写真風毛穴を強制しない。"
            "旧suffixの題材依存語horn/slimeは除去した。",
        ),
        paragraph(styles, "2.5 局所超標本化", "h2"),
        paragraph(
            styles,
            "512px payload Bを1536/2048へLanczos拡大Uし、Krea2候補K(U(B))を同じlinear-light area経路Dで縮小する。"
            "候補画像は貼らず、C1-C0から低周波を除いた差分だけをquality gate通過時に戻す。2候補は平均せず、一方を代表、他方をagreement証拠に使う。",
        ),
        paragraph(
            styles,
            "C0 = D(U(B)),   C1 = D(K(U(B))),   Deltalocal = BandPass(C1-C0)",
            "equation",
        ),
        PageBreak(),
        paragraph(styles, "3. 実験設定と4K結果", "h1"),
        paper_table(
            styles,
            [
                ["設定", "値", "設定", "値"],
                ["GPU", "RTX 3090 24 GiB", "Checkpoint", "custom Krea2 Turbo NF4"],
                ["Input", "1915x821", "Target", "4096x1756"],
                ["Stages", "2800x1200 -> 4096x1756", "Tiles", "16 + 28 = 44"],
                ["Tile geometry", "1280 / 160 / 960 / 80", "Phases", "2"],
                ["Steps", "adaptive 3-4", "Denoise", "0.16 -> 0.13"],
                ["Sampler", "DPM++ 2M SDE / Simple", "Seed", "3883506083"],
            ],
            [28 * mm, 56 * mm, 28 * mm, 55 * mm],
        ),
        Spacer(1, 5),
        scaled_image(assets["tile_map"], 167 * mm),
        paragraph(
            styles,
            "Fig. 3. 最終段の2位相core配置。線はcore境界であり、実payloadは外側haloを含む。",
            "caption",
        ),
        paper_table(
            styles,
            [
                ["測定項目", "結果", "測定項目", "結果"],
                ["成功 / skip", "44 / 0", "Diffusion", "387.911 s"],
                ["Peak VRAM", f"{peak_vram:,} MiB", "Peak temp", f"{peak_temp} C"],
                [
                    "Stage 2 consensus",
                    f"{stage2['consensus_stats']['mean_consensus_gate']:.6f}",
                    "Novel consensus",
                    f"{stage2['consensus_stats']['mean_novel_consensus_gate']:.6f}",
                ],
                [
                    "Finish changed",
                    f"{finish['changed_percent']:.6f}%",
                    "Flat changed",
                    f"{finish['flat_region_changed_pixels']} px",
                ],
                [
                    "Finish clipping",
                    f"{finish['clipped_channel_fraction'] * 100:.7f}%",
                    "Detail energy",
                    f"{finish['detail_energy_ratio']:.6f}x",
                ],
                [
                    "Lanczos比 HP mean",
                    f"{metrics['highpass_mean_ratio']:.6f}x",
                    "Lanczos比 HP p95",
                    f"{metrics['highpass_p95_ratio']:.6f}x",
                ],
                [
                    "Residual seam p95比",
                    f"{metrics['residual_boundary_to_global_p95_ratio']:.6f}",
                    "Low-frequency drift p95",
                    f"{metrics['low_frequency_luma_drift_p95']:.3f} code",
                ],
            ],
            [35 * mm, 49 * mm, 37 * mm, 46 * mm],
        ),
        paragraph(
            styles,
            "Residual seam p95比はSmart 4Kと同寸法Lanczos基準との差分について、計画core境界の隣接jump p95を全画面jump p95で割った参考診断であり、単独の合否gateではない。"
            "全体像と固定7地点の1024x1024原寸cropでは、人物増殖、明瞭な二重輪郭、周期模様、tile seamを認めなかった。",
        ),
        PageBreak(),
        paragraph(styles, "4. 人物・顔の評価", "h1"),
        scaled_image(assets["face"], 167 * mm),
        paragraph(
            styles,
            "Fig. 4. 同一座標のLanczos基準、最終4K、p99正規化差分。差分は真値誤差ではなく生成処理量を示す。",
            "caption",
        ),
        paragraph(
            styles,
            "顔、髪流れ、新聞、衣装、脚、車、アーチ窓、床反射を100% cropで確認した。小さな人物の顔は全体4Kパスで破綻していないが、"
            "追加の局所再生成が必ず有効とは限らないため、顔周辺ROI (2580,560)-(2870,930) を別途検査した。",
        ),
        scaled_image(assets["local_qa"], 167 * mm),
        paragraph(
            styles,
            "Fig. 5. 顔を含む代表payload。1536候補は縮小後にdetail energyが増えず、採用残差は全ゼロ。",
            "caption",
        ),
        paper_table(
            styles,
            [
                ["Local 1536測定", "結果", "Local 1536測定", "結果"],
                ["Tiles / candidates", "4 / 8", "Accepted / no-op", "0 / 4"],
                [
                    "Duration",
                    f"{local['duration_seconds']:.3f} s",
                    "Peak VRAM",
                    f"{local_peak_vram:,} MiB",
                ],
                ["Peak temperature", f"{local_peak_temp} C", "Changed pixels", "0"],
                [
                    "Agreement coverage",
                    "0",
                    "Pixel hash",
                    EXPECTED_PIXEL_SHA256[:16] + "...",
                ],
            ],
            [39 * mm, 44 * mm, 42 * mm, 42 * mm],
        ),
        callout(
            styles,
            "<b>Fail-closed結果 - </b>全4 tileの候補は <font name='Courier'>detail_energy_did_not_increase</font> で棄却された。"
            "ROI外だけでなく画像全体がbit-identicalである。顔だから無条件に生成候補を貼るのではなく、改善証拠がなければ元4Kを採用した。",
            accent=True,
        ),
        Spacer(1, 6),
        paragraph(styles, "2048負荷観測", "h2"),
        paragraph(
            styles,
            "ROI Ultra 2048は最初の1候補が806秒時点でも完了せず、VRAM約23.1 GiB、GPU使用率約100%を維持したため明示的に中断した。"
            "暗黙に1536へfallbackして画質設定を変えることはしていない。本RTX 3090環境では1536を実用既定とする。",
        ),
        PageBreak(),
        paragraph(styles, "5. 考察と限界", "h1"),
        paragraph(
            styles,
            "全体4Kパスでは2位相が支持した帯域制限残差とcoherent detailを採用し、顔の追加超標本化では全候補を棄却した。"
            "この非対称な結果が本手法の意図である。モデル出力をdetailと同一視せず、局所・帯域・振幅・複数位相の証拠で採否を決める。",
        ),
        paragraph(
            styles,
            "ただし本稿は単一画像・単一seed・単一観察者の結果であり、4K ground truthを持たない。PSNR/SSIM型の超解像ベンチマークではない。"
            "生成細部は未知の真値復元ではないため、科学、医療、法的証拠画像へ適用すべきでない。文字や極小の顔はモデルpriorの限界を受ける。",
        ),
        paragraph(
            styles,
            "2位相overlapは計算量を増やし、最高解像度段が総コストの中心となる。一方、GPU空間活性をtile edgeで制限し、7.19MP全体をモデルへ一括投入せずRTX 3090で完走した。",
        ),
        paragraph(styles, "6. Forge GUI手順", "h1"),
        paper_table(
            styles,
            [
                ["Step", "操作"],
                ["1", "Krea2 checkpoint、Qwen Image VAE、Qwen3-VLを選択"],
                ["2", "img2imgへ入力画像と正確なpromptを設定"],
                ["3", "Script: VRAM-Canvas 4K/8K Highres"],
                ["4", "4K Smart - long edge 4096 + profile"],
                ["5", "Krea2 Dense Detail 4K、Grid Phases 2を確認してGenerate"],
                ["6", "全体と100% cropを目視確認"],
                [
                    "7",
                    "必要なROIだけKrea2 Local Supersample Detail / Ultra Detail 1536",
                ],
            ],
            [18 * mm, 149 * mm],
        ),
        paragraph(styles, "7. 再現性", "h1"),
        paragraph(
            styles,
            "4K file SHA-256: <font name='Courier'>"
            + EXPECTED_FINAL_SHA256
            + "</font><br/>"
            "Input file SHA-256: <font name='Courier'>"
            + EXPECTED_SOURCE_SHA256
            + "</font><br/>"
            "4K PNG text chunks: <font name='Courier'>parameters, vram_canvas, krea2_smart_finish</font><br/>"
            "Manifests: <font name='Courier'>smart8k_manifest.json, run_manifest.json, experiment_manifest.json</font>",
            "small",
        ),
        paragraph(styles, "8. 結論", "h1"),
        paragraph(
            styles,
            "1915x821画像を4096x1756へ、44/44 tile成功、skip 0で変換した。全体4Kは合意付き高周波残差を採用し、"
            "顔ROIは改善根拠のない4/4 tileをno-opへ戻した。本手法は画質向上を無条件に保証せず、証拠が弱い場合に元画像を保つ。",
        ),
        paragraph(styles, "参考文献", "h1"),
        paragraph(
            styles,
            "[1] Krea AI. Krea 2 official inference code - Usage. https://github.com/krea-ai/krea-2#usage (2026).",
            "reference",
        ),
        paragraph(
            styles,
            "[2] S. Lee et al. Krea 2 Technical Report. https://www.krea.ai/blog/krea-2-technical-report (2026).",
            "reference",
        ),
        paragraph(
            styles,
            "[3] O. Bar-Tal et al. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation. ICML (2023). https://proceedings.mlr.press/v202/bar-tal23a.html",
            "reference",
        ),
        paragraph(
            styles,
            "[4] A. Barbero Jimenez. Mixture of Diffusers for scene composition and high resolution image generation. arXiv:2302.02412 (2023).",
            "reference",
        ),
        paragraph(
            styles,
            "[5] R. Du et al. DemoFusion: Democratising High-Resolution Image Generation With No $$$. CVPR (2024).",
            "reference",
        ),
    ]
    return story


def build_pdf(
    smart_manifest: Path,
    local_manifest: Path,
    qa_dir: Path,
    output: Path,
    assets_dir: Path,
    regular_font: Path,
    bold_font: Path,
) -> dict:
    smart = read_json(smart_manifest)
    local = read_json(local_manifest)
    case = validate_case(smart, local, qa_dir)
    vram = case["vram"]
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    register_fonts(regular_font, bold_font)

    with Image.open(case["source"]) as opened:
        source = opened.convert("RGB")
    with Image.open(case["final"]) as opened:
        final = opened.convert("RGB")
    baseline = source.resize(final.size, Image.Resampling.LANCZOS)
    stage2 = vram["stage_reports"][1]
    metrics = compute_metrics(baseline, final, stage2)
    metrics_path = assets_dir / "case_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assets = {
        "overview": make_overview(
            source, final, assets_dir / "overview.png", bold_font
        ),
        "face": make_face_comparison(
            baseline, final, assets_dir / "face_comparison.png", bold_font
        ),
        "tile_map": make_tile_map(
            final, stage2, assets_dir / "tile_map.png", regular_font, bold_font
        ),
        "local_qa": make_local_qa(
            qa_dir, assets_dir / "local_qa.png", regular_font, bold_font
        ),
    }

    styles = make_styles()
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=21 * mm,
        leftMargin=21 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm,
        title="位相合意付き帯域制限タイル拡散による任意画像の4K化",
        author="AiWithYou",
        subject="Krea2 Smart 4K measured technical report",
    )
    document.build(
        build_story(styles, smart, local, vram, metrics, assets),
        onFirstPage=page_chrome,
        onLaterPages=page_chrome,
    )

    pdf_bytes = output.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-1024:]:
        raise RuntimeError("generated file does not have a complete PDF envelope")
    page_count = len(re.findall(rb"/Type\s*/Page(?:\s|/|>>)", pdf_bytes))
    if page_count != 5:
        raise RuntimeError(
            f"paper unexpectedly has {page_count} page objects; expected 5"
        )
    return {
        "output": str(output),
        "pages": page_count,
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "metrics": str(metrics_path),
        "assets": {name: str(path) for name, path in assets.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smart-manifest", type=Path, default=DEFAULT_SMART_MANIFEST)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--font", default=None)
    parser.add_argument("--bold-font", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    regular = find_font(args.font, bold=False)
    bold = find_font(args.bold_font, bold=True)
    result = build_pdf(
        args.smart_manifest.resolve(),
        args.local_manifest.resolve(),
        args.qa_dir.resolve(),
        args.output.resolve(),
        args.assets_dir.resolve(),
        regular,
        bold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))  # noqa: T201 - CLI result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
