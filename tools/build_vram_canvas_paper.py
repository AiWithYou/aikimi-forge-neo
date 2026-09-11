"""Build the two-page Japanese B5 VRAM-Canvas short paper."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

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
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#555555")
RULE = colors.HexColor("#777777")
HAIRLINE = colors.HexColor("#B8B8B8")
TABLE_FILL = colors.HexColor("#EFEFEF")
BOX_FILL = colors.HexColor("#F7F7F7")
JP_REGULAR = "VRAMCanvasJP-Regular"
JP_BOLD = "VRAMCanvasJP-Bold"
REPO_ROOT = Path(__file__).resolve().parents[1]

JIS_B5 = (182 * mm, 257 * mm)
PAGE_WIDTH, PAGE_HEIGHT = JIS_B5
MARGIN_X = 10 * mm
GUTTER = 5 * mm
BOTTOM = 13 * mm
TOP = 10 * mm
HEADER_HEIGHT = 50 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
COLUMN_WIDTH = (CONTENT_WIDTH - GUTTER) / 2


def find_japanese_font(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("VRAM_CANVAS_JP_FONT"),
        "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "/tmp/google-fonts/ofl/notosansjp/NotoSansJP[wght].ttf",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("A Japanese TrueType font is required. Pass --font or set " "VRAM_CANVAS_JP_FONT (Noto Sans JP is recommended).")


def find_japanese_bold_font(regular: Path, explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("VRAM_CANVAS_JP_BOLD_FONT"),
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
        str(regular),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("A Japanese bold TrueType font is required. Pass --bold-font.")


def register_fonts(regular_path: Path, bold_path: Path):
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


def style(
    name: str,
    *,
    font: str = JP_REGULAR,
    size: float = 6.8,
    leading: float = 9.0,
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
    "title": style(
        "title",
        font=JP_BOLD,
        size=13.4,
        leading=16.2,
        alignment=TA_CENTER,
        after=2.0,
    ),
    "subtitle": style(
        "subtitle",
        size=6.5,
        leading=8.0,
        alignment=TA_CENTER,
        color=MUTED,
        after=3.2,
    ),
    "author": style(
        "author",
        size=6.7,
        leading=8.2,
        alignment=TA_CENTER,
        after=2.2,
    ),
    "abstract": style("abstract", size=7.1, leading=9.2, alignment=TA_JUSTIFY),
    "keywords": style(
        "keywords",
        size=6.3,
        leading=8.0,
        alignment=TA_LEFT,
        color=MUTED,
        before=2.0,
    ),
    "section": style(
        "section",
        font=JP_BOLD,
        size=9.2,
        leading=11.5,
        alignment=TA_LEFT,
        before=4.0,
        after=1.7,
        keep_with_next=True,
    ),
    "subsection": style(
        "subsection",
        font=JP_BOLD,
        size=7.8,
        leading=9.8,
        alignment=TA_LEFT,
        before=2.5,
        after=0.8,
        keep_with_next=True,
    ),
    "body": style("body", size=7.2, leading=9.8, first_indent=7.2),
    "body0": style("body0", size=7.2, leading=9.8),
    "small": style("small", size=6.6, leading=8.5),
    "caption": style(
        "caption",
        size=5.9,
        leading=7.4,
        alignment=TA_LEFT,
        color=MUTED,
        before=1.2,
        after=2.0,
    ),
    "table": style("table", size=5.7, leading=7.2, alignment=TA_LEFT),
    "table_center": style("table-center", size=5.7, leading=7.2, alignment=TA_CENTER),
    "table_head": style(
        "table-head",
        font=JP_BOLD,
        size=5.7,
        leading=7.2,
        alignment=TA_CENTER,
    ),
    "equation": style(
        "equation",
        size=6.6,
        leading=8.2,
        alignment=TA_CENTER,
    ),
    "eqno": style("eqno", size=6.2, leading=8.2, alignment=TA_CENTER),
    "algorithm": style("algorithm", size=5.8, leading=7.3, alignment=TA_LEFT),
    "reference": style("reference", size=5.65, leading=7.2, alignment=TA_LEFT),
}


def p(text: str, name: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[name])


def equation(text: str, number: int) -> Table:
    table = Table(
        [[p(text, "equation"), p(f"({number})", "eqno")]],
        colWidths=[COLUMN_WIDTH - 9 * mm, 9 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
            ]
        )
    )
    return table


def paper_table(
    rows: list[list[str]],
    widths: list[float],
    caption: str,
    *,
    centered_columns: tuple[int, ...] = (),
) -> KeepTogether:
    cells = []
    for row_index, row in enumerate(rows):
        cells.append(
            [
                p(
                    value,
                    "table_head" if row_index == 0 else ("table_center" if column in centered_columns else "table"),
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
                ("LEFTPADDING", (0, 0), (-1, -1), 1.4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
            ]
        )
    )
    return KeepTogether([p(caption, "caption"), table, Spacer(1, 2.0)])


class PipelineFigure(Flowable):
    """Compact monochrome pipeline sized for a single paper column."""

    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 39 * mm

    def draw(self):
        c = self.canv
        nodes = [
            "dense native基準像 → 4K preflight",
            "VRAM予算 → tile T / halo / phase",
            "Krea2によるhalo付き局所refine",
            "safe残差 + bounded novel-detail候補",
            "2 phase独立証拠 + consensus合成",
            "8K → Smart Finish → 品質gate",
        ]
        box_x = 5 * mm
        box_width = self.width - 10 * mm
        box_height = 4.4 * mm
        top = self.height - 1.5 * mm
        c.setLineWidth(0.45)
        for index, label in enumerate(nodes):
            y = top - (index + 1) * 5.45 * mm
            c.setFillColor(BOX_FILL if index % 2 == 0 else colors.white)
            c.setStrokeColor(RULE)
            c.rect(box_x, y, box_width, box_height, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont(JP_REGULAR, 5.5)
            c.drawCentredString(self.width / 2, y + 1.45 * mm, label)
            if index < len(nodes) - 1:
                arrow_y0 = y - 0.25 * mm
                arrow_y1 = y - 0.82 * mm
                c.setStrokeColor(INK)
                c.line(self.width / 2, arrow_y0, self.width / 2, arrow_y1)
                c.line(self.width / 2, arrow_y1, self.width / 2 - 0.65 * mm, arrow_y1 + 0.65 * mm)
                c.line(self.width / 2, arrow_y1, self.width / 2 + 0.65 * mm, arrow_y1 + 0.65 * mm)


def algorithm_block() -> KeepTogether:
    lines = [
        "<b>Algorithm 1</b>　Krea2 Smart 4K/8K",
        "<b>Input:</b> 基準像 B0, target (W,H), 予算 M, seed s",
        "1:  stages, T, halo, phases ← PLAN(B0,W,H,M)",
        "2:  for stage k do",
        "3:　　B ← RESIZE(B, stage[k]); safe/novel moments ← 0",
        "4:　　for phase q and halo tile i do",
        "5:　　　R_i ← LOCAL-DIFFUSION(B_i, hash(s,k,q,x_i,y_i))",
        "6:　　　Delta_safe ← BASE-ANCHORED-HIGHPASS(R_i,B_i)",
        "7:　　　Delta_novel ← BOUNDED-BANDPASS(R_i,B_i)",
        "8:　　　ACCUMULATE(PHASE-NORMALIZED-WEIGHT, moments)",
        "9:　　novel ← CONSENSUS(novel) if independent evidence ≥ 2",
        "10:　 B ← clip(B + safe + novel);  B ← SMART-FINISH(B)",
        "11: REQUIRE(all tiles, lower/upper detail gates, 100% QA)",
        "<b>Output:</b> exact-size 4K/8K image B + manifest",
    ]
    row = [[p(line, "algorithm")] for line in lines]
    table = Table(row, colWidths=[COLUMN_WIDTH], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_FILL),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
                ("LINEBELOW", (0, 0), (-1, 0), 0.45, INK),
                ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.0),
                ("TOPPADDING", (0, 0), (-1, -1), 0.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8),
            ]
        )
    )
    return KeepTogether([table, Spacer(1, 2.2)])


def draw_page_chrome(c, doc):
    c.saveState()
    c.setStrokeColor(HAIRLINE)
    c.setLineWidth(0.35)
    c.line(MARGIN_X, 9.2 * mm, PAGE_WIDTH - MARGIN_X, 9.2 * mm)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 5.1)
    c.drawString(MARGIN_X, 5.8 * mm, "Krea2 Smart 4K/8K / Technical Short Paper")
    c.drawRightString(PAGE_WIDTH - MARGIN_X, 5.8 * mm, f"{doc.page} / 2")
    if doc.page > 1:
        c.setFont("Helvetica", 4.8)
        c.drawString(MARGIN_X, PAGE_HEIGHT - 6.4 * mm, "KREA2 SMART 4K/8K: VRAM-BOUNDED DETAIL-PRESERVING REFINEMENT")
        c.line(MARGIN_X, PAGE_HEIGHT - 8.0 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 8.0 * mm)
    c.restoreState()


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
        PageTemplate(
            id="later",
            frames=[later_left, later_right],
            onPage=draw_page_chrome,
        ),
    ]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _elapsed_seconds(data: dict) -> float:
    started = datetime.fromisoformat(data["created_at_utc"])
    completed = datetime.fromisoformat(data["completed_at_utc"])
    return (completed - started).total_seconds()


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _qa_by_name(records: list[dict]) -> dict[str, Path]:
    result = {record["name"]: _artifact_path(record["path"]) for record in records}
    missing = [name for name, path in result.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing QA crops: {', '.join(missing)}")
    return result


def load_case(final_manifest_path: Path, preflight_manifest_path: Path) -> dict:
    final_manifest = _read_json(final_manifest_path)
    preflight_manifest = _read_json(preflight_manifest_path)
    if final_manifest.get("status") != "complete_8k":
        raise RuntimeError("The final manifest must have status complete_8k.")
    if preflight_manifest.get("status") != "complete_4k":
        raise RuntimeError("The preflight manifest must have status complete_4k.")
    if final_manifest["base_prompt"] != preflight_manifest["base_prompt"]:
        raise RuntimeError("4K and 8K manifests do not share the exact base prompt.")

    source = final_manifest["steps"]["source"]
    four = preflight_manifest["steps"]["preflight_4k"]
    eight = final_manifest["steps"]["final_8k"]
    if source["sha256"] != four["final"]["sha256"]:
        raise RuntimeError("The 8K source hash does not match the approved 4K result.")
    telemetry = eight["telemetry"]
    if telemetry.get("subprocess_exit_code") != 0 or telemetry.get("sample_count", 0) <= 0:
        raise RuntimeError("8K telemetry must contain successful GPU polling samples.")
    vram = eight["vram_canvas"]
    if (
        vram["processed_tile_count"] != vram["tile_count"]
        or vram["skipped_tile_count"] != 0
    ):
        raise RuntimeError("8K tile completion gate is not satisfied.")

    return {
        "base_prompt": final_manifest["base_prompt"],
        "seed": final_manifest["seed"],
        "checkpoint": final_manifest["backend"]["checkpoint"],
        "source": source,
        "four": four,
        "eight": eight,
        "four_seconds": _elapsed_seconds(preflight_manifest),
        "eight_seconds": _elapsed_seconds(final_manifest),
        "source_crops": _qa_by_name(source["qa_crops_100pct"]),
        "eight_crops": _qa_by_name(eight["qa_crops_100pct"]),
        "telemetry": telemetry,
    }


def comparison_figure(case: dict) -> KeepTogether:
    rows = []
    for label, crop_name in (
        ("顔・瞳・髪", "upper_center"),
        ("衣装・縫製", "lower_center"),
    ):
        rows.append(
            [
                p(f"4K {label}", "table_head"),
                p(f"8K {label}", "table_head"),
            ]
        )
        rows.append(
            [
                RLImage(
                    str(case["source_crops"][crop_name]),
                    width=34.5 * mm,
                    height=34.5 * mm,
                ),
                RLImage(
                    str(case["eight_crops"][crop_name]),
                    width=34.5 * mm,
                    height=34.5 * mm,
                ),
            ]
        )
    table = Table(
        rows,
        colWidths=[COLUMN_WIDTH / 2, COLUMN_WIDTH / 2],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_FILL),
                ("BACKGROUND", (0, 2), (-1, 2), TABLE_FILL),
                ("BOX", (0, 0), (-1, -1), 0.45, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, HAIRLINE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.0),
                ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
            ]
        )
    )
    return KeepTogether(
        [
            table,
            p(
                "Fig. 2.　同一正規化座標から切り出した1024×1024原寸crop。"
                "8K側は各辺2倍のため4K側の半分の画角を示す。紙面上は縮小表示だが、"
                "元PNG・座標・SHA-256をmanifestへ保存する。",
                "caption",
            ),
        ]
    )


def build_story(case: dict) -> list:
    four = case["four"]
    eight = case["eight"]
    four_finish = four["finish_report"]["detail_guard"]
    eight_finish = eight["finish_report"]["detail_guard"]
    four_retention = four["detail_retention_vs_source"]
    eight_retention = eight["detail_retention_vs_source"]
    stage = eight["vram_canvas"]["stage_reports"][-1]
    consensus = stage["consensus_stats"]
    source_band = case["source"]["analysis_metrics"]["normalized_multiband"]
    eight_band = eight["analysis_metrics"]["normalized_multiband"]
    micro_ratio = (
        eight_band["sigma_0_1"]["abs_p95"]
        / max(source_band["sigma_0_1"]["abs_p95"], 1e-6)
    )

    four_size = "×".join(str(value) for value in four["final"]["size"])
    eight_size = "×".join(str(value) for value in eight["final"]["size"])
    four_tiles = four["vram_canvas"]["tile_count"]
    eight_tiles = eight["vram_canvas"]["tile_count"]
    telemetry = case["telemetry"]
    eight_finish_text = (
        f'{eight_finish["detail_energy_ratio"]:.3f}×'
        if eight_finish.get("applied")
        else "1.000× (安全no-op)"
    )

    story = [
        p(
            "Krea2 Smart 4K/8K：独立位相合意による<br/>高密度・高解像度リファイン",
            "title",
        ),
        p(
            "Krea2 Smart 4K/8K: Dense High-Resolution Refinement with Independent-Phase Consensus",
            "subtitle",
        ),
        p("AiWithYou　　Algorithm Note / Single-case Validation　　2026年7月13日", "author"),
        HRFlowable(
            width="100%",
            thickness=0.55,
            color=INK,
            spaceBefore=0,
            spaceAfter=3.0,
        ),
        p(
            "<b>要旨 - </b>本稿はKrea2と24 GiB GPUで、4Kを主成果、8Kを承認済み4Kの"
            "条件付き2倍拡張として生成するtraining-free手法を示す。密描写promptで基準像の情報量を先に確保し、"
            "halo付きtileから、既存細部を守るsafe残差と、平坦域にも限定的な新規描線を許す2–8 px帯域残差を分離する。"
            "後者は2つのshift位相の独立証拠と低分散合意がある画素だけ採用する。指定prompt・seedの単一事例で"
            f"{four_size}と{eight_size}を全tile完了・skip 0で生成し、原寸cropを保存した。"
            "比較優位や意味的一致の一般保証ではなく、実装と単一事例の検証である。",
            "abstract",
        ),
        p(
            "<b>キーワード - </b>Krea2、4K/8K、VRAM制約、band-pass residual、phase consensus、100% crop",
            "keywords",
        ),
        FrameBreak,
        p("1　目的と設計原則", "section"),
        p(
            "目的は、顔・瞳・角・髪流れ・スライム形状を保ちながら、zoom時に読める髪束、虹彩、角の面、"
            "レース、縫い目、透明材質の内部表現を増やすことである。高画素化だけでは情報は増えず、強いdenoiseは"
            "同一性を壊すため、4Kで必ず目視し、8Kはその各辺を正確に2倍する。",
            "body0",
        ),
        p(
            "生成tileそのものは貼らず、局所候補と基準cropの周波数差だけをCPU上のcanvasへ蓄積する。"
            "GPUへ載る空間項はcanvas全体ではなく1024角payloadで決まり、全画面統計はdisk memmapへ退避する。",
        ),
        p("1.1　貢献", "subsection"),
        paper_table(
            [
                ["機構", "役割", "fail-closed条件"],
                ["dense native prompt", "元から髪・瞳・衣装・slimeを描く", "指定tagをprefixで厳密保持"],
                ["safe residual", "既存細部を形状に沿って保存", "低周波不一致・平坦域を抑制"],
                ["novel residual", "不足する2–8 px描線を限定追加", "2 phase証拠・±6/8 level"],
                ["consensus", "位置ずれ・偶発noiseを減衰", "分散gate＋coverage≥1.5"],
                ["4K→8K", "費用の高い8Kを条件付き実行", "3840–4096入力の正確な2倍"],
                ["QA/telemetry", "zoom品質と資源を再現可能に記録", "全tile、上下detail比、原寸crop"],
            ],
            [23 * mm, 31 * mm, COLUMN_WIDTH - 54 * mm],
            "Table 1.　実装した防御層。",
        ),
        PipelineFigure(COLUMN_WIDTH),
        p("Fig. 1.　4Kを品質境界とする処理系。", "caption"),
        FrameBreak,
        p("2　提案アルゴリズム", "section"),
        p(
            "基準cropをB、Krea2候補をR、低域をL、高域をH、2–8 px帯域をBPとする。"
            "低周波構造整合a∈[0,1]と基準細部energy eBを共通に用いる。",
            "body0",
        ),
        equation(
            "Δ<sub>safe</sub>=clip(γa·e<sub>B</sub>/(e<sub>B</sub>+τ)[H(R)-H(B)], −d<sub>s</sub>, d<sub>s</sub>)",
            1,
        ),
        p(
            "safe枝は基準に既存する細部だけを強める。これだけでは平坦な髪面・衣装面へ描き込みを増やせないため、"
            "相補gate c=τ/(eB+τ)でnovel枝を設ける。",
        ),
        equation(
            "Δ<sub>novel</sub>=clip(ηa·c[BP<sub>2–8</sub>(R)-BP<sub>2–8</sub>(B)], −d<sub>n</sub>, d<sub>n</sub>)",
            2,
        ),
        p(
            "novel枝は輝度のみ、既定±8（4K）/±6（8K）levelに制限する。shiftした2位相を個別正規化し、"
            "十分統計S0,S1,S2から平均μ、二次moment E2、候補間分散Vを求める。",
        ),
        equation(
            "g=exp[−λV/(E<sub>2</sub>+κ²)],　B′=clip(B+g<sub>s</sub>μ<sub>s</sub>+1[S<sub>0</sub>≥1.5]g<sub>n</sub>μ<sub>n</sub>)",
            3,
        ),
        p(
            "一致する微細線は残し、位相間で符号・位置が揺れるtextureを減衰する。"
            "ただし全候補が同じ誤りを出す場合は検出できないため、原寸目視を機械gateの代用にしない。",
        ),
        algorithm_block(),
        paper_table(
            [
                ["profile", "phase", "denoise", "novel gain / cap"],
                ["Structure Safe", "1", ".12→.08", "0 / —"],
                ["Dense Detail 4K", "2", ".16→.13", "1.0 / ±8"],
                ["Dense Detail 8K", "2", ".12→.11", "0.8 / ±6"],
            ],
            [26 * mm, 10 * mm, 18 * mm, COLUMN_WIDTH - 54 * mm],
            "Table 2.　Forge Neo GUIとCLIの共有profile。",
            centered_columns=(1, 2, 3),
        ),
        NextPageTemplate("later"),
        PageBreak(),
        p("3　Forge Neo統合と実験条件", "section"),
        p(
            "img2imgのVRAM-Canvas GUIへ4K/8K Smart button、profile自動適用、novel gain/cap、"
            "Krea2 checkpoint検査、8K入力guard、seed/prompt状態復元、中断時の未完成tile拒否を統合した。"
            "CLIはGPUを1秒間隔で監視し、commandをprompt非表示でJSONへ保存する。",
            "body0",
        ),
        p(
            '<b>GPU:</b> RTX 3090 24 GiB　　<b>checkpoint:</b> Krea2 Turbo NF4 '
            '(exact filename in manifest)<br/>'
            f'<b>seed:</b> {case["seed"]}　　<b>base prompt:</b><br/>{case["base_prompt"]}',
            "reference",
        ),
        paper_table(
            [
                ["段", "寸法", "tile", "wall time", "VRAM"],
                [
                    "4K",
                    four_size,
                    f"{four_tiles}/{four_tiles}",
                    f'{case["four_seconds"]:.1f} s',
                    "連続計測なし",
                ],
                [
                    "8K",
                    eight_size,
                    f"{eight_tiles}/{eight_tiles}",
                    f'{case["eight_seconds"]:.1f} s',
                    f'{telemetry["max_memory_used_mib"]:,} MiB*',
                ],
            ],
            [9 * mm, 20 * mm, 15 * mm, 20 * mm, COLUMN_WIDTH - 64 * mm],
            "Table 3.　実行結果。*1秒pollingの観測最大で真の瞬間peak保証ではない。",
            centered_columns=(0, 1, 2, 3, 4),
        ),
        paper_table(
            [
                ["gate / metric", "4K", "8K"],
                ["gradient p95 retention", f'{four_retention["gradient_p95"]:.3f}', f'{eight_retention["gradient_p95"]:.3f}'],
                ["high-pass p95 retention", f'{four_retention["highpass_abs_p95"]:.3f}', f'{eight_retention["highpass_abs_p95"]:.3f}'],
                ["4096基準 σ0–1 p95 ratio", "—", f"{micro_ratio:.3f}"],
                ["Smart Finish energy", f'{four_finish["detail_energy_ratio"]:.3f}×', eight_finish_text],
                ["flat-region edits", f'{four_finish["flat_region_changed_pixels"]} px', f'{eight_finish["flat_region_changed_pixels"]} px'],
                ["clipped channel fraction", f'{four_finish["clipped_channel_fraction"]:.6f}', f'{eight_finish["clipped_channel_fraction"]:.6f}'],
            ],
            [36 * mm, 20 * mm, COLUMN_WIDTH - 56 * mm],
            "Table 4.　下限だけでなく1.8倍の上限を設け、noise/oversharpeningも拒否する。",
            centered_columns=(1, 2),
        ),
        p(
            f'8K telemetryは{telemetry["sample_count"]} samples、GPU利用率最大'
            f'{telemetry["max_utilization_percent"]}%、温度最大{telemetry["max_temperature_c"]} ℃。'
            f'novel枝はcoverage {consensus["novel_evidence_percent"]:.1f}%、'
            f'平均consensus {consensus["mean_novel_consensus_gate"]:.3f}、'
            f'採用残差の平均絶対値 {consensus["mean_abs_novel_residual"]:.3f} levelだった。',
            "body0",
        ),
        p(
            f'<b>4K SHA-256:</b> {four["final"]["sha256"][:16]}…<br/>'
            f'<b>8K SHA-256:</b> {eight["final"]["sha256"][:16]}…',
            "reference",
        ),
        FrameBreak,
        p("4　原寸QA", "section"),
        comparison_figure(case),
        p(
            "対応cropを目視し、顔・両目・角数、髪の大流れ、黒衣装のlace/seam、緑slimeの外形・透明感を維持し、"
            "明瞭なtile seam、二重輪郭、人物・角の増殖を認めなかった。8Kでは4K由来の構造を保ったまま、"
            "局所線が細分化している。ただし、この判断は本1事例の人手観察である。",
            "body0",
        ),
        p("5　限界", "section"),
        p(
            "単一prompt・seed・portraitの検証であり、他の画風、手指、文字、写実顔、一般的な比較優位を示さない。"
            "高周波metricは知覚品質や意味的一致の代理ではなく、noiseも上昇させ得るため上限gateと目視を併用した。"
            "consensusは共有誤りを検出できず、8Kは4Kより時間・熱・disk I/Oが大きい。4Kを既定の納品点とし、"
            "8Kは必要時のみ選ぶ。",
            "body0",
        ),
        p("6　結論", "section"),
        p(
            "密描写をnative段で確保し、base-anchored safe枝とbounded novel枝を分離し、2位相合意で後者を"
            "fail-closedに採用する手法をForge Neoへ実装した。指定条件で4K/8Kを全tile完走し、数値gate、"
            "実測telemetry、原寸crop、hashを一体で保存した。",
            "body0",
        ),
        p("参考文献", "section"),
        p(
            "[1] O. Bar-Tal et al., “MultiDiffusion,” ICML, 2023.<br/>"
            "[2] R. Du et al., “DemoFusion,” CVPR, 2024.<br/>"
            "[3] R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022.<br/>"
            "[4] S. Lee et al., “Krea 2 Technical Report,” Krea, 2026.",
            "reference",
        ),
    ]
    return story

def build_pdf(output: Path, font_path: Path, bold_font_path: Path, case: dict):
    register_fonts(font_path, bold_font_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(output),
        pagesize=JIS_B5,
        leftMargin=0,
        rightMargin=0,
        topMargin=0,
        bottomMargin=0,
        pageTemplates=page_templates(),
        title="Krea2 Smart 4K/8K: 独立位相合意による高密度・高解像度リファイン",
        author="AiWithYou",
        subject="Krea2向けVRAM制約4K/8K pipelineのアルゴリズム短報と単一事例検証",
    )
    doc.build(build_story(case))
    reader = PdfReader(str(output))
    page_count = len(reader.pages)
    if page_count != 2:
        raise RuntimeError(f"Expected exactly 2 JIS B5 pages, generated {page_count}.")
    expected = tuple(value / mm for value in JIS_B5)
    for index, page in enumerate(reader.pages, start=1):
        actual = (
            float(page.mediabox.width) / mm,
            float(page.mediabox.height) / mm,
        )
        if any(abs(got - want) > 0.05 for got, want in zip(actual, expected)):
            raise RuntimeError(
                f"Page {index} is {actual[0]:.2f}x{actual[1]:.2f} mm; "
                f"expected JIS B5 {expected[0]:.2f}x{expected[1]:.2f} mm."
            )
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    required_text = (
        "Krea2 Smart 4K/8K",
        "225/225",
        "light blue hair,long_wavy_hair",
        "green_slime",
        "Expressionless",
        case["eight"]["final"]["sha256"][:12],
    )
    missing = [value for value in required_text if value not in extracted]
    if missing:
        raise RuntimeError(
            "Generated PDF text extraction is missing required evidence: "
            + ", ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/pdf/vram_canvas_b5_ja.pdf")
    parser.add_argument(
        "--run-manifest",
        required=True,
        help="Completed Krea2 Smart 8K manifest with telemetry and 100% QA crops.",
    )
    parser.add_argument(
        "--preflight-manifest",
        required=True,
        help="Completed Krea2 Smart 4K manifest used as the approved 8K source.",
    )
    parser.add_argument(
        "--repo-copy",
        default="docs/vram_canvas_b5_ja.pdf",
        help="Optional second copy committed with the implementation; empty disables.",
    )
    parser.add_argument("--font", default=None, help="Japanese TTF/TTC; Noto Sans JP is recommended.")
    parser.add_argument("--bold-font", default=None, help="Japanese bold TTF/TTC.")
    args = parser.parse_args()
    output = Path(args.output)
    case = load_case(Path(args.run_manifest), Path(args.preflight_manifest))
    regular_font = find_japanese_font(args.font)
    build_pdf(
        output,
        regular_font,
        find_japanese_bold_font(regular_font, args.bold_font),
        case,
    )
    if args.repo_copy:
        repo_copy = Path(args.repo_copy)
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, repo_copy)
    sys.stdout.write(f"{output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
