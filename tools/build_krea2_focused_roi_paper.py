"""Build the measured two-page Japanese B5 Focused ROI Rewrite paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw
from pypdf import PdfReader
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Spacer,
)

import build_krea2_local_supersample_paper as base


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "output/krea2_smart4k_snow_station/focused_face_rewrite_1536/"
    "local_supersample_20260714_122916_778319"
)
DEFAULT_MANIFEST = RUN_DIR / "experiment_manifest.json"
DEFAULT_VALIDATION = RUN_DIR / "focused_roi_validation.json"
DEFAULT_COMPARISON = RUN_DIR / "focused_face_comparison.png"
DEFAULT_OUTPUT = ROOT / "output/pdf/krea2_focused_roi_rewrite_b5_ja.pdf"
DEFAULT_REPO_COPY = ROOT / "docs/krea2_focused_roi_rewrite_b5_ja.pdf"
DEFAULT_ASSETS = ROOT / "output/pdf/krea2_focused_roi_rewrite_assets"
EXPECTED_OUTPUT_FILE_SHA256 = (
    "47EE4FAA7E42A277FC6581BE91F6B9413CEBDDBE3ABE42FCF52574EA2428F0D7"
)
EXPECTED_OUTPUT_PIXEL_SHA256 = (
    "618cfba205a005d3251f2dc6ab3f447e4cf349b568b62ae2d8bf7fc050c3ca75"
)
EXPECTED_PROMPT_SHA256 = (
    "F3D288BFA6AB2B6533EB187F903257F9E30CD632728F2141BBC3D967493D7BE0"
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_case(manifest_path: Path, validation_path: Path, comparison_path: Path) -> dict:
    manifest = read_json(manifest_path)
    validation = read_json(validation_path)
    embedded = manifest["embedded_krea2_local_supersample"]
    tile = embedded["tiles"][0]
    input_path = Path(manifest["input"]["path"])
    output_path = Path(manifest["output"]["path"])
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()

    checks = {
        "focused rewrite": embedded["focused_rewrite"] is True,
        "one focused region": embedded["focused_region_count"] == 1,
        "selected accepted candidate": tile["selected_candidate"] == 2
        and tile["quality_gate_override_reason"] is None,
        "5.12x zoom": abs(float(tile["effective_zoom"]) - 5.12) < 1e-9,
        "outside unchanged": validation["changed_pixels_outside_roi"] == 0,
        "inside changed": validation["changed_pixels_inside_roi"] == 16646,
        "prompt hash": manifest["request"]["base_prompt_sha256"] == EXPECTED_PROMPT_SHA256,
        "pixel hash": manifest["output"]["pixel_sha256"] == EXPECTED_OUTPUT_PIXEL_SHA256,
        "file hash": manifest["output"]["file_sha256"] == EXPECTED_OUTPUT_FILE_SHA256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Focused paper evidence check failed: " + ", ".join(failed))
    if sha256_file(output_path) != EXPECTED_OUTPUT_FILE_SHA256:
        raise RuntimeError("Output file bytes do not match the measured manifest")
    for path in (input_path, output_path, comparison_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "manifest": manifest,
        "validation": validation,
        "embedded": embedded,
        "tile": tile,
        "input_path": input_path,
        "output_path": output_path,
        "comparison_path": comparison_path,
    }


def make_locator(case: dict, output: Path, regular_font: Path, bold_font: Path) -> Path:
    source = Image.open(case["input_path"]).convert("RGB")
    display = source.resize((1600, round(1600 * source.height / source.width)), Image.Resampling.LANCZOS)
    top = 88
    canvas = Image.new("RGB", (display.width, display.height + top), (244, 246, 249))
    canvas.paste(display, (0, top))
    draw = ImageDraw.Draw(canvas)
    title = base.pil_font(bold_font, 40)
    label = base.pil_font(regular_font, 28)
    draw.text((35, 19), "Focused target and generation context", font=title, fill=(20, 23, 29))
    sx = display.width / source.width
    sy = display.height / source.height
    target = case["tile"]["core"]
    context = case["tile"]["payload_box"]

    def scaled_box(box: list[int]) -> tuple[int, int, int, int]:
        left, y0, right, y1 = box
        return (
            round(left * sx),
            round(top + y0 * sy),
            round(right * sx),
            round(top + y1 * sy),
        )

    draw.rectangle(scaled_box(context), outline=(20, 196, 214), width=8)
    draw.rectangle(scaled_box(target), outline=(239, 44, 103), width=10)
    legend_y = top + 20
    draw.rectangle((34, legend_y, 70, legend_y + 22), outline=(20, 196, 214), width=5)
    draw.text((82, legend_y - 6), "300×300 context → 1536×1536", font=label, fill=(25, 30, 38))
    draw.rectangle((34, legend_y + 42, 70, legend_y + 64), outline=(239, 44, 103), width=5)
    draw.text((82, legend_y + 36), "120×150 writable target", font=label, fill=(25, 30, 38))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return output


class FocusedPipelineFigure(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 32 * mm

    def draw(self) -> None:
        canvas = self.canv
        canvas.setLineWidth(0.45)
        canvas.setStrokeColor(base.RULE)
        canvas.setFont(base.JP_REGULAR, 5.1)
        nodes = (
            ("tight target T\n120×150", 1, 21, 22),
            ("context B\n300×300", 27, 21, 22),
            ("Lanczos U(B)\n1536×1536", 53, 21, 23),
            ("Krea2 K(U)\n6 steps / .38", 53, 10, 23),
            ("same D\nC1 − C0", 27, 10, 22),
            ("inward mask\nwrite T only", 1, 10, 22),
            ("4096×1756 output; exterior = source uint8", 14, 0.5, 48),
        )
        for label, x, y, width in nodes:
            canvas.setFillColor(base.BOX_FILL)
            canvas.rect(x * mm, y * mm, width * mm, 7.2 * mm, fill=1, stroke=1)
            canvas.setFillColor(base.INK)
            lines = label.split("\n")
            for index, line in enumerate(lines):
                canvas.drawCentredString(
                    (x + width / 2) * mm,
                    (y + 4.5 - index * 2.4) * mm,
                    line,
                )
        canvas.setStrokeColor(base.INK)
        for x1, y1, x2, y2 in (
            (23, 24.6, 27, 24.6),
            (49, 24.6, 53, 24.6),
            (64.5, 21, 64.5, 17.2),
            (53, 13.6, 49, 13.6),
            (27, 13.6, 23, 13.6),
            (12, 10, 28, 7.7),
        ):
            canvas.line(x1 * mm, y1 * mm, x2 * mm, y2 * mm)


def draw_page_chrome(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(base.HAIRLINE)
    canvas.setLineWidth(0.35)
    canvas.line(base.MARGIN_X, 9.2 * mm, base.PAGE_WIDTH - base.MARGIN_X, 9.2 * mm)
    canvas.setFillColor(base.MUTED)
    canvas.setFont("Helvetica", 5.0)
    canvas.drawString(
        base.MARGIN_X,
        5.8 * mm,
        "Krea2 Focused ROI Rewrite / Measured Technical Short Paper",
    )
    canvas.drawRightString(
        base.PAGE_WIDTH - base.MARGIN_X,
        5.8 * mm,
        f"{doc.page} / 2",
    )
    if doc.page > 1:
        canvas.setFont("Helvetica", 4.7)
        canvas.drawString(
            base.MARGIN_X,
            base.PAGE_HEIGHT - 6.4 * mm,
            "CONTEXT-MAGNIFIED SINGLE-ROI FACE REGENERATION",
        )
        canvas.line(
            base.MARGIN_X,
            base.PAGE_HEIGHT - 8.0 * mm,
            base.PAGE_WIDTH - base.MARGIN_X,
            base.PAGE_HEIGHT - 8.0 * mm,
        )
    canvas.restoreState()


def page_templates() -> list[PageTemplate]:
    header_height = 45 * mm
    header_bottom = base.PAGE_HEIGHT - base.TOP - header_height
    first_column_top = header_bottom - 3 * mm
    first_column_height = first_column_top - base.BOTTOM
    header = Frame(
        base.MARGIN_X,
        header_bottom,
        base.CONTENT_WIDTH,
        header_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first-header",
    )
    first_left = Frame(
        base.MARGIN_X,
        base.BOTTOM,
        base.COLUMN_WIDTH,
        first_column_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first-left",
    )
    first_right = Frame(
        base.MARGIN_X + base.COLUMN_WIDTH + base.GUTTER,
        base.BOTTOM,
        base.COLUMN_WIDTH,
        first_column_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first-right",
    )
    later_height = base.PAGE_HEIGHT - 11 * mm - base.BOTTOM
    later_left = Frame(
        base.MARGIN_X,
        base.BOTTOM,
        base.COLUMN_WIDTH,
        later_height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="later-left",
    )
    later_right = Frame(
        base.MARGIN_X + base.COLUMN_WIDTH + base.GUTTER,
        base.BOTTOM,
        base.COLUMN_WIDTH,
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


def build_story(case: dict, locator: Path) -> list:
    manifest = case["manifest"]
    validation = case["validation"]
    telemetry = manifest["telemetry"]
    candidate = case["tile"]["candidate_metrics"][1]
    title_block = [
        base.p("文脈拡大・単一領域再生成による4K顔局所再描画", "title"),
        base.p(
            "Context-Magnified Single-ROI Regeneration for Focused Face Rewriting in 4K Images",
            "subtitle",
        ),
        base.p("AiWithYou　—　Technical Short Paper / 2026-07-14", "author"),
        HRFlowable(width="100%", thickness=0.45, color=base.RULE, spaceAfter=1.8),
        base.p(
            "<b>要旨—</b> 固定tileで分断されていた小さな顔を、tight targetと周辺contextへ分離し、顔全体を1枚の1536入力へ"
            "5.12倍拡大してKrea2で再生成するFocused ROI Rewriteを実装した。同じlinear-light縮小を通すC1−C0をtarget内だけへ"
            "20 px inward featherで合成する。4096×1756画像で120×150 targetを評価し、detail gate合格候補2を採用、target内"
            "16,646画素（92.4778%）を変更、target外0画素、RGB差mean 9.007 / p95 26 / max 77、49.1 s、peak 20,957 MiBを"
            "得た。単一事例で知覚品質の一般性は示さないが、実拡大再生成と領域外bit-exact保護を同時に実証した。",
            "abstract",
        ),
        base.p(
            "Keywords: Krea2, focused regeneration, ROI, context crop, round-trip compensation, 4K, RTX 3090",
            "keywords",
        ),
        FrameBreak(),
    ]
    page1_left = [
        base.p("1. 問題設定", "section"),
        base.p(
            "固定payloadを1536へ拡大しても、顔が複数tileへ掛かれば左右の目・輪郭・髪は別seedで処理される。これは顔全体の"
            "再生成ではない。さらに低denoiseの保守的detail残差は元画像を守る一方、顔形状をほぼ変えない。本研究は「1顔を1生成"
            "単位にする」「実際の再描画を採用する」「target外を一画素も変えない」を同時に満たす。",
        ),
        base.p("2. 手法", "section"),
        FocusedPipelineFigure(base.COLUMN_WIDTH),
        base.p(
            "target Tの長辺LとContext Scale kから、正方形context辺sB=ceil(kL)を得る。Tは唯一の書き込み領域、Bは周辺の髪・頭・"
            "照明を含む生成文脈である。targetごとに1つのBを作るため、1顔を複数sampleへ分断しない。",
            "body0",
        ),
        base.p(
            "C0 = D(U(B))　 C1,i = D(K(U(B); zi))　 Δi = C1,i − C0",
            "equation",
        ),
        base.p(
            "UはsRGB Lanczos拡大、KはKrea2 img2img、Dは共通のlinear-light area縮小。同じDでround-trip誤差を相殺する。"
            "Focusedでは選択候補のフルΔを使うが、高解像度候補の直接貼付や別resizerは使わない。",
        ),
        base.p("2.1 候補と限定書き戻し", "subsection"),
        base.p(
            "2候補を別seedで生成する。合格候補があればその中のquality score最小を、全件不合格なら全体の最小を選ぶ。平均はしない。"
            "target境界で0、内側feather後に1となるsmoothstepをΔへ掛け、target外は元uint8を直接コピーする。",
        ),
        base.callout(
            "<b>核心:</b> 生成contextと書き込みtargetを分離する。300×300全体を1536へ拡大して顔を一度に生成するが、"
            "書き戻すのは120×150だけである。",
            accent=True,
        ),
        base.p("2.2 Algorithm 1", "subsection"),
        base.paper_table(
            [
                ["Step", "Focused single-context rewrite"],
                ["1", "target Tを検証し、中心・長辺Lを得る"],
                ["2", "一辺ceil(kL)の正方形context Bを計画"],
                ["3", "Bを1536へ拡大し、別seedの2候補を生成"],
                ["4", "C0と各C1を同じlinear-light Dで縮小"],
                ["5", "合格候補を優先し、平均せず1候補を選択"],
                ["6", "full C1−C0をTでcropし、内向きmaskを適用"],
                ["7", "T外をsource uint8のまま最終化・hash記録"],
            ],
            [9 * mm, base.COLUMN_WIDTH - 9 * mm],
        ),
        base.p(
            "不変条件は、(i) 1 target = 1 sampling context、(ii) C0/C1の縮小演算一致、(iii) 合成weight非負・finite、"
            "(iv) target外をaccumulatorへ通さない、の4点である。",
        ),
        FrameBreak(),
    ]
    page1_right = [
        base.p("3. 実装", "section"),
        base.p(
            "Forge img2img ScriptへFocused ROI Rewrite modeとFocused Face Rewrite 1536 profileを追加した。profileは6 steps、"
            "denoise .38、2候補、context 2.0×、source側20 px feather。Krea2/Qwen実体、target非重複、context<1536、disk、"
            "tile上限をmodel処理前に検査する。",
        ),
        base.paper_table(
            [
                ["設定", "値"],
                ["canvas", "4096×1756"],
                ["target", "(2608,635,2728,785) / 120×150"],
                ["context", "(2518,560,2818,860) / 300×300"],
                ["process", "1536×1536 / 5.12×"],
                ["Krea2", "6 steps / d=.38 / 2 candidates"],
            ],
            [22 * mm, base.COLUMN_WIDTH - 22 * mm],
            "表1　実験条件。global seed 2846268111。",
        ),
        base.scaled_image(locator, base.COLUMN_WIDTH),
        base.p(
            "図1　水色は生成context、赤は唯一の書き込みtarget。顔はcontext中央にありtile境界へ掛からない。",
            "caption",
        ),
        base.scaled_image(case["comparison_path"], base.COLUMN_WIDTH),
        base.p(
            "図2　元4K顔、選択された1536候補、4K書き戻しの同一target比較。候補2はdetail gate合格。",
            "caption",
        ),
        base.p("3.1 測定と再現性", "subsection"),
        base.p(
            "入力・出力のdecoded RGB、target内外の差、candidate統計、PNG text chunk、file/pixel SHA-256を保存した。GPU memory、"
            "利用率、温度、電力はnvidia-smiで1秒間隔に取得したGPU全体値で、プロセス専有値や瞬間ピーク保証ではない。base promptは"
            "元4K manifestの駅・雪・白髪人物を記述した文字列を使い、prompt SHAはF3D288BFA6AB…である。",
        ),
        base.callout(
            "<b>Preflight:</b> targetなし、重複target、context辺≥1536、誤model/Qwen構成、disk不足、tile上限超過は"
            "最初のcandidateより前に明示失敗する。",
        ),
        PageBreak(),
    ]
    page2_left = [
        base.p("4. 結果", "section"),
        base.paper_table(
            [
                ["指標", "実測"],
                ["選択候補", "2 / accepted"],
                ["target内変更", "16,646 / 18,000 (92.4778%)"],
                ["target外変更", "0 px"],
                ["RGB |Δ|", "mean 9.007 / p95 26 / max 77"],
                ["時間", f"{manifest['duration_seconds']:.1f} s"],
                ["peak VRAM", f"{telemetry['peak_memory_used_mib']:,} MiB"],
            ],
            [25 * mm, base.COLUMN_WIDTH - 25 * mm],
            "表2　decoded RGB差分とGPU実測。VRAMはGPU全体の1秒標本。",
        ),
        base.p(
            f"候補2のdetail-energy増分は{candidate['detail_increase']:+.3f} code相当で品質gateを通過した。候補1は負で不合格のため、"
            "修正後の選択則は候補2を採用した。PNG metadataはselected_candidate=2、quality overrideなし、effective_zoom=5.12を記録する。",
        ),
        base.paper_table(
            [
                ["候補", "detail Δ", "score", "gate", "選択"],
                ["1", "−0.568", "35.314", "fail", "-"],
                ["2", "+0.503", "38.143", "pass", "yes"],
            ],
            [9 * mm, 16 * mm, 15 * mm, 13 * mm, base.COLUMN_WIDTH - 53 * mm],
            "表3　同一contextの2候補。score比較は合格集合内で行う。",
            centered_columns=(0, 1, 2, 3, 4),
        ),
        base.callout(
            "<b>領域契約:</b> target内16,646画素が変化した一方、target外は0画素。出力寸法は入力と同じ4096×1756。"
            "これは「処理を実行した」だけでなく「再生成結果を反映した」ことを差分で示す。",
            accent=True,
        ),
        base.p("4.1 視覚観察", "subsection"),
        base.p(
            "元4Kで潰れていた両目、虹彩、上まつ毛、鼻口、顎線が1536候補で再構成された。4Kへ縮小・feather後も異なる線が残り、"
            "単なるLanczos拡大ではない。contextで背景・衣装も生成されるが、それらはtarget外なので最終4Kへ書き込まれない。",
        ),
        base.p("4.2 境界と不変性", "subsection"),
        base.p(
            "source側20 pxのinward smoothstepはtarget境界で差分を0へ収束させる。Focusedではweight分母を1に固定し、通常tileの"
            "normalizerでfeatherが相殺されることを防ぐ。外側はlinear RGBへのdecode/encodeも行わず、元uint8を直接返すため、"
            "差分検証でtarget外0画素になった。",
        ),
        base.p("5. 考察", "section"),
        base.p(
            "旧方式の問題はprocess edgeの小ささではなく、意味単位と生成単位の不一致だった。Focusedはtargetとcontextを分け、顔全体を"
            "1回のsamplingへ収める。inward featherは矩形境界を抑えるが、targetが小さすぎると全強度領域も狭くなるため、前髪・顎まで"
            "含むtight boxを起点とする。",
        ),
        base.p("5.1 旧固定tileとの相違", "subsection"),
        base.paper_table(
            [
                ["観点", "固定tile detail", "Focused rewrite"],
                ["生成単位", "512 payloadごと", "1 target = 1 context"],
                ["顔分断", "境界次第で発生", "target単位では発生しない"],
                ["denoise", ".10-.15", ".38"],
                ["採用差", "band-limited", "full C1−C0"],
                ["目的", "微細残差", "意味単位の再描画"],
            ],
            [17 * mm, 29 * mm, base.COLUMN_WIDTH - 46 * mm],
            centered_columns=(0,),
        ),
        FrameBreak(),
    ]
    page2_right = [
        base.p("5.1 再生成と超解像の区別", "subsection"),
        base.p(
            "本手法はground truthの未知細部を復元する決定論的超解像ではなく、Krea2による条件付き再描画である。identity、表情、目形状、"
            "明度が変わり得る。quality scoreも知覚的顔品質を直接測らないため、高解像度候補と4K書き戻しの原寸確認が必要である。",
        ),
        base.p("6. 安全性と失敗時動作", "section"),
        base.p(
            "重複target、targetなし、context辺がprocess edge以上、誤model、OOMを明示失敗にする。処理中はRestore faces、Tiling、mask、"
            "内部保存をOFFにし、成功・例外・中断・skip・stop・OOMでprocessing stateを復元する。途中candidateやpartial canvasを最終"
            "画像として返さない。",
        ),
        base.p("6.1 GUI運用", "subsection"),
        base.p(
            "(1) 承認済み4Kと正確なbase promptを通常img2imgへ設定する。(2) Focused ROI Rewrite / Focused Face Rewrite 1536を選ぶ。"
            "(3) 1顔を前髪・顎まで囲むtight boxで指定する。(4) Save QA cropsを有効にして生成する。(5) high-resolution candidate、"
            "before/after payload、4K原寸、effective zoom、target外差を確認する。",
        ),
        base.p(
            "Context Scale 2.0は出発点であり、背景が支配的なら下げ、頭部の文脈が不足するなら上げる。ただしcontextが大きいほど"
            "実効拡大率は下がる。featherは矩形境界を隠す一方、targetが小さいと再描画の有効面積も減らすため、box寸法と同時に調整する。",
        ),
        base.paper_table(
            [
                ["証跡", "値"],
                ["prompt SHA", "F3D288BFA6AB…"],
                ["input pixel", "78d02e584a4a…"],
                ["output pixel", "618cfba205a0…"],
                ["output file", "47EE4FAA7E42…"],
                ["outside diff", str(validation["changed_pixels_outside_roi"])],
            ],
            [24 * mm, base.COLUMN_WIDTH - 24 * mm],
            "表3　再現性・不変性の証跡。",
        ),
        base.p("7. 限界", "section"),
        base.p(
            "画像1枚、target 1個、prompt 1件、seed系列1件のcase studyである。ground truth、他方式との盲検比較、複数評価者、identity"
            "embedding距離を含まない。denoise .38、context 2.0×、feather 20 pxの一般最適性は未検証で、別seedでは形状変化や平滑化が"
            "生じ得る。",
        ),
        base.p(
            "quality gate通過はdetail-energy等の数値条件であり、同一性や美的品質の保証ではない。複数顔、画像端の極小顔、写真、"
            "文字と重なる顔、異なるKrea2 checkpoint、24GB未満のGPUは未評価である。",
        ),
        base.p("8. 結論", "section"),
        base.p(
            "顔targetと生成contextを分離し、顔全体を1536へ5.12倍拡大して再生成し、round-trip補償後の差分をtarget内だけへ戻した。"
            "detail gate合格候補を採用してtarget内92.48%を変更しながら、target外を0画素変更に保った。固定tileの形式的拡大から、"
            "意味単位を保つ局所再生成へ移行した。",
        ),
        base.p("参考文献", "section"),
        base.p(
            "[1] S. Lee et al., “Krea 2 Technical Report,” Krea, 2026.<br/>"
            "[2] R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022.<br/>"
            "[3] O. Bar-Tal et al., “MultiDiffusion,” ICML, 2023.",
            "reference",
        ),
        Spacer(1, 1.5),
        base.p(
            "<b>Reproduction</b><br/>runner: tools/run_krea2_local_supersample_experiment.py<br/>"
            "manifest: local_supersample_20260714_122916_778319/experiment_manifest.json",
            "reference",
        ),
    ]
    return title_block + page1_left + page1_right + page2_left + page2_right


def build_pdf(
    output: Path,
    repo_copy: Path,
    case: dict,
    locator: Path,
    regular_font: Path,
    bold_font: Path,
) -> None:
    base.register_fonts(regular_font, bold_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(output),
        pagesize=base.JIS_B5,
        leftMargin=0,
        rightMargin=0,
        topMargin=0,
        bottomMargin=0,
        pageTemplates=page_templates(),
        title="文脈拡大・単一領域再生成による4K顔局所再描画",
        author="AiWithYou",
        subject="Krea2 Focused ROI Rewriteの単一画像実機評価",
    )
    doc.build(build_story(case, locator))
    reader = PdfReader(str(output))
    if len(reader.pages) != 2:
        raise RuntimeError(f"Expected exactly 2 B5 pages, generated {len(reader.pages)}")
    expected = tuple(value / mm for value in base.JIS_B5)
    for index, page in enumerate(reader.pages, start=1):
        actual = (float(page.mediabox.width) / mm, float(page.mediabox.height) / mm)
        if any(abs(got - want) > 0.05 for got, want in zip(actual, expected)):
            raise RuntimeError(
                f"Page {index}: {actual[0]:.2f}x{actual[1]:.2f} mm; "
                f"expected {expected[0]:.2f}x{expected[1]:.2f} mm"
            )
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = (
        "Focused ROI Rewrite",
        "5.12",
        "16,646",
        "92.4778%",
        "0 px",
        "20,957 MiB",
        "618cfba205a0",
    )
    missing = [text for text in required if text not in extracted]
    if missing:
        raise RuntimeError("PDF text extraction is missing evidence: " + ", ".join(missing))
    repo_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, repo_copy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-copy", type=Path, default=DEFAULT_REPO_COPY)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--font", default=None)
    parser.add_argument("--bold-font", default=None)
    args = parser.parse_args()

    regular_font = base.find_japanese_font(args.font)
    bold_font = base.find_japanese_bold_font(regular_font, args.bold_font)
    case = load_case(args.manifest, args.validation, args.comparison)
    locator = make_locator(
        case,
        args.assets_dir / "focused_target_locator.png",
        regular_font,
        bold_font,
    )
    build_pdf(
        args.output,
        args.repo_copy,
        case,
        locator,
        regular_font,
        bold_font,
    )
    sys.stdout.write(f"PDF: {args.output}\n")
    sys.stdout.write(f"Repository copy: {args.repo_copy}\n")
    sys.stdout.write(f"Locator: {locator}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
