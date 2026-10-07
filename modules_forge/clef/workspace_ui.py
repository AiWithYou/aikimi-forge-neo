"""Preserved input drafts with one question set and one saved-result workbench."""

from __future__ import annotations

import copy
import json
import math
import shutil
import uuid
from functools import lru_cache, wraps
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from .collection import generation_metadata, merge_images, parse_records, scan_files, scan_folder, thumbnails
from .core import OUTPUTS, PROFILES, ROOT, TEMPLATES, ClefError, atomic_json, canonical_hash, parse_schema, sha256
from .curation import REVIEW_LABELS, copy_accepted, latest_export, matching_ids, read_review, set_decision
from .filters import active_rules, apply_filter_event, filter_payload, filter_state
from .service import STUDIO, execution_stopped
from .ui import (
    COMPARISON_NOTE,
    NEW_ITEM,
    PRIVATE,
    TYPE_LABELS,
    _history,
    _read_run,
    _request,
    _schema_outputs,
    build_conditions,
    comparison_view,
    detail_html,
    edit_fields,
    environment_status,
    gallery_token,
    progress_label,
    run_heading,
    schema_update,
    selected_token,
    table_view,
    visible_selection,
)

RECORD_TEMPLATES = {
    "問い合わせの振り分け": TEMPLATES["文章・JSONの分類"],
    "制作メモの整理": {
        "category": {
            "type": "choice",
            "instructions": "この制作メモを用途で分類してください。",
            "criteria": {
                "idea": "新しい案や着想",
                "fix": "修正や改善の依頼",
                "setting": "設定値や制作条件の記録",
                "other": "その他",
            },
        },
        "action": {"type": "noul", "instructions": "このメモは具体的な作業や対応を求めている。"},
    },
    "文章の用途分け": {
        "purpose": {
            "type": "choice",
            "instructions": "文章の主な用途を分類してください。",
            "criteria": {
                "instruction": "操作手順や説明",
                "story": "物語や創作",
                "record": "事実や作業の記録",
                "request": "質問や依頼",
                "other": "その他",
            },
        },
        "procedure": {"type": "noul", "instructions": "文章に具体的な操作手順が含まれている。"},
    },
}
HTML_CSS = (ROOT / "extensions-builtin/clef-studio/style.css").read_text(encoding="utf-8")
FILTER_JS = (ROOT / "extensions-builtin/clef-studio/assets/filters.js").read_text(encoding="utf-8")
CLIPBOARD_JS = (ROOT / "extensions-builtin/clef-studio/assets/clipboard.js").read_text(encoding="utf-8")


def formatted_html(value):
    return gr.HTML(value, css_template=HTML_CSS, apply_default_css=False)


def guarded(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ClefError, OSError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    return invoke


def collection_label(collection):
    items = (collection or {}).get("items", [])
    label = f"{len(items)}件 · {(collection or {}).get('bytes', 0) / 2**20:.1f}MiB · 除外{len((collection or {}).get('skipped', []))}件"
    return label + ("\n" + collection["root"] if (collection or {}).get("root") else "")


def history_for():
    choices = []
    for label, key in _history():
        try:
            items = _read_run(key)["items"]
            if items:
                source = "画像" if items[0].get("kind", "image") == "image" else "文章・JSON"
                choices.append((f"{source} · {label}", key))
        except (OSError, ValueError, KeyError):
            pass
    return choices


def resolve_input(source, images, text, text_mode, json_data, json_mode):
    """Use the client-selected source and current draft, never a cached preview."""
    if source == "image":
        return {**images, "source": source}
    if source not in {"text", "json"}:
        raise ClefError("入力タブを選択してください。")
    body, mode = (text, text_mode) if source == "text" else (json_data, json_mode)
    allowed = {"text", "lines", "paragraphs"} if source == "text" else {"json", "array", "jsonl"}
    if mode not in allowed:
        raise ClefError("入力の区切り方を選択してください。")
    return {
        "source": source,
        "items": parse_records(body, mode),
        "bytes": len(body.encode()),
        "root": "",
        "skipped": [],
    }


def current_request(profile, context, schema, pixels, length, collection):
    image = (collection or {}).get("source", "image") == "image"
    return _request(profile, context if image else "", schema, pixels if image else 262144, length)


def is_image_run(run):
    return bool(run and run.get("items") and run["items"][0].get("kind", "image") == "image")


def same_input(run, collection):
    def identity(items):
        return [(item.get("kind", "image"), item["sha256"]) for item in items]

    return identity(run["items"]) == identity((collection or {}).get("items", []))


@lru_cache(maxsize=128)
def saved_metadata(path, directory, item_id, digest):
    if sha256(path) != digest:
        raise ClefError("選択した画像が変更されています。読み直してください。")
    metadata = generation_metadata(path)
    files = {}
    for kind in ("workflow", "prompt"):
        if metadata[kind] is not None:
            target = Path(directory) / "metadata" / f"{item_id}.{kind}.json"
            atomic_json(target, metadata[kind])
            files[kind] = str(target)
    text = "\n".join(f"{key}: {value}" for key, value in metadata["settings"].items())
    if not files:
        text = "生成設定の記録なし"
    elif len(files) == 1:
        text = "一部の記録のみ\n" + text
    text += "\n" + "\n".join(metadata["errors"]) if metadata["errors"] else ""
    return text or "JSONの記録あり", files


def result_thumbnails(run, ids, directory, review):
    items = {item["id"]: item for item in run["items"]}
    previews = []
    for key in ids:
        item = items[key]
        target = Path(directory) / "previews" / f"{run['id']}__{key}.jpg"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(item["path"]) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((400, 400))
                image.save(target, quality=85)
        decision = REVIEW_LABELS[review["decisions"].get(key, "unreviewed")]
        previews.append((str(target), f"{int(key) + 1} · {item['name']} · {decision}"))
    return previews


def schema_editor(initial, schema, selector, schema_table, outputs_root, notice):
    events = {
        **PRIVATE,
        "concurrency_id": "clef-view",
        "concurrency_limit": 1,
        "show_progress": "hidden",
        "show_progress_on": [],
    }

    def save(value):
        target = outputs_root / "schemas" / (canonical_hash(value)[:16] + ".json")
        atomic_json(target, value)
        return str(target)

    first = next(iter(initial))
    with gr.Accordion("質問を編集・追加", open=False):
        selector.render()
        with gr.Row():
            qid = gr.Textbox(value=first, label="質問ID", scale=1)
            kind = gr.Dropdown(TYPE_LABELS, value=initial[first]["type"], label="判断形式", scale=2)
        instruction = gr.Textbox(value=initial[first]["instructions"], label="質問", lines=2)
        options = gr.Textbox(value=edit_fields(initial, first)[3], label="選択肢 · 1行に ID: 説明", lines=3)
        with gr.Row():
            apply_item = gr.Button("質問の変更を適用", size="sm")
            delete_item = gr.Button("項目を削除", size="sm")
    with gr.Accordion("質問セットのJSON · 保存・読み込み", open=False):
        schema_json = gr.Code(
            value=json.dumps(initial, ensure_ascii=False, indent=2),
            language="json",
            label="質問セットのJSON",
            lines=8,
            wrap_lines=True,
        )
        with gr.Row():
            import_json = gr.Button("JSONを質問セットに適用", size="sm")
            schema_download = gr.DownloadButton("質問セットを保存", value=save(initial), size="sm")
        schema_file = gr.File(label="保存した質問セットを読み込む", file_types=[".json"], type="filepath")
    outputs = [schema, selector, schema_table, schema_json]

    def question_notice(schema, key, qid, kind, instruction, options):
        pending = [qid, kind, instruction, options] != list(edit_fields(schema, key))
        return gr.update(
            value="質問に未適用の編集があります。「質問の変更を適用」で確定してください。", visible=pending
        )

    for field in (qid, kind, instruction, options):
        field.input(
            question_notice,
            [schema, selector, qid, kind, instruction, options],
            notice,
            trigger_mode="always_last",
            **events,
        )
    for control in (schema, selector):
        control.change(lambda: gr.update(visible=False), outputs=notice, **events)

    def fields(schema, key):
        values = edit_fields(schema, key)
        return (
            *values[:3],
            gr.update(
                value=values[3],
                visible=values[1] != "noul",
                label="段階 · 上から低い順、1行に1段階" if values[1] == "score" else "選択肢 · 1行に ID: 説明",
            ),
        )

    selector.change(fields, [schema, selector], [qid, kind, instruction, options], **events)
    schema.change(fields, [schema, selector], [qid, kind, instruction, options], **events)
    kind.change(
        lambda value: gr.update(
            visible=value != "noul",
            label="段階 · 上から低い順、1行に1段階" if value == "score" else "選択肢 · 1行に ID: 説明",
        ),
        kind,
        options,
        **events,
    )
    apply_item.click(
        guarded(
            lambda schema, previous, *values: _schema_outputs(
                schema_update(schema, previous, *values), values[0].strip()
            )
        ),
        [schema, selector, qid, kind, instruction, options],
        outputs,
        **events,
    )

    @guarded
    def remove(schema, key):
        if len(schema) <= 1:
            raise ClefError("判断項目は少なくとも1件必要です。")
        value = copy.deepcopy(schema)
        value.pop(key, None)
        return _schema_outputs(value)

    delete_item.click(remove, [schema, selector], outputs, **events)
    import_json.click(guarded(lambda value: _schema_outputs(parse_schema(value))), schema_json, outputs, **events)
    schema_file.change(
        guarded(
            lambda path: (
                _schema_outputs(parse_schema(Path(path).read_text(encoding="utf-8-sig"))) if path else [gr.update()] * 4
            )
        ),
        schema_file,
        outputs,
        **events,
    )

    schema.change(save, schema, schema_download, **events)
    return outputs


def workbench(profile, outputs_root):
    initial = copy.deepcopy(TEMPLATES["人物イラストの仕分け"])
    schema = gr.State(initial)
    images = gr.State({"items": [], "root": "", "skipped": [], "bytes": 0})
    folder_images = gr.State({"items": [], "root": "", "skipped": [], "bytes": 0})
    collection = gr.State({"source": "image", "items": [], "root": "", "skipped": [], "bytes": 0})
    # A client value avoids waiting for a queued Tab.select before capturing a run.
    source = gr.Textbox(value="image", visible=False, elem_id="clef-input-source")
    displayed = gr.State(None)
    active_job = gr.State("")
    previous_schema = gr.State(None)
    selection = gr.JSON(None, visible=False)
    selection_state = gr.State(None)
    view_state = gr.State(["入力順", "すべて", None, 1])
    timer = gr.Timer(1, active=False)
    view_events = {
        **PRIVATE,
        "concurrency_id": "clef-view",
        "concurrency_limit": 1,
        "show_progress": "hidden",
        "show_progress_on": [],
    }
    with gr.Row():
        with gr.Column(scale=4, min_width=310):
            gr.Markdown("### 入力")
            with gr.Tabs(selected="image", elem_id="clef-input-tabs"):
                with gr.Tab("画像", id="image") as image_tab:
                    folder = gr.Textbox(
                        label="フォルダの絶対パス", placeholder=r"G:\Tool\ComfyUI-Output\20250729", lines=1
                    )
                    with gr.Row():
                        scan = gr.Button("フォルダを読み込む", variant="secondary")
                        recursive = gr.Checkbox(label="サブフォルダも含める", value=False)
                    source_note = gr.Textbox(
                        value="フォルダを読み込むか、個別画像を追加してください。",
                        interactive=False,
                        show_label=False,
                        lines=2,
                    )
                    with gr.Accordion("個別画像を追加", open=False, elem_id="clef-individual-images"):
                        files = gr.File(
                            label="画像を追加 · 複数可",
                            file_count="multiple",
                            file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                            type="filepath",
                            height=140,
                            elem_id="clef-images",
                        )
                        gr.HTML(
                            '<p class="clef-paste-hint" role="status" aria-live="polite">'
                            "この欄で Ctrl+V（⌘V）を押すと、コピーした画像を追加できます。"
                            "</p>",
                            js_on_load=CLIPBOARD_JS,
                            elem_id="clef-image-paste",
                        )
                    gallery = gr.Gallery(
                        label="入力画像 · 12件ずつ表示",
                        columns=3,
                        rows=1,
                        height=210,
                        interactive=False,
                        object_fit="contain",
                        visible=False,
                        elem_id="clef-input-gallery",
                    )
                    page = gr.Number(label="プレビューページ", value=1, precision=0, minimum=1, visible=False)
                    with gr.Accordion("除外したファイル", open=False):
                        skipped = gr.Dataframe(
                            headers=["ファイル", "理由"],
                            value=[],
                            interactive=False,
                            type="array",
                            column_count=(2, "fixed"),
                            wrap=True,
                        )
                    with gr.Accordion("画像に添える補足・解像度", open=False):
                        context = gr.Textbox(value="添付画像を評価してください。", label="補足の文章・JSON", lines=2)
                        pixels = gr.Dropdown(
                            [
                                ("256² px 相当", 65536),
                                ("512² px 相当", 262144),
                                ("768² px 相当", 589824),
                                ("1024² px 相当", 1048576),
                            ],
                            value=262144,
                            label="画像の最大画素数 · 元の縦横比を維持",
                        )
                    with gr.Accordion("生成設定を取り出す", open=False):
                        metadata_tab(images, outputs_root)
                with gr.Tab("文章", id="text") as text_tab:
                    text = gr.Textbox(
                        label="判定する文章",
                        placeholder="改行を含む本文をそのまま入力",
                        lines=8,
                        elem_id="clef-text-input",
                    )
                    text_mode = gr.Dropdown(
                        [("全体で1件", "text"), ("1行1件", "lines"), ("空行区切り", "paragraphs")],
                        value="text",
                        label="文章の区切り",
                    )
                    with gr.Row():
                        text_example = gr.Button("空欄に文章の入力例を入れる", size="sm", scale=0)
                with gr.Tab("JSONデータ", id="json") as json_tab:
                    json_data = gr.Code(
                        label="判定するJSONデータ",
                        language="json",
                        lines=8,
                        wrap_lines=True,
                        elem_id="clef-json-input",
                    )
                    json_mode = gr.Dropdown(
                        [("全体で1件", "json"), ("配列の要素ごと", "array"), ("JSONL · 1行1件", "jsonl")],
                        value="json",
                        label="JSONの区切り",
                    )
                    with gr.Row():
                        json_example = gr.Button("空欄にJSONの入力例を入れる", size="sm", scale=0)
            gr.Markdown("### 質問セット")
            with gr.Row(elem_classes=["clef-template-row"]):
                template = gr.Dropdown(
                    [name for name in TEMPLATES if name != "文章・JSONの分類"] + list(RECORD_TEMPLATES),
                    value=None,
                    label="テンプレートから質問を読み込む",
                    scale=3,
                )
                apply_template = gr.Button("テンプレートを適用", size="sm", scale=1, interactive=False)
                undo = gr.Button("直前の質問に戻す", size="sm", scale=0, visible=False)
            schema_table = formatted_html(_schema_outputs(initial)[2])
            question_warning = gr.Markdown("", visible=False)
            run_summary = gr.Textbox(
                value="画像 · 入力0件 · 質問3件", interactive=False, show_label=False, elem_id="clef-input-summary"
            )
            with gr.Row(elem_classes=["clef-run-toolbar"]):
                judge = gr.Button("入力を追加してください", variant="primary", interactive=False, elem_id="clef-judge")
                stop = gr.Button("停止", interactive=False, scale=0)
            status = gr.Textbox(
                value="入力と質問を確認して実行してください。", show_label=False, interactive=False, lines=2
            )
            selector = gr.Dropdown(
                choices=[(question.get("instructions") or key, key) for key, question in initial.items()]
                + [("＋ 新しい項目", NEW_ITEM)],
                value=next(iter(initial)),
                label="編集する質問",
                render=False,
            )
            schema_outputs = schema_editor(initial, schema, selector, schema_table, outputs_root, question_warning)
            with gr.Accordion("条件を1行ずつ質問にする", open=False):
                conditions = gr.Textbox(
                    label="判定する質問 · 1行に1条件",
                    placeholder="具体的な対応を求めている。\nすぐに対処する必要がある。",
                    lines=3,
                )
                apply_conditions = gr.Button("条件を判断項目に適用", size="sm")
            with gr.Accordion("入力トークン上限", open=False):
                length = gr.Slider(512, 8192, value=4096, step=512, label="総入力トークン上限 · 超過は切り捨てずエラー")
        with gr.Column(scale=6, min_width=350):
            gr.Markdown("### 結果")
            heading = gr.Textbox(value="まだ判定していません", show_label=False, interactive=False)
            with gr.Accordion("過去の実行", open=False):
                with gr.Row():
                    history = gr.Dropdown(choices=history_for(), value=None, label="保存した判定を開く", scale=3)
                    refresh = gr.Button("更新", size="sm", scale=0)
                return_active = gr.Button("現在の実行に戻る", size="sm")
            with gr.Column(elem_id="clef-result-content", elem_classes=["clef-results-empty"]) as result_content:
                with gr.Row():
                    visibility = gr.Dropdown(
                        ["すべて", "候補だけ", "未選択", "採用", "保留", "見送り"], value="すべて", label="表示する項目"
                    )
                    order = gr.Dropdown(["入力順", "曖昧な順"], value="入力順", label="一覧の順序")
                with gr.Accordion("この実行の確率で絞る · 再判定なし", open=False):
                    filter_panel = gr.HTML(
                        None,
                        html_template="<div></div>",
                        css_template=HTML_CSS,
                        js_on_load=FILTER_JS,
                        apply_default_css=False,
                        elem_id="clef-filters-common",
                    )
                counts = gr.Textbox(value="採用 0件 · 保留 0件", interactive=False, show_label=False)
                result_gallery = gr.Gallery(
                    label="画像を選ぶ · 12件ずつ表示",
                    value=[],
                    columns=4,
                    rows=3,
                    height=430,
                    interactive=False,
                    allow_preview=False,
                    object_fit="contain",
                    buttons=[],
                    type="filepath",
                    visible=False,
                    elem_id="clef-result-gallery",
                )
                result_page = gr.Number(label="結果ページ", value=1, precision=0, minimum=1, visible=False)
                with gr.Accordion("全項目の判定一覧", open=True) as result_table:
                    results = gr.Dataframe(
                        headers=["#", "入力", "状態", "人の判断", "判断結果"],
                        value=[],
                        interactive=False,
                        type="array",
                        column_count=(5, "fixed"),
                        wrap=True,
                        max_height=220,
                        column_widths=["5%", "25%", "8%", "10%", "52%"],
                        elem_id="clef-results",
                    )
                selected_image = gr.Image(
                    label="選択した画像", interactive=False, height=300, type="filepath", visible=False
                )
                selected_record = gr.Code(
                    label="選択した入力 · 実行時の本文・JSON",
                    language="json",
                    visible=False,
                    lines=5,
                    wrap_lines=True,
                    elem_id="clef-selected-record",
                )
                with gr.Row():
                    accept = gr.Button("採用", size="sm", interactive=False)
                    hold = gr.Button("保留", size="sm", interactive=False)
                    reject = gr.Button("見送り", size="sm", interactive=False)
                    clear = gr.Button("選択を解除", size="sm", interactive=False)
                with gr.Accordion("質問ごとの確率", open=True):
                    detail = formatted_html(detail_html(None, None))
                with gr.Accordion("選択画像の生成設定", open=False, visible=False) as result_metadata:
                    settings = gr.Textbox(label="画像に残っている設定", interactive=False, lines=5)
                    with gr.Row():
                        workflow_file = gr.DownloadButton("workflow JSONを保存", interactive=False)
                        prompt_file = gr.DownloadButton("API prompt JSONを保存", interactive=False)
                with gr.Column(
                    elem_id="clef-image-exports", elem_classes=["clef-image-exports-hidden"]
                ) as image_exports:
                    with gr.Row():
                        export = gr.Button("採用した画像をコピー · 0件", variant="secondary", interactive=False)
                        with_settings = gr.Checkbox(value=True, label="生成設定のJSONも保存")
                    export_note = gr.Textbox(label="コピー先", interactive=False, visible=False)
                    export_file = gr.DownloadButton("採用画像 ZIPを保存", interactive=False)
                gr.Markdown("確率は選択肢間の分布です。候補と、人による採用は別に保存します。")
                with gr.Row():
                    json_file = gr.DownloadButton("表示中の結果 JSONを保存", interactive=False)
                    csv_file = gr.DownloadButton("表示中の一覧 CSVを保存", interactive=False)
                with gr.Row():
                    resume = gr.Button("保存条件で未完了分を再開", visible=False)
                    retry_errors = gr.Checkbox(label="エラーも再試行", value=False, visible=False)
                with gr.Accordion("同じ入力・質問の結果と比較", open=False):
                    compare = gr.Dropdown(choices=history_for(), value=None, label="比較対象の判定")
                    comparison_note = gr.Markdown(COMPARISON_NOTE)
                    comparison = gr.Dataframe(
                        headers=["画像・レコード", "判断の変化", "最大確率差", "項目ごとの差"],
                        interactive=False,
                        type="array",
                        column_count=(4, "fixed"),
                        wrap=True,
                    )

    filters = [order, visibility, result_page]
    current = [profile, context, schema, pixels, length, collection]
    source_inputs = [source, images, text, text_mode, json_data, json_mode]
    result_outputs = [
        displayed,
        results,
        detail,
        selected_image,
        selected_record,
        heading,
        json_file,
        csv_file,
        selection_state,
        counts,
        export,
        resume,
        retry_errors,
        accept,
        hold,
        reject,
        clear,
        result_gallery,
        result_page,
        filter_panel,
        selection,
        export_note,
        export_file,
        result_table,
        result_metadata,
        image_exports,
        result_content,
        view_state,
    ]

    def paint(
        run,
        token,
        profile,
        context,
        schema,
        pixels,
        length,
        collection,
        order,
        visibility,
        conditions,
        page_number,
    ):
        run = _read_run(run["id"])
        image_mode = is_image_run(run)
        directory = outputs_root / run["id"]
        review = read_review(directory, run)
        new_panel = not conditions or conditions.get("run_id") != run["id"]
        conditions = filter_state(run, conditions)
        rules = active_rules(conditions)
        table = table_view(run, order, review, visibility, rules, conditions["mode"])
        total_pages = max(1, math.ceil(len(table["data"]) / 12))
        page_number = max(1, min(int(page_number or 1), total_pages))
        visible_ids = [selected_token(run, row)[1] for row in table["data"]]
        page_ids = visible_ids[(page_number - 1) * 12 : page_number * 12] if image_mode else visible_ids
        item = visible_selection(run, token, page_ids)
        index = run["items"].index(item) if item else None
        image_path = item["path"] if item and item.get("kind", "image") == "image" else None
        record = (
            json.loads(Path(item["path"]).read_text(encoding="utf-8"))["state"]
            if item and item.get("kind") == "record"
            else None
        )
        record_text = (
            json.dumps(record, ensure_ascii=False, indent=2) if item and item.get("kind") == "record" else None
        )
        try:
            request = current_request(profile, context, schema, pixels, length, collection)
        except ClefError:
            request = None
        label = run_heading(run, request)
        if not same_input(run, collection) and "不一致" not in label:
            label += " · 現在の入力と不一致"
        decisions = review["decisions"]
        accepted = sum(value == "accepted" for value in decisions.values())
        held = sum(value == "hold" for value in decisions.values())
        candidate = str(len(matching_ids(run, rules, conditions["mode"]))) if rules else "条件未設定"
        count = (
            f"表示 {len(table['data'])}/{len(run['items'])}件 · 候補 {candidate} · 採用 {accepted}件 · 保留 {held}件"
        )
        partial = (run["status"] != "running" or execution_stopped(directory)) and any(
            item["status"] != "done" for item in run["items"]
        )
        csv = directory / "results.csv"
        saved_export = latest_export(directory, run, decisions) if image_mode else None
        return [
            run,
            table,
            detail_html(run, index),
            gr.update(value=image_path, visible=bool(image_path)),
            gr.update(value=record_text, visible=record_text is not None),
            label,
            gr.update(value=str(directory / "result.json"), interactive=True),
            gr.update(value=str(csv) if csv.is_file() else None, interactive=csv.is_file()),
            [run["id"], item["id"]] if item else None,
            count,
            gr.update(
                value=f"採用した画像をコピー · {accepted}件",
                interactive=image_mode
                and accepted > 0
                and all(item["status"] == "done" for item in run["items"] if decisions.get(item["id"]) == "accepted"),
            ),
            gr.update(visible=partial),
            gr.update(visible=partial),
            *[gr.update(interactive=item is not None) for _ in range(4)],
            gr.update(
                value=result_thumbnails(run, page_ids, directory, review) if image_mode else [],
                visible=image_mode,
                selected_index=page_ids.index(item["id"]) if image_mode and item else None,
            ),
            gr.update(value=page_number, visible=image_mode and total_pages > 1, maximum=total_pages),
            gr.update(value=filter_payload(run, conditions)) if new_panel else gr.skip(),
            [run["id"], item["id"]] if item else None,
            gr.update(value=saved_export[0] if saved_export else None, visible=bool(saved_export)),
            gr.update(value=saved_export[1] if saved_export else None, interactive=bool(saved_export)),
            gr.update(open=not image_mode) if new_panel else gr.skip(),
            gr.update(visible=image_mode),
            gr.update(elem_classes=[] if image_mode else ["clef-image-exports-hidden"]),
            gr.update(elem_classes=[]),
            [order, visibility, conditions, page_number],
        ]

    @guarded
    def submit(
        source,
        images,
        text,
        text_mode,
        json_data,
        json_mode,
        profile,
        context,
        schema,
        pixels,
        length,
        request: gr.Request,
    ):
        collection = resolve_input(source, images, text, text_mode, json_data, json_mode)
        if not (collection or {}).get("items"):
            raise ClefError("実行する入力がありません。フォルダまたは入力を確認してください。")
        identifier = STUDIO.start(
            current_request(profile, context, schema, pixels, length, collection),
            collection["items"],
            request.session_hash,
        )
        return (
            identifier,
            "実行を開始しました。判定済みの結果は順次表示します。",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(active=True),
            None,
            None,
            _read_run(identifier),
            collection,
        )

    judge.click(
        submit,
        [*source_inputs, profile, context, schema, pixels, length],
        [active_job, status, judge, stop, timer, selection_state, selection, displayed, collection],
        **view_events,
    )

    @guarded
    def poll(
        identifier,
        run,
        token,
        profile,
        context,
        schema,
        pixels,
        length,
        collection,
        view,
        request: gr.Request,
    ):
        if not identifier:
            return [gr.update()] * (len(result_outputs) + 5)
        info = STUDIO.status(identifier, request.session_hash)
        done = info["done"]
        values = [
            profile,
            context,
            schema,
            pixels,
            length,
            collection,
            *view,
        ]
        outputs = (
            paint(info["result"], token, *values)
            if not run or run["id"] == identifier
            else [gr.update()] * len(result_outputs)
        )
        return [
            *outputs,
            progress_label(info["result"], info["message"]),
            gr.update(interactive=done and bool((collection or {}).get("items"))),
            gr.update(interactive=not done),
            gr.update(active=not done),
            gr.update(choices=history_for()),
        ]

    timer.tick(
        poll,
        [active_job, displayed, selection_state, *current, view_state],
        [*result_outputs, status, judge, stop, timer, history],
        **view_events,
    )

    @guarded
    def cancel_job(key, request: gr.Request):
        if key:
            STUDIO.cancel(key, request.session_hash)
        return "停止を要求しました。完了分を保存し、プロセス終了を確認しています。"

    stop.click(cancel_job, active_job, status, queue=False, **PRIVATE)

    @guarded
    def resume_job(run, displayed_heading, retry, request: gr.Request):
        if not run:
            raise ClefError("再開する判定記録を開いてください。")
        if not displayed_heading.startswith(run["id"]):
            raise ClefError("表示する実行が変わりました。結果を確認してから再開してください。")
        identifier = STUDIO.resume(run["id"], request.session_hash, retry)
        return (
            identifier,
            _read_run(identifier),
            "保存した条件で未完了分を再開します。",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(active=True),
        )

    resume.click(
        resume_job,
        [displayed, heading, retry_errors],
        [active_job, displayed, status, judge, stop, timer],
        **view_events,
    )

    def refresh_view(run, token, profile, context, schema, pixels, length, collection, view, order, visibility, page):
        return (
            paint(run, token, profile, context, schema, pixels, length, collection, order, visibility, view[2], page)
            if run
            else [gr.update()] * len(result_outputs)
        )

    for control in filters:
        control.input(
            guarded(refresh_view),
            [displayed, selection_state, *current, view_state, *filters],
            result_outputs,
            **view_events,
        )

    @guarded
    def change_filters(run, token, profile, context, schema, pixels, length, collection, view, event: gr.EventData):
        conditions = apply_filter_event(run, view[2], event._data)
        if conditions is None:
            return [gr.skip()] * len(result_outputs)
        return paint(run, token, profile, context, schema, pixels, length, collection, view[0], view[1], conditions, 1)

    filter_panel.input(
        change_filters,
        [displayed, selection_state, *current, view_state],
        result_outputs,
        trigger_mode="multiple",
        **view_events,
    )
    history.change(
        guarded(
            lambda key, *values: (
                paint(_read_run(key), None, *values[:-1], *values[-1]) if key else [gr.update()] * len(result_outputs)
            )
        ),
        [history, *current, view_state],
        result_outputs,
        **view_events,
    )
    refresh.click(
        lambda: (gr.update(choices=history_for()), gr.update(choices=history_for())),
        outputs=[history, compare],
        **PRIVATE,
    )

    @guarded
    def return_current(key, *values):
        result = paint(_read_run(key), None, *values[:-1], *values[-1]) if key else [gr.update()] * len(result_outputs)
        return [*result, gr.update(value=None)]

    return_active.click(
        return_current,
        [active_job, *current, view_state],
        [*result_outputs, history],
        **view_events,
    )

    @guarded
    def select(run, profile, context, schema, pixels, length, collection, view, event: gr.SelectData):
        token = selected_token(run, event.row_value)
        run = _read_run(run["id"])
        review = read_review(outputs_root / run["id"], run)
        order, visibility, conditions, page_number = view
        conditions = filter_state(run, conditions)
        table = table_view(run, order, review, visibility, active_rules(conditions), conditions["mode"])
        ids = [selected_token(run, row)[1] for row in table["data"]]
        if token[1] not in ids:
            raise ClefError("表示が更新されています。現在の一覧から選び直してください。")
        if is_image_run(run):
            view = [order, visibility, conditions, ids.index(token[1]) // 12 + 1]
        return paint(run, token, profile, context, schema, pixels, length, collection, *view)

    @guarded
    def select_image(run, previews, profile, context, schema, pixels, length, collection, view, event: gr.SelectData):
        index = int(event.index)
        if not previews or not 0 <= index < len(previews):
            raise ClefError("現在の画像一覧から選び直してください。")
        token = gallery_token(run, previews[index][0])
        return paint(run, token, profile, context, schema, pixels, length, collection, *view)

    results.select(select, [displayed, *current, view_state], result_outputs, **view_events)
    result_gallery.select(
        select_image, [displayed, result_gallery, *current, view_state], result_outputs, **view_events
    )

    for button, decision in [(accept, "accepted"), (hold, "hold"), (reject, "rejected"), (clear, "unreviewed")]:

        @guarded
        def review(run, token, view_token, *values, decision=decision):
            if not run:
                raise ClefError("判定済みの項目を選んでください。")
            set_decision(outputs_root / run["id"], run, token, decision)
            return paint(run, view_token, *values[:-1], *values[-1])

        button.click(
            review, [displayed, selection, selection_state, *current, view_state], result_outputs, **view_events
        )

    def dirty_heading(run, profile, context, schema, pixels, length, collection):
        label = "まだ判定していません"
        if run:
            try:
                request = current_request(profile, context, schema, pixels, length, collection)
            except ClefError:
                request = None
            label = run_heading(run, request)
            if not same_input(run, collection) and "不一致" not in label:
                label += " · 現在の入力と不一致"
        value = collection or {}
        count = len(value.get("items", []))
        source_label = {"image": "画像", "text": "文章", "json": "JSONデータ"}.get(value.get("source", "image"), "入力")
        summary = f"{source_label} · 入力{count}件 · 質問{len(schema)}件"
        if value.get("error"):
            summary += "\n" + value["error"]
        busy = any(not job.done.is_set() or job.lease_held for job in STUDIO.jobs.values())
        action = f"{count}件を判定"
        return (
            label,
            gr.update(value=action if count else "入力を追加してください", interactive=count > 0 and not busy),
            summary,
        )

    for control in current:
        control.change(dirty_heading, [displayed, *current], [heading, judge, run_summary], **view_events)

    def preview_input(source, images, text, text_mode, json_data, json_mode):
        try:
            return resolve_input(source, images, text, text_mode, json_data, json_mode)
        except (ClefError, ValueError) as exc:
            return {"source": source, "items": [], "bytes": 0, "root": "", "skipped": [], "error": str(exc)}

    for control in source_inputs:
        control.change(preview_input, source_inputs, collection, trigger_mode="always_last", **view_events)
    for tab, name in [(image_tab, "image"), (text_tab, "text"), (json_tab, "json")]:
        tab.select(fn=None, outputs=source, js=f"() => ['{name}']", queue=False, **PRIVATE)

    def collection_outputs(value):
        total = len(value["items"])
        return (
            value,
            collection_label(value),
            gr.update(value=thumbnails(value), visible=bool(total)),
            gr.update(value=1, visible=total > 12, maximum=max(1, math.ceil(total / 12))),
            [[row["name"], row["reason"]] for row in value["skipped"]],
        )

    @guarded
    def load_folder(path, recursive, values):
        base = scan_folder(path, recursive)
        return (base, *collection_outputs(merge_images(base, scan_files(values))))

    @guarded
    def load_files(base, values):
        return collection_outputs(merge_images(base, scan_files(values)))

    scan.click(
        load_folder,
        [folder, recursive, files],
        [folder_images, images, source_note, gallery, page, skipped],
        **view_events,
    )
    files.change(
        load_files,
        [folder_images, files],
        [images, source_note, gallery, page, skipped],
        **view_events,
    )
    page.change(
        guarded(
            lambda value, number: thumbnails(value, max(1, min(int(number), math.ceil(len(value["items"]) / 12) or 1)))
        ),
        [images, page],
        gallery,
        **view_events,
    )

    @guarded
    def load_template(name, old_schema):
        if not name:
            raise ClefError("読み込むテンプレートを選んでください。")
        value = RECORD_TEMPLATES[name] if name in RECORD_TEMPLATES else TEMPLATES[name]
        return (
            *_schema_outputs(copy.deepcopy(value)),
            copy.deepcopy(old_schema),
            gr.update(value=None),
            gr.update(interactive=True, visible=True),
            gr.update(interactive=False),
        )

    @guarded
    def undo_template(value):
        if not value:
            raise ClefError("戻す質問セットがありません。")
        return (*_schema_outputs(value), None, gr.update(interactive=False, visible=False))

    apply_template.click(
        load_template,
        [template, schema],
        [*schema_outputs, previous_schema, template, undo, apply_template],
        **view_events,
    )
    undo.click(undo_template, previous_schema, [*schema_outputs, previous_schema, undo], **view_events)
    template.input(lambda name: gr.update(interactive=bool(name)), template, apply_template, **view_events)
    apply_conditions.click(
        guarded(lambda value: _schema_outputs(build_conditions(value))),
        conditions,
        schema_outputs,
        **view_events,
    )

    def text_example_value(value):
        return (
            value
            if value and value.strip()
            else "請求された金額が契約と違うので確認したいです。\n契約書と請求書を確認してもらえますか？"
        )

    def json_example_value(value):
        return (
            value
            if value and value.strip()
            else json.dumps(
                {"id": "inquiry-1", "body": "サービスに接続できず、全員の作業が止まっています。", "priority": "high"},
                ensure_ascii=False,
                indent=2,
            )
        )

    text_example.click(text_example_value, text, text, **view_events)
    json_example.click(json_example_value, json_data, json_data, **view_events)
    for control, button in [(text, text_example), (json_data, json_example)]:
        control.change(
            lambda value: gr.update(interactive=not bool(value and value.strip())), control, button, **view_events
        )

    @guarded
    def image_settings(run, token):
        item = next(
            (
                item
                for item in (run or {}).get("items", [])
                if token and token[0] == run["id"] and item["id"] == token[1]
            ),
            None,
        )
        if not item or item.get("kind", "image") != "image":
            return (
                "画像を選択してください。",
                gr.update(value=None, interactive=False),
                gr.update(value=None, interactive=False),
            )
        text, files = saved_metadata(item["path"], str(outputs_root / run["id"]), item["id"], item["sha256"])
        return (
            text,
            gr.update(value=files.get("workflow"), interactive="workflow" in files),
            gr.update(value=files.get("prompt"), interactive="prompt" in files),
        )

    selection_state.change(
        image_settings, [displayed, selection_state], [settings, workflow_file, prompt_file], **view_events
    )

    @guarded
    def export_images(run, displayed_heading, include_settings):
        if not is_image_run(run):
            raise ClefError("判定した画像を採用してからコピーしてください。")
        if not displayed_heading.startswith(run["id"]):
            raise ClefError("表示する実行が変わりました。結果を確認してからコピーしてください。")
        run = _read_run(run["id"])
        directory = outputs_root / run["id"]
        target = copy_accepted(directory, run, include_settings)
        archive = shutil.make_archive(str(target), "zip", target)
        atomic_json(directory / "last-export.json", {"directory": target.relative_to(directory).as_posix()})
        return gr.update(value=str(target), visible=True), gr.update(value=archive, interactive=True)

    export.click(export_images, [displayed, heading, with_settings], [export_note, export_file], **view_events)

    compare.change(
        lambda run, key: comparison_view(run, _read_run(key)) if run and key else comparison_view(None, None),
        [displayed, compare],
        [comparison, comparison_note],
        **PRIVATE,
    )
    displayed.change(
        lambda run, key: comparison_view(run, _read_run(key)) if run and key else comparison_view(None, None),
        [displayed, compare],
        [comparison, comparison_note],
        **PRIVATE,
    )
    return displayed, selection_state


def metadata_tab(collection, outputs_root):
    data = gr.State([])
    directory = gr.State("")
    gr.Markdown("入力画像から生成条件とComfyUIのJSONを取り出します。")
    read = gr.Button("生成設定を読む", variant="primary")
    note = gr.Textbox(
        value="先にフォルダを読み込むか、個別画像を追加してください。", show_label=False, interactive=False
    )
    table = gr.Dataframe(
        headers=["#", "画像", "記録", "seed", "steps", "sampler", "モデル"],
        value=[],
        type="array",
        interactive=False,
        column_count=(7, "fixed"),
        wrap=True,
    )
    image = gr.Image(label="選択画像", type="pil", interactive=False, height=360, visible=False)
    settings = gr.Textbox(label="残っている生成条件", interactive=False, lines=6)
    with gr.Row():
        workflow = gr.DownloadButton("workflow JSONを保存", interactive=False)
        prompt = gr.DownloadButton("API prompt JSONを保存", interactive=False)

    @guarded
    def inspect(collection):
        if not (collection or {}).get("items"):
            raise ClefError("フォルダを読み込むか、個別画像を追加してください。")
        target = outputs_root / "metadata" / uuid.uuid4().hex[:12]
        items, rows = [], []
        for index, item in enumerate(collection["items"]):
            if sha256(item["path"]) != item["sha256"]:
                raise ClefError("読み込み後に画像が変更されています。フォルダを読み直してください。")
            metadata = generation_metadata(item["path"])
            settings = metadata["settings"]
            present = [key for key in ("workflow", "prompt") if metadata[key] is not None]
            items.append({**item, "metadata": metadata})
            rows.append(
                [
                    str(index + 1),
                    item["name"],
                    " + ".join(present) if present else "記録なし",
                    settings.get("seed", ""),
                    settings.get("steps", ""),
                    settings.get("sampler_name", ""),
                    settings.get("unet_name", settings.get("ckpt_name", "")),
                ]
            )
        return (
            items,
            str(target),
            rows,
            f"{len(items)}件確認 · workflow {sum(item['metadata']['workflow'] is not None for item in items)}件 · API prompt {sum(item['metadata']['prompt'] is not None for item in items)}件",
        )

    read.click(inspect, collection, [data, directory, table, note], **PRIVATE)

    @guarded
    def choose(items, directory, event: gr.SelectData):
        index = int(event.row_value[0]) - 1
        if not 0 <= index < len(items):
            raise ClefError("現在の一覧から選択してください。")
        item = items[index]
        text, files = saved_metadata(item["path"], directory, item["id"], item["sha256"])
        preview = thumbnails({"items": [item]}, per_page=1)[0][0]
        return (
            gr.update(value=preview, visible=True),
            text,
            gr.update(value=files.get("workflow"), interactive="workflow" in files),
            gr.update(value=files.get("prompt"), interactive="prompt" in files),
        )

    table.select(choose, [data, directory], [image, settings, workflow, prompt], **PRIVATE)
    collection.change(
        lambda: (
            [],
            "",
            [],
            "入力を変更しました。生成設定を読み直してください。",
            gr.update(value=None, visible=False),
            "",
            gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False),
        ),
        outputs=[data, directory, table, note, image, settings, workflow, prompt],
        **PRIVATE,
    )


def build(outputs=OUTPUTS):
    with gr.Column(elem_id="clef-studio") as studio:
        gr.Markdown("# Clef\n画像・文章・JSONデータを、共通の質問で判定します。")
        with gr.Row(elem_classes=["clef-toolbar"]):
            profile = gr.Dropdown(
                [(value["label"], key) for key, value in PROFILES.items()],
                value="flash-int8",
                label="実行モデル",
                scale=4,
            )
            unload = gr.Button("Clefモデルを解放", size="sm", scale=1)
        with gr.Accordion("実行環境", open=False):
            gr.Textbox(value=environment_status(), show_label=False, interactive=False)

        def release():
            gr.Info(STUDIO.unload())

        unload.click(release, queue=False, **PRIVATE)
        workbench(profile, outputs)

    def exclude_defaults(block):
        block.do_not_save_to_config = True
        for child in getattr(block, "children", []):
            exclude_defaults(child)

    exclude_defaults(studio)
    return studio
