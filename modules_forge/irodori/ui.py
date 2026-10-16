"""Native speech controls; all inference stays in the managed worker."""

import gradio as gr

from modules.aikimi_status import studio_status_html

from .core import IrodoriError, ready
from .service import SERVICE

FIELDS = (
    "text",
    "caption",
    "precision",
    "steps",
    "cfg_text",
    "cfg_caption",
    "cfg_speaker",
    "seed",
    "seconds",
    "duration_scale",
)
PRIVATE = {"api_name": False}


def owner(request):
    return request.session_hash or ""


def submit(
    text,
    caption,
    precision,
    steps,
    cfg_text,
    cfg_caption,
    cfg_speaker,
    seed,
    seconds,
    duration_scale,
    references,
    request: gr.Request,
):
    values = (text, caption, precision, steps, cfg_text, cfg_caption, cfg_speaker, seed, seconds, duration_scale)
    try:
        identifier = SERVICE.start(dict(zip(FIELDS, values, strict=True)), references, owner(request))
        return identifier, "生成中…", assistant_state(identifier, {"status": "running", "error": ""})
    except (IrodoriError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def choices(request):
    return [
        (identifier[:12], identifier)
        for identifier, job in reversed(list(SERVICE.jobs.items()))
        if job["owner"] == owner(request) and job["status"] == "complete"
    ]


def status_message(state):
    if state["status"] == "complete":
        result = state["result"]
        watermark = "透かし適用" if result["watermarked"] else "透かし未適用（モデルを利用できませんでした）"
        return f"完了 · {result['seconds']:.2f}秒 · 48kHz · seed {result['seed']} · {watermark}"
    if state["status"] == "cancelled":
        return "中断しました。"
    if state["status"] == "error":
        return "生成に失敗しました。" + state.get("error", "")
    if state["status"] == "stopping":
        return "停止を確認中…"
    return "生成中…"


def assistant_state(identifier, state):
    return studio_status_html(
        "irodori",
        "running" if state["status"] == "stopping" else state["status"],
        status_message(state),
        job_id=identifier,
        model_name="Irodori TTS",
        result_id="irodori-audio" if state["status"] == "complete" else "",
    )


def poll(identifier, request: gr.Request):
    if not identifier:
        return (gr.skip(),) * 6
    try:
        state = SERVICE.poll(identifier, owner(request))
    except IrodoriError as exc:
        return "", str(exc), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    complete = state["status"] == "complete"
    finished = state["status"] in {"complete", "error", "cancelled"}
    return (
        "" if finished else identifier,
        status_message(state),
        state["audio"] if complete else gr.skip(),
        state["files"] if complete else gr.skip(),
        gr.update(choices=choices(request), value=identifier if complete else None) if finished else gr.skip(),
        assistant_state(identifier, state),
    )


def cancel(identifier, request: gr.Request):
    if not identifier:
        return "実行中の処理はありません。"
    try:
        return SERVICE.cancel(identifier, owner(request))
    except IrodoriError as exc:
        raise gr.Error(str(exc)) from exc


def history(identifier, request: gr.Request):
    if not identifier:
        return (gr.skip(),) * 3
    try:
        state = SERVICE.poll(identifier, owner(request))
        if state["status"] != "complete":
            return (gr.skip(),) * 3
        return state["audio"], state["files"], status_message(state)
    except IrodoriError as exc:
        raise gr.Error(str(exc)) from exc


def build_ui():
    with gr.Blocks(analytics_enabled=False, elem_id="aikimi-irodori") as tab:
        gr.Markdown("日本語の文章から音声を生成します。声の説明や、使用許可のある参照音声を追加できます。")
        if not ready():
            gr.Markdown("初回は **`aikimi-irodori-setup.bat`** を実行してからForge Neoを再起動してください。")
        identifier = gr.State("")
        assistant = gr.HTML(visible="hidden", elem_id="irodori-assistant-state")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="読み上げ文",
                    lines=6,
                    placeholder="こんにちは。今日はどんな一日でしたか。",
                    elem_id="irodori-text",
                )
                caption = gr.Textbox(
                    label="声の説明（任意）",
                    lines=2,
                    placeholder="落ち着いた、やわらかい声。自然な速さで。",
                    elem_id="irodori-caption",
                )
                with gr.Accordion("参照音声（任意）", open=False):
                    references = gr.File(
                        label="同じ話者・PCM WAV・合計120秒まで",
                        file_count="multiple",
                        file_types=[".wav"],
                        type="filepath",
                        elem_id="irodori-references",
                    )
                with gr.Accordion("詳細設定", open=False):
                    precision = gr.Dropdown(
                        label="モデルの精度（codecはFP32）",
                        choices=[("BF16", "bf16"), ("FP32", "fp32")],
                        value="bf16",
                        elem_id="irodori-precision",
                    )
                    steps = gr.Slider(1, 100, value=40, step=1, label="ステップ")
                    cfg_text = gr.Slider(0, 15, value=3, step=0.1, label="テキストCFG")
                    cfg_caption = gr.Slider(0, 15, value=3, step=0.1, label="声の説明CFG（説明がある場合のみ）")
                    cfg_speaker = gr.Slider(0, 15, value=5, step=0.1, label="参照音声CFG（参照がある場合のみ）")
                    seed = gr.Number(value=-1, precision=0, label="seed（-1でランダム）")
                    seconds = gr.Number(value=0, label="目標の長さ（0で自動・最大30秒・末尾無音を除去）")
                    duration_scale = gr.Slider(0.5, 2, value=1, step=0.05, label="長さ倍率（自動のときだけ）")
                with gr.Row():
                    generate = gr.Button("生成", variant="primary", elem_id="irodori-generate")
                    stop = gr.Button("中断", elem_id="irodori-cancel")
            with gr.Column():
                status = gr.Textbox(label="状態", value="読み上げ文を入力してください。", interactive=False)
                audio = gr.Audio(label="生成音声", type="filepath", interactive=False, elem_id="irodori-audio")
                files = gr.File(label="音声と生成条件", file_count="multiple", interactive=False)
                previous = gr.Dropdown(label="このセッションの履歴", choices=[], value=None)
        inputs = [
            text,
            caption,
            precision,
            steps,
            cfg_text,
            cfg_caption,
            cfg_speaker,
            seed,
            seconds,
            duration_scale,
            references,
        ]
        generate.click(
            submit,
            inputs,
            [identifier, status, assistant],
            concurrency_limit=1,
            concurrency_id="irodori-events",
            **PRIVATE,
        )
        stop.click(cancel, [identifier], [status], queue=False, **PRIVATE)
        timer = gr.Timer(0.75)
        timer.tick(
            poll,
            [identifier],
            [identifier, status, audio, files, previous, assistant],
            concurrency_limit=1,
            concurrency_id="irodori-events",
            **PRIVATE,
        )
        previous.change(history, [previous], [audio, files, status], **PRIVATE)
    return tab
