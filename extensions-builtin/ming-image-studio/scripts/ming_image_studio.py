"""A focused design workspace for Ming Image's text and native RGBA output."""

from __future__ import annotations

import html
import json
import queue
import threading
import uuid
from dataclasses import replace
from functools import partial

import gradio as gr

from modules import script_callbacks
from modules.aikimi_status import studio_status_html
from modules.gradio_compat import keep_hidden_component_mounted
from modules_forge import ming_image_studio as studio

PRESETS = {
    "正方形 · 1024×1024": (1024, 1024),
    "横長 · 1152×864": (1152, 864),
    "縦長 · 864×1152": (864, 1152),
    "ワイド · 1344×768": (1344, 768),
    "縦ワイド · 768×1344": (768, 1344),
    "正方形 · 2048×2048": (2048, 2048),
    "横長 · 2304×1728": (2304, 1728),
    "縦長 · 1728×2304": (1728, 2304),
    "ワイド · 2560×1440": (2560, 1440),
    "カスタム": None,
}
EXAMPLE = (
    "A botanical exhibition poster. Cream paper background, a large dark green fern in the lower half, "
    'generous margins, elegant editorial typography. At the top, the headline "BOTANICA"; '
    'below it, the small subtitle "A quiet collection". Two colors: forest green and warm cream.'
)


def _integer(value):
    try:
        result = int(value)
        if isinstance(value, bool) or str(value).strip() not in {str(result), str(float(result))}:
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise studio.MingImageError("寸法・Steps・Seedには整数を指定してください。") from exc


def _request(
    prompt,
    text,
    transparent,
    width,
    height,
    steps,
    seed,
    precision="int8",
    model_path="",
    text_encoder_path="",
    vae_path="",
    loras=None,
    strengths=None,
):
    from modules_forge.local_assets import lora_settings

    if studio.is_json_prompt(prompt or ""):
        text, transparent = "", False
    result = studio.MingImageRequest(
        prompt or "",
        text or "",
        bool(transparent),
        _integer(width),
        _integer(height),
        _integer(steps),
        _integer(seed),
        precision=precision,
        model_path=model_path or "",
        text_encoder_path=text_encoder_path or "",
        vae_path=vae_path or "",
        loras=lora_settings(loras, strengths),
    )
    result.validate()
    return result


def _prompt_mode(prompt):
    structured = studio.is_json_prompt(prompt or "")
    if structured:
        try:
            value = json.loads(prompt)
            hint = "JSONをそのまま送信します。追加の文字・透過指定は適用しません（入力は保持）。"
            if not isinstance(value, dict):
                hint = "JSONオブジェクトを指定してください。"
        except json.JSONDecodeError as exc:
            hint = f"JSONの{exc.lineno}行目・{exc.colno}文字目を確認してください。"
        return hint, gr.update(interactive=False), gr.update(interactive=False)
    return "自然文・JSONを入力できます。", gr.update(interactive=True), gr.update(interactive=True)


def _summary(width, height, steps, transparent, prompt):
    if any(value is None for value in (width, height, steps)):
        return "寸法・Stepsを入力してください。"
    summary = f"{width:g} × {height:g} px · {steps:g} steps"
    if transparent and not studio.is_json_prompt(prompt or ""):
        summary += " · 透過を指示"
        if width * height <= 1024**2:
            summary += "（公式推奨は2048相当）"
    return summary


def _caption(data):
    fraction = data.get("near_transparent_fraction", 0)
    alpha = ""
    if fraction >= 0.01:
        alpha = f" · 透明に近い部分 約{fraction:.0%}"
    elif data["request"]["transparent"]:
        alpha = " · 透過を指示しましたが、ほぼ不透明です（透明に近い部分1%未満）"
    precision = " · 本体W4A8" if data["request"].get("precision") == "w4a8" else ""
    return f"{data['width']} × {data['height']} px{alpha} · Seed {data['seed']}{precision}"


def _can_double(record):
    if not record:
        return False
    data = record["metadata"]
    return data["width"] * data["height"] <= 1024**2 and max(data["width"], data["height"]) <= 2048


def _result_controls(record):
    data = record.get("metadata", {})
    label = (
        f"この画像の条件で {data['width'] * 2}×{data['height'] * 2} を再生成"
        if _can_double(record)
        else "2倍での再生成は2048相当まで"
    )
    return (
        gr.update(visible=bool(record)),
        gr.update(visible=bool(data.get("has_transparency"))),
        gr.update(value=label, interactive=_can_double(record)),
    )


def _restore(record):
    if not record:
        return (gr.update(),) * 10
    data = record["metadata"]
    request = data["request"]
    preset = next((k for k, v in PRESETS.items() if v == (request["width"], request["height"])), "カスタム")
    return (
        request["prompt"],
        request["text"],
        request["transparent"],
        request["width"],
        request["height"],
        request["steps"],
        str(data["seed"]),
        preset,
        "表示中の画像の条件を戻しました。",
        request.get("precision", "int8"),
    )


def _run(request_factory, previous):
    notification_id = uuid.uuid4().hex

    def status(stage, message):
        return studio_status_html(
            "ming_image",
            stage,
            message,
            job_id=notification_id,
            model_name="Ming Image",
            result_id="ming-result" if stage == "complete" else "",
            visible=True,
        )

    unchanged = gr.update
    yield (
        status("prepare", "入力を確認しています。"),
        unchanged(),
        unchanged(),
        unchanged(),
        {},
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        unchanged(),
        unchanged(),
        gr.update(interactive=False),
    )
    try:
        request = request_factory()
        for event in studio.run_generation(request):
            complete = event["stage"] == "complete"
            record = (
                {"path": event["path"], "files": event["files"], "metadata": event["metadata"]} if complete else None
            )
            job = {"prompt_id": event["prompt_id"]} if event.get("prompt_id") and not complete else {}
            message = event["message"]
            caption = unchanged()
            if complete:
                data = record["metadata"]
                caption = _caption(data)
                message = "完了 · " + caption
            if "elapsed" in event:
                message += f" · {event['elapsed']:.0f}秒"
            yield (
                status(event["stage"], message),
                event.get("path", unchanged()),
                event.get("files", unchanged()),
                record if complete else unchanged(),
                job,
                gr.update(interactive=complete),
                gr.update(interactive=bool(job)),
                gr.update(interactive=complete),
                gr.update(interactive=_can_double(record)),
                event["metadata"]["effective_prompt"] if complete else unchanged(),
                caption,
                gr.update(interactive=complete),
            )
    except Exception as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPUメモリが不足しました。寸法を1024相当に下げて再試行してください。"
        stage = "cancelled" if isinstance(exc, studio.MingImageCancelled) else "error"
        yield (
            status(stage, message),
            unchanged(),
            unchanged(),
            unchanged(),
            {},
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=bool(previous)),
            gr.update(interactive=_can_double(previous)),
            unchanged(),
            unchanged(),
            gr.update(interactive=True),
        )


def _generate(
    prompt,
    text,
    transparent,
    width,
    height,
    steps,
    seed,
    previous,
    precision="int8",
    model_path="",
    text_encoder_path="",
    vae_path="",
    loras=None,
    strengths=None,
):
    yield from _run(
        lambda: _request(
            prompt,
            text,
            transparent,
            width,
            height,
            steps,
            seed,
            precision,
            model_path,
            text_encoder_path,
            vae_path,
            loras,
            strengths,
        ),
        previous,
    )


def _double(previous):
    def request():
        if not _can_double(previous):
            raise studio.MingImageError("1024相当以下の生成結果を選んでください。")
        data = previous["metadata"]
        return replace(
            studio.MingImageRequest(**data["request"]),
            width=data["width"] * 2,
            height=data["height"] * 2,
            seed=data["seed"],
            parent=data["prompt_id"],
        )

    yield from _run(request, previous)


def _cancel(job):
    if job and job.get("prompt_id"):
        try:
            studio.cancel_generation(job["prompt_id"])
            return "停止を要求しました。処理の終了を待っています。"
        except Exception as exc:
            return "停止要求を送信できません: " + html.escape(str(exc))
    return "実行中のジョブはありません。"


def _readiness(precision="int8", model_path="", text_encoder_path="", vae_path=""):
    from tools.setup_ming_image import profiles, runtime_ready

    size_gb = sum(item.size for item in profiles(precision)["models"].artifacts) / 1_000_000_000
    label = f"Ming Imageを準備（{precision.upper()} · 約{size_gb:.1f}GB）"
    try:
        if model_path or text_encoder_path or vae_path:
            from modules_forge.ming_local import selected_assets

            if model_path:
                label = "実行環境・共通部品を準備（標準本体なし）"
            selected_assets(
                studio.MingImageRequest(
                    "check",
                    precision=precision,
                    model_path=model_path or "",
                    text_encoder_path=text_encoder_path or "",
                    vae_path=vae_path or "",
                ),
                studio.runtime_root(),
                verify_hash=False,
            )
            ready = runtime_ready()
        else:
            ready = runtime_ready() and studio.model_ready(
                studio.runtime_root(), verify_hash=False, precision=precision
            )
        message = (
            (
                "選択したローカルモデルを使用します。"
                if model_path
                else f"本体{precision.upper()}は導入済み。専用環境は生成時に自動起動します。"
            )
            if ready
            else f"Ming Image（本体{precision.upper()}）の準備が必要です。"
        )
    except (OSError, ValueError) as exc:
        ready, message = False, "準備状態を確認できません: " + str(exc)
    return (
        message,
        gr.update(visible=keep_hidden_component_mounted(ready), interactive=ready),
        gr.update(value=label, visible=keep_hidden_component_mounted(not ready), interactive=True),
        html.escape(message) if not ready else "",
        gr.update(interactive=not bool(model_path)),
    )


def _setup(precision="int8", *, repair=False, components_only=False, runtime_only=False):
    yield "専用環境とモデルを準備しています。取得済みのファイルは再利用します。"
    updates = queue.Queue()

    def prepare():
        try:
            from tools.setup_ming_image import run

            bridge = studio._bridge()
            active = bridge.server_runtime_root(studio.SERVER_URL)
            with bridge.runtime_setup_session(active or studio.runtime_root(), studio.SERVER_URL):
                result = run(
                    repair=repair,
                    precision=precision,
                    progress=updates.put,
                    **({"components_only": True} if components_only else {}),
                    **({"runtime_only": True} if runtime_only else {}),
                )
            if not result["ok"]:
                raise studio.MingImageError("導入後の検証に失敗しました。")
            updates.put("準備できました。プロンプトを入力して生成できます。")
        except Exception as exc:
            updates.put("準備を完了できませんでした: " + str(exc))
        finally:
            updates.put(None)

    threading.Thread(target=prepare, name="ming-setup", daemon=True).start()
    while True:
        message = updates.get()
        if message is None:
            return
        for filename, label in (
            (studio.MODEL, "画像生成モデル"),
            (studio.W4A8_MODEL, "画像生成モデル（W4A8）"),
            (studio.W4A8_MODEL.replace(".safetensors", ".json"), "モデルの変換記録"),
            (studio.ENCODER, "テキストエンコーダー"),
            (studio.VAE, "VAE"),
        ):
            prefix = "ming-image-" + filename + ":"
            if message == prefix + " verified and installed":
                message = label + "の取得・検証が完了しました。"
            else:
                message = message.replace(prefix, label + "を取得中 ·")
        yield message


def _prepare(precision="int8", model_path="", text_encoder_path="", vae_path="", *, repair=False):
    last_message = ""
    options = {}
    if model_path:
        options["runtime_only" if text_encoder_path and vae_path else "components_only"] = True
    for message in _setup(precision, repair=repair, **options):
        last_message = message
        yield (
            message,
            gr.update(visible=keep_hidden_component_mounted(False)),
            gr.update(interactive=False),
            html.escape(message),
            gr.update(interactive=False),
        )
    message, generate, prepare, status, selector = _readiness(precision, model_path, text_encoder_path, vae_path)
    if last_message.startswith("準備を完了できませんでした:"):
        message, status = last_message, html.escape(last_message)
    yield message, generate, prepare, status, selector


def _restore_assets(record):
    request = (record or {}).get("metadata", {}).get("request", {})
    loras = request.get("loras", [])
    return (
        *_restore(record),
        request.get("model_path", ""),
        request.get("text_encoder_path", ""),
        request.get("vae_path", ""),
        [x["name"] for x in loras],
        [[x["name"], x["strength"]] for x in loras],
    )


def on_ui_tabs():
    from modules_forge import local_assets

    saved = local_assets.selection("ming")
    root = studio.runtime_root()
    initial_precision = "int8"
    if not studio.model_ready(root, verify_hash=False) and studio.model_ready(
        root, verify_hash=False, precision="w4a8"
    ):
        initial_precision = "w4a8"
    initial_precision = saved.get("precision", initial_precision)
    saved_loras = saved.get("loras", [])
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row(elem_id="ming-workspace"):
            with gr.Column(scale=1, min_width=300, elem_id="ming-controls"):
                gr.Markdown("### Ming Image Design", elem_id="ming-heading")
                with gr.Accordion("モデル・LoRA", open=False, elem_id="ming-models") as model_section:
                    model_path = gr.Dropdown(
                        label="本体モデル",
                        choices=[
                            ("標準モデル", ""),
                            *local_assets.choices("ming_model", [root / "models/diffusion_models"]),
                        ],
                        value=saved.get("model_path", ""),
                        allow_custom_value=True,
                        elem_id="ming-local-model",
                        info="一覧から選択、または手元のMing .safetensorsのフルパスを貼り付けてEnter。",
                    )
                    loras = gr.Dropdown(
                        label="追加LoRA（複数選択）",
                        choices=local_assets.choices(
                            "ming_lora", [root / "models/loras", local_assets.ROOT / "models/Lora/Ming"]
                        ),
                        value=[x["name"] for x in saved_loras],
                        multiselect=True,
                        allow_custom_value=True,
                        elem_id="ming-loras",
                    )
                    strengths = gr.Dataframe(
                        headers=["LoRA", "強度"],
                        datatype=["str", "number"],
                        type="array",
                        value=local_assets.lora_rows(
                            [x["name"] for x in saved_loras], [[x["name"], x["strength"]] for x in saved_loras]
                        ),
                        static_columns=[0],
                        column_count=(2, "fixed"),
                        row_count=len(saved_loras),
                        max_chars=64,
                        interactive=True,
                        elem_id="ming-lora-strengths",
                        label="強度 −2〜2 · 0で無効",
                        visible=False,
                    )
                    with gr.Accordion("共通部品を指定 · 任意", open=False):
                        text_encoder_path = gr.Textbox(
                            label="テキストエンコーダー",
                            value=saved.get("text_encoder_path", ""),
                            placeholder="空欄: 標準の共通部品",
                            elem_id="ming-local-encoder",
                        )
                        vae_path = gr.Textbox(
                            label="VAE",
                            value=saved.get("vae_path", ""),
                            placeholder="空欄: 標準の共通部品",
                            elem_id="ming-local-vae",
                        )
                    refresh_assets = gr.Button("モデル・LoRA一覧を更新", size="sm")
                prompt = gr.Textbox(label="プロンプト", lines=7, placeholder=EXAMPLE, elem_id="ming-prompt")
                hint = gr.Markdown("自然文・JSONを入力できます。", elem_id="ming-input-hint")
                with gr.Accordion("画像に載せる文字 · 任意", open=False):
                    text = gr.Textbox(label="1行に1つ。本文に書いた文字は追加不要です。", lines=2, elem_id="ming-text")
                transparent = gr.Checkbox(label="背景の透過を指示（RGBA PNG）", value=False, elem_id="ming-transparent")
                preset = gr.Dropdown(
                    label="キャンバス",
                    choices=list(PRESETS),
                    value="正方形 · 1024×1024",
                    filterable=False,
                    elem_id="ming-preset",
                )
                with gr.Row():
                    width = gr.Number(
                        label="幅 px", value=1024, precision=0, step=16, min_width=85, elem_id="ming-width"
                    )
                    swap = gr.Button("⇄", size="sm", min_width=40, scale=0, elem_id="ming-swap")
                    height = gr.Number(
                        label="高さ px", value=1024, precision=0, step=16, min_width=85, elem_id="ming-height"
                    )
                with gr.Accordion("詳細設定", open=False):
                    seed = gr.Textbox(label="Seed（-1でランダム）", value="-1", elem_id="ming-seed")
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=12, step=1)
                    gr.Markdown("既定は12 steps・CFG 1。まず1024相当で試せます。")
                    precision = gr.Radio(
                        choices=[("INT8（標準）", "int8"), ("W4A8（省メモリ・試験版）", "w4a8")],
                        value=initial_precision,
                        label="標準モデルの精度",
                        elem_id="ming-precision",
                    )
                summary = gr.Markdown("1024 × 1024 px · 12 steps", elem_id="ming-summary")
                with gr.Row(elem_id="ming-actions"):
                    generate = gr.Button("デザインを生成", variant="primary", elem_id="ming-generate")
                    cancel = gr.Button("停止", interactive=False, scale=0, min_width=65, elem_id="ming-cancel")
                prepare = gr.Button(
                    "Ming Imageを準備",
                    variant="primary",
                    visible=keep_hidden_component_mounted(False),
                    elem_id="ming-prepare",
                )
                status = gr.HTML("", elem_id="ming-status")
                with gr.Accordion("実行環境とモデル", open=False):
                    setup_status = gr.Markdown("準備状態を確認しています…")
                    setup = gr.Button("破損したモデルを退避して修復")
                    refresh = gr.Button("準備状態を更新")
                    gr.Markdown("テキストW4A8 · 専用ComfyUI。保存先: `outputs/ming-image/`")
            with gr.Column(scale=2, min_width=300, elem_id="ming-output"):
                background = gr.Radio(
                    ["チェック", "白", "黒"],
                    value="チェック",
                    label="プレビュー背景",
                    visible=False,
                    elem_id="ming-background",
                )
                result = gr.Image(
                    label="生成結果",
                    type="filepath",
                    format="png",
                    image_mode="RGBA",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                    height=570,
                    elem_id="ming-result",
                )
                caption = gr.Markdown("生成結果がここに表示されます。", elem_id="ming-caption")
                with gr.Group(visible=False) as result_tools:
                    with gr.Row():
                        restore = gr.Button("この画像の条件をフォームに戻す", interactive=False, elem_id="ming-restore")
                        double = gr.Button("この画像の条件で再生成", interactive=False, elem_id="ming-double")
                    gr.Markdown("解像度を変えると、構図や文字も変わる場合があります。", elem_id="ming-double-note")
                    files = gr.File(
                        label="PNG・生成条件・プロンプトを保存",
                        file_count="multiple",
                        interactive=False,
                        elem_id="ming-files",
                    )
                    with gr.Accordion("この画像に使用したプロンプト", open=False):
                        effective = gr.Textbox(label="モデルへ送信した文", lines=5, interactive=False)
        record = gr.State({})
        job = gr.State({})
        private = {"api_visibility": "private", "show_progress": "hidden"}
        source_inputs = [precision, model_path, text_encoder_path, vae_path]
        for component in [*source_inputs, loras]:
            component.do_not_save_to_config = True  # Do not replace the asset library with old UI defaults.
        inputs = [
            prompt,
            text,
            transparent,
            width,
            height,
            steps,
            seed,
            record,
            precision,
            model_path,
            text_encoder_path,
            vae_path,
            loras,
            strengths,
        ]
        outputs = [status, result, files, record, job, generate, cancel, restore, double, effective, caption, precision]
        prompt.change(_prompt_mode, inputs=prompt, outputs=[hint, text, transparent], queue=False, **private)
        preset.change(
            lambda p: PRESETS.get(p) or (gr.update(), gr.update()),
            inputs=preset,
            outputs=[width, height],
            queue=False,
            **private,
        )
        swap.click(
            lambda w, h: (h, w, "カスタム"),
            inputs=[width, height],
            outputs=[width, height, preset],
            queue=False,
            **private,
        )
        for control in (width, height):
            control.input(lambda: "カスタム", outputs=preset, queue=False, **private)
        for control in (width, height, steps, transparent, prompt):
            control.change(
                _summary, inputs=[width, height, steps, transparent, prompt], outputs=summary, queue=False, **private
            )
        restore.click(
            _restore_assets,
            inputs=record,
            outputs=[
                prompt,
                text,
                transparent,
                width,
                height,
                steps,
                seed,
                preset,
                hint,
                precision,
                model_path,
                text_encoder_path,
                vae_path,
                loras,
                strengths,
            ],
            queue=False,
            **private,
        )
        generate.click(
            _generate,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
            concurrency_id="minimax-h3-generation",
            trigger_mode="once",
            **private,
        )
        double.click(
            _double,
            inputs=record,
            outputs=outputs,
            concurrency_limit=1,
            concurrency_id="minimax-h3-generation",
            trigger_mode="once",
            **private,
        )
        cancel.click(_cancel, inputs=job, outputs=status, queue=False, **private)
        readiness_outputs = [setup_status, generate, prepare, status, precision]
        for control, repair in ((setup, True), (prepare, False)):
            control.click(
                partial(_prepare, repair=repair),
                inputs=source_inputs,
                outputs=readiness_outputs,
                concurrency_id="h3-runtime-control",
                concurrency_limit=1,
                **private,
            )
        refresh.click(_readiness, inputs=source_inputs, outputs=readiness_outputs, **private)
        for control in source_inputs:
            control.change(_readiness, inputs=source_inputs, outputs=readiness_outputs, queue=False, **private)
        loras.change(
            lambda names, rows: gr.update(value=local_assets.lora_rows(names, rows), visible=bool(names)),
            inputs=[loras, strengths],
            outputs=strengths,
            queue=False,
            **private,
        )
        model_section.expand(
            lambda names, rows: gr.update(value=local_assets.lora_rows(names, rows), visible=bool(names)),
            inputs=[loras, strengths],
            outputs=strengths,
            queue=False,
            **private,
        )

        def refresh_choices(model, selected):
            models = [("標準モデル", ""), *local_assets.choices("ming_model", [root / "models/diffusion_models"])]
            adapters = local_assets.choices(
                "ming_lora", [root / "models/loras", local_assets.ROOT / "models/Lora/Ming"]
            )
            return gr.update(choices=models, value=model), gr.update(choices=adapters, value=selected)

        refresh_assets.click(refresh_choices, inputs=[model_path, loras], outputs=[model_path, loras], **private)
        record.change(_result_controls, inputs=record, outputs=[result_tools, background, double], **private)
        tab.load(_readiness, inputs=source_inputs, outputs=readiness_outputs, **private)
    return [(tab, "Ming Image", "ming_image_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
