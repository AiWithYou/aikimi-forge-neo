"""MOSS-Image-SFT text generation in the shared image workspace."""

import gradio as gr

from modules import script_callbacks
from modules_forge import local_assets
from modules_forge.moss_image import lora as local_lora
from modules_forge.moss_image.core import MODEL_ROOT
from modules_forge.moss_image.service import SERVICE


def start(
    state,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    shift,
    seed,
    skip_layer,
    precision="int8",
    text_precision="int8",
    loras=None,
    lora_rows=None,
):
    state = dict(state or {})
    if state.get("job"):
        raise gr.Error("生成中です。完了または中断を待ってください。")
    try:
        state["job"] = SERVICE.start(
            dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                shift=shift,
                seed=seed,
                skip_layer=skip_layer,
                precision=precision,
                text_precision=text_precision,
                loras=list(local_assets.lora_settings(loras, lora_rows)),
            )
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return (
        state,
        "生成を開始しました。",
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(active=True),
    )


def select_loras(names, rows):
    try:
        for name in names or []:
            local_lora.header_info(local_lora.resolve(MODEL_ROOT, name))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return gr.update(value=local_assets.lora_rows(names, rows), visible=True if names else "hidden")


def refresh_loras(names):
    choices = local_lora.inventory(MODEL_ROOT)
    return gr.update(choices=choices, value=[name for name in names or [] if name in choices])


def poll(state):
    state = dict(state or {})
    identifier = state.get("job")
    if not identifier:
        return state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(active=False)
    result = SERVICE.poll(identifier)
    status = result["status"]
    terminal = status in {"complete", "error", "cancelled", "idle"}
    if terminal:
        state.pop("job", None)
    stage = result.get("stage")
    message = {
        "verifying": "生成準備中…",
        "initializing": "生成準備中…",
        "loading_text_encoder": "生成準備中…",
        "loading_diffusion": "生成準備中…",
        "encoding_prompt": "プロンプトを処理中…",
        "sampling": "画像を生成中…",
        "decoding": "画像を仕上げ中…",
        "saving": "保存中…",
        "complete": "保存中…",
    }.get(stage, "生成中…")
    current, total = result.get("current"), result.get("total")
    if stage == "sampling" and type(current) is int and type(total) is int and 0 <= current <= total:
        message += f" {current}/{total}"
    image, files = gr.skip(), gr.skip()
    if status == "complete":
        message = f"完了 · {result['width']}×{result['height']} · seed {result['seed']}"
        image, files = result["image_path"], result["files"]
    elif status in {"error", "cancelled", "stopping", "idle"}:
        message = (
            result.get("error")
            or {
                "error": "生成に失敗しました。",
                "cancelled": "中断しました。",
                "stopping": "停止を確認中…",
                "idle": "実行中の処理はありません。",
            }[status]
        )
    return (
        state,
        message,
        image,
        files,
        gr.update(interactive=terminal),
        gr.update(interactive=not terminal),
        gr.update(active=not terminal),
    )


def on_ui_tabs():
    from modules_forge.image_workspace import ImageWorkspace

    with gr.Blocks(analytics_enabled=False) as blocks:
        model = gr.Textbox(value="MOSS-Image-SFT", label="モデル", interactive=False, elem_id="moss-image-model-name")
        model.do_not_save_to_config = True
        with gr.Row() as precisions:
            precision = gr.Dropdown(
                [("INT8 ConvRot", "int8"), ("BF16", "bf16")],
                value="int8",
                label="画像生成本体",
                elem_id="moss-image-precision",
            )
            text_precision = gr.Dropdown(
                [("INT8 ConvRot", "int8"), ("W4A8", "w4a8"), ("BF16", "bf16")],
                value="int8",
                label="テキストエンコーダー",
                elem_id="moss-image-text-precision",
            )
            gr.Textbox(value="FP32", label="VAE", interactive=False, elem_id="moss-image-vae-precision")
        with gr.Column() as lora_controls:
            with gr.Row():
                loras = gr.Dropdown(
                    local_lora.inventory(MODEL_ROOT),
                    value=[],
                    multiselect=True,
                    label="LoRA",
                    elem_id="moss-image-loras",
                )
                lora_refresh = gr.Button("一覧を更新", size="sm", scale=0)
            lora_rows = gr.Dataframe(
                value=[],
                headers=["LoRA", "強度"],
                datatype=["str", "number"],
                static_columns=[0],
                type="array",
                column_count=(2, "fixed"),
                row_count=0,
                interactive=True,
                visible="hidden",
                label="LoRA強度 −2〜2 · 0で無効",
                elem_id="moss-image-lora-strengths",
            )
            loras.do_not_save_to_config = True
            lora_rows.do_not_save_to_config = True
        prompt = gr.Textbox(label="プロンプト", lines=4, placeholder="生成したい画像を自然文、タグ列、JSONで記述")
        negative = gr.Textbox(label="ネガティブプロンプト", lines=2)
        run = gr.Button("生成", variant="primary")
        cancel = gr.Button("中断", interactive=False)
        with gr.Column() as settings:
            with gr.Row():
                width = gr.Number(value=1024, precision=0, minimum=512, maximum=2048, step=16, label="幅")
                height = gr.Number(value=1024, precision=0, minimum=512, maximum=2048, step=16, label="高さ")
            steps = gr.Slider(1, 200, value=50, step=1, label="Steps")
            cfg = gr.Slider(0, 100, value=3, step=0.1, label="CFG")
            shift = gr.Slider(0.1, 100, value=5, step=0.1, label="Shift")
            seed = gr.Number(value=-1, precision=0, label="Seed")
            skip_layer = gr.Checkbox(value=True, label="負条件の22層をスキップ", elem_id="moss-image-skip-layer")
        with gr.Column() as output:
            status = gr.Markdown("")
            image = gr.Image(label="生成結果", type="filepath", format="png", interactive=False)
            files = gr.File(label="PNG・生成条件", file_count="multiple", interactive=False)
        ImageWorkspace(
            blocks,
            "moss-image",
            models=[model, precisions],
            prompts=[prompt, negative],
            actions=[run, cancel],
            settings=[settings],
            outputs=[output],
            panels={"LoRA": [lora_controls]},
            badges={"LoRA": loras},
        )
        state = gr.State({})
        timer = gr.Timer(0.5, active=False)
        run.click(
            start,
            inputs=[
                state,
                prompt,
                negative,
                width,
                height,
                steps,
                cfg,
                shift,
                seed,
                skip_layer,
                precision,
                text_precision,
                loras,
                lora_rows,
            ],
            outputs=[state, status, run, cancel, timer],
            api_visibility="private",
        )
        loras.change(select_loras, inputs=[loras, lora_rows], outputs=lora_rows, queue=False, api_visibility="private")
        lora_refresh.click(refresh_loras, inputs=loras, outputs=loras, queue=False, api_visibility="private")
        cancel.click(
            lambda current: SERVICE.cancel((current or {}).get("job")),
            inputs=state,
            outputs=status,
            queue=False,
            api_visibility="private",
        )
        timer.tick(
            poll,
            inputs=state,
            outputs=[state, status, image, files, run, cancel, timer],
            queue=False,
            api_visibility="private",
        )
    return [(blocks, "MOSS Image", "moss_image_studio")]


def app_started(_blocks, app):
    from modules_forge.moss_image.api import register_api

    register_api(app)


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(app_started)
