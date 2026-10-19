"""MOSS generation, reference editing and distillation in the shared workspace."""

import gradio as gr

from modules import script_callbacks
from modules_forge import local_assets
from modules_forge.moss_image import checkpoints, core
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
    checkpoint="official",
    mode="txt2img",
    images=None,
    edit_method="redraw",
    denoise=0.65,
    edit_lora="",
    edit_strength=1.0,
    kv_cache=True,
    hires_enabled=False,
    hires_scale=1.5,
    hires_steps=20,
    hires_denoise=0.35,
    rope_x=1.0,
    rope_y=1.0,
):
    state = dict(state or {})
    if state.get("job"):
        raise gr.Error("生成中です。完了または中断を待ってください。")
    try:
        request = dict(
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
            rope={"scale_x": rope_x, "scale_y": rope_y},
        )
        if checkpoint != "official":
            request["checkpoint"] = checkpoint
        if mode == "txt2img" and hires_enabled:
            request["width"], request["height"] = core.hires_output_size(width, height, hires_scale)
            request["hires"] = {
                "base_width": width,
                "base_height": height,
                "steps": hires_steps,
                "denoise": hires_denoise,
            }
        if mode == "img2img":
            request.update(
                mode=mode,
                images=[item[0] if isinstance(item, (list, tuple)) else item for item in images or []],
                edit_method=edit_method,
                denoise=denoise,
                kv_cache=kv_cache,
            )
            if edit_method == "reference":
                if not edit_lora or edit_strength <= 0:
                    raise ValueError("MOSSの編集用LoRAを選択し、強度を0より大きくしてください。")
                if any(item["name"] == edit_lora for item in request["loras"]):
                    raise ValueError("編集用LoRAを通常のLoRA欄と重複して選択しないでください。")
                request["loras"].append({"name": edit_lora, "strength": edit_strength})
        state["job"] = SERVICE.start(request)
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


def refresh_models(name):
    choices = checkpoints.inventory(MODEL_ROOT)
    if name not in {value for _, value in choices}:
        raise gr.Error("選択したモデルが見つからないか、形式を確認できません。モデルを選び直してください。")
    return gr.update(choices=choices, value=name)


def select_model(name):
    if name == "official":
        return gr.update(interactive=True)
    try:
        path = checkpoints.resolve(MODEL_ROOT, name)
        precision = checkpoints.header_precision(checkpoints.local_assets.read_header(path))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return gr.update(value=precision, interactive=False)


def edit_controls(method):
    multiple = method == "reference"
    return (
        gr.update(visible=not multiple),
        gr.update(visible=multiple),
        gr.update(label="参照画像（最大3枚・入力順にPicture 1〜3）" if multiple else "参照画像（1枚）"),
    )


def reference_size(images):
    if not images:
        raise gr.Error("参照画像を入力してください。")
    image = images[0][0] if isinstance(images[0], (tuple, list)) else images[0]
    from PIL import ImageOps

    width, height = ImageOps.exif_transpose(image).size
    scale = min(4096 / max(width, height), max(1, 512 / min(width, height)))
    return max(512, int(width * scale) // 16 * 16), max(512, int(height * scale) // 16 * 16)


def hires_preview(enabled, width, height, scale):
    if not enabled:
        return ""
    try:
        width, height = core.hires_output_size(width, height, scale)
    except ValueError as error:
        return str(error)
    return f"最終サイズ: **{width}×{height}**"


def hires_controls(enabled, width, height, scale):
    visibility = gr.update(visible=True if enabled else "hidden")
    if not enabled:
        return visibility, ""
    try:
        dimensions = []
        for value in (width, height):
            if value is None:
                dimensions.append(None)
            else:
                gr.Number.raise_if_out_of_bounds(value, 512, 4096)
                dimensions.append(gr.Number.round_to_precision(value, 0))
        return visibility, hires_preview(True, *dimensions, scale)
    except (gr.Error, TypeError, ValueError, OverflowError) as error:
        return visibility, str(error)


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
        "encoding_references": "参照画像を処理中…",
        "sampling": "画像を生成中…",
        "hires_upscaling": "画像を拡大中…",
        "hires_encoding": "拡大した画像を処理中…",
        "hires_sampling": "拡大した画像を仕上げ中…",
        "decoding": "画像を仕上げ中…",
        "saving": "保存中…",
        "complete": "保存中…",
    }.get(stage, "生成中…")
    current, total = result.get("current"), result.get("total")
    if stage in {"sampling", "hires_sampling"} and type(current) is int and type(total) is int and 0 <= current <= total:
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
        with gr.Row() as model_controls:
            model = gr.Dropdown(
                checkpoints.inventory(MODEL_ROOT),
                value="official",
                label="モデル",
                interactive=True,
                elem_id="moss-image-model-name",
            )
            model_refresh = gr.Button("モデル一覧を更新", size="sm", scale=0)
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
        with gr.Column() as reference_controls:
            images = gr.Gallery(
                label="参照画像（1枚）",
                type="pil",
                columns=3,
                height=220,
                object_fit="contain",
                interactive=True,
                elem_id="moss-image-references-gallery",
            )
            match_size = gr.Button("画像のサイズを使う", size="sm")
            method = gr.Dropdown(
                [("元画像から再描画", "redraw"), ("編集LoRAで複数参照", "reference")],
                value="redraw",
                label="編集方法",
                elem_id="moss-image-edit-method",
            )
            denoise = gr.Slider(0.01, 1, value=0.65, step=0.01, label="変更量", elem_id="moss-image-denoise")
            with gr.Column(visible=False, elem_id="moss-image-edit-settings") as edit_settings:
                edit_lora = gr.Dropdown(
                    [("選択してください", ""), *local_lora.inventory(MODEL_ROOT)],
                    value="",
                    label="編集用LoRA",
                    elem_id="moss-image-edit-lora",
                )
                edit_refresh = gr.Button("編集用LoRA一覧を更新", size="sm")
                edit_strength = gr.Slider(0, 2, value=1, step=0.05, label="編集LoRAの強度")
                kv_cache = gr.Checkbox(value=True, label="参照キャッシュ", elem_id="moss-image-kv-cache")
            for component in (images, method, edit_lora):
                component.do_not_save_to_config = True
        run = gr.Button("生成", variant="primary")
        cancel = gr.Button("中断", interactive=False)
        with gr.Column() as settings:
            with gr.Row():
                width = gr.Number(value=1024, precision=0, minimum=512, maximum=4096, step=16, label="幅")
                height = gr.Number(value=1024, precision=0, minimum=512, maximum=4096, step=16, label="高さ")
            with gr.Group(elem_id="moss-image-hires-controls"):
                hires_enabled = gr.Checkbox(
                    value=False, label="拡大して仕上げる（Hires.fix）", elem_id="moss-image-hires-enabled"
                )
                with gr.Group(visible="hidden", elem_id="moss-image-hires-settings") as hires_settings:
                    hires_scale = gr.Slider(
                        1.05, 4, value=1.5, step=0.05, label="倍率", elem_id="moss-image-hires-scale"
                    )
                    hires_steps = gr.Slider(
                        1, 200, value=20, step=1, label="仕上げSteps", elem_id="moss-image-hires-steps"
                    )
                    hires_denoise = gr.Slider(
                        0.01, 1, value=0.35, step=0.01, label="仕上げの変更量", elem_id="moss-image-hires-denoise"
                    )
                    hires_size = gr.Markdown("", elem_id="moss-image-hires-size")
            steps = gr.Slider(1, 200, value=50, step=1, label="Steps")
            cfg = gr.Slider(0, 100, value=3, step=0.1, label="CFG")
            shift = gr.Slider(0.1, 100, value=5, step=0.1, label="Shift")
            seed = gr.Number(value=-1, precision=0, label="Seed")
            skip_layer = gr.Checkbox(value=True, label="負条件の22層をスキップ", elem_id="moss-image-skip-layer")
            with gr.Accordion("高解像度の構図調整（RoPE）", open=False):
                rope_x = gr.Slider(0.25, 4, value=1, step=0.05, label="横スケール · 1で標準", elem_id="moss-image-rope-x")
                rope_y = gr.Slider(0.25, 4, value=1, step=0.05, label="縦スケール · 1で標準", elem_id="moss-image-rope-y")
        with gr.Column() as output:
            status = gr.Markdown("")
            image = gr.Image(label="生成結果", type="filepath", format="png", interactive=False)
            files = gr.File(label="PNG・生成条件", file_count="multiple", interactive=False)
        workspace = ImageWorkspace(
            blocks,
            "moss-image",
            editable=True,
            models=[model_controls],
            prompts=[prompt, negative],
            actions=[run, cancel],
            settings=[settings],
            references=[reference_controls],
            outputs=[output],
            panels={"LoRA": [lora_controls], "モデル詳細": [precisions]},
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
                model,
                workspace.mode,
                images,
                method,
                denoise,
                edit_lora,
                edit_strength,
                kv_cache,
                hires_enabled,
                hires_scale,
                hires_steps,
                hires_denoise,
                rope_x,
                rope_y,
            ],
            outputs=[state, status, run, cancel, timer],
            api_visibility="private",
        )
        loras.change(select_loras, inputs=[loras, lora_rows], outputs=lora_rows, queue=False, api_visibility="private")
        lora_refresh.click(refresh_loras, inputs=loras, outputs=loras, queue=False, api_visibility="private")
        model.change(select_model, inputs=model, outputs=precision, queue=False, api_visibility="private")
        model_refresh.click(refresh_models, inputs=model, outputs=model, queue=False, api_visibility="private")
        gr.on(
            triggers=[blocks.load, hires_enabled.change, width.change, height.change, hires_scale.change],
            fn=hires_controls,
            inputs=[hires_enabled, width, height, hires_scale],
            outputs=[hires_settings, hires_size],
            trigger_mode="always_last",
            preprocess=False,
            queue=False,
            api_visibility="private",
        )
        method.change(
            edit_controls,
            inputs=method,
            outputs=[denoise, edit_settings, images],
            queue=False,
            api_visibility="private",
        )
        match_size.click(reference_size, inputs=images, outputs=[width, height], queue=False, api_visibility="private")
        edit_refresh.click(
            lambda name: gr.update(
                choices=[("選択してください", ""), *local_lora.inventory(MODEL_ROOT)],
                value=name,
            ),
            inputs=edit_lora,
            outputs=edit_lora,
            queue=False,
            api_visibility="private",
        )
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
