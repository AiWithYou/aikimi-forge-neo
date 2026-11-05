"""FLUX 3 generation, references and layout in the shared image workspace."""

import json

import gradio as gr

from modules import script_callbacks
from modules_forge import local_assets
from modules_forge.flux3_klein import checkpoints as local_checkpoints
from modules_forge.flux3_klein import lora as local_lora
from modules_forge.flux3_klein.core import RUNTIME, model_ready
from modules_forge.flux3_klein.history import History
from modules_forge.flux3_klein.service import SERVICE

HISTORY = History(SERVICE.outputs)

SIZES = {
    "参照画像に合わせる": (None, None),
    "1024×1024": (1024, 1024),
    "1344×768": (1344, 768),
    "768×1344": (768, 1344),
    "1536×1024": (1536, 1024),
    "1024×1536": (1024, 1536),
    "2048×2048": (2048, 2048),
    "3840×2160": (3840, 2160),
    "2160×3840": (2160, 3840),
    "4096×4096": (4096, 4096),
}


def model_defaults(model):
    return (
        (16, gr.update(value=1, interactive=False))
        if model == "distilled"
        else (50, gr.update(value=3.5, interactive=True))
    )


def start(
    state,
    mode,
    model,
    precision,
    prompt,
    images,
    boxes,
    size,
    steps,
    cfg,
    seed,
    pup,
    lora=None,
    lora_strength=1.0,
    checkpoint=None,
    checkpoint_identity=None,
    checkpoint_ready=False,
):
    state = dict(state or {})
    if state.get("job"):
        raise gr.Error("生成中です。完了または中断を待ってください。")
    if checkpoint_ready is not True:
        raise gr.Error("本体の確認が完了していません。確認後に実行してください。")
    try:
        boxes = json.loads(boxes or "[]")
        references = (
            [(item[0] if isinstance(item, (tuple, list)) else item) for item in images or []]
            if mode == "img2img"
            else []
        )
        width, height = SIZES[size]
        request = dict(
            model=model,
            precision=precision,
            prompt=prompt,
            images=references,
            boxes=boxes,
            steps=steps,
            cfg=cfg,
            seed=seed,
            pup=pup,
            lora=lora or None,
            lora_strength=lora_strength,
            checkpoint=checkpoint or None,
        )
        if width is not None:
            request.update(width=width, height=height)
        if checkpoint:
            if not isinstance(checkpoint_identity, dict) or checkpoint_identity.get("name") != checkpoint:
                raise ValueError("本体の選択・確認が完了していません。選び直してください。")
            state["job"] = SERVICE.start(request, checkpoint_identity=checkpoint_identity)
        else:
            if checkpoint_identity is not None:
                raise ValueError("本体の選択・確認が完了していません。選び直してください。")
            state["job"] = SERVICE.start(request)
    except (ValueError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc
    return (
        state,
        "生成を開始しました。",
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(active=True),
    )


def poll(state, checkpoint_ready=True):
    state = dict(state or {})
    identifier = state.get("job")
    if not identifier:
        return state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(active=False)
    result = SERVICE.poll(identifier)
    status = result["status"]
    terminal = status in {"complete", "error", "cancelled"}
    if terminal:
        state.pop("job", None)
    message = result.get("stage", "生成中…")
    image, files, caption = gr.skip(), gr.skip(), gr.skip()
    if status == "complete":
        message = f"完了 · {result['width']}×{result['height']} · seed {result['seed']}"
        image, files, caption = result["image_path"], result["files"], result.get("prompt", "")
    elif status in {"error", "cancelled", "stopping"}:
        message = result.get("error") or ("中断しました。" if status == "cancelled" else "停止を確認中…")
    return state, message, image, files, caption, gr.update(interactive=terminal and checkpoint_ready is True), gr.update(active=not terminal)


def refresh_history(selected):
    choices = HISTORY.list()
    return gr.update(choices=choices, value=selected if selected in {value for _, value in choices} else None)


def select_history(identifier):
    if not identifier:
        return None, [], ""
    try:
        result = HISTORY.load(identifier)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return result["image_path"], result["files"], result.get("prompt", "")


def lora_choices():
    return [("なし", ""), *[(name.removesuffix(".safetensors"), name) for name in local_lora.inventory(RUNTIME)]]


def refresh_loras(selected):
    choices = lora_choices()
    return gr.update(choices=choices, value=selected if selected in {value for _, value in choices} else "")


def select_lora(selected):
    try:
        if selected:
            local_lora.header_info(local_lora.resolve(RUNTIME, selected))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return gr.update(interactive=bool(selected))


def checkpoint_choices(current=""):
    names = local_checkpoints.inventory(RUNTIME)
    choices = [("公式本体", ""), *[(name.removesuffix(".safetensors"), name) for name in names]]
    if current and current not in names:
        choices.append((f"{current}（見つかりません）", current))
    return choices


def refresh_checkpoints(selected):
    return gr.update(choices=checkpoint_choices(selected or ""), value=selected or "")


def select_checkpoint(selected):
    if not selected:
        local_assets.save_selection("flux3_klein", {"checkpoint": ""})
        return None, gr.skip(), gr.skip(), gr.skip(), gr.skip(), "", True
    try:
        identity = local_checkpoints.inspect(RUNTIME, selected)
        local_assets.save_selection("flux3_klein", {"checkpoint": selected, "identity": identity})
        steps, cfg = model_defaults(identity["model"])
        return (
            identity,
            gr.update(value=identity["model"]),
            gr.update(value=identity["precision"]),
            steps,
            cfg,
            "本体を確認しました。",
            True,
        )
    except (OSError, ValueError) as error:
        raise gr.Error(str(error)) from error


def restore_checkpoint(_current):
    saved = local_assets.selection("flux3_klein")
    selected = saved.get("checkpoint") or ""
    if not isinstance(selected, str):
        return (
            gr.update(interactive=True),
            None,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "本体を選び直してください。",
            False,
        )
    choice = gr.update(choices=checkpoint_choices(selected), value=selected, interactive=True)
    if not selected:
        return choice, None, gr.skip(), gr.skip(), gr.skip(), gr.skip(), "", True
    try:
        expected = saved.get("identity")
        if not isinstance(expected, dict) or expected.get("name") != selected:
            raise ValueError("missing identity")
        actual = local_checkpoints.inspect(RUNTIME, selected)
        if actual != expected:
            raise ValueError("changed identity")
    except (OSError, ValueError, TypeError, KeyError):
        return (
            choice,
            None,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            ("選択した本体が見つからないか変更されました。別の本体を選んでから、目的の本体を選び直してください。"),
            False,
        )
    steps, cfg = model_defaults(actual["model"])
    return (
        choice,
        actual,
        gr.update(value=actual["model"]),
        gr.update(value=actual["precision"]),
        steps,
        cfg,
        "本体を確認しました。",
        True,
    )


def generation_ready(state, checkpoint_ready):
    return gr.update(interactive=checkpoint_ready is True and not bool((state or {}).get("job")))


def readiness(selected, fmt, upsample, checkpoint, identity):
    if fmt == "w4a8" and not checkpoint:
        return "W4A8は独自本体を選択してください。"
    request = {"model": selected, "precision": fmt, "pup": upsample, "checkpoint": checkpoint or None}
    if checkpoint:
        if not isinstance(identity, dict) or identity.get("name") != checkpoint:
            return "本体を選び直してください。"
        request["checkpoint_identity"] = identity
    return "準備完了" if model_ready(RUNTIME, request) else "選択した本体と必要な共通部品を確認してください。"


def on_ui_tabs():
    from modules_forge.image_workspace import ImageWorkspace

    initial = local_assets.selection("flux3_klein").get("checkpoint", "")
    if not isinstance(initial, str):
        initial = ""
    with gr.Blocks(analytics_enabled=False) as blocks:
        with gr.Row() as models:
            model = gr.Dropdown(
                [("Klein（蒸留版）", "distilled"), ("Klein Base", "base")],
                value="distilled",
                label="モデル",
                elem_id="flux3-model-choice",
            )
            precision = gr.Dropdown(
                [("INT8 ConvRot", "int8"), ("BF16", "bf16"), ("W4A8（独自本体）", "w4a8"), ("NVFP4（Blackwell）", "nvfp4")],
                value="int8",
                label="保存形式",
            )
        with gr.Row() as checkpoint_controls:
            checkpoint = gr.Dropdown(
                checkpoint_choices(initial),
                value=initial,
                label="本体",
                elem_id="flux3-checkpoint-choice",
                interactive=False,
            )
            checkpoint.do_not_save_to_config = True
            checkpoint_refresh = gr.Button("更新", size="sm", scale=0, min_width=60)
        with gr.Row() as lora_controls:
            lora = gr.Dropdown(lora_choices(), value="", label="LoRA", elem_id="flux3-lora-choice")
            lora.do_not_save_to_config = True
            lora_strength = gr.Slider(0, 2, value=1, step=0.05, label="LoRA強度", interactive=False)
            lora_refresh = gr.Button("更新", size="sm", scale=0, min_width=60)
        prompt = gr.Textbox(label="プロンプト", lines=4, placeholder="生成したい完成画像や、参照画像への変更を記述")
        run = gr.Button("生成", variant="primary", interactive=False)
        cancel = gr.Button("中断", interactive=True)
        with gr.Column() as settings:
            size = gr.Dropdown(list(SIZES), value="1024×1024", label="出力サイズ")
            steps = gr.Slider(1, 200, value=16, step=1, label="Steps")
            cfg = gr.Slider(1, 15, value=1, step=0.1, label="CFG", interactive=False)
            seed = gr.Number(value=-1, precision=0, label="Seed")
            pup = gr.Checkbox(value=False, label="プロンプト補完（PUP）", info="任意。追加モデルの導入が必要です。")
        references = gr.Gallery(
            label="参照画像（入力順・最大10枚）",
            type="pil",
            format="png",
            interactive=True,
            columns=5,
            elem_id="flux3-reference-images",
        )
        with gr.Column() as layout:
            gr.HTML(
                """<div class="flux3-layout">
                  <div class="flux3-layout-toolbar"><span>ドラッグして枠を描き、説明を入力</span>
                    <button type="button" class="flux3-layout-clear">描いた枠をクリア</button></div>
                  <div class="flux3-layout-body">
                    <div class="flux3-layout-stage">
                      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" tabindex="0"
                        aria-label="配置キャンバス。ドラッグして矩形を追加">
                        <title>要素の位置と大きさを指定</title>
                      </svg>
                    </div>
                    <div class="flux3-layout-rows" aria-label="配置した要素"></div>
                  </div>
                  <div class="flux3-layout-note" role="status"></div>
                </div>""",
                elem_id="flux3-layout-editor",
            )
            with gr.Column(elem_classes=["flux3-layout-advanced"]):
                gr.HTML(
                    "<details><summary>高度な編集・座標</summary></details>",
                    elem_id="flux3-layout-disclosure",
                )
                boxes = gr.Textbox(value="[]", label="配置ボックス（JSON）", lines=5, elem_id="flux3-layout-json")
        with gr.Column() as output:
            status = gr.Markdown("")
            image = gr.Image(label="生成結果", type="filepath", interactive=False)
            files = gr.File(label="PNG・生成条件", file_count="multiple", interactive=False)
            caption = gr.Textbox(label="生成に使用した文章", lines=4, interactive=False)
        with gr.Column() as preparation:
            gr.Markdown(
                "導入：`aikimi-flux3-klein-setup.bat` · [使い方](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/flux3-klein.md)"
            )
            check = gr.Button("準備状態を確認", size="sm")
        with gr.Column() as history:
            gr.Markdown("[FLUX 3の使い方](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/flux3-klein.md)")
            with gr.Row():
                history_choice = gr.Dropdown([], label="保存した結果（最新50件）", elem_id="flux3-history-choice")
                history_choice.do_not_save_to_config = True
                history_refresh = gr.Button("更新", size="sm", scale=0)
            history_image = gr.Image(label="履歴の画像", type="filepath", format="png", interactive=False)
            history_files = gr.File(label="履歴のPNG・生成条件", file_count="multiple", interactive=False)
            history_caption = gr.Textbox(label="履歴の生成に使用した文章", lines=4, interactive=False)
        workspace = ImageWorkspace(
            blocks,
            "flux3",
            editable=True,
            models=[models, checkpoint_controls, lora_controls],
            prompts=[prompt],
            actions=[run, cancel],
            settings=[settings],
            references=[references],
            outputs=[output],
            panels={"配置": [layout], "モデル導入": [preparation], "履歴": [history]},
        )
        for button in (workspace.panels["履歴"].button, history_refresh):
            button.click(
                refresh_history, inputs=history_choice, outputs=history_choice, queue=False, api_visibility="private"
            )
        history_choice.change(
            select_history,
            inputs=history_choice,
            outputs=[history_image, history_files, history_caption],
            queue=False,
            api_visibility="private",
        )
        state = gr.State({})
        checkpoint_identity = gr.State(None)
        checkpoint_ready = gr.State(False)
        timer = gr.Timer(0.5, active=False)
        model.change(model_defaults, inputs=model, outputs=[steps, cfg], queue=False, api_visibility="private")
        checkpoint_refresh.click(
            refresh_checkpoints,
            inputs=checkpoint,
            outputs=checkpoint,
            concurrency_id="flux3-checkpoint-selection",
            api_visibility="private",
        )
        checkpoint.input(
            select_checkpoint,
            inputs=checkpoint,
            outputs=[checkpoint_identity, model, precision, steps, cfg, status, checkpoint_ready],
            concurrency_id="flux3-checkpoint-selection",
            api_visibility="private",
        )
        blocks.load(
            fn=restore_checkpoint,
            inputs=checkpoint,
            outputs=[checkpoint, checkpoint_identity, model, precision, steps, cfg, status, checkpoint_ready],
            concurrency_id="flux3-checkpoint-selection",
            api_visibility="private",
        )
        lora_refresh.click(refresh_loras, inputs=lora, outputs=lora, queue=False, api_visibility="private")
        lora.change(select_lora, inputs=lora, outputs=lora_strength, queue=False, api_visibility="private")
        run.click(
            start,
            inputs=[
                state,
                workspace.mode,
                model,
                precision,
                prompt,
                references,
                boxes,
                size,
                steps,
                cfg,
                seed,
                pup,
                lora,
                lora_strength,
                checkpoint,
                checkpoint_identity,
                checkpoint_ready,
            ],
            outputs=[state, status, run, cancel, timer],
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
            inputs=[state, checkpoint_ready],
            outputs=[state, status, image, files, caption, run, timer],
            queue=False,
            api_visibility="private",
        )
        checkpoint_ready.change(
            generation_ready, inputs=[state, checkpoint_ready], outputs=run, queue=False, api_visibility="private"
        )
        check.click(
            readiness,
            inputs=[model, precision, pup, checkpoint, checkpoint_identity],
            outputs=status,
            queue=False,
            api_visibility="private",
        )
    return [(blocks, "FLUX 3", "flux3_klein_studio")]


def app_started(_blocks, app):
    from modules_forge.flux3_klein.api import register_api

    register_api(app)


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(app_started)
