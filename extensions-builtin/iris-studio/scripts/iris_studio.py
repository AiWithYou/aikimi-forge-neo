"""Iris tasks share controls and retain a separate result for each task."""

from pathlib import Path

import gradio as gr

from modules import script_callbacks
from modules_forge import local_assets, model_catalog
from modules_forge.iris.core import (
    GENERATION_SIZES,
    PRECISIONS,
    RUNTIME,
    TASKS,
    checkpoint_choices,
    model_ready,
    runtime_status,
)
from modules_forge.iris.local_source import resolve
from modules_forge.iris.preparation import PREPARATION
from modules_forge.iris.service import SERVICE

LABELS = {"generate": "生成", "depth": "深度を推定", "upscale": "拡大・復元"}


def refresh_checkpoints(current):
    try:
        choices = checkpoint_choices(RUNTIME, current)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return gr.update(choices=choices, value=current or "base")


def ready(task, precision, checkpoint="base"):
    return model_ready(RUNTIME, precision, task, checkpoint if task == "generate" else "base")


def button(task, precision, checkpoint="base"):
    checkpoint = checkpoint if task == "generate" else "base"
    available = ready(task, precision, checkpoint)
    valid = checkpoint == "base"
    if not available and not valid:
        try:
            resolve(RUNTIME, checkpoint)
            valid = True
        except ValueError:
            pass
    return gr.update(
        value=LABELS[task]
        if available
        else "モデルを準備"
        if checkpoint == "base"
        else "実行環境・共通部品を準備"
        if valid
        else "選択を確認",
        interactive=available or valid,
        variant="primary" if available else "secondary",
    )


def switch(state, task, precision, checkpoint="base", *, persist=True, saved=None):
    state = dict(state or {})
    if saved is None:
        saved = local_assets.selection("iris")
    if task == "generate" and state.get("selection_task", "generate") != "generate":
        checkpoint = saved.get("checkpoint", "base")
    base_precisions = dict(state.get("base_precisions", saved.get("base_precisions", {})))
    result = (state or {}).get("results", {}).get(task, {})
    selected = checkpoint if task == "generate" else "base"
    if selected == "base":
        if state.get("selection_task", task) == task and state.get("selection_checkpoint", "base") == "base":
            base_precisions[task] = precision
        else:
            precision = base_precisions.get(task, "int8")
    error = ""
    if selected != "base":
        try:
            source = resolve(RUNTIME, selected)
            precision = source["precision"]
            checkpoint = selected = str(source["weights"])
        except ValueError as exc:
            error = str(exc)
    state.update(selection_task=task, selection_checkpoint=selected, base_precisions=base_precisions)
    if persist:
        local_assets.save_selection(
            "iris",
            {
                **{key: value for key, value in saved.items() if key != "directory"},
                "base_precisions": base_precisions,
                **({"checkpoint": checkpoint, "precision": precision} if task == "generate" else {}),
            },
        )
    status = runtime_status()
    if selected != "base":
        path = Path(selected)
        status = (
            error
            or f"{path.name} · {PRECISIONS[precision]} · {'準備完了' if ready(task, precision, selected) else '実行環境・共有エンコーダーを確認してください。'}"
        )
    run_button = button(task, precision, selected)
    if state.get("job"):
        run_button["interactive"] = False
    if task == "generate":
        try:
            choices = checkpoint_choices(RUNTIME, selected)
        except ValueError:
            choices = [
                ("オリジナル", "base"),
                *model_catalog.choice_labels([], current=selected if selected != "base" else ""),
            ]
    else:
        choices = [(f"公式 · {TASKS[task]}", "base")]
    return (
        gr.update(visible=task == "generate"),
        gr.update(visible=task != "generate"),
        gr.update(visible=task == "upscale"),
        gr.update(visible=task == "generate"),
        gr.update(visible=task == "generate"),
        gr.update(visible=task == "depth"),
        gr.update(visible=task == "upscale"),
        run_button,
        status,
        result.get("files", []),
        result.get("information", ""),
        gr.update(visible=True),
        gr.update(value=precision, interactive=selected == "base" and not (state or {}).get("job")),
        gr.update(
            value=selected,
            interactive=task == "generate" and not state.get("job"),
            allow_custom_value=task == "generate",
            choices=choices,
        ),
        state,
    )


def start(state, task, precision, prompt, image, seed, size, steps, cfg, negative, checkpoint="base"):
    state = dict(state or {})
    checkpoint = checkpoint if task == "generate" else "base"
    try:
        if state.get("job"):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), "実行中です。", gr.skip(), gr.skip(), gr.skip()
        if checkpoint != "base":
            try:
                source = resolve(RUNTIME, checkpoint)
            except ValueError as exc:
                raise ValueError(f"チェックポイント: {exc}") from exc
            checkpoint, precision = str(source["weights"]), source["precision"]
        if ready(task, precision, checkpoint):
            width, height = 1024, 1024
            if task == "generate":
                if size not in GENERATION_SIZES:
                    raise ValueError("生成サイズを選び直してください。")
                width, height = GENERATION_SIZES[size]
            identifier = SERVICE.start(
                {
                    "task": task,
                    "precision": precision,
                    "checkpoint": checkpoint,
                    "prompt": prompt,
                    "image": image,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "negative_prompt": negative,
                }
            )
            kind = "run"
        else:
            identifier = (
                PREPARATION.start(task, precision)
                if checkpoint == "base"
                else PREPARATION.start(task, precision, components_only=True)
            )
            kind = "setup"
        state.update(job=identifier, kind=kind, task=task, precision=precision, checkpoint=checkpoint)
        return (
            state,
            gr.update(interactive=False),
            gr.update(value=precision, interactive=False),
            gr.update(interactive=False),
            "準備中…",
            gr.update(interactive=True),
            gr.update(active=True),
            gr.update(value=checkpoint, interactive=False),
        )
    except Exception as exc:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), str(exc), gr.skip(), gr.skip(), gr.skip()


def cancel(state):
    if state and state.get("job"):
        if state["kind"] == "setup":
            PREPARATION.cancel(state["job"])
        else:
            SERVICE.cancel(state["job"])
        return "中断処理中…", gr.update(interactive=False)
    return gr.skip(), gr.skip()


def poll(state):
    if not state or not state.get("job"):
        return (gr.skip(),) * 13
    state = dict(state)
    result = PREPARATION.poll(state["job"]) if state["kind"] == "setup" else SERVICE.poll(state["job"])
    terminal = result["status"] in {"complete", "error", "cancelled"}
    if not terminal:
        message = result.get("stage", result.get("message", "処理中…"))
        if result.get("current"):
            message += f" · {result['current']} / {result['total']}"
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), message, *(gr.skip(),) * 8)
    generation = depth = upscale = files = information = gr.skip()
    if state["kind"] == "run" and result["status"] == "complete":
        image = result["image"]
        task = state["task"]
        if task == "generate":
            generation = image
        else:
            original = str(Path(result["directory"]) / "input.png")
            if task == "depth":
                depth = (original, image)
            else:
                upscale = (original, image)
        files = result["files"]
        information = f"{TASKS[task]} · {PRECISIONS[state['precision']]} · {result['size'][0]}×{result['size'][1]} · {result['seconds']:.1f}秒"
        if state.get("checkpoint", "base") != "base":
            label = dict((value, label) for label, value in checkpoint_choices()).get(
                state["checkpoint"], state["checkpoint"]
            )
            information = f"{label} · {information}"
        if result.get("seed") is not None:
            information += f" · Seed {result['seed']}"
        state.setdefault("results", {})[task] = {"image": image, "files": files, "information": information}
        message = "完了"
    elif result["status"] == "complete":
        message = "準備完了。入力を確認して実行してください。"
    elif result["status"] == "cancelled":
        message = "中断しました。"
    else:
        message = result.get("error", result.get("message", "処理に失敗しました。"))
    state.pop("job")
    return (
        state,
        gr.update(interactive=True),
        gr.update(interactive=state.get("checkpoint", "base") == "base"),
        button(state["task"], state["precision"], state.get("checkpoint", "base")),
        message,
        gr.update(interactive=False),
        gr.update(active=False),
        generation,
        depth,
        upscale,
        files,
        information,
        gr.update(interactive=state["task"] == "generate"),
    )


def load_selection():
    saved = local_assets.selection("iris")
    selected = saved.get("checkpoint", "base")
    initial_precision = saved.get("precision", "int8")
    if selected == "base":
        initial_precision = saved.get("base_precisions", {}).get("generate", "int8")
    if initial_precision not in PRECISIONS:
        initial_precision = "int8"
    state = {
        "selection_task": "generate",
        "selection_checkpoint": selected,
        "base_precisions": saved.get("base_precisions", {}),
    }
    view = switch(state, "generate", initial_precision, selected, persist=False, saved=saved)
    selected = view[13]["value"]
    try:
        choices = checkpoint_choices(RUNTIME, selected)
    except ValueError:
        choices = [
            ("オリジナル", "base"),
            *model_catalog.choice_labels([], current=selected if selected != "base" else ""),
        ]
    view[13]["choices"] = choices
    return (*view, gr.update(value="generate"))


def on_ui_tabs():
    initial_view = load_selection()
    initial_precision = initial_view[12]["value"]
    selected = initial_view[13]["value"]
    choices = initial_view[13]["choices"]
    with gr.Blocks() as ui:
        state = gr.State(initial_view[14])
        timer = gr.Timer(0.5, active=False)
        with gr.Row():
            task = gr.Radio(
                [(v, k) for k, v in TASKS.items()], value="generate", show_label=False, elem_id="iris-task", scale=3
            )
            precision = gr.Radio(
                [(v, k) for k, v in PRECISIONS.items()],
                value=initial_precision,
                label="保存形式",
                interactive=selected == "base",
                scale=1,
            )
        with gr.Group() as checkpoint_group:
            with gr.Row():
                checkpoint = gr.Dropdown(
                    choices,
                    value=selected,
                    label="チェックポイント",
                    allow_custom_value=True,
                    elem_id="iris-checkpoint",
                    scale=4,
                )
                refresh = gr.Button("一覧を更新", scale=1)
        for component in (checkpoint, precision):
            model_catalog.configure_component(component)
        with gr.Row():
            with gr.Column(scale=5, min_width=320):
                with gr.Group() as text_input:
                    prompt = gr.Textbox(
                        label="プロンプト（英語）", lines=5, placeholder="a red fox sleeping in fresh snow, golden hour"
                    )
                    seed = gr.Number(label="Seed（-1でランダム）", value=-1, precision=0)
                with gr.Group(visible=False) as image_input:
                    image = gr.Image(label="入力画像", type="pil", height=320)
                    upscale_limit = gr.Markdown("入力を短辺512・長辺1024以内に収めてから4倍にします。", visible=False)
                with gr.Row():
                    initial_button = initial_view[7]
                    run = gr.Button(
                        initial_button["value"],
                        variant=initial_button["variant"],
                        interactive=initial_button["interactive"],
                        elem_id="iris-run",
                        scale=4,
                    )
                    stop = gr.Button("中断", interactive=False, scale=1)
                status = gr.Textbox(value=initial_view[8], show_label=False, interactive=False, lines=2)
                with gr.Accordion("詳細設定", open=False) as detail:
                    size = gr.Dropdown(list(GENERATION_SIZES), value="1024×1024", label="生成サイズ")
                    with gr.Row():
                        steps = gr.Slider(1, 200, value=100, step=1, label="ステップ数")
                        cfg = gr.Slider(1, 15, value=3, step=0.1, label="CFG")
                    negative = gr.Textbox(label="ネガティブプロンプト（英語）", lines=2)
                with gr.Accordion("モデルの準備", open=False):
                    gr.Markdown(
                        "選択した処理・モデルに必要なファイルを取得します。"
                        "量子化モデルはHugging Faceの[INT8](https://huggingface.co/Aikimi/iris-3b-int8)・"
                        "[W4A8](https://huggingface.co/Aikimi/iris-3b-w4a8)から取得します。"
                    )
            with gr.Column(scale=7, min_width=360):
                with gr.Group() as generation_group:
                    generation = gr.Image(label="生成結果", height=560, interactive=False)
                    with gr.Row():
                        to_depth = gr.Button("深度推定へ送る")
                        to_upscale = gr.Button("復元・4倍拡大へ送る")
                with gr.Group(visible=False) as depth_group:
                    depth = gr.ImageSlider(label="元画像 ↔ 相対深度", height=560)
                with gr.Group(visible=False) as upscale_group:
                    upscale = gr.ImageSlider(label="元画像 ↔ 復元・4倍拡大", height=560)
                information = gr.Markdown()
                files = gr.File(label="画像・数値データ・実行条件", file_count="multiple", interactive=False)
        change_outputs = [
            text_input,
            image_input,
            upscale_limit,
            detail,
            generation_group,
            depth_group,
            upscale_group,
            run,
            status,
            files,
            information,
            checkpoint_group,
            precision,
            checkpoint,
            state,
        ]
        ui.load(load_selection, [], [*change_outputs, task], queue=False, show_progress="hidden")
        task.change(switch, [state, task, precision, checkpoint], change_outputs, queue=False)
        precision.input(switch, [state, task, precision, checkpoint], change_outputs, queue=False)
        checkpoint.input(switch, [state, task, precision, checkpoint], change_outputs, queue=False)

        refresh.click(refresh_checkpoints, [checkpoint], [checkpoint], queue=False).then(
            switch, [state, task, precision, checkpoint], change_outputs, queue=False
        )

        def send(state, destination):
            result = (state or {}).get("results", {}).get("generate")
            if not result:
                raise gr.Error("画像を生成してから送ってください。")
            return destination, result["image"]

        to_depth.click(lambda state: send(state, "depth"), [state], [task, image], queue=False).then(
            switch, [state, task, precision, checkpoint], change_outputs, queue=False
        )
        to_upscale.click(lambda state: send(state, "upscale"), [state], [task, image], queue=False).then(
            switch, [state, task, precision, checkpoint], change_outputs, queue=False
        )
        run.click(
            start,
            [state, task, precision, prompt, image, seed, size, steps, cfg, negative, checkpoint],
            [state, task, precision, run, status, stop, timer, checkpoint],
            queue=False,
        )
        stop.click(cancel, [state], [status, stop], queue=False)
        timer.tick(
            poll,
            [state],
            [
                state,
                task,
                precision,
                run,
                status,
                stop,
                timer,
                generation,
                depth,
                upscale,
                files,
                information,
                checkpoint,
            ],
            queue=False,
        )
    return [(ui, "Iris", "iris_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
