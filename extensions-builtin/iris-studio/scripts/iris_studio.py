"""Iris tasks share controls and retain a separate result for each task."""

from pathlib import Path

import gradio as gr

from modules import script_callbacks
from modules_forge.iris.core import GENERATION_SIZES, PRECISIONS, RUNTIME, TASKS, model_ready, runtime_status
from modules_forge.iris.preparation import PREPARATION
from modules_forge.iris.service import SERVICE

LABELS = {"generate": "生成", "depth": "深度を推定", "upscale": "拡大・復元"}


def ready(task, precision):
    return model_ready(RUNTIME, precision, task)


def button(task, precision):
    return gr.update(
        value=LABELS[task] if ready(task, precision) else "モデルを準備",
        interactive=True,
        variant="primary" if ready(task, precision) else "secondary",
    )


def switch(state, task, precision):
    result = (state or {}).get("results", {}).get(task, {})
    return (
        gr.update(visible=task == "generate"),
        gr.update(visible=task != "generate"),
        gr.update(visible=task == "upscale"),
        gr.update(visible=task == "generate"),
        gr.update(visible=task == "generate"),
        gr.update(visible=task == "depth"),
        gr.update(visible=task == "upscale"),
        button(task, precision),
        runtime_status(),
        result.get("files", []),
        result.get("information", ""),
    )


def start(state, task, precision, prompt, image, seed, size, steps, cfg, negative):
    state = dict(state or {})
    try:
        if state.get("job"):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), "実行中です。", gr.skip(), gr.skip()
        if ready(task, precision):
            width, height = 1024, 1024
            if task == "generate":
                if size not in GENERATION_SIZES:
                    raise ValueError("生成サイズを選び直してください。")
                width, height = GENERATION_SIZES[size]
            identifier = SERVICE.start(
                {
                    "task": task,
                    "precision": precision,
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
            identifier = PREPARATION.start(task, precision)
            kind = "setup"
        state.update(job=identifier, kind=kind, task=task, precision=precision)
        return (
            state,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "準備中…",
            gr.update(interactive=True),
            gr.update(active=True),
        )
    except Exception as exc:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), str(exc), gr.skip(), gr.skip()


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
        return (gr.skip(),) * 12
    state = dict(state)
    result = PREPARATION.poll(state["job"]) if state["kind"] == "setup" else SERVICE.poll(state["job"])
    terminal = result["status"] in {"complete", "error", "cancelled"}
    if not terminal:
        message = result.get("stage", result.get("message", "処理中…"))
        if result.get("current"):
            message += f" · {result['current']} / {result['total']}"
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            message,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
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
        gr.update(interactive=True),
        button(state["task"], state["precision"]),
        message,
        gr.update(interactive=False),
        gr.update(active=False),
        generation,
        depth,
        upscale,
        files,
        information,
    )


def on_ui_tabs():
    with gr.Blocks() as ui:
        state = gr.State({})
        timer = gr.Timer(0.5, active=False)
        with gr.Row():
            task = gr.Radio(
                [(v, k) for k, v in TASKS.items()], value="generate", show_label=False, elem_id="iris-task", scale=3
            )
            precision = gr.Radio([(v, k) for k, v in PRECISIONS.items()], value="int8", label="モデル", scale=1)
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
                    run = gr.Button(
                        "生成" if ready("generate", "int8") else "モデルを準備",
                        variant="primary" if ready("generate", "int8") else "secondary",
                        elem_id="iris-run",
                        scale=4,
                    )
                    stop = gr.Button("中断", interactive=False, scale=1)
                status = gr.Textbox(value=runtime_status(), show_label=False, interactive=False, lines=2)
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
                    refresh = gr.Button("準備状態を再確認")
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
        ]
        task.change(switch, [state, task, precision], change_outputs, queue=False)
        precision.change(switch, [state, task, precision], change_outputs, queue=False)
        refresh.click(switch, [state, task, precision], change_outputs, queue=False)

        def send(state, destination):
            result = (state or {}).get("results", {}).get("generate")
            if not result:
                raise gr.Error("画像を生成してから送ってください。")
            return destination, result["image"]

        to_depth.click(lambda state: send(state, "depth"), [state], [task, image], queue=False).then(
            switch, [state, task, precision], change_outputs, queue=False
        )
        to_upscale.click(lambda state: send(state, "upscale"), [state], [task, image], queue=False).then(
            switch, [state, task, precision], change_outputs, queue=False
        )
        run.click(
            start,
            [state, task, precision, prompt, image, seed, size, steps, cfg, negative],
            [state, task, precision, run, status, stop, timer],
            queue=False,
        )
        stop.click(cancel, [state], [status, stop], queue=False)
        timer.tick(
            poll,
            [state],
            [state, task, precision, run, status, stop, timer, generation, depth, upscale, files, information],
            queue=False,
        )
    return [(ui, "Iris", "iris_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
