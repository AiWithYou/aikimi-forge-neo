"""Iris tasks share controls and retain a separate result for each task."""

from pathlib import Path

import gradio as gr

from modules import script_callbacks
from modules_forge import local_assets, model_catalog
from modules_forge.iris import lora as iris_lora
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
from modules_forge.iris.official_turbo import checkpoint_settings, combine_settings, generation_sizes
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


def update_sizes(checkpoint, current="1024×1024", names=None, rows=None):
    try:
        sampling = (
            lora_sampling(list(local_assets.lora_settings(names, rows)), checkpoint) if names is not None else None
        )
        sizes = generation_sizes(RUNTIME, checkpoint, sampling=sampling)
    except ValueError:
        return gr.skip()
    return gr.update(choices=list(sizes), value=current if current in sizes else "1024×1024")


def lora_sampling(loras, checkpoint="base"):
    active = [{**item, "metadata": iris_lora.inspect(RUNTIME, item["name"])} for item in loras if item["strength"] != 0]
    sampling = combine_settings(checkpoint_settings(RUNTIME, checkpoint), iris_lora.sampling_settings(active))
    if sampling["mode"] == "turbo":
        return {**sampling, "turbo_name": next(item["name"] for item in active if item["metadata"].get("turbo_steps"))}
    return sampling


def lora_controls(state, task, names, rows, standard, steps, cfg, negative, *, persist=True):
    names = list(names or [])
    rows = local_assets.lora_rows(names, rows)
    standard = dict(standard or {})
    if task == "generate" and standard.get("mode", "standard") == "standard":
        standard.update(steps=steps, cfg=cfg, negative=negative)
    sampling = None
    message = ""
    try:
        settings = list(local_assets.lora_settings(names, rows))
        if persist and local_assets.selection("iris_lora") != {"loras": settings}:
            local_assets.save_selection("iris_lora", {"loras": settings})
        if task == "generate":
            sampling = lora_sampling(settings, (state or {}).get("selection_checkpoint", "base"))
    except ValueError as exc:
        message = str(exc)
    update_sampling = False
    if sampling is not None:
        if sampling["mode"] == "official_turbo":
            update_sampling = True
            steps, cfg, negative = len(sampling["sigmas"]), 1, ""
            standard.pop("turbo_name", None)
            message = f"公式Turbo · {steps}ステップ・CFG 1"
        elif sampling["mode"] == "turbo":
            update_sampling = standard.get("turbo_name") != sampling["turbo_name"]
            if update_sampling:
                steps, cfg, negative = sampling["default_steps"], 1, ""
            standard["turbo_name"] = sampling["turbo_name"]
            message = f"Turbo · 推奨 {sampling['default_steps']}ステップ・CFG 1"
        else:
            if standard.get("mode") in {"turbo", "official_turbo"}:
                update_sampling = True
                steps, cfg, negative = standard["steps"], standard["cfg"], standard["negative"]
            standard.pop("turbo_name", None)
        standard["mode"] = sampling["mode"]
    editable = task == "generate" and not (state or {}).get("job")
    sampling_editable = task == "generate" and (sampling or {}).get("mode") != "official_turbo"
    choices = list(dict.fromkeys([*iris_lora.inventory(RUNTIME), *names]))
    return (
        standard,
        gr.update(choices=choices, value=names, interactive=editable),
        gr.update(
            value=rows, row_count=(len(rows), "fixed"), visible=True if names else "hidden", interactive=editable
        ),
        gr.update(interactive=editable),
        message,
        gr.update(**({"value": steps} if update_sampling else {}), interactive=sampling_editable),
        gr.update(**({"value": cfg} if update_sampling else {}), interactive=sampling_editable),
        gr.update(**({"value": negative} if update_sampling else {}), interactive=sampling_editable),
        gr.update(visible=task == "generate"),
    )


def sync_loras(state, task, names, rows, standard, steps, cfg, negative):
    if (standard or {}).get("initialized") is False:
        return tuple(gr.skip() for _ in range(9))
    updates = list(lora_controls(state, task, names, rows, standard, steps, cfg, negative, persist=False))
    updates[1].pop("value")
    return tuple(updates)


def edit_loras(state, task, names, rows, standard, steps, cfg, negative):
    if (standard or {}).get("initialized") is False:
        return tuple(gr.skip() for _ in range(9))
    updates = list(lora_controls(state, task, names, rows, standard, steps, cfg, negative))
    updates[1] = gr.skip()
    return tuple(updates)


def load_loras(state=None, task="generate"):
    settings = local_assets.selection("iris_lora").get("loras", [])
    settings = [item for item in settings if isinstance(item, dict) and isinstance(item.get("name"), str)]
    names = [item["name"] for item in settings]
    rows = [[item["name"], item.get("strength", 1)] for item in settings]
    updates = list(
        lora_controls(state, task, names, rows, {"initialized": True, "mode": "standard"}, 100, 3, "", persist=False)
    )
    for index, value in zip((5, 6, 7), (100, 3, ""), strict=True):
        updates[index].setdefault("value", value)
    return tuple(updates)


def start(
    state,
    task,
    precision,
    prompt,
    image,
    seed,
    size,
    steps,
    cfg,
    negative,
    checkpoint="base",
    lora_names=None,
    lora_rows=None,
    image_mode="txt2img",
    edit_image=None,
    strength=0.5,
):
    state = dict(state or {})
    checkpoint = checkpoint if task == "generate" else "base"
    try:
        if state.get("job"):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), "実行中です。", gr.skip(), gr.skip(), gr.skip()
        loras = []
        if task == "generate":
            loras = list(local_assets.lora_settings(lora_names, lora_rows))
        if checkpoint != "base":
            try:
                source = resolve(RUNTIME, checkpoint)
            except ValueError as exc:
                raise ValueError(f"チェックポイント: {exc}") from exc
            checkpoint, precision = str(source["weights"]), source["precision"]
        if task == "generate":
            sampling = lora_sampling(loras, checkpoint)
        if ready(task, precision, checkpoint):
            width, height = 1024, 1024
            if task == "generate":
                sizes = generation_sizes(RUNTIME, checkpoint, sampling=sampling)
                if size not in sizes:
                    raise ValueError("生成サイズを選び直してください。")
                width, height = sizes[size]
            identifier = SERVICE.start(
                {
                    "task": task,
                    "precision": precision,
                    "checkpoint": checkpoint,
                    "prompt": prompt,
                    "image": edit_image if task == "generate" and image_mode == "img2img" else image,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "negative_prompt": negative,
                    **({"loras": loras} if task == "generate" else {}),
                    **({"image_mode": image_mode, "strength": strength} if task == "generate" else {}),
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
        state.update(
            job=identifier,
            kind=kind,
            task=task,
            precision=precision,
            checkpoint=checkpoint,
            image_mode=image_mode if task == "generate" else None,
        )
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


def start_with_loras(*args):
    result = start(*args)
    current = result[0] if isinstance(result[0], dict) and "__type__" not in result[0] else args[0]
    editable = args[1] == "generate" and not (current or {}).get("job")
    return (*result, *(gr.update(interactive=editable) for _ in range(3)), *send_controls(current))


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
        if result.get("image_mode") == "img2img":
            information += f" · img2img · 変更強度 {result['strength']:g} · {result['executed_steps']} steps"
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


def poll_with_loras(state):
    result = poll(state)
    current = result[0]
    if not isinstance(current, dict) or "__type__" in current or current.get("job"):
        return (*result, *(gr.skip(),) * 6)
    return (*result, *(gr.update(interactive=current["task"] == "generate") for _ in range(3)), *send_controls(current))


def send_controls(state):
    state = state or {}
    available = not state.get("job") and bool(state.get("results", {}).get("generate", {}).get("image"))
    return tuple(gr.update(interactive=available) for _ in range(3))


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
    initial_loras = load_loras(initial_view[14])
    initial_precision = initial_view[12]["value"]
    selected = initial_view[13]["value"]
    choices = initial_view[13]["choices"]
    with gr.Blocks() as ui:
        state = gr.State(initial_view[14])
        standard_sampling = gr.State({**initial_loras[0], "initialized": False})
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
                with gr.Group() as lora_group:
                    with gr.Row():
                        loras = gr.Dropdown(
                            initial_loras[1]["choices"],
                            value=initial_loras[1]["value"],
                            multiselect=True,
                            allow_custom_value=True,
                            label="LoRA（複数選択・フルパスも入力可）",
                            elem_id="iris-loras",
                            scale=4,
                            min_width=240,
                        )
                        lora_refresh = gr.Button("一覧を更新", scale=1, elem_id="iris-lora-refresh")
                    lora_strengths = gr.Dataframe(
                        value=initial_loras[2]["value"],
                        headers=["LoRA", "強度"],
                        datatype=["str", "number"],
                        type="array",
                        static_columns=[0],
                        column_count=(2, "fixed"),
                        row_count=len(initial_loras[2]["value"]),
                        interactive=True,
                        visible=initial_loras[2]["visible"],
                        label="強度（−2〜2・0で無効）",
                        column_widths=["75%", "25%"],
                        max_chars=64,
                        max_height=240,
                        buttons=[],
                        elem_id="iris-lora-strengths",
                    )
                    lora_information = gr.Markdown(initial_loras[4], elem_id="iris-lora-information")
                with gr.Group(visible=False) as image_input:
                    image = gr.Image(label="入力画像", type="pil", height=320)
                    upscale_limit = gr.Markdown("入力を短辺512・長辺1024以内に収めてから4倍にします。", visible=False)
                with gr.Group() as edit_input:
                    edit_image = gr.Image(label="入力画像", type="pil", height=320, elem_id="iris-img2img-input")
                    strength = gr.Slider(0, 1, value=0.5, step=0.01, label="変更強度", elem_id="iris-img2img-strength")
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
                status = gr.Textbox(
                    value=initial_view[8], show_label=False, interactive=False, lines=2, elem_id="iris-status"
                )
                with gr.Accordion("詳細設定", open=False) as detail:
                    initial_sizes = update_sizes(selected)
                    size = gr.Dropdown(
                        initial_sizes.get("choices", list(GENERATION_SIZES)), value="1024×1024", label="生成サイズ"
                    )
                    with gr.Row():
                        steps = gr.Slider(
                            1,
                            200,
                            value=initial_loras[5]["value"],
                            step=1,
                            label="ステップ数",
                            interactive=initial_loras[5]["interactive"],
                        )
                        cfg = gr.Slider(
                            1,
                            15,
                            value=initial_loras[6]["value"],
                            step=0.1,
                            label="CFG",
                            interactive=initial_loras[6]["interactive"],
                        )
                    negative = gr.Textbox(
                        label="ネガティブプロンプト（CFG > 1で反映）",
                        lines=2,
                        value=initial_loras[7]["value"],
                        interactive=initial_loras[7]["interactive"],
                    )
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
                        to_edit = gr.Button("img2imgへ送る", elem_id="iris-send-img2img", interactive=False)
                        to_depth = gr.Button("深度推定へ送る", interactive=False)
                        to_upscale = gr.Button("復元・4倍拡大へ送る", interactive=False)
                with gr.Group(visible=False) as depth_group:
                    depth = gr.ImageSlider(label="元画像 ↔ 相対深度", height=560)
                with gr.Group(visible=False) as upscale_group:
                    upscale = gr.ImageSlider(label="元画像 ↔ 復元・4倍拡大", height=560)
                information = gr.Markdown()
                files = gr.File(label="画像・数値データ・実行条件", file_count="multiple", interactive=False)
        from modules_forge.image_workspace import ImageWorkspace

        workspace = ImageWorkspace(
            ui,
            "iris-common",
            editable=True,
            models=[checkpoint_group, precision],
            prompts=[text_input, negative],
            actions=[run, stop, status],
            settings=[steps, cfg, size, seed],
            references=[edit_input],
            outputs=[generation_group, depth_group, upscale_group, information, files],
            panels={"LoRA": [lora_group], "画像処理": [task, image_input]},
            badges={"LoRA": loras},
        )
        task.change(
            lambda value: (
                gr.update(visible=value == "generate"),
                gr.update(visible=value == "generate"),
                gr.update(visible=True if value == "generate" else "hidden"),
            ),
            inputs=task,
            outputs=[workspace.prompts, workspace.settings, workspace.tabs],
            queue=False,
        )
        processing_panel = workspace.panels["画像処理"]
        task.change(
            lambda value: processing_panel.reveal() if value != "generate" else (gr.skip(), gr.skip()),
            inputs=task,
            outputs=[processing_panel.state, processing_panel.group],
            queue=False,
        )
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
        lora_inputs = [state, task, loras, lora_strengths, standard_sampling, steps, cfg, negative]
        lora_outputs = [
            standard_sampling,
            loras,
            lora_strengths,
            lora_refresh,
            lora_information,
            steps,
            cfg,
            negative,
            lora_group,
        ]
        for component in (loras, lora_strengths):
            model_catalog.configure_component(component)
        size_inputs = [checkpoint, size, loras, lora_strengths]
        loras.change(edit_loras, lora_inputs, lora_outputs, queue=False, trigger_mode="always_last").then(
            update_sizes, size_inputs, [size], queue=False
        )
        lora_strengths.input(edit_loras, lora_inputs, lora_outputs, queue=False).then(
            update_sizes, size_inputs, [size], queue=False
        )
        lora_refresh.click(sync_loras, lora_inputs, lora_outputs, queue=False).then(
            update_sizes, size_inputs, [size], queue=False
        )
        ui.load(load_selection, [], [*change_outputs, task], queue=False, show_progress="hidden").then(
            load_loras, [state, task], lora_outputs, queue=False, show_progress="hidden"
        ).then(update_sizes, size_inputs, [size], queue=False)
        task.change(switch, [state, task, precision, checkpoint], change_outputs, queue=False).then(
            sync_loras, lora_inputs, lora_outputs, queue=False
        ).then(update_sizes, size_inputs, [size], queue=False)
        precision.input(switch, [state, task, precision, checkpoint], change_outputs, queue=False)
        checkpoint.input(switch, [state, task, precision, checkpoint], change_outputs, queue=False).then(
            sync_loras, lora_inputs, lora_outputs, queue=False
        ).then(update_sizes, size_inputs, [size], queue=False)

        refresh.click(refresh_checkpoints, [checkpoint], [checkpoint], queue=False).then(
            switch, [state, task, precision, checkpoint], change_outputs, queue=False
        ).then(sync_loras, lora_inputs, lora_outputs, queue=False).then(update_sizes, size_inputs, [size], queue=False)

        def send(state, destination):
            if (state or {}).get("job"):
                raise gr.Error("処理が終わってから画像を送ってください。")
            result = (state or {}).get("results", {}).get("generate")
            if not result:
                raise gr.Error("画像を生成してから送ってください。")
            return destination, result["image"]

        to_depth.click(lambda state: send(state, "depth"), [state], [task, image], queue=False).success(
            switch, [state, task, precision, checkpoint], change_outputs, queue=False
        ).success(sync_loras, lora_inputs, lora_outputs, queue=False)
        to_upscale.click(lambda state: send(state, "upscale"), [state], [task, image], queue=False).success(
            switch, [state, task, precision, checkpoint], change_outputs, queue=False
        ).success(sync_loras, lora_inputs, lora_outputs, queue=False)
        to_edit.click(
            lambda state: send(state, "generate"),
            [state],
            [task, edit_image],
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        ).success(switch, [state, task, precision, checkpoint], change_outputs, queue=False).success(
            lambda: workspace.select_mode("img2img"),
            [],
            [workspace.mode, workspace.tabs, workspace.references],
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )
        run.click(
            start_with_loras,
            [
                state,
                task,
                precision,
                prompt,
                image,
                seed,
                size,
                steps,
                cfg,
                negative,
                checkpoint,
                loras,
                lora_strengths,
                workspace.mode,
                edit_image,
                strength,
            ],
            [
                state,
                task,
                precision,
                run,
                status,
                stop,
                timer,
                checkpoint,
                loras,
                lora_strengths,
                lora_refresh,
                to_edit,
                to_depth,
                to_upscale,
            ],
            queue=False,
        )
        stop.click(cancel, [state], [status, stop], queue=False)
        timer.tick(
            poll_with_loras,
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
                loras,
                lora_strengths,
                lora_refresh,
                to_edit,
                to_depth,
                to_upscale,
            ],
            queue=False,
        )
    return [(ui, "Iris", "iris_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
