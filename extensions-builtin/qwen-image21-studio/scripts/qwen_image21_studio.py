"""Qwen Image 2.1 generation and editing without importing its model at startup."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr

from modules import gradio_compat, script_callbacks
from modules.paths import data_path, script_path
from modules_forge.qwen_image21.annotations import annotation_preview, reference_index, resolve_annotation
from modules_forge.qwen_image21.core import (
    MAX_REFERENCE_IMAGES,
    QwenImage21Error,
    Request,
    runtime_status,
)
from modules_forge.qwen_image21.service import JobNotFound, Studio

RUNTIME = Path(script_path) / "models" / "Qwen-Image-2.1"
STUDIO = Studio(RUNTIME, Path(data_path) / "outputs" / "qwen-image-2.1")
PRIVATE = {"api_visibility": "private", "show_progress": "hidden"}
RESOLUTIONS = [
    ("1024 × 1024 · 1:1", "1024x1024"),
    ("1024 × 1280 · 4:5", "1024x1280"),
    ("1280 × 1024 · 5:4", "1280x1024"),
    ("2048 × 2048 · 1:1 · 2K", "2048x2048"),
    ("2400 × 1792 · 4:3 · 2K", "2400x1792"),
    ("1792 × 2400 · 3:4 · 2K", "1792x2400"),
    ("2528 × 1696 · 3:2 · 2K", "2528x1696"),
    ("1696 × 2528 · 2:3 · 2K", "1696x2528"),
    ("2752 × 1536 · 16:9 · 2K", "2752x1536"),
    ("1536 × 2752 · 9:16 · 2K", "1536x2752"),
]


def owner(request: gr.Request) -> str:
    if not request.session_hash:
        raise QwenImage21Error("ブラウザーのQwen Image 2.1タブから操作してください。")
    return f"{request.username or ''}:{request.session_hash}"


def reference_paths(gallery) -> list[str]:
    values = gallery or []
    if len(values) > MAX_REFERENCE_IMAGES:
        raise QwenImage21Error(f"参照画像は最大{MAX_REFERENCE_IMAGES}枚です。不要な画像を削除してください。")
    return [str(value[0] if isinstance(value, (tuple, list)) else value) for value in values]


def reference_gallery(paths):
    return [(path, f"Image {index}") for index, path in enumerate(paths, 1)]


def update_reference_gallery(paths):
    # Gradio 6 hides upload/paste controls for a non-null selected_index even
    # with allow_preview=False. Selection belongs to our separate gr.State.
    return gr.update(value=reference_gallery(paths), selected_index=None)


def reference_controls_visibility(gallery):
    return gr.update(visible=bool(gallery))


def select_reference(event: gr.SelectData):
    return event.index if event.selected and isinstance(event.index, int) else -1


def move_reference(gallery, selected, direction):
    paths = reference_paths(gallery)
    if not isinstance(selected, int) or not 0 <= selected < len(paths):
        raise gr.Error("並べ替える参照画像を選んでください。")
    target = max(0, min(len(paths) - 1, selected + direction))
    paths[selected], paths[target] = paths[target], paths[selected]
    return update_reference_gallery(paths), target


def remove_reference(gallery, selected):
    # Permit deleting excess uploads so an 11th image cannot strand the control.
    paths = [str(value[0] if isinstance(value, (tuple, list)) else value) for value in (gallery or [])]
    if not isinstance(selected, int) or not 0 <= selected < len(paths):
        raise gr.Error("削除する参照画像を選んでください。")
    paths.pop(selected)
    target = min(selected, len(paths) - 1)
    return (
        update_reference_gallery(paths),
        target,
        reference_controls_visibility(paths),
    )


def open_annotation(gallery, selected):
    paths = reference_paths(gallery)
    if len(paths) == 1:
        selected = 0
    if not isinstance(selected, int) or not 0 <= selected < len(paths):
        raise gr.Error("大きく表示する参照画像を選んでください。")
    path = paths[selected]
    background = annotation_preview(path)
    return (
        path,
        gr.update(value=background, label=f"Image {selected + 1} · 変更したい場所を囲む"),
        gr.update(visible=True),
    )


def close_annotation():
    return "", None, gr.update(visible=gradio_compat.keep_hidden_component_mounted(False))


def refresh_references(gallery, annotation_target):
    # Keep removal available even if one upload exceeds the ten-reference limit.
    paths = [str(value[0] if isinstance(value, (tuple, list)) else value) for value in (gallery or [])]
    if annotation_target:
        try:
            index = reference_index(paths, annotation_target)
            return (
                reference_controls_visibility(paths),
                paths[index],
                gr.update(label=f"Image {index + 1} · 変更したい場所を囲む"),
                gr.update(visible=True),
            )
        except QwenImage21Error:
            pass
    elif len(paths) == 1:
        target, editor, panel = open_annotation(paths, 0)
        return reference_controls_visibility(paths), target, editor, panel
    return reference_controls_visibility(paths), *close_annotation()


def continue_edit(identifier, request: gr.Request):
    path = str(STUDIO.artifact(identifier, owner(request)))
    target, editor, panel = open_annotation([path], 0)
    return (
        update_reference_gallery([path]),
        0,
        gr.update(visible=True),
        target,
        editor,
        panel,
        gr.update(value="edit"),
    )


def start(
    prompt,
    gallery,
    resolution,
    transparent,
    precision,
    memory_mode,
    seed,
    steps,
    request: gr.Request,
    annotation_target="",
    annotation_editor=None,
    previous_output=None,
):
    try:
        if resolution not in {value for _, value in RESOLUTIONS}:
            raise QwenImage21Error("出力サイズを選択してください。")
        width, height = (int(value) for value in resolution.split("x"))
        paths = reference_paths(gallery)
        annotation_reference, annotation_layers = resolve_annotation(paths, annotation_target, annotation_editor)
        generation = Request(
            prompt=prompt,
            input_images=tuple(paths),
            width=width,
            height=height,
            transparent=transparent,
            precision=precision,
            memory_mode=memory_mode,
            seed=seed,
            steps=steps,
            annotation_reference=annotation_reference,
            annotation_layers=annotation_layers,
        )
        identifier = STUDIO.start(generation, owner(request))
        return (
            identifier,
            "開始しました。初回のモデル読み込みには時間がかかります。",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(active=True),
            gr.update(label="前の結果（新しい画像を生成中）" if previous_output else "生成結果"),
            gr.update(value=None, visible=gradio_compat.keep_hidden_component_mounted(False)),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    except Exception as exc:
        return gr.update(), str(exc), *[gr.update() for _ in range(7)]


def poll(identifier, request: gr.Request):
    if not identifier:
        return [gr.update()] * 9
    done = False
    try:
        state = STUDIO.status(identifier, owner(request))
        done = state["done"]
        text = f"{state['message']}  ·  経過 {state['elapsed']:.0f} 秒"
        output, files, usable = gr.update(), gr.update(), False
        if done and state["state"] == "complete":
            path = STUDIO.artifact(identifier, owner(request))
            output, usable = gr.update(value=str(path), label="生成結果"), True
            files = gr.update(value=[str(path), str(path.with_name("result.json"))], visible=True)
        elif done:
            output = gr.update(label="生成結果")
        return (
            text,
            gr.update(interactive=done),
            gr.update(interactive=not done),
            gr.update(active=not done),
            output,
            files,
            gr.update(interactive=usable),
            gr.update(interactive=usable),
            gr.update(value="result") if usable else gr.update(),
        )
    except JobNotFound as exc:
        return (
            str(exc),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(active=False),
            gr.update(label="生成結果"),
            gr.update(),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(),
        )
    except Exception as exc:
        if done:
            return (
                str(exc),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(active=False),
                gr.update(label="生成結果"),
                gr.update(),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(),
            )
        return str(exc), *[gr.update() for _ in range(8)]


def cancel(identifier, request: gr.Request):
    if STUDIO.cancel(identifier, owner(request)):
        return "停止を要求しました。workerの終了を確認中です。", gr.update(interactive=False)
    return "この画面で停止できる実行中ジョブはありません。", gr.update(interactive=False)


def use_result(identifier, gallery, request: gr.Request):
    paths = reference_paths(gallery)
    if len(paths) >= MAX_REFERENCE_IMAGES:
        raise gr.Error("参照画像は10枚までです。追加する前に1枚削除してください。")
    paths.append(str(STUDIO.artifact(identifier, owner(request))))
    return (
        update_reference_gallery(paths),
        len(paths) - 1,
        reference_controls_visibility(paths),
    )


def check_runtime():
    return runtime_status(RUNTIME)


def _canvas_updates(editor):
    if editor is None:
        return None, None, ""
    if "value" in editor:
        return editor["value"], None, editor.get("label", "編集する画像")
    return gr.update(), gr.update(), editor.get("label", gr.update())


def open_canvas(gallery, selected):
    target, editor, panel = open_annotation(gallery, selected)
    background, foreground, title = _canvas_updates(editor)
    return target, background, foreground, panel, title


def refresh_canvas(gallery, target):
    controls, target, editor, panel = refresh_references(gallery, target)
    background, foreground, title = _canvas_updates(editor)
    return controls, target, background, foreground, panel, title


def close_canvas():
    target, editor, panel = close_annotation()
    return target, None, None, panel, ""


def continue_canvas(identifier, request: gr.Request):
    gallery, selected, controls, target, editor, panel, view = continue_edit(identifier, request)
    background, foreground, title = _canvas_updates(editor)
    return gallery, selected, controls, target, background, foreground, panel, title, view


def start_canvas(
    prompt,
    gallery,
    resolution,
    transparent,
    precision,
    memory_mode,
    seed,
    steps,
    request: gr.Request,
    annotation_target="",
    background=None,
    foreground=None,
    previous_output=None,
):
    # Keep the bridge files alive until Studio.start snapshots the request.
    # The original reference is still read from Gallery, never from this preview.
    try:
        with TemporaryDirectory(prefix="qwen-canvas-") as temporary:
            editor = None
            if annotation_target and foreground is not None and foreground.getchannel("A").getbbox() is not None:
                if background is None:
                    raise QwenImage21Error("編集元を読み込み直してください。")
                directory = Path(temporary)
                background.save(directory / "background.png", format="PNG")
                foreground.save(directory / "marks.png", format="PNG")
                editor = {"background": str(directory / "background.png"), "layers": [str(directory / "marks.png")]}
            return start(
                prompt,
                gallery,
                resolution,
                transparent,
                precision,
                memory_mode,
                seed,
                steps,
                request,
                annotation_target,
                editor,
                previous_output,
            )
    except Exception as exc:
        return gr.update(), str(exc), *[gr.update() for _ in range(7)]


def switch_workspace(view):
    return (
        gr.update(visible=True if view == "edit" else "hidden"),
        gr.update(visible=True if view == "result" else "hidden"),
    )


def on_ui_tabs():
    from modules_forge.forge_canvas.canvas import ForgeCanvas

    with gr.Blocks(analytics_enabled=False, elem_id="qwen-image21-studio") as tab:
        with gr.Row(elem_id="qwen21-workspace"):
            with gr.Column(scale=8, min_width=340, elem_id="qwen21-visual"):
                workspace_view = gr.Radio(
                    [("編集元・描き込み", "edit"), ("生成結果", "result")],
                    value="edit",
                    label="表示",
                    show_label=False,
                    elem_id="qwen21-view",
                )
                with gr.Group(elem_id="qwen21-edit-tab") as edit_view:
                    gallery = gr.Gallery(
                        label="参照画像（任意・最大10枚、左からImage 1、Image 2…）",
                        type="filepath",
                        format="png",
                        columns=6,
                        height=140,
                        object_fit="contain",
                        allow_preview=False,
                        file_types=["image"],
                        sources=["upload", "clipboard"],
                        buttons=[],
                        interactive=True,
                        elem_id="qwen21-references",
                    )
                    with gr.Row(visible=False, elem_id="qwen21-reference-controls") as reference_controls:
                        annotate = gr.Button("選択画像を開く・囲む", size="sm", elem_id="qwen21-open-annotation")
                        previous = gr.Button("前へ", size="sm")
                        following = gr.Button("後へ", size="sm")
                        remove = gr.Button("選択画像を削除", size="sm")
                        clear = gr.Button("参照をクリア", size="sm")
                    with gr.Group(
                        visible=gradio_compat.keep_hidden_component_mounted(False),
                        elem_id="qwen21-annotation-panel",
                    ) as annotation_panel:
                        annotation_title = gr.Markdown("編集する画像")
                        gr.Markdown("**Fで拡大・縮小 / Shiftで消しゴム / Ctrl+Zで取り消し**")
                        annotation_canvas = ForgeCanvas(
                            no_upload=True,
                            height=640,
                            scribble_color="#ef4444",
                            scribble_width=3,
                            scribble_alpha=100,
                            scribble_alpha_fixed=True,
                            scribble_softness_fixed=True,
                            numpy=False,
                            elem_id="qwen21-annotation-editor",
                            file_background=True,
                        )
                        close_editor = gr.Button("囲みを使わず閉じる", size="sm")
                        gr.Markdown("元画像は保持します。囲みは目印のため、範囲外も変わる場合があります。")
                with gr.Group(visible="hidden", elem_id="qwen21-result-tab") as result_view:
                    output = gr.Image(
                        label="生成結果",
                        type="filepath",
                        image_mode=None,
                        format="png",
                        height=680,
                        interactive=False,
                        buttons=["download", "fullscreen"],
                        elem_id="qwen21-output",
                    )
                    with gr.Row():
                        edit_result = gr.Button(
                            "この画像を続けて編集",
                            interactive=False,
                            elem_id="qwen21-edit-result",
                            variant="primary",
                        )
                        use = gr.Button("参照画像に追加", interactive=False)
                    files = gr.File(
                        label="PNG・生成条件を保存",
                        file_count="multiple",
                        interactive=False,
                        visible=gradio_compat.keep_hidden_component_mounted(False),
                    )
                    gr.Markdown("保存先: `outputs/qwen-image-2.1/`")
            with gr.Column(scale=3, min_width=280, elem_id="qwen21-controls"):
                gr.Markdown("### Qwen Image 2.1")
                prompt = gr.Textbox(
                    label="プロンプト・編集指示",
                    lines=4,
                    max_lines=8,
                    placeholder="例：赤い囲みの中を消して、背景になじませて。参照なしなら新規生成。",
                    elem_id="qwen21-prompt",
                )
                with gr.Row():
                    generate = gr.Button("生成・編集", variant="primary", elem_id="qwen21-generate")
                    stop = gr.Button("停止", interactive=False, elem_id="qwen21-stop")
                status = gr.Textbox(
                    value="未実行", label="進行状況", lines=2, interactive=False, elem_id="qwen21-status"
                )
                precision = gr.Radio(
                    [("INT8 · メモリ節約", "int8"), ("BF16", "bf16")],
                    value="int8",
                    label="精度",
                )
                resolution = gr.Dropdown(RESOLUTIONS, value="1024x1024", label="出力サイズ")
                transparent = gr.Checkbox(value=False, label="透過背景を指示（RGBA PNG）")
                with gr.Accordion("生成設定", open=False):
                    memory_mode = gr.Radio(
                        [("CPUへ退避 · VRAM節約", "offload"), ("GPUに配置", "gpu")],
                        value="offload",
                        label="モデルの配置",
                    )
                    seed = gr.Textbox(value="-1", label="Seed（-1: 毎回ランダム）")
                    steps = gr.Slider(1, 100, value=40, step=1, label="Steps")
                    gr.Markdown("2K・複数参照・BF16は必要メモリが増えます。最初は1024px程度で確認してください。")
                with gr.Accordion("実行環境", open=False):
                    gr.Markdown("初回は `aikimi-qwen-image21-setup.bat` で専用環境とモデルを準備します。")
                    check = gr.Button("導入状態を確認", size="sm")
                    environment = gr.Textbox(value=check_runtime(), label="導入状態", interactive=False, lines=2)
        job, selected, annotation_target = gr.State(""), gr.State(-1), gr.State("")
        back, forward = gr.State(-1), gr.State(1)
        timer = gr.Timer(1, active=False)
        workspace_view.change(switch_workspace, inputs=workspace_view, outputs=[edit_view, result_view], **PRIVATE)
        generate.click(
            start_canvas,
            inputs=[
                prompt,
                gallery,
                resolution,
                transparent,
                precision,
                memory_mode,
                seed,
                steps,
                annotation_target,
                annotation_canvas.background,
                annotation_canvas.foreground,
                output,
            ],
            outputs=[job, status, generate, stop, timer, output, files, use, edit_result],
            concurrency_limit=1,
            concurrency_id="qwen-image21-submit",
            trigger_mode="once",
            **PRIVATE,
        )
        timer.tick(
            poll,
            inputs=job,
            outputs=[status, generate, stop, timer, output, files, use, edit_result, workspace_view],
            **PRIVATE,
        )
        stop.click(cancel, inputs=job, outputs=[status, stop], queue=False, **PRIVATE)
        gallery.select(select_reference, outputs=selected, **PRIVATE)
        gallery.change(
            refresh_canvas,
            inputs=[gallery, annotation_target],
            outputs=[
                reference_controls,
                annotation_target,
                annotation_canvas.background,
                annotation_canvas.foreground,
                annotation_panel,
                annotation_title,
            ],
            **PRIVATE,
        )
        annotate.click(
            open_canvas,
            inputs=[gallery, selected],
            outputs=[
                annotation_target,
                annotation_canvas.background,
                annotation_canvas.foreground,
                annotation_panel,
                annotation_title,
            ],
            **PRIVATE,
        )
        close_editor.click(
            close_canvas,
            outputs=[
                annotation_target,
                annotation_canvas.background,
                annotation_canvas.foreground,
                annotation_panel,
                annotation_title,
            ],
            **PRIVATE,
        )
        previous.click(move_reference, inputs=[gallery, selected, back], outputs=[gallery, selected], **PRIVATE)
        following.click(move_reference, inputs=[gallery, selected, forward], outputs=[gallery, selected], **PRIVATE)
        remove.click(
            remove_reference, inputs=[gallery, selected], outputs=[gallery, selected, reference_controls], **PRIVATE
        )
        clear.click(
            lambda: ([], -1, gr.update(visible=False), *close_canvas()),
            outputs=[
                gallery,
                selected,
                reference_controls,
                annotation_target,
                annotation_canvas.background,
                annotation_canvas.foreground,
                annotation_panel,
                annotation_title,
            ],
            **PRIVATE,
        )
        use.click(use_result, inputs=[job, gallery], outputs=[gallery, selected, reference_controls], **PRIVATE)
        edit_result.click(
            continue_canvas,
            inputs=job,
            outputs=[
                gallery,
                selected,
                reference_controls,
                annotation_target,
                annotation_canvas.background,
                annotation_canvas.foreground,
                annotation_panel,
                annotation_title,
                workspace_view,
            ],
            **PRIVATE,
        )
        check.click(check_runtime, outputs=environment, **PRIVATE)
    return [(tab, "Qwen Image 2.1", "qwen_image21_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
