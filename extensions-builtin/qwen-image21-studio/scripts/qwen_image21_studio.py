"""Qwen Image 2.1 generation and editing without importing its model at startup."""

from __future__ import annotations

import html
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
    precision_label,
    runtime_manifest,
    runtime_status,
)
from modules_forge.qwen_image21.quantized_cache import saved_status
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
    ("編集元と同じサイズ", "reference"),
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
    # A component update preserves None; gr.update drops it before serialization.
    return gr.Gallery(value=reference_gallery(paths), selected_index=None, render=False)


def reference_controls_visibility(gallery):
    return gr.update(visible=bool(gallery))


def select_reference(event: gr.SelectData):
    selected = event.index if event.selected and isinstance(event.index, int) else -1
    return selected, gr.Gallery(selected_index=None, render=False)


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


def result_path(identifier, request: gr.Request, variant="preferred"):
    # Preserve the original two-argument callback contract for ordinary jobs.
    return (
        STUDIO.artifact(identifier, owner(request))
        if variant == "preferred"
        else STUDIO.artifact(identifier, owner(request), variant)
    )


def continue_edit(identifier, request: gr.Request, variant="preferred"):
    path = str(result_path(identifier, request, variant))
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
    rewrite_prompt=False,
    sparse_mode="off",
    sparse_keep_percent=75,
    sparse_jev_cadence="legacy",
    sparse_jev_interval=2,
    rewrite_edit_prompt=False,
    preserve_unmasked=False,
    edit_mask_reference=-1,
    edit_mask_path="",
    mask_feather=0,
    sparse_jev_max_calls=0,
    sparse_jev_max_wait_seconds=0,
):
    try:
        if resolution not in {value for _, value in RESOLUTIONS}:
            raise QwenImage21Error("出力サイズを選択してください。")
        paths = reference_paths(gallery)
        annotation_reference, annotation_layers = resolve_annotation(paths, annotation_target, annotation_editor)
        if resolution == "reference":
            if not paths:
                raise QwenImage21Error("同じサイズにする編集元の参照画像を指定してください。")
            index = (
                edit_mask_reference
                if preserve_unmasked
                else (reference_index(paths, annotation_target) if annotation_target else 0)
            )
            if not isinstance(index, int) or not 0 <= index < len(paths):
                raise QwenImage21Error("編集元の参照画像を選択してください。")
            with annotation_preview(paths[index]) as image:
                width, height = image.size
        else:
            width, height = (int(value) for value in resolution.split("x"))
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
            rewrite_prompt=rewrite_prompt,
            sparse_mode=sparse_mode,
            sparse_keep_percent=sparse_keep_percent,
            sparse_jev_cadence=sparse_jev_cadence,
            sparse_jev_interval=sparse_jev_interval,
            rewrite_edit_prompt=rewrite_edit_prompt,
            preserve_unmasked=preserve_unmasked,
            edit_mask_reference=edit_mask_reference,
            edit_mask_path=edit_mask_path,
            mask_feather=mask_feather,
            sparse_jev_max_calls=sparse_jev_max_calls,
            sparse_jev_max_wait_seconds=sparse_jev_max_wait_seconds,
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
            gr.update(value=""),
        )
    except Exception as exc:
        return gr.update(), str(exc), *[gr.update() for _ in range(8)]


def poll(identifier, request: gr.Request, variant="preferred"):
    if not identifier:
        return [gr.update()] * 10
    done = False
    try:
        state = STUDIO.status(identifier, owner(request))
        done = state["done"]
        text = f"{state['message']}  ·  経過 {state['elapsed']:.0f} 秒"
        output, files, usable = gr.update(), gr.update(), False
        if done and state["state"] == "complete":
            path = result_path(identifier, request, variant)
            output, usable = (
                gr.update(
                    value=str(path),
                    label=("生成そのまま" if variant == "original" else "範囲外固定")
                    if state.get("preserved_output_path")
                    else "生成結果",
                ),
                True,
            )
            paths = [str(path)]
            if state.get("preserved_output_path"):
                if variant == "original":
                    paths.insert(0, str(STUDIO.artifact(identifier, owner(request))))
                else:
                    paths.append(str(STUDIO.artifact(identifier, owner(request), "original")))
            files = gr.update(value=[*paths, str(path.with_name("result.json"))], visible=True)
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
            gr.update(value=state.get("effective_prompt", "")) if usable else gr.update(),
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
                gr.update(),
            )
        return str(exc), *[gr.update() for _ in range(9)]


def cancel(identifier, request: gr.Request):
    if STUDIO.cancel(identifier, owner(request)):
        return "停止を要求しました。workerの終了を確認中です。", gr.update(interactive=False)
    return "この画面で停止できる実行中ジョブはありません。", gr.update(interactive=False)


def use_result(identifier, gallery, request: gr.Request, variant="preferred"):
    paths = reference_paths(gallery)
    if len(paths) >= MAX_REFERENCE_IMAGES:
        raise gr.Error("参照画像は10枚までです。追加する前に1枚削除してください。")
    paths.append(str(result_path(identifier, request, variant)))
    return (
        update_reference_gallery(paths),
        len(paths) - 1,
        reference_controls_visibility(paths),
    )


def check_runtime():
    from modules_forge.qwen_image21.prompt_rewriter import rewriter_status

    return runtime_status(RUNTIME) + "\n" + rewriter_status(RUNTIME) + "\n" + rewriter_status(RUNTIME, editing=True)


def save_quantized(precision, request: gr.Request):
    try:
        identifier = STUDIO.prepare(precision, owner(request))
        return (
            identifier,
            f"{precision.upper()}モデルを準備しています。",
            gr.update(interactive=False),
            gr.update(visible=True, interactive=True),
            gr.update(active=True),
            gr.update(interactive=False),
        )
    except Exception as exc:
        return gr.update(), str(exc), *[gr.update() for _ in range(4)]


def poll_save(identifier, precision, request: gr.Request):
    if not identifier:
        return [gr.update()] * 5
    try:
        state = STUDIO.status(identifier, owner(request))
        done = state["done"]
        text = state["message"]
        if not done:
            text += f" · 経過 {state['elapsed']:.0f} 秒"
        return (
            text,
            gr.update(interactive=done and precision in {"int8", "w4a8"}),
            gr.update(visible=not done, interactive=not done),
            gr.update(active=not done),
            gr.update(interactive=done),
        )
    except JobNotFound as exc:
        return (
            str(exc),
            gr.update(interactive=precision in {"int8", "w4a8"}),
            gr.update(visible=False),
            gr.update(active=False),
            gr.update(interactive=True),
        )


def model_save_status(precision, identifier, request: gr.Request):
    if identifier:
        try:
            if not STUDIO.status(identifier, owner(request))["done"]:
                return gr.update(), gr.update(interactive=False)
        except JobNotFound:
            pass
    return saved_status(RUNTIME, precision), gr.update(interactive=precision in {"int8", "w4a8"})


def profile_settings(precision, previous_precision):
    turbo = precision in {"turbo_bf16", "turbo_q4_k_m"}
    if turbo:
        return gr.update(value=4, interactive=False), gr.update(value="off", interactive=False), precision
    steps = (
        gr.update(value=40, interactive=True)
        if previous_precision.startswith("turbo_")
        else gr.update(interactive=True)
    )
    if precision == "base_q4_k_m":
        return steps, gr.update(value="off", interactive=False), precision
    return steps, gr.update(interactive=True), precision


def default_precision():
    try:
        runtime_manifest(RUNTIME, "base_q4_k_m")
    except QwenImage21Error:
        try:
            runtime_manifest(RUNTIME, "int8")
        except QwenImage21Error:
            return "base_q4_k_m"
        return "int8"
    return "base_q4_k_m"


def refresh_saved_after_generation(identifier, precision, request: gr.Request):
    try:
        state = STUDIO.status(identifier, owner(request)) if identifier else {}
        if state.get("done") and state.get("state") == "complete":
            return saved_status(RUNTIME, precision)
    except JobNotFound:
        pass
    return gr.update()


def assistant_status(identifier, request: gr.Request):
    """Expose only this browser's job state to the mascot, without prompts/paths."""
    from modules.aikimi_security.redaction import safe_error_message

    if not identifier:
        return ""
    try:
        state = STUDIO.status(identifier, owner(request))
    except JobNotFound:
        return '<span data-state="idle"></span>'
    stage = state.get("stage", "loading") if not state["done"] else state["state"]
    label = safe_error_message(state.get("message", ""), limit=240)
    precision = precision_label(state.get("precision", ""))
    return '<span data-state="{}" data-progress="{}" data-message="{}" data-model="{}" data-result="{}" data-job="{}"></span>'.format(
        html.escape(str(stage), quote=True),
        html.escape(str(state.get("progress", 0)), quote=True),
        html.escape(label, quote=True),
        html.escape(f"Qwen Image 2.1 · {precision}", quote=True),
        "qwen21-output"
        if state.get("operation") != "prepare" and state["done"] and state["state"] == "complete"
        else "",
        html.escape(identifier, quote=True),
    )


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


def continue_canvas(identifier, request: gr.Request, variant="preferred"):
    gallery, selected, controls, target, editor, panel, view = continue_edit(identifier, request, variant)
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
    rewrite_prompt=False,
    sparse_mode="off",
    sparse_keep_percent=75,
    sparse_jev_cadence="legacy",
    sparse_jev_interval=2,
    rewrite_edit_prompt=False,
    preserve_unmasked=False,
    mask_target="",
    mask_background=None,
    mask_foreground=None,
    mask_upload=None,
    mask_source="paint",
    mask_feather=0,
    sparse_jev_max_calls=0,
    sparse_jev_max_wait_seconds=0,
):
    # Keep the bridge files alive until Studio.start snapshots the request.
    # The original reference is still read from Gallery, never from this preview.
    try:
        with TemporaryDirectory(prefix="qwen-canvas-") as temporary:
            directory = Path(temporary)
            editor = None
            if annotation_target and foreground is not None and foreground.getchannel("A").getbbox() is not None:
                if background is None:
                    raise QwenImage21Error("編集元を読み込み直してください。")
                background.save(directory / "background.png", format="PNG")
                foreground.save(directory / "marks.png", format="PNG")
                editor = {"background": str(directory / "background.png"), "layers": [str(directory / "marks.png")]}
            mask_reference, mask_path = -1, ""
            if preserve_unmasked:
                paths = reference_paths(gallery)
                if not mask_target:
                    raise QwenImage21Error("範囲外を固定する編集元を開いてください。")
                mask_reference = reference_index(paths, mask_target)
                if mask_source == "upload":
                    if not isinstance(mask_upload, (str, Path)) or not mask_upload:
                        raise QwenImage21Error("編集範囲のマスク画像をアップロードしてください。")
                    mask_path = str(mask_upload)
                elif mask_source == "paint":
                    if mask_background is None or mask_foreground is None:
                        raise QwenImage21Error("範囲外を固定するには変更する範囲を塗ってください。")
                    mask_background.save(directory / "mask-background.png", format="PNG")
                    mask_foreground.save(directory / "mask-paint.png", format="PNG")
                    painted_reference, _ = resolve_annotation(
                        paths,
                        mask_target,
                        {
                            "background": str(directory / "mask-background.png"),
                            "layers": [str(directory / "mask-paint.png")],
                        },
                    )
                    if painted_reference != mask_reference:
                        raise QwenImage21Error("編集マスクが空です。変更する範囲を塗ってください。")
                    mask_path = str(directory / "mask.png")
                    mask_foreground.getchannel("A").save(mask_path, format="PNG")
                else:
                    raise QwenImage21Error("マスクの指定方法が不正です。")
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
                rewrite_prompt,
                sparse_mode,
                sparse_keep_percent,
                sparse_jev_cadence,
                sparse_jev_interval,
                rewrite_edit_prompt,
                preserve_unmasked,
                mask_reference,
                mask_path,
                mask_feather,
                sparse_jev_max_calls,
                sparse_jev_max_wait_seconds,
            )
    except Exception as exc:
        return gr.update(), str(exc), *[gr.update() for _ in range(8)]


def switch_workspace(view):
    return (
        gr.update(visible=True if view == "edit" else "hidden"),
        gr.update(visible=True if view == "result" else "hidden"),
    )


def refresh_mask(gallery, annotation_target, current_target, enabled=True):
    paths = reference_paths(gallery)
    if not enabled or not annotation_target:
        return "", None, None, None
    try:
        index = reference_index(paths, annotation_target)
        if current_target and reference_index(paths, current_target) == index:
            return paths[index], gr.update(), gr.update(), gr.update()
    except QwenImage21Error:
        if not paths or annotation_target not in paths:
            return "", None, None, None
        index = paths.index(annotation_target)
    return paths[index], annotation_preview(paths[index]), None, None


def result_variants(identifier, request: gr.Request, current_identifier=""):
    try:
        state = STUDIO.status(identifier, owner(request)) if identifier else {}
    except JobNotFound:
        state = {}
    if identifier == current_identifier and state.get("done"):
        return gr.update(), gr.update()
    complete = state.get("done") and state.get("state") == "complete"
    return (
        gr.update(value="preferred", visible=True if complete and state.get("preserved_output_path") else "hidden"),
        identifier if complete else "",
    )


def select_result_variant(identifier, variant, request: gr.Request):
    state = STUDIO.status(identifier, owner(request))
    if not state.get("done") or state.get("state") != "complete":
        return gr.update()
    path = result_path(identifier, request, variant)
    label = (
        ("生成そのまま" if variant == "original" else "範囲外固定")
        if state.get("preserved_output_path")
        else "生成結果"
    )
    return gr.update(value=str(path), label=label)


def on_ui_tabs():
    from modules_forge.forge_canvas.canvas import ForgeCanvas

    selected_precision = default_precision()
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
                        preserve_unmasked = gr.Checkbox(
                            value=False,
                            label="マスク範囲外を元画像に固定",
                            info="塗った部分だけ生成結果を反映します。編集元と同じ出力サイズが必要です。",
                            elem_id="qwen21-preserve-unmasked",
                        )
                        # Keep the Canvas HTML lifecycle mounted. CSS follows
                        # the checkbox/source radio without remounting its JS.
                        with gr.Group(elem_id="qwen21-mask-panel"):
                            mask_source = gr.Radio(
                                [("塗って指定", "paint"), ("マスク画像", "upload")],
                                value="paint",
                                label="編集範囲",
                                elem_id="qwen21-mask-source",
                            )
                            with gr.Group(elem_id="qwen21-mask-paint"):
                                gr.Markdown("**変更する部分を塗る / Shiftで消す / Ctrl+Zで取り消し**")
                                mask_canvas = ForgeCanvas(
                                    no_upload=True,
                                    height=460,
                                    scribble_color="#38bdf8",
                                    scribble_color_fixed=True,
                                    scribble_width=36,
                                    scribble_alpha=100,
                                    scribble_alpha_fixed=True,
                                    scribble_softness_fixed=True,
                                    numpy=False,
                                    elem_id="qwen21-mask-editor",
                                    file_background=True,
                                )
                                # Gradio deduplicates head scripts by URL. A
                                # second HTML component can run before the
                                # first component's shared script has loaded.
                                mask_canvas.block.js_on_load = (
                                    "const initializeMask = () => {"
                                    "if (!element.isConnected) return;"
                                    "if (typeof ForgeCanvas !== 'function') { setTimeout(initializeMask, 50); return; }"
                                    + mask_canvas.block.js_on_load
                                    + "}; initializeMask();"
                                )
                            mask_upload = gr.Image(
                                label="白＝編集・黒＝保持（編集元と同じサイズ）",
                                type="filepath",
                                image_mode=None,
                                format="png",
                                sources=["upload", "clipboard"],
                                interactive=True,
                                elem_id="qwen21-mask-upload",
                            )
                            mask_feather = gr.Slider(0, 64, value=0, step=1, label="境界を内側へぼかす px")
                with gr.Group(visible="hidden", elem_id="qwen21-result-tab") as result_view:
                    result_variant = gr.Radio(
                        [("範囲外固定", "preferred"), ("生成そのまま", "original")],
                        value="preferred",
                        label="表示・次の編集に使う画像",
                        visible="hidden",
                        elem_id="qwen21-result-variant",
                    )
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
                    effective_prompt = gr.Textbox(
                        label="使用したプロンプト",
                        lines=4,
                        max_lines=8,
                        interactive=False,
                        buttons=["copy"],
                        elem_id="qwen21-effective-prompt",
                    )
            with gr.Column(scale=3, min_width=280, elem_id="qwen21-controls"):
                gr.Markdown("### Qwen Image 2.1")
                prompt = gr.Textbox(
                    label="プロンプト・編集指示",
                    lines=4,
                    max_lines=8,
                    placeholder="例：赤い囲みの中を消して、背景になじませて。参照なしなら新規生成。",
                    elem_id="qwen21-prompt",
                )
                rewrite_prompt = gr.Checkbox(
                    value=False,
                    label="プロンプトを書き換える（4bit）",
                    info="新規生成用。参照画像がある編集では自動で省略します。",
                    elem_id="qwen21-rewrite-prompt",
                )
                rewrite_edit_prompt = gr.Checkbox(
                    value=False,
                    label="編集指示を書き換える（4bit）",
                    info="参照画像を使う編集用。画像と編集指示から使用するプロンプトを整えます。",
                    elem_id="qwen21-rewrite-edit-prompt",
                )
                with gr.Row():
                    generate = gr.Button("生成・編集", variant="primary", elem_id="qwen21-generate")
                    stop = gr.Button("停止", interactive=False, elem_id="qwen21-stop")
                status = gr.Textbox(
                    value="未実行", label="進行状況", lines=2, interactive=False, elem_id="qwen21-status"
                )
                precision = gr.Radio(
                    [
                        ("通常 · Q4_K_M (Unsloth)", "base_q4_k_m"),
                        ("通常 · INT8", "int8"),
                        ("通常 · W4A8", "w4a8"),
                        ("通常 · BF16", "bf16"),
                        ("Viggle Turbo · BF16", "turbo_bf16"),
                        ("Viggle Turbo · Q4_K_M", "turbo_q4_k_m"),
                    ],
                    value=selected_precision,
                    label="モデル・精度",
                    elem_id="qwen21-precision",
                )
                gr.Markdown(
                    "通常版: [Unsloth Q4_K_M](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)"
                    " · Turbo: [Viggle BF16](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)"
                    " · [Q4_K_M GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF)"
                )
                with gr.Row():
                    save_model = gr.Button(
                        "変換モデルを保存",
                        size="sm",
                        interactive=selected_precision in {"int8", "w4a8"},
                        elem_id="qwen21-save-model",
                    )
                    stop_save = gr.Button("保存を停止", size="sm", visible=False, elem_id="qwen21-stop-save")
                save_status = gr.Textbox(
                    value=saved_status(RUNTIME, selected_precision),
                    label="モデルの保存状態",
                    show_label=False,
                    interactive=False,
                    lines=2,
                    elem_id="qwen21-save-status",
                )
                assistant = gr.HTML(visible="hidden", elem_id="qwen21-assistant-state")
                resolution = gr.Dropdown(
                    RESOLUTIONS, value="1024x1024", label="出力サイズ", elem_id="qwen21-resolution"
                )
                transparent = gr.Checkbox(value=False, label="透過背景を指示（RGBA PNG）")
                with gr.Accordion("生成設定", open=False):
                    memory_mode = gr.Radio(
                        [("CPUへ退避 · VRAM節約", "offload"), ("GPUに配置", "gpu")],
                        value="offload",
                        label="モデルの配置",
                    )
                    seed = gr.Textbox(value="-1", label="Seed（-1: 毎回ランダム）")
                    steps = gr.Slider(1, 100, value=40, step=1, label="Steps")
                    from modules_forge.jev_sparse.qwen21_integration import launch_defaults

                    sparse_defaults = launch_defaults()
                    sparse_mode = gr.Dropdown(
                        [
                            ("OFF · 通常生成", "off"),
                            ("Dense · 速度計測", "dense"),
                            ("固定Sparse · 通信なし", "fixed"),
                            ("数値ルール · 通信なし", "rules"),
                            ("Jev速度優先 · 集約統計を外部送信", "jev"),
                        ],
                        value="off" if selected_precision == "base_q4_k_m" else sparse_defaults.mode,
                        interactive=selected_precision != "base_q4_k_m",
                        label="Sparse Attention",
                        elem_id="qwen21-sparse-mode",
                        info="Jevは保存済みのキーを使用します。画像・プロンプトは送信しません。方式の変更は次の生成から適用します。",
                    )
                    sparse_keep = gr.Slider(
                        1,
                        100,
                        value=sparse_defaults.keep_percent,
                        step=1,
                        label="固定Sparseの保持率 %",
                        visible=selected_precision != "base_q4_k_m" and sparse_defaults.mode == "fixed",
                    )
                    sparse_mode.change(
                        lambda mode: gr.update(visible=mode == "fixed"),
                        inputs=sparse_mode,
                        outputs=sparse_keep,
                        **PRIVATE,
                    )
                    from modules_forge.jev_sparse.ui import budget_controls, decision_controls

                    sparse_cadence, sparse_interval = decision_controls("qwen21", sparse_mode)
                    sparse_max_calls, sparse_max_wait = budget_controls(
                        "qwen21", sparse_mode, steps=steps, cadence=sparse_cadence, interval=sparse_interval
                    )
                    gr.Markdown(
                        "Jevのstepはモデル評価単位です。最初の判定に必要な統計を集めた後、指定した頻度で更新します。"
                    )
                    gr.Markdown("2K・複数参照・BF16は必要メモリが増えます。最初は1024px程度で確認してください。")
                with gr.Accordion("実行環境", open=False):
                    from modules_forge.jev_sparse.ui import credential_controls

                    credential_controls("qwen21")
                    gr.Markdown("初回は `aikimi-qwen-image21-setup.bat` で専用環境とモデルを準備します。")
                    gr.Markdown("公式フル版 (INT8 / W4A8 / BF16): `aikimi-qwen-image21-setup.bat --official-full`")
                    gr.Markdown("Turbo BF16: `aikimi-qwen-image21-setup.bat --turbo-bf16-only`")
                    gr.Markdown("Turbo Q4_K_M: `aikimi-qwen-image21-setup.bat --turbo-q4-only`")
                    gr.Markdown("書き換えを追加: `aikimi-qwen-image21-setup.bat --prompt-rewriter-only`")
                    gr.Markdown("編集用の書き換えを追加: `aikimi-qwen-image21-setup.bat --edit-prompt-rewriter-only`")
                    check = gr.Button("導入状態を確認", size="sm")
                    environment = gr.Textbox(value=check_runtime(), label="導入状態", interactive=False, lines=5)
        job, selected, annotation_target = gr.State(""), gr.State(-1), gr.State("")
        mask_target, variant_job = gr.State(""), gr.State("")
        back, forward = gr.State(-1), gr.State(1)
        timer = gr.Timer(1, active=False)
        save_job = gr.State("")
        profile_state = gr.State(selected_precision)
        save_timer = gr.Timer(1, active=False)
        save_model.click(
            save_quantized,
            inputs=precision,
            outputs=[save_job, save_status, save_model, stop_save, save_timer, generate],
            concurrency_id="qwen-image21-submit",
            concurrency_limit=1,
            **PRIVATE,
        ).then(assistant_status, inputs=save_job, outputs=assistant, **PRIVATE)
        save_timer.tick(
            poll_save,
            inputs=[save_job, precision],
            outputs=[save_status, save_model, stop_save, save_timer, generate],
            **PRIVATE,
        ).then(assistant_status, inputs=save_job, outputs=assistant, **PRIVATE)
        stop_save.click(cancel, inputs=save_job, outputs=[save_status, stop_save], queue=False, **PRIVATE)
        precision.change(model_save_status, inputs=[precision, save_job], outputs=[save_status, save_model], **PRIVATE)
        precision.change(
            profile_settings, inputs=[precision, profile_state], outputs=[steps, sparse_mode, profile_state], **PRIVATE
        )
        tab.load(model_save_status, inputs=[precision, save_job], outputs=[save_status, save_model], **PRIVATE)
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
                rewrite_prompt,
                sparse_mode,
                sparse_keep,
                sparse_cadence,
                sparse_interval,
                rewrite_edit_prompt,
                preserve_unmasked,
                mask_target,
                mask_canvas.background,
                mask_canvas.foreground,
                mask_upload,
                mask_source,
                mask_feather,
                sparse_max_calls,
                sparse_max_wait,
            ],
            outputs=[job, status, generate, stop, timer, output, files, use, edit_result, effective_prompt],
            concurrency_limit=1,
            concurrency_id="qwen-image21-submit",
            trigger_mode="once",
            **PRIVATE,
        )
        job.change(assistant_status, inputs=job, outputs=assistant, **PRIVATE)
        timer.tick(
            poll,
            inputs=[job, result_variant],
            outputs=[status, generate, stop, timer, output, files, use, edit_result, workspace_view, effective_prompt],
            **PRIVATE,
        ).then(assistant_status, inputs=job, outputs=assistant, **PRIVATE).then(
            refresh_saved_after_generation,
            inputs=[job, precision],
            outputs=save_status,
            **PRIVATE,
        )
        stop.click(cancel, inputs=job, outputs=[status, stop], queue=False, **PRIVATE)
        preserve_unmasked.change(
            lambda enabled: gr.update(value="reference") if enabled else gr.update(),
            inputs=preserve_unmasked,
            outputs=resolution,
            **PRIVATE,
        )
        for component in (annotation_target, preserve_unmasked):
            component.change(
                refresh_mask,
                inputs=[gallery, annotation_target, mask_target, preserve_unmasked],
                outputs=[mask_target, mask_canvas.background, mask_canvas.foreground, mask_upload],
                **PRIVATE,
            )
        for component in (job, output):
            component.change(
                result_variants,
                inputs=[job, variant_job],
                outputs=[result_variant, variant_job],
                **PRIVATE,
            )
        result_variant.input(select_result_variant, inputs=[job, result_variant], outputs=output, **PRIVATE)
        gallery.select(select_reference, outputs=[selected, gallery], **PRIVATE)
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
        use.click(
            use_result,
            inputs=[job, gallery, result_variant],
            outputs=[gallery, selected, reference_controls],
            **PRIVATE,
        )
        edit_result.click(
            continue_canvas,
            inputs=[job, result_variant],
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
