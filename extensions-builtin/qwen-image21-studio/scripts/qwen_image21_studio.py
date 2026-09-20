"""Qwen Image 2.1 generation and editing without importing its model at startup."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from modules import gradio_compat, script_callbacks
from modules.paths import data_path, script_path
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
    return gr.update(value=reference_gallery(paths), selected_index=target), target


def remove_reference(gallery, selected):
    # Permit deleting excess uploads so an 11th image cannot strand the control.
    paths = [str(value[0] if isinstance(value, (tuple, list)) else value) for value in (gallery or [])]
    if not isinstance(selected, int) or not 0 <= selected < len(paths):
        raise gr.Error("削除する参照画像を選んでください。")
    paths.pop(selected)
    target = min(selected, len(paths) - 1)
    return (
        gr.update(value=reference_gallery(paths), selected_index=max(0, target) if paths else None),
        target,
        reference_controls_visibility(paths),
    )


def start(prompt, gallery, resolution, transparent, precision, memory_mode, seed, steps, request: gr.Request):
    try:
        if resolution not in {value for _, value in RESOLUTIONS}:
            raise QwenImage21Error("出力サイズを選択してください。")
        width, height = (int(value) for value in resolution.split("x"))
        generation = Request(
            prompt=prompt,
            input_images=tuple(reference_paths(gallery)),
            width=width,
            height=height,
            transparent=transparent,
            precision=precision,
            memory_mode=memory_mode,
            seed=seed,
            steps=steps,
        )
        identifier = STUDIO.start(generation, owner(request))
        return (
            identifier,
            "開始しました。初回のモデル読み込みには時間がかかります。",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(active=True),
            None,
            gr.update(value=None, visible=gradio_compat.keep_hidden_component_mounted(False)),
            gr.update(interactive=False),
        )
    except Exception as exc:
        return gr.update(), str(exc), *[gr.update() for _ in range(6)]


def poll(identifier, request: gr.Request):
    if not identifier:
        return [gr.update()] * 7
    done = False
    try:
        state = STUDIO.status(identifier, owner(request))
        done = state["done"]
        text = f"{state['message']}  ·  経過 {state['elapsed']:.0f} 秒"
        output, files, usable = gr.update(), gr.update(), False
        if done and state["state"] == "complete":
            path = STUDIO.artifact(identifier, owner(request))
            output, usable = str(path), True
            files = gr.update(value=[str(path), str(path.with_name("result.json"))], visible=True)
        return (
            text,
            gr.update(interactive=done),
            gr.update(interactive=not done),
            gr.update(active=not done),
            output,
            files,
            gr.update(interactive=usable),
        )
    except JobNotFound as exc:
        return (
            str(exc),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(active=False),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
        )
    except Exception as exc:
        if done:
            return (
                str(exc),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(active=False),
                gr.update(),
                gr.update(),
                gr.update(interactive=False),
            )
        return str(exc), *[gr.update() for _ in range(6)]


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
        gr.update(value=reference_gallery(paths), selected_index=len(paths) - 1),
        len(paths) - 1,
        reference_controls_visibility(paths),
    )


def check_runtime():
    return runtime_status(RUNTIME)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, elem_id="qwen-image21-studio") as tab:
        with gr.Row():
            with gr.Column(scale=5, min_width=300):
                prompt = gr.Textbox(
                    label="プロンプト・編集指示",
                    lines=5,
                    placeholder="作りたい画像を入力。参照画像があれば、残す要素と変更する内容を指定。",
                    elem_id="qwen21-prompt",
                )
                gallery = gr.Gallery(
                    label="参照画像（任意・最大10枚、左からImage 1、Image 2…）",
                    type="filepath",
                    format="png",
                    columns=5,
                    height=180,
                    object_fit="contain",
                    allow_preview=False,
                    file_types=["image"],
                    sources=["upload", "clipboard"],
                    buttons=[],
                    interactive=True,
                    elem_id="qwen21-references",
                )
                with gr.Row(visible=False, elem_id="qwen21-reference-controls") as reference_controls:
                    previous = gr.Button("選択画像を前へ", size="sm")
                    following = gr.Button("選択画像を後へ", size="sm")
                    remove = gr.Button("選択画像を削除", size="sm")
                    clear = gr.Button("参照をクリア", size="sm")
                with gr.Row():
                    resolution = gr.Dropdown(RESOLUTIONS, value="1024x1024", label="出力サイズ")
                    transparent = gr.Checkbox(value=False, label="透過背景を指示（RGBA PNG）")
                with gr.Accordion("生成設定", open=False):
                    precision = gr.Radio(
                        [("INT8 · メモリ節約", "int8"), ("BF16", "bf16")],
                        value="int8",
                        label="精度",
                    )
                    memory_mode = gr.Radio(
                        [("CPUへ退避 · VRAM節約", "offload"), ("GPUに配置", "gpu")],
                        value="offload",
                        label="モデルの配置",
                    )
                    with gr.Row():
                        seed = gr.Textbox(value="-1", label="Seed（-1: 毎回ランダム）")
                        steps = gr.Slider(1, 100, value=40, step=1, label="Steps")
                    gr.Markdown("2K・複数参照・BF16は必要メモリが増えます。まず1024 × 1024で確認してください。")
                with gr.Row():
                    generate = gr.Button("生成・編集", variant="primary", elem_id="qwen21-generate")
                    stop = gr.Button("停止", interactive=False, elem_id="qwen21-stop")
                status = gr.Textbox(
                    value="未実行", label="進行状況", lines=2, interactive=False, elem_id="qwen21-status"
                )
                with gr.Accordion("実行環境", open=False):
                    gr.Markdown(
                        "初回は **`aikimi-qwen-image21-setup.bat`** を実行します。専用環境とモデルを準備します。"
                    )
                    check = gr.Button("導入状態を確認", size="sm")
                    environment = gr.Textbox(value=check_runtime(), label="導入状態", interactive=False, lines=2)
            with gr.Column(scale=5, min_width=300):
                output = gr.Image(
                    label="生成結果",
                    type="filepath",
                    image_mode=None,
                    format="png",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                    elem_id="qwen21-output",
                )
                use = gr.Button("この結果を参照画像に追加", interactive=False)
                files = gr.File(
                    label="PNG・生成条件を保存",
                    file_count="multiple",
                    interactive=False,
                    visible=gradio_compat.keep_hidden_component_mounted(False),
                )
                gr.Markdown("保存先: `outputs/qwen-image-2.1/`。透過はPNGのアルファチャンネルに保持します。")
        job, selected = gr.State(""), gr.State(-1)
        back, forward = gr.State(-1), gr.State(1)
        timer = gr.Timer(1, active=False)
        generate.click(
            start,
            inputs=[prompt, gallery, resolution, transparent, precision, memory_mode, seed, steps],
            outputs=[job, status, generate, stop, timer, output, files, use],
            concurrency_limit=1,
            concurrency_id="qwen-image21-submit",
            trigger_mode="once",
            **PRIVATE,
        )
        timer.tick(poll, inputs=job, outputs=[status, generate, stop, timer, output, files, use], **PRIVATE)
        stop.click(cancel, inputs=job, outputs=[status, stop], queue=False, **PRIVATE)
        gallery.select(select_reference, outputs=selected, **PRIVATE)
        gallery.change(reference_controls_visibility, inputs=gallery, outputs=reference_controls, **PRIVATE)
        previous.click(move_reference, inputs=[gallery, selected, back], outputs=[gallery, selected], **PRIVATE)
        following.click(move_reference, inputs=[gallery, selected, forward], outputs=[gallery, selected], **PRIVATE)
        remove.click(
            remove_reference, inputs=[gallery, selected], outputs=[gallery, selected, reference_controls], **PRIVATE
        )
        clear.click(
            lambda: ([], -1, gr.update(visible=False)), outputs=[gallery, selected, reference_controls], **PRIVATE
        )
        use.click(use_result, inputs=[job, gallery], outputs=[gallery, selected, reference_controls], **PRIVATE)
        check.click(check_runtime, outputs=environment, **PRIVATE)
    return [(tab, "Qwen Image 2.1", "qwen_image21_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
