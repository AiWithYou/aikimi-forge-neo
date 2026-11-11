"""Controls for MiniMax's native still-image model."""

from functools import partial

import gradio as gr

from modules_forge.p_h3_image_ui import _cancel, _generate, _setup

PRESETS = {
    "4 MP · 正方形": (2048, 2048),
    "動作確認 · 512×512": (512, 512),
    "16 MP · 正方形": (4096, 4096),
    "16 MP · 横長 4:3": (4736, 3552),
    "16 MP · 縦長 3:4": (3552, 4736),
    "16 MP · 横長 3:2": (5024, 3360),
    "16 MP · 縦長 2:3": (3360, 5024),
    "16 MP · 横長 16:9": (5472, 3072),
    "16 MP · 縦長 9:16": (3072, 5472),
    "16 MP · 横長 21:9": (6272, 2688),
}


def _run(prompt, images, width, height, seed, *, data_directory):
    images = images or []
    for status, image, files, _, _, job, generate, cancel in _generate(
        None, "low_ram", "edit" if images else "text", prompt, images[0] if images else None,
        images[1:], width, height, seed, False, None, data_directory=data_directory, official=True,
    ):
        yield status, image, files, job, generate, cancel


def _prepare():
    for _, status, button in _setup(None, script="setup_minimax_h3_image.py"):
        yield status, button


def render(*, data_directory):
    private = {"api_visibility": "private", "show_progress": "hidden"}
    with gr.Accordion("初回セットアップ", open=False):
        gr.Markdown(
            "ComfyUI 0.45.0と画像モデルを準備します。モデルは約37.8 GB "
            "（共通エンコーダーがあれば約22.1 GB）。"
            "[モデルと利用条件](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)"
        )
        setup = gr.Button("環境とモデルを準備")
        setup_status = gr.Markdown()
    with gr.Row():
        with gr.Column(min_width=300):
            prompt = gr.Textbox(label="プロンプト", lines=5)
            images = gr.File(label="入力画像（最大16枚）", file_count="multiple", type="filepath",
                             file_types=[".png", ".jpg", ".jpeg", ".webp"], allow_reordering=True)
            gr.Markdown("画像編集では先頭が編集対象の `<Picture 1>` です。参照番号は表示順で決まります。")
            preset = gr.Dropdown(list(PRESETS), value="4 MP · 正方形", label="画像サイズ")
            with gr.Accordion("サイズ・Seed", open=False):
                with gr.Row():
                    width = gr.Number(value=2048, minimum=32, maximum=16384, step=32, precision=0, label="幅 px")
                    height = gr.Number(value=2048, minimum=32, maximum=16384, step=32, precision=0, label="高さ px")
                seed = gr.Textbox(value="-1", label="Seed（-1でランダム）")
            with gr.Row():
                generate = gr.Button("生成", variant="primary")
                cancel = gr.Button("中断", interactive=False)
            status = gr.Markdown()
        with gr.Column(min_width=300):
            result = gr.Image(label="生成画像", type="filepath", format="png", interactive=False)
            files = gr.File(label="PNG・生成条件", file_count="multiple", interactive=False)
    job = gr.State({})
    preset.input(lambda value: PRESETS[value], [preset], [width, height], queue=False, **private)
    for dimension in (width, height):
        dimension.input(lambda: gr.update(value=None), outputs=preset, queue=False, **private)
    setup.click(_prepare, outputs=[setup_status, setup], concurrency_limit=1,
                concurrency_id="h3-runtime-control", trigger_mode="once", **private)
    generate.click(partial(_run, data_directory=data_directory), [prompt, images, width, height, seed],
                   [status, result, files, job, generate, cancel], concurrency_limit=1,
                   concurrency_id="minimax-h3-generation", trigger_mode="once", **private)
    cancel.click(_cancel, [job], [status, cancel], queue=False, **private)
