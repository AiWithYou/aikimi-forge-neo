"""Optional CPU-only outpaint preparation tab; never starts a model worker."""

from __future__ import annotations

import gradio as gr

from modules import script_callbacks
from modules_forge.qwen_image21.outpaint import SOURCE, prepare, recipe, stitch

PRIVATE = {"api_visibility": "private", "show_progress": "hidden", "queue": False}


def prepare_canvas(source, left, top, right, bottom, version, scene):
    try:
        original, canvas, plan = prepare(source, left, top, right, bottom)
        settings = recipe(plan, version, scene)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    summary = (
        f"**{plan.size[0]} × {plan.size[1]} px** · 実際の余白（左／上／右／下）: "
        f"{plan.left}／{plan.top}／{plan.right}／{plan.bottom} px。元画像の拡縮・切り抜きはしていません。\n\n"
        + "\n\n".join(settings["warnings"])
    )
    return (original, plan), canvas, settings["prompt"], settings, summary, None, None


def restore_original(state, generated, feather):
    if not state:
        raise gr.Error("先に余白付き画像を準備してください。")
    try:
        original, plan = state
        return stitch(original, generated, plan, feather)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc


def clear_prepared():
    return None, None, "", None, "余白付き画像を準備してください。", None, None


def clear_result():
    return None


def on_ui_tabs():
    with gr.Blocks() as tab:
        state = gr.State(None)
        gr.Markdown(
            "## Qwen 2.1 Outpaint · 前後処理\n"
            "**このタブはLoRAの読み込み・画像生成を行いません。** "
            "灰色の余白付き参照を作成し、**別途ComfyUIで生成した画像**へ元画像を戻すための補助です。 "
            "Forge側のINT8・GGUF用LoRAローダーではありません。\n\n"
            f"[ausbossのモデル・手順]({SOURCE}) · "
            "調査記録: `docs/qwen-image21-community-2026-09-26.md`"
        )
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="1 · 元画像", type="pil", image_mode=None, format="png")
                with gr.Row():
                    left = gr.Slider(0, 2048, value=128, step=1, label="左の余白 px")
                    right = gr.Slider(0, 2048, value=128, step=1, label="右の余白 px")
                with gr.Row():
                    top = gr.Slider(0, 2048, value=128, step=1, label="上の余白 px")
                    bottom = gr.Slider(0, 2048, value=128, step=1, label="下の余白 px")
                version = gr.Radio(
                    choices=[("v1 · 大きめの拡張／約1 MP", "v1"), ("v2 · 日常的な拡張／1〜2 MP", "v2")],
                    value="v2", label="ComfyUIで使用するLoRA（ここでは読み込みません）",
                )
                scene = gr.Textbox(label="場面の説明（任意）", lines=3, max_lines=6)
                gr.Markdown(
                    "32の倍数への調整は指定した拡張方向だけに足します。"
                    "拡張しない方向の長さが32の倍数でない場合は、その方向にも少量の余白を指定してください。"
                    "キャンバスは約2 MPまでです。"
                )
                prepare_button = gr.Button("余白付き画像と設定を準備", variant="primary")
            with gr.Column():
                padded = gr.Image(label="2 · ComfyUIへ渡す参照PNG（ダウンロード）", type="pil", format="png", interactive=False)
                summary = gr.Markdown("余白付き画像を準備してください。")
                prompt = gr.Textbox(label="ComfyUIへコピーする指示", lines=4, interactive=False)
                with gr.Accordion("ComfyUI用の設定メモ（ワークフローJSONではありません）", open=False):
                    settings = gr.JSON(label="生成設定とキャンバス位置")
        gr.Markdown(
            "### 3 · ComfyUIで生成後、元画像を復元\n"
            "参照は `image_1`、参照のresolutionは `0`。エンコーダーのlatentを使用します。"
            "25 steps／CFG 1／Euler／Simple／denoise 1。**Set Latent Noise Maskは使いません。** "
            "生成結果は準備したPNGと同じサイズにしてください。\n\n"
            "境界ぼかし0は元画像の全画素を復元します。32 pxでは拡張側の境界だけを混ぜ、内側は元画素を保持します。"
            "元画像・余白・LoRA選択・説明を変えた場合は準備をやり直してください。"
        )
        with gr.Row():
            with gr.Column():
                generated = gr.Image(label="ComfyUIで生成した同サイズの画像", type="pil", image_mode=None, format="png")
                feather = gr.Slider(0, 128, value=32, step=1, label="元画像の内側で境界をぼかす幅 px")
                restore_button = gr.Button("元画像を復元してPNGを作成")
            restored = gr.Image(label="復元済みPNG（ダウンロード）", type="pil", format="png", interactive=False)

        prepared_outputs = [state, padded, prompt, settings, summary, restored, generated]
        prepare_inputs = [source, left, top, right, bottom, version, scene]
        prepare_button.click(prepare_canvas, inputs=prepare_inputs, outputs=prepared_outputs, **PRIVATE)
        restore_button.click(restore_original, inputs=[state, generated, feather], outputs=restored, **PRIVATE)
        for component in prepare_inputs:
            component.change(clear_prepared, inputs=[], outputs=prepared_outputs, **PRIVATE)
        for component in (generated, feather):
            component.change(clear_result, inputs=[], outputs=restored, **PRIVATE)
    return [(tab, "Qwen Outpaint 補助", "qwen_image21_outpaint")]


script_callbacks.on_ui_tabs(on_ui_tabs)
