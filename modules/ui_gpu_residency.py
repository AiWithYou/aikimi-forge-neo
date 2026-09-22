"""全Studioで共通のモデル保持設定。通常のUI再読み込みの対象に含める。"""

import uuid

import gradio as gr

from modules import shared
from modules_forge import gpu_residency


def change_policy(value):
    if value not in {"auto", "keep", "release"}:
        raise gr.Error("モデル保持の設定を選んでください。")
    previous = gpu_residency.policy()
    try:
        shared.opts.set("aikimi_model_retention", value)
        if shared.opts.aikimi_model_retention != value:
            raise RuntimeError("モデルを解放できませんでした。")
        shared.opts.save(shared.config_filename)
    except Exception as exc:
        shared.opts.set("aikimi_model_retention", previous)
        raise gr.Error("設定を保存できませんでした。保存先と実行環境の状態を確認してください。") from exc
    return "設定を保存しました。実行中の生成がある場合は、終了後に適用します。"


def create_ui(blocks):
    with gr.Group(elem_id="aikimi-gpu-retention"):
        with gr.Row():
            with gr.Column(scale=5, min_width=240):
                with gr.Accordion("GPU・モデル保持", open=False):
                    mode = gr.Radio(
                        [("保持（手動で解放）", "keep"), ("自動（5分後に解放）", "auto"), ("毎回解放", "release")],
                        value=gpu_residency.policy(),
                        label="生成後のモデル保持",
                    )
                    gr.Markdown("同じモデルは再利用し、別の機能がGPUを使うときは待機モデルを解放します。")
            release = gr.Button("モデルを解放", scale=1, min_width=160, elem_id="aikimi-unload-model")
        status = gr.HTML(elem_id="aikimi-model-release-status")
        mode.input(change_policy, inputs=mode, outputs=status, queue=False, api_visibility="private")
        release.click(release_models, outputs=status, queue=False, api_visibility="private")
        blocks.load(gpu_residency.policy, outputs=mode, queue=False, api_visibility="private")


def release_models():
    from modules.aikimi_status import studio_status_html

    try:
        message = gpu_residency.release_idle()
        stage = "released" if "解放しました" in message else "warning"
    except Exception as exc:
        message, stage = str(exc), "error"
    return studio_status_html("model-release", stage, message, job_id=uuid.uuid4().hex, visible=True)
