"""Text-to-image tab for well9472/Nanosaur2-670M."""

from __future__ import annotations

import html
import uuid

import gradio as gr

from modules import script_callbacks
from modules.aikimi_status import studio_status_html
from modules_forge import nanosaur2_studio as studio
from modules_forge.minimax_h3_runtime import SERVER_URL

PRESETS = {
    "動作確認 · 512×512": (512, 512),
    "正方形 · 768×768": (768, 768),
    "横長 · 1216×832": (1216, 832),
    "縦長 · 832×1216": (832, 1216),
    "正方形 · 1024×1024": (1024, 1024),
}


def _integer(value, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise studio.Nanosaur2Error(f"{label}は整数で指定してください。")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise studio.Nanosaur2Error(f"{label}は整数で指定してください。") from exc
    if str(value).strip() not in {str(result), str(float(result))} or not minimum <= result <= maximum:
        raise studio.Nanosaur2Error(f"{label}は{minimum}〜{maximum}の整数で指定してください。")
    return result


def _request(prompt, negative, width, height, steps, cfg, seed, guidance):
    return studio.Nanosaur2Request(
        prompt=prompt or "",
        negative_prompt=negative or "",
        width=_integer(width, "幅", 256, 2048),
        height=_integer(height, "高さ", 256, 2048),
        steps=_integer(steps, "Steps", 1, 100),
        cfg=cfg,
        seed=_integer(seed, "Seed", -1, 2**53 - 1),
        guidance=guidance,
    )


def _setup():
    yield "Nanosaur2の実行環境・ノード・モデルを準備しています。初回は約2.13 GBのモデルを取得します。"
    try:
        from tools.setup_nanosaur2 import run

        bridge = studio._bridge()
        active_root = bridge.server_runtime_root(SERVER_URL)
        with bridge.runtime_setup_session(active_root or studio.runtime_root(), SERVER_URL):
            result = run()
        if not result["ok"]:
            raise studio.Nanosaur2Error("導入後の検証に失敗しました。")
        yield "準備できました。「接続・起動」を押すか、そのまま画像を生成してください。"
    except Exception as exc:
        yield "**セットアップに失敗しました:** " + html.escape(str(exc))


def _setup_state():
    """Check installation without starting ComfyUI or hashing gigabytes on page load."""
    from tools.setup_nanosaur2 import runtime_ready

    try:
        root = studio.runtime_root()
        if not runtime_ready() or not studio.source_ready(root):
            return "未導入、または準備が未完了です。「環境とモデルを準備」を押してください。"
        for entry in studio._entries("models"):
            path = root / "models" / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                return "モデルの準備が未完了です。「環境とモデルを準備」で不足分を用意してください。"
        return "導入済みです。生成時にモデルを検証し、接続・起動します。そのままプロンプトを入力できます。"
    except (OSError, ValueError) as exc:
        return "**準備状態を確認できません:** " + html.escape(str(exc))


def _connect(restart=False):
    yield "再起動しています…" if restart else "接続・起動しています… 初回は少し時間がかかります。"
    try:
        readiness = studio.ensure_runtime(restart=restart)
        yield (
            "Nanosaur2の接続準備ができています。 "
            f"ComfyUI: {html.escape(readiness.comfy_version or '不明')} / "
            f"GPU: {html.escape(readiness.gpu_name or '不明')}"
        )
    except Exception as exc:
        yield "**起動に失敗しました:** " + html.escape(str(exc))


def _restart():
    yield from _connect(True)


def _generate(prompt, negative, width, height, steps, cfg, seed, guidance):
    notification_id = uuid.uuid4().hex

    def notify(stage, message):
        return studio_status_html(
            "nanosaur2",
            stage,
            message,
            job_id=notification_id,
            model_name="Nanosaur2",
            result_id="nanosaur2-result" if stage == "complete" else "",
            visible=True,
        )

    yield (
        notify("prepare", "入力を確認しています。"),
        None,
        [],
        None,
        {},
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    try:
        request = _request(prompt, negative, width, height, steps, cfg, seed, guidance)
        for event in studio.run_generation(request):
            complete = event["stage"] == "complete"
            job = {"prompt_id": event["prompt_id"]} if event.get("prompt_id") and not complete else {}
            message = event["message"]
            if "elapsed" in event:
                message += f" 経過 {event['elapsed']:.0f} 秒。"
            yield (
                notify(event["stage"], message),
                event.get("path", gr.update()),
                event.get("files", gr.update()),
                event.get("metadata", gr.update()),
                job,
                gr.update(interactive=complete),
                gr.update(interactive=bool(job)),
            )
    except Exception as exc:
        yield (
            notify("cancelled" if isinstance(exc, studio.Nanosaur2Cancelled) else "error", str(exc)),
            gr.update(),
            gr.update(),
            gr.update(),
            {},
            gr.update(interactive=True),
            gr.update(interactive=False),
        )


def _cancel(job):
    if not isinstance(job, dict) or not job.get("prompt_id"):
        return "実行中の画像ジョブはありません。", gr.update(interactive=False)
    try:
        studio.cancel_generation(job["prompt_id"])
        return "画像ジョブに停止要求を送りました。", gr.update(interactive=False)
    except Exception as exc:
        return "**停止要求を送信できません:** " + html.escape(str(exc)), gr.update(interactive=True)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.Markdown(
            "## Nanosaur2\nイラスト向けテキスト画像生成。モデル配布元は[well9472/Nanosaur2-670M](https://huggingface.co/well9472/Nanosaur2-670M)です。"
        )
        with gr.Accordion("実行環境とモデル", open=False):
            gr.Markdown("専用ComfyUI・Python・Nanosaur2の3モデルを準備します。H3のモデルは不要です。")
            with gr.Row():
                setup = gr.Button("環境とモデルを準備")
                connect = gr.Button("接続・起動")
                restart = gr.Button("ComfyUIを再起動")
            setup_status = gr.Markdown("準備状態を確認しています…")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                prompt = gr.Textbox(
                    label="プロンプト",
                    lines=6,
                    placeholder="newest, masterpiece, 青い目のキツネ耳の旅人、朝の森、柔らかな日光",
                )
                negative = gr.Textbox(label="ネガティブプロンプト", value="oldest, low quality", lines=2)
                preset = gr.Dropdown(choices=[*PRESETS, "カスタム"], value="正方形 · 768×768", label="解像度プリセット", filterable=False)
                with gr.Row():
                    width = gr.Number(value=768, precision=0, minimum=256, maximum=2048, step=16, label="幅 px", min_width=95, elem_id="nanosaur2-width")
                    swap_size = gr.Button("縦横入替", size="sm", scale=0, min_width=70, elem_id="nanosaur2-swap-size")
                    height = gr.Number(value=768, precision=0, minimum=256, maximum=2048, step=16, label="高さ px", min_width=95, elem_id="nanosaur2-height")
                with gr.Accordion("詳細設定", open=False):
                    steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                    cfg = gr.Slider(1, 12, value=4, step=0.1, label="CFG")
                    seed = gr.Textbox(value="-1", label="Seed（-1でランダム）")
                    guidance = gr.Dropdown(
                        choices=["alternate", "cfg", "path_drop"], value="alternate", label="Guidance"
                    )
                    gr.Markdown("作者推奨はEuler / simple、50 steps、CFG 4、alternate guidanceです。")
                with gr.Row():
                    generate = gr.Button("画像を生成", variant="primary")
                    cancel = gr.Button("停止", interactive=False)
                status = gr.HTML("未実行。", elem_id="nanosaur2-status")
            with gr.Column(scale=1, min_width=320):
                result = gr.Image(
                    label="生成画像", type="filepath", format="png", interactive=False, elem_id="nanosaur2-result"
                )
                files = gr.File(label="PNG・生成条件JSON", file_count="multiple", interactive=False)
                gr.Markdown("保存先: `outputs/nanosaur2/`")
                with gr.Accordion("生成条件", open=False):
                    metadata = gr.JSON(label="生成条件")
        job = gr.State({})
        private = {"api_visibility": "private", "show_progress": "hidden"}
        tab.load(_setup_state, outputs=setup_status, **private)
        preset.change(
            lambda value: PRESETS.get(value, (gr.update(), gr.update())),
            inputs=preset,
            outputs=[width, height],
            queue=False,
            **private,
        )
        swap_size.click(lambda w, h: (h, w, "カスタム"), inputs=[width, height], outputs=[width, height, preset], queue=False, **private)
        for control in (width, height):
            control.input(lambda: "カスタム", outputs=preset, queue=False, **private)
        setup.click(
            _setup,
            outputs=setup_status,
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
            trigger_mode="once",
            **private,
        )
        connect.click(
            _connect,
            outputs=setup_status,
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
            trigger_mode="once",
            **private,
        )
        restart.click(
            _restart,
            outputs=setup_status,
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
            trigger_mode="once",
            **private,
        )
        generate.click(
            _generate,
            inputs=[prompt, negative, width, height, steps, cfg, seed, guidance],
            outputs=[status, result, files, metadata, job, generate, cancel],
            concurrency_limit=1,
            concurrency_id="minimax-h3-generation",
            trigger_mode="once",
            **private,
        )
        cancel.click(_cancel, inputs=job, outputs=[status, cancel], queue=False, **private)
    return [(tab, "Nanosaur2", "nanosaur2_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
