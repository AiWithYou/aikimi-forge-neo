"""Add an image-only tab without replacing the existing H3 video controls."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from modules import script_callbacks
from modules.paths import data_path, script_path
from modules_forge import minimax_h3_images as images
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_negpip import H3NegPiP
from modules_forge.minimax_h3_negpip_ui import create_negpip_controls
from modules_forge.minimax_h3_runtime import SERVER_URL, managed_runtime_root

OUTPUT_DIRECTORY = Path(data_path) / "outputs" / "minimax_h3" / "images"
LOG_DIRECTORY = Path(data_path) / "logs" / "minimax_h3"
PRESETS = {
    "動作確認 · 512×512": (512, 512),
    "正方形 · 768×768": (768, 768),
    "横長 · 1344×768": (1344, 768),
    "縦長 · 768×1344": (768, 1344),
    "正方形 · 1024×1024": (1024, 1024),
}


def _initial_runtime() -> str:
    return str(managed_runtime_root(Path(script_path)))


def _preset(value):
    return PRESETS.get(value, (gr.update(), gr.update()))


def _mode(value):
    return gr.update(visible=value == "references")


def _reference_tags(files):
    count = len(images._bridge().normalize_file_list(files))
    if count > 9:
        return "**参照画像は9枚までです。**"
    tags = " / ".join(f"`<Picture {i}>`" for i in range(1, count + 1))
    return f"アップロード順: {tags}" if tags else "参照画像を追加してください。"


def _runtime_action(runtime, url, profile, *negpip_values, restart=False):
    bridge = images._bridge()
    try:
        root = bridge.resolve_runtime_root(runtime)
        with bridge._RUNTIME_LIFECYCLE_LOCK:
            action = bridge.restart_runtime if restart else bridge.ensure_ready
            readiness = action(root, url, LOG_DIRECTORY, runtime_profile=profile, acceleration=H3Acceleration(negpip=H3NegPiP.from_values(negpip_values)))
            client = bridge.ComfyH3Client(url)
            try:
                images.check_image_nodes(client)
            finally:
                client.close()
        return (
            "画像生成の接続準備ができています。GPUでの生成成功を示すものではありません。 "
            f"ComfyUI: {html.escape(readiness.comfy_version or '不明')} / "
            f"GPU: {html.escape(readiness.gpu_name or '不明')}"
        )
    except (bridge.H3BridgeError, images.H3ImageError, OSError, ValueError) as exc:
        return "**確認が必要です:** " + html.escape(str(exc))


def _restart(runtime, url, profile, *negpip_values):
    return _runtime_action(runtime, url, profile, *negpip_values, restart=True)


def _request(mode, prompt, references, width, height, steps, seed, scheduler, frame, candidates, *negpip_values):
    return images.H3ImageRequest(
        mode=mode, prompt=prompt or "",
        reference_images=images._bridge().normalize_file_list(references) if mode == "references" else (),
        width=images.integer(width, "幅", 256, 2048),
        height=images.integer(height, "高さ", 256, 2048),
        steps=images.integer(steps, "Steps", 1, 100),
        seed=images.integer(seed, "Seed", -1, 2**63 - 1),
        scheduler=scheduler,
        frame_index=images.integer(frame, "採用フレーム", 0, 4),
        save_candidates=candidates,
        negpip=H3NegPiP.from_values(negpip_values),
    )


def _generate(runtime, url, profile, mode, prompt, references, width, height, steps, seed, scheduler, frame, candidates, *negpip_values):
    # Keep the submitted URL in server-side State; changing a textbox must not
    # redirect a subsequent cancellation to another ComfyUI process.
    yield (
        "入力を確認しています。", None, [], [], None, {},
        gr.update(interactive=False), gr.update(interactive=False),
    )
    try:
        request = _request(mode, prompt, references, width, height, steps, seed, scheduler, frame, candidates, *negpip_values)
        for event in images.run_image_generation(
            request, Path(runtime), url, LOG_DIRECTORY, OUTPUT_DIRECTORY, runtime_profile=profile,
        ):
            complete = event["stage"] == "complete"
            job = {"prompt_id": event["prompt_id"], "server_url": url} if event.get("prompt_id") and not complete else {}
            message = html.escape(event["message"])
            if "elapsed" in event:
                message += f" 経過 {event['elapsed']:.0f} 秒。"
            yield (
                message, event.get("path", gr.update()),
                [(path, f"候補 {index}") for index, path in enumerate(event["candidates"])] if complete else gr.update(),
                event.get("files", gr.update()), event.get("metadata", gr.update()), job,
                gr.update(interactive=complete), gr.update(interactive=bool(job)),
            )
    except Exception as exc:
        yield (
            "**停止またはエラー:** " + html.escape(str(exc)), gr.update(), gr.update(),
            gr.update(), gr.update(), {}, gr.update(interactive=True), gr.update(interactive=False),
        )


def _cancel(job):
    bridge = images._bridge()
    if not isinstance(job, dict) or not job.get("prompt_id"):
        return "実行中の画像ジョブはありません。", gr.update(interactive=False)
    try:
        bridge.cancel_generation(job["prompt_id"], job["server_url"])
        return "この画像ジョブに停止要求を送りました。", gr.update(interactive=False)
    except (bridge.H3BridgeError, OSError) as exc:
        return "**停止要求を送信できません:** " + html.escape(str(exc)), gr.update(interactive=True)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.Markdown(
            "## MiniMax H3 Image\n"
            "テキスト生成・参照画像編集をPNGで保存します。"
            "H3の最小5フレームから1枚を採用する実験的な静止画モードです。"
            "動画ファイルの生成・音声の復号は行いません。"
        )
        with gr.Accordion("実行環境とモデル", open=False):
            runtime = gr.Textbox(value=_initial_runtime(), visible=False, interactive=False)
            url = gr.Textbox(value=SERVER_URL, visible=False, interactive=False)
            runtime.do_not_save_to_config = True
            url.do_not_save_to_config = True
            profile = gr.Radio(
                choices=[("高速（Pinned + Async 2）", "fast"), ("省RAM", "low_ram")],
                value="fast", label="起動設定（H3動画と共通）",
            )
            gr.Markdown(
                "既存H3 Studioのモデル一式を再利用します。参照編集にはRef2VAが必要です。"
                "この画像タブは標準モデル・通常VAE・Dense Attentionを使用します。"
                "動画側で別の高速化設定を使っている場合は、キューを空にして再起動してください。"
            )
            with gr.Row():
                connect = gr.Button("接続・起動")
                restart = gr.Button("選択設定で再起動")
            runtime_status = gr.Markdown("未確認。生成時にも実行環境を検証します。")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                mode = gr.Radio(
                    choices=[("テキスト → 画像", "text"), ("参照画像から編集", "references")],
                    value="text", label="生成モード",
                )
                prompt = gr.Textbox(
                    label="プロンプト", lines=6,
                    placeholder="完成させたい静止画を記述してください。参照編集例: <Picture 1>の人物を保ち、背景だけを夕暮れの街に変更する。",
                )
                with gr.Group(visible=False) as reference_group:
                    references = gr.File(
                        label="参照画像（アップロード順・最大9枚）", file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".webp"], type="filepath",
                    )
                    tags = gr.Markdown("参照画像を追加してください。")
                    gr.Markdown("プロンプトで全画像のタグと役割を指定してください。画素や人物の完全な保持は保証しません。")
                preset = gr.Dropdown(
                    choices=[*PRESETS, "カスタム"], value="正方形 · 768×768", label="解像度プリセット",
                )
                with gr.Row():
                    width = gr.Number(value=768, precision=0, minimum=256, maximum=2048, step=32, label="幅")
                    height = gr.Number(value=768, precision=0, minimum=256, maximum=2048, step=32, label="高さ")
                with gr.Accordion("詳細設定", open=False):
                    gr.Markdown("32の倍数で指定します。約1MPを超える直接生成は実験的で、負荷が増えます。")
                    steps = gr.Slider(1, 100, value=20, step=1, label="Steps（標準20）")
                    seed = gr.Textbox(value="-1", label="Seed（-1: 毎回ランダム）")
                    scheduler = gr.Dropdown(["simple", "beta", "normal"], value="simple", label="Scheduler")
                    frame = gr.Dropdown(
                        choices=[(f"候補 {i}" + ("（標準）" if i == 0 else ""), i) for i in range(5)],
                        value=0, label="採用フレーム（0〜4）",
                    )
                    candidates = gr.Checkbox(value=False, label="5候補もPNG保存して比較する")
                    gr.Markdown("5候補は同じ時間系列の画像で、独立した5シードのバッチではありません。")
                negpip_controls = create_negpip_controls(prefix="h3-image", image_mode=True)
                with gr.Row():
                    generate = gr.Button("画像を生成", variant="primary")
                    cancel = gr.Button("停止", interactive=False)
                status = gr.Markdown("未実行。")
            with gr.Column(scale=1, min_width=320):
                result = gr.Image(label="生成画像", type="filepath", format="png", interactive=False)
                gallery = gr.Gallery(label="5フレームの候補（候補保存を有効にした場合）", columns=3, height=280, interactive=False)
                files = gr.File(label="PNG・生成条件JSON", file_count="multiple", interactive=False)
                gr.Markdown("保存先: `outputs/minimax_h3/images/`。原画像は上書きしません。")
                with gr.Accordion("生成条件（確定Seedを含む）", open=False):
                    metadata = gr.JSON(label="生成条件")
        job = gr.State({})
        private = {"api_visibility": "private", "show_progress": "hidden"}
        mode.change(_mode, inputs=mode, outputs=reference_group, queue=False, **private)
        references.change(_reference_tags, inputs=references, outputs=tags, queue=False, **private)
        preset.change(_preset, inputs=preset, outputs=[width, height], queue=False, **private)
        for control in (width, height):
            control.input(lambda: "カスタム", outputs=preset, queue=False, **private)
        for button, callback in ((connect, _runtime_action), (restart, _restart)):
            button.click(
                callback, inputs=[runtime, url, profile, *negpip_controls], outputs=runtime_status,
                concurrency_limit=1, concurrency_id="h3-runtime-control", trigger_mode="once", **private,
            )
        generate.click(
            _generate,
            inputs=[runtime, url, profile, mode, prompt, references, width, height, steps, seed, scheduler, frame, candidates, *negpip_controls],
            outputs=[status, result, gallery, files, metadata, job, generate, cancel],
            concurrency_limit=1, concurrency_id="minimax-h3-generation", trigger_mode="once", **private,
        )
        cancel.click(_cancel, inputs=job, outputs=[status, cancel], queue=False, **private)
    return [(tab, "H3 Image", "minimax_h3_image_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
