"""Ming Image Studio controls and authenticated job API."""

from functools import partial

import gradio as gr

from modules import script_callbacks
from modules.aikimi_status import studio_status_html
from modules_forge import gpu_fault
from modules_forge.image_workspace import PRIVATE, ImageWorkspace
from modules_forge.ming_image import checkpoints, core
from modules_forge.ming_image.service import SERVICE

PROFILES = list(
    zip(
        (
            "24GB 高画質 (Aikimi HQ)",
            "16GB 省メモリ高画質 (Aikimi HQ)",
            "Comfy-Org 標準 (W4A8)",
            "Comfy-Org 標準 (INT8)",
        ),
        core.PROFILES,
        strict=True,
    )
)
PRESETS = ((2048, 2048), (2560, 1440), (2432, 1824), (1664, 2496), (1024, 1024))
TEMPLATES = {
    "UI/UX モック": "Design a booking app. Show the date picker, room details and a clearly labeled Reserve button.",
    "ポスター": "Design an exhibition poster titled PAPER CITY. Use folded paper buildings, deep blue ink and a clear date line.",
    "EC 商品ヒーロー": "Photograph a glass coffee brewer on a pale stone counter. Leave the left third free for a headline.",
    "透過アセット": "A single hand-painted fern leaf, isolated on a transparent background with a clean alpha edge.",
    "Figma 風 JSON": '{"type":"FRAME","name":"Checkout","layoutMode":"VERTICAL","children":[{"type":"TEXT","characters":"Order summary"}]}',
}


def readiness(profile):
    if core.manifest()["profiles"][profile]["status"] != "published":
        return "HQモデルの公開待ちです。Comfy-Org 標準を選択できます。"
    return "準備完了" if core.model_ready(request={"profile": profile}) else "モデルの準備が必要です。"


def prepare(profile):
    from tools.setup_ming_image import run

    try:
        run(profile=profile)
    except ValueError as exc:
        return str(exc)
    return readiness(profile)


def start(mode, images, prompt, negative, profile, dit, text, transparent, width, height, steps, cfg, shift, seed):
    refs = [item[0] for item in (images or [])] if mode == "img2img" else []
    try:
        job = SERVICE.start(
            dict(
                mode=mode,
                images=refs,
                prompt=prompt,
                negative_prompt=negative,
                profile=profile,
                checkpoint=dit,
                text_checkpoint=text,
                transparent=transparent,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                shift=shift,
                seed=seed,
            )
        )
    except (ValueError, gpu_fault.GPUFaultError) as exc:
        return gr.skip(), gr.skip(), status_html("", "error", str(exc))
    return job, gr.Timer(active=True), status_html(job, "running", "生成中…")


def status_html(job, stage, message, progress=None):
    return studio_status_html(
        "ming_image",
        stage,
        message,
        job_id=job or "",
        progress=progress,
        visible=True,
        result_id="ming-image-preview" if stage == "complete" else "",
    )


def poll(job):
    record = SERVICE.poll(job)
    state = record["status"]
    if state == "complete":
        return record["image_path"], status_html(job, state, "完了"), gr.Timer(active=False)
    if state in ("error", "cancelled", "idle"):
        message = record.get("error") or {"cancelled": "中断しました。", "idle": ""}.get(state, "生成に失敗しました。")
        return gr.skip(), status_html(job, state, message), gr.Timer(active=False)
    if state == "stopping":
        return gr.skip(), status_html(job, "running", "中断処理中…"), gr.skip()
    current, total = record.get("current", 0), record.get("total", 1)
    message = f"{record.get('stage', '準備中')} {current}/{total}"
    return gr.skip(), status_html(job, "running", message, current / total if total else None), gr.skip()


def cancel_job(job):
    SERVICE.cancel(job)
    return poll(job)[1]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as blocks:
        with gr.Column() as models:
            profile = gr.Dropdown(PROFILES, value=core.PROFILES[0], label="モデルプロファイル")
            dit = gr.Dropdown(checkpoints.inventory(core.MODEL_ROOT, "dit"), value="official", label="DiT")
            text = gr.Dropdown(
                checkpoints.inventory(core.MODEL_ROOT, "text"), value="official", label="テキストエンコーダー"
            )
            refresh = gr.Button("モデル一覧を更新")
            setup = gr.Button("モデルを準備")
            ready = gr.Markdown(readiness(core.PROFILES[0]))
        prompt = gr.Textbox(label="プロンプト", lines=4, elem_id="ming-image-prompt")
        negative = gr.Textbox(
            label="負のプロンプト（CFG > 1）", interactive=False, elem_id="ming-image-negative-prompt"
        )
        with gr.Column() as settings:
            transparent = gr.Checkbox(label="透過背景 (RGBA)")
            with gr.Row():
                width = gr.Number(value=2048, precision=0, label="幅")
                height = gr.Number(value=2048, precision=0, label="高さ")
            with gr.Row():
                for w, h in PRESETS:
                    gr.Button(f"{w}×{h}", size="sm").click(partial(tuple, (w, h)), outputs=[width, height], **PRIVATE)
            with gr.Row():
                for label, template in TEMPLATES.items():
                    gr.Button(label, size="sm").click(partial(str, template), outputs=prompt, **PRIVATE)
            steps = gr.Slider(1, 100, value=12, step=1, precision=0, label="Steps")
            cfg = gr.Slider(1, 20, value=1, step=0.1, label="CFG")
            shift = gr.Number(value=1.15, label="Max shift")
            seed = gr.Number(value=-1, precision=0, label="Seed")
        images = gr.Gallery(
            type="pil",
            elem_id="ming-image-references-gallery",
            label="参照画像（表示順に最大8枚）",
            interactive=True,
            format="png",
        )
        generate = gr.Button("生成", variant="primary")
        cancel = gr.Button("中断")
        with gr.Column() as output:
            image = gr.Image(
                label="生成結果",
                type="pil",
                image_mode=None,
                interactive=False,
                format="png",
                elem_id="ming-image-preview",
            )
            status = gr.HTML()
            send = gr.Button("img2img へ送る")
        workspace = ImageWorkspace(
            blocks,
            "ming-image",
            editable=True,
            models=[models],
            prompts=[prompt, negative],
            actions=[generate, cancel],
            settings=[settings],
            references=[images],
            outputs=[output],
        )
        job = gr.State(None)
        timer = gr.Timer(0.5, active=False)
        generate.click(
            start,
            [
                workspace.mode,
                images,
                prompt,
                negative,
                profile,
                dit,
                text,
                transparent,
                width,
                height,
                steps,
                cfg,
                shift,
                seed,
            ],
            [job, timer, status],
            api_visibility="private",
        )
        timer.tick(poll, job, [image, status, timer], **PRIVATE)
        cancel.click(cancel_job, job, status, **PRIVATE)

        def send_result(value):
            if value is None:
                raise gr.Error("生成結果がありません。")
            return [value], *workspace.select_mode("img2img")

        send.click(send_result, image, [images, workspace.mode, workspace.tabs, workspace.references], **PRIVATE)
        profile.change(readiness, profile, ready, **PRIVATE)
        setup.click(prepare, profile, ready, api_visibility="private")
        refresh.click(
            lambda: tuple(
                gr.update(choices=checkpoints.inventory(core.MODEL_ROOT, role), value="official")
                for role in ("dit", "text")
            ),
            outputs=[dit, text],
            **PRIVATE,
        )
        cfg.change(lambda value: gr.update(interactive=value > 1), cfg, negative, **PRIVATE)
    return [(blocks, "Ming Image", "ming_image_studio")]


def app_started(_blocks, app):
    from modules_forge.ming_image.api import register_api

    register_api(app)


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(app_started)
