"""CPU-only outpaint preparation with an explicit external-generation handoff."""

import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from hashlib import sha256

import gradio as gr
from PIL import Image, ImageDraw

from modules import script_callbacks
from modules_forge.qwen_image21.outpaint import Plan, normalize_image, prepare, recipe, stitch

PRIVATE = {"api_visibility": "private", "show_progress": "hidden", "queue": False}
EMPTY = "元画像を選び、広げたい方向の余白を指定してください。"
READY = "作成すると、保存用の参照PNGとComfyUIへの指示が下に表示されます。"
PREVIEW_SIDE = 360
FRAME = (255, 145, 71)
_EXPORT_DIRECTORY = None
_EXPORT_LOCK = threading.Lock()
# Order matches the left/top/right/bottom margin inputs.
DIRECTIONS = {
    "四方": (1, 1, 1, 1),
    "左右": (1, 0, 1, 0),
    "上下": (0, 1, 0, 1),
    "左": (1, 0, 0, 0),
    "右": (0, 0, 1, 0),
    "上": (0, 1, 0, 0),
    "下": (0, 0, 0, 1),
}


@dataclass(frozen=True)
class PreparedCanvas:
    original: Image.Image
    plan: Plan
    token: str
    inputs: tuple
    summary: str


def input_signature(source, left, top, right, bottom, version, scene):
    original = normalize_image(source)
    pixels = (original.mode, original.size, sha256(original.tobytes()).hexdigest())
    # int and integral float inputs from Gradio describe the same request.
    pads = tuple(float(value) for value in (left, top, right, bottom))
    return pixels, pads, version, scene.strip()


def save_png(image, name):
    # A readable, size-bearing file name helps pick the right file in ComfyUI.
    # Reuse identical exports and clean this source cache on process shutdown.
    global _EXPORT_DIRECTORY
    try:
        digest = sha256(image.mode.encode() + str(image.size).encode() + image.tobytes()).hexdigest()
        with _EXPORT_LOCK:
            if _EXPORT_DIRECTORY is None:
                _EXPORT_DIRECTORY = tempfile.TemporaryDirectory(prefix="aikimi-outpaint-")
            directory = os.path.join(_EXPORT_DIRECTORY.name, digest)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{name}.png")
            if not os.path.exists(path):
                image.save(path, format="PNG")
    except OSError as exc:
        raise gr.Error(f"PNGを一時保存できません: {exc}") from exc
    return path


def margin(value):
    """Lenient reading for shortcuts/preview only; prepare() still validates."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return 0
    return int(min(value, 4096))


def direction_of(left, top, right, bottom):
    enabled = tuple(int(margin(value) > 0) for value in (left, top, right, bottom))
    return next((name for name, pattern in DIRECTIONS.items() if pattern == enabled), None)


def apply_direction(direction, left, top, right, bottom):
    if direction not in DIRECTIONS:
        return tuple(gr.update() for _ in range(4))
    current = [margin(value) for value in (left, top, right, bottom)]
    amount = max(current) or 128
    # Keep manually typed amounts on sides that stay enabled.
    return tuple(
        gr.update(value=(value or amount) if enabled else 0)
        for value, enabled in zip(current, DIRECTIONS[direction], strict=True)
    )


def preview_image(original, plan):
    width, height = plan.size
    scale = min(1.0, PREVIEW_SIDE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.new("RGB", size, (128, 128, 128))
    draw = ImageDraw.Draw(image)
    for offset in range(-size[1], size[0], 12):
        draw.line([(offset, size[1]), (offset + size[1], 0)], fill=(158, 158, 158), width=4)
    box = [round(value * scale) for value in plan.box]
    box[2], box[3] = max(box[2], box[0] + 1), max(box[3], box[1] + 1)
    # Transparent source pixels become plain gray, exactly as in the reference PNG.
    draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=(128, 128, 128))
    thumbnail = original.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR, reducing_gap=2.0)
    image.paste(thumbnail, box[:2], thumbnail.getchannel("A") if thumbnail.mode == "RGBA" else None)
    draw.rectangle([box[0] - 1, box[1] - 1, box[2], box[3]], outline=FRAME, width=2)
    return image


def preview_canvas(source, left, top, right, bottom):
    direction = gr.update(value=direction_of(left, top, right, bottom))
    if source is None:
        return gr.update(visible=False), None, "", direction
    try:
        original, _, plan = prepare(source, left, top, right, bottom)
    except (OSError, ValueError) as exc:
        return (
            gr.update(visible=True),
            gr.update(value=None, visible=False),
            f"**このままでは作成できません。** {exc}",
            direction,
        )
    width, height = plan.size
    caption = (
        f"**完成サイズ {width} × {height} px** · 斜線部（左 {plan.left} / 上 {plan.top} / 右 {plan.right} / 下 {plan.bottom} px）"
        "をComfyUIで生成します。橙枠が元画像の範囲です。境界のなじませ方は最後に調整できます。"
    )
    if (plan.left, plan.top, plan.right, plan.bottom) != tuple(margin(value) for value in (left, top, right, bottom)):
        caption += "  \n32 px単位に合わせて余白を調整しています。"
    return gr.update(visible=True), gr.update(value=preview_image(original, plan), visible=True), caption, direction


def handoff_steps(settings):
    comfy = settings["comfyui"]
    width, height = settings["canvas_size"]
    return (
        f"1. [ausboss配布ページ]({settings['source']})のワークフローで `{settings['weights']}` を使う\n"
        f"2. 参照PNGを `{comfy['reference_input']}` に読み込み、resolutionを `{comfy['reference_resolution']}` にする\n"
        "3. 下の指示をプロンプトへ貼り付ける\n"
        f"4. latentは `{comfy['latent_source']}` をそのまま使い、"
        f"{comfy['steps']} steps · CFG {comfy['cfg']:g} · {comfy['sampler']} / {comfy['scheduler']} · "
        f"denoise {comfy['denoise']:g}。Set Latent Noise Maskは使わない\n"
        f"5. **{width} × {height} px** で出力された画像を保存し、[3 · 完成画像](#qwen21-outpaint-restore)へ"
    )


def prepare_canvas(source, left, top, right, bottom, version, scene):
    try:
        original, canvas, plan = prepare(source, left, top, right, bottom)
        settings = recipe(plan, version, scene)
        signature = input_signature(original, left, top, right, bottom, version, scene)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    summary = f"**参照PNGを作成しました · {plan.size[0]} × {plan.size[1]} px · LoRA {version}**"
    if settings["warnings"][1:]:
        summary += "\n\n" + "\n\n".join(settings["warnings"][1:])
    summary += "  \n[2 · ComfyUIで生成 ↓](#qwen21-outpaint-handoff) · [3 · 完成画像 ↓](#qwen21-outpaint-restore)"
    token = sha256(json.dumps(signature, ensure_ascii=False).encode("utf-8")).hexdigest()
    return PreparedCanvas(original, plan, token, signature, summary), canvas, settings["prompt"], settings, summary


def draft_changed(source, left, top, right, bottom, version, scene, state, binding, generated):
    changed = bool(state)
    if state:
        try:
            changed = input_signature(source, left, top, right, bottom, version, scene) != state.inputs
        except (OSError, TypeError, ValueError):
            changed = True
    message = (
        "**未反映の変更があります。** 下の参照PNG・指示・画像は前回の作成時のままです。"
        "「参照PNGを作成」で更新すると、生成画像の選び直しが必要です。"
        if changed
        else state.summary
        if state
        else READY
        if source is not None
        else EMPTY
    )
    valid_binding = bool(
        state and not changed and binding == state.token and generated_status(state, generated, False)[0] == state.token
    )
    validation = (
        "未反映の変更があります。[参照PNGの作成へ戻る ↑](#qwen21-outpaint-source)。"
        "入力を作成時の値に戻すと、前回の生成画像のまま完成画像を作れます。"
        if changed
        else generated_status(state, generated, False)[1]
        if valid_binding
        else "今回の参照から生成した画像を選んでください。"
    )
    return (
        changed,
        message,
        gr.update(interactive=source is not None),
        gr.update(interactive=valid_binding),
        gr.update(label="前回の完成画像（変更は未反映）" if changed else "完成画像"),
        validation,
        gr.update(label="前回の参照PNG（変更は未反映）" if changed else "参照PNG"),
        gr.update(label="前回の指示（変更は未反映）" if changed else "プロンプトへ貼り付ける指示"),
        gr.update(label="前回の参照PNGを保存" if changed else "参照PNGを保存"),
        gr.update(label="前回の完成画像を保存" if changed else "完成画像を保存"),
    )


def generated_status(state, generated, dirty):
    if not state:
        return None, "先に参照PNGを作成してください。", gr.update(interactive=False)
    width, height = state.plan.size
    if dirty:
        return None, "未反映の変更があります。参照PNGを作成し直してください。", gr.update(interactive=False)
    if generated is None:
        return (
            None,
            f"今回の参照からComfyUIで生成した **{width} × {height} px** の画像を選んでください。",
            gr.update(interactive=False),
        )
    try:
        image = normalize_image(generated)
    except (OSError, ValueError) as exc:
        return None, str(exc), gr.update(interactive=False)
    if image.size != state.plan.size:
        return (
            None,
            (
                f"**サイズが違います: {image.width} × {image.height} px**。"
                f"必要なサイズは **{width} × {height} px** です。ComfyUIの出力サイズを合わせて生成し直してください。"
            ),
            gr.update(interactive=False),
        )
    return (
        state.token,
        (f"**サイズ一致 · {width} × {height} px**。今回の参照から生成した画像なら「完成画像を作成」を押してください。"),
        gr.update(interactive=True),
    )


def restore_original(state, generated, feather, binding, dirty):
    if not state:
        raise gr.Error("先に参照PNGを作成してください。")
    if dirty:
        raise gr.Error("未反映の変更があります。参照PNGを作成し直してください。")
    if binding != state.token:
        raise gr.Error("参照が新しくなりました。今回の参照から生成した画像を選び直してください。")
    try:
        return stitch(state.original, generated, state.plan, feather)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc


def finish_for_ui(state, generated, feather, binding, dirty):
    result = restore_original(state, generated, feather, binding, dirty)
    path = save_png(result, f"qwen-outpaint-{result.width}x{result.height}")
    return result, gr.update(value=path, interactive=True, label="完成画像を保存")


def cleared_result():
    return gr.update(value=None, label="完成画像"), gr.update(value=None, interactive=False, label="完成画像を保存")


def prepare_for_ui(source, left, top, right, bottom, version, scene, generated, previous=None, binding=None):
    state, canvas, prompt, settings, summary = prepare_canvas(source, left, top, right, bottom, version, scene)
    maximum = min(128, (min(state.plan.source_width, state.plan.source_height) - 1) // 2)
    message = generated_status(state, None, False)[1]
    valid_binding = bool(binding == state.token and generated_status(state, generated, False)[0] == state.token)
    if valid_binding:
        _, message, _ = generated_status(state, generated, False)
    elif generated is not None:
        message = (
            "**参照を更新しました。** 以前の生成画像は残しています。今回の参照から生成した画像を選び直してください。"
        )
    feather = gr.update(maximum=maximum)
    if not previous or previous.token != state.token:
        feather["value"] = min(32, maximum)
    width, height = state.plan.size
    reference_file = save_png(canvas, f"qwen-outpaint-reference-{width}x{height}")
    restored, final_download = cleared_result()
    return (
        state,
        gr.update(value=canvas, label="参照PNG"),
        gr.update(value=prompt, label="プロンプトへ貼り付ける指示"),
        settings,
        summary,
        False,
        binding,
        gr.update(visible=True),
        gr.update(visible=True),
        feather,
        message,
        gr.update(interactive=valid_binding),
        restored,
        gr.update(value=reference_file, interactive=True, label="参照PNGを保存"),
        handoff_steps(settings),
        final_download,
    )


def uploaded(state, generated, dirty):
    binding, message, button = generated_status(state, generated, dirty)
    return binding, message, button, *cleared_result()


def feather_changed():
    # Recomputing this CPU-only result is cheap; never offer a stale download.
    return cleared_result()


def on_ui_tabs():
    with gr.Blocks() as tab:
        state = gr.State(None)
        dirty = gr.State(False)
        binding = gr.State(None)
        with gr.Column(elem_id="qwen21-outpaint"):
            gr.Markdown(
                "## Qwen Outpaint\n"
                "余白付きの参照PNGを作り、**ComfyUI（ausboss Outpaint LoRA）で別途生成**した画像から、"
                "元画像の画素を保った完成画像を作ります。このタブでは画像生成を行いません。"
            )
            gr.Markdown("### 1 · 参照PNGを作る")
            with gr.Row(elem_classes="qwen21-outpaint-stage"):
                with gr.Column(min_width=260):
                    source = gr.Image(
                        label="元画像",
                        type="pil",
                        image_mode=None,
                        format="png",
                        height=260,
                        sources=["upload", "clipboard"],
                        elem_id="qwen21-outpaint-source",
                    )
                with gr.Column(
                    min_width=260, visible=False, elem_id="qwen21-outpaint-preview-column"
                ) as preview_column:
                    preview = gr.Image(
                        label="仕上がりの範囲",
                        type="pil",
                        format="png",
                        interactive=False,
                        height=260,
                        buttons=[],
                        elem_id="qwen21-outpaint-preview",
                    )
                    preview_caption = gr.Markdown(elem_id="qwen21-outpaint-preview-caption")
                with gr.Column(min_width=260):
                    direction = gr.Radio(
                        choices=list(DIRECTIONS),
                        value="四方",
                        label="広げる方向",
                        elem_id="qwen21-outpaint-direction",
                    )
                    with gr.Row(elem_id="qwen21-outpaint-margins"):
                        left = gr.Number(value=128, minimum=0, maximum=2048, step=1, label="左の余白 px", min_width=70)
                        right = gr.Number(value=128, minimum=0, maximum=2048, step=1, label="右の余白 px", min_width=70)
                        top = gr.Number(value=128, minimum=0, maximum=2048, step=1, label="上の余白 px", min_width=70)
                        bottom = gr.Number(
                            value=128, minimum=0, maximum=2048, step=1, label="下の余白 px", min_width=70
                        )
                    version = gr.Radio(
                        choices=[("v1 · 目安 約1 MP", "v1"), ("v2 · 目安 1〜2 MP", "v2")],
                        value="v2",
                        label="ComfyUIで使うOutpaint LoRA",
                        elem_id="qwen21-outpaint-version",
                    )
                    scene = gr.Textbox(
                        label="広げる場面の説明（任意）",
                        lines=2,
                        max_lines=4,
                        max_length=10000,
                        placeholder="例: Extend the forest path into the distance.",
                        elem_id="qwen21-outpaint-scene",
                    )
                    prepare_button = gr.Button(
                        "参照PNGを作成", variant="primary", interactive=False, elem_id="qwen21-outpaint-prepare"
                    )
                    summary = gr.Markdown(EMPTY, elem_id="qwen21-outpaint-summary")
            with gr.Column(visible=False, elem_id="qwen21-outpaint-handoff") as handoff:
                gr.Markdown("### 2 · ComfyUIで生成（このタブの外）")
                with gr.Row(elem_classes="qwen21-outpaint-stage"):
                    with gr.Column(min_width=280):
                        padded = gr.Image(
                            label="参照PNG",
                            type="pil",
                            format="png",
                            interactive=False,
                            height=260,
                            buttons=["fullscreen"],
                            elem_id="qwen21-outpaint-padded",
                        )
                        reference_download = gr.DownloadButton(
                            "参照PNGを保存",
                            interactive=False,
                            elem_id="qwen21-outpaint-reference-download",
                        )
                    with gr.Column(min_width=280):
                        steps = gr.Markdown(elem_id="qwen21-outpaint-steps")
                        prompt = gr.Textbox(
                            label="プロンプトへ貼り付ける指示",
                            lines=3,
                            interactive=False,
                            buttons=["copy"],
                            elem_id="qwen21-outpaint-prompt",
                        )
                        with gr.Accordion("設定メモ・互換性", open=False):
                            settings = gr.JSON(label="生成設定とキャンバス位置")
                            gr.Markdown(
                                "このJSONは設定メモです。ComfyUIワークフローではありません。"
                                "この補助はLoRAを読み込まず、ForgeのINT8・GGUFローダーとの互換性も未検証です。"
                            )
            with gr.Column(visible=False, elem_id="qwen21-outpaint-restore") as restore_section:
                gr.Markdown("### 3 · 完成画像を作る")
                with gr.Row(elem_classes="qwen21-outpaint-stage"):
                    with gr.Column(min_width=280):
                        generated = gr.Image(
                            label="ComfyUIで生成した画像",
                            type="pil",
                            image_mode=None,
                            format="png",
                            sources=["upload", "clipboard"],
                            height=260,
                            elem_id="qwen21-outpaint-generated",
                        )
                        validation = gr.Markdown(elem_id="qwen21-outpaint-validation")
                        feather = gr.Slider(
                            0,
                            128,
                            value=32,
                            step=1,
                            label="元画像との境界をなじませる幅 px",
                            info="0なら元画像を全画素そのまま保持。増やすと元画像の外周をこの幅だけ生成画像と混ぜます。",
                            elem_id="qwen21-outpaint-feather",
                        )
                        restore_button = gr.Button(
                            "完成画像を作成", variant="primary", interactive=False, elem_id="qwen21-outpaint-stitch"
                        )
                    with gr.Column(min_width=280):
                        restored = gr.Image(
                            label="完成画像",
                            type="pil",
                            format="png",
                            interactive=False,
                            height=260,
                            buttons=["fullscreen"],
                            elem_id="qwen21-outpaint-restored",
                        )
                        final_download = gr.DownloadButton(
                            "完成画像を保存",
                            interactive=False,
                            elem_id="qwen21-outpaint-final-download",
                        )
        prepare_inputs = [source, left, top, right, bottom, version, scene]
        prepare_button.click(
            prepare_for_ui,
            inputs=[*prepare_inputs, generated, state, binding],
            outputs=[
                state,
                padded,
                prompt,
                settings,
                summary,
                dirty,
                binding,
                handoff,
                restore_section,
                feather,
                validation,
                restore_button,
                restored,
                reference_download,
                steps,
                final_download,
            ],
            **PRIVATE,
        )
        restore_button.click(
            finish_for_ui,
            inputs=[state, generated, feather, binding, dirty],
            outputs=[restored, final_download],
            **PRIVATE,
        )
        for component in prepare_inputs:
            component.change(
                draft_changed,
                inputs=[*prepare_inputs, state, binding, generated],
                outputs=[
                    dirty,
                    summary,
                    prepare_button,
                    restore_button,
                    restored,
                    validation,
                    padded,
                    prompt,
                    reference_download,
                    final_download,
                ],
                **PRIVATE,
            )
        # Forge wraps event methods for compatibility; gr.on() cannot consume
        # those wrappers. Bind through each component's supported event method.
        for component in (source, left, top, right, bottom):
            component.change(
                preview_canvas,
                inputs=[source, left, top, right, bottom],
                outputs=[preview_column, preview, preview_caption, direction],
                **PRIVATE,
            )
        direction.input(
            apply_direction, inputs=[direction, left, top, right, bottom], outputs=[left, top, right, bottom], **PRIVATE
        )
        generated.change(
            uploaded,
            inputs=[state, generated, dirty],
            outputs=[binding, validation, restore_button, restored, final_download],
            **PRIVATE,
        )
        feather.change(feather_changed, inputs=[], outputs=[restored, final_download], **PRIVATE)
    return [(tab, "Qwen Outpaint 補助", "qwen_image21_outpaint")]


script_callbacks.on_ui_tabs(on_ui_tabs)
