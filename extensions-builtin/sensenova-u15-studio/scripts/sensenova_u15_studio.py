import html
import os
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from modules import script_callbacks
from modules.paths import data_path
from modules_forge.sensenova_u15_bridge import (
    AUTO_RESOLUTION_PIXELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MODEL_ID,
    DEFAULT_SOURCE_PATH,
    MAX_REFERENCE_IMAGES,
    MODE_EDIT,
    MODE_TEXT,
    PROFILE_OFFICIAL_8STEP,
    PROFILE_QUALITY,
    QUANT_INT8_CONVROT,
    SenseNovaBridgeError,
    SenseNovaGenerationCancelled,
    SenseNovaRequest,
    cancel_generation,
    inspect_runtime,
    normalize_gallery_images,
    parse_resolution,
    progress_html,
    reference_order_html,
    request_summary_html,
    resolution_choices,
    run_generation,
    runtime_status_html,
    validate_request,
)

OUTPUT_DIRECTORY = Path(data_path) / "outputs" / "sensenova_u15"
CACHE_DIRECTORY = Path(data_path) / "cache" / "sensenova_u15"
LOG_DIRECTORY = Path(data_path) / "logs" / "sensenova_u15"
PROMPT_DRAFT_KEY = "forge-neo:sensenova-u15:prompt-draft:v1"


def _gallery_list(value: Any) -> list[Any]:
    return list(value or [])


def _reference_update(values: list[Any], selected: int):
    labeled = []
    for number, item in enumerate(values, 1):
        image = item[0] if isinstance(item, (tuple, list)) else item
        size = f" · {image.width}×{image.height}" if isinstance(image, Image.Image) else ""
        labeled.append((image, f"Image-{number}{size}"))
    return gr.update(value=labeled, selected_index=selected if selected >= 0 else None, preview=False)


def _selection_controls(gallery: Any, index: int):
    values = _gallery_list(gallery)
    valid = 0 <= index < len(values)
    return (
        gr.update(interactive=valid),
        gr.update(interactive=valid and index > 0),
        gr.update(interactive=valid and index < len(values) - 1),
        gr.update(interactive=valid),
        gr.update(interactive=bool(values)),
        reference_order_html(values, index if valid else -1),
    )


def _append_reference(gallery: Any, upload: Image.Image | None):
    values = _gallery_list(gallery)
    if upload is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    if len(values) >= MAX_REFERENCE_IMAGES:
        gr.Warning(f"参照画像は最大{MAX_REFERENCE_IMAGES}枚です。追加する前に不要な参照を削除してください。")
        return gr.update(), gr.update(), gr.update(), gr.update()
    values.append((normalize_gallery_images([upload])[0], None))
    message = reference_order_html(values)
    return _reference_update(values, len(values) - 1), gr.update(value=None), message, len(values) - 1


def _append_reference_files(gallery: Any, uploads: Any):
    values = _gallery_list(gallery)
    paths = list(uploads or [])
    if not paths:
        return gr.update(), gr.update(), gr.update(), gr.update()
    capacity = max(0, MAX_REFERENCE_IMAGES - len(values))
    if not capacity:
        gr.Warning(f"参照画像は最大{MAX_REFERENCE_IMAGES}枚です。選択ファイルは保持しています。")
        return gr.update(), gr.update(), gr.update(), gr.update()
    images = normalize_gallery_images(paths[:capacity])
    values.extend((image, None) for image in images)
    truncated = len(paths) > capacity
    message = reference_order_html(values)
    if truncated:
        notice = f"{len(images)}枚を追加しました。残り{len(paths) - capacity}枚は未追加です。参照画像は最大{MAX_REFERENCE_IMAGES}枚です。"
        gr.Warning(notice)
        message = f'<p class="sn-inline-error" role="alert">{notice}</p>' + message
    return (
        _reference_update(values, len(values) - 1),
        gr.update(value=paths[capacity:] if truncated else None),
        message,
        len(values) - 1,
    )


def _limit_reference_gallery(gallery: Any):
    values = _gallery_list(gallery)
    truncated = len(values) > MAX_REFERENCE_IMAGES
    values = values[:MAX_REFERENCE_IMAGES]
    message = reference_order_html(values)
    if truncated:
        message = (
            f'<p class="sn-inline-error" role="alert">先頭{MAX_REFERENCE_IMAGES}枚を残しました。参照画像は最大{MAX_REFERENCE_IMAGES}枚です。</p>'
            + message
        )
    index = len(values) - 1
    return _reference_update(values, index), message, index


def _replace_reference(index: int, gallery: Any, upload: Image.Image | None):
    values = _gallery_list(gallery)
    if upload is None or index < 0 or index >= len(values):
        return gr.update(), gr.update(), reference_order_html(values), index
    values[index] = (normalize_gallery_images([upload])[0], None)
    return (
        _reference_update(values, index),
        gr.update(value=None),
        reference_order_html(values),
        index,
    )


def _remove_reference(index: int, gallery: Any):
    values = _gallery_list(gallery)
    if 0 <= index < len(values):
        values.pop(index)
    next_index = min(index, len(values) - 1) if values else -1
    return _reference_update(values, next_index), reference_order_html(values), next_index


def _move_reference(index: int, gallery: Any, offset: int):
    values = _gallery_list(gallery)
    target = index + offset
    if 0 <= index < len(values) and 0 <= target < len(values):
        values[index], values[target] = values[target], values[index]
        index = target
    return _reference_update(values, index), reference_order_html(values), index


def _clear_references():
    return _reference_update([], -1), reference_order_html([]), -1


def _after_gallery_delete(gallery: Any, evt: gr.EventData):
    values = _gallery_list(gallery)
    # Gallery.deleteは削除前の値と、削除対象のindexを送る。
    values.pop(int(evt.index))
    return _reference_update(values, -1), reference_order_html(values), -1


def _select_reference(evt: gr.SelectData = None) -> int:
    return int(evt.index) if evt is not None and evt.selected else -1


REFERENCE_TASKS = {
    "identity": (
        "被写体を保って編集",
        1,
        "Use Image-1 as the primary reference. Preserve the subject's identity, facial features, proportions and composition. Apply only the requested changes.",
    ),
    "style": (
        "Image-2の画風・色調を使う",
        2,
        "Preserve the subject and composition of Image-1. Use Image-2 only as a reference for art style, color palette and lighting; do not copy its subject.",
    ),
    "outfit": (
        "Image-2の衣装を着せる",
        2,
        "Preserve the person, face, hair, pose and background of Image-1. Replace only their outfit with the clothing design from Image-2.",
    ),
    "background": (
        "Image-2の背景へ移す",
        2,
        "Preserve the subject's identity, face, hair and outfit from Image-1. Place that subject in the environment from Image-2, matching its perspective and lighting.",
    ),
}


def _append_reference_instruction(prompt: str, task: str, gallery: Any):
    _, required, instruction = REFERENCE_TASKS[task]
    if len(_gallery_list(gallery)) < required:
        gr.Warning(f"この用途には参照画像が{required}枚必要です。")
        return gr.update()
    current = str(prompt or "").strip()
    if instruction in current:
        return gr.update()
    return (current + "\n\n" + instruction).strip()


def _reuse_result(result: str | None, gallery: Any):
    if not result:
        gr.Warning("先に画像を生成してください。")
        return gr.update(), gr.update(), gr.update(), gr.update()
    image = normalize_gallery_images([result])[0]
    values = _gallery_list(gallery)
    if values:
        values[0] = (image, None)
    else:
        values.append((image, None))
    return _reference_update(values, 0), reference_order_html(values, 0), 0, MODE_EDIT


def _mode_updates(mode: str, current_resolution: str, fast_available: bool):
    choices = resolution_choices(mode)
    values = {value for _, value in choices}
    if mode == MODE_EDIT:
        selected = "auto"
    else:
        selected = (
            current_resolution
            if current_resolution in values and current_resolution != "auto"
            else "2048x2048"
        )
    text_profile = (
        PROFILE_OFFICIAL_8STEP if fast_available else PROFILE_QUALITY
    )
    return (
        gr.update(visible=mode == MODE_EDIT),
        gr.update(choices=choices, value=selected),
        gr.update(visible=mode == MODE_EDIT),
        gr.update(
            visible=mode == MODE_EDIT,
            value=str(512 * 512) if mode == MODE_EDIT else "auto",
        ),
        gr.update(
            value=(
                f'<div class="sn-mode-note"><b>MULTI-IMAGE EDIT</b><span>モデル上限は{MAX_REFERENCE_IMAGES}枚。参照優先モードでは最大8枚を合成できます。各画像の役割を番号で指定してください。入力の精細さは参照解像度で調整できます。</span></div>'
                if mode == MODE_EDIT
                else (
                    '<div class="sn-mode-note"><b>TEXT TO IMAGE</b><span>参照画像を使わず、公式8-Step高速プリセットまたは50-Step品質プリセットで生成します。</span></div>'
                    if fast_available
                    else '<div class="sn-mode-note"><b>TEXT TO IMAGE</b><span>公式8-Step LoRAが未準備のため、Quality 50-Stepを選択しています。</span></div>'
                )
            )
        ),
        gr.update(
            visible=mode == MODE_TEXT,
            value=(
                text_profile if mode == MODE_TEXT else PROFILE_QUALITY
            ),
        ),
        gr.update(
            value=8 if mode == MODE_TEXT and fast_available else 50,
            interactive=mode != MODE_TEXT or not fast_available,
        ),
        gr.update(
            value=1.0 if mode == MODE_TEXT and fast_available else 4.0,
            interactive=mode != MODE_TEXT or not fast_available,
        ),
        gr.update(
            value=3.0,
            interactive=mode != MODE_TEXT or not fast_available,
        ),
    )


def _profile_updates(profile: str):
    if profile == PROFILE_OFFICIAL_8STEP:
        return (
            gr.update(value=8, interactive=False),
            gr.update(value=1.0, interactive=False),
            gr.update(value=3.0, interactive=False),
        )
    if profile == PROFILE_QUALITY:
        return (
            gr.update(value=50, interactive=True),
            gr.update(value=4.0, interactive=True),
            gr.update(value=3.0, interactive=True),
        )
    raise SenseNovaBridgeError(f"未対応の生成プロファイルです: {profile!r}")


def _refresh_runtime(source_path: str, checkpoint_path: str, mode: str, current_profile: str):
    try:
        status = inspect_runtime(source_path, checkpoint=checkpoint_path)
        profile = PROFILE_QUALITY if mode == MODE_EDIT or not status.lora_ready else current_profile
        profile_update = gr.update()
        steps, cfg, shift = gr.update(), gr.update(), gr.update()
        if profile != current_profile:
            profile_update = gr.update(value=profile)
            steps, cfg, shift = _profile_updates(profile)
        return (
            runtime_status_html(status),
            status.lora_ready,
            profile_update,
            steps,
            cfg,
            shift,
        )
    except (OSError, ValueError, SenseNovaBridgeError) as exc:
        return (
            (
                '<section class="sn-runtime sn-runtime-setup" role="alert">'
                f"<strong>準備状況を確認できません</strong><p>{html.escape(str(exc))}</p></section>"
            ),
            False,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _request_from_ui(
    mode: str,
    prompt: str,
    gallery: Any,
    model_path: str,
    quantization: str,
    checkpoint_path: str,
    source_path: str,
    resolution: str,
    input_max_pixels: str,
    generation_profile: str,
    steps: int,
    cfg_scale: float,
    img_cfg_scale: float,
    timestep_shift: float,
    seed: int,
    vram_mode: str,
    attn_backend: str,
    dtype: str,
    *,
    should_validate: bool = True,
) -> SenseNovaRequest:
    width, height = parse_resolution(resolution, mode)
    images = normalize_gallery_images(gallery) if mode == MODE_EDIT else ()
    request = SenseNovaRequest(
        mode=mode,
        prompt=str(prompt or ""),
        model_path=str(model_path or ""),
        quantization=quantization,
        checkpoint=str(checkpoint_path or ""),
        source_path=str(source_path or ""),
        input_images=images,
        width=width,
        height=height,
        target_pixels=AUTO_RESOLUTION_PIXELS[resolution] if resolution in AUTO_RESOLUTION_PIXELS else 2048 * 2048,
        input_max_pixels=input_max_pixels,
        generation_profile=generation_profile,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        img_cfg_scale=float(img_cfg_scale),
        timestep_shift=float(timestep_shift),
        seed=int(seed),
        vram_mode=vram_mode,
        attn_backend=attn_backend,
        dtype=dtype,
    )
    if should_validate:
        validate_request(request)
    return request


def _summary_from_ui(
    mode,
    prompt,
    gallery,
    model_path,
    quantization,
    checkpoint_path,
    source_path,
    resolution,
    input_max_pixels,
    generation_profile,
    steps,
    cfg_scale,
    img_cfg_scale,
    timestep_shift,
    seed,
    vram_mode,
    attn_backend,
    dtype,
):
    try:
        request = _request_from_ui(
            mode,
            prompt,
            gallery,
            model_path,
            quantization,
            checkpoint_path,
            source_path,
            resolution,
            input_max_pixels,
            generation_profile,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
            should_validate=False,
        )
        try:
            validate_request(request)
            message = ""
        except (SenseNovaBridgeError, ValueError, TypeError, OverflowError) as exc:
            message = (
                f'<p class="sn-inline-error" role="alert">{html.escape(str(exc))}</p>'
            )
        return request_summary_html(request), message
    except (SenseNovaBridgeError, ValueError, TypeError, OverflowError) as exc:
        return (
            gr.update(),
            f'<p class="sn-inline-error" role="alert">{html.escape(str(exc))}</p>',
        )


def _generate(
    mode,
    prompt,
    gallery,
    model_path,
    quantization,
    checkpoint_path,
    source_path,
    resolution,
    input_max_pixels,
    generation_profile,
    steps,
    cfg_scale,
    img_cfg_scale,
    timestep_shift,
    seed,
    vram_mode,
    attn_backend,
    dtype,
):
    yield (
        progress_html("prepare", "生成条件を確認しています", 0.01),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        "",
        gr.update(interactive=False),
        gr.update(value="生成を準備中…", interactive=False),
        "",
    )
    try:
        request = _request_from_ui(
            mode,
            prompt,
            gallery,
            model_path,
            quantization,
            checkpoint_path,
            source_path,
            resolution,
            input_max_pixels,
            generation_profile,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
        )
        for update in run_generation(
            request,
            output_directory=OUTPUT_DIRECTORY,
            cache_directory=CACHE_DIRECTORY,
            log_directory=LOG_DIRECTORY,
        ):
            completed = update["stage"] == "complete"
            yield (
                progress_html(
                    update["stage"],
                    update["message"],
                    update.get("progress", 0.0),
                    update.get("elapsed"),
                ),
                update.get("path") or gr.update(),
                update.get("path") or gr.update(),
                update.get("metadata_path") or gr.update(),
                update.get("metadata") or gr.update(),
                "" if completed else update.get("job_id", ""),
                gr.update(interactive=bool(update.get("job_id")) and not completed),
                gr.update(
                    value="画像を生成" if completed else "生成中…",
                    interactive=completed,
                ),
                "",
            )
    except SenseNovaGenerationCancelled as exc:
        yield (
            progress_html("cancelled", str(exc), 0.0),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
            gr.update(interactive=False),
            gr.update(value="画像を生成", interactive=True),
            "",
        )
    except (SenseNovaBridgeError, OSError, ValueError, TypeError, OverflowError) as exc:
        yield (
            progress_html("error", "生成を開始できませんでした", 0.0),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
            gr.update(interactive=False),
            gr.update(value="画像を生成", interactive=True),
            f'<p class="sn-inline-error" role="alert">{html.escape(str(exc))}</p>',
        )


def _cancel(job_id: str):
    if not job_id:
        return progress_html("idle", "停止する生成はありません。", 0.0), gr.update(interactive=False)
    message = cancel_generation(job_id)
    return progress_html("cancel", message, 0.0), gr.update(interactive=False)


def _build_ui():
    initial_status = inspect_runtime(DEFAULT_SOURCE_PATH, checkpoint=DEFAULT_CHECKPOINT_PATH)
    initial_profile = (
        PROFILE_OFFICIAL_8STEP if initial_status.lora_ready else PROFILE_QUALITY
    )
    initial_steps = 8 if initial_status.lora_ready else 50
    initial_cfg = 1.0 if initial_status.lora_ready else 4.0
    initial_request = SenseNovaRequest(
        mode=MODE_TEXT,
        prompt="",
        generation_profile=initial_profile,
        vram_mode="reference",
        steps=initial_steps,
        cfg_scale=initial_cfg,
    )

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Column(elem_id="sensenova-u15-studio", elem_classes=["sn-shell"]):
            gr.HTML(
                """
                <section class="sn-hero">
                  <div>
                    <h2>SenseNova U1.5 Studio</h2>
                  </div>
                </section>
                """
            )
            runtime_status = gr.HTML(
                value=runtime_status_html(initial_status), elem_id="sn-runtime-status"
            )
            quantization = gr.State(QUANT_INT8_CONVROT)
            job_id = gr.State("")
            lora_ready = gr.State(initial_status.lora_ready)

            with gr.Row(elem_classes=["sn-main-grid"]):
                with gr.Column(scale=6, min_width=430, elem_classes=["sn-compose"]):
                    gr.HTML(
                        '<div class="sn-section-title"><h3>つくる</h3></div>'
                    )
                    mode = gr.Radio(
                        choices=[
                            ("テキストから生成", MODE_TEXT),
                            ("複数画像を編集", MODE_EDIT),
                        ],
                        value=MODE_TEXT,
                        label="生成モード",
                        show_label=False,
                        elem_id="sn-mode",
                        elem_classes=["sn-mode"],
                    )
                    mode_note = gr.HTML(
                        (
                            '<div class="sn-mode-note"><b>TEXT TO IMAGE</b><span>参照画像を使わず、公式8-Step高速プリセットまたは50-Step品質プリセットで生成します。</span></div>'
                            if initial_status.lora_ready
                            else '<div class="sn-mode-note"><b>TEXT TO IMAGE</b><span>公式8-Step LoRAが未準備のため、Quality 50-Stepを選択しています。</span></div>'
                        ),
                        elem_id="sn-mode-note",
                    )
                    with gr.Group(elem_classes=["sn-prompt-card"]):
                        gr.Markdown(
                            "### プロンプト\n被写体、構図、質感、照明、文字、残したい要素を具体的に書きます。"
                        )
                        prompt = gr.Textbox(
                            label="SenseNovaプロンプト",
                            show_label=False,
                            lines=6,
                            max_lines=18,
                            placeholder="例: Image-1の人物を保ち、Image-2の衣装を着せる。顔・髪型・背景の構図は変えない。",
                            elem_id="sn-prompt",
                        )
                        gr.HTML(
                            '<div class="sn-prompt-meta" id="sn-prompt-help"><span>Ctrl / ⌘ + Enter で生成</span>'
                            '<span id="sn-draft-status">下書きをこの端末に自動保存</span>'
                            '<span id="sn-prompt-count">0 / 20,000</span></div>'
                        )

                    with gr.Group(
                        visible=False,
                        elem_classes=["sn-reference-card"],
                        elem_id="sn-references",
                    ) as reference_group:
                        gr.Markdown(
                            "### 参照画像\n追加順がモデルへの入力順です。プロンプトでは `Image-1`、`Image-2` のように指定し、選択画像は左右へ移動できます。"
                        )
                        reference_gallery = gr.Gallery(
                            value=[],
                            type="pil",
                            interactive=True,
                            label=f"参照画像（最大{MAX_REFERENCE_IMAGES}枚）",
                            show_label=False,
                            columns=4,
                            rows=2,
                            height=280,
                            allow_preview=True,
                            object_fit="contain",
                            elem_id="sn-reference-gallery",
                        )
                        selected_reference = gr.State(-1)
                        reference_order = gr.HTML(
                            value=reference_order_html([]), elem_id="sn-reference-order"
                        )
                        gr.Markdown(
                            "参照優先では8枚、24GB Safeでは2枚まで。番号は現在の表示順です。並べ替えた後はプロンプトの役割も確認してください。"
                        )
                        with gr.Row():
                            reference_task = gr.Dropdown(
                                choices=[(value[0], key) for key, value in REFERENCE_TASKS.items()],
                                value="identity",
                                label="参照の使い方",
                                elem_id="sn-reference-task",
                            )
                            instruction_button = gr.Button(
                                "役割の文をプロンプトへ追記", elem_id="sn-reference-instruction"
                            )
                        with gr.Row(equal_height=False, elem_classes=["sn-bulk-row"]):
                            bulk_upload = gr.File(
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                                label="複数画像を一括選択",
                                height=92,
                                elem_id="sn-reference-bulk-upload",
                            )
                            bulk_append_button = gr.Button(
                                "選択ファイルを一括追加",
                                variant="secondary",
                                min_width=190,
                                elem_id="sn-bulk-add",
                            )
                        with gr.Row():
                            upload = gr.Image(
                                type="pil",
                                sources=["upload", "clipboard"],
                                label="追加・差し替え画像",
                                height=170,
                                elem_id="sn-reference-upload",
                            )
                            with gr.Column(min_width=190):
                                append_button = gr.Button(
                                    "末尾へ追加", variant="secondary"
                                )
                                replace_button = gr.Button("選択画像を差し替え", variant="secondary", interactive=False)
                                with gr.Row():
                                    move_left_button = gr.Button("← 前へ", size="sm", interactive=False)
                                    move_right_button = gr.Button("後ろへ →", size="sm", interactive=False)
                                remove_button = gr.Button("選択画像を削除", variant="stop", interactive=False)
                                clear_button = gr.Button("すべて消去", variant="stop", interactive=False)

                    gr.HTML(
                        '<div class="sn-section-title"><h3>生成設定</h3></div>'
                    )
                    with gr.Group(elem_classes=["sn-settings-card"]):
                        generation_profile = gr.Radio(
                            choices=[
                                (
                                    "公式8-Step · 高速T2I · 推奨",
                                    PROFILE_OFFICIAL_8STEP,
                                ),
                                ("Quality 50-Step · 基本モデル", PROFILE_QUALITY),
                            ],
                            value=initial_profile,
                            label="生成プロファイル",
                        )
                        resolution = gr.Dropdown(
                            choices=resolution_choices(MODE_TEXT),
                            value="2048x2048",
                            label="出力解像度",
                            elem_id="sn-resolution",
                        )
                        input_max_pixels = gr.Dropdown(
                            choices=[
                                ("省メモリ · 各約0.26MP · 比率保護", str(512 * 512)),
                                ("高精細参照 · 各約1.05MP · 比率保護", str(1024 * 1024)),
                                ("高忠実度 · 各約4.19MP · 比率保護", str(2048 * 2048)),
                                ("自動 · モデル上限まで配分 · 実験用", "auto"),
                            ],
                            value=str(512 * 512),
                            label="参照画像の情報量",
                            visible=False,
                        )
                        with gr.Accordion("詳細設定", open=False, elem_id="sn-advanced"):
                            with gr.Row():
                                steps = gr.Slider(
                                    1,
                                    100,
                                    value=initial_steps,
                                    step=1,
                                    label="Steps",
                                    interactive=not initial_status.lora_ready,
                                )
                                seed = gr.Number(
                                    value=42,
                                    precision=0,
                                    label="Seed",
                                    minimum=0,
                                    maximum=2**32 - 1,
                                )
                            with gr.Row():
                                cfg_scale = gr.Slider(
                                    0,
                                    20,
                                    value=initial_cfg,
                                    step=0.1,
                                    label="CFG",
                                    interactive=not initial_status.lora_ready,
                                )
                                img_cfg_scale = gr.Slider(
                                    0,
                                    20,
                                    value=1.0,
                                    step=0.1,
                                    label="Image CFG",
                                    visible=False,
                                    elem_id="sn-image-cfg",
                                )
                                timestep_shift = gr.Slider(
                                    0.1,
                                    20,
                                    value=3.0,
                                    step=0.1,
                                    label="Timestep Shift",
                                    interactive=not initial_status.lora_ready,
                                )
                            with gr.Row():
                                vram_mode = gr.Dropdown(
                                    choices=[
                                        ("参照優先 · 最大8枚 · KVをCPUへ退避", "reference"),
                                        ("24GB Safe · 2K出力優先 · RTX 3090", "low"),
                                        ("Uncapped streaming · 大容量GPU・実験用", "unrestricted"),
                                        ("Full GPU · 全重み配置・実験用", "full"),
                                    ],
                                    value="reference",
                                    label="VRAMモード",
                                )
                                attn_backend = gr.Dropdown(
                                    choices=[
                                        ("自動", "auto"),
                                        ("PyTorch SDPA", "sdpa"),
                                        ("FlashAttention", "flash"),
                                    ],
                                    value="auto",
                                    label="Attention",
                                )
                                dtype = gr.State("bfloat16")

                    validation = gr.HTML(value="", elem_id="sn-validation")
                    with gr.Row(elem_classes=["sn-actions"]):
                        cancel_button = gr.Button(
                            "キャンセル",
                            variant="stop",
                            interactive=False,
                            elem_id="sn-cancel",
                        )
                        generate_button = gr.Button(
                            "画像を生成",
                            variant="primary",
                            elem_id="sn-generate",
                        )

                with gr.Column(
                    scale=5, min_width=390, elem_classes=["sn-result-column"]
                ):
                    gr.HTML(
                        '<div class="sn-section-title"><h3>生成結果</h3></div>'
                    )
                    result_image = gr.Image(
                        type="filepath",
                        label="生成結果",
                        interactive=False,
                        height=420,
                        elem_id="sn-result-image",
                    )
                    reuse_button = gr.Button("結果をImage-1にして続ける", elem_id="sn-reuse-result", interactive=False)
                    with gr.Row(elem_classes=["sn-downloads"]):
                        result_file = gr.File(
                            label="PNG",
                            interactive=False,
                            elem_id="sn-result-png",
                        )
                        metadata_file = gr.File(
                            label="設定JSON",
                            interactive=False,
                            elem_id="sn-result-json",
                        )
                    progress = gr.HTML(
                        value=progress_html("idle", "生成待ち", 0.0),
                        elem_id="sn-progress",
                    )
                    summary = gr.HTML(
                        value=request_summary_html(initial_request),
                        elem_id="sn-summary",
                    )
                    with gr.Accordion(
                        "生成メタデータ",
                        open=False,
                        elem_id="sn-metadata-panel",
                    ):
                        metadata = gr.JSON(label="生成メタデータ", visible=True)
                    with gr.Accordion(
                        "実行環境とモデル",
                        open=False,
                        elem_id="sn-runtime-setup",
                    ):
                        with gr.Group(elem_classes=["sn-model-card"]):
                            gr.HTML(
                                '<p class="sn-quant-note"><strong>正式版 · INT8 ConvRot</strong><br>参照優先では参照キャッシュを精度を保ったままCPUへ退避し、必要な層だけGPUへ転送します。出力は最大2K、参照は最大8枚です。被写体の比率は保ち、32pxグリッドの余白は端の画素で補います。</p>'
                            )
                            model_path = gr.Textbox(
                                value=DEFAULT_MODEL_ID,
                                label="正式版モデルID",
                                interactive=False,
                            )
                            checkpoint_path = gr.Textbox(
                                value=os.fspath(DEFAULT_CHECKPOINT_PATH),
                                label="INT8 ConvRot checkpoint",
                            )
                            source_path = gr.Textbox(
                                value=os.fspath(DEFAULT_SOURCE_PATH),
                                label="固定済みConvRotランタイム",
                            )
                            gr.Markdown(
                                "未準備の場合はAikimi Studio Neo直下の`download_sensenova_u15_int8.bat`を実行します。"
                            )
                            refresh_button = gr.Button(
                                "準備状況を再確認",
                                variant="secondary",
                                elem_id="sn-refresh-runtime",
                            )

        selection_event = reference_gallery.select(
            _select_reference,
            outputs=[selected_reference],
            queue=False,
            show_progress=False,
        )
        for component in (reference_gallery, selected_reference):
            component.change(
                _selection_controls,
                inputs=[reference_gallery, selected_reference],
                outputs=[
                    replace_button,
                    move_left_button,
                    move_right_button,
                    remove_button,
                    clear_button,
                    reference_order,
                ],
                queue=False,
                trigger_mode="always_last",
            )
        instruction_event = instruction_button.click(
            _append_reference_instruction,
            inputs=[prompt, reference_task, reference_gallery],
            outputs=[prompt],
            queue=False,
        )
        delete_event = reference_gallery.delete(
            _after_gallery_delete,
            inputs=[reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        result_image.change(
            lambda value: gr.update(interactive=bool(value)), inputs=[result_image], outputs=[reuse_button], queue=False
        )
        reuse_event = reuse_button.click(
            _reuse_result,
            inputs=[result_image, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference, mode],
            queue=False,
        )
        gallery_upload_event = reference_gallery.upload(
            _limit_reference_gallery,
            inputs=[reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        bulk_append_event = bulk_append_button.click(
            _append_reference_files,
            inputs=[reference_gallery, bulk_upload],
            outputs=[
                reference_gallery,
                bulk_upload,
                reference_order,
                selected_reference,
            ],
            queue=False,
        )
        append_event = append_button.click(
            _append_reference,
            inputs=[reference_gallery, upload],
            outputs=[reference_gallery, upload, reference_order, selected_reference],
            queue=False,
        )
        replace_event = replace_button.click(
            _replace_reference,
            inputs=[selected_reference, reference_gallery, upload],
            outputs=[reference_gallery, upload, reference_order, selected_reference],
            queue=False,
        )
        remove_event = remove_button.click(
            _remove_reference,
            inputs=[selected_reference, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        move_left_event = move_left_button.click(
            lambda index, gallery: _move_reference(index, gallery, -1),
            inputs=[selected_reference, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        move_right_event = move_right_button.click(
            lambda index, gallery: _move_reference(index, gallery, 1),
            inputs=[selected_reference, reference_gallery],
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )
        clear_event = clear_button.click(
            _clear_references,
            outputs=[reference_gallery, reference_order, selected_reference],
            queue=False,
        )

        mode.change(
            _mode_updates,
            inputs=[mode, resolution, lora_ready],
            outputs=[
                reference_group,
                resolution,
                img_cfg_scale,
                input_max_pixels,
                mode_note,
                generation_profile,
                steps,
                cfg_scale,
                timestep_shift,
            ],
            queue=False,
        )
        generation_profile.change(
            _profile_updates,
            inputs=[generation_profile],
            outputs=[steps, cfg_scale, timestep_shift],
            queue=False,
        )
        refresh_button.click(
            _refresh_runtime,
            inputs=[source_path, checkpoint_path, mode, generation_profile],
            outputs=[
                runtime_status,
                lora_ready,
                generation_profile,
                steps,
                cfg_scale,
                timestep_shift,
            ],
            queue=False,
        )

        request_inputs = [
            mode,
            prompt,
            reference_gallery,
            model_path,
            quantization,
            checkpoint_path,
            source_path,
            resolution,
            input_max_pixels,
            generation_profile,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
            dtype,
        ]
        for component in [
            mode,
            reference_gallery,
            checkpoint_path,
            source_path,
            resolution,
            input_max_pixels,
            generation_profile,
            steps,
            cfg_scale,
            img_cfg_scale,
            timestep_shift,
            seed,
            vram_mode,
            attn_backend,
        ]:
            component.change(
                _summary_from_ui,
                inputs=request_inputs,
                outputs=[summary, validation],
                queue=False,
                trigger_mode="always_last",
            )
        for reference_event in [
            instruction_event,
            delete_event,
            reuse_event,
            gallery_upload_event,
            bulk_append_event,
            append_event,
            replace_event,
            remove_event,
            move_left_event,
            move_right_event,
            clear_event,
        ]:
            reference_event.then(
                _summary_from_ui,
                inputs=request_inputs,
                outputs=[summary, validation],
                queue=False,
            )
            reference_event.then(
                _selection_controls,
                inputs=[reference_gallery, selected_reference],
                outputs=[
                    replace_button,
                    move_left_button,
                    move_right_button,
                    remove_button,
                    clear_button,
                    reference_order,
                ],
                queue=False,
            )
        selection_event.then(
            _selection_controls,
            inputs=[reference_gallery, selected_reference],
            outputs=[replace_button, move_left_button, move_right_button, remove_button, clear_button, reference_order],
            queue=False,
        )

        prompt.blur(
            _summary_from_ui,
            inputs=request_inputs,
            outputs=[summary, validation],
            queue=False,
            trigger_mode="always_last",
        )

        generation_event = generate_button.click(
            _generate,
            inputs=request_inputs,
            outputs=[
                progress,
                result_image,
                result_file,
                metadata_file,
                metadata,
                job_id,
                cancel_button,
                generate_button,
                validation,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="sensenova-u15-generation",
        )
        generation_event.then(
            lambda value: gr.update(interactive=bool(value)),
            inputs=[result_image],
            outputs=[reuse_button],
            queue=False,
        )
        cancel_button.click(
            _cancel,
            inputs=[job_id],
            outputs=[progress, cancel_button],
            queue=False,
        )
        interface.load(
            None,
            outputs=[prompt],
            js=f'''() => {{
                try {{
                    return window.localStorage.getItem("{PROMPT_DRAFT_KEY}") || "";
                }} catch (_error) {{
                    return "";
                }}
            }}''',
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        interface.load(
            _refresh_runtime,
            inputs=[source_path, checkpoint_path, mode, generation_profile],
            outputs=[
                runtime_status,
                lora_ready,
                generation_profile,
                steps,
                cfg_scale,
                timestep_shift,
            ],
            queue=False,
        )

    return [(interface, "SenseNova U1.5", "sensenova_u15_studio")]


script_callbacks.on_ui_tabs(_build_ui, name="sensenova_u15_studio")
