from __future__ import annotations

import html
import math
import os
from pathlib import Path

import gradio as gr

from modules import script_callbacks
from modules.paths import data_path, script_path
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_acceleration_ui import create_acceleration_controls
from modules_forge.minimax_h3_fun_control import H3FunControl
from modules_forge.minimax_h3_bridge import (
    H3BridgeError,
    H3GenerationCancelled,
    H3Request,
    H3_SERVER_URL,
    MODE_KEYFRAMES,
    MODE_REFERENCES,
    MODE_TEXT,
    RUNTIME_PROFILE_FAST,
    RUNTIME_PROFILE_LOW_RAM,
    append_prompt_section,
    cache_history_video,
    cancel_generation,
    generation_preset_values,
    history_choices,
    history_html,
    inspect_readiness,
    list_history,
    load_history_request,
    normalize_file_list,
    progress_html,
    prompt_template,
    readiness_html,
    reference_guide_html,
    restart_runtime,
    resolve_runtime_root,
    run_generation,
    settings_summary_html,
    ensure_ready,
    validate_request,
)
from modules_forge.minimax_h3_runtime import installed_runtime_root, managed_runtime_root
from modules_forge.minimax_h3_setup_ui import initial_setup_state, setup_updates


OUTPUT_DIRECTORY = Path(data_path) / "outputs" / "minimax_h3"
LOG_DIRECTORY = Path(data_path) / "logs" / "minimax_h3"

MODE_HELP = {
    MODE_TEXT: (
        "TEXT · FL2VA",
        "言葉だけから映像と32kHzステレオ音声を同時生成します。最初の一作におすすめです。",
    ),
    MODE_KEYFRAMES: (
        "KEYFRAMES · FL2VA",
        "開始画像、終了画像、または両方を指定して、その間の動きと音を生成します。",
    ),
    MODE_REFERENCES: (
        "REFERENCES · REF2VA",
        "人物・画風・動き・声を画像／動画／音声から参照します。タグ順序を確認して使います。",
    ),
}

QUALITY_LABELS = {
    "draft": "動作確認 · 約0.2 MP · 速い",
    "preview": "標準 · 公式Fast Preview · 約0.4 MP",
    "balanced": "バランス · 約0.5 MP",
    "native": "高品質 · Native 768p · 重い",
}

PRESET_STATE_LABELS = {
    "quick": "動作確認",
    "recommended": "標準",
    "final": "高品質",
    "custom": "カスタム",
}


def _initial_runtime() -> Path | None:
    return managed_runtime_root(Path(script_path))


def _setup_runtime(shared_models: str):
    yield from setup_updates(shared_models, Path(script_path))


def _setup_state():
    shared, message = initial_setup_state(Path(script_path))
    return gr.update(value=shared, interactive=True), message, gr.update(interactive=True)


def _initial_generation_preset(readiness, runtime_profile: str) -> str:
    memory_values = [
        float(value)
        for value in (readiness.ram_free_gib, readiness.commit_free_gib)
        if isinstance(value, (int, float)) and value >= 0
    ]
    if not readiness.connected or not memory_values:
        return "recommended"
    request = H3Request(mode=MODE_TEXT, prompt="initial-preset", quality="preview")
    width, height = request.dimensions
    safety_gib = 2.0 if runtime_profile == RUNTIME_PROFILE_LOW_RAM else 4.0
    required_gib = width * height * request.frame_count * 3 * 4 / 1024**3 + safety_gib
    return "quick" if min(memory_values) < required_gib else "recommended"


def _mode_help_html(mode: str) -> str:
    eyebrow, message = MODE_HELP.get(mode, MODE_HELP[MODE_TEXT])
    return (
        '<div class="h3-mode-help">'
        f'<span>{html.escape(eyebrow)}</span><p>{html.escape(message)}</p>'
        "</div>"
    )


def _mode_updates(
    mode: str,
    aspect: str = "16:9",
    quality: str = "preview",
    duration: float = 5.0,
    steps: int = 20,
    scheduler: str = "simple",
    ref_image_size: str = "match",
    current_validation: str = "",
):
    reference_mode = mode == MODE_REFERENCES
    effective_ref_image_size = str(ref_image_size) if reference_mode else "match"
    return (
        gr.update(visible=True if mode == MODE_KEYFRAMES else "hidden"),
        gr.update(visible=True if reference_mode else "hidden"),
        _mode_help_html(mode),
        gr.update(visible=True, value=effective_ref_image_size)
        if reference_mode
        else gr.update(visible="hidden", value=effective_ref_image_size),
        settings_summary_html(
            aspect,
            quality,
            duration,
            steps,
            scheduler,
            effective_ref_image_size,
        ),
        _clear_validation_targets(current_validation, "keyframes", "references"),
    )


def _runtime_notice_html(
    title: str,
    message: str,
    *,
    tone: str = "active",
    role: str = "status",
) -> str:
    return (
        f'<div class="h3-runtime-card" data-tone="{html.escape(tone, quote=True)}" '
        f'role="{html.escape(role, quote=True)}" aria-live="polite" aria-atomic="true">'
        '<div class="h3-runtime-summary">'
        f'<span class="h3-runtime-primary" data-tone="{html.escape(tone, quote=True)}">'
        f'<i aria-hidden="true"></i><strong>{html.escape(title)}</strong></span>'
        f'<p>{html.escape(message)}</p></div></div>'
    )


def _status_error(message: str) -> str:
    return _runtime_notice_html("確認が必要です", message, tone="error", role="alert")


def _preset_state_html(preset: str) -> str:
    label = PRESET_STATE_LABELS.get(preset, PRESET_STATE_LABELS["custom"])
    safe_preset = preset if preset in PRESET_STATE_LABELS else "custom"
    return (
        f'<p class="h3-preset-state" data-h3-preset="{safe_preset}" '
        'role="status" aria-live="polite" aria-atomic="true">'
        f'現在の設定: <strong>{html.escape(label)}</strong></p>'
    )


def _apply_generation_preset(preset: str, aspect: str):
    return (*generation_preset_values(preset, aspect), _preset_state_html(preset))


def _apply_quick_preset(aspect: str):
    return _apply_generation_preset("quick", aspect)


def _apply_recommended_preset(aspect: str):
    return _apply_generation_preset("recommended", aspect)


def _apply_final_preset(aspect: str):
    return _apply_generation_preset("final", aspect)


def _mark_preset_custom(_value: object | None = None) -> str:
    return _preset_state_html("custom")


def _custom_settings_updates(
    aspect: str,
    quality: str,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
    ref_image_size: str,
    current_validation: str = "",
) -> tuple[str, str, str | dict]:
    return (
        settings_summary_html(aspect, quality, duration, steps, scheduler, ref_image_size),
        _mark_preset_custom(),
        _clear_settings_validation(
            aspect,
            quality,
            duration,
            steps,
            seed,
            scheduler,
            ref_image_size,
            current_validation,
        ),
    )


def _aspect_settings_updates(
    aspect: str,
    quality: str,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
    ref_image_size: str,
    current_validation: str = "",
):
    return (
        settings_summary_html(aspect, quality, duration, steps, scheduler, ref_image_size),
        _clear_settings_validation(
            aspect,
            quality,
            duration,
            steps,
            seed,
            scheduler,
            ref_image_size,
            current_validation,
        ),
    )


def _input_validation_html(message: str, target: str, control: str | None = None) -> str:
    safe_target = target if target in {"prompt", "keyframes", "references", "settings"} else "settings"
    valid_controls = {
        "prompt",
        "first_frame",
        "reference_images",
        "reference_videos",
        "reference_audios",
        "aspect",
        "quality",
        "duration",
        "steps",
        "seed",
        "scheduler",
        "ref_image_size",
    }
    safe_control = control if control in valid_controls else {
        "prompt": "prompt",
        "keyframes": "first_frame",
        "references": "reference_images",
        "settings": "aspect",
    }[safe_target]
    return (
        f'<div class="h3-input-error" data-h3-invalid="{safe_target}" '
        f'data-h3-control="{safe_control}" role="alert">'
        '<strong>入力を確認してください</strong>'
        f'<span id="h3-input-validation-message">{html.escape(message)}</span>'
        "</div>"
    )


def _validation_target(request: H3Request) -> str:
    if not request.prompt or not request.prompt.strip() or len(request.prompt) > 20_000:
        return "prompt"
    try:
        request.dimensions
        request.frame_count
    except (H3BridgeError, TypeError, ValueError, OverflowError):
        return "settings"
    if not 1 <= int(request.steps) <= 100:
        return "settings"
    if int(request.seed) < -1 or int(request.seed) >= 2**63:
        return "settings"
    if request.scheduler not in {"simple", "beta", "normal"}:
        return "settings"
    if request.ref_image_size not in {"match", "max"}:
        return "settings"
    if request.mode == MODE_KEYFRAMES:
        return "keyframes"
    if request.mode == MODE_REFERENCES:
        return "references"
    return "settings"


def _validation_control(request: H3Request | None, message: str, target: str) -> str:
    lowered = str(message or "").lower()
    if target == "prompt":
        return "prompt"
    if "seed" in lowered:
        return "seed"
    if "steps" in lowered:
        return "steps"
    if "scheduler" in lowered:
        return "scheduler"
    if "参照画像サイズ" in lowered or "reference image size" in lowered:
        return "ref_image_size"
    if "アスペクト" in lowered or "aspect" in lowered:
        return "aspect"
    if "品質" in lowered or "quality" in lowered or "解像度" in lowered:
        return "quality"
    if "長さ" in lowered or "秒数" in lowered or "duration" in lowered:
        return "duration"
    if target == "keyframes":
        return "first_frame"
    if target == "references":
        if "動画" in lowered:
            return "reference_videos"
        if "音声" in lowered and "音声だけ" not in lowered:
            return "reference_audios"
        return "reference_images"
    if request is not None and request.mode == MODE_REFERENCES:
        return "reference_images"
    return "aspect"


def _clear_validation_targets(current_validation: str, *targets: str):
    rendered = str(current_validation or "")
    if any(f'data-h3-invalid="{target}"' in rendered for target in targets):
        return ""
    return gr.update()


def _clear_prompt_validation(prompt: str, current_validation: str):
    if str(prompt or "").strip() and len(str(prompt or "")) <= 20_000:
        return _clear_validation_targets(current_validation, "prompt")
    return gr.update()


def _clear_keyframe_validation(first_frame, last_frame, current_validation: str):
    if first_frame or last_frame:
        return _clear_validation_targets(current_validation, "keyframes")
    return gr.update()


def _clear_settings_validation(
    aspect: str,
    quality: str,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
    ref_image_size: str,
    current_validation: str,
):
    try:
        request = _request_from_ui(
            MODE_TEXT,
            "settings-validation",
            None,
            None,
            None,
            None,
            None,
            aspect,
            quality,
            duration,
            steps,
            seed,
            scheduler,
            ref_image_size,
        )
        validate_request(request)
    except (H3BridgeError, TypeError, ValueError, OverflowError):
        return gr.update()
    return _clear_validation_targets(current_validation, "settings")


def _reference_guide_updates(
    image_values,
    video_values,
    audio_values,
    current_validation: str,
):
    rendered_guide = reference_guide_html(image_values, video_values, audio_values)
    has_visual_reference = bool(normalize_file_list(image_values) or normalize_file_list(video_values))
    validation_update = gr.update()
    if has_visual_reference and 'data-tone="error"' not in rendered_guide:
        validation_update = _clear_validation_targets(current_validation, "references")
    return (
        rendered_guide,
        validation_update,
    )


def _prompt_action_updates(prompt: str, action: str, current_validation: str):
    updated_prompt = prompt_template(prompt) if action == "template" else append_prompt_section(prompt, action)
    return updated_prompt, _clear_prompt_validation(updated_prompt, current_validation)


def _prompt_template_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "template", current_validation)


def _prompt_camera_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "camera", current_validation)


def _prompt_dialogue_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "dialogue", current_validation)


def _prompt_sfx_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "sfx", current_validation)


def _prompt_music_updates(prompt: str, current_validation: str):
    return _prompt_action_updates(prompt, "music", current_validation)


def _connect_runtime(runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values) -> str:
    acceleration = H3Acceleration.from_values(acceleration_values)
    try:
        root = resolve_runtime_root(runtime_value)
        readiness = ensure_ready(
            root,
            server_url,
            LOG_DIRECTORY,
            runtime_profile=runtime_profile,
            acceleration=acceleration,
        )
        return readiness_html(readiness, runtime_profile)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _restart_runtime(runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values) -> str:
    acceleration = H3Acceleration.from_values(acceleration_values)
    try:
        root = resolve_runtime_root(runtime_value)
        readiness = restart_runtime(
            root,
            server_url,
            LOG_DIRECTORY,
            runtime_profile=runtime_profile,
            acceleration=acceleration,
        )
        return readiness_html(readiness, runtime_profile)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _rescan_runtime(runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values) -> str:
    acceleration = H3Acceleration.from_values(acceleration_values)
    try:
        root = resolve_runtime_root(runtime_value)
        return readiness_html(inspect_readiness(root, server_url, acceleration=acceleration), runtime_profile)
    except H3BridgeError as exc:
        return _status_error(str(exc))


def _runtime_operation(callback, message: str, runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values):
    yield (
        _runtime_notice_html("実行環境を更新中", message),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    try:
        rendered = callback(runtime_value, server_url, runtime_profile, *acceleration_values)
    except Exception as exc:
        rendered = _status_error(str(exc))
    yield (
        rendered,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def _connect_runtime_updates(runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values):
    yield from _runtime_operation(
        _connect_runtime,
        "H3 backendへ接続しています…",
        runtime_value,
        server_url,
        runtime_profile,
        *acceleration_values,
    )


def _restart_runtime_updates(runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values):
    yield from _runtime_operation(
        _restart_runtime,
        "選択した起動プロファイルで再起動しています…",
        runtime_value,
        server_url,
        runtime_profile,
        *acceleration_values,
    )


def _rescan_runtime_updates(runtime_value: str, server_url: str, runtime_profile: str, *acceleration_values):
    yield from _runtime_operation(
        _rescan_runtime,
        "H3 backendとモデルの状態を確認しています…",
        runtime_value,
        server_url,
        runtime_profile,
        *acceleration_values,
    )


def _history_state(runtime_value: str):
    try:
        root = resolve_runtime_root(runtime_value) if runtime_value else None
    except H3BridgeError:
        root = None
    try:
        items = list_history(root, OUTPUT_DIRECTORY)
        rendered = history_html(items)
        choices = history_choices(items)
    except (OSError, ValueError, OverflowError) as exc:
        raise H3BridgeError(f"生成履歴を読み込めません: {exc}") from exc
    return items, rendered, choices


def _history_error_html(message: str) -> str:
    return (
        '<div class="h3-history-empty" data-tone="error" role="alert">'
        '<span>履歴を読み込めませんでした</span>'
        f'<small>{html.escape(message)}</small></div>'
    )


def _initial_history_state(runtime_value: str):
    try:
        return _history_state(runtime_value)
    except H3BridgeError as exc:
        return [], _history_error_html(str(exc)), []


def _refresh_history(runtime_value: str, current_selected: str | None):
    try:
        _, rendered, choices = _history_state(runtime_value)
        available_values = {value for _, value in choices}
        selected = current_selected if current_selected in available_values else (
            choices[0][1] if choices else None
        )
        return rendered, gr.update(choices=choices, value=selected)
    except H3BridgeError as exc:
        return _history_error_html(str(exc)), gr.update()


def _load_history_video(selected: str, runtime_value: str):
    if not selected:
        return gr.update(), progress_html("idle", "表示する履歴を選択してください", 0.0)
    try:
        items, _, _ = _history_state(runtime_value)
        path = cache_history_video(selected, items, OUTPUT_DIRECTORY)
        return path, progress_html("complete", "履歴の動画を読み込みました", 1.0)
    except H3BridgeError as exc:
        return gr.update(), progress_html("error", str(exc), 0.0)


def _matching_generation_preset(request: H3Request) -> str:
    if request.acceleration != H3Acceleration():
        return "custom"
    for preset in ("quick", "recommended", "final"):
        quality, duration, steps, scheduler, ref_image_size, _ = generation_preset_values(
            preset,
            request.aspect,
        )
        if (
            request.quality == quality
            and request.duration_seconds == duration
            and request.steps == steps
            and request.scheduler == scheduler
            and request.ref_image_size == ref_image_size
        ):
            return preset
    return "custom"


def _restore_history_settings(selected: str, runtime_value: str, include_acceleration: bool = False):
    try:
        if not selected:
            raise H3BridgeError("設定を復元する履歴を選択してください。")
        items, _, _ = _history_state(runtime_value)
        request = load_history_request(selected, items, OUTPUT_DIRECTORY)
    except H3BridgeError as exc:
        return (*[gr.update() for _ in range(21)], progress_html("error", str(exc), 0.0), *([gr.update() for _ in range(len(H3Acceleration().values()) + 3)] if include_acceleration else []))

    effective_ref_image_size = request.ref_image_size if request.mode == MODE_REFERENCES else "match"
    restored_request = H3Request(
        mode=request.mode,
        prompt=request.prompt,
        aspect=request.aspect,
        quality=request.quality,
        duration_seconds=request.duration_seconds,
        steps=request.steps,
        seed=request.seed,
        scheduler=request.scheduler,
        ref_image_size=effective_ref_image_size,
        acceleration=request.acceleration,
        control=request.control,
    )
    mode_updates = _mode_updates(
        restored_request.mode,
        restored_request.aspect,
        restored_request.quality,
        restored_request.duration_seconds,
        restored_request.steps,
        restored_request.scheduler,
        restored_request.ref_image_size,
        "",
    )
    preset = _matching_generation_preset(restored_request)
    message = "履歴からプロンプトと生成設定を復元しました。入力素材は安全のため引き継ぎません。"
    if restored_request.mode == MODE_KEYFRAMES:
        message += "開始・終了フレームはもう一度追加してください。"
    elif restored_request.mode == MODE_REFERENCES:
        message += "参照素材はもう一度追加してください。"
    if restored_request.control.enabled:
        message += "Fun ControlNetの制御動画も追加してください。"
    return (
        gr.update(value=restored_request.mode),
        gr.update(value=restored_request.prompt),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        mode_updates[0],
        mode_updates[1],
        mode_updates[2],
        reference_guide_html(None, None, None),
        gr.update(value=restored_request.aspect),
        gr.update(value=restored_request.quality),
        gr.update(value=restored_request.duration_seconds),
        gr.update(value=restored_request.steps),
        gr.update(value=restored_request.seed),
        gr.update(value=restored_request.scheduler),
        mode_updates[3],
        mode_updates[4],
        gr.update(value=_preset_state_html(preset)),
        "",
        progress_html("idle", message, 0.0),
        *([gr.update(value=value) for value in request.acceleration.values()] if include_acceleration else []),
        *([gr.update(value=request.control.mode), gr.update(value=None), gr.update(value=request.control.strength)] if include_acceleration else []),
    )


def _request_from_ui(
    mode,
    prompt,
    first_frame,
    last_frame,
    reference_images,
    reference_videos,
    reference_audios,
    aspect,
    quality,
    duration,
    steps,
    seed,
    scheduler,
    ref_image_size,
    control_mode="off",
    control_video=None,
    control_strength=1.0,
    *acceleration_values,
) -> H3Request:
    try:
        duration_value = float(duration)
        if not math.isfinite(duration_value):
            raise ValueError("non-finite duration")
    except (TypeError, ValueError, OverflowError) as exc:
        raise H3BridgeError("長さは秒数で指定してください。") from exc
    try:
        steps_value = int(steps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise H3BridgeError("Steps は整数で指定してください。") from exc
    try:
        seed_value = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise H3BridgeError("Seed は整数で指定してください。") from exc
    return H3Request(
        mode=str(mode),
        prompt=str(prompt or ""),
        first_frame=os.fspath(first_frame) if first_frame else None,
        last_frame=os.fspath(last_frame) if last_frame else None,
        reference_images=normalize_file_list(reference_images),
        reference_videos=normalize_file_list(reference_videos),
        reference_audios=normalize_file_list(reference_audios),
        aspect=str(aspect),
        quality=str(quality),
        duration_seconds=duration_value,
        steps=steps_value,
        seed=seed_value,
        scheduler=str(scheduler),
        ref_image_size=str(ref_image_size),
        acceleration=H3Acceleration.from_values(acceleration_values),
        control=H3FunControl(str(control_mode), float(control_strength)),
        control_video=os.fspath(control_video) if control_video else None,
    )


def _generate(
    runtime_value,
    server_url,
    runtime_profile,
    mode,
    prompt,
    first_frame,
    last_frame,
    reference_images,
    reference_videos,
    reference_audios,
    aspect,
    quality,
    duration,
    steps,
    seed,
    scheduler,
    ref_image_size,
    control_mode="off",
    control_video=None,
    control_strength=1.0,
    *acceleration_values,
):
    yield (
        progress_html("prepare", "生成条件を確認しています", 0.02),
        gr.update(),
        "",
        gr.update(),
        gr.update(),
        gr.update(interactive=False),
        gr.update(value="生成を準備中…", interactive=False),
        "",
    )
    try:
        request: H3Request | None = None
        try:
            request = _request_from_ui(
                mode,
                prompt,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                aspect,
                quality,
                duration,
                steps,
                seed,
                scheduler,
                ref_image_size,
                control_mode,
                control_video,
                control_strength,
                *acceleration_values,
            )
            validate_request(request)
        except (H3BridgeError, ValueError, TypeError, OverflowError) as exc:
            target = _validation_target(request) if request is not None else "settings"
            control = _validation_control(request, str(exc), target)
            yield (
                progress_html("validation", "入力欄を確認してください", 0.0),
                gr.update(),
                "",
                gr.update(),
                gr.update(),
                gr.update(interactive=False),
                gr.update(value="映像＋音声を生成", interactive=True),
                _input_validation_html(str(exc), target, control),
            )
            return
        root = resolve_runtime_root(runtime_value)
        for update in run_generation(
            request,
            root,
            server_url,
            LOG_DIRECTORY,
            OUTPUT_DIRECTORY,
            runtime_profile=runtime_profile,
        ):
            video_update = update.get("path") or gr.update()
            prompt_id = update.get("prompt_id") or ""
            rendered_history = gr.update()
            selector_update = gr.update()
            if update["stage"] == "complete":
                try:
                    _, rendered_history, choices = _history_state(runtime_value)
                    selected = choices[0][1] if choices else None
                    selector_update = gr.update(choices=choices, value=selected)
                except H3BridgeError as exc:
                    rendered_history = _history_error_html(str(exc))
            cancel_update = gr.update(
                interactive=bool(prompt_id) and update["stage"] not in {"complete", "error"}
            )
            generation_finished = update["stage"] in {"complete", "error"}
            if update["stage"] == "runtime":
                generate_label = "H3 backendを確認中…"
            elif update["stage"] == "reconnecting":
                generate_label = "H3 backendへ再接続中…"
            else:
                generate_label = "映像＋音声を生成" if generation_finished else "生成中…"
            generate_update = gr.update(
                value=generate_label,
                interactive=generation_finished,
            )
            yield (
                progress_html(
                    update["stage"],
                    update["message"],
                    update.get("progress", 0.0),
                    update.get("elapsed"),
                ),
                video_update,
                prompt_id,
                rendered_history,
                selector_update,
                cancel_update,
                generate_update,
                gr.update(),
            )
    except H3GenerationCancelled as exc:
        yield (
            progress_html("cancelled", str(exc), 0.0),
            gr.update(),
            "",
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(value="映像＋音声を生成", interactive=True),
            gr.update(),
        )
    except (H3BridgeError, OSError, ValueError, TypeError, OverflowError) as exc:
        yield (
            progress_html("error", str(exc), 0.0),
            gr.update(),
            "",
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            gr.update(value="映像＋音声を生成", interactive=True),
            gr.update(),
        )


def _cancel(prompt_id: str, server_url: str):
    try:
        if prompt_id:
            cancel_generation(prompt_id, server_url)
            return progress_html("active", "停止要求を送りました", 0.0), "", gr.update(interactive=False)
        return (
            progress_html("idle", "実行中の H3 ジョブはありません", 0.0),
            "",
            gr.update(interactive=False),
        )
    except H3BridgeError as exc:
        return progress_html("error", str(exc), 0.0), prompt_id, gr.update(interactive=bool(prompt_id))


def _runtime_checking_html() -> str:
    return _runtime_notice_html(
        "生成準備を確認中",
        "backend・モデル・利用可能なメモリを確認しています…",
    )


def _initial_ui_updates(
    runtime_value: str,
    server_url: str,
    selected_profile: str,
    aspect: str,
):
    runtime_profile = (
        selected_profile
        if selected_profile in {RUNTIME_PROFILE_FAST, RUNTIME_PROFILE_LOW_RAM}
        else RUNTIME_PROFILE_FAST
    )
    try:
        root = resolve_runtime_root(runtime_value) if runtime_value else None
        readiness = inspect_readiness(root, server_url)
        if readiness.runtime_profile in {RUNTIME_PROFILE_FAST, RUNTIME_PROFILE_LOW_RAM}:
            runtime_profile = readiness.runtime_profile
        rendered_status = readiness_html(readiness, runtime_profile)
        initial_preset = _initial_generation_preset(readiness, runtime_profile)
    except Exception as exc:
        rendered_status = _status_error(str(exc))
        initial_preset = "recommended"
    quality, duration, steps, scheduler, ref_image_size, summary = generation_preset_values(
        initial_preset,
        aspect,
    )
    return (
        rendered_status,
        gr.update(value=runtime_profile, interactive=True),
        gr.update(interactive=True),
        gr.update(value=quality, interactive=True),
        gr.update(value=duration, interactive=True),
        gr.update(value=steps, interactive=True),
        gr.update(interactive=True),
        gr.update(value=scheduler, interactive=True),
        gr.update(value=ref_image_size, interactive=True),
        summary,
        _preset_state_html(initial_preset),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value="映像＋音声を生成", interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def _runtime_profile_pending(runtime_profile: str) -> str:
    labels = {
        RUNTIME_PROFILE_FAST: "高速（Async 2）",
        RUNTIME_PROFILE_LOW_RAM: "省RAM（Async無効）",
    }
    label = labels.get(runtime_profile)
    if label is None:
        return _status_error(f"未対応のH3 runtime profileです: {runtime_profile}")
    return _runtime_notice_html(
        "再起動すると反映されます",
        f"{label}を選択しました。「選択設定で再起動」を押してください。",
        tone="warn",
    )


def _restore_history_with_acceleration(selected: str, runtime_value: str):
    return _restore_history_settings(selected, runtime_value, include_acceleration=True)


def _initial_ui_with_acceleration(*args):
    # Do not permit a Turbo preset before initial readiness can overwrite Steps.
    return (*_initial_ui_updates(*args), *[gr.update(interactive=True) for _ in range(len(H3Acceleration().values()) + 6)])


def _acceleration_pending(*values):
    try:
        option = H3Acceleration.from_values(values)
    except ValueError as exc:
        return _runtime_notice_html("追加設定を確認してください", str(exc), tone="warn")
    message = "選択した構成で「状態を再確認」を押してください。重みは自動取得しません。"
    message += "Fast VAE / NegPiP / 長尺生成の有効・無効を切り替えた場合は「選択設定で再起動」が必要です。"
    if option.negpip.enabled:
        message += "NegPiPは同梱の固定版だけを選択したローカルComfyUIへ導入します。"
    return _runtime_notice_html("追加構成を変更しました", message, tone="warn")


def _turbo_preset_updates(steps, aspect, quality, duration, ref_image_size):
    # Only the explicit button changes Steps; it leaves VAE / Sparse choices alone.
    return (
        "fused_turbo", steps, "simple",
        settings_summary_html(aspect, quality, duration, steps, "simple", ref_image_size),
        _preset_state_html("custom"),
    )


def _turbo_four_updates(aspect, quality, duration, ref_image_size):
    return _turbo_preset_updates(4, aspect, quality, duration, ref_image_size)


def _turbo_eight_updates(aspect, quality, duration, ref_image_size):
    return _turbo_preset_updates(8, aspect, quality, duration, ref_image_size)


def _reset_acceleration_updates(aspect, quality, duration, ref_image_size):
    return (
        *H3Acceleration().values(), 20, "simple",
        settings_summary_html(aspect, quality, duration, 20, "simple", ref_image_size),
        _preset_state_html("custom"),
    )


def _build_ui():
    runtime_root = _initial_runtime()
    runtime_value = os.fspath(runtime_root) if runtime_root else ""
    shared_models_value, setup_message = initial_setup_state(Path(script_path))
    initial_runtime_profile = RUNTIME_PROFILE_FAST
    initial_preset = "recommended"
    (
        initial_quality,
        initial_duration,
        initial_steps,
        initial_scheduler,
        initial_ref_image_size,
        initial_settings_summary,
    ) = generation_preset_values(initial_preset, "16:9")
    initial_history, initial_history_html, initial_history_choices = _initial_history_state(runtime_value)
    initial_history_value = initial_history_choices[0][1] if initial_history_choices else None

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Column(elem_id="h3-studio", elem_classes=["h3-studio-shell"]):
            gr.HTML(
                """
                <section class="h3-hero">
                  <div class="h3-hero-copy">
                    <h2>MiniMax H3 Studio</h2>
                  </div>
                </section>
                """
            )
            runtime_status = gr.HTML(
                value=_runtime_checking_html(),
                elem_id="h3-runtime-status",
            )

            with gr.Row(elem_classes=["h3-main-grid"]):
                with gr.Column(scale=6, min_width=420, elem_classes=["h3-compose"]):
                    gr.HTML(
                        '<div class="h3-section-kicker"><h3>つくり方</h3></div>'
                    )
                    mode = gr.Radio(
                        choices=[
                            ("テキスト", MODE_TEXT),
                            ("キーフレーム", MODE_KEYFRAMES),
                            ("参照素材", MODE_REFERENCES),
                        ],
                        value=MODE_TEXT,
                        label="生成モード",
                        show_label=False,
                        elem_id="h3-mode",
                        elem_classes=["h3-mode-selector"],
                    )
                    mode_help = gr.HTML(value=_mode_help_html(MODE_TEXT), elem_id="h3-mode-help")

                    with gr.Group(elem_classes=["h3-prompt-card"]):
                        with gr.Row(elem_classes=["h3-card-heading"]):
                            gr.Markdown("### プロンプト\n映像・カメラ・台詞・効果音・音楽を、同じ時系列で書きます。")
                            prompt_template_button = gr.Button(
                                "構成テンプレートを挿入",
                                size="sm",
                                elem_id="h3-prompt-template",
                                elem_classes=["h3-quiet-button"],
                            )
                        prompt = gr.Textbox(
                            value="",
                            placeholder=(
                                "例: 雨上がりの東京。赤い傘を持つ人物を低いカメラで追う。\n"
                                "3秒で振り返り、遠くの雷に合わせて街の環境音が一瞬静まる。"
                            ),
                            lines=4,
                            max_lines=14,
                            label="H3 プロンプト",
                            show_label=False,
                            elem_id="h3-prompt",
                        )
                        gr.HTML(
                            '<div class="h3-prompt-meta"><span>Ctrl / ⌘ + Enter で生成</span>'
                            '<span id="h3-draft-status">下書きをこの端末に自動保存</span>'
                            '<span id="h3-prompt-count">0 / 20,000</span></div>',
                            elem_id="h3-prompt-meta",
                        )
                        input_validation = gr.HTML(value="", elem_id="h3-input-validation")

                    with gr.Group(visible="hidden", elem_id="h3-keyframes", elem_classes=["h3-media-panel"]) as keyframe_group:
                        gr.Markdown("### キーフレーム\n片方だけでも使えます。両方指定すると、その間の動きを補間します。")
                        with gr.Row():
                            first_frame = gr.Image(
                                type="filepath",
                                label="開始フレーム",
                                height=190,
                                elem_id="h3-first-frame",
                            )
                            last_frame = gr.Image(
                                type="filepath",
                                label="終了フレーム",
                                height=190,
                                elem_id="h3-last-frame",
                            )

                    with gr.Group(visible="hidden", elem_id="h3-references", elem_classes=["h3-media-panel"]) as reference_group:
                        gr.Markdown(
                            "### 参照素材\n"
                            "画像9枚・動画3本・音声3本、合計12個まで。動画と音声の合計は15秒以内です。"
                        )
                        reference_images = gr.File(
                            label="参照画像 · Picture 1–9",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                            elem_id="h3-reference-images",
                        )
                        reference_videos = gr.File(
                            label="参照動画 · Video 1–3（2〜15秒）",
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                            elem_id="h3-reference-videos",
                        )
                        reference_audios = gr.File(
                            label="参照音声 · Audio 1–3（2〜15秒）",
                            file_count="multiple",
                            file_types=["audio"],
                            type="filepath",
                            elem_id="h3-reference-audios",
                        )
                        reference_guide = gr.HTML(
                            value=reference_guide_html(None, None, None),
                            elem_id="h3-reference-guide",
                        )

                    gr.HTML(
                        '<div class="h3-section-kicker"><h3>生成設定</h3></div>'
                    )
                    with gr.Group(elem_classes=["h3-settings-card"]):
                        gr.Markdown("#### 用途で選ぶ\n解像度・5秒・20 Stepsを用途別の標準値へまとめて揃えます。")
                        with gr.Row(elem_classes=["h3-speed-presets"]):
                            quick_preset_button = gr.Button(
                                "動作確認\n速い・低解像度",
                                size="sm",
                                interactive=False,
                                elem_id="h3-preset-quick",
                            )
                            recommended_preset_button = gr.Button(
                                "標準\n公式Fast Preview相当",
                                size="sm",
                                interactive=False,
                                elem_id="h3-preset-recommended",
                            )
                            final_preset_button = gr.Button(
                                "高品質\nNative・重い",
                                size="sm",
                                interactive=False,
                                elem_id="h3-preset-final",
                            )
                        preset_state = gr.HTML(
                            value=_preset_state_html(initial_preset),
                            elem_id="h3-preset-state",
                        )
                        with gr.Row():
                            aspect = gr.Dropdown(
                                choices=["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                                value="16:9",
                                label="アスペクト比",
                                interactive=False,
                                elem_id="h3-aspect",
                            )
                            quality = gr.Dropdown(
                                choices=[(label, value) for value, label in QUALITY_LABELS.items()],
                                value=initial_quality,
                                label="解像度・速度",
                                interactive=False,
                                elem_id="h3-quality",
                            )
                        duration = gr.Slider(
                            minimum=5,
                            maximum=15,
                            step=0.5,
                            value=initial_duration,
                            label="長さ（秒）",
                            interactive=False,
                            elem_id="h3-duration",
                        )
                        acceleration_controls, acceleration_buttons = create_acceleration_controls(duration)
                        with gr.Accordion("Fun ControlNet · INT8 / 動きと構図", open=False, elem_id="h3-fun-control"):
                            control_mode = gr.Dropdown(
                                choices=[("オフ", "off"), ("元動画から輪郭を抽出 · Canny", "canny"), ("前処理済み動画 · Depth / Pose / HED / MLSDなど", "preprocessed")],
                                value="off", label="制御動画の使い方", interactive=False, elem_id="h3-control-mode",
                            )
                            control_video = gr.File(
                                label="制御動画", file_count="single", file_types=["video"], type="filepath",
                                interactive=False, elem_id="h3-control-video",
                            )
                            control_strength = gr.Slider(
                                0.05, 2.0, value=1.0, step=0.05, label="ControlNetの強さ",
                                interactive=False, elem_id="h3-control-strength",
                            )
                            gr.Markdown("INT8 ConvRot版（約2.30GB）を使用します。動画の先頭から生成する長さを24fpsで取り出し、中央を切り抜いて解像度を合わせます。短い動画は最後のフレームで補います。元の音声は参照しません。")
                        fun_control_controls = [control_mode, control_video, control_strength]
                        with gr.Row(elem_classes=["h3-generate-row"]):
                            generate_button = gr.Button(
                                "映像＋音声を生成",
                                variant="primary",
                                size="lg",
                                interactive=False,
                                elem_id="h3-generate",
                                elem_classes=["h3-generate-button"],
                            )
                            cancel_button = gr.Button(
                                "停止",
                                size="lg",
                                interactive=False,
                                elem_id="h3-cancel",
                                elem_classes=["h3-cancel-button"],
                            )
                        with gr.Accordion(
                            "プロンプト補助",
                            open=False,
                            elem_id="h3-prompt-assists",
                        ):
                            with gr.Row(elem_classes=["h3-prompt-chips"]):
                                camera_button = gr.Button("＋ カメラ", size="sm")
                                dialogue_button = gr.Button("＋ 台詞", size="sm")
                                sfx_button = gr.Button("＋ 効果音", size="sm")
                                music_button = gr.Button("＋ 音楽", size="sm")
                        settings_summary = gr.HTML(
                            value=initial_settings_summary,
                            elem_id="h3-settings-summary",
                        )

                        with gr.Accordion("詳細設定", open=False, elem_id="h3-advanced"):
                            with gr.Row():
                                steps = gr.Slider(
                                    minimum=1,
                                    maximum=40,
                                    step=1,
                                    value=initial_steps,
                                    label="Steps（生成ステップ）",
                                    interactive=False,
                                    elem_id="h3-steps",
                                )
                                seed = gr.Number(
                                    value=-1,
                                    precision=0,
                                    label="Seed（-1 = ランダム）",
                                    interactive=False,
                                    elem_id="h3-seed",
                                )
                            with gr.Row():
                                scheduler = gr.Dropdown(
                                    choices=[
                                        ("simple · 公式workflow推奨", "simple"),
                                        ("beta · 実験的", "beta"),
                                        ("normal · 実験的", "normal"),
                                    ],
                                    value=initial_scheduler,
                                    label="Scheduler（生成方式）",
                                    interactive=False,
                                    elem_id="h3-scheduler",
                                )
                                ref_image_size = gr.Radio(
                                    choices=[
                                        ("Match · 推奨・高速", "match"),
                                        ("Max · 同一性優先・非常に重い", "max"),
                                    ],
                                    value=initial_ref_image_size,
                                    label="参照画像サイズ",
                                    interactive=False,
                                    visible="hidden",
                                    elem_id="h3-ref-image-size",
                                )

                    gr.HTML(
                        """
                        <nav class="h3-mobile-actions" aria-label="MiniMax H3 生成操作">
                          <button type="button" id="h3-mobile-generate-proxy">映像＋音声を生成</button>
                          <button type="button" id="h3-mobile-cancel-proxy">停止</button>
                        </nav>
                        """,
                        elem_id="h3-mobile-action-bar",
                    )

                with gr.Column(scale=5, min_width=390, elem_classes=["h3-output"]):
                    gr.HTML(
                        '<div class="h3-section-kicker"><h3>生成結果</h3></div>'
                    )
                    result_video = gr.Video(
                        label="MiniMax H3 生成結果",
                        show_label=False,
                        interactive=False,
                        elem_id="h3-result-video",
                        elem_classes=["h3-video-stage"],
                    )
                    progress = gr.HTML(
                        value=progress_html(
                            "idle",
                            "生成時にH3 backendを確認し、未起動なら自動起動します",
                            0.0,
                        ),
                        elem_id="h3-progress",
                    )
                    gr.HTML(
                        '<p class="h3-sr-only" role="status" aria-live="polite" '
                        'aria-atomic="true"></p>',
                        elem_id="h3-progress-announcer",
                    )
                    gr.HTML(
                        '<p class="h3-output-note">映像と32kHzステレオ音声は同じ推論から生成され、MP4へ同期保存されます。</p>'
                    )

                    with gr.Accordion(
                        "実行環境とモデル", open=installed_runtime_root(Path(script_path)) is None,
                        elem_id="h3-runtime-setup",
                    ):
                        shared_models = gr.Textbox(
                            value=shared_models_value, interactive=False,
                            label="共有するモデルフォルダー（任意）",
                            placeholder="空欄ならNeo内にダウンロード",
                            elem_id="h3-shared-models",
                        )
                        shared_models.do_not_save_to_config = True
                        setup_button = gr.Button("環境とモデルを準備", elem_id="h3-setup", interactive=False)
                        setup_status = gr.Markdown(
                            setup_message,
                            elem_id="h3-setup-status",
                        )
                        runtime_profile = gr.Radio(
                            choices=[
                                (
                                    "高速・推奨 · Pinned Memory + Async 2",
                                    RUNTIME_PROFILE_FAST,
                                ),
                                (
                                    "省RAM・低速 · cacheなし + Pinned/Async無効",
                                    RUNTIME_PROFILE_LOW_RAM,
                                ),
                            ],
                            value=initial_runtime_profile,
                            label="起動プロファイル",
                            interactive=False,
                            elem_id="h3-runtime-profile",
                        )
                        gr.Markdown(
                            "profile変更は「選択設定で再起動」を押すと反映されます。"
                            "外部起動processは自動停止しません。"
                        )
                        with gr.Row():
                            runtime_path = gr.Textbox(
                                value=runtime_value,
                                visible=False, interactive=False,
                                elem_id="h3-runtime-path",
                            )
                            server_url = gr.Textbox(
                                value=H3_SERVER_URL,
                                label="ローカルBackend URL",
                                visible=False, interactive=False,
                                elem_id="h3-server-url",
                            )
                            runtime_path.do_not_save_to_config = True
                            server_url.do_not_save_to_config = True
                        with gr.Row():
                            connect_button = gr.Button(
                                "接続 / 起動",
                                variant="primary",
                                interactive=False,
                                elem_id="h3-connect",
                            )
                            restart_button = gr.Button(
                                "選択設定で再起動",
                                interactive=False,
                                elem_id="h3-restart",
                            )
                            rescan_button = gr.Button(
                                "状態を再確認",
                                interactive=False,
                                elem_id="h3-rescan",
                            )
                        gr.HTML(
                            """
                            <div class="h3-license-note">
                              <strong>Local Base · 768p</strong>
                              <span>ローカル公開weightはBaseです。Context-IRと2K Regenerateは外部の有料API専用で、この画面からは呼びません。</span>
                              <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">MiniMax H3 Community License</a>
                              <a href="https://docs.comfy.org/tutorials/video/minimax/minimax-h3" target="_blank" rel="noopener noreferrer">Official ComfyUI guide</a>
                            </div>
                            """
                        )

                    with gr.Accordion("最近の生成", open=True, elem_id="h3-history-accordion"):
                        history_panel = gr.HTML(value=initial_history_html, elem_id="h3-history")
                        history_selector = gr.Dropdown(
                            choices=initial_history_choices,
                            value=initial_history_value,
                            label="履歴を選択",
                            show_label=False,
                            elem_id="h3-history-selector",
                        )
                        with gr.Row():
                            load_history_button = gr.Button(
                                "動画を表示",
                                size="sm",
                                elem_id="h3-history-load",
                            )
                            restore_history_button = gr.Button(
                                "設定を復元",
                                size="sm",
                                interactive=False,
                                elem_id="h3-history-restore",
                            )
                            refresh_history_button = gr.Button(
                                "更新",
                                size="sm",
                                elem_id="h3-history-refresh",
                            )

            prompt_id_state = gr.State("")
            initialize_trigger = gr.Button(
                "H3 Studioを初期化",
                elem_id="h3-initialize-trigger",
            )

        initialize_trigger.click(
            fn=_initial_ui_with_acceleration,
            inputs=[runtime_path, server_url, runtime_profile, aspect],
            outputs=[
                runtime_status,
                runtime_profile,
                aspect,
                quality,
                duration,
                steps,
                seed,
                scheduler,
                ref_image_size,
                settings_summary,
                preset_state,
                quick_preset_button,
                recommended_preset_button,
                final_preset_button,
                restore_history_button,
                generate_button,
                runtime_path,
                server_url,
                connect_button,
                restart_button,
                rescan_button,
                *acceleration_controls,
                *acceleration_buttons,
                *fun_control_controls,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
        )

        summary_inputs = [aspect, quality, duration, steps, scheduler, ref_image_size]
        acceleration_controls[-5].change(
            lambda enabled: (gr.update(visible=not enabled), gr.update(label="各区間の終了構図" if enabled else "終了フレーム")),
            inputs=[acceleration_controls[-5]], outputs=[settings_summary, last_frame], queue=False, show_progress="hidden",
        )
        mode.change(
            fn=_mode_updates,
            inputs=[mode, *summary_inputs, input_validation],
            outputs=[
                keyframe_group,
                reference_group,
                mode_help,
                ref_image_size,
                settings_summary,
                input_validation,
            ],
            queue=False,
        )
        prompt.blur(
            fn=_clear_prompt_validation,
            inputs=[prompt, input_validation],
            outputs=[input_validation],
            queue=False,
            trigger_mode="always_last",
        )
        for component in [first_frame, last_frame]:
            component.change(
                fn=_clear_keyframe_validation,
                inputs=[first_frame, last_frame, input_validation],
                outputs=[input_validation],
                queue=False,
            )
        reference_inputs = [reference_images, reference_videos, reference_audios]
        for component in reference_inputs:
            component.change(
                fn=_reference_guide_updates,
                inputs=[*reference_inputs, input_validation],
                outputs=[reference_guide, input_validation],
                queue=False,
            )
        prompt_action_outputs = [prompt, input_validation]
        for button, callback in [
            (prompt_template_button, _prompt_template_updates),
            (camera_button, _prompt_camera_updates),
            (dialogue_button, _prompt_dialogue_updates),
            (sfx_button, _prompt_sfx_updates),
            (music_button, _prompt_music_updates),
        ]:
            button.click(
                fn=callback,
                inputs=[prompt, input_validation],
                outputs=prompt_action_outputs,
                queue=False,
            )

        setting_inputs = [aspect, quality, duration, steps, seed, scheduler, ref_image_size]
        aspect.input(
            fn=_aspect_settings_updates,
            inputs=[*setting_inputs, input_validation],
            outputs=[settings_summary, input_validation],
            queue=False,
            trigger_mode="always_last",
        )

        preset_outputs = [
            quality,
            duration,
            steps,
            scheduler,
            ref_image_size,
            settings_summary,
            preset_state,
        ]
        quick_preset_button.click(
            fn=_apply_quick_preset,
            inputs=[aspect],
            outputs=preset_outputs,
            queue=False,
            api_name="h3_apply_quick_preset",
        )
        recommended_preset_button.click(
            fn=_apply_recommended_preset,
            inputs=[aspect],
            outputs=preset_outputs,
            queue=False,
            api_name="h3_apply_recommended_preset",
        )
        final_preset_button.click(
            fn=_apply_final_preset,
            inputs=[aspect],
            outputs=preset_outputs,
            queue=False,
            api_name="h3_apply_final_preset",
        )
        for component in [quality, scheduler, ref_image_size]:
            component.input(
                fn=_custom_settings_updates,
                inputs=[*setting_inputs, input_validation],
                outputs=[settings_summary, preset_state, input_validation],
                queue=False,
                trigger_mode="always_last",
            )
        for component in [duration, steps]:
            component.input(
                fn=_custom_settings_updates,
                inputs=[*setting_inputs, input_validation],
                outputs=[settings_summary, preset_state, input_validation],
                queue=False,
                trigger_mode="always_last",
            )
        seed.change(
            fn=_clear_settings_validation,
            inputs=[*setting_inputs, input_validation],
            outputs=[input_validation],
            queue=False,
            trigger_mode="always_last",
        )

        interface.load(
            fn=_setup_state, outputs=[shared_models, setup_status, setup_button], queue=False,
            show_progress="hidden", api_visibility="private",
        )
        setup_button.click(
            fn=_setup_runtime,
            inputs=[shared_models],
            outputs=[runtime_path, setup_status, setup_button, shared_models],
            show_progress="hidden", trigger_mode="once", concurrency_limit=1,
            concurrency_id="h3-runtime-control", api_visibility="private",
        ).then(
            fn=_rescan_runtime,
            inputs=[runtime_path, server_url, runtime_profile, *acceleration_controls],
            outputs=[runtime_status], show_progress="hidden",
            concurrency_id="h3-runtime-control",
        )

        connect_button.click(
            fn=_connect_runtime_updates,
            inputs=[runtime_path, server_url, runtime_profile, *acceleration_controls],
            outputs=[runtime_status, connect_button, restart_button, rescan_button, runtime_profile],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
        )
        restart_button.click(
            fn=_restart_runtime_updates,
            inputs=[runtime_path, server_url, runtime_profile, *acceleration_controls],
            outputs=[runtime_status, connect_button, restart_button, rescan_button, runtime_profile],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
        )
        rescan_button.click(
            fn=_rescan_runtime_updates,
            inputs=[runtime_path, server_url, runtime_profile, *acceleration_controls],
            outputs=[runtime_status, connect_button, restart_button, rescan_button, runtime_profile],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="h3-runtime-control",
        )
        runtime_profile.input(
            fn=_runtime_profile_pending,
            inputs=[runtime_profile],
            outputs=[runtime_status],
            queue=False,
            trigger_mode="always_last",
        )
        refresh_history_button.click(
            fn=_refresh_history,
            inputs=[runtime_path, history_selector],
            outputs=[history_panel, history_selector],
            queue=False,
        )
        load_history_button.click(
            fn=_load_history_video,
            inputs=[history_selector, runtime_path],
            outputs=[result_video, progress],
        )
        restore_history_button.click(
            fn=_restore_history_with_acceleration,
            inputs=[history_selector, runtime_path],
            outputs=[
                mode,
                prompt,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                keyframe_group,
                reference_group,
                mode_help,
                reference_guide,
                aspect,
                quality,
                duration,
                steps,
                seed,
                scheduler,
                ref_image_size,
                settings_summary,
                preset_state,
                input_validation,
                progress,
                *acceleration_controls,
                *fun_control_controls,
            ],
            queue=False,
            show_progress="hidden",
        )

        for control in acceleration_controls:
            control.change(
                _acceleration_pending, inputs=acceleration_controls, outputs=[runtime_status],
                queue=False, show_progress="hidden", trigger_mode="always_last",
            )
        turbo_inputs = [aspect, quality, duration, ref_image_size]
        turbo_outputs = [acceleration_controls[0], steps, scheduler, settings_summary, preset_state]
        for button, callback in zip(acceleration_buttons[:2], [_turbo_four_updates, _turbo_eight_updates]):
            button.click(callback, inputs=turbo_inputs, outputs=turbo_outputs, queue=False, show_progress="hidden")
        acceleration_buttons[2].click(
            _reset_acceleration_updates, inputs=turbo_inputs,
            outputs=[*acceleration_controls, steps, scheduler, settings_summary, preset_state],
            queue=False, show_progress="hidden",
        )

        generate_button.click(
            fn=_generate,
            inputs=[
                runtime_path,
                server_url,
                runtime_profile,
                mode,
                prompt,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                aspect,
                quality,
                duration,
                steps,
                seed,
                scheduler,
                ref_image_size,
                *fun_control_controls,
                *acceleration_controls,
            ],
            outputs=[
                progress,
                result_video,
                prompt_id_state,
                history_panel,
                history_selector,
                cancel_button,
                generate_button,
                input_validation,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="minimax-h3-generation",
        )
        cancel_button.click(
            fn=_cancel,
            inputs=[prompt_id_state, server_url],
            outputs=[progress, prompt_id_state, cancel_button],
            queue=False,
        )

    return [(interface, "H3 Studio", "minimax_h3_studio")]


script_callbacks.on_ui_tabs(_build_ui, name="minimax_h3_studio")
