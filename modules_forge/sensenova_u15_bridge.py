"""Validated bridge between Forge Neo's UI and the SenseNova U1.5 worker."""

from __future__ import annotations

import atexit
import hashlib
import html
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from modules.aikimi_security.redaction import (
    redact_mapping,
    redact_text,
    sanitized_subprocess_environment,
)
from modules_forge.sensenova_u15_environment import WORKER_PYTHON, environment_status

MODE_TEXT = "text"
MODE_EDIT = "edit"
QUANT_INT8_CONVROT = "int8_convrot"
PROFILE_QUALITY = "quality"
PROFILE_OFFICIAL_8STEP = "official_8step"

DEFAULT_MODEL_ID = "sensenova/SenseNova-U1.5-8B-MoT"
SOURCE_REVISION = "e6dfd45762eb46f805067fe079c14bcb643ccccd"
CHECKPOINT_REVISION = "57de22ad4e2fc24c77f56dfe45dbb87a60dfebee"
CONVROT_FILE_NAME = "SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors"
CONVROT_EXPECTED_BYTES = 17_734_813_848
CONVROT_SHA256 = "cf6ed9ee3be516612b7fe083edfc7c9dd5d059cc759e300d2cf1f2726c0d250e"
EXPECTED_CONVROT_LAYERS = 588
OFFICIAL_LORA_REVISION = "e909f4636d119d65fe4cba8770c19daff2ac102e"
OFFICIAL_LORA_FILE_NAME = "SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"
OFFICIAL_LORA_EXPECTED_BYTES = 814_867_236
OFFICIAL_LORA_SHA256 = (
    "3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708"
)
EXPECTED_LORA_TARGETS = 294

MAX_REFERENCE_IMAGES = 64
MAX_PROMPT_LENGTH = 20_000
MAX_SOURCE_PIXELS = 100_000_000
GRID_FACTOR = 32
MIN_OUTPUT_SIDE = 512
MAX_OUTPUT_SIDE = 4096
MIN_PIXEL_BUDGET = 512 * 512
MAX_PIXEL_BUDGET = 2048 * 2048
LOW_VRAM_MAX_OUTPUT_PIXELS = 2048 * 2048
LOW_VRAM_MAX_INPUT_PIXELS = 512 * 512
LOW_VRAM_MAX_REFERENCE_IMAGES = 2
REFERENCE_MAX_IMAGES = 8
REFERENCE_MAX_INPUT_PIXELS = 1024 * 1024

VRAM_MODES = ("low", "reference", "unrestricted", "full")
ATTENTION_BACKENDS = ("auto", "sdpa", "flash")
DTYPES = ("bfloat16",)
GENERATION_PROFILES = (PROFILE_QUALITY, PROFILE_OFFICIAL_8STEP)
AUTO_RESOLUTION_PIXELS = {"auto": 2048 * 2048, "auto_1mp": 1024 * 1024}

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "2048x2048": (2048, 2048),
    "2720x1536": (2720, 1536),
    "1536x2720": (1536, 2720),
    "2496x1664": (2496, 1664),
    "1664x2496": (1664, 2496),
    "2368x1760": (2368, 1760),
    "1760x2368": (1760, 2368),
    "1440x2880": (1440, 2880),
    "2880x1440": (2880, 1440),
    "1152x3456": (1152, 3456),
    "3456x1152": (3456, 1152),
}

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = _ROOT / "models" / "SenseNova-U1" / "runtime-final"
DEFAULT_CHECKPOINT_PATH = _ROOT / "models" / "SenseNova-U1" / CONVROT_FILE_NAME
DEFAULT_LORA_PATH = _ROOT / "models" / "SenseNova-U1" / OFFICIAL_LORA_FILE_NAME
DEFAULT_WORKER_PATH = _ROOT / "tools" / "sensenova_u15_worker.py"

_PROCESS_LOCK = threading.RLock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None
_ACTIVE_JOB_ID: str | None = None
_CANCELLED_JOB_IDS: set[str] = set()
# Only populated when the generator has exited but its worker has not.
_PENDING_CLEANUP: tuple[str, Path, Path] | None = None


def _shutdown_active_worker() -> None:
    with _PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


atexit.register(_shutdown_active_worker)


class SenseNovaBridgeError(RuntimeError):
    pass


class SenseNovaGenerationCancelled(SenseNovaBridgeError):
    pass


@dataclass(frozen=True)
class SenseNovaRequest:
    mode: str
    prompt: str
    generation_profile: str
    model_path: str = DEFAULT_MODEL_ID
    quantization: str = QUANT_INT8_CONVROT
    checkpoint: str = os.fspath(DEFAULT_CHECKPOINT_PATH)
    source_path: str = os.fspath(DEFAULT_SOURCE_PATH)
    input_images: tuple[Image.Image, ...] = ()
    width: int | None = 2048
    height: int | None = 2048
    target_pixels: int = 2048 * 2048
    input_max_pixels: int | str = 512 * 512
    steps: int = 50
    cfg_scale: float = 4.0
    img_cfg_scale: float = 1.0
    timestep_shift: float = 3.0
    seed: int = 42
    vram_mode: str = "low"
    attn_backend: str = "auto"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class RuntimeStatus:
    ready: bool
    source_ready: bool
    dependencies_ready: bool
    checkpoint_ready: bool
    source_path: Path
    checkpoint_path: Path | None
    messages: tuple[str, ...]
    partial_bytes: int = 0
    lora_ready: bool = False


def parse_resolution(value: str, mode: str) -> tuple[int | None, int | None]:
    normalized = str(value or "").strip()
    if normalized in AUTO_RESOLUTION_PIXELS:
        if mode != MODE_EDIT:
            raise SenseNovaBridgeError("自動解像度は画像編集でのみ使用できます。")
        return None, None
    if normalized not in RESOLUTIONS:
        raise SenseNovaBridgeError(f"未対応の解像度です: {normalized or '(empty)'}")
    return RESOLUTIONS[normalized]


def resolution_choices(mode: str) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    if mode == MODE_EDIT:
        choices.append(("元の入力1枚目を基準 · 約4MP · 2K出力優先", "auto"))
        choices.append(("元の入力1枚目を基準 · 約1MP · 試作・省VRAM（学習解像度外）", "auto_1mp"))
    choices.extend(
        [
            ("512 × 512 · 最小VRAM動作確認", "512x512"),
            ("1024 × 1024 · 省VRAM（学習解像度外）", "1024x1024"),
            ("2048 × 2048 · 1:1 · 24GB Safe実測済み", "2048x2048"),
            ("2720 × 1536 · 16:9 公式", "2720x1536"),
            ("1536 × 2720 · 9:16 公式", "1536x2720"),
            ("2496 × 1664 · 3:2 公式", "2496x1664"),
            ("1664 × 2496 · 2:3 公式", "1664x2496"),
            ("2368 × 1760 · 4:3 公式", "2368x1760"),
            ("1760 × 2368 · 3:4 公式", "1760x2368"),
            ("1440 × 2880 · 1:2 公式", "1440x2880"),
            ("2880 × 1440 · 2:1 公式", "2880x1440"),
            ("1152 × 3456 · 1:3 公式", "1152x3456"),
            ("3456 × 1152 · 3:1 公式", "3456x1152"),
        ]
    )
    return choices


def _validate_reference_size(image: Image.Image) -> None:
    if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_SOURCE_PIXELS:
        raise SenseNovaBridgeError("参照画像の解像度が大きすぎます。最大100MPです。")


def _flatten_rgb(image: Image.Image) -> Image.Image:
    _validate_reference_size(image)
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _image_from_gallery_item(item: Any) -> Image.Image:
    if isinstance(item, (tuple, list)) and item:
        item = item[0]
    if isinstance(item, dict):
        item = item.get("image") or item.get("path") or item.get("name")
    if isinstance(item, Image.Image):
        _validate_reference_size(item)
        return _flatten_rgb(item)
    if isinstance(item, np.ndarray):
        return _flatten_rgb(Image.fromarray(item))
    if isinstance(item, (str, os.PathLike)):
        path = Path(item)
        if not path.is_file():
            raise SenseNovaBridgeError(f"参照画像が見つかりません: {path}")
        with Image.open(path) as opened:
            _validate_reference_size(opened)
            return _flatten_rgb(opened)
    raise SenseNovaBridgeError(f"参照画像の形式を読み取れません: {type(item).__name__}")


def normalize_gallery_images(value: Any) -> tuple[Image.Image, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, (list, tuple)) else [value]
    items = [item for item in items if item is not None]
    if len(items) > MAX_REFERENCE_IMAGES:
        raise SenseNovaBridgeError(f"参照画像は最大{MAX_REFERENCE_IMAGES}枚です。")
    return tuple(_image_from_gallery_item(item) for item in items)


def validate_request(request: SenseNovaRequest) -> None:
    if request.mode not in {MODE_TEXT, MODE_EDIT}:
        raise SenseNovaBridgeError(f"未対応の生成モードです: {request.mode!r}")
    prompt = request.prompt.strip()
    if not prompt:
        raise SenseNovaBridgeError("プロンプトを入力してください。")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise SenseNovaBridgeError(
            f"プロンプトは{MAX_PROMPT_LENGTH:,}文字以内にしてください。"
        )
    if not request.model_path.strip():
        raise SenseNovaBridgeError(
            "モデルIDまたはローカルモデルパスを入力してください。"
        )
    if request.quantization != QUANT_INT8_CONVROT:
        raise SenseNovaBridgeError(f"未対応の量子化方式です: {request.quantization!r}")
    if request.model_path.strip() != DEFAULT_MODEL_ID:
        raise SenseNovaBridgeError(
            f"INT8 ConvRotのconfigは正式版 {DEFAULT_MODEL_ID} に固定されています。"
        )
    if request.generation_profile not in GENERATION_PROFILES:
        raise SenseNovaBridgeError(
            "生成プロファイルは quality または official_8step を指定してください。"
        )
    if request.generation_profile == PROFILE_OFFICIAL_8STEP:
        if request.mode != MODE_TEXT:
            raise SenseNovaBridgeError(
                "公式8-Step LoRAはテキスト生成専用です。画像編集ではQuality 50-Stepを使ってください。"
            )
        if int(request.steps) != 8 or float(request.cfg_scale) != 1.0:
            raise SenseNovaBridgeError(
                "公式8-StepはSteps 8・CFG 1.0の固定プリセットです。"
            )
        if float(request.timestep_shift) != 3.0:
            raise SenseNovaBridgeError(
                "公式8-StepはTimestep Shift 3.0の固定プリセットです。"
            )
        if not _official_lora_is_ready(DEFAULT_LORA_PATH):
            raise SenseNovaBridgeError(
                "公式8-Step LoRAが未準備です。download_sensenova_u15_int8.batを再実行してください。"
            )
    if not request.checkpoint.strip():
        raise SenseNovaBridgeError("INT8 ConvRot checkpointを指定してください。")
    if Path(request.checkpoint).suffix.lower() != ".safetensors":
        raise SenseNovaBridgeError(
            "INT8 ConvRotには拡張子 .safetensors のcheckpointが必要です。"
        )

    if request.mode == MODE_EDIT:
        if not request.input_images:
            raise SenseNovaBridgeError(
                "画像編集には参照画像を1枚以上追加してください。"
            )
        if len(request.input_images) > MAX_REFERENCE_IMAGES:
            raise SenseNovaBridgeError(f"参照画像は最大{MAX_REFERENCE_IMAGES}枚です。")
    elif request.input_images:
        raise SenseNovaBridgeError(
            "テキスト生成では参照画像を使用しません。画像編集へ切り替えてください。"
        )

    if (request.width is None) != (request.height is None):
        raise SenseNovaBridgeError(
            "幅と高さは両方指定するか、両方を自動にしてください。"
        )
    if request.width is None and request.mode != MODE_EDIT:
        raise SenseNovaBridgeError("テキスト生成では出力解像度を指定してください。")
    if request.width is not None:
        for name, value in (("幅", request.width), ("高さ", request.height)):
            if (
                value is None
                or value < MIN_OUTPUT_SIDE
                or value > MAX_OUTPUT_SIDE
                or value % GRID_FACTOR
            ):
                raise SenseNovaBridgeError(
                    f"{name}は{MIN_OUTPUT_SIDE}〜{MAX_OUTPUT_SIDE}の32の倍数にしてください。"
                )
    if not MIN_PIXEL_BUDGET <= int(request.target_pixels) <= MAX_PIXEL_BUDGET:
        raise SenseNovaBridgeError(
            "自動出力の画素数は512²〜2048²の範囲にしてください。"
        )
    input_budget: int | None = None
    if request.input_max_pixels != "auto":
        try:
            input_budget = int(request.input_max_pixels)
        except (TypeError, ValueError) as exc:
            raise SenseNovaBridgeError(
                "入力画像予算は auto または整数で指定してください。"
            ) from exc
        if not MIN_PIXEL_BUDGET <= input_budget <= MAX_PIXEL_BUDGET:
            raise SenseNovaBridgeError(
                "入力画像予算は512²〜2048²の範囲にしてください。"
            )
    if not 1 <= int(request.steps) <= 100:
        raise SenseNovaBridgeError("Stepsは1〜100にしてください。")
    if not 0.0 <= float(request.cfg_scale) <= 20.0:
        raise SenseNovaBridgeError("CFGは0〜20にしてください。")
    if not 0.0 <= float(request.img_cfg_scale) <= 20.0:
        raise SenseNovaBridgeError("Image CFGは0〜20にしてください。")
    if not 0.1 <= float(request.timestep_shift) <= 20.0:
        raise SenseNovaBridgeError("Timestep Shiftは0.1〜20にしてください。")
    if not 0 <= int(request.seed) <= 2**32 - 1:
        raise SenseNovaBridgeError("Seedは0〜4,294,967,295にしてください。")
    if request.vram_mode not in VRAM_MODES:
        raise SenseNovaBridgeError(
            f"VRAMモードは {', '.join(VRAM_MODES)} から選択してください。"
        )
    if request.vram_mode == "low":
        output_pixels = (
            int(request.target_pixels)
            if request.width is None
            else int(request.width) * int(request.height or 0)
        )
        low_vram_errors: list[str] = []
        if output_pixels > LOW_VRAM_MAX_OUTPUT_PIXELS:
            low_vram_errors.append(
                f"出力{request.width or '自動'}×{request.height or '自動'}"
            )
        if request.mode == MODE_EDIT:
            if len(request.input_images) > LOW_VRAM_MAX_REFERENCE_IMAGES:
                low_vram_errors.append(f"参照{len(request.input_images)}枚")
            if input_budget is None or input_budget > LOW_VRAM_MAX_INPUT_PIXELS:
                budget_label = (
                    "自動"
                    if input_budget is None
                    else f"{round(input_budget**0.5):,}²/画像"
                )
                low_vram_errors.append(f"入力予算{budget_label}")
        if low_vram_errors:
            details = "、".join(low_vram_errors)
            raise SenseNovaBridgeError(
                "Low（RTX 3090・24GB安全）では、出力を2048²以下、参照画像を2枚以下、"
                "参照画像の予算を1枚あたり約0.26MPにしてください。"
                f"現在の超過条件: {details}。大きい条件はUncapped streamingを選択してください。"
            )
    if request.vram_mode == "reference":
        pixels = request.target_pixels if request.width is None else request.width * request.height
        if pixels > LOW_VRAM_MAX_OUTPUT_PIXELS:
            raise SenseNovaBridgeError("参照優先では出力を2048²以下にしてください。")
        if request.mode == MODE_EDIT and (
            len(request.input_images) > REFERENCE_MAX_IMAGES or input_budget is None or input_budget > REFERENCE_MAX_INPUT_PIXELS
        ):
            raise SenseNovaBridgeError("参照優先では参照8枚まで、各画像の予算を約1.05MP以下にしてください。")
    if request.attn_backend not in ATTENTION_BACKENDS:
        raise SenseNovaBridgeError("Attention backendの指定が不正です。")
    if request.dtype not in DTYPES:
        raise SenseNovaBridgeError("計算精度の指定が不正です。")


def _convrot_safetensors_header_is_valid(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                return False
            header_length = int.from_bytes(length_bytes, "little")
            if header_length <= 2 or header_length > 256 * 1024 * 1024:
                return False
            header = stream.read(header_length)
        return (
            len(header) == header_length
            and b".comfy_quant" in header
            and header.count(b'.comfy_quant"') == EXPECTED_CONVROT_LAYERS
            and b"fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
            in header
        )
    except OSError:
        return False


def _official_lora_header_is_valid(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                return False
            header_length = int.from_bytes(length_bytes, "little")
            if header_length <= 2 or header_length > 256 * 1024 * 1024:
                return False
            header = stream.read(header_length)
        return (
            len(header) == header_length
            and b'"tensor_kind":"neo_hf_lora"' in header.replace(b" ", b"")
            and header.count(b'.lora_down.weight"') == EXPECTED_LORA_TARGETS
            and header.count(b'.lora_up.weight"') == EXPECTED_LORA_TARGETS
            and header.count(b'.alpha"') == EXPECTED_LORA_TARGETS
        )
    except OSError:
        return False


def _official_lora_is_ready(path: Path = DEFAULT_LORA_PATH) -> bool:
    sidecar = Path(str(path) + ".sha256")
    try:
        sidecar_sha256 = (
            sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
            if sidecar.is_file()
            else ""
        )
        return (
            path.is_file()
            and path.stat().st_size == OFFICIAL_LORA_EXPECTED_BYTES
            and _official_lora_header_is_valid(path)
            and sidecar_sha256 == OFFICIAL_LORA_SHA256
        )
    except (OSError, IndexError):
        return False


def inspect_runtime(
    source_path: str | os.PathLike[str] = DEFAULT_SOURCE_PATH,
    *,
    checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT_PATH,
) -> RuntimeStatus:
    source = Path(source_path).expanduser().resolve()
    checkpoint_path = (
        Path(checkpoint).expanduser().resolve()
        if str(checkpoint).strip()
        else None
    )
    messages: list[str] = []
    package_init = source / "SenseNova" / "src" / "sensenova_u1" / "__init__.py"
    inference_file = source / "SenseNova" / "examples" / "editing" / "inference.py"
    config_file = source / "SenseNova-U1.5-8B-MoT" / "config.json"
    revision_file = source / ".sensenova_runtime_revision"
    revision = (
        revision_file.read_text(encoding="utf-8").strip()
        if revision_file.is_file()
        else ""
    )
    source_ready = (
        package_init.is_file()
        and inference_file.is_file()
        and config_file.is_file()
        and revision == SOURCE_REVISION
    )
    if source_ready:
        messages.append("正式版ConvRotランタイムは固定リビジョンで準備済みです。")
    elif package_init.is_file():
        messages.append(
            "推論コードのリビジョンが一致しません。セットアップを再実行してください。"
        )
    else:
        messages.append(
            "正式版ランタイムが未準備です。download_sensenova_u15_int8.bat を実行してください。"
        )

    dependencies_ready, dependency_message = environment_status()
    messages.append(dependency_message)

    partial_bytes = 0
    partial = Path(str(checkpoint_path) + ".part") if checkpoint_path else None
    if partial and partial.is_file():
        partial_bytes = partial.stat().st_size
    chunk_directory = (
        Path(str(checkpoint_path) + ".part.chunks") if checkpoint_path else None
    )
    if chunk_directory and chunk_directory.is_dir():
        chunk_bytes = sum(
            path.stat().st_size
            for pattern in ("chunk-*.bin", "chunk-*.bin.resume")
            for path in chunk_directory.glob(pattern)
            if path.is_file()
        )
        partial_bytes = max(partial_bytes, chunk_bytes)
    sidecar = Path(str(checkpoint_path) + ".sha256") if checkpoint_path else None
    try:
        sidecar_sha256 = (
            sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
            if sidecar and sidecar.is_file()
            else ""
        )
    except (OSError, IndexError):
        sidecar_sha256 = ""
    checkpoint_ready = bool(
        checkpoint_path
        and checkpoint_path.is_file()
        and checkpoint_path.stat().st_size == CONVROT_EXPECTED_BYTES
        and _convrot_safetensors_header_is_valid(checkpoint_path)
        and sidecar_sha256 == CONVROT_SHA256
    )
    if checkpoint_ready:
        messages.append("正式版INT8 ConvRot checkpointを確認しました。")
    elif partial_bytes:
        messages.append(
            "INT8 ConvRotをダウンロード中または中断中です: "
            f"{partial_bytes / (1024**3):.2f} / {CONVROT_EXPECTED_BYTES / (1024**3):.2f} GiB"
        )
    elif checkpoint_path and checkpoint_path.is_file():
        messages.append(
            "INT8 ConvRotの完全性記録が一致しません。セットアップでSHA-256を再検証してください。"
        )
    else:
        messages.append("正式版INT8 ConvRotが未準備です。セットアップを実行してください。")

    lora_ready = _official_lora_is_ready(DEFAULT_LORA_PATH)
    if lora_ready:
        messages.append("公式8-Step T2I LoRAを確認しました。")
    else:
        messages.append(
            "公式8-Step T2I LoRAは未準備です。Quality 50-Stepは利用できます。"
        )

    return RuntimeStatus(
        ready=source_ready and dependencies_ready and checkpoint_ready,
        source_ready=source_ready,
        dependencies_ready=dependencies_ready,
        checkpoint_ready=checkpoint_ready,
        source_path=source,
        checkpoint_path=checkpoint_path,
        messages=tuple(messages),
        partial_bytes=partial_bytes,
        lora_ready=lora_ready,
    )


def runtime_status_html(status: RuntimeStatus) -> str:
    state = "ready" if status.ready else "setup"
    title = (
        "生成できます"
        if status.ready and status.lora_ready
        else "Quality生成できます"
        if status.ready
        else "準備が必要です"
    )
    items = "".join(f"<li>{html.escape(message)}</li>" for message in status.messages)
    summary = (
        "Quality 50-Stepを使用できます。公式8-Step LoRAは未準備です。"
        if status.ready and not status.lora_ready
        else status.messages[0]
        if status.messages
        else "正式版INT8 ConvRotを使用できます。"
        if status.ready
        else "実行環境の設定を確認してください。"
    )
    return (
        f'<section class="sn-runtime sn-runtime-{state}" role="status">'
        '<div class="sn-runtime-summary">'
        f'<span class="sn-status-dot" aria-hidden="true"></span><strong>{title}</strong>'
        f'<p>{html.escape(summary)}</p></div>'
        '<details class="sn-runtime-details"><summary>詳細を開く</summary>'
        f"<ul>{items}</ul></details></section>"
    )


def progress_html(
    stage: str, message: str, progress: float, elapsed: float | None = None
) -> str:
    pct = max(0, min(100, round(float(progress) * 100)))
    elapsed_text = f" · {elapsed:.0f}秒" if elapsed is not None else ""
    safe_stage = html.escape(stage)
    safe_message = html.escape(message)
    return (
        f'<section class="sn-progress" data-stage="{safe_stage}" role="status" aria-live="polite">'
        f'<div class="sn-progress-head"><strong>{safe_message}</strong><span>{pct}%{elapsed_text}</span></div>'
        f'<div class="sn-progress-track"><i style="width:{pct}%"></i></div></section>'
    )


def reference_order_html(images: Sequence[Any], selected: int = -1) -> str:
    count = len(images or [])
    if not count:
        return '<p class="sn-reference-empty">画像編集では、参照画像を順番に追加してください。</p>'
    chips = []
    for index in range(1, count + 1):
        current = ' aria-current="true"' if index - 1 == selected else ""
        label = " · 選択中" if index - 1 == selected else ""
        chips.append(f"<span{current}><b>{index}</b> Image-{index}{label}</span>")
    return f'<div class="sn-reference-order"><strong>モデルへ渡す順序</strong><div>{"".join(chips)}</div></div>'


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _cleanup_job_directory(job_directory: Path, cache_root: Path) -> None:
    if not job_directory.exists():
        return
    expected_parent = (cache_root / "jobs").resolve()
    resolved = job_directory.resolve()
    if resolved.parent != expected_parent or not _is_within(resolved, cache_root):
        raise SenseNovaBridgeError(
            "一時jobの削除先が許可範囲外です。削除を中止しました。"
        )
    for attempt in range(3):
        try:
            shutil.rmtree(resolved)
            return
        except OSError as exc:
            if attempt == 2:
                print(
                    "SenseNova temporary cleanup did not complete "
                    f"({type(exc).__name__})."
                )
                return
            time.sleep(0.1 * (attempt + 1))


def _finish_pending_cleanup() -> None:
    """Reap a stopped worker left behind by a failed generator finalizer."""
    global _ACTIVE_JOB_ID, _ACTIVE_PROCESS, _PENDING_CLEANUP

    with _PROCESS_LOCK:
        pending = _PENDING_CLEANUP
        process = _ACTIVE_PROCESS
        if pending is None or (process is not None and process.poll() is None):
            return
        job_id, job_directory, cache_root = pending
        _PENDING_CLEANUP = None
        _ACTIVE_JOB_ID = None
        _ACTIVE_PROCESS = None
        _CANCELLED_JOB_IDS.discard(job_id)

    # A stream error must not prevent state recovery or removal of staged inputs.
    try:
        if process is not None and process.stdout is not None:
            process.stdout.close()
    finally:
        _cleanup_job_directory(job_directory, cache_root)


def _release_forge_vram() -> None:
    sd_models = sys.modules.get("modules.sd_models")
    if sd_models is None:
        # Standalone worker/tests have no in-process Forge model to release.
        # Avoid importing the complete WebUI loader solely for a no-op.
        return
    try:
        from modules import devices

        sd_models.unload_model_weights()
        devices.torch_gc()
    except Exception as exc:
        raise SenseNovaBridgeError(
            f"ForgeモデルのVRAM解放に失敗しました: {exc}"
        ) from exc


def _stage_images(images: Sequence[Image.Image], job_directory: Path) -> list[str]:
    staged: list[str] = []
    inputs_directory = job_directory / "inputs"
    inputs_directory.mkdir(parents=True, exist_ok=False)
    for index, image in enumerate(images, 1):
        path = inputs_directory / f"reference_{index:02d}.png"
        image.save(path, format="PNG", optimize=False)
        staged.append(os.fspath(path.resolve()))
    return staged


def _request_payload(
    request: SenseNovaRequest,
    *,
    input_paths: Sequence[str],
    output_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "mode": request.mode,
        "prompt": request.prompt.strip(),
        "model_path": request.model_path.strip(),
        "quantization": request.quantization,
        "checkpoint": os.fspath(Path(request.checkpoint).expanduser().resolve()),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CONVROT_SHA256,
        "source_path": os.fspath(Path(request.source_path).expanduser().resolve()),
        "input_images": list(input_paths),
        "width": request.width,
        "height": request.height,
        "target_pixels": int(request.target_pixels),
        "input_max_pixels": request.input_max_pixels,
        "generation_profile": request.generation_profile,
        "lora_path": (
            os.fspath(DEFAULT_LORA_PATH.resolve())
            if request.generation_profile == PROFILE_OFFICIAL_8STEP
            else ""
        ),
        "lora_revision": (
            OFFICIAL_LORA_REVISION
            if request.generation_profile == PROFILE_OFFICIAL_8STEP
            else ""
        ),
        "lora_sha256": (
            OFFICIAL_LORA_SHA256
            if request.generation_profile == PROFILE_OFFICIAL_8STEP
            else ""
        ),
        "steps": int(request.steps),
        "cfg_scale": float(request.cfg_scale),
        "img_cfg_scale": float(request.img_cfg_scale),
        "cfg_norm": "none",
        "timestep_shift": float(request.timestep_shift),
        "seed": int(request.seed),
        "vram_mode": request.vram_mode,
        "attn_backend": request.attn_backend,
        "dtype": request.dtype,
        "device": "cuda",
        "fast_vram_fraction": 0.90,
        "fast_vram_headroom_gib": 2.0,
        "fast_activation_reserve_gib": 4.0,
        "fast_vram_budget_gib": 0.0,
        "output_path": os.fspath(output_path.resolve()),
        "metadata_path": os.fspath(metadata_path.resolve()),
    }


def _parse_event(line: str) -> dict[str, Any] | None:
    prefix = "SENSENOVA_EVENT "
    if not line.startswith(prefix):
        return None
    try:
        payload = json.loads(line[len(prefix) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("stage"), str):
        return None
    return payload


def _read_process_output(
    process: subprocess.Popen[str], events: queue.Queue[str | None]
) -> None:
    assert process.stdout is not None
    try:
        for line in process.stdout:
            events.put(line.rstrip("\r\n"))
    finally:
        events.put(None)


def _validate_worker_result(output_path: Path, metadata_path: Path, request: SenseNovaRequest) -> dict[str, Any]:
    """workerの終了コードとは別に、画像と今回の生成条件の一致を確認する。"""
    try:
        if metadata_path.stat().st_size > 1024 * 1024:
            raise SenseNovaBridgeError("生成メタデータが大きすぎるため読み込めません。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise SenseNovaBridgeError("生成メタデータの形式が不正です。")
        expected = {
            "schema_version": 3,
            "mode": request.mode,
            "prompt": request.prompt.strip(),
            "seed": int(request.seed),
            "steps": int(request.steps),
            "generation_profile": request.generation_profile,
            "input_image_count": len(request.input_images),
        }
        for key, value in expected.items():
            if type(metadata.get(key)) is not type(value) or metadata[key] != value:
                raise SenseNovaBridgeError(f"生成結果の{key}が今回の生成条件と一致しません。")
        dimensions = (metadata.get("width"), metadata.get("height"))
        if any(
            type(value) is not int or not 32 <= value <= MAX_OUTPUT_SIDE or value % GRID_FACTOR for value in dimensions
        ):
            raise SenseNovaBridgeError("生成結果の解像度が不正です。")
        if request.width is not None and dimensions != (request.width, request.height):
            raise SenseNovaBridgeError("生成画像の解像度が指定と一致しません。")
        with Image.open(output_path) as image:
            if image.format != "PNG" or image.size != dimensions:
                raise SenseNovaBridgeError("生成PNGの形式または解像度がメタデータと一致しません。")
            image.verify()
        with Image.open(output_path) as image:
            image.load()
        with output_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if metadata.get("output_sha256") != digest:
            raise SenseNovaBridgeError("生成画像のSHA-256がメタデータと一致しません。")
    except SenseNovaBridgeError:
        raise
    except (OSError, ValueError, SyntaxError) as exc:
        raise SenseNovaBridgeError(f"生成結果を確認できません: {exc}") from exc
    return metadata


def run_generation(
    request: SenseNovaRequest,
    *,
    output_directory: str | os.PathLike[str],
    cache_directory: str | os.PathLike[str],
    log_directory: str | os.PathLike[str],
    worker_path: str | os.PathLike[str] = DEFAULT_WORKER_PATH,
) -> Iterator[dict[str, Any]]:
    global _ACTIVE_JOB_ID, _ACTIVE_PROCESS, _PENDING_CLEANUP

    _finish_pending_cleanup()
    validate_request(request)
    runtime = inspect_runtime(
        request.source_path,
        checkpoint=request.checkpoint,
    )
    if not runtime.ready:
        raise SenseNovaBridgeError(" ".join(runtime.messages))

    output_root = Path(output_directory).resolve()
    cache_root = Path(cache_directory).resolve()
    log_root = Path(log_directory).resolve()
    worker = Path(worker_path).resolve()
    if not worker.is_file():
        raise SenseNovaBridgeError(f"SenseNova workerが見つかりません: {worker}")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    job_directory = cache_root / "jobs" / job_id
    job_directory.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_root / f"sensenova_u15_{stamp}_{job_id[:8]}.png"
    metadata_path = output_path.with_suffix(".json")
    log_path = log_root / f"{stamp}_{job_id[:8]}.log"
    process: subprocess.Popen[str] | None = None

    with _PROCESS_LOCK:
        if _ACTIVE_JOB_ID is not None:
            _cleanup_job_directory(job_directory, cache_root)
            raise SenseNovaBridgeError(
                "別のSenseNova生成が実行中です。完了またはキャンセルを待ってください。"
            )
        _ACTIVE_JOB_ID = job_id
        _CANCELLED_JOB_IDS.discard(job_id)

    try:
        started = time.monotonic()
        yield {
            "stage": "prepare",
            "message": "入力画像と実行環境を確認しました",
            "progress": 0.02,
            "job_id": job_id,
        }
        if job_id in _CANCELLED_JOB_IDS:
            raise SenseNovaGenerationCancelled("SenseNova生成をキャンセルしました。")
        input_paths = _stage_images(request.input_images, job_directory)
        if job_id in _CANCELLED_JOB_IDS:
            raise SenseNovaGenerationCancelled("SenseNova生成をキャンセルしました。")
        payload = _request_payload(
            request,
            input_paths=input_paths,
            output_path=output_path,
            metadata_path=metadata_path,
        )
        request_path = job_directory / "request.json"
        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if job_id in _CANCELLED_JOB_IDS:
            raise SenseNovaGenerationCancelled("SenseNova生成をキャンセルしました。")
        _release_forge_vram()
        yield {
            "stage": "vram",
            "message": "Forgeモデルを退避し、VRAMを確保しました",
            "progress": 0.05,
            "job_id": job_id,
        }
        if job_id in _CANCELLED_JOB_IDS:
            raise SenseNovaGenerationCancelled("SenseNova生成をキャンセルしました。")

        environment = sanitized_subprocess_environment(os.environ)
        environment.pop("PYTHONUTF8", None)
        environment.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with _PROCESS_LOCK:
            # キャンセル確認、起動、所有processの登録を一つの操作にする。
            if job_id in _CANCELLED_JOB_IDS:
                raise SenseNovaGenerationCancelled("SenseNova生成をキャンセルしました。")
            process = subprocess.Popen(
                [
                    os.fspath(WORKER_PYTHON),
                    "-B",
                    os.fspath(worker),
                    "--request",
                    os.fspath(request_path),
                ],
                cwd=os.fspath(_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creationflags,
            )
            _ACTIVE_PROCESS = process

        event_queue: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(
            target=_read_process_output, args=(process, event_queue), daemon=True
        )
        reader.start()
        output_closed = False
        log_tail: list[str] = []
        last_event: dict[str, Any] = {
            "stage": "loading",
            "message": "SenseNova U1.5を読み込んでいます",
            "progress": 0.06,
        }
        last_yield = time.monotonic()

        with log_path.open("w", encoding="utf-8", newline="\n") as log_stream:
            while not output_closed or process.poll() is None:
                try:
                    line = event_queue.get(timeout=0.5)
                except queue.Empty:
                    line = ""
                if line is None:
                    output_closed = True
                elif line:
                    safe_line = redact_text(line)
                    log_stream.write(safe_line + "\n")
                    log_stream.flush()
                    log_tail.append(safe_line)
                    del log_tail[:-20]
                    event = _parse_event(line)
                    if event is not None:
                        event = redact_mapping(event)
                        last_event = event
                        if event["stage"] != "complete":
                            event["elapsed"] = time.monotonic() - started
                            event["job_id"] = job_id
                            yield event
                            last_yield = time.monotonic()

                now = time.monotonic()
                if (
                    process.poll() is None
                    and last_event.get("stage") != "complete"
                    and now - last_yield >= 5.0
                ):
                    heartbeat = dict(last_event)
                    heartbeat["elapsed"] = now - started
                    heartbeat["job_id"] = job_id
                    yield heartbeat
                    last_yield = now

        return_code = process.wait()
        if job_id in _CANCELLED_JOB_IDS:
            raise SenseNovaGenerationCancelled("SenseNova生成をキャンセルしました。")
        if return_code != 0:
            error_message = str(last_event.get("message") or "SenseNova worker failed.")
            details = [line for line in log_tail if not line.startswith("Traceback")][
                -4:
            ]
            if details:
                error_message += " / " + " | ".join(details)
            raise SenseNovaBridgeError(error_message)
        if not output_path.is_file() or not metadata_path.is_file():
            raise SenseNovaBridgeError(
                "workerは正常終了しましたが、出力画像またはメタデータがありません。"
            )
        if not _is_within(output_path, output_root) or not _is_within(
            metadata_path, output_root
        ):
            raise SenseNovaBridgeError("worker出力が許可された保存先の外にあります。")
        metadata = _validate_worker_result(output_path, metadata_path, request)
        yield {
            "stage": "complete",
            "message": "生成が完了しました",
            "progress": 1.0,
            "elapsed": time.monotonic() - started,
            "job_id": job_id,
            "path": os.fspath(output_path),
            "metadata_path": os.fspath(metadata_path),
            "metadata": redact_mapping(metadata),
        }
    finally:
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            with _PROCESS_LOCK:
                stopped = process is None or process.poll() is not None
                if stopped:
                    if _ACTIVE_JOB_ID == job_id:
                        _ACTIVE_JOB_ID = None
                    if _ACTIVE_PROCESS is process:
                        _ACTIVE_PROCESS = None
                    _CANCELLED_JOB_IDS.discard(job_id)
                else:
                    # Retain ownership and inputs until a later cancel/start
                    # confirms exit. Closing a live worker's pipe may also block.
                    _PENDING_CLEANUP = (job_id, job_directory, cache_root)
            if stopped:
                try:
                    if process is not None and process.stdout is not None:
                        process.stdout.close()
                finally:
                    _cleanup_job_directory(job_directory, cache_root)


def cancel_generation(job_id: str | None = None) -> str:
    _finish_pending_cleanup()
    with _PROCESS_LOCK:
        process = _ACTIVE_PROCESS
        active_job = _ACTIVE_JOB_ID
        if active_job is None:
            return "実行中のSenseNova生成はありません。"
        if job_id and job_id != active_job:
            return "指定された生成はすでに終了しています。"
        _CANCELLED_JOB_IDS.add(active_job)
        if process is None or process.poll() is not None:
            return "キャンセルを受け付けました。worker起動前に停止します。"
        process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    _finish_pending_cleanup()
    return "キャンセルを受け付けました。モデルworkerを停止しています。"


def request_summary_html(request: SenseNovaRequest) -> str:
    mode = "複数画像編集" if request.mode == MODE_EDIT else "テキスト生成"
    quantization = "正式版 · INT8 ConvRot"
    profile = {
        PROFILE_OFFICIAL_8STEP: "公式8-Step T2I",
        PROFILE_QUALITY: "Quality",
    }.get(
        request.generation_profile,
        f"UNKNOWN ({request.generation_profile})",
    )
    size = (
        f"元の入力1枚目を基準 · 約{request.target_pixels / 1_000_000:.1f}MP"
        if request.width is None
        else f"{request.width} × {request.height}"
    )
    vram_label = {
        "low": "24GB SAFE",
        "reference": "参照優先",
        "unrestricted": "UNCAPPED",
        "full": "FULL GPU",
    }.get(request.vram_mode, request.vram_mode.upper())
    return (
        '<div class="sn-summary">'
        f"<span><small>MODE</small>{html.escape(mode)}</span>"
        f"<span><small>WEIGHTS</small>{html.escape(quantization)}</span>"
        f"<span><small>OUTPUT</small>{html.escape(size)}</span>"
        f"<span><small>REFERENCES</small>{len(request.input_images)}枚</span>"
        f"<span><small>SAMPLING</small>{html.escape(profile)} · {html.escape(vram_label)} · {request.steps} steps · CFG {request.cfg_scale:g}</span>"
        "</div>"
    )
