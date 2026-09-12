from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import html
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import httpx

from modules.aikimi_security.redaction import sanitized_subprocess_environment
from modules_forge.gpu_ownership import GPUOwnership, release_forge_vram
from modules_forge.minimax_h3_runtime import SERVER_URL, installed_runtime_root, managed_runtime_root, model_root, setup_lock
from modules_forge.minimax_h3_acceleration import FAST_VAE_PACK, SPARSE_COMMIT, H3Acceleration
from modules_forge.minimax_h3_clipcache import CLIP_CACHE_PACK, CLIP_CACHE_REVISION, require_clipcache
from modules_forge import minimax_h3_negpip_cache as negpip_cache
from modules_forge import minimax_h3_fun_control as fun_control
from modules_forge import minimax_h3_hybrid as hybrid
from modules_forge.minimax_h3_negpip import NEGPIP_NODE, NEGPIP_PACK, custom_node_whitelist, install_bundled_negpip


H3_FPS = 24
H3_MIN_SECONDS = 5.0
H3_MAX_SECONDS = 15.0
H3_SERVER_URL = SERVER_URL
H3_FL_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_REF_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_MINIMUM_COMFY_COMMIT = "62b3c94bd45154f6486c7abf1b9efcacee96ea69"
H3_HISTORY_METADATA_MAX_BYTES = 1024 * 1024
H3_STATUS_POLL_MAX_FAILURES = 3

RUNTIME_PROFILE_FAST = "fast"
RUNTIME_PROFILE_LOW_RAM = "low_ram"
RUNTIME_PROFILES = {RUNTIME_PROFILE_FAST, RUNTIME_PROFILE_LOW_RAM}
RUNTIME_PROFILE_LABELS = {
    RUNTIME_PROFILE_FAST: "高速（Pinned Memory + Async 2）",
    RUNTIME_PROFILE_LOW_RAM: "省RAM（cacheなし + Pinned/Async無効）",
}

MODE_TEXT = "text"
MODE_KEYFRAMES = "keyframes"
MODE_REFERENCES = "references"
MODES = {MODE_TEXT, MODE_KEYFRAMES, MODE_REFERENCES}

ASPECTS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
QUALITY_DIMENSIONS: dict[str, dict[str, tuple[int, int]]] = {
    "draft": {
        "21:9": (672, 288),
        "16:9": (608, 352),
        "4:3": (512, 384),
        "1:1": (448, 448),
        "3:4": (384, 512),
        "9:16": (352, 608),
    },
    "preview": {
        "21:9": (960, 416),
        "16:9": (864, 480),
        "4:3": (736, 544),
        "1:1": (640, 640),
        "3:4": (544, 736),
        "9:16": (480, 864),
    },
    "balanced": {
        "21:9": (1120, 480),
        "16:9": (960, 544),
        "4:3": (832, 640),
        "1:1": (736, 736),
        "3:4": (640, 832),
        "9:16": (544, 960),
    },
    "native": {
        "21:9": (1344, 576),
        "16:9": (1344, 768),
        "4:3": (1024, 768),
        "1:1": (768, 768),
        "3:4": (768, 1024),
        "9:16": (768, 1344),
    },
}

REQUIRED_NODE_TYPES = {
    "BasicGuider",
    "BasicScheduler",
    "CLIPLoader",
    "CreateVideo",
    "GetVideoComponents",
    "KSamplerSelect",
    "LoadAudio",
    "LoadImage",
    "LoadVideo",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "ModelAttentionBackend",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "SaveVideo",
    "UNETLoader",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAELoader",
}

MODEL_FILES = {
    "FL2VA": ("diffusion_models", H3_FL_MODEL),
    "Ref2VA": ("diffusion_models", H3_REF_MODEL),
    "Qwen3-VL 32B": ("text_encoders", H3_TEXT_ENCODER),
    "Video VAE": ("vae", H3_VIDEO_VAE),
    "Audio VAE": ("vae", H3_AUDIO_VAE),
}

_MANAGED_PROCESS: subprocess.Popen | None = None
_MANAGED_PROCESS_IDENTITY: tuple[Path, str] | None = None
_PROCESS_LOCK = threading.Lock()
_RUNTIME_LIFECYCLE_LOCK = threading.RLock()
_RUNTIME_SETUP_ACTIVE = False
_CANCELLED_JOB_LOCK = threading.Lock()
_CANCELLED_JOB_IDS: set[str] = set()
_ACTIVE_GENERATION_LOCK = threading.Lock()
_ACTIVE_GENERATION_IDS: set[str] = set()
_GPU_OWNERSHIPS: dict[str, GPUOwnership] = {}
_LOG = logging.getLogger(__name__)
_TERMINAL_JOB_STATUSES = {"completed", "success", "failed", "error", "cancelled", "canceled"}


class H3BridgeError(RuntimeError):
    """A user-actionable MiniMax H3 bridge error."""


class H3GenerationCancelled(H3BridgeError):
    """The user intentionally stopped an in-flight MiniMax H3 generation."""


class H3JobNotFound(H3BridgeError):
    """A ComfyUI job endpoint no longer knows the requested prompt ID."""


class _WindowsPerformanceInformation(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    ]


def _local_commit_free_gib() -> float | None:
    if os.name != "nt":
        return None
    information = _WindowsPerformanceInformation()
    information.cb = ctypes.sizeof(information)
    try:
        get_performance_info = ctypes.windll.psapi.GetPerformanceInfo
        get_performance_info.argtypes = [
            ctypes.POINTER(_WindowsPerformanceInformation),
            ctypes.c_ulong,
        ]
        get_performance_info.restype = ctypes.c_int
        if not get_performance_info(ctypes.byref(information), information.cb):
            return None
    except (AttributeError, OSError, ValueError):
        return None
    free_pages = max(0, int(information.CommitLimit) - int(information.CommitTotal))
    return free_pages * int(information.PageSize) / 1024**3


def _mark_cancelled_job(prompt_id: str) -> None:
    with _CANCELLED_JOB_LOCK:
        _CANCELLED_JOB_IDS.add(prompt_id)


def _is_cancelled_job(prompt_id: str) -> bool:
    with _CANCELLED_JOB_LOCK:
        return prompt_id in _CANCELLED_JOB_IDS


def _clear_cancelled_job(prompt_id: str) -> None:
    with _CANCELLED_JOB_LOCK:
        _CANCELLED_JOB_IDS.discard(prompt_id)


def _mark_active_generation(prompt_id: str) -> None:
    with _ACTIVE_GENERATION_LOCK:
        _ACTIVE_GENERATION_IDS.add(prompt_id)


def _clear_active_generation(prompt_id: str) -> None:
    with _ACTIVE_GENERATION_LOCK:
        _ACTIVE_GENERATION_IDS.discard(prompt_id)


def _active_generation_count() -> int:
    with _ACTIVE_GENERATION_LOCK:
        return len(_ACTIVE_GENERATION_IDS)


@dataclass(frozen=True)
class H3Request:
    mode: str
    prompt: str
    aspect: str = "16:9"
    quality: str = "preview"
    duration_seconds: float = H3_MIN_SECONDS
    steps: int = 20
    seed: int = -1
    scheduler: str = "simple"
    ref_image_size: str = "match"
    first_frame: str | None = None
    last_frame: str | None = None
    reference_images: tuple[str, ...] = ()
    reference_videos: tuple[str, ...] = ()
    reference_audios: tuple[str, ...] = ()
    acceleration: H3Acceleration = field(default_factory=H3Acceleration)
    control: fun_control.H3FunControl = field(default_factory=fun_control.H3FunControl)
    control_video: str | None = None

    @property
    def dimensions(self) -> tuple[int, int]:
        return dimensions_for(self.aspect, self.quality)

    @property
    def frame_count(self) -> int:
        frames = snap_h3_frames(self.duration_seconds)
        return self.acceleration.hybrid.frame_count(frames) if self.acceleration.hybrid.enabled else frames

    @property
    def effective_seconds(self) -> float:
        return self.frame_count / H3_FPS

    @property
    def resolved_seed(self) -> int:
        if self.seed == -1:
            return secrets.randbelow(2**53)
        return self.seed


@dataclass(frozen=True)
class RuntimeReadiness:
    runtime_root: Path | None
    server_url: str
    connected: bool
    comfy_version: str | None = None
    gpu_name: str | None = None
    vram_gib: float | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    runtime_args: tuple[str, ...] = ()
    ck_attention_available: bool = False
    core_revision: str | None = None
    h3_core_optimized: bool = False
    runtime_profile: str | None = None
    ram_free_gib: float | None = None
    ram_total_gib: float | None = None
    commit_free_gib: float | None = None
    model_files: dict[str, bool] = field(default_factory=dict)
    server_model_files: dict[str, bool] = field(default_factory=dict)
    missing_nodes: tuple[str, ...] = ()
    error: str | None = None
    acceleration: H3Acceleration = field(default_factory=H3Acceleration)
    node_schemas: dict[str, Any] = field(default_factory=dict)

    @property
    def ready_for_fl2va(self) -> bool:
        required = ("FL2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")
        return (
            self.connected
            and not self.missing_nodes
            and self.ck_attention_available
            and self.h3_core_optimized
            and all(self.model_files.get(name) for name in required)
            and all(self.server_model_files.get(name) for name in required)
        )

    @property
    def ready_for_ref2va(self) -> bool:
        return (
            self.ready_for_fl2va
            and bool(self.model_files.get("Ref2VA"))
            and bool(self.server_model_files.get("Ref2VA"))
        )


@dataclass(frozen=True)
class HistoryItem:
    path: Path
    modified_at: float
    source: str
    size_bytes: int | None = None

    @property
    def public_id(self) -> str:
        """Return an opaque browser value without disclosing the local path."""

        normalized = os.path.normcase(os.fspath(self.path.resolve(strict=False)))
        return f"h3-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]}"

    @property
    def label(self) -> str:
        stamp = datetime.fromtimestamp(self.modified_at).strftime("%m/%d %H:%M")
        return f"{stamp} · {self.path.name}"


def snap_h3_frames(seconds: float) -> int:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise H3BridgeError("長さは秒数で指定してください。") from exc
    if not math.isfinite(seconds) or not H3_MIN_SECONDS <= seconds <= H3_MAX_SECONDS:
        raise H3BridgeError("MiniMax H3 の長さは 5〜15 秒で指定してください。")
    requested = max(5, int(seconds * H3_FPS + 0.5))
    return requested + (5 - requested % 17) % 17


def dimensions_for(aspect: str, quality: str) -> tuple[int, int]:
    if quality not in QUALITY_DIMENSIONS:
        raise H3BridgeError(f"未対応の品質プリセットです: {quality}")
    try:
        width, height = QUALITY_DIMENSIONS[quality][aspect]
    except KeyError as exc:
        raise H3BridgeError(f"未対応のアスペクト比です: {aspect}") from exc
    if width % 32 or height % 32:
        raise AssertionError("MiniMax H3 dimensions must be multiples of 32")
    return width, height


def normalize_file_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, os.PathLike)):
        return (os.fspath(value),)
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        elif hasattr(item, "name"):
            item = item.name
        if item:
            result.append(os.fspath(item))
    return tuple(result)


def validate_request(request: H3Request) -> None:
    try:
        if not isinstance(request.acceleration, H3Acceleration):
            raise ValueError("H3 高速化設定の形式が不正です。")
        request.acceleration.validate()
        if not isinstance(request.control, fun_control.H3FunControl):
            raise ValueError("Fun ControlNet設定の形式が不正です。")
        request.control.validate()
    except ValueError as exc:
        raise H3BridgeError(str(exc)) from exc
    if request.mode not in MODES:
        raise H3BridgeError("生成モードを選択してください。")
    if request.control.enabled and not request.control_video:
        raise H3BridgeError("Fun ControlNetの制御動画を追加してください。")
    if not request.prompt or not request.prompt.strip():
        raise H3BridgeError("映像と音のプロンプトを入力してください。")
    if len(request.prompt) > 20_000:
        raise H3BridgeError("プロンプトが長すぎます。20,000文字以内にしてください。")
    dimensions_for(request.aspect, request.quality)
    snap_h3_frames(request.duration_seconds)
    if request.acceleration.hybrid.enabled:
        try:
            request.acceleration.hybrid.frame_count(snap_h3_frames(request.duration_seconds))
        except ValueError as exc:
            raise H3BridgeError(str(exc)) from exc
        if request.control.enabled:
            raise H3BridgeError("長尺生成ではFun ControlNetをオフにしてください。")
        if request.reference_videos or request.reference_audios:
            raise H3BridgeError("長尺生成の参照素材には画像を指定してください。")
    try:
        steps = int(request.steps)
        seed = int(request.seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise H3BridgeError("Steps と Seed は有限の整数で指定してください。") from exc
    if not 1 <= steps <= 100:
        raise H3BridgeError("Steps は 1〜100 で指定してください。")
    if request.acceleration.hybrid.enabled and request.acceleration.hybrid.switch_step > steps:
        raise H3BridgeError("長尺の切替ステップはSteps以下にしてください。")
    if seed < -1 or seed >= 2**63:
        raise H3BridgeError("Seed は -1、または 0〜2^63-1 で指定してください。")
    if request.scheduler not in {"simple", "beta", "normal"}:
        raise H3BridgeError("Scheduler は simple / beta / normal から選択してください。")
    if request.ref_image_size not in {"match", "max"}:
        raise H3BridgeError("参照画像サイズは match または max を選択してください。")

    if request.mode == MODE_KEYFRAMES and not (request.first_frame or request.last_frame):
        raise H3BridgeError("キーフレームモードでは開始画像または終了画像を追加してください。")

    if request.mode == MODE_REFERENCES:
        image_count = len(request.reference_images)
        video_count = len(request.reference_videos)
        audio_count = len(request.reference_audios)
        if image_count > 9:
            raise H3BridgeError("参照画像は最大9枚です。")
        if video_count > 3:
            raise H3BridgeError("参照動画は最大3本です。")
        if audio_count > 3:
            raise H3BridgeError("参照音声は最大3本です。")
        if image_count + video_count + audio_count > 12:
            raise H3BridgeError("参照素材は合計12個までです。")
        if not image_count and not video_count:
            raise H3BridgeError("参照モードには画像または動画が必要です。音声だけでは生成できません。")


def normalize_loopback_url(url: str) -> str:
    parsed = urllib.parse.urlsplit((url or "").strip())
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise H3BridgeError("H3 backend は安全のため 127.0.0.1 / localhost の HTTP URL のみ利用できます。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise H3BridgeError("H3 backend URL に認証情報、query、fragmentは指定できません。")
    if parsed.path not in {"", "/"}:
        raise H3BridgeError("H3 backend URL にはパスを付けないでください。")
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise H3BridgeError("H3 backend のポート番号が不正です。") from exc
    return f"http://{parsed.hostname}:{port}"


def _local_filesystem_path(value: str | os.PathLike[str], label: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    normalized = expanded.replace("/", "\\")
    if normalized.startswith("\\\\"):
        raise H3BridgeError(f"{label}にUNCネットワークパスは使用できません。ローカルディスクを指定してください。")
    path = Path(expanded)
    if os.name == "nt" and path.drive:
        drive_root = path.drive.rstrip("\\/") + "\\"
        drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(drive_root))
        if drive_type == 4:
            raise H3BridgeError(
                f"{label}にネットワークドライブ {path.drive} は使用できません。ローカルディスクを指定してください。"
            )
        if drive_type in {0, 1}:
            raise H3BridgeError(f"{label}のドライブ種別を確認できません: {path.drive}")
    return path


def _resolve_local_path(value: str | os.PathLike[str], label: str) -> Path:
    unresolved = _local_filesystem_path(value, label)
    resolved = unresolved.resolve()
    _local_filesystem_path(resolved, label)
    return resolved


def resolve_runtime_root(value: str | os.PathLike[str] | None) -> Path:
    if not value:
        raise H3BridgeError("ComfyUI のフォルダーを指定してください。")
    root = _resolve_local_path(value, "ComfyUI runtime")
    if root == managed_runtime_root() and installed_runtime_root() is None:
        raise H3BridgeError("H3 Studioの「環境とモデルを準備」を実行してください。")
    if not (root / "main.py").is_file():
        raise H3BridgeError(f"ComfyUI main.py が見つかりません: {root}")
    _resolve_local_path(root / "main.py", "ComfyUI main.py")
    if not (root / "models").is_dir():
        raise H3BridgeError(f"ComfyUI models フォルダーが見つかりません: {root}")
    return root


def model_file_status(runtime_root: Path | None, acceleration: H3Acceleration | None = None) -> dict[str, bool]:
    files = (acceleration or H3Acceleration()).model_files(MODEL_FILES)
    if runtime_root is None:
        return {name: False for name in files}
    try:
        models = model_root(runtime_root)
    except ValueError as exc:
        raise H3BridgeError(str(exc)) from exc
    return {
        name: (models / directory / filename).is_file()
        for name, (directory, filename) in files.items()
    }


def server_model_file_status(nodes: dict[str, Any], acceleration: H3Acceleration | None = None) -> dict[str, bool]:
    loader_inputs = {
        "UNETLoader": "unet_name",
        "CLIPLoader": "clip_name",
        "VAELoader": "vae_name",
    }
    choices: dict[str, set[str]] = {}
    for node_name, input_name in loader_inputs.items():
        try:
            values = nodes[node_name]["input"]["required"][input_name][0]
        except (KeyError, IndexError, TypeError):
            values = []
        choices[node_name] = {str(value) for value in values if isinstance(value, str)}
    loaders = {"diffusion_models": "UNETLoader", "text_encoders": "CLIPLoader", "vae": "VAELoader"}
    return {
        name: filename in choices[loaders[directory]]
        for name, (directory, filename) in (acceleration or H3Acceleration()).model_files(MODEL_FILES).items()
    }


def ck_attention_available(nodes: dict[str, Any]) -> bool:
    try:
        definition = nodes["ModelAttentionBackend"]["input"]["required"]["attention"]
        # ComfyUIはV1の列挙値とV3のCOMBO/optionsを同じAPIで返す。
        values = definition[1].get("options", ()) if definition[0] == "COMBO" else definition[0]
    except (KeyError, IndexError, TypeError):
        return False
    return "comfy kitchen attention" in values


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _cli_option_value(arguments: Sequence[str], option: str) -> str | None:
    values = _cli_option_values(arguments, option)
    return values[0] if len(values) == 1 else None


def _cli_option_values(arguments: Sequence[str], option: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            values.append(argument.split("=", 1)[1])
        if argument == option:
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                values.append(arguments[index + 1])
            else:
                values.append("")
    return tuple(values)


def _runtime_arguments_are_allowed(arguments: Sequence[str]) -> bool:
    value_options = {
        "--async-offload",
        "--listen",
        "--port",
        "--preview-method",
        "--reserve-vram",
        "--vram-headroom",
        "--whitelist-custom-nodes",
    }
    flag_options = {
        "--auto-launch",
        "--cache-none",
        "--disable-all-custom-nodes",
        "--disable-api-nodes",
        "--disable-async-offload",
        "--disable-pinned-memory",
    }
    arguments = tuple(str(argument) for argument in arguments)
    whitelist = custom_node_whitelist(arguments)
    if whitelist is None or set(whitelist) - {FAST_VAE_PACK, NEGPIP_PACK, CLIP_CACHE_PACK, negpip_cache.PACK, hybrid.PACK}:
        return False
    index = 0
    saw_main = False
    while index < len(arguments):
        argument = arguments[index]
        if not argument.startswith("--"):
            if index == 0 and Path(argument).name.lower() == "main.py":
                saw_main = True
                index += 1
                continue
            return False
        option, has_equals, _ = argument.partition("=")
        if option in flag_options:
            if has_equals:
                return False
            index += 1
            continue
        if option not in value_options:
            return False
        if option == "--whitelist-custom-nodes":
            index += 1
            while index < len(arguments) and not arguments[index].startswith("--"):
                index += 1
            continue
        if has_equals:
            index += 1
            continue
        if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            index += 2
            continue
        if option == "--async-offload":
            index += 1
            continue
        return False
    return saw_main


def runtime_profile_from_args(
    arguments: Sequence[str],
    expected_port: int | None = None,
) -> str | None:
    arguments = tuple(str(argument) for argument in arguments)
    async_value = _cli_option_value(arguments, "--async-offload")
    if async_value == "":
        async_value = "2"
    headroom_value = _cli_option_value(arguments, "--vram-headroom")
    try:
        reserve_matches = float(_cli_option_value(arguments, "--reserve-vram") or "nan") == 2.0
        headroom_matches = headroom_value is None or float(headroom_value) == 0.0
        port_matches = (
            expected_port is None
            or int(_cli_option_value(arguments, "--port") or "-1") == expected_port
        )
    except ValueError:
        return None
    forbidden = {
        "--cpu",
        "--disable-dynamic-vram",
        "--fast",
        "--fast-disk",
        "--gpu-only",
        "--highvram",
        "--novram",
    }
    common_profile = (
        _runtime_arguments_are_allowed(arguments)
        and len(_cli_option_values(arguments, "--listen")) == 1
        and _cli_option_value(arguments, "--listen") == "127.0.0.1"
        and len(_cli_option_values(arguments, "--port")) == 1
        and port_matches
        and len(_cli_option_values(arguments, "--disable-all-custom-nodes")) == 1
        and len(_cli_option_values(arguments, "--disable-api-nodes")) == 1
        and len(_cli_option_values(arguments, "--reserve-vram")) == 1
        and reserve_matches
        and len(_cli_option_values(arguments, "--preview-method")) == 1
        and _cli_option_value(arguments, "--preview-method") == "none"
        and len(_cli_option_values(arguments, "--vram-headroom")) <= 1
        and headroom_matches
        and not any(_cli_option_values(arguments, option) for option in forbidden)
    )
    if not common_profile:
        return None
    cache_options = ("--cache-none", "--cache-lru", "--cache-classic")
    if (
        len(_cli_option_values(arguments, "--async-offload")) == 1
        and async_value == "2"
        and not _cli_option_values(arguments, "--disable-async-offload")
        and not _cli_option_values(arguments, "--disable-pinned-memory")
        and not any(_cli_option_values(arguments, option) for option in cache_options)
    ):
        return RUNTIME_PROFILE_FAST
    if (
        not _cli_option_values(arguments, "--async-offload")
        and len(_cli_option_values(arguments, "--disable-async-offload")) == 1
        and len(_cli_option_values(arguments, "--disable-pinned-memory")) == 1
        and len(_cli_option_values(arguments, "--cache-none")) == 1
        and not _cli_option_values(arguments, "--cache-lru")
        and not _cli_option_values(arguments, "--cache-classic")
    ):
        return RUNTIME_PROFILE_LOW_RAM
    return None


def h3_core_optimization_status(runtime_root: Path, minimum_commit: str = H3_MINIMUM_COMFY_COMMIT) -> tuple[bool, str | None]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=runtime_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=creationflags,
        )
        revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
        if not revision:
            return False, None
        ancestor_result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", minimum_commit, revision],
            cwd=runtime_root,
            check=False,
            capture_output=True,
            timeout=5,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    return ancestor_result.returncode == 0, revision


def _loopback_server_process(server_url: str) -> Any | None:
    normalized_url = normalize_loopback_url(server_url)
    port = urllib.parse.urlsplit(normalized_url).port or 80
    try:
        import psutil
    except ImportError as exc:
        raise H3BridgeError(f"H3 backend processを確認できません: {exc}") from exc
    try:
        listeners = [
            connection
            for connection in psutil.net_connections(kind="tcp")
            if connection.status == psutil.CONN_LISTEN
            and connection.laddr
            and int(connection.laddr.port) == port
        ]
    except (OSError, psutil.Error) as exc:
        raise H3BridgeError(f"H3 backend processを確認できません: {exc}") from exc
    if not listeners:
        return None

    loopback = [
        connection
        for connection in listeners
        if str(connection.laddr.ip).split("%", 1)[0] in {"127.0.0.1", "::1"}
    ]
    if len(loopback) != len(listeners):
        raise H3BridgeError(
            f"port {port} のprocessはloopback専用ではありません。H3 backendは127.0.0.1だけで起動してください。"
        )
    process_ids = {connection.pid for connection in loopback}
    if None in process_ids:
        raise H3BridgeError(f"port {port} を使用中のprocess IDを確認できません。")
    if len(process_ids) != 1:
        raise H3BridgeError(
            f"port {port} を複数のprocessが待ち受けています。H3 backendを1つだけ起動してください。"
        )
    process_id = process_ids.pop()
    try:
        return psutil.Process(process_id)
    except psutil.Error as exc:
        raise H3BridgeError(f"port {port} のprocessを確認できません: {exc}") from exc


def server_runtime_root(server_url: str) -> Path | None:
    process = _loopback_server_process(server_url)
    if process is None:
        return None
    port = urllib.parse.urlsplit(normalize_loopback_url(server_url)).port or 80
    try:
        import psutil
    except ImportError as exc:
        raise H3BridgeError(f"H3 backend processを確認できません: {exc}") from exc
    try:
        cwd = Path(process.cwd())
        for argument in process.cmdline()[1:]:
            candidate = Path(argument)
            if candidate.name.lower() != "main.py":
                continue
            main_path = candidate if candidate.is_absolute() else cwd / candidate
            if main_path.is_file():
                return _resolve_local_path(main_path.parent, "接続中のComfyUI runtime")
        if (cwd / "main.py").is_file():
            return _resolve_local_path(cwd, "接続中のComfyUI runtime")
    except (OSError, psutil.Error) as exc:
        raise H3BridgeError(f"port {port} のComfyUI processを確認できません: {exc}") from exc
    raise H3BridgeError(f"port {port} を使用中のprocessがComfyUI main.pyから起動されたことを確認できません。")


def _same_local_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(left.resolve())) == os.path.normcase(os.fspath(right.resolve()))


class ComfyH3Client:
    def __init__(self, server_url: str = H3_SERVER_URL, timeout: float = 15.0):
        self.server_url = normalize_loopback_url(server_url)
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self._client = httpx.Client(
            base_url=self.server_url,
            trust_env=False,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=1,
                max_keepalive_connections=1,
                keepalive_expiry=30.0,
            ),
        )

    def close(self) -> None:
        self._client.close()

    def _request_json(self, path: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            response = self._client.request(
                "POST" if body is not None else "GET",
                path,
                content=body,
                headers=headers,
                timeout=self.timeout if timeout is None else timeout,
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise H3BridgeError(f"ComfyUI に接続できません: {exc}") from exc
        if response.status_code >= 400:
            details = response.text
            try:
                parsed = json.loads(details)
                details = _format_comfy_error(parsed)
            except json.JSONDecodeError:
                pass
            error_type = H3JobNotFound if response.status_code == 404 else H3BridgeError
            raise error_type(
                f"ComfyUI が要求を拒否しました (HTTP {response.status_code}): {details[:1200]}"
            )
        raw = response.content
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise H3BridgeError("ComfyUI から不正な応答を受信しました。") from exc

    def system_stats(self) -> dict[str, Any]:
        value = self._request_json("/system_stats")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI system_stats の形式が不正です。")
        return value

    def queue_counts(self) -> tuple[int, int]:
        value = self._request_json("/queue")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI queue の形式が不正です。")
        running = value.get("queue_running") or []
        pending = value.get("queue_pending") or []
        if not isinstance(running, list) or not isinstance(pending, list):
            raise H3BridgeError("ComfyUI queue の形式が不正です。")
        return len(running), len(pending)

    def object_info(self, node_types: Sequence[str] | None = None, timeout: float = 8.0) -> dict[str, Any]:
        if node_types is None:
            value = self._request_json("/object_info", timeout=timeout)
            if not isinstance(value, dict):
                raise H3BridgeError("ComfyUI object_info の形式が不正です。")
            return value

        deadline = time.monotonic() + timeout
        nodes: dict[str, Any] = {}
        for class_type in sorted(set(node_types)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise H3BridgeError("ComfyUI H3 node情報の取得が時間切れになりました。")
            value = self._request_json(
                f"/object_info/{urllib.parse.quote(class_type, safe='')}",
                timeout=remaining,
            )
            if not isinstance(value, dict):
                raise H3BridgeError(f"ComfyUI object_info/{class_type} の形式が不正です。")
            nodes.update(value)
        return nodes

    def submit(self, workflow: dict[str, Any]) -> str:
        prompt_id = str(uuid.uuid4())
        response = self._request_json(
            "/prompt",
            {
                "prompt": workflow,
                "client_id": self.client_id,
                "prompt_id": prompt_id,
                "extra_data": {"workflow_id": "forge-neo-minimax-h3-studio"},
            },
            timeout=90,
        )
        result = response.get("prompt_id") if isinstance(response, dict) else None
        if not result:
            raise H3BridgeError("ComfyUI が prompt_id を返しませんでした。")
        return str(result)

    def job(self, prompt_id: str) -> dict[str, Any]:
        value = self._request_json(f"/api/jobs/{urllib.parse.quote(prompt_id)}")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI job status の形式が不正です。")
        return value

    def history(self, prompt_id: str) -> dict[str, Any]:
        value = self._request_json(f"/history/{urllib.parse.quote(prompt_id)}")
        if not isinstance(value, dict):
            raise H3BridgeError("ComfyUI history の形式が不正です。")
        return value

    def cancel(self, prompt_id: str) -> None:
        if prompt_id:
            _mark_cancelled_job(prompt_id)
            try:
                self._request_json(f"/api/jobs/{urllib.parse.quote(prompt_id)}/cancel", {})
            except H3BridgeError:
                _clear_cancelled_job(prompt_id)
                raise


def _queue_counts(server_url: str) -> tuple[int, int]:
    client = ComfyH3Client(server_url)
    try:
        return client.queue_counts()
    finally:
        client.close()


def _server_api_responding(server_url: str) -> bool:
    client: ComfyH3Client | None = None
    try:
        client = ComfyH3Client(server_url, timeout=1.0)
        client.system_stats()
        return True
    except H3BridgeError:
        return False
    finally:
        if client is not None:
            client.close()


def _format_comfy_error(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    error = value.get("error")
    if isinstance(error, dict):
        parts = [error.get("message"), error.get("details")]
        node_errors = value.get("node_errors")
        if node_errors:
            parts.append(json.dumps(node_errors, ensure_ascii=False))
        return " / ".join(str(part) for part in parts if part)
    return str(error or value)


def inspect_readiness(
    runtime_root: Path | None,
    server_url: str = H3_SERVER_URL,
    object_timeout: float = 8.0,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    acceleration = acceleration or H3Acceleration()
    acceleration.validate()
    required_nodes = REQUIRED_NODE_TYPES | acceleration.extra_nodes()
    files = model_file_status(runtime_root, acceleration=acceleration)
    client: ComfyH3Client | None = None
    try:
        client = ComfyH3Client(server_url, timeout=2.0)
        stats = client.system_stats()
        connected_root = server_runtime_root(client.server_url)
        if connected_root is None:
            raise H3BridgeError("ComfyUI APIは応答しましたが、対応するloopback processを確認できません。")
        if runtime_root is None:
            raise H3BridgeError(f"接続中のComfyUI runtimeを選択してください: {connected_root}")
        selected_root = resolve_runtime_root(runtime_root)
        if not _same_local_path(connected_root, selected_root):
            raise H3BridgeError(
                "別のComfyUIがこのportを使用中です。"
                f" 選択: {selected_root} / 接続中: {connected_root}"
            )
        system = stats.get("system") or {}
        devices = stats.get("devices") or []
        version = str(system.get("comfyui_version") or "")
        if _version_tuple(version) < (0, 31, 0):
            raise H3BridgeError(
                f"最新のMiniMax H3最適化には ComfyUI 0.31.0 以降が必要です（現在 {version or '不明'}）。"
            )
        nodes = client.object_info(required_nodes, timeout=object_timeout)
        missing_nodes = tuple(sorted(required_nodes - set(nodes)))
        server_files = server_model_file_status(nodes, acceleration=acceleration)
        h3_core_optimized, core_revision = h3_core_optimization_status(
            selected_root, SPARSE_COMMIT if acceleration.attention != "dense" else H3_MINIMUM_COMFY_COMMIT
        )
        packages = {
            str(item.get("name")): str(item.get("installed"))
            for item in system.get("comfy_package_versions") or []
            if isinstance(item, dict) and item.get("name") and item.get("installed")
        }
        runtime_args = tuple(str(argument) for argument in system.get("argv") or ())
        device = devices[0] if devices else {}
        return RuntimeReadiness(
            runtime_root=selected_root,
            server_url=client.server_url,
            connected=True,
            comfy_version=version or None,
            gpu_name=str(device.get("name") or "") or None,
            vram_gib=(float(device["vram_total"]) / 1024**3) if device.get("vram_total") else None,
            package_versions=packages,
            runtime_args=runtime_args,
            ck_attention_available=ck_attention_available(nodes),
            core_revision=core_revision,
            h3_core_optimized=h3_core_optimized,
            runtime_profile=runtime_profile_from_args(
                runtime_args,
                expected_port=urllib.parse.urlsplit(client.server_url).port or 80,
            ),
            ram_free_gib=(float(system["ram_free"]) / 1024**3) if system.get("ram_free") else None,
            ram_total_gib=(float(system["ram_total"]) / 1024**3) if system.get("ram_total") else None,
            commit_free_gib=_local_commit_free_gib(),
            model_files=files,
            server_model_files=server_files,
            missing_nodes=missing_nodes,
            acceleration=acceleration,
            node_schemas=nodes,
        )
    except H3BridgeError as exc:
        return RuntimeReadiness(
            runtime_root=runtime_root,
            server_url=server_url,
            connected=False,
            model_files=files,
            error=str(exc),
            acceleration=acceleration,
        )
    finally:
        if client is not None:
            client.close()


def _python_for_runtime(runtime_root: Path) -> Path:
    python = runtime_root.parent / ".venv" / "Scripts" / "python.exe"
    if python.is_file():
        return _resolve_local_path(python, "H3専用Python")
    raise H3BridgeError(
        "H3専用Pythonがありません。H3 Studioの「環境とモデルを準備」を実行してください。"
    )


def _runtime_command(
    python: Path,
    port: int,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    acceleration: H3Acceleration | None = None,
) -> list[str]:
    acceleration = acceleration or H3Acceleration()
    acceleration.validate()
    if runtime_profile not in RUNTIME_PROFILES:
        raise H3BridgeError(f"未対応のH3 runtime profileです: {runtime_profile}")
    command = [
        os.fspath(python),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-all-custom-nodes",
        "--disable-api-nodes",
        "--reserve-vram",
        "2",
        "--preview-method",
        "none",
    ]
    if runtime_profile == RUNTIME_PROFILE_FAST:
        command.extend(["--async-offload", "2"])
    else:
        command.extend(["--cache-none", "--disable-async-offload", "--disable-pinned-memory"])
    if packs := acceleration.runtime_packs():
        command.extend(["--whitelist-custom-nodes", *packs])
    return command


def start_runtime(
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    wait_seconds: float = 120.0,
    initial_readiness: RuntimeReadiness | None = None,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    if _RUNTIME_SETUP_ACTIVE:
        raise H3BridgeError("H3のセットアップが完了するまでお待ちください。")
    runtime_root = resolve_runtime_root(runtime_root)
    try:
        with setup_lock(runtime_root):
            return _start_runtime_locked(
                runtime_root, server_url, log_directory, runtime_profile, wait_seconds, initial_readiness, acceleration,
            )
    except ValueError as exc:
        raise H3BridgeError(str(exc)) from exc


def _start_runtime_locked(
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    wait_seconds: float = 120.0,
    initial_readiness: RuntimeReadiness | None = None,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    global _MANAGED_PROCESS, _MANAGED_PROCESS_IDENTITY
    acceleration = acceleration or H3Acceleration()
    acceleration.validate()
    if initial_readiness is not None and initial_readiness.acceleration != acceleration:
        raise H3BridgeError("H3 backendの事前確認と高速化設定が一致しません。状態を再確認してください。")
    runtime_root = resolve_runtime_root(runtime_root)
    if acceleration.hybrid.enabled:
        try:
            hybrid.verify_directory(runtime_root / "custom_nodes" / hybrid.PACK)
        except (OSError, ValueError) as exc:
            raise H3BridgeError(str(exc)) from exc
    if acceleration.clip_cache != "off":
        try:
            require_clipcache(runtime_root)
        except (OSError, ValueError) as exc:
            raise H3BridgeError(str(exc)) from exc
    normalized_url = normalize_loopback_url(server_url)
    parsed = urllib.parse.urlsplit(normalized_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise H3BridgeError("ローカル以外のComfyUIは起動できません。")
    port = parsed.port or 80
    identity = (runtime_root, normalized_url)

    if initial_readiness is None:
        current = inspect_readiness(runtime_root, normalized_url, acceleration=acceleration)
    else:
        try:
            initial_root = (
                resolve_runtime_root(initial_readiness.runtime_root)
                if initial_readiness.runtime_root is not None
                else None
            )
            initial_url = normalize_loopback_url(initial_readiness.server_url)
        except H3BridgeError as exc:
            raise H3BridgeError(f"H3 backendの事前確認結果が不正です: {exc}") from exc
        if initial_root is None or not _same_local_path(initial_root, runtime_root) or initial_url != normalized_url:
            raise H3BridgeError("H3 backendの事前確認結果が、選択中のruntimeまたはURLと一致しません。")
        current = initial_readiness
    if current.connected:
        return current
    listening_root = server_runtime_root(normalized_url)
    if listening_root is not None and not _same_local_path(listening_root, runtime_root):
        raise H3BridgeError(
            f"別のComfyUIがport {port}を使用中です。選択: {runtime_root} / 接続中: {listening_root}"
        )
    if listening_root is not None and current.error and "接続できません" not in current.error:
        raise H3BridgeError(current.error)

    python = _python_for_runtime(runtime_root)
    log_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = log_directory / f"minimax-h3-{stamp}.stdout.log"
    stderr_path = log_directory / f"minimax-h3-{stamp}.stderr.log"
    command = _runtime_command(python, port, runtime_profile, acceleration=acceleration)

    with _PROCESS_LOCK:
        if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is None:
            if _MANAGED_PROCESS_IDENTITY != identity:
                managed_root, managed_url = _MANAGED_PROCESS_IDENTITY or (Path("不明"), "不明")
                raise H3BridgeError(
                    "H3 Studioが別のruntimeを管理中です。"
                    f" 実行中: {managed_root} ({managed_url}) / 要求: {runtime_root} ({normalized_url})"
                )
        elif listening_root is None:
            if acceleration.negpip.enabled:
                try:
                    install_bundled_negpip(runtime_root)
                    if acceleration.clip_cache != "off":
                        negpip_cache.install(runtime_root)
                except (OSError, ValueError) as exc:
                    raise H3BridgeError(str(exc)) from exc
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                _MANAGED_PROCESS = subprocess.Popen(
                    command,
                    cwd=runtime_root,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=sanitized_subprocess_environment(os.environ),
                    creationflags=creationflags,
                )
                _MANAGED_PROCESS_IDENTITY = identity

    deadline = time.monotonic() + wait_seconds
    last = current
    while time.monotonic() < deadline:
        if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is not None:
            raise H3BridgeError(
                f"ComfyUI が起動直後に終了しました。ログを確認してください: {stderr_path}"
            )
        time.sleep(1.0)
        if not _server_api_responding(normalized_url):
            continue
        last = inspect_readiness(runtime_root, normalized_url, acceleration=acceleration)
        if last.connected:
            return last
    raise H3BridgeError(
        f"ComfyUI の起動を {int(wait_seconds)} 秒待ちましたが接続できません。ログ: {stderr_path}"
    )


def _owns_runtime_process(listener: Any, identity: tuple[Path, str]) -> bool:
    """Pythonの起動用プロセスを含め、このセッションのプロセス系統か確認する。"""
    with _PROCESS_LOCK:
        parent = _MANAGED_PROCESS
        if parent is None or parent.poll() is not None or _MANAGED_PROCESS_IDENTITY != identity or listener is None:
            return False
    if parent.pid == int(listener.pid):
        return True
    import psutil

    try:
        return any(ancestor.pid == parent.pid for ancestor in listener.parents())
    except (OSError, psutil.Error):
        return False


def _stop_managed_runtime(listener: Any | None = None) -> None:
    global _MANAGED_PROCESS, _MANAGED_PROCESS_IDENTITY
    with _PROCESS_LOCK:
        process = _MANAGED_PROCESS
        identity = _MANAGED_PROCESS_IDENTITY
    if process is not None and process.poll() is None:
        listener_stopped = False
        if identity is not None:
            import psutil

            try:
                listener = listener if listener is not None else _loopback_server_process(identity[1])
                if listener is not None and _owns_runtime_process(listener, identity):
                    listener.terminate()
                    try:
                        listener.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        listener.kill()
                        listener.wait(timeout=10)
                    listener_stopped = True
            except psutil.NoSuchProcess:
                listener_stopped = True
            except (H3BridgeError, OSError, psutil.Error) as exc:
                _LOG.warning("Managed H3 listener could not be stopped (%s).", type(exc).__name__)
                return
        try:
            if not listener_stopped:
                process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                _LOG.warning(
                    "Managed H3 runtime did not stop cleanly (%s).",
                    type(exc).__name__,
                )
        except OSError as exc:
            _LOG.warning(
                "Managed H3 runtime could not be stopped (%s).", type(exc).__name__
            )
    with _PROCESS_LOCK:
        if _MANAGED_PROCESS is process:
            _MANAGED_PROCESS = None
            _MANAGED_PROCESS_IDENTITY = None


atexit.register(_stop_managed_runtime)


@contextmanager
def runtime_setup_session(runtime_root: Path, server_url: str = H3_SERVER_URL):
    """生成・保存が終わった自分のH3だけを停止して、導入中の再起動を防ぐ。"""
    global _RUNTIME_SETUP_ACTIVE
    with _RUNTIME_LIFECYCLE_LOCK:
        if _RUNTIME_SETUP_ACTIVE:
            raise H3BridgeError("H3のセットアップを実行中です。")
        if _active_generation_count():
            raise H3BridgeError("H3の生成・保存が完了してからセットアップを実行してください。")
        listening = server_runtime_root(server_url)
        if listening is not None:
            process = _loopback_server_process(server_url)
            owned = _owns_runtime_process(process, (runtime_root, server_url))
            if not _same_local_path(listening, runtime_root) or not owned:
                raise H3BridgeError("別のセッションがH3を使用中です。そのセッションを終了してください。")
            if any(_queue_counts(server_url)):
                raise H3BridgeError("H3の生成キューが空になってからセットアップを実行してください。")
            _stop_managed_runtime(process)
            if server_runtime_root(server_url) is not None:
                raise H3BridgeError("H3を停止できませんでした。セットアップを中止します。")
        _RUNTIME_SETUP_ACTIVE = True
    # Gradioのgeneratorは別workerで再開され得るため、RLockをyield越しに保持しない。
    try:
        yield
    finally:
        with _RUNTIME_LIFECYCLE_LOCK:
            _RUNTIME_SETUP_ACTIVE = False


def _restart_runtime_locked(
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    wait_seconds: float = 120.0,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    runtime_root = resolve_runtime_root(runtime_root)
    normalized_url = normalize_loopback_url(server_url)
    active_generations = _active_generation_count()
    if active_generations:
        raise H3BridgeError(
            f"Forge Neoが生成結果を処理中のため再起動しません（処理中 {active_generations}件）。"
        )
    listening_root = server_runtime_root(normalized_url)
    if listening_root is None:
        return start_runtime(
            runtime_root,
            normalized_url,
            log_directory,
            runtime_profile=runtime_profile,
            wait_seconds=wait_seconds,
            acceleration=acceleration,
        )
    if not _same_local_path(listening_root, runtime_root):
        raise H3BridgeError(
            "別のComfyUIは再起動できません。"
            f" 選択: {runtime_root} / 接続中: {listening_root}"
        )

    running, pending = _queue_counts(normalized_url)
    if running or pending:
        raise H3BridgeError(
            f"生成キューが空ではないため再起動しません（実行中 {running} / 待機 {pending}）。"
        )

    process = _loopback_server_process(normalized_url)
    if process is None:
        return start_runtime(
            runtime_root,
            normalized_url,
            log_directory,
            runtime_profile=runtime_profile,
            wait_seconds=wait_seconds,
            acceleration=acceleration,
        )
    identity = (runtime_root, normalized_url)
    if not _owns_runtime_process(process, identity):
        raise H3BridgeError(
            "接続中のH3 backendはこのForgeセッションが起動したprocessではないため、自動停止しません。"
            " 外部ランチャーを閉じてから「接続 / 起動」を押してください。"
        )

    running, pending = _queue_counts(normalized_url)
    if running or pending:
        raise H3BridgeError(
            f"生成キューが空ではないため再起動しません（実行中 {running} / 待機 {pending}）。"
        )
    _stop_managed_runtime(process)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if _loopback_server_process(normalized_url) is None:
            break
        time.sleep(0.25)
    else:
        raise H3BridgeError("H3 backendの停止を15秒待ちましたがportが解放されません。")

    return start_runtime(
        runtime_root,
        normalized_url,
        log_directory,
        runtime_profile=runtime_profile,
        wait_seconds=wait_seconds,
        acceleration=acceleration,
    )


def restart_runtime(
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    wait_seconds: float = 120.0,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    with _RUNTIME_LIFECYCLE_LOCK:
        readiness = _restart_runtime_locked(
            runtime_root,
            server_url,
            log_directory,
            runtime_profile=runtime_profile,
            wait_seconds=wait_seconds,
            acceleration=acceleration,
        )
        return validate_readiness(readiness, runtime_profile)


def validate_readiness(
    readiness: RuntimeReadiness,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
) -> RuntimeReadiness:
    acceleration = readiness.acceleration
    acceleration.validate()
    if runtime_profile not in RUNTIME_PROFILES:
        raise H3BridgeError(f"未対応のH3 runtime profileです: {runtime_profile}")
    if not readiness.connected:
        raise H3BridgeError(readiness.error or "H3 backendへ接続できません。")
    if NEGPIP_NODE in readiness.missing_nodes:
        raise H3BridgeError("NegPiPノードがありません。生成キューを空にして「選択設定で再起動」を押すと、同梱の固定版だけを導入・許可します。")
    if readiness.missing_nodes:
        raise H3BridgeError("ComfyUI に H3 必須ノードがありません: " + ", ".join(readiness.missing_nodes))
    if not readiness.h3_core_optimized:
        revision = readiness.core_revision[:12] if readiness.core_revision else "未確認"
        raise H3BridgeError(
            "選択した設定に必要なH3 core修正を確認できません。"
            f" ComfyUIを {(SPARSE_COMMIT if acceleration.attention != 'dense' else H3_MINIMUM_COMFY_COMMIT)[:12]} 以降へ更新してください（現在 {revision}）。"
        )
    kitchen = readiness.package_versions.get("comfy-kitchen", "")
    if not readiness.ck_attention_available or _version_tuple(kitchen) < (0, 2, 30):
        raise H3BridgeError(
            "Comfy Kitchen INT8 attentionを利用できません。"
            f" comfy-kitchen 0.2.30以上を導入してください（現在 {kitchen or '未確認'}）。"
        )
    if readiness.runtime_profile != runtime_profile:
        expected = RUNTIME_PROFILE_LABELS[runtime_profile]
        detected = RUNTIME_PROFILE_LABELS.get(readiness.runtime_profile or "", "未対応の起動設定")
        raise H3BridgeError(
            f"H3 backendの起動設定が一致しません（選択: {expected} / 接続中: {detected}）。"
            " 実行環境とモデルの「選択設定で再起動」を押してください。"
        )
    whitelist = custom_node_whitelist(readiness.runtime_args)
    expected_whitelist = acceleration.runtime_packs()
    if whitelist != expected_whitelist:
        raise H3BridgeError(
            "追加ノード（Fast VAE / NegPiP / CLIPCached / 長尺生成）の許可設定が一致しません。「選択設定で再起動」を押してください。"
            " 選択した追加ノードだけを許可し、他のcustom nodeは無効のままにします。"
        )
    if acceleration.attention != "dense" and _version_tuple(kitchen) < (0, 2, 33):
        raise H3BridgeError("Sparse Attentionにはcomfy-kitchen 0.2.33以上が必要です。ComfyUIのrequirementsを更新してください。")
    try:
        acceleration.validate_nodes(readiness.node_schemas)
        if acceleration.hybrid.enabled:
            hybrid.verify_directory(readiness.runtime_root / "custom_nodes" / hybrid.PACK)
        if acceleration.clip_cache != "off":
            require_clipcache(readiness.runtime_root)
            if acceleration.negpip.enabled:
                negpip_cache.verify(readiness.runtime_root)
    except ValueError as exc:
        raise H3BridgeError(str(exc)) from exc
    if not readiness.ready_for_fl2va:
        required = ("FL2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")
        selected = acceleration.model_files(MODEL_FILES)
        missing_local = [f"{name}: models/{selected[name][0]}/{selected[name][1]}" for name in required if not readiness.model_files.get(name)]
        missing_server = [f"{name}: {selected[name][1]}" for name in required if not readiness.server_model_files.get(name)]
        details = []
        if missing_local:
            details.append("filesystem: " + ", ".join(missing_local))
        if missing_server:
            details.append("接続先のmodel一覧: " + ", ".join(missing_server))
        raise H3BridgeError("選択したH3構成のモデルが不足しています: " + " / ".join(details))
    return readiness


def _ensure_ready_locked(
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    if runtime_profile not in RUNTIME_PROFILES:
        raise H3BridgeError(f"未対応のH3 runtime profileです: {runtime_profile}")
    readiness = inspect_readiness(runtime_root, server_url, acceleration=acceleration)
    if not readiness.connected:
        readiness = start_runtime(
            runtime_root,
            server_url,
            log_directory,
            runtime_profile=runtime_profile,
            initial_readiness=readiness,
            acceleration=acceleration,
        )
    return validate_readiness(readiness, runtime_profile)


def ensure_ready(
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    acceleration: H3Acceleration | None = None,
) -> RuntimeReadiness:
    with _RUNTIME_LIFECYCLE_LOCK:
        if _RUNTIME_SETUP_ACTIVE:
            raise H3BridgeError("H3のセットアップが完了するまでお待ちください。")
        return _ensure_ready_locked(
            runtime_root,
            server_url,
            log_directory,
            runtime_profile=runtime_profile,
            acceleration=acceleration,
        )


def _validate_media_path(path_value: str, expected: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise H3BridgeError(f"参照ファイルが見つかりません: {path}")
    allowed = {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"},
        "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
        "audio": {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"},
    }[expected]
    if path.suffix.lower() not in allowed:
        raise H3BridgeError(f"{expected}として未対応の形式です: {path.suffix or '(拡張子なし)'}")
    return path


@functools.lru_cache(maxsize=64)
def _probe_media_cached(
    path_value: str,
    _size_bytes: int,
    _modified_ns: int,
    _changed_ns: int,
) -> tuple[float, bool, float | None]:
    path = Path(path_value)
    try:
        import av

        with av.open(os.fspath(path)) as container:
            duration = float(container.duration or 0) / float(av.time_base)
            has_audio = bool(container.streams.audio)
            video_stream = container.streams.video[0] if container.streams.video else None
            video_rate = None
            if video_stream is not None:
                video_rate = video_stream.average_rate or video_stream.base_rate
            video_fps = float(video_rate) if video_rate else None
    except Exception as exc:
        raise H3BridgeError(f"参照メディアを解析できません: {path.name}: {exc}") from exc
    return duration, has_audio, video_fps


def _probe_media(path: Path) -> tuple[float, bool, float | None]:
    try:
        resolved = path.resolve()
        file_stat = resolved.stat()
    except OSError as exc:
        raise H3BridgeError(f"参照メディアを確認できません: {path.name}: {exc}") from exc
    return _probe_media_cached(
        os.fspath(resolved),
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _managed_input_root(runtime_root: Path, create: bool = False) -> tuple[Path, Path]:
    input_root = _resolve_local_path(runtime_root / "input", "ComfyUI input")
    managed_candidate = input_root / "forge_h3"
    if create:
        managed_candidate.mkdir(parents=True, exist_ok=True)
    managed_root = _resolve_local_path(managed_candidate, "MiniMax H3 input")
    if input_root not in managed_root.parents:
        raise H3BridgeError("MiniMax H3 input フォルダーが ComfyUI input の外を指しています。")
    return input_root, managed_root


def _copy_to_comfy_input(source: Path, runtime_root: Path) -> str:
    input_root, target_dir = _managed_input_root(runtime_root, create=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name).strip("._") or "reference"
    target_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    target = _resolve_local_path(target_dir / target_name, "MiniMax H3 input copy")
    if input_root not in target.parents:
        raise H3BridgeError("ComfyUI input フォルダー外への書き込みを拒否しました。")
    shutil.copy2(source, target)
    os.utime(target, None)
    return f"forge_h3/{target_name}"


def _prepared_media_names(prepared: dict[str, Any]) -> Iterator[str]:
    for key in ("first_frame", "last_frame", "control_video"):
        if prepared.get(key):
            yield str(prepared[key])
    for name in prepared.get("images") or []:
        yield str(name)
    for video in prepared.get("videos") or []:
        if isinstance(video, dict) and video.get("name"):
            yield str(video["name"])
    for name in prepared.get("audios") or []:
        yield str(name)


def cleanup_prepared_media(prepared: dict[str, Any], runtime_root: Path) -> None:
    input_root, managed_root = _managed_input_root(runtime_root)
    for relative_name in set(_prepared_media_names(prepared)):
        candidate = _resolve_local_path(input_root / relative_name, "MiniMax H3 input copy")
        if managed_root not in candidate.parents:
            continue
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                candidate.unlink(missing_ok=True)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
        if last_error is not None:
            _LOG.warning("MiniMax H3 input copy could not be removed: %s: %s", candidate, last_error)


def cleanup_stale_prepared_media(
    runtime_root: Path,
    max_age_seconds: float = 7 * 24 * 60 * 60,
) -> None:
    _, managed_root = _managed_input_root(runtime_root)
    if not managed_root.is_dir():
        return
    cutoff = time.time() - max(60.0, float(max_age_seconds))
    stale_pattern = re.compile(r"^[0-9a-f]{12}_.+")
    for candidate in managed_root.iterdir():
        try:
            if not candidate.is_file() or not stale_pattern.fullmatch(candidate.name):
                continue
            if candidate.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        cleanup_prepared_media({"images": [f"forge_h3/{candidate.name}"]}, runtime_root)


def _cleanup_after_terminal(
    client: ComfyH3Client,
    prompt_id: str,
    prepared: dict[str, Any],
    runtime_root: Path,
    wait_seconds: float = 600.0,
) -> None:
    try:
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            try:
                status = str(client.job(prompt_id).get("status") or "").lower()
                if status in _TERMINAL_JOB_STATUSES:
                    try:
                        cleanup_prepared_media(prepared, runtime_root)
                    finally:
                        _clear_cancelled_job(prompt_id)
                        _finish_gpu_generation(prompt_id)
                    return
            except H3JobNotFound:
                if _is_cancelled_job(prompt_id):
                    try:
                        cleanup_prepared_media(prepared, runtime_root)
                    finally:
                        _clear_cancelled_job(prompt_id)
                        _finish_gpu_generation(prompt_id)
                    return
            except H3BridgeError:
                pass
            time.sleep(1.0)
        _LOG.warning("MiniMax H3 deferred input cleanup timed out for job %s", prompt_id)
    finally:
        try:
            client.close()
        except (OSError, httpx.HTTPError) as exc:
            _LOG.warning("MiniMax H3 cleanup HTTP client could not be closed: %s", exc)


def _schedule_deferred_cleanup(
    client: ComfyH3Client,
    prompt_id: str,
    prepared: dict[str, Any],
    runtime_root: Path,
) -> None:
    worker = threading.Thread(
        target=_cleanup_after_terminal,
        args=(client, prompt_id, prepared, runtime_root),
        name=f"minimax-h3-cleanup-{prompt_id[:8]}",
        daemon=True,
    )
    worker.start()


def prepare_media(request: H3Request, runtime_root: Path) -> dict[str, Any]:
    validate_request(request)
    prepared: dict[str, Any] = {
        "first_frame": None,
        "last_frame": None,
        "images": [],
        "videos": [],
        "audios": [],
    }
    try:
        if request.control.enabled:
            source = _validate_media_path(request.control_video, "video")
            _, managed_root = _managed_input_root(runtime_root)
            managed_root.mkdir(parents=True, exist_ok=True)
            name = f"{uuid.uuid4().hex[:12]}_fun_control.mp4"
            prepared["control_video"] = f"forge_h3/{name}"
            fun_control.prepare_video(source, managed_root / name, *request.dimensions, request.frame_count, request.control.mode)
        if request.mode == MODE_KEYFRAMES:
            if request.first_frame:
                source = _validate_media_path(request.first_frame, "image")
                prepared["first_frame"] = _copy_to_comfy_input(source, runtime_root)
            if request.last_frame:
                source = _validate_media_path(request.last_frame, "image")
                prepared["last_frame"] = _copy_to_comfy_input(source, runtime_root)
            return prepared

        if request.mode != MODE_REFERENCES:
            return prepared

        total_timed_seconds = 0.0
        for value in request.reference_images:
            source = _validate_media_path(value, "image")
            prepared["images"].append(_copy_to_comfy_input(source, runtime_root))
        for value in request.reference_videos:
            source = _validate_media_path(value, "video")
            duration, has_audio, video_fps = _probe_media(source)
            if not 2.0 <= duration <= 15.0:
                raise H3BridgeError(f"参照動画 {source.name} は 2〜15 秒にしてください（{duration:.1f}秒）。")
            if video_fps is None or abs(video_fps - H3_FPS) > 0.05:
                shown_fps = "不明" if video_fps is None else f"{video_fps:.3f}"
                raise H3BridgeError(
                    f"参照動画 {source.name} は24fpsに変換してください（現在 {shown_fps}fps）。"
                )
            total_timed_seconds += duration
            prepared["videos"].append(
                {
                    "name": _copy_to_comfy_input(source, runtime_root),
                    "has_audio": has_audio,
                    "duration": duration,
                    "fps": video_fps,
                }
            )
        for value in request.reference_audios:
            source = _validate_media_path(value, "audio")
            duration, _, _ = _probe_media(source)
            if not 2.0 <= duration <= 15.0:
                raise H3BridgeError(f"参照音声 {source.name} は 2〜15 秒にしてください（{duration:.1f}秒）。")
            total_timed_seconds += duration
            prepared["audios"].append(_copy_to_comfy_input(source, runtime_root))
        if total_timed_seconds > 15.0 + 1e-6:
            raise H3BridgeError(f"参照動画と音声の合計は15秒までです（現在 {total_timed_seconds:.1f}秒）。")
        return prepared
    except Exception:
        cleanup_prepared_media(prepared, runtime_root)
        raise


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def expected_reference_tags(prepared_media: dict[str, Any]) -> tuple[str, ...]:
    tags = [f"<Picture {index}>" for index, _ in enumerate(prepared_media.get("images") or [], 1)]
    audio_index = 1
    for video_index, video in enumerate(prepared_media.get("videos") or [], 1):
        if video.get("has_audio"):
            tags.append(f"<Audio {audio_index}>")
            audio_index += 1
        tags.append(f"<Video {video_index}>")
    for _ in prepared_media.get("audios") or []:
        tags.append(f"<Audio {audio_index}>")
        audio_index += 1
    return tuple(tags)


def validate_reference_prompt_tags(prompt: str, prepared_media: dict[str, Any]) -> None:
    expected = expected_reference_tags(prepared_media)
    related_tags = re.findall(r"<(?:Picture|Video|Audio)\b[^<>]*>", prompt, flags=re.IGNORECASE)
    exact_tags = re.findall(r"<(?:Picture|Video|Audio) [1-9][0-9]*>", prompt)
    malformed = sorted(set(related_tags) - set(exact_tags))
    if malformed:
        raise H3BridgeError(
            "参照タグの形式が不正です: "
            + ", ".join(malformed)
            + "。大文字小文字と半角スペースを含め、<Picture 1> の形式で入力してください。"
        )

    expected_set = set(expected)
    supplied_set = set(exact_tags)
    unknown = [tag for tag in exact_tags if tag not in expected_set]
    if unknown:
        raise H3BridgeError(
            "選択した素材に存在しない参照タグです: " + ", ".join(dict.fromkeys(unknown))
        )
    missing = [tag for tag in expected if tag not in supplied_set]
    if missing:
        raise H3BridgeError(
            "Prompt に未使用の参照素材があります。次のタグを追加してください: " + ", ".join(missing)
        )


def build_workflow(request: H3Request, prepared_media: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    validate_request(request)
    width, height = request.dimensions
    seed = request.resolved_seed if seed is None else int(seed)
    model_name = H3_REF_MODEL if request.mode == MODE_REFERENCES else H3_FL_MODEL

    workflow: dict[str, Any] = {
        "1": _node("UNETLoader", unet_name=model_name, weight_dtype="default"),
        "15": _node(
            "ModelAttentionBackend",
            model=["1", 0],
            attention="comfy kitchen attention",
        ),
        "2": _node("CLIPLoader", clip_name=H3_TEXT_ENCODER, type="minimax", device="default"),
        "3": _node("VAELoader", vae_name=H3_VIDEO_VAE),
        "4": _node("VAELoader", vae_name=H3_AUDIO_VAE),
        "6": _node("RandomNoise", noise_seed=seed),
        "7": _node("KSamplerSelect", sampler_name="res_multistep"),
        "8": _node(
            "BasicScheduler",
            model=["1", 0],
            scheduler=request.scheduler,
            steps=int(request.steps),
            denoise=1.0,
        ),
        "9": _node("BasicGuider", model=["15", 0], conditioning=["5", 0]),
        "10": _node(
            "SamplerCustomAdvanced",
            noise=["6", 0],
            guider=["9", 0],
            sampler=["7", 0],
            sigmas=["8", 0],
            latent_image=["5", 1],
        ),
        "11": _node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": _node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": _node("CreateVideo", images=["11", 0], audio=["12", 0], fps=H3_FPS, bit_depth=8),
        "14": _node(
            "SaveVideo",
            video=["13", 0],
            filename_prefix="video/Forge_Neo_MiniMax_H3",
            format="auto",
            codec="auto",
        ),
    }

    conditioning_inputs: dict[str, Any]
    if request.mode == MODE_REFERENCES:
        validate_reference_prompt_tags(request.prompt, prepared_media)
        if request.acceleration.hybrid.enabled:
            for continuation in request.acceleration.hybrid.prompts.strip().splitlines():
                validate_reference_prompt_tags(request.prompt + "\n" + continuation, prepared_media)
        conditioning_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": request.prompt.strip(),
            "width": width,
            "height": height,
            "length": request.frame_count,
            "ref_image_size": request.ref_image_size,
        }
        next_id = 20
        for index, image_name in enumerate(prepared_media.get("images") or []):
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = _node("LoadImage", image=image_name)
            conditioning_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        for index, video in enumerate(prepared_media.get("videos") or []):
            load_id = str(next_id)
            components_id = str(next_id + 1)
            next_id += 2
            workflow[load_id] = _node("LoadVideo", file=video["name"])
            workflow[components_id] = _node("GetVideoComponents", video=[load_id, 0])
            conditioning_inputs[f"ref_videos.ref_video_{index}"] = [components_id, 0]
            if video.get("has_audio"):
                conditioning_inputs[f"ref_video_audios.ref_video_audio_{index}"] = [components_id, 1]
        for index, audio_name in enumerate(prepared_media.get("audios") or []):
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = _node("LoadAudio", audio=audio_name)
            conditioning_inputs[f"ref_audios.ref_audio_{index}"] = [node_id, 0]
        workflow["5"] = _node("MiniMaxH3ReferenceToVideo", **conditioning_inputs)
    else:
        conditioning_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": request.prompt.strip(),
            "width": width,
            "height": height,
            "length": request.frame_count,
        }
        if request.mode == MODE_KEYFRAMES:
            if prepared_media.get("first_frame"):
                workflow["20"] = _node("LoadImage", image=prepared_media["first_frame"])
                conditioning_inputs["first_frame"] = ["20", 0]
            if prepared_media.get("last_frame"):
                workflow["21"] = _node("LoadImage", image=prepared_media["last_frame"])
                conditioning_inputs["last_frame"] = ["21", 0]
        workflow["5"] = _node("MiniMaxH3ImageToVideo", **conditioning_inputs)

    request.acceleration.apply_workflow(workflow, MODEL_FILES, request.mode)
    fun_control.apply_workflow(workflow, request.control, prepared_media.get("control_video"))
    if request.acceleration.hybrid.enabled:
        hybrid.apply_workflow(workflow, request.acceleration.hybrid, snap_h3_frames(request.duration_seconds), int(request.steps), request.scheduler, seed)
    return workflow


def extract_history_video(history: dict[str, Any], prompt_id: str, runtime_root: Path) -> Path:
    item = history.get(prompt_id)
    if not isinstance(item, dict):
        raise H3BridgeError("完了履歴に生成結果がありません。")
    outputs = item.get("outputs") or {}
    candidates: list[dict[str, Any]] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for key in ("images", "videos"):
            values = output.get(key) or []
            if isinstance(values, list):
                candidates.extend(value for value in values if isinstance(value, dict))
    for candidate in reversed(candidates):
        filename = str(candidate.get("filename") or "")
        if Path(filename).suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv"}:
            continue
        folder_type = candidate.get("type") or "output"
        base = runtime_root / ("output" if folder_type == "output" else "temp")
        path = (base / str(candidate.get("subfolder") or "") / filename).resolve()
        if base.resolve() not in path.parents or not path.is_file():
            continue
        return path
    raise H3BridgeError("ComfyUI の完了履歴から動画ファイルを特定できません。")


def _execution_error(job: dict[str, Any]) -> str:
    error = job.get("execution_error")
    if isinstance(error, dict):
        return str(error.get("exception_message") or error.get("message") or error)
    return str(error or "生成処理が失敗しました。ComfyUIログを確認してください。")


def _validate_video_result(path: Path, request: H3Request) -> dict[str, Any]:
    """全フレームを順に復号し、要求した映像と音声が最後まで存在することを確認する。"""
    import av

    expected_frames = request.frame_count
    expected_size = request.dimensions
    frame_count = audio_samples = 0
    try:
        with av.open(os.fspath(path)) as container:
            if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
                raise H3BridgeError("完成動画に映像と音声が一つずつ含まれていません。")
            video, audio = container.streams.video[0], container.streams.audio[0]
            if (video.width, video.height) != expected_size or video.average_rate != H3_FPS:
                raise H3BridgeError("完成動画の解像度またはフレームレートが生成条件と一致しません。")
            sample_rate = audio.codec_context.sample_rate
            channels = len(audio.codec_context.layout.channels)
            if sample_rate != 32000 or channels != 2:
                raise H3BridgeError("完成動画の音声が32kHzステレオではありません。")
            for packet in container.demux(video, audio):
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        if (frame.width, frame.height) != expected_size:
                            raise H3BridgeError("完成動画の途中で解像度が変化しています。")
                        if frame.time is None or abs(frame.time - frame_count / H3_FPS) > 0.001:
                            raise H3BridgeError("完成動画のフレーム時刻が連続していません。")
                        frame_count += 1
                        if frame_count > expected_frames:
                            raise H3BridgeError("完成動画のフレーム数が生成条件を超えています。")
                    else:
                        if frame.time is None or abs(frame.time - audio_samples / sample_rate) > 0.001:
                            raise H3BridgeError("完成動画の音声時刻が連続していません。")
                        audio_samples += frame.samples
                        if audio_samples / sample_rate > request.effective_seconds + 0.1:
                            raise H3BridgeError("完成動画の音声が指定した長さを超えています。")
        if frame_count != expected_frames:
            raise H3BridgeError(f"完成動画が途中で終わっています（{frame_count}/{expected_frames}フレーム）。")
        # AACの末尾パディングを許容し、音声の欠落や大きな尺違いを拒否する。
        if not audio_samples or abs(audio_samples / sample_rate - request.effective_seconds) > 2 / H3_FPS:
            raise H3BridgeError("完成動画の音声の長さが映像と一致しません。")
    except H3BridgeError:
        raise
    except (OSError, ValueError, av.FFmpegError) as exc:
        raise H3BridgeError(f"完成動画を最後まで読み込めません: {exc}") from exc
    return {
        "width": expected_size[0],
        "height": expected_size[1],
        "frames": frame_count,
        "fps": H3_FPS,
        "audio_sample_rate": sample_rate,
        "audio_channels": channels,
        "audio_samples": audio_samples,
    }


def mirror_result(
    source: Path,
    output_directory: Path,
    request: H3Request,
    prompt_id: str,
    seed: int,
    readiness: RuntimeReadiness,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex
    target = output_directory / f"MiniMax_H3_{stamp}_{prompt_id[:8]}_{token[:8]}{source.suffix.lower()}"
    metadata_path = target.with_suffix(".json")
    staged_video = target.with_name(f".{target.name}.{token}.part")
    staged_metadata = metadata_path.with_name(f".{metadata_path.name}.{token}.part")
    metadata = {
        "schema_version": 2,
        "model": "MiniMax H3",
        "prompt_id": prompt_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": request.mode,
        "prompt": request.prompt,
        "aspect": request.aspect,
        "quality": request.quality,
        "dimensions": list(request.dimensions),
        "requested_seconds": request.duration_seconds,
        "frames": request.frame_count,
        "effective_seconds": request.effective_seconds,
        "hybrid_revision": hybrid.REVISION if request.acceleration.hybrid.enabled else None,
        "steps": request.steps,
        "seed": seed,
        "scheduler": request.scheduler,
        "ref_image_size": request.ref_image_size,
        "attention_backend": "comfy-kitchen-int8" if request.acceleration.attention == "dense" else f"comfy-kitchen-sparse-{request.acceleration.attention}",
        "acceleration": request.acceleration.to_dict(),
        "fun_control": request.control.to_dict(),
        "fun_control_model": fun_control.MODEL if request.control.enabled else None,
        "clip_cache_revision": CLIP_CACHE_REVISION if request.acceleration.clip_cache != "off" else None,
        "selected_models": {
            name: filename for name, (_, filename) in request.acceleration.model_files(MODEL_FILES).items()
        },
        "video_decoder": "MiniMaxH3FastVAEDecode" if request.acceleration.decode_mode == "fast" else "VAEDecode",
        "comfyui_version": readiness.comfy_version,
        "comfy_kitchen_version": readiness.package_versions.get("comfy-kitchen"),
        "comfyui_revision": readiness.core_revision,
        "runtime_profile": readiness.runtime_profile,
        "source_backend": "comfyui",
        "source_file": source.name,
    }
    try:
        shutil.copy2(source, staged_video)
        metadata["output_validation"] = _validate_video_result(staged_video, request)
        with staged_video.open("rb") as stream:
            metadata["output_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        staged_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staged_metadata, metadata_path)
        try:
            os.replace(staged_video, target)
        except OSError:
            metadata_path.unlink(missing_ok=True)
            raise
    except OSError as exc:
        if not target.exists():
            metadata_path.unlink(missing_ok=True)
        raise H3BridgeError(f"完成動画を Forge Neo の出力へ保存できません: {exc}") from exc
    finally:
        staged_video.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
    return target


def _estimated_required_free_gib(request: H3Request, runtime_profile: str) -> float:
    if runtime_profile not in RUNTIME_PROFILES:
        raise H3BridgeError(f"未対応のH3 runtime profileです: {runtime_profile}")
    width, height = request.dimensions
    decoded_video_gib = width * height * request.frame_count * 3 * 4 / 1024**3
    safety_gib = 4.0 if runtime_profile == RUNTIME_PROFILE_FAST else 2.0
    control_gib = decoded_video_gib + fun_control.MODEL_BYTES / 1024**3 if request.control.enabled else 0.0
    return decoded_video_gib + safety_gib + control_gib


def _validate_request_runtime_constraints(
    request: H3Request,
    readiness: RuntimeReadiness,
    runtime_profile: str,
) -> None:
    if runtime_profile not in RUNTIME_PROFILES:
        raise H3BridgeError(f"未対応のH3 runtime profileです: {runtime_profile}")
    if readiness.acceleration != request.acceleration:
        raise H3BridgeError("生成要求と確認済みの高速化構成が一致しません。状態を再確認してください。")
    if request.mode == MODE_REFERENCES and not readiness.ready_for_ref2va:
        raise H3BridgeError("参照モード用 Ref2VA モデルがありません。")
    if request.control.enabled:
        control_client = ComfyH3Client(readiness.server_url)
        try:
            fun_control.validate_model(readiness.runtime_root)
            fun_control.validate_nodes(control_client.object_info(fun_control.NODES))
        except ValueError as exc:
            raise H3BridgeError(str(exc)) from exc
        finally:
            control_client.close()
    memory_values = {
        "空き物理RAM": readiness.ram_free_gib,
        "OS commit余力": readiness.commit_free_gib,
    }
    available_values: dict[str, float] = {}
    for label, value in memory_values.items():
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise H3BridgeError(f"{label}の情報が不正です。backendを再起動してください。")
        available_values[label] = float(value)
    if not available_values:
        return
    required_free_gib = _estimated_required_free_gib(request, runtime_profile)
    limiting_label, limiting_free_gib = min(available_values.items(), key=lambda item: item[1])
    if limiting_free_gib >= required_free_gib:
        return
    profile_hint = (
        "省RAM profileへ切り替えてbackendを再起動するか、他のアプリを閉じてください。"
        if runtime_profile == RUNTIME_PROFILE_FAST
        else "他のアプリを閉じるか、解像度・長さを下げてください。"
    )
    raise H3BridgeError(
        "生成前の空きRAMが不足しています。"
        f" 必要目安 {required_free_gib:.1f} GiB / {limiting_label} {limiting_free_gib:.1f} GiB。"
        f" {profile_hint}"
    )


def _generation_poll_interval(running: bool, elapsed_seconds: float) -> float:
    if not running:
        return 1.0
    return 2.0 if elapsed_seconds < 60.0 else 5.0


def run_generation(
    request: H3Request,
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    output_directory: Path,
    runtime_profile: str = RUNTIME_PROFILE_FAST,
    poll_seconds: float | None = None,
) -> Iterator[dict[str, Any]]:
    ownership = GPUOwnership()
    waiting_id = str(uuid.uuid4())
    try:
        while not ownership.acquire():
            if _is_cancelled_job(waiting_id):
                raise H3GenerationCancelled("生成を停止しました。")
            yield {"stage": "queued", "message": "GPUの使用終了を待っています", "progress": 0.01, "prompt_id": waiting_id}
            time.sleep(0.1)
        if _is_cancelled_job(waiting_id):
            raise H3GenerationCancelled("生成を停止しました。")
        release_forge_vram()
        yield from _run_generation(request, runtime_root, server_url, log_directory, output_directory, runtime_profile, poll_seconds, ownership)
    finally:
        _clear_cancelled_job(waiting_id)
        with _ACTIVE_GENERATION_LOCK:
            retained = ownership in _GPU_OWNERSHIPS.values()
        if not retained:
            ownership.release()


def _finish_gpu_generation(prompt_id: str) -> None:
    with _ACTIVE_GENERATION_LOCK:
        ownership = _GPU_OWNERSHIPS.pop(prompt_id, None)
        _ACTIVE_GENERATION_IDS.discard(prompt_id)
    if ownership is not None:
        ownership.release()


def _run_generation(
    request: H3Request,
    runtime_root: Path,
    server_url: str,
    log_directory: Path,
    output_directory: Path,
    runtime_profile: str,
    poll_seconds: float | None,
    ownership: GPUOwnership,
) -> Iterator[dict[str, Any]]:
    validate_request(request)
    yield {
        "stage": "runtime",
        "message": "H3 backendを確認しています。未起動なら自動起動します（初回は1〜2分）。",
        "progress": 0.03,
        "prompt_id": "",
    }
    readiness = ensure_ready(
        runtime_root,
        server_url,
        log_directory,
        runtime_profile=runtime_profile,
        acceleration=request.acceleration,
    )
    _validate_request_runtime_constraints(request, readiness, runtime_profile)
    cleanup_stale_prepared_media(runtime_root)
    yield {"stage": "prepare", "message": "入力素材を検証しています", "progress": 0.06, "prompt_id": ""}
    prepared = prepare_media(request, runtime_root)
    client: ComfyH3Client | None = None
    prompt_id = ""
    terminal = False
    deferred_cleanup_scheduled = False
    try:
        seed = request.resolved_seed
        workflow = build_workflow(request, prepared, seed=seed)
        with _RUNTIME_LIFECYCLE_LOCK:
            readiness = ensure_ready(
                runtime_root,
                server_url,
                log_directory,
                runtime_profile=runtime_profile,
                acceleration=request.acceleration,
            )
            _validate_request_runtime_constraints(request, readiness, runtime_profile)
            client = ComfyH3Client(server_url, timeout=hybrid.POLL_TIMEOUT_SECONDS if request.acceleration.hybrid.enabled else 15.0)
            prompt_id = client.submit(workflow)
            _mark_active_generation(prompt_id)
            with _ACTIVE_GENERATION_LOCK:
                _GPU_OWNERSHIPS[prompt_id] = ownership
        yield {
            "stage": "queued",
            "message": "ComfyUI のキューに追加しました",
            "progress": 0.12,
            "prompt_id": prompt_id,
            "seed": seed,
        }
        started = time.monotonic()
        consecutive_poll_failures = 0
        while True:
            try:
                job = client.job(prompt_id)
            except H3JobNotFound:
                if _is_cancelled_job(prompt_id):
                    terminal = True
                    _clear_cancelled_job(prompt_id)
                    raise H3GenerationCancelled("生成を停止しました。") from None
                raise
            except H3BridgeError as exc:
                consecutive_poll_failures += 1
                if consecutive_poll_failures >= H3_STATUS_POLL_MAX_FAILURES:
                    raise H3BridgeError(
                        "H3 backendの状態確認に連続して失敗しました。"
                        f" 最後のエラー: {exc}"
                    ) from exc
                elapsed = time.monotonic() - started
                yield {
                    "stage": "reconnecting",
                    "message": (
                        "H3 backendの応答を待っています"
                        f"（再試行 {consecutive_poll_failures}/{H3_STATUS_POLL_MAX_FAILURES - 1}）"
                    ),
                    "progress": 0.13,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "elapsed": elapsed,
                }
                retry_wait = min(
                    3.0,
                    max(0.5, poll_seconds) if poll_seconds is not None else float(consecutive_poll_failures),
                )
                time.sleep(retry_wait)
                continue
            consecutive_poll_failures = 0
            status = str(job.get("status") or "pending").lower()
            elapsed = time.monotonic() - started
            if status in {"completed", "success"}:
                terminal = True
                _clear_cancelled_job(prompt_id)
                history_error: H3BridgeError | None = None
                for attempt in range(1, H3_STATUS_POLL_MAX_FAILURES + 1):
                    try:
                        history = client.history(prompt_id)
                        source = extract_history_video(history, prompt_id, runtime_root)
                        break
                    except H3BridgeError as exc:
                        history_error = exc
                        if attempt >= H3_STATUS_POLL_MAX_FAILURES:
                            raise H3BridgeError(
                                "完成済みH3ジョブの結果取得に連続して失敗しました。"
                                f" 最後のエラー: {exc}"
                            ) from exc
                        time.sleep(
                            min(
                                3.0,
                                max(0.5, poll_seconds) if poll_seconds is not None else float(attempt),
                            )
                        )
                else:  # pragma: no cover - loop either breaks or raises
                    raise H3BridgeError(f"完成済みH3ジョブの結果を取得できません: {history_error}")
                target = mirror_result(source, output_directory, request, prompt_id, seed, readiness)
                yield {
                    "stage": "complete",
                    "message": "音声付き動画を保存しました",
                    "progress": 1.0,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "path": os.fspath(target),
                    "elapsed": elapsed,
                }
                return
            if status in {"failed", "error"}:
                terminal = True
                _clear_cancelled_job(prompt_id)
                raise H3BridgeError(_execution_error(job))
            if status in {"cancelled", "canceled"}:
                terminal = True
                _clear_cancelled_job(prompt_id)
                raise H3GenerationCancelled("生成を停止しました。")
            running = status in {"running", "in_progress"}
            message = "生成中 — RTX 3090 では長時間かかる場合があります" if running else "キューで待機しています"
            progress = min(0.88, 0.16 + elapsed / 7200.0) if running else 0.13
            yield {
                "stage": "running" if running else "queued",
                "message": message,
                "progress": progress,
                "prompt_id": prompt_id,
                "seed": seed,
                "elapsed": elapsed,
            }
            if poll_seconds is None:
                wait_seconds = _generation_poll_interval(running, elapsed)
            else:
                wait_seconds = max(0.5, poll_seconds)
            time.sleep(wait_seconds)
    finally:
        try:
            if prompt_id and not terminal and client is not None:
                try:
                    client.cancel(prompt_id)
                except H3BridgeError as exc:
                    _LOG.warning("MiniMax H3 job %s could not be cancelled during cleanup: %s", prompt_id, exc)
                _schedule_deferred_cleanup(client, prompt_id, prepared, runtime_root)
                deferred_cleanup_scheduled = True
            if not prompt_id or terminal:
                cleanup_prepared_media(prepared, runtime_root)
        finally:
            if prompt_id and terminal:
                _finish_gpu_generation(prompt_id)
            if client is not None and not deferred_cleanup_scheduled:
                try:
                    client.close()
                except (OSError, httpx.HTTPError) as exc:
                    _LOG.warning("MiniMax H3 HTTP client could not be closed: %s", exc)


def cancel_generation(prompt_id: str, server_url: str) -> None:
    if prompt_id:
        _mark_cancelled_job(prompt_id)
        with _ACTIVE_GENERATION_LOCK:
            if prompt_id not in _ACTIVE_GENERATION_IDS:
                return
        client = ComfyH3Client(server_url)
        try:
            client.cancel(prompt_id)
        finally:
            client.close()


def list_history(runtime_root: Path | None, output_directory: Path, limit: int = 12) -> list[HistoryItem]:
    found: dict[Path, HistoryItem] = {}
    if output_directory.is_dir():
        for path in output_directory.glob("MiniMax_H3_*.mp4"):
            try:
                resolved = path.resolve()
                file_stat = path.stat()
            except OSError:
                continue
            found[resolved] = HistoryItem(
                resolved,
                file_stat.st_mtime,
                "Forge Neo",
                file_stat.st_size,
            )
    if runtime_root is not None:
        runtime_output = runtime_root / "output" / "video"
        if runtime_output.is_dir():
            for pattern in ("*MiniMax*H3*.mp4", "*minimax*h3*.mp4"):
                for path in runtime_output.glob(pattern):
                    try:
                        resolved = path.resolve()
                        file_stat = path.stat()
                    except OSError:
                        continue
                    found.setdefault(
                        resolved,
                        HistoryItem(
                            resolved,
                            file_stat.st_mtime,
                            "ComfyUI",
                            file_stat.st_size,
                        ),
                    )
    return sorted(found.values(), key=lambda item: item.modified_at, reverse=True)[:limit]


def history_html(items: Sequence[HistoryItem]) -> str:
    if not items:
        return (
            '<div class="h3-history-empty"><span>履歴はまだありません</span>'
            '<small>生成が完了すると、音声付きMP4がここに並びます。</small></div>'
        )
    rows = []
    for item in items[:6]:
        size_text = (
            f"{item.size_bytes / 1024**2:.1f} MiB"
            if item.size_bytes is not None
            else "サイズ不明"
        )
        rows.append(
            '<div class="h3-history-row">'
            '<span class="h3-history-thumb" aria-hidden="true">MP4</span>'
            '<span class="h3-history-copy">'
            f'<strong>{html.escape(item.path.stem)}</strong>'
            f'<small>{html.escape(item.label.split(" · ", 1)[0])} · {size_text} · {html.escape(item.source)}</small>'
            "</span></div>"
        )
    return '<div class="h3-history-list">' + "".join(rows) + "</div>"


def history_choices(items: Sequence[HistoryItem]) -> list[tuple[str, str]]:
    return [(item.label, item.public_id) for item in items]


def _resolve_history_selection(selected: str, items: Sequence[HistoryItem]) -> Path:
    try:
        public_matches = {
            item.public_id: item.path.resolve() for item in items
        }
        if selected in public_matches:
            selected_path = public_matches[selected]
        else:
            # Preserve the server-side Python API during the UI migration. The
            # browser only receives opaque IDs from history_choices().
            selected_path = Path(selected).resolve()
        allowed = set(public_matches.values())
    except (OSError, RuntimeError) as exc:
        raise H3BridgeError(f"履歴ファイルを確認できません: {exc}") from exc
    if selected_path not in allowed:
        raise H3BridgeError("選択された履歴ファイルは現在の一覧にありません。")
    if not selected_path.is_file():
        raise H3BridgeError("選択された履歴ファイルは削除されたか、移動されています。")
    return selected_path


def load_history_request(
    selected: str,
    items: Sequence[HistoryItem],
    output_directory: Path,
) -> H3Request:
    selected_path = _resolve_history_selection(selected, items)
    try:
        output_root = output_directory.resolve()
    except OSError as exc:
        raise H3BridgeError(f"Forge Neoの出力フォルダーを確認できません: {exc}") from exc
    if selected_path.parent != output_root:
        raise H3BridgeError("設定を復元できるのはForge Neoで保存した生成履歴だけです。")

    try:
        metadata_path = selected_path.with_suffix(".json").resolve()
    except OSError as exc:
        raise H3BridgeError(f"生成履歴の設定ファイルを確認できません: {exc}") from exc
    if metadata_path.parent != output_root:
        raise H3BridgeError("生成履歴の設定ファイルがForge Neoの出力フォルダー外を参照しています。")
    try:
        metadata_stat = metadata_path.stat()
        if metadata_stat.st_size > H3_HISTORY_METADATA_MAX_BYTES:
            raise H3BridgeError("生成履歴の設定ファイルが大きすぎるため読み込めません。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except H3BridgeError:
        raise
    except FileNotFoundError as exc:
        raise H3BridgeError("この生成履歴には復元用の設定ファイルがありません。") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3BridgeError(f"生成履歴の設定ファイルを読み込めません: {exc}") from exc

    if not isinstance(metadata, dict) or metadata.get("model") != "MiniMax H3":
        raise H3BridgeError("生成履歴の設定ファイルがMiniMax H3形式ではありません。")
    required = {
        "mode",
        "prompt",
        "aspect",
        "quality",
        "requested_seconds",
        "steps",
        "seed",
        "scheduler",
        "ref_image_size",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise H3BridgeError("復元用の設定が不足しています: " + ", ".join(missing))
    for name in ("mode", "prompt", "aspect", "quality", "scheduler", "ref_image_size"):
        if not isinstance(metadata[name], str):
            raise H3BridgeError(f"復元用の設定 {name} の形式が不正です。")
    if isinstance(metadata["requested_seconds"], bool) or isinstance(metadata["steps"], bool) or isinstance(
        metadata["seed"], bool
    ):
        raise H3BridgeError("復元用の数値設定の形式が不正です。")

    try:
        request = H3Request(
            mode=metadata["mode"],
            prompt=metadata["prompt"],
            aspect=metadata["aspect"],
            quality=metadata["quality"],
            duration_seconds=float(metadata["requested_seconds"]),
            steps=int(metadata["steps"]),
            seed=int(metadata["seed"]),
            scheduler=metadata["scheduler"],
            ref_image_size=metadata["ref_image_size"],
            acceleration=H3Acceleration.from_dict(metadata.get("acceleration")),
            control=fun_control.H3FunControl.from_dict(metadata.get("fun_control")),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise H3BridgeError(f"復元用の数値設定が不正です: {exc}") from exc
    if request.mode not in MODES:
        raise H3BridgeError("復元用の生成モードが不正です。")
    validate_request(
        H3Request(
            mode=MODE_TEXT,
            prompt=request.prompt,
            aspect=request.aspect,
            quality=request.quality,
            duration_seconds=request.duration_seconds,
            steps=request.steps,
            seed=request.seed,
            scheduler=request.scheduler,
            ref_image_size=request.ref_image_size,
            acceleration=request.acceleration,
        )
    )
    return request


def reference_guide_html(image_values: Any, video_values: Any, audio_values: Any) -> str:
    images = normalize_file_list(image_values)
    videos = normalize_file_list(video_values)
    audios = normalize_file_list(audio_values)
    total = len(images) + len(videos) + len(audios)
    issues: list[str] = []
    if len(images) > 9:
        issues.append("画像は9枚まで")
    if len(videos) > 3:
        issues.append("動画は3本まで")
    if len(audios) > 3:
        issues.append("単独音声は3本まで")
    if total > 12:
        issues.append("参照素材は合計12個まで")
    if (audios or videos) and not (images or videos):
        issues.append("音声だけでは生成できません")

    chips: list[str] = []
    for index, value in enumerate(images, 1):
        title = html.escape(Path(value).name, quote=True)
        chips.append(f'<code title="{title}">&lt;Picture {index}&gt;</code>')

    audio_index = 1
    total_timed_seconds = 0.0
    for video_index, value in enumerate(videos, 1):
        path = Path(value)
        has_audio = False
        try:
            path = _validate_media_path(value, "video")
            duration, has_audio, video_fps = _probe_media(path)
            total_timed_seconds += duration
            if not 2.0 <= duration <= 15.0:
                issues.append(f"{path.name}: 動画は2〜15秒")
            if video_fps is None or abs(video_fps - H3_FPS) > 0.05:
                shown_fps = "不明" if video_fps is None else f"{video_fps:.3f}"
                issues.append(f"{path.name}: 24fpsへ変換（現在 {shown_fps}fps）")
        except H3BridgeError as exc:
            issues.append(str(exc))
        title = html.escape(path.name, quote=True)
        if has_audio:
            chips.append(f'<code title="{title} · paired audio">&lt;Audio {audio_index}&gt;</code>')
            audio_index += 1
        chips.append(f'<code title="{title}">&lt;Video {video_index}&gt;</code>')

    for value in audios:
        path = Path(value)
        try:
            path = _validate_media_path(value, "audio")
            duration, _, _ = _probe_media(path)
            total_timed_seconds += duration
            if not 2.0 <= duration <= 15.0:
                issues.append(f"{path.name}: 音声は2〜15秒")
        except H3BridgeError as exc:
            issues.append(str(exc))
        title = html.escape(path.name, quote=True)
        chips.append(f'<code title="{title}">&lt;Audio {audio_index}&gt;</code>')
        audio_index += 1

    if total_timed_seconds > 15.0 + 1e-6:
        issues.append(f"動画＋音声は合計15秒まで（現在 {total_timed_seconds:.1f}秒）")

    tone = "error" if issues else "ready"
    issue_html = ""
    if issues:
        issue_html = '<span class="h3-reference-issue">' + html.escape(" / ".join(issues)) + "</span>"
    elif not chips:
        issue_html = "<span>素材を追加すると、実際の presentation 順でタグを表示します。</span>"
    else:
        issue_html = "<span>このタグをクリックせず、そのままPromptへ記述してください。</span>"
    chips_html = "".join(chips) or "<code>&lt;Picture 1&gt;</code><code>&lt;Video 1&gt;</code><code>&lt;Audio 1&gt;</code>"
    return (
        f'<div class="h3-reference-note" data-tone="{tone}" role="status" aria-live="polite">'
        f'<strong>画像 {len(images)}/9 · 動画 {len(videos)}/3 · 音声 {len(audios)}/3 · 合計 {total}/12</strong>'
        f"{chips_html}{issue_html}</div>"
    )


def cache_history_video(selected: str, items: Sequence[HistoryItem], output_directory: Path) -> str:
    selected_path = _resolve_history_selection(selected, items)
    if selected_path.parent == output_directory.resolve():
        return os.fspath(selected_path)
    source_key = hashlib.sha256(
        os.path.normcase(os.fspath(selected_path)).encode("utf-8")
    ).hexdigest()[:12]
    target = output_directory / f"Imported_{source_key}_{selected_path.name}"
    staged = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        source_stat = selected_path.stat()
        refresh = not target.exists()
        if not refresh:
            target_stat = target.stat()
            refresh = (
                target_stat.st_size != source_stat.st_size
                or target_stat.st_mtime_ns < source_stat.st_mtime_ns
            )
        if refresh:
            shutil.copy2(selected_path, staged)
            os.replace(staged, target)
    except OSError as exc:
        staged.unlink(missing_ok=True)
        raise H3BridgeError(f"履歴動画をプレイヤー用cacheへ準備できません: {exc}") from exc
    return os.fspath(target)


def readiness_html(
    readiness: RuntimeReadiness,
    expected_profile: str = RUNTIME_PROFILE_FAST,
) -> str:
    if expected_profile not in RUNTIME_PROFILES:
        expected_profile = RUNTIME_PROFILE_FAST
    selected_files = readiness.acceleration.model_files(MODEL_FILES)
    validation_error = None
    if readiness.connected:
        try:
            validate_readiness(readiness, expected_profile)
        except H3BridgeError as exc:
            validation_error = str(exc)
    connected_tone = "ready" if readiness.connected else "warn"
    connected_text = "ComfyUI 接続済み" if readiness.connected else "ComfyUI 未接続"
    files_ready = sum(1 for present in readiness.model_files.values() if present)
    total_files = len(readiness.model_files)
    server_files_ready = sum(1 for present in readiness.server_model_files.values() if present)
    server_files_total = len(MODEL_FILES)
    gpu = readiness.gpu_name or "GPU 未確認"
    if readiness.vram_gib:
        gpu += f" · {readiness.vram_gib:.0f} GiB"
    node_text = "H3 nodes OK" if readiness.connected and not readiness.missing_nodes else "H3 nodes 未確認"
    kitchen_version = readiness.package_versions.get("comfy-kitchen")
    attention_text = "Kitchen INT8" if readiness.ck_attention_available else "Attention 未確認"
    profile_matches = readiness.connected and readiness.runtime_profile == expected_profile
    runtime_text = {
        RUNTIME_PROFILE_FAST: "高速 · Async 2",
        RUNTIME_PROFILE_LOW_RAM: "省RAM · Async無効",
    }.get(readiness.runtime_profile or "", "起動設定 未確認")
    core_text = (
        f"Core {readiness.core_revision[:8]}+"
        if readiness.h3_core_optimized and readiness.core_revision
        else "H3 core 未確認"
    )
    memory_values = [
        float(value)
        for value in (readiness.ram_free_gib, readiness.commit_free_gib)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    ]
    memory_free_gib = min(memory_values) if memory_values else None
    recommended_required_gib = _estimated_required_free_gib(
        H3Request(mode=MODE_TEXT, prompt="ready-check"),
        expected_profile,
    )
    memory_low = (
        readiness.connected
        and memory_free_gib is not None
        and memory_free_gib < recommended_required_gib
    )
    memory_hint = (
        "他のアプリを閉じるか、省RAM profileまたは動作確認設定を使ってください。"
        if expected_profile == RUNTIME_PROFILE_FAST
        else "他のアプリを閉じるか、動作確認設定を使ってください。"
    )
    missing_server_files = [
        name for name in MODEL_FILES if not readiness.server_model_files.get(name)
    ] if readiness.connected else []
    details = readiness.error or validation_error or (
        "不足ノード: " + ", ".join(readiness.missing_nodes)
        if readiness.missing_nodes
        else "接続先で未検出のmodel: " + ", ".join(missing_server_files)
        if missing_server_files
        else "Comfy Kitchen INT8 attentionを利用できません。runtimeを更新してください。"
        if readiness.connected and not readiness.ck_attention_available
        else f"H3 coreを {(SPARSE_COMMIT if readiness.acceleration.attention != 'dense' else H3_MINIMUM_COMFY_COMMIT)[:12]} 以降へ更新してください。"
        if readiness.connected and not readiness.h3_core_optimized
        else (
            "選択した起動profileと接続中の設定が一致しません。"
            "「選択設定で再起動」を押してください。"
        )
        if readiness.connected and not profile_matches
        else (
            f"backendは準備完了ですが、標準5秒には約 {recommended_required_gib:.1f} GiB の余力が必要です。"
            f"{memory_hint}"
        )
        if memory_low
        else "ローカル音声付き生成の準備ができています。"
    )
    root = os.fspath(readiness.runtime_root) if readiness.runtime_root else "runtime未検出"
    gpu_detail = readiness.gpu_name or "未確認"
    ram_detail = (
        f"空き {readiness.ram_free_gib:.1f} / 合計 {readiness.ram_total_gib:.1f} GiB"
        if readiness.ram_free_gib is not None and readiness.ram_total_gib is not None
        else "未確認"
    )
    if readiness.commit_free_gib is not None:
        ram_detail += f" · commit余力 {readiness.commit_free_gib:.1f} GiB"
    memory_badge = ""
    if memory_free_gib is not None:
        memory_badge = (
            f'<span data-tone="{"warn" if memory_low else "ready"}" '
            'data-mobile="primary" '
            f'title="{html.escape(ram_detail, quote=True)}">'
            f'<i></i>RAM余力 {memory_free_gib:.1f} GiB</span>'
        )
    profile_detail = RUNTIME_PROFILE_LABELS.get(readiness.runtime_profile or "", "未対応の起動設定")
    expected_detail = RUNTIME_PROFILE_LABELS[expected_profile]
    revision_detail = readiness.core_revision or "未確認"
    runtime_ready = (
        readiness.connected
        and files_ready == total_files
        and server_files_ready == server_files_total
        and not readiness.missing_nodes
        and readiness.ck_attention_available
        and readiness.h3_core_optimized
        and profile_matches
        and not memory_low
        and not validation_error
    )
    if runtime_ready:
        summary_title = "生成できます"
        summary_tone = "ready"
    elif not readiness.connected and files_ready == total_files:
        summary_title = "生成時にH3を起動します"
        summary_tone = "warn"
    else:
        summary_title = "準備を確認してください"
        summary_tone = "error" if readiness.error else "warn"
    return (
        f'<div class="h3-runtime-card" data-tone="{summary_tone}" role="status" '
        'aria-live="polite" aria-atomic="true">'
        '<div class="h3-runtime-summary">'
        f'<span class="h3-runtime-primary" data-tone="{summary_tone}" data-mobile="primary">'
        f'<i aria-hidden="true"></i><strong>{html.escape(summary_title)}</strong></span>'
        f'<p>{html.escape(details)}</p>'
        '</div><details class="h3-runtime-details"><summary>詳細を開く</summary>'
        '<div class="h3-runtime-badges">'
        f'<span data-tone="{connected_tone}" data-mobile="primary"><i></i>{html.escape(connected_text)}</span>'
        f'<span title="{html.escape(gpu_detail, quote=True)}"><i></i>{html.escape(gpu)}</span>'
        f'<span data-tone="{"ready" if files_ready == total_files else "warn"}"><i></i>Files {files_ready}/{total_files}</span>'
        f'<span data-tone="{"ready" if server_files_ready == server_files_total else "warn"}"><i></i>Backend {server_files_ready}/{server_files_total}</span>'
        f'<span data-tone="{"ready" if readiness.connected and not readiness.missing_nodes else "warn"}"><i></i>{html.escape(node_text)}</span>'
        f'<span data-tone="{"ready" if readiness.ck_attention_available else "warn"}"><i></i>{html.escape(attention_text)}</span>'
        f'<span data-tone="{"ready" if readiness.h3_core_optimized else "warn"}" title="{html.escape(revision_detail, quote=True)}"><i></i>{html.escape(core_text)}</span>'
        f'<span data-tone="{"ready" if profile_matches else "warn"}" data-mobile="primary" title="{html.escape(profile_detail, quote=True)}"><i></i>{html.escape(runtime_text)}</span>'
        f"{memory_badge}"
        f'<span title="ComfyUI {html.escape(readiness.comfy_version or "不明", quote=True)} / comfy-kitchen {html.escape(kitchen_version or "不明", quote=True)}"><i></i>Comfy {html.escape(readiness.comfy_version or "不明")} · Kitchen {html.escape(kitchen_version or "不明")}</span>'
        '</div><dl>'
        f'<dt>GPU</dt><dd>{html.escape(gpu_detail)}</dd>'
        f'<dt>RAM</dt><dd>{html.escape(ram_detail)}</dd>'
        f'<dt>接続中profile</dt><dd>{html.escape(profile_detail)}</dd>'
        f'<dt>選択中profile</dt><dd>{html.escape(expected_detail)}</dd>'
        f'<dt>ComfyUI / Kitchen</dt><dd>{html.escape(readiness.comfy_version or "不明")} / {html.escape(kitchen_version or "不明")}</dd>'
        f'<dt>Core revision</dt><dd><code>{html.escape(revision_detail)}</code></dd>'
        f'<dt>Runtime</dt><dd><code title="{html.escape(root, quote=True)}">{html.escape(root)}</code></dd>'
        + "".join(
            f"<dt>{html.escape(name)}</dt><dd><code>{html.escape(filename)}</code></dd>"
            for name, (_, filename) in selected_files.items()
        )
        + '</dl></details>'
        "</div>"
    )


GENERATION_PRESETS: dict[str, tuple[str, float, int]] = {
    "quick": ("draft", 5.0, 20),
    "recommended": ("preview", 5.0, 20),
    "final": ("native", 5.0, 20),
}


def generation_preset_values(
    preset: str,
    aspect: str,
) -> tuple[str, float, int, str, str, str]:
    try:
        quality, duration, steps = GENERATION_PRESETS[preset]
    except KeyError as exc:
        raise H3BridgeError(f"未対応の生成プリセットです: {preset}") from exc
    scheduler = "simple"
    ref_image_size = "match"
    return (
        quality,
        duration,
        steps,
        scheduler,
        ref_image_size,
        settings_summary_html(
            aspect,
            quality,
            duration,
            steps,
            scheduler,
            ref_image_size,
        ),
    )


def relative_workload(aspect: str, quality: str, duration: float, steps: int) -> float:
    width, height = dimensions_for(aspect, quality)
    frames = snap_h3_frames(duration)
    baseline = 864 * 480 * 124 * 20
    return (width * height * frames * int(steps)) / baseline


def settings_summary_html(
    aspect: str,
    quality: str,
    duration: float,
    steps: int,
    scheduler: str = "simple",
    ref_image_size: str = "match",
) -> str:
    try:
        width, height = dimensions_for(aspect, quality)
        frames = snap_h3_frames(duration)
        effective = frames / H3_FPS
        workload = relative_workload(aspect, quality, duration, steps)
        official_preview = (
            quality == "preview"
            and frames == 124
            and int(steps) == 20
            and scheduler == "simple"
            and ref_image_size == "match"
        )
        tone = (
            "warn"
            if quality == "native"
            or scheduler != "simple"
            or ref_image_size == "max"
            or workload >= 2.0
            else "ready"
        )
        if ref_image_size == "max":
            note = "Reference Maxは非常に重い設定です。まずMatchで内容を確認してください。"
        elif scheduler != "simple":
            note = "実験的schedulerです。再現性重視ならsimpleへ戻してください。"
        elif quality == "native":
            note = "Native最終出力です。RTX 3090では生成時間とRAM使用量が大きく増えます。"
        elif workload >= 5.0:
            note = "非常に重い設定です。まず「標準」で構図と音を確認してください。"
        elif workload >= 2.0:
            note = "重い設定です。RTX 3090では生成時間が大きく伸びます。"
        elif workload < 0.75:
            note = "高速な動作確認向けです。解像度は低くなります。"
        elif official_preview:
            note = "公式Fast Preview相当の標準設定です。"
        else:
            note = "H3の32pxキャンバスと17-frameグリッドへ整列済みです。"
        return (
            f'<div class="h3-settings-summary" data-tone="{tone}" role="status" aria-live="polite">'
            f'<strong>{width} × {height}</strong><span>{frames} frames · {effective:.2f} sec · {int(steps)} steps · 24fps stereo · 相対負荷 {workload:.2f}×</span>'
            f'<small>{html.escape(note)} 相対負荷は所要時間の予測ではありません。</small></div>'
        )
    except (H3BridgeError, TypeError, ValueError, OverflowError) as exc:
        return f'<div class="h3-settings-summary" data-tone="error" role="alert">{html.escape(str(exc))}</div>'


def progress_html(stage: str, message: str, progress: float = 0.0, elapsed: float | None = None) -> str:
    try:
        progress = float(progress)
    except (TypeError, ValueError, OverflowError):
        progress = 0.0
    if not math.isfinite(progress):
        progress = 0.0
    progress = min(1.0, max(0.0, progress))
    percent = int(progress * 100)
    elapsed_text = ""
    if elapsed is not None:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_text = f" · {minutes:02d}:{seconds:02d}"
    stage_labels = {
        "idle": "待機中",
        "validation": "入力修正待ち",
        "runtime": "接続確認中",
        "prepare": "準備中",
        "queued": "キュー待ち",
        "running": "生成中",
        "reconnecting": "再接続中",
        "complete": "完了",
        "cancelled": "停止済み",
        "error": "エラー",
        "active": "処理中",
    }
    stage_label = stage_labels.get(stage, stage)
    tone = "ready" if stage == "complete" else "error" if stage == "error" else "active"
    if stage == "running":
        progress_track = (
            '<div class="h3-progress-track" role="progressbar" aria-label="MiniMax H3 progress" '
            f'aria-valuetext="{html.escape(message, quote=True)}"><i></i></div>'
        )
    else:
        progress_track = (
            '<div class="h3-progress-track" role="progressbar" aria-label="MiniMax H3 progress" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
            f'<i style="width:{percent}%"></i></div>'
        )
    return (
        f'<div class="h3-progress" data-tone="{tone}" data-stage="{html.escape(stage, quote=True)}"'
        '>'
        '<div class="h3-progress-copy">'
        f'<strong>{html.escape(message)}</strong><span>{html.escape(stage_label)}{elapsed_text}</span></div>'
        f"{progress_track}</div>"
    )


def append_prompt_section(prompt: str, section: str) -> str:
    templates = {
        "camera": "Camera: ゆっくり安定したプッシュイン。自然な動きと明確な構図。",
        "dialogue": 'Dialogue: 「」を自然に発話。正確なリップシンク。',
        "sfx": "SFX: 空間が伝わる環境音と、動きに同期した細かな効果音。",
        "music": "Music: 編集の流れに寄り添い、最後にきれいに収束する控えめな劇伴。",
    }
    addition = templates.get(section)
    if addition is None:
        return prompt
    prompt = (prompt or "").rstrip()
    if addition in prompt:
        return prompt
    return f"{prompt}\n\n{addition}".lstrip()


def prompt_template(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if any(marker in prompt for marker in ("Scene:", "Camera:", "Audio:")):
        return prompt
    scene = prompt or "主役、場所、画の質感、起こしたい動きを書いてください。"
    return (
        f"Scene: {scene}\n\n"
        "Shot plan:\n"
        "[0s-2s] 主役と場所を一目で伝える。\n"
        "[2s-4s] 連続性を保ちながら中心の動きを展開する。\n"
        "[4s-end] 意図のある最後の画で締める。\n\n"
        "Camera: 構図、レンズ感、カメラ移動、カットを書く。\n"
        "Audio: 台詞、環境音、効果音、音楽をショットと同じ時系列で書く。"
    )
