"""Union 2.0 model contracts and explicitly selected native H3 VAE optimizations.

No Torch import, GPU allocation, installation, or network calls at import time.
Contracts: ComfyUI PR #16471 (Union v2) and #16485 (offloaded VAE fix).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO

UNION2_COMMIT = "95539f56344958339e39b7582a476267d489b0ee"
VAE_FIXED_COMMIT = "912fca4f39b875a0360f2c5170568176ea813ded"
UNION2_MODEL = "minimax_h3_fun_controlnet_union_v2_pruned.safetensors"
OPTIMIZED_MODES = frozenset({"fp16_accumulation"})
CONTROL_CHOICES = [
    ("オフ", "off"),
    ("Union 1 · 元動画からCanny", "canny"),
    ("Union 1 · 前処理済み動画", "preprocessed"),
    ("Union 2.0 · 元動画からCanny", "v2_canny"),
    ("Union 2.0 · 元動画からGray", "v2_gray"),
    ("Union 2.0 · 前処理済み動画（8条件）", "v2_preprocessed"),
]
DECODE_CHOICES = [
    ("VAEDecode · 標準（起動中のCore）", "standard"),
    ("更新版 ＋ FP16積算 · 精度との比較用", "fp16_accumulation"),
    ("Fast VAE Decode · 外部ノード", "fast"),
]
MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_FILE_BYTES = 32 * 1024**3
_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"safetensorsヘッダーに重複キーがあります: {key}")
        result[key] = value
    return result


def read_header(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("safetensorsのヘッダーを読み込めません。")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= MAX_HEADER_BYTES:
        raise ValueError("safetensorsヘッダーのサイズが不正です。")
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("safetensorsヘッダーが途中で切れています。")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensorsヘッダーの形式が不正です。") from exc
    if not isinstance(data, dict):
        raise ValueError("safetensorsヘッダーはオブジェクトが必要です。")
    return data, length + 8


def validate_tensor_table(header: dict[str, Any], payload_bytes: int | None = None) -> None:
    intervals = []
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in entry.items()
            ):
                raise ValueError("safetensorsメタデータが不正です。")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"テンソル情報が不正です: {name}")
        shape, offsets, dtype = entry.get("shape"), entry.get("data_offsets"), entry.get("dtype")
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"テンソル形状が不正です: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f"テンソルoffsetが不正です: {name}")
        begin, end = offsets
        if not 0 <= begin <= end <= _MAX_FILE_BYTES or not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            raise ValueError(f"テンソル範囲またはdtypeが未対応です: {name}")
        count = math.prod(shape)
        if count * _DTYPE_BITS[dtype] != (end - begin) * 8:
            raise ValueError(f"テンソル形状とバイト数が一致しません: {name}")
        if end > begin:
            intervals.append((begin, end))
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("safetensorsデータに重複または欠落があります。")
        cursor = end
    if not intervals or (payload_bytes is not None and cursor != payload_bytes):
        raise ValueError("safetensorsが空、またはファイル末尾とテンソル範囲が一致しません。")


def validate_union2_header(header: dict[str, Any]) -> str:
    """Only native, 10-block AdaLN-basis v2 patches match Neo's pruned base.

    Returns bf16/int8. W4A8/raw VideoX-Fun/full-AdaLN files are not relabelled.
    This verifies a structural contract, not learned quality or GPU execution.
    """
    validate_tensor_table(header)
    metadata = header.get("__metadata__", {})
    if metadata.get("minimax_h3_fun_controlnet") != "adaln_basis":
        raise ValueError(
            "NeoのprunedモデルにはAdaLN-basis変換済みUnion 2.0が必要です。元のVideoX-Fun重みは使用しません。"
        )
    if metadata.get("inpaint_masked_pixel_mode") != "post_norm":
        raise ValueError("Union 2.0のpost_normメタデータがありません。v1からの改名では使用できません。")
    blocks = {int(match.group(1)) for key in header if (match := re.match(r"^control_blocks\.(\d+)\.", key))}
    if blocks != set(range(10)):
        raise ValueError("Union 2.0は連続した10個の制御ブロックが必要です。")
    if "control_blocks_places" in metadata:
        try:
            places = json.loads(metadata["control_blocks_places"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("制御ブロック配置のメタデータが不正です。") from exc
        if places != list(range(0, 50, 5)) or any(type(x) is not int for x in places):
            raise ValueError("Union 2.0の制御層は0,5,...,45である必要があります。")

    def tensor(name, dims=None):
        item = header.get(name)
        if not isinstance(item, dict) or (dims is not None and len(item.get("shape", [])) != dims):
            raise ValueError(f"ネイティブUnion 2.0の必須テンソルがありません: {name}")
        return item

    projection = tensor("control_proj_in.weight", 2)
    hidden = projection["shape"][0]
    if hidden != 5376 or projection["shape"][1] != 49 * 2 * 2:
        raise ValueError("Union 2.0の入力次元が49chのH3用ではありません。")
    for i in range(10):
        prefix = f"control_blocks.{i}."
        adaln = tensor(prefix + "adaln_proj.linear.weight", 2)
        if adaln["shape"][1] != 8 or adaln["shape"][0] <= 0 or adaln["shape"][0] % hidden:
            raise ValueError("制御ブロックのAdaLNがpruned H3の8次元basisと一致しません。")
        if tensor(prefix + "after_proj.weight", 2)["shape"] != [hidden, hidden]:
            raise ValueError("制御出力の形状がH3と一致しません。")
        qkv = tensor(prefix + "attn.qkv_proj.weight", 2)
        head = tensor(prefix + "attn.q_norm.weight", 1)
        if qkv["shape"][1] != hidden or head["shape"] != [128] or qkv["shape"][0] != 56 * 128 * 3:
            raise ValueError("ControlNet Attentionの形状がH3と一致しません。")
        tensor(prefix + "mlp.fc1.weight", 2)
        tensor(prefix + "mlp.fc2.weight", 2)
    tensor("control_blocks.0.before_proj.weight", 2)
    dtypes = {header[f"control_blocks.{i}.attn.qkv_proj.weight"]["dtype"] for i in range(10)}
    if dtypes <= {"BF16", "F16", "F32"}:
        return "bf16"
    if "I8" in dtypes and dtypes <= {"I8", "BF16", "F16", "F32"} and any(k.endswith(".comfy_quant") for k in header):
        # Mixed-precision loaders read the comfy_quant JSON tensor themselves.
        return "int8"
    raise ValueError("この追加機能で未確認の量子化形式です。BF16またはINT8 ConvRot版を使用してください。")


def inspect_union2(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Union 2.0の通常ファイルを指定してください。")
    stat = path.stat()
    with path.open("rb") as stream:
        header, start = read_header(stream)
    validate_tensor_table(header, stat.st_size - start)
    precision = validate_union2_header(header)
    return {
        "precision": precision,
        "size": stat.st_size,
        "blocks": 10,
        "injection_layers": list(range(0, 50, 5)),
        "inpaint_masked_pixel_mode": "post_norm",
    }


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def strip_selected_fast(arguments: Sequence[str]) -> tuple[str, ...]:
    """Accept only the single explicit FP16 accumulation flag, never bare --fast."""
    values = tuple(arguments)
    indices = [i for i, value in enumerate(values) if value == "--fast" or value.startswith("--fast=")]
    if not indices:
        return values
    if len(indices) != 1:
        raise ValueError("--fastの重複指定は許可しません。")
    i = indices[0]
    if values[i] != "--fast" or i + 1 >= len(values) or values[i + 1] != "fp16_accumulation":
        raise ValueError("許可する高速化フラグは --fast fp16_accumulation だけです。")
    if i + 2 < len(values) and not values[i + 2].startswith("--"):
        raise ValueError("FP16積算以外の--fastオプションは許可しません。")
    return values[:i] + values[i + 2 :]


def has_fp16_accumulation(arguments: Sequence[str]) -> bool:
    return strip_selected_fast(arguments) != tuple(arguments)


def _at_least(version: str, expected: tuple[int, ...]) -> bool:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\+[^\s]+)?$", version or "")
    return bool(match and tuple(map(int, match.groups())) >= expected)


def check_runtime(readiness, *, decode_mode: str = "standard", union2: bool = False, ancestor=None) -> None:
    """Called before submission/export. Fail visibly; do not silently downgrade."""
    if not getattr(readiness, "connected", True):
        return  # validate_readiness reports the connection error next.
    selected_fast = decode_mode == "fp16_accumulation"
    if has_fp16_accumulation(readiness.runtime_args) != selected_fast:
        raise ValueError("FP16積算の選択と起動引数が一致しません。「選択設定で再起動」を実行してください。")
    # Native optimizations come from the Core update, not a duplicate sampler
    # option. Once the opt-in runtime is active, standard decode checks it too.
    upgraded = preferred_python(Path(readiness.runtime_root)) is not None if readiness.runtime_root else False
    if not union2 and decode_mode not in OPTIMIZED_MODES and not upgraded:
        return
    commit = VAE_FIXED_COMMIT if decode_mode in OPTIMIZED_MODES or upgraded else UNION2_COMMIT
    if ancestor is None:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=readiness.runtime_root,
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        ready = result.returncode == 0
    else:
        ready = ancestor(commit)
    if not ready:
        raise ValueError(
            f"H3専用ComfyUIに必要な更新がありません（{commit[:12]}以降）。tools/upgrade_minimax_h3_union2_vae.pyで確認・更新してください。"
        )
    if decode_mode in OPTIMIZED_MODES or upgraded or union2:
        packages = readiness.package_versions
        if not _at_least(packages.get("comfy-kitchen", ""), (0, 2, 35)) or not _at_least(
            packages.get("comfy-aimdo", ""), (0, 5, 5)
        ):
            raise ValueError("更新版VAEにはcomfy-kitchen 0.2.35以上・comfy-aimdo 0.5.5以上の専用環境が必要です。")


def preferred_python(runtime_root: Path) -> Path | None:
    """The updater writes this receipt last. The original .venv stays intact."""
    pending = runtime_root.parent / "union2-vae-update-in-progress.json"
    if pending.exists() or pending.is_symlink():
        raise ValueError("H3更新が未完了です。更新ログを確認してください。未確定の環境では起動しません。")
    receipt = runtime_root.parent / "union2-vae-runtime.json"
    if not receipt.exists() and not receipt.is_symlink():
        return None
    if receipt.is_symlink() or receipt.stat().st_size > 65536:
        raise ValueError("更新版H3の導入記録が不正です。")
    data = json.loads(receipt.read_text("utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("revision") != VAE_FIXED_COMMIT:
        raise ValueError("更新版H3の導入記録の版が一致しません。")
    python = runtime_root.parent / ".venv-union2-vae" / "Scripts" / "python.exe"
    if not python.is_file() or python.is_symlink() or python.resolve() != python.absolute():
        raise ValueError("更新版H3の専用Pythonがありません。更新を完了するかロールバックしてください。")
    return python
