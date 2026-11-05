"""CPU-only validation of a selected YuE2 checkpoint in its pinned environment."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

from .core import YuE2Error, official_weight_files

_DIMENSIONS = {
    "hidden_size": (2048, 16384),
    "num_hidden_layers": (28, 128),
    "num_attention_heads": (16, 128),
    "num_key_value_heads": (8, 128),
    "head_dim": (128, 1024),
    "intermediate_size": (6144, 65536),
    "vocab_size": (184704, 1000000),
    "max_position_embeddings": (24576, 1000000),
    "latent_dim": (64, 4096),
    "max_latent_frames": (24576, 1000000),
}
_FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise YuE2Error(f"YuE2のJSONに重複するキーがあります: {key}")
        result[key] = value
    return result


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=lambda _: _invalid_json())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise YuE2Error("YuE2のJSONを読み取れません。") from error


def _invalid_json():
    raise YuE2Error("YuE2のJSONに非有限値があります。")


def validate_config(config):
    if not isinstance(config, dict) or config.get("model_type") != "yue2":
        raise YuE2Error("YuE2のモデル設定が不正です。")
    if config.get("tie_word_embeddings", False) is not False or config.get("latent_type", "vae") != "vae":
        raise YuE2Error("YuE2の共有語彙またはlatent設定には対応していません。")
    dimensions = {}
    for name, (default, maximum) in _DIMENSIONS.items():
        value = config.get(name, default)
        if type(value) is not int or not 0 < value <= maximum:
            raise YuE2Error(f"YuE2の構造設定が範囲外です: {name}")
        dimensions[name] = value
    if (dimensions["num_attention_heads"] % dimensions["num_key_value_heads"]
            or dimensions["head_dim"] % 2
            or dimensions["num_attention_heads"] * dimensions["head_dim"] > 32768):
        raise YuE2Error("YuE2のattention設定が一致しません。")
    for name, default in (("rms_norm_eps", 1e-6), ("rope_theta", 1000000.0), ("timestep_shift", 1.0)):
        value = config.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise YuE2Error(f"YuE2の数値設定が不正です: {name}")


def _read_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise YuE2Error("YuE2のsafetensorsヘッダーが不完全です。")
        size = struct.unpack("<Q", prefix)[0]
        file_size = os.fstat(stream.fileno()).st_size
        if not 2 <= size <= 32 * 1024 * 1024 or size + 8 > file_size:
            raise YuE2Error("YuE2のsafetensorsヘッダーが不正です。")
        header = _json(stream.read(size))
    if not isinstance(header, dict):
        raise YuE2Error("YuE2のsafetensorsヘッダーが不正です。")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or metadata.get("format", "pt") != "pt":
        raise YuE2Error("YuE2のsafetensors保存形式が未対応です。")
    ranges = []
    for name, tensor in header.items():
        if not isinstance(tensor, dict):
            raise YuE2Error(f"YuE2のテンソル情報が不正です: {name}")
        shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
        if (not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape)
                or dtype not in _FLOAT_BYTES or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(item) is not int for item in offsets)):
            raise YuE2Error(f"YuE2のテンソル形式が未対応です: {name}")
        start, end = offsets
        if start < 0 or end < start or end - start != math.prod(shape) * _FLOAT_BYTES[dtype]:
            raise YuE2Error(f"YuE2のテンソルサイズが不正です: {name}")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise YuE2Error("YuE2のテンソル範囲が重複または欠落しています。")
        cursor = end
    if cursor != file_size - size - 8:
        raise YuE2Error("YuE2のsafetensorsデータ長が一致しません。")
    return header


def validate_tensor_headers(directory, expected):
    directory = Path(directory)
    files = official_weight_files(directory)
    index_path = directory / "model.safetensors.index.json"
    mapping = None
    if index_path.is_file():
        if index_path.stat().st_size > 32 * 1024 * 1024:
            raise YuE2Error("YuE2の重み索引が大きすぎます。")
        index = _json(index_path.read_bytes())
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict):
            raise YuE2Error("YuE2の重み索引が不正です。")
    actual = set()
    for name in sorted(files):
        for key, tensor in _read_header(directory / name).items():
            if key in actual:
                raise YuE2Error(f"YuE2のテンソルが重複しています: {key}")
            if mapping is not None and mapping.get(key) != name:
                raise YuE2Error(f"YuE2の重み索引と所属が一致しません: {key}")
            if key not in expected or tuple(tensor["shape"]) != tuple(expected[key]):
                raise YuE2Error(f"YuE2のテンソル名または形状が一致しません: {key}")
            actual.add(key)
    if actual != set(expected) or (mapping is not None and set(mapping) != actual):
        missing = sorted(set(expected) - actual)[:5]
        raise YuE2Error(f"YuE2のテンソルまたは索引が不完全です: {', '.join(missing)}")


def validate_checkpoint(directory):
    directory = Path(directory)
    config_path = directory / "config.json"
    if config_path.stat().st_size > 512 * 1024:
        raise YuE2Error("YuE2のモデル設定が大きすぎます。")
    config = _json(config_path.read_bytes())
    validate_config(config)
    import torch
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    if torch.cuda.is_initialized():
        raise YuE2Error("YuE2のCPU検査中にGPUが初期化されています。")
    with torch.device("meta"):
        model = YuE2ForCausalLM(YuE2Config(**config))
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    if torch.cuda.is_initialized():
        raise YuE2Error("YuE2のCPU検査中にGPUが初期化されています。")
    validate_tensor_headers(directory, expected)


if __name__ == "__main__":
    from .worker import parent_guard

    try:
        parent_guard()
        if len(sys.argv) != 2:
            raise YuE2Error("YuE2の本体フォルダーを指定してください。")
        validate_checkpoint(sys.argv[1])
    except Exception as error:
        print(f"YuE2の本体を検証できません: {error}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from error
