"""Local Iris checkpoint candidates, validated without loading tensors or models."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

from modules_forge import local_assets

_INT8 = "iris-rowwise-int8-v1"
_W4A8 = "iris-convrot-w4a8-v1"
_FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_BYTES = {**_FLOAT_BYTES, "I8": 1, "F8_E4M3": 1}
_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


def _unquote(value):
    if isinstance(value, str):
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
    return value


def _configuration(root, path):
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"隣接するconfig.yamlを読み込めません: {path}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict) or not raw["model"]:
        raise ValueError("config.yamlには空でないmodel設定が必要です。")
    # The pinned upstream generation schema has no task marker. Its downstream
    # checkpoints add task.name=depth/restoration, which must never enter this UI.
    if "task" in raw:
        raise ValueError("画像生成用の設定を指定してください。task設定には対応していません。")
    encoder = raw.get("text_encoder", {})
    if not isinstance(encoder, dict):
        raise ValueError("text_encoder設定は共有Qwenエンコーダーに対応する辞書が必要です。")
    expected = {"name": "qwen3_vl", "dim": 2560, "max_length": 300, "hidden_layers": _LAYERS}
    for key, value in expected.items():
        actual = encoder.get(key, value)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"共有エンコーダーに対応しないtext_encoder.{key}です。")
        if key == "hidden_layers" and any(type(layer) is not int for layer in actual):
            raise ValueError("共有エンコーダーに対応しないtext_encoder.hidden_layersです。")
    pretrained = encoder.get("pretrained", "Qwen/Qwen3-VL-4B-Instruct")
    if pretrained != "Qwen/Qwen3-VL-4B-Instruct":
        if not isinstance(pretrained, str) or not pretrained.strip() or "://" in pretrained:
            raise ValueError("共有エンコーダーに対応しないtext_encoder.pretrainedです。")
        reference = Path(_unquote(pretrained)).expanduser()
        if not reference.is_absolute():
            reference = local_assets.ROOT / reference
        if reference.resolve() != (root / "text-encoder").resolve():
            raise ValueError("共有エンコーダーに対応しないtext_encoder.pretrainedです。")
    for key, expected_value in (
        ("text_dim", 2560),
        ("text_len", 300),
        ("text_adapter", "lap_blocks2"),
        ("text_lap_num_layers", len(_LAYERS)),
    ):
        value = raw["model"].get(key, expected_value)
        if type(value) is not type(expected_value) or value != expected_value:
            raise ValueError(f"共有エンコーダーに対応しないmodel.{key}です。")
    return raw


def _tensors(header):
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    for key, tensor in tensors.items():
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        offsets = tensor["data_offsets"]
        if (
            not isinstance(shape, list)
            or any(type(size) is not int or size <= 0 for size in shape)
            or dtype not in _BYTES
            or offsets[1] - offsets[0] != math.prod(shape) * _BYTES[dtype]
        ):
            raise ValueError(f"非対応または不完全なIris浮動小数・量子化テンソルです: {key}")
    for prefix in ("s_embedder.proj", "t_embedder.mlp.0"):
        found = [tensors[key] for key in (prefix + ".weight", prefix + ".qweight") if key in tensors]
        if not found or any(len(tensor["shape"]) != 2 for tensor in found):
            raise ValueError("IrisのFull FTチェックポイントではありません。")
    final = tensors.get("final_layer.linear.weight", {})
    if final.get("dtype") not in _FLOAT_BYTES or len(final.get("shape", [])) != 2:
        raise ValueError("IrisのFull FTチェックポイントではありません。")
    if not any(key.startswith("blocks.0.") for key in tensors):
        raise ValueError("IrisのFull FTチェックポイントではありません。")
    return tensors


def _precision(header):
    tensors = _tensors(header)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Irisの保存形式メタデータが不正です。")
    marker = metadata.get("format")
    if marker not in (None, "pt", _INT8, _W4A8):
        raise ValueError("対応しないIrisの保存形式です。")
    quantized = [key.removesuffix(".qweight") for key in tensors if key.endswith(".qweight")]
    precision = "w4a8" if marker == _W4A8 else "int8" if quantized else "normal"
    if (
        (precision == "normal" and marker not in (None, "pt"))
        or (precision == "int8" and marker not in (None, _INT8))
        or (precision == "w4a8" and not quantized)
    ):
        raise ValueError("Irisの保存形式と量子化テンソルが一致しません。")
    for key, value in metadata.items():
        if "quant" in key.lower() and (precision == "normal" or value != {"int8": _INT8, "w4a8": _W4A8}[precision]):
            raise ValueError("対応しない量子化メタデータです。")
        if key.lower() == "precision" and value != precision:
            raise ValueError("Irisの保存形式とprecisionメタデータが一致しません。")
    allowed = _FLOAT_BYTES if precision != "w4a8" else {**_FLOAT_BYTES, "F8_E4M3": 1}
    for key, tensor in tensors.items():
        if not key.endswith(".qweight") and tensor["dtype"] not in allowed:
            raise ValueError(f"対応しない量子化テンソルです: {key}")
        if precision != "w4a8" and key.endswith((".s_channel", ".codebook")):
            raise ValueError("Irisの保存形式と量子化テンソルが一致しません。")
    for name in quantized:
        weight, scale = tensors[name + ".qweight"], tensors.get(name + ".scale", {})
        shape = weight["shape"]
        if name + ".weight" in tensors or weight["dtype"] != "I8" or len(shape) != 2:
            raise ValueError(f"Irisの量子化重みが不正です: {name}")
        rows, columns = shape
        if precision == "int8":
            valid = scale.get("dtype") == "F32" and scale.get("shape") == [rows, 1]
        else:
            channel = tensors.get(name + ".s_channel", {})
            codebook = tensors.get(name + ".codebook")
            valid = (
                rows % 16 == 0
                and columns % 128 == 0
                and scale.get("dtype") == "F8_E4M3"
                and scale.get("shape") == [rows, columns // 8]
                and channel.get("dtype") == "F32"
                and channel.get("shape") == [rows]
                and (codebook is None or (codebook.get("dtype") == "F32" and codebook.get("shape") == [16]))
            )
        if not valid:
            raise ValueError(f"Irisの{'INT8' if precision == 'int8' else 'W4A8'}重み・scaleが一致しません: {name}")
    return precision


def resolve(root, value) -> dict:
    """Validate a candidate only; strict runtime loading remains authoritative."""
    root = Path(root).resolve()
    value = _unquote(value)
    try:
        directory = local_assets.local_path(value, directory=True)
    except ValueError:
        weights = local_assets.local_path(value, suffixes=(".safetensors",))
    else:
        files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".safetensors")
        if len(files) > 1:
            raise ValueError("フォルダーに複数の重みがあります。使用するファイルを指定してください。")
        if not files:
            raise ValueError(f"safetensorsファイルがありません: {directory}")
        weights = files[0].resolve()
    config = weights.with_name("config.yaml")
    raw = _configuration(root, config)
    try:
        precision = _precision(local_assets.read_header(weights))
    except (OSError, UnicodeError, TypeError, KeyError) as exc:
        raise ValueError(f"Irisのsafetensorsヘッダーを読み込めません: {weights}") from exc
    result = {"weights": weights, "config": config, "precision": precision, "raw": raw}
    distill = weights.with_name("distill.yaml")
    if distill.exists():
        from .official_turbo import read_settings

        result["sampling"] = read_settings(distill)
        flow = raw.get("flow", {})
        if not isinstance(flow, dict) or flow.get("prediction", "v") not in {"v", "x"}:
            raise ValueError("公式Turboにはvまたはxのflow predictionが必要です。")
    return result


def discover(root, directory) -> list[tuple[str, str]]:
    """Recursively list structurally plausible candidates with distinct labels."""
    if directory == "" or directory is None:
        folder = Path(root).resolve() / "checkpoints"
        if not folder.exists():
            return []
        folder = local_assets.local_path(str(folder), directory=True)
    else:
        folder = local_assets.local_path(_unquote(directory), directory=True)
    candidates = {}
    for current, directories, files in os.walk(folder, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name.lower() not in {".git", ".cache", "worker-env"} and not name.lower().startswith("iris-source-")
        )
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix.lower() != ".safetensors":
                continue
            try:
                weights = resolve(root, str(path))["weights"]
            except (OSError, ValueError):
                continue
            candidates[str(weights)] = f"{weights.name} · {weights.parent}"
    return [(candidates[path], path) for path in sorted(candidates)]
