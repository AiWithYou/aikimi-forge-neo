"""Validate and stream one float Qwen checkpoint into a temporary Diffusers folder."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path

from modules_forge import local_assets
from modules_forge.qwen_image21.local_source import TRANSFORMER_CONFIG, validate_folder_tensors

RECIPE = "qwen21-single-to-diffusers-v1"
CHUNK_BYTES = 8 * 1024 * 1024
DISK_MARGIN = 64 * 1024 * 1024


def source_guard(path, expected_identity, *, check_cancel=lambda: None):
    path = local_assets.local_path(str(path), suffixes=(".safetensors",))
    version = local_assets.file_version(path)

    def check():
        check_cancel()
        if local_assets.file_version(path) != version:
            raise ValueError("選択した本体が変更されました。再実行してください。")

    def verify():
        check()
        if local_assets.file_identity(path, check_cancel=check) != expected_identity:
            raise ValueError("選択した本体が変更されました。再実行してください。")
        check()

    verify()
    return check, verify


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"safetensorsのキーが重複しています: {key}")
        result[key] = value
    return result


def _plan(path, model_class):
    from accelerate import init_empty_weights

    # local_assets verifies bounds; the second parse detects duplicate JSON keys.
    local_assets.read_header(path)
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size), object_pairs_hook=_unique_object)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or "_quantization_metadata" in metadata:
        raise ValueError("事前量子化モデルの再量子化には対応しません。")
    with init_empty_weights():
        expected = {name: tuple(value.shape) for name, value in model_class.from_config(TRANSFORMER_CONFIG).state_dict().items()}

    result, ranges = {}, []
    for original, item in header.items():
        if not isinstance(item, dict) or item.get("dtype") not in {"BF16", "F16", "F32"}:
            raise ValueError("未量子化のBF16・FP16・FP32本体を選択してください。")
        shape, offsets = item.get("shape"), item.get("data_offsets")
        if (not isinstance(shape, list) or any(type(value) is not int or value < 0 for value in shape)
                or not isinstance(offsets, list) or len(offsets) != 2):
            raise ValueError("safetensorsの形状が不正です。")
        start, end = offsets
        element_bytes = 4 if item["dtype"] == "F32" else 2
        if end - start != math.prod(shape) * element_bytes:
            raise ValueError(f"safetensorsのデータ長が一致しません: {original}")
        ranges.append((start, end))
        name = original.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
        if name.endswith(".img_mlp.gate_up.weight"):
            if not shape or shape[0] % 2:
                raise ValueError(f"gate_upの寸法が不正です: {original}")
            split = start + (end - start) // 2
            half_shape = [shape[0] // 2, *shape[1:]]
            entries = ((name.replace("gate_up", "gate_layer"), half_shape, start, split),
                       (name.replace("gate_up", "proj"), half_shape, split, end))
        else:
            entries = ((name, shape, start, end),)
        for key, dimensions, first, last in entries:
            if key in result:
                raise ValueError(f"正規化後のテンソルが重複しています: {key}")
            result[key] = (item["dtype"], dimensions, first, last)
    cursor = 0
    for first, last in sorted(ranges):
        if first != cursor:
            raise ValueError("safetensorsのテンソル範囲が重複または欠落しています。")
        cursor = last
    if cursor != path.stat().st_size - 8 - header_size:
        raise ValueError("safetensorsのデータ長が一致しません。")
    if set(result) != set(expected) or any(tuple(item[1]) != expected[name] for name, item in result.items()):
        raise ValueError("ローカル本体のテンソル名・寸法がQwen Image 2.1と一致しません。部分読み込みは行いません。")
    return header_size, result


@contextmanager
def staged_transformer(path, model_class, runtime, expected_identity, *, check_cancel=lambda: None, plan=None):
    """Keep the original unchanged; clean up this job's temp folder on exit."""
    path = local_assets.local_path(str(path), suffixes=(".safetensors",))
    check, verify = source_guard(path, expected_identity, check_cancel=check_cancel)
    header_size, tensors = plan if plan is not None else _plan(path, model_class)
    check()
    root = Path(runtime).resolve()
    temporary_root = root / "tmp"
    if temporary_root.is_symlink() or not temporary_root.resolve().is_relative_to(root):
        raise ValueError("一時フォルダーの保存先が不正です。")
    temporary_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(temporary_root).free < path.stat().st_size + DISK_MARGIN:
        raise OSError("単一ファイルの変換に必要な空き容量がありません。")
    with tempfile.TemporaryDirectory(prefix="qwen21-transformer-", dir=temporary_root) as temporary:
        folder = Path(temporary)
        (folder / "config.json").write_text(json.dumps(TRANSFORMER_CONFIG), encoding="utf-8")
        output_header = {"__metadata__": {"format": "pt"}}
        offset = 0
        for name, (dtype, dimensions, first, last) in tensors.items():
            output_header[name] = {"dtype": dtype, "shape": dimensions, "data_offsets": [offset, offset + last - first]}
            offset += last - first
        encoded = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with path.open("rb") as source, (folder / "diffusion_pytorch_model.safetensors").open("xb") as target:
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for _, _, first, last in tensors.values():
                source.seek(8 + header_size + first)
                remaining = last - first
                while remaining:
                    check()
                    block = source.read(min(remaining, CHUNK_BYTES))
                    if not block:
                        raise ValueError("本体の読み込み中にデータが不足しました。")
                    target.write(block)
                    remaining -= len(block)
            target.flush()
            os.fsync(target.fileno())
        check()
        validate_folder_tensors(folder, model_class)
        verify()
        yield folder
        verify()
