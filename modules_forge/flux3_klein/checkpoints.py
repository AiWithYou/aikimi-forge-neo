"""FLUX 3 Full FT files: exact architecture, format and content identities."""

import json
import math
import struct
from pathlib import Path, PurePosixPath

from modules_forge import local_assets

from .core import MODEL_REVISION, Flux3Error
from .paths import canonical_path

W4_CONFIG = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}


def expected_shapes(model):
    shapes = {}
    for stream in ("image", "image_cond", "txt"):
        for group in ("early_stream_modulations", "single_stream_modulations"):
            shapes[f"{group}.{stream}.lin.weight"] = (9216, 3072)
    for stream in ("image", "image_cond"):
        shapes[f"emb_in.{stream}.weight"] = (3072, 128)
    shapes.update(
        {
            "final_layer.image.adaLN_modulation.1.weight": (6144, 3072),
            "final_layer.image.linear.weight": (128, 3072),
            "time_in.in_layer.weight": (3072, 256),
            "time_in.out_layer.weight": (3072, 3072),
            "txt_in.weight": (3072, 20480),
        }
    )
    if model == "distilled":
        shapes.update({"ref_time_in.in_layer.weight": (3072, 256), "ref_time_in.out_layer.weight": (3072, 3072)})
    for prefix in block_prefixes():
        for name, shape in (
            ("qkv.weight", (9216, 3072)),
            ("attn_out.weight", (3072, 3072)),
            ("mlp_in.weight", (18432, 3072)),
            ("mlp_out.weight", (3072, 9216)),
            ("norm.key_norm.scale", (128,)),
            ("norm.query_norm.scale", (128,)),
        ):
            shapes[f"{prefix}.{name}"] = shape
    return shapes


def block_prefixes():
    return [
        *(f"content_mode_blocks.{stream}.{i}" for stream in ("image", "image_cond") for i in range(5)),
        *(f"txt_mode_blocks.{i}" for i in range(5)),
        *(f"single_blocks.{i}" for i in range(28)),
    ]


def quantized_keys():
    return {
        f"{prefix}.{layer}.weight" for prefix in block_prefixes() for layer in ("qkv", "attn_out", "mlp_in", "mlp_out")
    }


def layout_quantized_keys(fused):
    keys = set(quantized_keys())
    if not fused:
        for key in list(keys):
            if key.endswith(".qkv.weight"):
                keys.remove(key)
                prefix = key.removesuffix("qkv.weight")
                keys.update(prefix + part for part in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
    return keys


def _descriptor(stream, data_start, value):
    offsets = value.get("data_offsets")
    shape = value.get("shape")
    if value.get("dtype") != "U8" or not isinstance(shape, list) or len(shape) != 1 or not 0 < shape[0] <= 4096:
        raise Flux3Error("本体の量子化設定が不正です。")
    if offsets[1] - offsets[0] != shape[0]:
        raise Flux3Error("本体の量子化設定のサイズが不正です。")
    stream.seek(data_start + offsets[0])
    try:
        result = json.loads(stream.read(shape[0]))
    except (ValueError, UnicodeError) as error:
        raise Flux3Error("本体の量子化設定を読み取れません。") from error
    return result


def _validate_w4_values(stream, data_start, key, value):
    start, end = value["data_offsets"]
    stream.seek(data_start + start)
    remaining = end - start
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise Flux3Error("W4A8本体のスケールが不完全です。")
        remaining -= len(chunk)
        if key.endswith(".weight_s_rel"):
            # E4M3FN's only non-finite encodings are 0x7f and 0xff.
            valid = b"\x7f" not in chunk and b"\xff" not in chunk
        else:
            values = (item[0] for item in struct.iter_unpack("<f", chunk))
            valid = all(math.isfinite(item) for item in values)
        if not valid:
            raise Flux3Error(f"W4A8本体のスケールまたはコードブックが不正です: {key}")


def _validate_w4_layout(path, header, data_start):
    intervals = sorted(value["data_offsets"] for key, value in header.items() if key != "__metadata__")
    position = 0
    for start, end in intervals:
        if start != position or end <= start:
            raise Flux3Error("W4A8本体のテンソル範囲が不正です。")
        position = end
    if data_start + position != Path(path).stat().st_size:
        raise Flux3Error("W4A8本体のデータ範囲が不正です。")


def header_info(path):
    header = local_assets.read_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise Flux3Error("本体のメタデータが不正です。")
    if "_quantization_metadata" in metadata:
        raise Flux3Error("旧形式の量子化メタデータを持つ本体は受け付けません。")
    model = "distilled" if "ref_time_in.in_layer.weight" in header else "base"
    shapes = expected_shapes(model)
    fused = "single_blocks.0.qkv.weight" in header
    quantized = layout_quantized_keys(fused)
    probe = "single_blocks.0.qkv.weight" if fused else "single_blocks.0.q_proj.weight"
    precision = "int8" if header.get(probe, {}).get("dtype") == "I8" else "bf16"
    if precision == "int8":
        config = header.get(probe.removesuffix(".weight") + ".comfy_quant")
        if config is not None:
            with Path(path).open("rb") as stream:
                data_start = 8 + struct.unpack("<Q", stream.read(8))[0]
                settings = _descriptor(stream, data_start, config)
                if isinstance(settings, dict) and settings.get("format") == "asym_w4a8_int8":
                    if (
                        settings != W4_CONFIG
                        or type(settings.get("group_size")) is not int
                        or type(settings.get("convrot_groupsize")) is not int
                    ):
                        raise Flux3Error("未対応のW4A8量子化設定です。")
                    precision = "w4a8"
                    _validate_w4_layout(path, header, data_start)
    if not fused:
        if precision not in {"bf16", "w4a8"}:
            raise Flux3Error("分離Q・K・VはBF16またはW4A8本体のみ対応します。")
        for key in [name for name in shapes if name.endswith(".qkv.weight")]:
            rows, columns = shapes.pop(key)
            if rows % 3:
                raise Flux3Error("本体のQ・K・V形状が不正です。")
            prefix = key.removesuffix("qkv.weight")
            for part in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                shapes[prefix + part] = (rows // 3, columns)
    allowed_attention = {
        f"{prefix}.comfy_attention.config"
        for prefix in block_prefixes()
        if prefix.startswith(("content_mode_blocks.image.", "single_blocks."))
    }
    attention = set(header) & allowed_attention
    expected = dict(shapes)
    if precision in {"int8", "w4a8"}:
        for key in quantized:
            prefix = key.removesuffix(".weight")
            rows, columns = shapes[key]
            if precision == "w4a8":
                if columns % 256:
                    raise Flux3Error("W4A8本体の入力次元が不正です。")
                expected[key] = (rows, columns // 2)
                expected[prefix + ".weight_s_rel"] = (rows, columns // 16)
                expected[prefix + ".weight_s_channel"] = (rows,)
                if prefix + ".weight_codebook" in header:
                    expected[prefix + ".weight_codebook"] = (16,)
            else:
                expected[prefix + ".weight_scale"] = (rows, 1)
            expected[prefix + ".comfy_quant"] = None
    if precision != "int8" and attention:
        raise Flux3Error("本体に未対応のattention量子化設定があります。")
    expected.update({key: None for key in attention})
    if set(header) - {"__metadata__"} != set(expected):
        raise Flux3Error("FLUX 3の全レイヤーが揃った互換本体を指定してください。")
    with Path(path).open("rb") as stream:
        data_start = 8 + struct.unpack("<Q", stream.read(8))[0]
        for key, shape in expected.items():
            value = header[key]
            dtype = "I8" if precision in {"int8", "w4a8"} and key in quantized else "BF16"
            if key.endswith((".weight_scale", ".weight_s_channel", ".weight_codebook")):
                dtype = "F32"
            elif key.endswith(".weight_s_rel"):
                dtype = "F8_E4M3"
            if shape is None:
                config = _descriptor(stream, data_start, value)
                if key.endswith(".comfy_quant"):
                    if precision == "w4a8":
                        if (
                            config != W4_CONFIG
                            or type(config.get("group_size")) is not int
                            or type(config.get("convrot_groupsize")) is not int
                        ):
                            raise Flux3Error("未対応のW4A8量子化設定です。")
                    elif (
                        not isinstance(config, dict)
                        or set(config) != {"format", "convrot", "convrot_groupsize"}
                        or config["format"] != "int8_tensorwise"
                        or config["convrot"] is not True
                        or type(config["convrot_groupsize"]) is not int
                        or config["convrot_groupsize"] != 256
                    ):
                        raise Flux3Error("未対応の本体量子化設定です。")
                elif config != {"attention": "comfy_kitchen_int8"}:
                    raise Flux3Error("未対応のattention設定です。")
            elif (
                value.get("shape") != list(shape)
                or value.get("dtype") != dtype
                or (
                    value["data_offsets"][1] - value["data_offsets"][0]
                    != math.prod(shape) * {"I8": 1, "BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
                )
            ):
                raise Flux3Error(f"FLUX 3本体の形状または保存形式が一致しません: {key}")
            elif precision == "w4a8" and key.endswith((".weight_s_rel", ".weight_s_channel", ".weight_codebook")):
                _validate_w4_values(stream, data_start, key, value)
    return {"model": model, "precision": precision}


def validate_name(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 512
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(not part or part.startswith(".") for part in name.split("/"))
        or PurePosixPath(name).suffix.lower() != ".safetensors"
    ):
        raise Flux3Error("本体は専用checkpointsフォルダー内のsafetensors名で指定してください。")
    return name


def resolve(root, name):
    validate_name(name)
    try:
        directory = canonical_path(Path(root) / "checkpoints")
        path = canonical_path(directory / name)
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("missing")
        return path
    except (OSError, ValueError) as error:
        raise Flux3Error("本体が見つからないか管理フォルダー外です。一覧を更新してください。") from error


def inventory(root):
    try:
        directory = canonical_path(Path(root) / "checkpoints")
        result = []
        for path in directory.rglob("*.safetensors"):
            name = path.relative_to(directory).as_posix()
            try:
                resolve(root, name)
            except ValueError:
                continue
            result.append(name)
        return sorted(result, key=str.casefold)
    except (OSError, ValueError):
        return []


def inspect(root, name):
    path = resolve(root, name)
    before = local_assets.file_identity(path)
    info = header_info(path)
    if before != local_assets.file_identity(path):
        raise Flux3Error("確認中に本体が変更されました。再選択してください。")
    return {"name": name, "sha256": before["sha256"], "size": before["size"], **info}


def bind(root, request):
    name = request.get("checkpoint")
    if not name:
        return None
    current = inspect(root, name)
    if current["model"] != request.get("model") or current["precision"] != request.get("precision"):
        raise Flux3Error("本体と選択したモデル・保存形式が一致しません。")
    previous = request.get("checkpoint_identity")
    if previous is not None and previous != current:
        raise Flux3Error("選択した本体が変更されました。選び直してください。")
    request["checkpoint_identity"] = current
    return current


def record(request):
    if not request.get("checkpoint"):
        return {"source": "official", "revision": MODEL_REVISION}
    identity = request.get("checkpoint_identity")
    if not isinstance(identity, dict) or identity.get("name") != request["checkpoint"]:
        raise Flux3Error("本体の保存条件がありません。")
    from .core import model_specs

    spec = model_specs(request).get("dit")
    result = {"name": identity["name"], "sha256": identity["sha256"], "size": identity["size"]}
    if spec and identity["size"] == spec["size"] and identity["sha256"] == spec["sha256"]:
        return {"source": "official", "revision": MODEL_REVISION, **result}
    return {"source": "custom", **result}
