"""Managed single-file MOSS checkpoints with exact shapes and bound identities."""

import json
import struct
from pathlib import Path

from modules_forge import local_assets

from .core import MossImageError, safe_model_root

QUANTIZED_KEYS = frozenset(
    f"layers.{layer}.proj.{projection}.weight"
    for layer in range(42)
    for projection in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
)


def expected_shapes():
    shapes = {"rope.bands": (16,)}

    def linear(name, rows, columns, bias=True):
        shapes[name + ".weight"] = (rows, columns)
        if bias:
            shapes[name + ".bias"] = (rows,)

    def norms(prefix):
        for name, size in (("norm1", 3072), ("norm2", 3072), ("q_norm", 128), ("k_norm", 128)):
            shapes[prefix + name + ".weight"] = (size,)

    linear("video_embedder", 3072, 128)
    linear("text_embedder.proj", 3072, 4096)
    linear("time_embedding.0", 3072, 256)
    linear("time_embedding.2", 3072, 3072)
    linear("shared_block_modulation", 18432, 3072)
    linear("video_head_modulation", 6144, 3072)
    linear("final_linear_video", 128, 3072, False)
    shapes["final_norm_video.weight"] = (3072,)
    for layer in range(2):
        prefix = f"text_embedder.blocks.{layer}."
        norms(prefix)
        for name, rows, columns in (
            ("qkv", 9216, 3072),
            ("proj", 3072, 3072),
            ("ffn_gate", 8192, 3072),
            ("ffn_up", 8192, 3072),
            ("ffn_down", 3072, 8192),
        ):
            linear(prefix + name, rows, columns)
    for layer in range(42):
        prefix = f"layers.{layer}."
        norms(prefix)
        for name in ("q", "k", "v", "o"):
            linear(prefix + "proj." + name, 3072, 3072, False)
        linear(prefix + "proj.ffn_up_gate", 16384, 3072, False)
        linear(prefix + "proj.ffn_down", 3072, 8192, False)
    return shapes


def header_precision(header):
    expected = expected_shapes()
    quantized = header.get("layers.0.proj.q.weight", {}).get("dtype") == "I8"
    first_shape = header.get("layers.0.proj.q.weight", {}).get("shape")
    first_expected = expected.get("layers.0.proj.q.weight")
    w4a8 = bool(quantized and first_expected and first_shape == [first_expected[0], first_expected[1] // 2])
    shapes = dict(expected)
    if quantized:
        for key in QUANTIZED_KEYS:
            prefix = key.removesuffix(".weight")
            if w4a8:
                rows, columns = expected[key]
                shapes[key] = (rows, columns // 2)
                shapes[prefix + ".weight_s_rel"] = (rows, columns // 16)
                shapes[prefix + ".weight_s_channel"] = (rows,)
                if prefix + ".weight_codebook" in header:
                    shapes[prefix + ".weight_codebook"] = (16,)
            else:
                shapes[prefix + ".weight_scale"] = (expected[key][0], 1)
            shapes[prefix + ".comfy_quant"] = None
    if set(header) - {"__metadata__"} != set(shapes):
        raise MossImageError("MOSS Imageの全対象が揃った本体を指定してください。")
    for key, shape in shapes.items():
        value = header[key]
        actual = value.get("shape")
        if shape is None:
            valid = isinstance(actual, list) and len(actual) == 1 and 1 <= actual[0] <= 4096
        else:
            valid = actual == list(shape)
        dtype = value.get("dtype")
        allowed = {"BF16", "F16", "F32"}
        if key.endswith(".comfy_quant"):
            allowed = {"U8"}
        elif key in QUANTIZED_KEYS:
            allowed = {"I8"} if quantized else {"BF16"}
        elif key.endswith(".weight_s_rel"):
            allowed = {"F8_E4M3", "F32"}
        elif key.endswith((".weight_scale", ".weight_s_channel", ".weight_codebook")):
            allowed = {"F32"}
        if not valid or dtype not in allowed:
            raise MossImageError(f"MOSS Image本体の形状または保存形式が一致しません: {key}")
    return "w4a8" if w4a8 else "int8" if quantized else "bf16"


def precision(path):
    """Check every W4A8 descriptor as well as the architecture header."""
    header = local_assets.read_header(path)
    result = header_precision(header)
    if result != "w4a8":
        return result
    if "_quantization_metadata" in header.get("__metadata__", {}):
        raise MossImageError("旧形式とW4A8の量子化設定を同時に指定できません。")
    with Path(path).open("rb") as stream:
        data_start = 8 + struct.unpack("<Q", stream.read(8))[0]
        for key in QUANTIZED_KEYS:
            descriptor = header[key.removesuffix(".weight") + ".comfy_quant"]
            start, end = descriptor["data_offsets"]
            if end - start != descriptor["shape"][0]:
                raise MossImageError("W4A8の量子化設定のサイズが不正です。")
            stream.seek(data_start + start)
            try:
                config = json.loads(stream.read(end - start))
            except (ValueError, UnicodeError) as error:
                raise MossImageError("W4A8の量子化設定を読み取れません。") from error
            if (
                not isinstance(config, dict)
                or set(config) - {"format", "group_size", "convrot_groupsize", "convrot"}
                or config.get("format") != "asym_w4a8_int8"
                or type(config.get("group_size", 16)) is not int
                or config.get("group_size", 16) != 16
                or type(config.get("convrot_groupsize", 256)) is not int
                or config.get("convrot_groupsize", 256) != 256
                or ("convrot" in config and config["convrot"] is not True)
            ):
                raise MossImageError("MOSS Imageで未対応のW4A8量子化設定です。")
    return result


def validate_name(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 512
        or "\\" in name
        or ":" in name
        or any(not part or part.startswith(".") for part in name.split("/"))
        or name.split("/")[0].casefold() == "moss-image-sft"
        or Path(name).suffix.lower() != ".safetensors"
    ):
        raise MossImageError("モデルは専用checkpointsフォルダー内のsafetensors名で指定してください。")
    return name


def resolve(root, name):
    validate_name(name)
    directory = safe_model_root(Path(root) / "checkpoints")
    path = safe_model_root(directory / name)
    if not path.is_relative_to(directory) or not path.is_file():
        raise MossImageError("選択したモデルが見つかりません。一覧を更新してください。")
    return path


def inspect(root, name):
    path = resolve(root, name)
    before = local_assets.file_version(path)
    model_precision = precision(path)
    digest = local_assets.file_identity(path)["sha256"]
    if before != local_assets.file_version(path):
        raise MossImageError("確認中にモデルが変更されました。再実行してください。")
    return {"name": name, "precision": model_precision, "sha256": digest, "size": before[0], "mtime_ns": before[1]}


def bind(root, request):
    name = request.get("checkpoint", "official")
    if name == "official":
        return None
    identity = inspect(root, name)
    if identity["precision"] != request["precision"]:
        raise MossImageError("モデルと画像生成本体の保存形式が一致しません。モデルを選び直してください。")
    if request.get("checkpoint_identity", identity) != identity:
        raise MossImageError("選択したモデルが変更されました。選び直して再実行してください。")
    request["checkpoint_identity"] = identity
    return identity


def inventory(root):
    choices = [("MOSS-Image-SFT", "official")]
    directory = Path(root) / "checkpoints"
    for path in sorted(directory.rglob("*.safetensors"), key=lambda item: str(item).casefold()):
        name = path.relative_to(directory).as_posix()
        try:
            model_precision = precision(resolve(root, name))
        except (OSError, ValueError, TypeError, KeyError):
            continue
        choices.append((f"{name.removesuffix('.safetensors')} · {model_precision.upper()}", name))
    return choices
