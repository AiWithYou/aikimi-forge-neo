"""Managed single-file MOSS checkpoints with exact shapes and bound identities."""

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
    shapes = dict(expected)
    if quantized:
        for key in QUANTIZED_KEYS:
            prefix = key.removesuffix(".weight")
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
        elif key.endswith(".weight_scale"):
            allowed = {"F32"}
        if not valid or dtype not in allowed:
            raise MossImageError(f"MOSS Image本体の形状または保存形式が一致しません: {key}")
    return "int8" if quantized else "bf16"


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
    precision = header_precision(local_assets.read_header(path))
    digest = local_assets._digest(str(path), *before)
    if before != local_assets.file_version(path):
        raise MossImageError("確認中にモデルが変更されました。再実行してください。")
    return {"name": name, "precision": precision, "sha256": digest, "size": before[0], "mtime_ns": before[1]}


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
            precision = header_precision(local_assets.read_header(resolve(root, name)))
        except (OSError, ValueError, TypeError, KeyError):
            continue
        choices.append((f"{name.removesuffix('.safetensors')} · {precision.upper()}", name))
    return choices
