"""Identify native Ming weights from safetensors headers."""

import json
from pathlib import Path

from modules_forge import local_assets, studio_checkpoints

from .core import MingImageError, safe_model_root


def role(path):
    header = local_assets.read_header(path)
    try:
        raw = header.get("__metadata__", {}).get("config", "{}")
        if not isinstance(raw, str):
            raise TypeError
        transformer = json.loads(raw).get("transformer", {})
        if not isinstance(transformer, dict):
            raise TypeError
    except (AttributeError, TypeError, ValueError, RecursionError) as exc:
        raise MingImageError("safetensors の config が正しい JSON オブジェクトではありません。") from exc

    def shape(name):
        return header.get(name, {}).get("shape", [])

    if transformer.get("image_model") == "ming_image" and shape("cap_embedder.0.weight") == [2560]:
        if shape("layers.0.attention.qkv.weight") in ([11520, 3840], [11520, 1920]) or shape(
            "layers.0.attention.to_q.weight"
        ) == [3840, 3840]:
            return "dit"
    if shape("thinker.norm.weight") == [2048] and shape("connector.layers.0.self_attn.q_proj.weight") in (
        [1536, 1536],
        [1536, 768],
    ):
        return "text"
    raise MingImageError("Ming Image 用のモデルを選択してください。")


def validate_name(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 512
        or "\\" in name
        or ":" in name
        or any(not part or part.startswith(".") for part in name.split("/"))
        or Path(name).suffix.lower() != ".safetensors"
    ):
        raise MingImageError("モデルは専用checkpointsフォルダー内のsafetensors名で指定してください。")
    return name


def resolve(root, name):
    validate_name(name)
    directory = safe_model_root(Path(root) / "checkpoints")
    path = safe_model_root(directory / name)
    if not path.is_relative_to(directory) or not path.is_file():
        raise MingImageError("選択したモデルが見つかりません。一覧を更新してください。")
    return path


def bind(root, request):
    for key, expected in (("checkpoint", "dit"), ("text_checkpoint", "text")):
        name = request.get(key, "official")
        if name == "official":
            continue
        path = resolve(root, name)
        if role(path) != expected:
            raise MingImageError("選択したモデルの種類が一致しません。")
        identity = studio_checkpoints.snapshot_checkpoint(str(path))["identity"]["sha256"]
        field = key + "_identity"
        if request.get(field, identity) != identity:
            raise MingImageError("チェックポイントが変更されました。")
        request[field] = identity


def inventory(root, expected):
    choices = [("標準モデル", "official")]
    directory = Path(root) / "checkpoints"
    for path in sorted(directory.rglob("*.safetensors")):
        name = path.relative_to(directory).as_posix()
        try:
            if role(resolve(root, name)) == expected:
                choices.append((name, name))
        except (OSError, ValueError, KeyError, AttributeError):
            continue
    return choices
