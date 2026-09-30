"""Resolve Ming's selected files independently of the optional managed denoiser."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from modules_forge import local_assets

NODE_NAME = "Aikimi-Ming-Local"
NODE_SOURCE = local_assets.ROOT / "extensions-builtin/ming-image-studio/comfyui_nodes" / NODE_NAME


def helper_record() -> dict:
    helper = Path(local_assets.__file__).resolve()
    return {"path": str(helper), "sha256": hashlib.sha256(helper.read_bytes()).hexdigest()}


def source_ready(runtime: Path) -> bool:
    destination = runtime / "custom_nodes" / NODE_NAME
    files = list(NODE_SOURCE.glob("*.py"))
    record = destination / "asset-identity.json"
    try:
        helper_matches = json.loads(record.read_text(encoding="utf-8")) == helper_record()
    except (OSError, ValueError):
        helper_matches = False
    return (
        helper_matches
        and bool(files)
        and all(
            (destination / path.name).is_file() and (destination / path.name).read_bytes() == path.read_bytes()
            for path in files
        )
    )


def sync_nodes(runtime: Path) -> bool:
    destination = runtime / "custom_nodes" / NODE_NAME
    changed = False
    for source in NODE_SOURCE.glob("*.py"):
        target = destination / source.name
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            changed = True
    record = destination / "asset-identity.json"
    text = json.dumps(helper_record(), ensure_ascii=False)
    if not record.is_file() or record.read_text(encoding="utf-8") != text:
        destination.mkdir(parents=True, exist_ok=True)
        record.write_text(text, encoding="utf-8")
        changed = True
    return changed


def validate_denoiser(path: Path) -> None:
    header = local_assets.read_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("モデルのメタデータが不正です。")
    config = json.loads(metadata.get("config", "{}"))
    if not isinstance(config, dict) or not isinstance(config.get("transformer", {}), dict):
        raise ValueError("モデルの構成情報が不正です。")
    family = config.get("transformer", {}).get("image_model")
    keys = {k.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.") for k in header}
    if family not in {None, "ming_image"} or not any(k.startswith("layers.0.") for k in keys):
        raise ValueError("Ming Image用のComfyUI形式の本体を指定してください。")
    if family is None and ("cap_pad_token" in keys or any(k.startswith("dec_net.") for k in keys)):
        raise ValueError("Ming Imageと判定できません。Z-Image／Pixelモデルはここでは使用できません。")


def selected_assets(request, root: Path, *, verify_hash: bool = True) -> dict:
    from modules_forge import ming_image_studio as studio

    records = {Path(item["path"]).name: item for item in studio.manifest()["models"]}
    result = {}
    for kind, specified, default, folder in (
        ("model", request.model_path, studio.diffusion_filename(request.precision), "diffusion_models"),
        ("text_encoder", request.text_encoder_path, studio.ENCODER, "text_encoders"),
        ("vae", request.vae_path, studio.VAE, "vae"),
    ):
        path = local_assets.local_path(specified or str(root / "models" / folder / default), suffixes=(".safetensors",))
        local_assets.read_header(path)
        if kind == "model":
            validate_denoiser(path)
        if not specified:
            if kind == "model" and request.precision == "w4a8":
                if not studio.w4a8_receipt(root, verify_hash=verify_hash):
                    raise ValueError("標準W4A8の本体が未導入または破損しています。")
            else:
                entry = records[default]
                if path.stat().st_size != entry["size"]:
                    raise ValueError(f"標準の共通部品が不足または破損しています: {default}")
                if verify_hash and local_assets.file_identity(path)["sha256"] != entry["sha256"]:
                    raise ValueError(f"標準ファイルの検証に失敗しました: {default}")
        result[kind] = local_assets.file_identity(path) if verify_hash else {"path": str(path)}
    result["loras"] = []
    seen = set()
    for item in request.loras:
        if item["strength"] == 0:
            continue
        path = local_assets.local_path(item["name"], suffixes=(".safetensors",))
        if path in seen:
            raise ValueError("同じ実ファイルのLoRAを二重に指定できません。")
        seen.add(path)
        local_assets.read_header(path)
        record = local_assets.file_identity(path) if verify_hash else {"path": str(path)}
        result["loras"].append({**record, "strength": item["strength"]})
    return result


def apply_graph(graph: dict, request, assets: dict) -> dict:
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    graph["model"] = node("AikimiMingModel", path=assets["model"]["path"], sha256=assets["model"]["sha256"])
    graph["clip"] = node(
        "AikimiMingTextEncoder", path=assets["text_encoder"]["path"], sha256=assets["text_encoder"]["sha256"]
    )
    graph["vae"] = node("AikimiMingVAE", path=assets["vae"]["path"], sha256=assets["vae"]["sha256"])
    model = ["model", 0]
    for index, item in enumerate(assets["loras"]):
        name = f"local_lora_{index}"
        graph[name] = node(
            "AikimiMingLoRA", model=model, path=item["path"], sha256=item["sha256"], strength=item["strength"]
        )
        model = [name, 0]
    graph["sampling"]["inputs"]["model"] = model
    return graph
