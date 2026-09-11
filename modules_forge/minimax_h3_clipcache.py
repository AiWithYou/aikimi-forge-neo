"""MiniMax H3のCLIPCached接続と固定版インストールの検証。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

CLIP_CACHE_PACK = "ComfyUI-MiniMaxH3-CLIPCached"
CLIP_CACHE_REVISION = "80ef7eb3b01565ff4190b519ce0db2f42b5e14e2"
CLIP_CACHE_FL2VA = "MiniMaxH3CLIPCachedFL2VA"
CLIP_CACHE_REF2VA = "MiniMaxH3CLIPCachedRef2VA"
CLIP_CACHE_NODES = {CLIP_CACHE_FL2VA, CLIP_CACHE_REF2VA}
MANIFEST = Path(__file__).resolve().parents[1] / "tools/minimax_h3_clipcache_manifest.json"
REFERENCE_LIMITS = {"ref_image": 9, "ref_video": 3, "ref_video_audio": 3, "ref_audio": 3}


def clipcache_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest["repository"] != "Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached"
        or manifest["revision"] != CLIP_CACHE_REVISION
    ):
        raise ValueError("CLIPCachedの固定版マニフェストが一致しません。")
    paths = set()
    for entry in manifest["files"]:
        path = PurePosixPath(entry["path"])
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != entry["path"]
            or any(character in str(path) for character in "\\:\x00")
            or entry["path"] in paths
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
            or len(entry["blob"]) != 40
            or any(character not in "0123456789abcdef" for character in entry["blob"])
        ):
            raise ValueError("CLIPCachedマニフェストに不正なファイル指定があります。")
        paths.add(entry["path"])
    if not {"__init__.py", "nodes.py", "LICENSE"} <= paths:
        raise ValueError("CLIPCachedマニフェストに必須ファイルがありません。")
    return manifest


def blob_matches(content: bytes, entry: Mapping) -> bool:
    digest = hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content, usedforsecurity=False
    ).hexdigest()
    return len(content) == entry["bytes"] and digest == entry["blob"]


def clipcache_directory(runtime_root: Path) -> Path:
    root = runtime_root.resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "models").is_dir():
        raise ValueError("CLIPCachedの導入先にはComfyUIルートを指定してください。")
    target = root / "custom_nodes" / CLIP_CACHE_PACK
    if target.resolve() != target or target.is_symlink():
        raise ValueError("CLIPCachedの導入先にリンクされたフォルダーは使えません。")
    return target


def verify_clipcache_directory(directory: Path) -> None:
    manifest = clipcache_manifest()
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if path.resolve() != path.absolute() or path.is_symlink() or not path.is_file():
            raise ValueError(f"CLIPCachedの固定版ファイルが不足、またはリンクです: {entry['path']}")
        if path.stat().st_size != entry["bytes"] or not blob_matches(path.read_bytes(), entry):
            raise ValueError(f"CLIPCachedの内容が固定版と異なります: {entry['path']}。既存ファイルには上書きしません。")
    expected = {entry["path"] for entry in manifest["files"]}
    for root, directories, files in os.walk(directory, followlinks=False):
        relative = Path(root).relative_to(directory)
        for name in list(directories):
            child = Path(root) / name
            if child.resolve() != child.absolute():
                raise ValueError("CLIPCached内にリンクされたフォルダーがあります。")
            if name == "__pycache__" or (relative == Path(".") and name == "cache"):
                directories.remove(name)
        for name in files:
            if (relative / name).as_posix() not in expected:
                raise ValueError(f"CLIPCached内に管理対象外のファイルがあります: {relative / name}")


def require_clipcache(runtime_root: Path) -> None:
    target = clipcache_directory(runtime_root)
    if not target.is_dir():
        raise ValueError(
            "CLIPCachedが未導入です。tools/install_minimax_h3_clipcache.py --runtime-root <ComfyUIフォルダー>を実行してください。"
        )
    verify_clipcache_directory(target)


def apply_clipcache(workflow: dict, cache_mode: str) -> None:
    if cache_mode == "off":
        return
    if cache_mode not in {"auto", "refresh"}:
        raise ValueError("CLIPキャッシュの選択が不正です。")
    node = workflow["5"]
    classes = {"MiniMaxH3ImageToVideo": CLIP_CACHE_FL2VA, "MiniMaxH3ReferenceToVideo": CLIP_CACHE_REF2VA}
    if node["class_type"] not in classes:
        raise ValueError("CLIPCachedに接続できないconditioningノードです。")
    inputs = dict(node["inputs"])
    if inputs.pop("clip") != ["2", 0] or workflow["2"]["class_type"] != "CLIPLoader":
        raise ValueError("CLIPCachedには標準CLIPLoaderを使用してください。")
    for key in list(inputs):
        if "." not in key:
            continue
        slot = key.split(".", 1)[1]
        kind, _, index = slot.rpartition("_")
        if kind not in REFERENCE_LIMITS or not index.isdecimal() or int(index) >= REFERENCE_LIMITS[kind]:
            raise ValueError("CLIPCachedの参照上限は画像9枚、動画3本、音声3本です。")
        inputs[slot] = inputs.pop(key)
    inputs["clip_name"] = workflow["2"]["inputs"]["clip_name"]
    inputs["cache_mode"] = cache_mode
    workflow["5"] = {"class_type": classes[node["class_type"]], "inputs": inputs}
    del workflow["2"]


def _input_kind(inputs: Mapping, name: str, key: str):
    value = inputs.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name}の{key}の仕様が未対応です。")
    return value[0]


def validate_clipcache_nodes(nodes: Mapping) -> None:
    for name in CLIP_CACHE_NODES:
        schema = nodes.get(name, {})
        if not isinstance(schema, Mapping) or not isinstance(schema.get("input"), Mapping):
            raise ValueError(f"{name}の入力仕様を取得できません。")
        specification = schema.get("input", {})
        required = specification.get("required", {})
        optional = specification.get("optional", {})
        if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
            raise ValueError(f"{name}の入力仕様が未対応です。")
        inputs = {**required, **optional}

        expected = {"vae": "VAE", "prompt": "STRING", "width": "INT", "height": "INT", "length": "INT"}
        if name == CLIP_CACHE_REF2VA:
            expected["audio_vae"] = "VAE"
            expected.update(
                {
                    f"{kind}_{index}": "AUDIO" if "audio" in kind else "IMAGE"
                    for kind, count in REFERENCE_LIMITS.items()
                    for index in range(count)
                }
            )
        else:
            expected.update(first_frame="IMAGE", last_frame="IMAGE")
        for key, kind in expected.items():
            if _input_kind(inputs, name, key) != kind:
                raise ValueError(f"{name}の{key}の仕様が未対応です。固定版CLIPCachedを導入してください。")
        if not isinstance(_input_kind(inputs, name, "clip_name"), (list, tuple)):
            raise ValueError("CLIPCachedのencoder選択の仕様が未対応です。")
        known_required = {"clip_name", "vae", "prompt", "width", "height", "length"}
        if name == CLIP_CACHE_REF2VA:
            known_required.update({"audio_vae", "ref_image_size"})
            sizes = _input_kind(inputs, name, "ref_image_size")
            if not isinstance(sizes, (list, tuple)) or not {"match", "max"} <= set(sizes):
                raise ValueError("CLIPCachedの参照サイズ指定が未対応です。")
        if set(required) - known_required:
            raise ValueError("CLIPCachedに未対応の必須入力があります。")
        cache = _input_kind(inputs, name, "cache_mode")
        if not isinstance(cache, (list, tuple)) or not {"auto", "refresh"} <= set(cache):
            raise ValueError("CLIPCachedの保存モードの仕様が未対応です。")
        outputs = schema.get("output")
        if not isinstance(outputs, (list, tuple)) or tuple(outputs) != ("CONDITIONING", "LATENT"):
            raise ValueError("CLIPCachedの出力仕様が未対応です。")
