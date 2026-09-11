"""NegPiPとCLIPCachedの接続、専用ノードの管理。"""

import os
import shutil
import tempfile
from pathlib import Path

PACK = "Aikimi-H3-NegPiP-Cache"
NODE = "AikimiH3NegPiPCachedCLIP"
BUNDLE = Path(__file__).resolve().parents[1] / "extensions-builtin/minimax-h3-studio/comfyui_nodes" / PACK


def verify(runtime_root):
    target = Path(runtime_root).resolve(strict=True) / "custom_nodes" / PACK
    installed = target / "__init__.py"
    if installed.resolve() != installed or not installed.is_file() or installed.read_bytes() != (BUNDLE / "__init__.py").read_bytes():
        raise ValueError("NegPiPキャッシュの導入版が同梱版と異なります。既存ファイルを確認してください。")
    if {p.name for p in target.iterdir()} - {"__init__.py", "__pycache__", "cache"}:
        raise ValueError("NegPiPキャッシュの導入先に管理対象外のファイルがあります。")
    for name in ("cache", "__pycache__"):
        child = target / name
        if child.resolve() != child or child.is_symlink():
            raise ValueError("NegPiPキャッシュの保存先にリンクは使えません。")
    return target


def install(runtime_root):
    root = Path(runtime_root).resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "models").is_dir():
        raise ValueError("NegPiPキャッシュの導入先にはComfyUIルートを指定してください。")
    target = root / "custom_nodes" / PACK
    if target.resolve() != target or target.is_symlink():
        raise ValueError("NegPiPキャッシュの導入先にリンクは使えません。")
    source = BUNDLE / "__init__.py"
    if source.is_symlink() or not source.is_file():
        raise ValueError("NegPiPキャッシュの同梱ノードがありません。")
    content = source.read_bytes()
    if target.exists():
        return verify(root)
    target.parent.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".aikimi-negpip-cache-", dir=target.parent))
    try:
        (stage / "__init__.py").write_bytes(content)
        os.rename(stage, target)
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to(target.parent.resolve()):
                raise ValueError("一時導入先がcustom_nodesの外です。")
            shutil.rmtree(stage)
    return target


def apply_workflow(graph, mode):
    node = graph["17"]
    if node["class_type"] != "ApplyMiniMaxH3NegPiP" or node["inputs"]["clip"] != ["2", 0]:
        raise ValueError("NegPiPキャッシュには標準NegPiP接続が必要です。")
    node["class_type"] = NODE
    node["inputs"].pop("clip")
    node["inputs"].update(clip_name=graph["2"]["inputs"]["clip_name"], cache_mode=mode)
    del graph["2"]


def validate_nodes(nodes):
    spec = nodes.get(NODE, {})
    required = spec.get("input", {}).get("required", {})
    source = nodes.get("ApplyMiniMaxH3NegPiP", {}).get("input", {})
    original = {**source.get("required", {}), **source.get("optional", {})}
    expected = set(original) - {"clip"} | {"clip_name", "cache_mode"}
    if set(required) != expected or list(spec.get("output", [])) != ["MODEL", "CLIP"]:
        raise ValueError("NegPiPキャッシュのノード仕様が未対応です。選択設定で再起動してください。")
    for key in set(original) - {"clip"}:
        if required[key][0] != original[key][0]:
            raise ValueError(f"NegPiPキャッシュの{key}の仕様が未対応です。")
    if not isinstance(required["clip_name"][0], list) or required["cache_mode"][0] != ["auto", "refresh"]:
        raise ValueError("NegPiPキャッシュのエンコーダー・保存モード仕様が未対応です。")
