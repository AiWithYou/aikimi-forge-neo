"""NegPiP付き条件を、エンコーダーを遅延ロードして再利用する。"""

import hashlib
import importlib
import json
from pathlib import Path

import folder_paths
import nodes

NODE_NAME = "AikimiH3NegPiPCachedCLIP"
NEGPIP_NODE = "ApplyMiniMaxH3NegPiP"


def negpip_module():
    if NEGPIP_NODE not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("NegPiP拡張が未ロードです。選択設定で再起動してください。")
    return importlib.import_module(nodes.NODE_CLASS_MAPPINGS[NEGPIP_NODE].__module__)


def cache_identity(cfg):
    from minimaxh3_clipcache.encoder_abi import get_encoder_abi_id

    abi, available = get_encoder_abi_id()
    if not available:
        raise RuntimeError("エンコーダーの実装識別子を取得できません。NegPiPキャッシュを停止します。")
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(negpip_module().__file__)):
        digest.update(path.read_bytes())
    digest.update(json.dumps(cfg, sort_keys=True, allow_nan=False).encode("utf-8"))
    return f"{abi}:negpip:{digest.hexdigest()}"


def create_proxy(clip_name, cache_mode, cfg):
    import comfy.model_management
    from minimaxh3_clipcache.loader import build_clip_loader_fn, resolve_clip_stat
    from minimaxh3_clipcache.proxy import CachedClipProxy

    if cache_mode not in {"auto", "refresh"}:
        raise ValueError("NegPiPキャッシュのモードが不正です。")
    module = negpip_module()
    identity = cache_identity(cfg)
    size, mtime, ctime = resolve_clip_stat(clip_name)
    loader = build_clip_loader_fn(clip_name)

    def load_patched():
        return module.patch_clip(loader())

    return CachedClipProxy(
        load_patched, clip_name, size, mtime,
        cache_dir=Path(__file__).parent / "cache",
        force_refresh=cache_mode == "refresh",
        unload_fn=comfy.model_management.unload_model_and_clones,
        encoder_abi_id=identity, clip_ctime_ns=ctime,
    )


class NegPiPCachedCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        # 既存NegPiPと同じ設定仕様を使い、CLIP入力だけを遅延ロードへ置き換える。
        spec = nodes.NODE_CLASS_MAPPINGS[NEGPIP_NODE].INPUT_TYPES()
        required = {**spec["required"], **spec["optional"]}
        required.pop("clip")
        required["clip_name"] = (folder_paths.get_filename_list("text_encoders"),)
        required["cache_mode"] = (["auto", "refresh"],)
        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "model/conditioning/minimax/cached"

    @classmethod
    def IS_CHANGED(cls, clip_name, cache_mode, **cfg):
        from minimaxh3_clipcache.loader import resolve_clip_stat

        if cache_mode == "refresh":
            return float("nan")
        cfg.pop("model", None)
        return (resolve_clip_stat(clip_name), cache_identity(cfg))

    def apply(self, model, clip_name, cache_mode, **cfg):
        proxy = create_proxy(clip_name, cache_mode, cfg)
        # HITでも生成モデル側のAttention処理は毎回設定する。
        return negpip_module().patch_model(model, {**cfg, "enabled": True}), proxy


NODE_CLASS_MAPPINGS = {NODE_NAME: NegPiPCachedCLIP}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "MiniMax H3 NegPiP + CLIP Cache"}
