"""Opt-in H3 acceleration policy; no model downloads or GPU imports.

Contracts checked 2026-09-06 against ComfyUI e308cc73 (BlockSparseAttention),
MATLOWAI's fused-turbo model card and starsFriday's Fast VAE node. These are
independent experiments, not a promise of faster or equivalent output.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from modules_forge.minimax_h3_negpip import H3NegPiP, NEGPIP_NODE, NEGPIP_PACK
from modules_forge.minimax_h3_clipcache import CLIP_CACHE_NODES, CLIP_CACHE_PACK, apply_clipcache, validate_clipcache_nodes
from modules_forge import minimax_h3_negpip_cache as negpip_cache
from modules_forge import minimax_h3_hybrid as hybrid

TURBO_MODEL = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors"
INT8_VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
FAST_VAE_NODE = "MiniMaxH3FastVAEDecode"
FAST_VAE_PACK = "ComfyUI-MiniMax-H3-MotionCache"
SPARSE_NODE = "BlockSparseAttention"
SPARSE_COMMIT = "e308cc73b466584b0c17be695e5de1a17438bb40"
SPARSE_SELECTIONS = {"sol": "Sol-Attn (adaptive tau)", "sla": "top-k (SLA)"}


@dataclass(frozen=True)
class H3Acceleration:
    model_variant: str = "base"
    video_vae: str = "fp16"
    decode_mode: str = "standard"
    tile_batch_size: int = 4
    attention: str = "dense"
    sparse_tau: float = 1.3
    sparse_keep_percent: float = 10.0
    sparse_start_percent: float = 0.2
    negpip: H3NegPiP = field(default_factory=H3NegPiP)
    clip_cache: str = "off"
    hybrid: hybrid.H3Hybrid = field(default_factory=hybrid.H3Hybrid)

    def validate(self) -> None:
        if not isinstance(self.hybrid, hybrid.H3Hybrid):
            raise ValueError("長尺設定の形式が不正です。")
        self.hybrid.validate()
        for name, allowed in (
            ("model_variant", {"base", "fused_turbo"}),
            ("video_vae", {"fp16", "int8"}),
            ("decode_mode", {"standard", "fast"}),
            ("attention", {"dense", "sol", "sla"}),
            ("clip_cache", {"off", "auto", "refresh"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"H3 高速化設定 {name} が不正です。")
        value = self.tile_batch_size
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value) or not 1 <= value <= 8:
            raise ValueError("Fast VAE tile batchは1〜8の整数で指定してください。")
        for name, low, high in (
            ("sparse_tau", 0.0, 4.0),
            ("sparse_keep_percent", 0.5, 95.0),
            ("sparse_start_percent", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name}は{low}〜{high}の有限値で指定してください。")
        if not isinstance(self.negpip, H3NegPiP):
            raise ValueError("NegPiP設定の形式が不正です。")
        self.negpip.validate()
        if self.hybrid.enabled and (self.negpip.enabled or self.clip_cache != "off" or self.attention != "dense"):
            raise ValueError("長尺生成ではNegPiPとCLIPキャッシュをオフ、AttentionをKitchen denseにしてください。")
        if self.negpip.enabled and self.attention != "dense":
            raise ValueError("NegPiPとSparse Attentionの併用は未検証です。AttentionをKitchen denseへ戻してください。")

    @classmethod
    def from_dict(cls, value: Any) -> H3Acceleration:
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("H3 高速化設定の形式が不正です。")
        data = dict(value)
        data["negpip"] = H3NegPiP.from_dict(data.get("negpip"))
        data["hybrid"] = hybrid.H3Hybrid.from_dict(data.get("hybrid"))
        result = cls(**data)
        result.validate()
        return result

    @classmethod
    def from_values(cls, values: tuple) -> H3Acceleration:
        if not values:
            return cls()
        previous_count = 8 + len(H3NegPiP().values())
        if len(values) not in (8, previous_count, previous_count + 1, previous_count + 6):
            raise ValueError("H3 追加設定の項目数が一致しません。UIを再読み込みしてください。")
        result = cls(
            *values[:8], negpip=H3NegPiP.from_values(values[8:previous_count]),
            clip_cache=values[previous_count] if len(values) > previous_count else "off",
            hybrid=hybrid.H3Hybrid(*values[previous_count + 1:]) if len(values) == previous_count + 6 else hybrid.H3Hybrid(),
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def values(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self) if f.name not in {"negpip", "clip_cache", "hybrid"}) + self.negpip.values() + (self.clip_cache,) + self.hybrid.values()


    def runtime_packs(self) -> tuple[str, ...]:
        return ((FAST_VAE_PACK,) if self.decode_mode == "fast" else ()) + ((NEGPIP_PACK,) if self.negpip.enabled else ()) + ((CLIP_CACHE_PACK,) if self.clip_cache != "off" else ()) + ((negpip_cache.PACK,) if self.negpip.enabled and self.clip_cache != "off" else ()) + ((hybrid.PACK,) if self.hybrid.enabled else ())

    def model_files(self, base: Mapping) -> dict:
        result = dict(base)
        if self.model_variant == "fused_turbo":
            result["FL2VA"] = result["Ref2VA"] = ("diffusion_models", TURBO_MODEL)
        if self.video_vae == "int8":
            result["Video VAE"] = ("vae", INT8_VIDEO_VAE)
        return result

    def extra_nodes(self) -> set[str]:
        return ({FAST_VAE_NODE} if self.decode_mode == "fast" else set()) | ({SPARSE_NODE} if self.attention != "dense" else set()) | ({NEGPIP_NODE} if self.negpip.enabled else set()) | (({negpip_cache.NODE} if self.negpip.enabled else CLIP_CACHE_NODES) if self.clip_cache != "off" else set()) | (hybrid.NODES if self.hybrid.enabled else set())

    def apply_workflow(self, workflow: dict, base_files: Mapping, mode: str) -> None:
        """Patch only selected axes; the default graph stays byte-for-byte equal."""
        self.validate()
        model_files = self.model_files(base_files)
        workflow["1"]["inputs"]["unet_name"] = model_files["Ref2VA" if mode == "references" else "FL2VA"][1]
        workflow["3"]["inputs"]["vae_name"] = model_files["Video VAE"][1]
        if self.decode_mode == "fast":
            workflow["11"] = {
                "class_type": FAST_VAE_NODE,
                "inputs": {"samples": ["10", 0], "vae": ["3", 0], "tile_batch_size": int(self.tile_batch_size)},
            }
        if self.attention != "dense":
            inputs = {
                "model": ["15", 0],
                # DynamicCombo uses flattened API keys, NOT a nested JSON object.
                "selection": SPARSE_SELECTIONS[self.attention],
                "start_percent": self.sparse_start_percent,
                "end_percent": 1.0,
                "dense_blocks": "",
                "min_tokens": 12288,
                "extra_tokens": 256,
                "sink_conditioning": "exact_kv_and_rows",
                "verbose": True,
            }
            key = "tau" if self.attention == "sol" else "keep_percent"
            inputs[f"selection.{key}"] = self.sparse_tau if self.attention == "sol" else self.sparse_keep_percent
            workflow["16"] = {"class_type": SPARSE_NODE, "inputs": inputs}
            workflow["9"]["inputs"]["model"] = ["16", 0]
        self.negpip.apply_workflow(workflow)
        if self.negpip.enabled and self.clip_cache != "off":
            negpip_cache.apply_workflow(workflow, self.clip_cache)
        else:
            apply_clipcache(workflow, self.clip_cache)

    def validate_nodes(self, nodes: Mapping) -> None:
        """Reject incompatible optional-node APIs before sending a generation."""
        if self.hybrid.enabled:
            hybrid.validate_nodes(nodes)
        for node, names in (
            (FAST_VAE_NODE, {"samples", "vae", "tile_batch_size"}),
            (SPARSE_NODE, {"model", "selection", "start_percent", "end_percent", "dense_blocks", "min_tokens", "extra_tokens", "sink_conditioning", "verbose"}),
        ):
            if node not in self.extra_nodes():
                continue
            spec = nodes.get(node, {}).get("input", {})
            inputs = {**spec.get("required", {}), **spec.get("optional", {})}
            if not names <= inputs.keys():
                raise ValueError(f"{node}の入力仕様が未対応です。ComfyUI / 対応node packを更新して再起動してください。")
            if node == SPARSE_NODE:
                selection = inputs["selection"]
                if len(selection) < 2 or selection[0] != "COMFY_DYNAMICCOMBO_V3":
                    raise ValueError("BlockSparseAttentionのDynamicCombo仕様が未対応です。")
                options = selection[1].get("options", [])
                selected = next((o for o in options if o.get("key") == SPARSE_SELECTIONS[self.attention]), None)
                subkey = "tau" if self.attention == "sol" else "keep_percent"
                if selected is None or subkey not in selected.get("inputs", {}).get("required", {}):
                    raise ValueError("選択したSparse Attention方式を接続中のComfyUIが提供していません。")
                if "exact_kv_and_rows" not in inputs["sink_conditioning"][0]:
                    raise ValueError("Sparse Attentionの音声保護設定が未対応です。")
        self.negpip.validate_nodes(nodes)
        if self.clip_cache != "off":
            if self.negpip.enabled:
                negpip_cache.validate_nodes(nodes)
            else:
                validate_clipcache_nodes(nodes)
