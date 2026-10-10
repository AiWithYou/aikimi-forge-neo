"""Apply validated low-rank residuals without mutating quantized base weights."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from .outpaint_runtime import OutpaintLinear
from .style_lora import cache_key, fingerprint, validate_installed, validate_options


class StyleLinear(OutpaintLinear):
    def __init__(self, base, down, up, scale):
        super().__init__(base, down, up)
        self.scale = scale

    def forward(self, hidden_states):
        if self.scale == 0:
            return self.base(hidden_states)
        output = self.base(hidden_states)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(hidden_states.to(self.lora_down.dtype), self.lora_down), self.lora_up
        )
        return output + (delta * self.scale).to(output.dtype)


def load_adapters(transformer, runtime: Path, request: dict) -> list[dict]:
    infos = validate_installed(runtime, request)
    replacements, applied = {}, []
    # Stage every adapter before mutating the model, including shared targets.
    for info in infos:
        state = load_file(info["path"], device="cpu")
        adapter_key = (str(Path(info["path"]).resolve()), fingerprint(Path(info["path"])))
        targets_seen = set()
        for name, group in info["groups"].items():
            down, up = state[group["down"]], state[group["up"]]
            rank = down.shape[0]
            alpha = (
                float(state[group["alpha"]].item())
                if "alpha" in group
                else float(info["metadata"].get("lora_alpha", rank))
            )
            if not math.isfinite(alpha) or not torch.isfinite(down).all() or not torch.isfinite(up).all():
                raise ValueError(f"{info['name']}: LoRAに非有限の値があります: {name}")
            scale = info["strength"] * alpha / rank
            targets = [(name, up)]
            if name.endswith(".gate_up"):
                if up.shape[0] % 2:
                    raise ValueError(f"{info['name']}: 結合MLP LoRAの形状が不正です。")
                gate, projection = up.chunk(2, dim=0)
                targets = [(name.replace("gate_up", "gate_layer"), gate), (name.replace("gate_up", "proj"), projection)]
            for target, target_up in targets:
                if target in targets_seen:
                    raise ValueError(f"{info['name']}: LoRAの対象が重複しています: {target}")
                targets_seen.add(target)
                base = replacements.get(target)
                if base is None:
                    try:
                        base = transformer.get_submodule(target)
                    except AttributeError as exc:
                        raise ValueError(f"{info['name']}: 対象層がありません: {target}") from exc
                if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
                    raise ValueError(f"{info['name']}: 未対応のモデル層です: {target}")
                # Quantized weight storage can have packed dimensions (GGUF/W4A8).
                if (base.out_features, base.in_features) != (target_up.shape[0], down.shape[1]):
                    raise ValueError(f"{info['name']}: LoRAとモデルの形状が一致しません: {target}")
                replacement = StyleLinear(base, down, target_up, scale)
                replacement.adapter_key = adapter_key
                replacement.scale_per_strength = alpha / rank
                replacements[target] = replacement
        applied.append(
            {key: value for key, value in info.items() if key not in {"groups", "metadata"}}
            | {
                "sha256": adapter_key[1],
                "applied_layers": len(targets_seen),
                "base_precision": request["precision"],
                "experimental_base_mismatch": info["base_mismatch"],
            }
        )
    for target, module in replacements.items():
        parent, attribute = target.rsplit(".", 1)
        setattr(transformer.get_submodule(parent), attribute, module)
    return applied


def update_strengths(transformer, loaded: list[dict], runtime: Path, request: dict) -> list[dict]:
    """Update resident residuals atomically; leave disabled adapters ready to reuse."""
    validate_options(request)
    keys, _ = cache_key(runtime, request)
    active = [item for item in request.get("style_loras", ()) if item["strength"] != 0]
    available = {(str(Path(info["path"]).resolve()), info["sha256"]): info for info in loaded}
    if any(key not in available for key in keys):
        raise ValueError("追加LoRAの読み込み状態が変わりました。モデルを読み込み直してください。")
    if not request.get("allow_lora_base_mismatch", False) and any(available[key]["base_mismatch"] for key in keys):
        raise ValueError("ConvRot INT8用です。「異なる量子化で試す」をONにした実験のみ可能です。")
    strengths = {key: item["strength"] for key, item in zip(keys, active, strict=True)}
    for module in transformer.modules():
        if isinstance(module, StyleLinear) and hasattr(module, "adapter_key"):
            module.scale = strengths.get(module.adapter_key, 0) * module.scale_per_strength
    return [
        {**available[key], "name": item["name"], "strength": item["strength"]}
        for key, item in zip(keys, active, strict=True)
    ]
