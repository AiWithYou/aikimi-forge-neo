"""Map the pinned ComfyUI adapter onto native Diffusers without requantizing INT8."""

from __future__ import annotations

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


class OutpaintLinear(nn.Module):
    def __init__(self, base, down, up):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("lora_down", down.to(dtype=torch.bfloat16).contiguous())
        self.register_buffer("lora_up", up.to(dtype=torch.bfloat16).contiguous())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden_states):
        output = self.base(hidden_states)
        # The pinned adapter has rank=alpha=32, i.e. scale=1. Keep quantized
        # base weights intact, including their offload scales and disk cache.
        delta = F.linear(F.linear(hidden_states.to(self.lora_down.dtype), self.lora_down), self.lora_up)
        return output + delta.to(output.dtype)


def map_adapter(state: dict) -> dict:
    """Comfy gate_up is [gate, up]; split only B's output rows, sharing A."""
    expected = {
        f"diffusion_model.transformer_blocks.{block}.{target}.lora_{side}.weight"
        for block in range(32)
        for target in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out")
        for side in ("A", "B")
    }
    if set(state) != expected:
        raise ValueError("Outpaint LoRAの384テンソルが固定配布版と一致しません。")
    result = {}
    for key in sorted(expected):
        if not key.endswith(".lora_A.weight"):
            continue
        name = key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
        down, up = state[key], state[key.replace(".lora_A.", ".lora_B.")]
        inputs = 12288 if name.endswith("img_mlp.out") else 4096
        outputs = 24576 if name.endswith("img_mlp.gate_up") else 4096
        if down.shape != (32, inputs) or up.shape != (outputs, 32):
            raise ValueError(f"Outpaint LoRAの形状が一致しません: {name}")
        if not torch.isfinite(down).all() or not torch.isfinite(up).all():
            raise ValueError("Outpaint LoRAに非有限の値が含まれています。")
        if name.endswith("img_mlp.gate_up"):
            gate, projection = up.chunk(2, dim=0)
            result[name.replace("gate_up", "gate_layer")] = (down, gate)
            result[name.replace("gate_up", "proj")] = (down, projection)
        else:
            result[name] = (down, up)
    return result


def load_adapter(transformer, path: str) -> dict:
    mapped = map_adapter(load_file(path, device="cpu"))
    replacements = []
    # Validate every target before replacing any module. Never silently skip.
    for name, (down, up) in mapped.items():
        base = transformer.get_submodule(name)
        if isinstance(base, OutpaintLinear) or (base.out_features, base.in_features) != (up.shape[0], down.shape[1]):
            raise ValueError(f"Outpaint LoRAの対象層が一致しません: {name}")
        parent, attribute = name.rsplit(".", 1)
        replacements.append((transformer.get_submodule(parent), attribute, OutpaintLinear(base, down, up)))
    for parent, attribute, replacement in replacements:
        setattr(parent, attribute, replacement)
    return {"source_tensors": 384, "linear_layers": len(replacements), "rank": 32, "alpha": 32, "scale": 1.0}
