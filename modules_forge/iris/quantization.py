"""Per-output-channel weight-only INT8 with BF16/FP32 linear computation."""

from __future__ import annotations

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


def quantize_rows(weight):
    weight = weight.detach().float()
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("INT8変換には有限値の2次元重みが必要です。")
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (weight / scale).round().clamp(-127, 127).to(torch.int8), scale


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_linear(cls, linear):
        result = cls(linear.in_features, linear.out_features, linear.bias is not None, device=linear.weight.device)
        result.qweight, result.scale = quantize_rows(linear.weight)
        if linear.bias is not None:
            result.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
        return result

    def forward(self, inputs):
        dtype = (
            torch.get_autocast_dtype(inputs.device.type)
            if torch.is_autocast_enabled(inputs.device.type)
            else inputs.dtype
        )
        weight = (self.qweight.float() * self.scale).to(dtype)
        return F.linear(inputs.to(dtype), weight, self.bias.to(dtype) if self.bias is not None else None)


def convert_linears(model):
    count = 0
    for name, module in list(model.named_modules()):
        # Upstream blocks also hold these cores in unregistered tuples. Preserve
        # their identity and FP32 weights rather than leave stale references.
        if not isinstance(module, nn.Linear) or module.weight.numel() < 4096 or "modulation_cores." in name:
            continue
        model.set_submodule(name, Int8Linear.from_linear(module))
        count += 1
    return {
        "format": "iris-rowwise-int8-v1",
        "linear_count": count,
        "bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
    }


def load_int8(model, path):
    state = load_file(path, device="cpu")
    modules = dict(model.named_modules())
    for key, weight in state.items():
        if not key.endswith(".qweight"):
            continue
        name = key.removesuffix(".qweight")
        module = modules.get(name)
        scale = state.get(name + ".scale")
        if (
            not isinstance(module, nn.Linear)
            or tuple(weight.shape) != (module.out_features, module.in_features)
            or weight.dtype != torch.int8
            or scale is None
            or tuple(scale.shape) != (module.out_features, 1)
            or scale.dtype != torch.float32
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError(f"INT8重み・スケールがモデルに一致しません: {name}")
        model.set_submodule(
            name, Int8Linear(module.in_features, module.out_features, module.bias is not None, device="meta")
        )
    if not any(key.endswith(".qweight") for key in state):
        raise ValueError("INT8重みが含まれていません。")
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model
