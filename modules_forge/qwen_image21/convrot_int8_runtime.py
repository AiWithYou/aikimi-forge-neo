"""Run prequantized base and ControlNet linears using the same ConvRot kernel."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch
from torch import nn

from . import convrot_int8

KITCHEN_VERSION = "0.2.31"


def validate_dependency() -> None:
    try:
        version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != KITCHEN_VERSION:
        raise RuntimeError(
            f"INT8 ConvRotにはcomfy-kitchen=={KITCHEN_VERSION}が必要です。Qwenの専用環境を準備してください。"
        )


class ConvRotLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int, bias: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device="meta"))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device="meta"))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device="meta"))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from comfy_kitchen import int8_linear

        return int8_linear(
            hidden_states,
            self.weight,
            self.weight_scale,
            self.bias,
            out_dtype=hidden_states.dtype,
            convrot=True,
            convrot_groupsize=256,
        )


def load_model(path: Path, model_class) -> tuple[nn.Module, dict]:
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from .local_source import TRANSFORMER_CONFIG

    validate_dependency()
    info = convrot_int8.inspect_checkpoint(path)
    state = convrot_int8.normalize_keys(load_file(str(path), device="cpu"))
    with init_empty_weights():
        model = model_class.from_config(TRANSFORMER_CONFIG)
    descriptors = [name for name in state if name.endswith(".comfy_quant")]
    if len(descriptors) != info["quantized_linears"]:
        raise ValueError("ConvRot設定の層数が読み込み時に変わりました。")
    for name in descriptors:
        stem = name.removesuffix(".comfy_quant")
        convrot_int8.validate_settings(json.loads(bytes(state.pop(name).tolist())), stem)
        try:
            base = model.get_submodule(stem)
        except AttributeError as exc:
            raise ValueError(f"ConvRotの対象層がありません: {stem}") from exc
        if not isinstance(base, nn.Linear):
            raise ValueError(f"ConvRotの対象が線形層ではありません: {stem}")
        parent, attribute = stem.rsplit(".", 1) if "." in stem else ("", stem)
        setattr(
            model.get_submodule(parent),
            attribute,
            ConvRotLinear(base.out_features, base.in_features, base.bias is not None),
        )
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[name].shape != expected[name].shape for name in state):
        raise ValueError(
            "INT8 ConvRot本体の全テンソル名・寸法がQwen Image 2.1と一致しません。部分読み込みは行いません。"
        )
    state = {
        name: value if not value.is_floating_point() or name.endswith(".weight_scale") else value.to(torch.bfloat16)
        for name, value in state.items()
    }
    # Do not cast the model dtype: that would also round its FP32 scale buffers.
    model.load_state_dict(state, strict=True, assign=True)
    if any(tensor.is_meta for tensor in model.state_dict().values()):
        raise RuntimeError("INT8 ConvRotモデルに未読み込みのテンソルがあります。")
    model.requires_grad_(False).eval()
    return model, {**info, "backend": "comfy-kitchen", "weight_dtype": "int8", "scale_dtype": "float32"}
