"""Packed ConvRot W4 weights and dynamic A8 CUDA computation."""

from __future__ import annotations

import importlib
import importlib.metadata
from functools import cache

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

FORMAT = "iris-convrot-w4a8-v1"
KITCHEN_VERSION = "0.2.33"
GROUP_SIZE = 16
ROTATION_SIZE = 256


def storage_shape(out_features, in_features):
    return ((out_features + 15) // 16 * 16, (in_features + ROTATION_SIZE - 1) // ROTATION_SIZE * ROTATION_SIZE)


@cache
def dependency():
    import comfy_kitchen as ck

    if importlib.metadata.version("comfy-kitchen") != KITCHEN_VERSION:
        raise RuntimeError(f"W4A8にはcomfy-kitchen=={KITCHEN_VERSION}が必要です。実行環境を準備してください。")
    return ck


def reject_bf16_fallback(*args, **kwargs):
    raise RuntimeError("このGPU・入力寸法ではW4A8のCUDA kernelを使用できません。モデルを選び直してください。")


@cache
def require_native_cuda():
    ck = dependency()
    if not ck.list_backends()["cuda"]["available"]:
        raise RuntimeError("W4A8のCUDA kernelがありません。実行環境を準備してください。")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("W4A8にはAmpere以降のNVIDIA GPUが必要です。")
    # This process is Iris's dedicated worker. CK's CUDA implementation has an
    # internal BF16 fallback even when backend dispatch itself selects CUDA.
    backend = importlib.import_module("comfy_kitchen.backends.cuda")
    backend.eager_w4a8_int8_linear = reject_bf16_fallback


class W4A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, *, codebook=True, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        out_features, in_features = storage_shape(out_features, in_features)
        self.register_buffer("qweight", torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device))
        self.register_buffer(
            "scale", torch.empty(out_features, in_features // GROUP_SIZE, dtype=torch.float8_e4m3fn, device=device)
        )
        self.register_buffer("s_channel", torch.empty(out_features, dtype=torch.float32, device=device))
        self.register_buffer("codebook", torch.empty(16, device=device) if codebook else None)
        self.bias = nn.Parameter(torch.empty(self.out_features, device=device), requires_grad=False) if bias else None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, *, device="cuda"):
        ck = dependency()
        device = torch.device(device)
        n, k = storage_shape(linear.out_features, linear.in_features)
        weight = linear.weight.detach().to(device=device, dtype=torch.bfloat16)
        if not torch.isfinite(weight).all():
            raise ValueError("W4A8変換には有限値の重みが必要です。")
        if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
            raise RuntimeError("W4A8にはAmpere以降のNVIDIA GPUが必要です。")
        if k != linear.in_features or n != linear.out_features:
            weight = F.pad(weight, (0, k - linear.in_features, 0, n - linear.out_features))
        with ck.use_backend("cuda" if device.type == "cuda" else "eager"):
            packed, scale, channel, correction, codebook = ck.quantize_w4a8_int8_weight(
                weight,
                group_size=GROUP_SIZE,
                convrot_groupsize=ROTATION_SIZE,
                symmetric=True,
                codebook=True,
                scale_dtype=torch.float8_e4m3fn,
            )
        if correction is not None:
            raise ValueError("対称W4A8の変換に非対称補正が含まれています。")
        result = cls(linear.in_features, linear.out_features, linear.bias is not None, codebook=codebook is not None)
        result.qweight, result.scale, result.s_channel = packed.cpu(), scale.cpu(), channel.cpu()
        result.codebook = codebook.cpu() if codebook is not None else None
        if linear.bias is not None:
            result.bias = nn.Parameter(linear.bias.detach().cpu().clone(), requires_grad=False)
        return result.eval()

    def forward(self, inputs):
        ck = dependency()
        dtype = (
            torch.get_autocast_dtype(inputs.device.type)
            if torch.is_autocast_enabled(inputs.device.type)
            else inputs.dtype
        )
        n, k = storage_shape(self.out_features, self.in_features)
        values = inputs.to(dtype)
        if k != self.in_features:
            values = F.pad(values, (0, k - self.in_features))
        bias = self.bias.to(dtype) if self.bias is not None else None
        if bias is not None and n != self.out_features:
            bias = F.pad(bias, (0, n - self.out_features))
        with ck.use_backend("cuda" if inputs.is_cuda else "eager"):
            output = ck.w4a8_int8_linear(
                values,
                self.qweight,
                self.scale,
                self.s_channel,
                codebook=self.codebook,
                bias=bias,
                group_size=GROUP_SIZE,
                convrot_groupsize=ROTATION_SIZE,
                out_dtype=dtype,
            )
        return output[..., : self.out_features]


def eligible(name, module):
    return (
        isinstance(module, nn.Linear)
        and module.in_features >= ROTATION_SIZE
        and module.out_features >= 64
        and "modulation_cores." not in name
    )


@torch.no_grad()
def convert_linears(model, *, device="cuda"):
    count = 0
    targets = [name for name, module in model.named_modules() if eligible(name, module)]
    for name in targets:
        module = model.get_submodule(name)
        model.set_submodule(name, W4A8Linear.from_linear(module, device=device))
        count += 1
    if not count:
        raise ValueError("W4A8へ変換できるLinearがありません。")
    return {"format": FORMAT, "linear_count": count, "bytes": sum(t.nbytes for t in model.state_dict().values())}


def load_w4a8(model, path):
    with safe_open(path, framework="pt", device="cpu") as source:
        if (source.metadata() or {}).get("format") != FORMAT:
            raise ValueError("W4A8 checkpoint形式が一致しません。")
    state = load_file(path, device="cpu")
    expected = {name for name, module in model.named_modules() if eligible(name, module)}
    actual = {key.removesuffix(".qweight") for key in state if key.endswith(".qweight")}
    if not expected or actual != expected:
        raise ValueError("W4A8の対象Linearがモデルに一致しません。")
    for name in expected:
        original = model.get_submodule(name)
        weight, scale, channel = (state.get(name + suffix) for suffix in (".qweight", ".scale", ".s_channel"))
        codebook = state.get(name + ".codebook")
        n, k = original.out_features, original.in_features
        padded_n, padded_k = storage_shape(n, k)
        if (
            weight is None
            or weight.dtype != torch.int8
            or tuple(weight.shape) != (padded_n, padded_k // 2)
            or scale is None
            or scale.dtype != torch.float8_e4m3fn
            or tuple(scale.shape) != (padded_n, padded_k // GROUP_SIZE)
            or channel is None
            or channel.dtype != torch.float32
            or tuple(channel.shape) != (padded_n,)
            or not torch.isfinite(scale.float()).all()
            or not torch.isfinite(channel).all()
            or (
                codebook is not None
                and (
                    codebook.dtype != torch.float32
                    or tuple(codebook.shape) != (16,)
                    or not torch.isfinite(codebook).all()
                )
            )
        ):
            raise ValueError(f"W4A8重み・scaleがモデルに一致しません: {name}")
        model.set_submodule(
            name, W4A8Linear(k, n, original.bias is not None, codebook=codebook is not None, device="meta")
        )
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)
