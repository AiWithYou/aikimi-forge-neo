"""Run the Qwen Image 2.1 Fun Union branch with the pinned Diffusers worker.

Kijai's checkpoint uses ComfyUI INT8 ConvRot tensors. The 16 control blocks
share the base transformer's attention layout; hooks inject their residuals
after base blocks 0, 2, ... 30 without changing the installed Diffusers code.
The control stream and 129-channel condition follow VideoX-Fun's
qwenimage21_transformer2d_control and pipeline_qwenimage21_control modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


class ConvRotLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int, bias: bool = False):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device="meta"))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device="meta"))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device="meta"))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from comfy_kitchen import int8_linear

        return int8_linear(
            x, self.weight, self.weight_scale, self.bias,
            out_dtype=x.dtype, convrot=True, convrot_groupsize=256,
        )


class FusedControlMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate_up = ConvRotLinear(6 * dim, dim)
        self.out = ConvRotLinear(dim, 3 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * up)


class FunUnion(nn.Module):
    def __init__(self):
        super().__init__()
        dim, heads, head_dim = 4096, 32, 128
        with torch.device("meta"):
            self.control_img_in = nn.Linear(129, dim)
            self.control_blocks = nn.ModuleList(
                QwenImage21TransformerBlock(dim, heads, head_dim) for _ in range(16)
            )
        for index, block in enumerate(self.control_blocks):
            block.attn.to_q = ConvRotLinear(dim, dim)
            block.attn.to_k = ConvRotLinear(dim, dim)
            block.attn.to_v = ConvRotLinear(dim, dim)
            block.attn.to_out[0] = ConvRotLinear(dim, dim)
            block.img_mlp = FusedControlMLP(dim)
            block.after_proj = ConvRotLinear(dim, dim, bias=True)
            if index == 0:
                block.before_proj = ConvRotLinear(dim, dim, bias=True)
        self.context: torch.Tensor | None = None
        self.strength = 1.0
        self._hints: list[torch.Tensor] = []
        self._handles = []

    @classmethod
    def from_checkpoint(cls, path: Path) -> FunUnion:
        from .fun_controlnet import inspect_checkpoint

        inspect_checkpoint(path)
        state = load_file(str(path), device="cpu")
        for key in [key for key in state if key.endswith(".comfy_quant")]:
            settings = json.loads(state.pop(key).numpy().tobytes())
            if settings != {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}:
                raise ValueError(f"未対応の量子化設定: {key}")
        model = cls()
        model.load_state_dict(state, strict=True, assign=True)
        if any(t.is_meta for t in model.state_dict().values()):
            raise ValueError("ControlNetの重みが不足しています。")
        model.eval()
        return model

    def attach(self, transformer: nn.Module) -> None:
        transformer.fun_union = self

        def before_first(_block, args, kwargs):
            if self.context is None:
                self._hints = []
                return
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is None or self.context.shape[0] != x.shape[0] or self.context.shape[1] > x.shape[1]:
                raise RuntimeError("ControlNetの画像トークン数がベースモデルと一致しません。")
            condition = self.context.to(device=x.device, dtype=x.dtype)
            joint = torch.zeros_like(x)
            joint[:, -condition.shape[1]:] = self.control_img_in(condition)
            block_kwargs = {key: value for key, value in kwargs.items() if key != "hidden_states"}
            self._hints = []
            for index, block in enumerate(self.control_blocks):
                if index == 0:
                    joint = block.before_proj(joint) + x
                joint = block(hidden_states=joint, **block_kwargs)
                self._hints.append(block.after_proj(joint))

        self._handles.append(transformer.transformer_blocks[0].register_forward_pre_hook(before_first, with_kwargs=True))
        for index in range(16):
            def add_hint(_block, _args, output, hint_index=index):
                return output + self._hints[hint_index] * self.strength if self._hints else output

            self._handles.append(transformer.transformer_blocks[index * 2].register_forward_hook(add_hint))

    def set_control(
        self, pipe, image, strength: float, generator: torch.Generator,
        *, inpaint_image=None, mask_image=None,
    ) -> None:
        """Pack control, keep-mask, and masked source in VideoX-Fun's 129-channel order."""
        if (inpaint_image is None) != (mask_image is None):
            raise ValueError("Inpaintingには編集元とマスクを一緒に指定してください。")
        if inpaint_image is not None and (
            inpaint_image.size != image.size or mask_image.size != image.size
        ):
            raise ValueError("編集元・マスク・制御画像のサイズが一致しません。")
        processed = pipe.image_processor.preprocess(image, height=image.height, width=image.width)
        if processed.shape[1] != 3:
            raise ValueError("制御画像はRGB画像が必要です。")
        processed = torch.cat([processed, torch.ones_like(processed[:, :1])], dim=1)
        processed = processed.unsqueeze(2).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            if inpaint_image is not None:
                mask_array = np.asarray(mask_image.convert("L"), dtype=np.uint8).copy()
                keep = torch.from_numpy((mask_array < 128).astype(np.float32))
                keep = keep.unsqueeze(0).unsqueeze(0).to(
                    device=pipe._execution_device, dtype=pipe.vae.dtype
                )
                source = pipe.image_processor.preprocess(
                    inpaint_image, height=image.height, width=image.width
                )
                if source.shape[1] != 3:
                    raise ValueError("編集元はRGB画像が必要です。")
                source = source.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
                source = source * keep
                source = torch.cat([source, torch.ones_like(source[:, :1])], dim=1)
                source_latents = pipe._encode_vae_image(source.unsqueeze(2), generator)
            else:
                source_latents = None
            latents = pipe._encode_vae_image(processed, generator)
        if source_latents is None:
            mask_latent = torch.zeros_like(latents[:, :1])
            source_latents = torch.zeros_like(latents)
        else:
            mask_latent = F.interpolate(
                keep, size=latents.shape[-2:], mode="nearest"
            ).unsqueeze(2)
        context = torch.cat([latents, mask_latent, source_latents], dim=1)
        self.context = pipe._pack_latents(context, 1, 129, context.shape[-2], context.shape[-1])
        self.strength = strength

    def clear_control(self) -> None:
        self.context = None
        self._hints = []
