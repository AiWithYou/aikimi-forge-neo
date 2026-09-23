"""Shared local GGUF loading for the standard and Turbo denoisers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_unsloth_checkpoint(checkpoint: dict, **_kwargs) -> dict:
    """Undo stable-diffusion.cpp's prefix and 32 fused image SwiGLU weights."""
    prefix = "model.diffusion_model."
    result = {}
    fused = 0
    for name, weight in checkpoint.items():
        if not name.startswith(prefix):
            raise ValueError(f"Unsloth GGUFの予期しない重み名: {name}")
        name = name.removeprefix(prefix)
        if name.endswith(".img_mlp.gate_up.weight"):
            gate, projection = weight.chunk(2, dim=0)
            result[name.replace("gate_up", "gate_layer")] = gate
            result[name.replace("gate_up", "proj")] = projection
            fused += 1
        else:
            if name == "txt_in.text_norm.weight" and hasattr(weight, "quant_type"):
                # Diffusers wraps GGUF BF16 as raw uint8; RMSNorm is not a
                # GGUFLinear and otherwise multiplies by its 8192 bytes.
                import torch
                from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

                weight = dequantize_gguf_tensor(weight).to(torch.bfloat16)
            result[name] = weight
    if fused != 32 or len(result) != 297:
        raise ValueError(f"Unsloth GGUFの重み構成が一致しません: {fused} fused / {len(result)} tensors")
    return result


def load_transformer(model_path: Path, gguf_path: Path, *, unsloth: bool = False):
    """Use Diffusers' GGUF loader with the 2.1 mapping missing from our pin."""
    import torch
    from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel
    from diffusers.loaders import single_file_model

    mappings = single_file_model.SINGLE_FILE_LOADABLE_CLASSES
    previous = mappings.get("QwenImage21Transformer2DModel")
    mappings["QwenImage21Transformer2DModel"] = {
        "checkpoint_mapping_fn": map_unsloth_checkpoint if unsloth else lambda checkpoint, **_kwargs: checkpoint,
        "default_subfolder": "transformer",
    }
    try:
        return QwenImage21Transformer2DModel.from_single_file(
            str(gguf_path),
            config=str(model_path),
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    finally:
        if previous is None:
            mappings.pop("QwenImage21Transformer2DModel", None)
        else:
            mappings["QwenImage21Transformer2DModel"] = previous
