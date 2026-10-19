"""Export fixed LightVAE components for the native ComfyUI H3 VAE loader.

Tensor formats follow ModelTC/LightVAE, revision
38b0587d47df9188e3e05973265352962b20ada4, comfyui/h3_vae.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def native_state_dict(state, component, heads=32):
    import torch

    result = {}
    for key, value in state.items():
        if component == "encoder":
            key = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", key)
            key = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", key)
            key = key.replace(".conv_shortcut.", ".nin_shortcut.")
        else:
            if ".ff.net.0.proj." in key:
                value = value.reshape(2, -1, *value.shape[1:]).flip(0).flatten(0, 1)
            for old, new in ((".attn.to_out.0.", ".attn.to_out."),
                             ("decoder.proj_in.", "decoder.x_embedder."),
                             (".ff.net.0.proj.", ".ff.w1."), (".ff.net.2.", ".ff.w2.")):
                key = key.replace(old, new)
        result[key] = value
    if component == "decoder":
        for key in tuple(result):
            if ".attn.to_q." in key:
                projections = [result.pop(key.replace(".to_q.", f".to_{axis}.")) for axis in "qkv"]
                shape = projections[0].shape
                by_head = [tensor.reshape(heads, -1, *shape[1:]) for tensor in projections]
                result[key.replace(".to_q.", ".to_qkv.")] = torch.cat(by_head, dim=1).flatten(0, 1)
        if "decoder.register_tokens" in result:
            tokens = result["decoder.register_tokens"]
            result.setdefault("decoder.mask_token", tokens.new_zeros(tokens.shape[0], 1, tokens.shape[2]))
    return result


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_native(source: Path, target: Path, component: str, source_sha256: str,
                  expected_sha256: str, expected_size: int) -> None:
    """Verify both identities and publish an export only after complete validation."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if component not in {"encoder", "decoder"}:
        raise ValueError("LightVAE component must be encoder or decoder")
    for path in (source, target):
        if path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError("Linked LightVAE files are not supported")
    if target.is_file():
        if target.stat().st_size == expected_size and sha256(target) == expected_sha256:
            return
        raise ValueError("Existing native LightVAE differs from the fixed export")
    if sha256(source) != source_sha256:
        raise ValueError("LightVAE source SHA-256 mismatch")
    with safe_open(source, framework="pt", device="cpu") as handle:
        architecture = json.loads(handle.metadata()["architecture"])
        config = architecture.get("teacher_config", architecture)
        prefixes = ("decoder.", "post_quant_conv.") if component == "decoder" else ("encoder.", "quant_conv.")
        state = {name: handle.get_tensor(name) for name in handle.keys() if name.startswith(prefixes)}
    state = native_state_dict(state, component, config.get("decoder_num_attention_heads", 32))
    state.update(latents_mean=torch.tensor(config["latents_mean"], dtype=torch.float32),
                 latents_std=torch.tensor(config["latents_std"], dtype=torch.float32))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        # One metadata key keeps the SafeTensors header deterministic.
        save_file(state, temporary, metadata={"source_sha256": source_sha256})
        if temporary.stat().st_size != expected_size or sha256(temporary) != expected_sha256:
            raise ValueError("Native LightVAE export size or SHA-256 mismatch")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
