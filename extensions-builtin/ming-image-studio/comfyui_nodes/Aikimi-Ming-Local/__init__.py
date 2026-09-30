"""Explicit local Ming files and all-or-nothing linear denoiser LoRAs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def identity_helper():
    """Use Forge's stdlib-only identity cache, pinned by the managed installer."""
    record = Path(__file__).with_name("asset-identity.json")
    if not record.is_file():
        return None
    info = json.loads(record.read_text(encoding="utf-8"))
    helper = Path(info["path"])
    if not helper.is_absolute() or hashlib.sha256(helper.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Local asset helper changed. Restart the managed Ming runtime.")
    spec = importlib.util.spec_from_file_location("_aikimi_ming_asset_identity", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_file(path, sha256):
    source = Path(path)
    if not source.is_absolute() or source.suffix.lower() != ".safetensors" or not source.is_file():
        raise ValueError("A local .safetensors file is required.")
    helper = identity_helper()
    if helper is None:
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        actual = helper.file_identity(source)["sha256"]
    if actual != sha256:
        raise ValueError("The selected file changed after submission. Select it again.")
    return str(source)


class LocalFile:
    CATEGORY = "Aikimi/Ming"
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING",), "sha256": ("STRING",)}}

    @classmethod
    def IS_CHANGED(cls, path, sha256, **kwargs):
        stat = Path(path).stat()
        return (sha256, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class MingModel(LocalFile):
    RETURN_TYPES = ("MODEL",)

    def load(self, path, sha256):
        import comfy.model_base
        import comfy.sd

        model = comfy.sd.load_diffusion_model(checked_file(path, sha256), model_options={})
        if model is None or not isinstance(model.model, comfy.model_base.MingImage):
            raise ValueError("This checkpoint is not a Ming Image model.")
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as source:
            supplied = {
                key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.") for key in source.keys()
            }
        expected = {name.removeprefix("diffusion_model.") for name, _ in model.model.named_parameters()}
        if missing := expected - supplied:
            raise ValueError(f"Incomplete Ming checkpoint; missing parameter: {sorted(missing)[0]}")
        return (model,)


class MingTextEncoder(LocalFile):
    RETURN_TYPES = ("CLIP",)

    def load(self, path, sha256):
        import comfy.sd
        import comfy.text_encoders.ming_image

        clip = comfy.sd.load_clip(ckpt_paths=[checked_file(path, sha256)], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
        if not isinstance(clip.tokenizer, comfy.text_encoders.ming_image.MingImageTokenizer):
            raise ValueError("This text encoder is not for Ming Image.")
        return (clip,)


class MingVAE(LocalFile):
    RETURN_TYPES = ("VAE",)

    def load(self, path, sha256):
        import comfy.sd
        import comfy.utils

        state, metadata = comfy.utils.load_torch_file(checked_file(path, sha256), safe_load=True, return_metadata=True)
        vae = comfy.sd.VAE(sd=state, metadata=metadata)
        vae.throw_exception_if_invalid()
        if vae.latent_channels != 16 or vae.latent_dim != 3:
            raise ValueError("This VAE does not have Ming Image's 16-channel video latent layout.")
        return (vae,)


def linear_patches(model, weights):
    """Only complete, shape-compatible linear pairs are accepted; no ignored keys."""
    import comfy.lora

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    state = model.model.state_dict()
    for name in state:
        if name.startswith("diffusion_model.") and name.endswith(".weight"):
            plain = name.removeprefix("diffusion_model.").removesuffix(".weight")
            for prefix in ("", "transformer.", "base_model.model."):
                key_map[prefix + plain] = name
    groups = {}
    suffixes = (
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".alpha", "alpha"),
    )
    for key, tensor in weights.items():
        match = next(((key.removesuffix(suffix), side) for suffix, side in suffixes if key.endswith(suffix)), None)
        if match is None:
            raise ValueError(f"Unsupported LoRA key (nothing will be partially applied): {key}")
        name, side = match
        target = key_map.get(name)
        if not isinstance(target, str) or target not in state:
            raise ValueError(f"LoRA target is not present in this Ming model: {name}")
        group = groups.setdefault(target, {})
        if side in group:
            raise ValueError(f"Duplicate LoRA target: {name}")
        group[side] = tensor
    if not groups:
        raise ValueError("No applicable LoRA layers.")
    canonical = {}
    for target, group in groups.items():
        if not {"up", "down"}.issubset(group):
            raise ValueError(f"Incomplete LoRA pair: {target}")
        down, up = group["down"], group["up"]
        shape = state[target].shape
        if hasattr(model.model, "get_submodule"):
            layer = model.model.get_submodule(target.removesuffix(".weight"))
            if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                # state_dict serializes W4A8 as packed bytes; LoRA targets its logical dimensions.
                shape = (layer.out_features, layer.in_features)
        if len(shape) != 2 or down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or down.shape[0] <= 0:
            raise ValueError(f"Unsupported LoRA shape: {target}")
        if (up.shape[0], down.shape[1]) != tuple(shape):
            raise ValueError(f"LoRA dimensions do not match: {target}")
        base = target.removesuffix(".weight")
        canonical[base + ".lora_down.weight"] = down
        canonical[base + ".lora_up.weight"] = up
        if "alpha" in group:
            if group["alpha"].numel() != 1 or not math.isfinite(group["alpha"].item()):
                raise ValueError(f"Invalid LoRA alpha: {target}")
            canonical[base + ".alpha"] = group["alpha"]
    patches = comfy.lora.load_lora(canonical, {name.removesuffix(".weight"): name for name in groups})
    if set(patches) != set(groups):
        raise ValueError("Some LoRA layers could not be loaded.")
    return patches


class MingLoRA(LocalFile):
    RETURN_TYPES = ("MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0}),
            }
        }

    def load(self, model, path, sha256, strength):
        import comfy.utils

        if not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError("LoRA strength must be between -2 and 2.")
        if strength == 0:
            return (model,)
        weights = comfy.utils.load_torch_file(checked_file(path, sha256), safe_load=True)
        patches = linear_patches(model, weights)
        result = model.clone()
        applied = result.add_patches(patches, strength)
        if set(applied) != set(patches):
            raise ValueError("Some LoRA layers could not be applied.")
        return (result,)


NODE_CLASS_MAPPINGS = {
    "AikimiMingModel": MingModel,
    "AikimiMingTextEncoder": MingTextEncoder,
    "AikimiMingVAE": MingVAE,
    "AikimiMingLoRA": MingLoRA,
}
