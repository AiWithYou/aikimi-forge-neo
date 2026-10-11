"""Pinned Orbit LoRA loader; canonical bytes never use Comfy's path-only cache."""

from . import orbit_bytes


def target_pairs(tensors):
    down_suffix, up_suffix = ".lora_A.weight", ".lora_B.weight"
    down = {name.removesuffix(down_suffix) for name in tensors if name.endswith(down_suffix)}
    up = {name.removesuffix(up_suffix) for name in tensors if name.endswith(up_suffix)}
    if (
        not down
        or down != up
        or len(tensors) != 2 * len(down)
        or any(not name.startswith("diffusion_model.") for name in down)
    ):
        raise ValueError("Orbit LoRAの対象テンソル構成が不正です。")
    return {name + ".weight": (tensors[name + down_suffix], tensors[name + up_suffix]) for name in sorted(down)}


def validate_targets(model, pairs):
    state = model.model.state_dict()
    for target, (down, up) in pairs.items():
        if target not in state:
            raise ValueError(f"Orbit LoRAの対象が本体にありません: {target}")
        try:
            layer = model.model.get_submodule(target.removesuffix(".weight"))
        except (AttributeError, KeyError):
            layer = None
        in_features, out_features = getattr(layer, "in_features", None), getattr(layer, "out_features", None)
        shape = (
            (out_features, in_features)
            if type(in_features) is int and type(out_features) is int
            else tuple(state[target].shape)
        )
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or shape != (up.shape[0], down.shape[1]):
            raise ValueError(f"Orbit LoRAと本体の形状が一致しません: {target}")


def validate_patches(patched, before, pairs, strength):
    for target, (down, up) in pairs.items():
        previous = before[target]
        current = patched.patches.get(target, ())
        if len(current) != len(previous) + 1 or any(
            old is not new for old, new in zip(previous, current[:-1], strict=True)
        ):
            raise ValueError(f"Orbit LoRAの適用を確認できません: {target}")
        entry = current[-1]
        if len(entry) != 5 or entry[0] != strength or entry[2] != 1.0 or entry[3] is not None or entry[4] is not None:
            raise ValueError(f"Orbit LoRAの指定強度での適用を確認できません: {target}")
        adapter = entry[1]
        weights = getattr(adapter, "weights", ())
        if (
            getattr(adapter, "name", None) != "lora"
            or len(weights) != 6
            or weights[0] is not up
            or weights[1] is not down
            or any(value is not None for value in weights[2:])
        ):
            raise ValueError(f"Orbit LoRAの受付テンソルの適用を確認できません: {target}")


class MiniMaxH3OrbitLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "canonical_path": ("STRING",),
                "size": ("INT", {"min": 1, "max": 2**53 - 1}),
                "sha256": ("STRING",),
                "version": ("STRING",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, canonical_path, size, sha256, version, strength_model=1.0):
        import math

        import folder_paths
        from comfy.sd import load_lora_for_models
        from safetensors.torch import load

        if isinstance(strength_model, bool) or not math.isfinite(strength_model) or not 0.05 <= strength_model <= 2:
            raise ValueError("Orbit LoRAの強さは0.05〜2.0で指定してください。")
        data, receipt = orbit_bytes.read_verified(
            canonical_path, folder_paths.get_folder_paths("loras"), size, sha256, version
        )
        tensors = load(data)
        orbit_bytes.verify_version(canonical_path, receipt["version"])
        pairs = target_pairs(tensors)
        validate_targets(model, pairs)
        before = {target: tuple(model.patches.get(target, ())) for target in pairs}
        patched, _ = load_lora_for_models(model, None, tensors, strength_model, 0)
        orbit_bytes.verify_version(canonical_path, receipt["version"])
        validate_patches(patched, before, pairs, strength_model)
        receipt.update(applied_targets=list(pairs), strength_model=strength_model)
        return {"ui": {"orbit_receipt": [receipt]}, "result": (patched,)}


NODE_CLASS_MAPPINGS = {"MiniMaxH3OrbitLora": MiniMaxH3OrbitLora}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3OrbitLora": "MiniMax H3 Verified Orbit LoRA"}
