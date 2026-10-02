"""One interpretation of shared controls for Outpaint's summary and request."""

from pathlib import Path

from .capabilities import validate_sampling
from .core import integer, precision_label
from .style_lora import validate_options

FIELDS = (
    "precision",
    "memory_mode",
    "style_loras",
    "lora_strengths",
    "allow_lora_base_mismatch",
    "fun_acc",
    "sparse_mode",
    "sparse_keep_percent",
    "sparse_jev_cadence",
    "sparse_jev_interval",
    "sparse_jev_max_calls",
    "sparse_jev_max_wait_seconds",
    "control_kind",
    "control_image",
    "control_strength",
    "control_inpaint",
    "local_model",
    "local_components",
    "rewrite_edit_prompt",
    "transparent",
)


def resolve(values, steps):
    from modules_forge.local_assets import lora_settings

    if len(values) != len(FIELDS):
        raise ValueError("Qwenの共通設定を読み込めません。")
    profile = dict(zip(FIELDS, values, strict=True))
    rows = profile.pop("lora_strengths")
    profile["style_loras"] = tuple(lora_settings(profile["style_loras"], rows))
    profile["local_model"] = profile["local_model"] or ""
    profile["local_components"] = profile["local_components"] or ""
    if profile["control_kind"] == "off":
        profile["control_image"] = ""
        profile["control_inpaint"] = False
    else:
        profile["control_image"] = str(profile["control_image"] or "")
    profile["steps"] = (
        4 if profile["precision"].startswith("turbo_") or profile["fun_acc"] else integer(steps, "Steps", 1, 100)
    )
    validate_options(profile)
    validate_sampling(profile)
    return profile


def summary(profile, version):
    model = Path(profile["local_model"]).name if profile["local_model"] else precision_label(profile["precision"])
    styles = [
        f"{Path(item['name']).stem} × {item['strength']:g}" for item in profile["style_loras"] if item["strength"]
    ]
    parts = [
        model,
        "CPU退避" if profile["memory_mode"] == "offload" else "GPU配置",
        "画風LoRA: " + (" / ".join(styles) if styles else "なし"),
        "Outpaint: " + ("LoRAなし" if version == "none" else version),
        f"{profile['steps']} steps",
    ]
    if profile["fun_acc"]:
        parts.append("Fun Acc")
    if profile["sparse_mode"] != "off":
        sparse = "Sparse " + profile["sparse_mode"]
        if profile["sparse_mode"] == "fixed":
            sparse += f" {profile['sparse_keep_percent']:g}%"
        parts.append(sparse)
    if profile["control_kind"] != "off":
        parts.append(
            "ControlNet "
            + profile["control_kind"]
            + f" × {profile['control_strength']:g}"
            + ("＋外側Inpaint" if profile["control_inpaint"] else "")
        )
    if profile["precision"].startswith("turbo_") or profile["fun_acc"]:
        parts.append("追加LoRA・制御の併用は画質実験")
    return " · ".join(parts)
