"""Small, JSON-safe description of Forge's actually loaded diffusion model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field


class ForgeModelStatus(BaseModel):
    loaded: bool = Field(title="Loaded")
    architecture: str | None = Field(default=None, title="Diffusion engine")
    configuration: str | None = Field(default=None, title="Model configuration")
    transformer: str | None = Field(default=None, title="Transformer implementation")
    checkpoint: str | None = Field(default=None, title="Loaded checkpoint path")
    checkpoint_sha256: str | None = Field(default=None, title="Checkpoint SHA256")
    additional_modules: list[str] = Field(title="Loaded VAE and text encoder modules")
    quantization: dict[str, object] = Field(
        title="Loaded transformer quantization summary"
    )
    inspection_errors: list[str] = Field(title="Runtime inspection errors")


def qualified_type_name(value: object | None) -> str | None:
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def quantization_summary(transformer: object | None) -> dict[str, object]:
    formats: Counter[str] = Counter()
    convrot_layers = 0
    convrot_group_sizes: set[int] = set()
    if transformer is None or not callable(getattr(transformer, "modules", None)):
        return {
            "quantized_layer_count": 0,
            "formats": {},
            "convrot_layer_count": 0,
            "convrot_group_sizes": [],
        }

    for module in transformer.modules():
        quant_format = getattr(module, "quant_format", None)
        if quant_format:
            formats[str(quant_format)] += 1
        else:
            quant_type = getattr(module, "quant_type", None)
            if quant_type:
                formats[str(quant_type)] += 1

        weight = getattr(module, "weight", None)
        params = getattr(weight, "_params", None)
        if bool(getattr(params, "convrot", False)):
            convrot_layers += 1
            group_size = getattr(params, "convrot_groupsize", None)
            if group_size is not None:
                convrot_group_sizes.add(int(group_size))

    return {
        "quantized_layer_count": sum(formats.values()),
        "formats": dict(sorted(formats.items())),
        "convrot_layer_count": convrot_layers,
        "convrot_group_sizes": sorted(convrot_group_sizes),
    }


def describe_loaded_model(
    sd_model: object, loading_parameters: dict[str, object]
) -> dict[str, object]:
    errors: list[str] = []
    forge_objects = getattr(sd_model, "forge_objects", None)
    loaded = forge_objects is not None
    model_config = getattr(sd_model, "model_config", None) if loaded else None
    transformer = None
    if loaded:
        try:
            transformer = forge_objects.unet.model.diffusion_model
        except AttributeError as exc:
            errors.append(f"transformer inspection failed: {exc}")

    checkpoint_info = loading_parameters.get("checkpoint_info")
    checkpoint = getattr(checkpoint_info, "filename", None)
    if checkpoint is None:
        checkpoint = getattr(sd_model, "filename", None)
    checkpoint_sha256 = getattr(checkpoint_info, "sha256", None)
    additional_modules = loading_parameters.get("additional_modules") or []
    if isinstance(additional_modules, (str, Path)):
        additional_modules = [additional_modules]

    return {
        "loaded": loaded,
        "architecture": qualified_type_name(sd_model) if loaded else None,
        "configuration": qualified_type_name(model_config),
        "transformer": qualified_type_name(transformer),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": str(checkpoint_sha256) if checkpoint_sha256 else None,
        "additional_modules": [str(value) for value in additional_modules],
        "quantization": quantization_summary(transformer),
        "inspection_errors": errors,
    }
