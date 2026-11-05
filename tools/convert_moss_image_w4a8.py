"""Convert a merged MOSS BF16 checkpoint to Comfy W4A8 without changing the source."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from safetensors import safe_open

from modules_forge import gpu_fault, local_assets
from modules_forge.moss_image import checkpoints
from tools.convert_krea2_int8_convrot import (
    TensorSpec,
    build_safetensors_header,
    validate_conversion_resources,
    write_tensor_bytes,
)

LOGGER = logging.getLogger(__name__)
CONFIG = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
CONFIG_BYTES = json.dumps(CONFIG, separators=(",", ":")).encode("utf-8")


def quantize_weight(weight, device):
    from comfy_kitchen.registry import registry
    from comfy_kitchen.tensor import AsymW4A8Int8Layout

    if not registry.is_available("cuda"):
        raise RuntimeError("The MOSS runtime's W4A8 CUDA backend is unavailable")
    packed, params = AsymW4A8Int8Layout.quantize(
        weight.to(device=device), group_size=16, convrot_groupsize=256,
        symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True, bits=4,
    )
    if params.correction is not None or params.codebook is None:
        raise RuntimeError("Unsupported W4A8 correction or missing codebook")
    return packed.cpu(), params.scale.cpu(), params.s_channel.cpu(), params.codebook.cpu()


def output_specs(source):
    specs = []
    for key in source.keys():
        item = source.get_slice(key)
        shape = tuple(item.get_shape())
        if key not in checkpoints.QUANTIZED_KEYS:
            specs.append(TensorSpec(key, str(item.get_dtype()), shape))
            continue
        rows, columns = shape
        prefix = key.removesuffix(".weight")
        specs.extend((
            TensorSpec(key, "I8", (rows, columns // 2)),
            TensorSpec(prefix + ".weight_s_rel", "F8_E4M3", (rows, columns // 16)),
            TensorSpec(prefix + ".weight_s_channel", "F32", (rows,)),
            TensorSpec(prefix + ".weight_codebook", "F32", (16,)),
            TensorSpec(prefix + ".comfy_quant", "U8", (len(CONFIG_BYTES),)),
        ))
    return specs


def _convert(source_path, output_path, device):
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("Source and output checkpoints must be different files")
    if output_path.exists():
        raise FileExistsError("Output already exists; refusing to overwrite it")
    partial_path = Path(str(output_path) + ".part")
    if partial_path.exists():
        raise FileExistsError("Partial output already exists; refusing to overwrite it")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MOSS W4A8 conversion requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise RuntimeError("MOSS W4A8 conversion requires Compute Capability 8.0 or newer")
    if checkpoints.header_precision(local_assets.read_header(source_path)) != "bf16":
        raise ValueError("MOSS W4A8 conversion requires a complete BF16 source")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_conversion_resources(source_path, output_path, profile_name="MOSS W4A8")
    before = local_assets.file_version(source_path)
    digest = local_assets._digest(str(source_path), *before)
    created = False
    try:
        with safe_open(source_path, framework="pt", device="cpu") as source:
            metadata = dict(source.metadata() or {})
            metadata.pop("_quantization_metadata", None)
            tags = [x.strip() for x in metadata.get("modelspec.tags", "").split(",") if x.strip().lower() != "bf16"]
            metadata.update({
                "modelspec.title": output_path.stem,
                "modelspec.tags": ",".join([x for x in tags if x] + ["asym_w4a8_int8", "convrot"]),
                "modelspec.quantization": "asym_w4a8_int8+convrot",
                "modelspec.quantized_from": source_path.name,
                "forge.quantization.source_sha256": digest,
                "forge.quantization.quantized_layers": str(len(checkpoints.QUANTIZED_KEYS)),
            })
            specs = output_specs(source)
            by_name = {spec.name: spec for spec in specs}
            header, data_size = build_safetensors_header(specs, metadata)
            with partial_path.open("xb", buffering=0) as destination:
                created = True
                destination.write(header)
                data_start = destination.tell()
                converted = 0
                for key in source.keys():
                    weight = source.get_tensor(key)
                    if not bool(torch.isfinite(weight).all()):
                        raise ValueError(f"Source tensor contains non-finite values: {key}")
                    if key not in checkpoints.QUANTIZED_KEYS:
                        write_tensor_bytes(destination, weight, by_name[key].nbytes, key)
                        continue
                    prefix = key.removesuffix(".weight")
                    names = (key, prefix + ".weight_s_rel", prefix + ".weight_s_channel", prefix + ".weight_codebook")
                    values = quantize_weight(weight, device)
                    for name, value, dtype in zip(
                        names, values, (torch.int8, torch.float8_e4m3fn, torch.float32, torch.float32), strict=True
                    ):
                        spec = by_name[name]
                        if value.dtype != dtype or tuple(value.shape) != spec.shape:
                            raise ValueError(f"Unexpected W4A8 tensor shape or dtype: {name}")
                        if name != key and not bool(torch.isfinite(value.float()).all()):
                            raise ValueError(f"W4A8 tensor contains non-finite values: {name}")
                        write_tensor_bytes(destination, value, spec.nbytes, name)
                    config_key = prefix + ".comfy_quant"
                    config = torch.tensor(list(CONFIG_BYTES), dtype=torch.uint8)
                    write_tensor_bytes(destination, config, by_name[config_key].nbytes, config_key)
                    converted += 1
                    LOGGER.info("Converted tensor %s/%s: %s", converted, len(checkpoints.QUANTIZED_KEYS), key)
                if destination.tell() - data_start != data_size:
                    raise OSError("W4A8 output byte count is incomplete")
                destination.flush()
                os.fsync(destination.fileno())
        if checkpoints.precision(partial_path) != "w4a8":
            raise ValueError("W4A8 output validation failed")
        after = local_assets.file_version(source_path)
        if before != after or local_assets._digest(str(source_path), *after) != digest:
            raise ValueError("The source checkpoint changed during conversion")
        if local_assets.file_version(source_path) != after:
            raise ValueError("The source checkpoint changed during conversion")
        # A hard link publishes atomically without replacing a concurrent output.
        os.link(partial_path, output_path)
    finally:
        if created:
            partial_path.unlink(missing_ok=True)
    LOGGER.info("Wrote %s", output_path)


def convert(source_path, output_path, device):
    gpu_fault.STATE.ensure_available()
    try:
        return _convert(source_path, output_path, device)
    except BaseException as error:
        reason = gpu_fault.classify(error)
        if reason:
            gpu_fault.STATE.trip(reason, "moss-w4a8-converter")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    convert(args.source, args.output, torch.device(args.device))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
