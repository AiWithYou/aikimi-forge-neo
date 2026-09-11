from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import BinaryIO

import torch
from comfy_kitchen.tensor import TensorWiseINT8Layout
from safetensors import safe_open


logger = logging.getLogger(__name__)


GIB = 1024**3
CONVERSION_HEADROOM_BYTES = 2 * GIB
MAX_STREAMING_TENSOR_WINDOW_BYTES = 4 * GIB
ESTIMATED_OUTPUT_RATIO = 0.70
KREA2_BLOCK_COUNT = 28
KREA2_QUANTIZED_PROJECTIONS = (
    "attn.gate",
    "attn.wk",
    "attn.wo",
    "attn.wq",
    "attn.wv",
    "mlp.down",
    "mlp.gate",
    "mlp.up",
)
KREA2_QUANTIZED_WEIGHT_KEYS = frozenset(
    f"blocks.{block}.{projection}.weight"
    for block in range(KREA2_BLOCK_COUNT)
    for projection in KREA2_QUANTIZED_PROJECTIONS
)

SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        if self.dtype not in SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"Unsupported SafeTensors dtype: {self.dtype}")
        return math.prod(self.shape) * SAFETENSORS_DTYPE_BYTES[self.dtype]


def conversion_resource_requirements(source_bytes: int) -> dict[str, int]:
    if source_bytes <= 0:
        raise ValueError("source checkpoint size must be greater than 0")
    estimated_output_bytes = max(GIB, math.ceil(source_bytes * ESTIMATED_OUTPUT_RATIO))
    return {
        "source_bytes": int(source_bytes),
        "estimated_output_bytes": estimated_output_bytes,
        "required_free_disk_bytes": estimated_output_bytes
        + CONVERSION_HEADROOM_BYTES,
        # The converter writes one tensor at a time, so it does not need to
        # retain both the source and the complete output checkpoint in RAM.
        "required_available_memory_bytes": min(
            int(source_bytes), MAX_STREAMING_TENSOR_WINDOW_BYTES
        )
        + CONVERSION_HEADROOM_BYTES,
    }


def validate_conversion_resources(
    source_path: Path,
    output_path: Path,
    *,
    available_memory_bytes: int | None = None,
    free_disk_bytes: int | None = None,
    profile_name: str = "Krea2",
) -> dict[str, int]:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_path}")
    requirements = conversion_resource_requirements(source_path.stat().st_size)
    if available_memory_bytes is None:
        import psutil

        available_memory_bytes = int(psutil.virtual_memory().available)
    if free_disk_bytes is None:
        free_disk_bytes = int(shutil.disk_usage(output_path.parent).free)
    if available_memory_bytes < requirements["required_available_memory_bytes"]:
        raise RuntimeError(
            f"Insufficient available RAM for {profile_name} conversion: "
            f"need at least {requirements['required_available_memory_bytes'] / GIB:.2f} GiB, "
            f"have {available_memory_bytes / GIB:.2f} GiB. Stop memory-heavy processes "
            "before converting."
        )
    if free_disk_bytes < requirements["required_free_disk_bytes"]:
        raise RuntimeError(
            f"Insufficient free output-disk space for {profile_name} conversion: "
            f"need at least {requirements['required_free_disk_bytes'] / GIB:.2f} GiB, "
            f"have {free_disk_bytes / GIB:.2f} GiB."
        )
    return {
        **requirements,
        "available_memory_bytes": int(available_memory_bytes),
        "free_disk_bytes": int(free_disk_bytes),
    }


def quant_config(group_size: int) -> dict[str, object]:
    return {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": group_size,
    }


def quant_config_tensor(group_size: int) -> torch.Tensor:
    payload = json.dumps(quant_config(group_size)).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def companion_keys(weight_key: str) -> tuple[str, str]:
    layer_name, suffix = weight_key.rsplit(".", 1)
    if suffix != "weight":
        raise ValueError(f"Expected a weight key, got {weight_key}")
    return f"{layer_name}.weight_scale", f"{layer_name}.comfy_quant"


def output_tensor_specs(
    source_path: Path,
    source_keys: list[str],
    group_size: int,
    *,
    quantized_weight_keys: frozenset[str] = KREA2_QUANTIZED_WEIGHT_KEYS,
) -> list[TensorSpec]:
    config_size = quant_config_tensor(group_size).numel()
    specs: list[TensorSpec] = []
    with safe_open(source_path, framework="pt", device="cpu") as source:
        for key in source_keys:
            tensor_slice = source.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())
            if key not in quantized_weight_keys:
                specs.append(TensorSpec(key, dtype, shape))
                continue

            scale_key, config_key = companion_keys(key)
            specs.extend(
                (
                    TensorSpec(key, "I8", shape),
                    TensorSpec(scale_key, "F32", (shape[0], 1)),
                    TensorSpec(config_key, "U8", (config_size,)),
                )
            )
    return specs


def build_safetensors_header(
    specs: list[TensorSpec], metadata: dict[str, str]
) -> tuple[bytes, int]:
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for spec in specs:
        end = offset + spec.nbytes
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, end],
        }
        offset = end

    encoded = json.dumps(
        header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return struct.pack("<Q", len(encoded)) + encoded, offset


def write_tensor_bytes(
    destination: BinaryIO, tensor: torch.Tensor, expected_bytes: int, name: str
) -> None:
    cpu_tensor = tensor.detach().cpu().contiguous()
    byte_view = cpu_tensor.reshape(-1).view(torch.uint8).numpy()
    start = destination.tell()
    remaining = memoryview(byte_view)
    while remaining:
        written = destination.write(remaining[: 64 * 1024 * 1024])
        if not written:
            raise IOError(f"Failed while writing tensor bytes for {name}")
        remaining = remaining[written:]
    written = destination.tell() - start
    if written != expected_bytes:
        raise IOError(
            f"Tensor byte count mismatch for {name}: wrote {written}, expected {expected_bytes}"
        )


def source_layout(
    path: Path,
    group_size: int,
    *,
    quantized_weight_keys: frozenset[str] = KREA2_QUANTIZED_WEIGHT_KEYS,
    profile_name: str = "Krea2",
) -> tuple[list[str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"Source must be a .safetensors checkpoint: {path}")
    if group_size <= 0:
        raise ValueError(f"ConvRot group size must be positive, got {group_size}")

    with safe_open(path, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        key_set = set(keys)
        missing = sorted(quantized_weight_keys - key_set)
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Source is not the expected {profile_name} layout: {len(missing)} quantized weights are missing ({preview})"
            )
        prequantized = sorted(key for key in keys if key.endswith(".comfy_quant"))
        if prequantized:
            raise ValueError(
                f"Source already contains quantization metadata ({prequantized[0]}); convert the BF16 merged checkpoint, not a quantized model"
            )

        for key in sorted(quantized_weight_keys):
            tensor_slice = source.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())
            if len(shape) != 2:
                raise ValueError(f"ConvRot requires 2D weights, but {key} has shape {shape}")
            if shape[1] % group_size != 0:
                raise ValueError(
                    f"ConvRot group size {group_size} does not divide {key}'s input dimension {shape[1]}"
                )
            if dtype != "BF16":
                raise ValueError(
                    f"Expected a BF16 merged source, but {key} is {dtype}; refusing lossy re-quantization"
                )

        metadata = dict(source.metadata() or {})

    return keys, metadata


def output_metadata(
    source: Path,
    output: Path,
    source_metadata: dict[str, str],
    group_size: int,
    *,
    quantized_weight_keys: frozenset[str] = KREA2_QUANTIZED_WEIGHT_KEYS,
    metadata_updates: dict[str, str] | None = None,
) -> dict[str, str]:
    metadata = dict(source_metadata)
    tags = [tag.strip() for tag in metadata.get("modelspec.tags", "").split(",") if tag.strip()]
    tags = [tag for tag in tags if tag.lower() != "bf16"]
    for tag in ("int8_tensorwise", "convrot"):
        if tag not in tags:
            tags.append(tag)

    metadata.update(
        {
            "modelspec.title": output.stem,
            "modelspec.tags": ",".join(tags),
            "modelspec.quantization": "int8_tensorwise+convrot",
            "modelspec.quantized_from": source.name,
            "forge.quantization.format": "int8_tensorwise",
            "forge.quantization.convrot": "true",
            "forge.quantization.convrot_groupsize": str(group_size),
            "forge.quantization.quantized_layers": str(len(quantized_weight_keys)),
            "forge.quantization.comfy_kitchen": version("comfy-kitchen"),
        }
    )
    if metadata_updates:
        metadata.update(metadata_updates)
    return metadata


def validate_checkpoint(
    source_path: Path,
    output_path: Path,
    group_size: int,
    *,
    quantized_weight_keys: frozenset[str] = KREA2_QUANTIZED_WEIGHT_KEYS,
    profile_name: str = "Krea2",
) -> None:
    expected_config = quant_config(group_size)
    with safe_open(source_path, framework="pt", device="cpu") as source, safe_open(
        output_path, framework="pt", device="cpu"
    ) as output:
        source_keys = set(source.keys())
        expected_output_keys = set(source_keys)
        for weight_key in quantized_weight_keys:
            expected_output_keys.update(companion_keys(weight_key))

        output_keys = set(output.keys())
        missing = sorted(expected_output_keys - output_keys)
        unexpected = sorted(output_keys - expected_output_keys)
        if missing or unexpected:
            raise ValueError(
                f"Output key mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
            )

        for key in sorted(source_keys - quantized_weight_keys):
            source_slice = source.get_slice(key)
            output_slice = output.get_slice(key)
            if source_slice.get_shape() != output_slice.get_shape():
                raise ValueError(f"Non-quantized tensor shape changed: {key}")
            if source_slice.get_dtype() != output_slice.get_dtype():
                raise ValueError(f"Non-quantized tensor dtype changed: {key}")

        for weight_key in sorted(quantized_weight_keys):
            scale_key, config_key = companion_keys(weight_key)
            source_shape = tuple(source.get_slice(weight_key).get_shape())
            output_weight = output.get_slice(weight_key)
            output_scale = output.get_slice(scale_key)
            if str(output_weight.get_dtype()) != "I8" or tuple(output_weight.get_shape()) != source_shape:
                raise ValueError(f"Invalid INT8 weight tensor: {weight_key}")
            if str(output_scale.get_dtype()) != "F32" or tuple(output_scale.get_shape()) != (source_shape[0], 1):
                raise ValueError(f"Invalid ConvRot scale tensor: {scale_key}")
            scale = output.get_tensor(scale_key)
            if not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise ValueError(f"ConvRot scale must be finite and positive: {scale_key}")
            loaded_config = json.loads(output.get_tensor(config_key).numpy().tobytes())
            if loaded_config != expected_config:
                raise ValueError(f"Invalid quantization metadata for {config_key}: {loaded_config}")

        metadata = output.metadata() or {}
        if metadata.get("forge.quantization.quantized_layers") != str(len(quantized_weight_keys)):
            raise ValueError(
                f"Output metadata does not record all {len(quantized_weight_keys)} quantized {profile_name} layers"
            )


def convert_checkpoint(
    source_path: Path,
    output_path: Path,
    device: torch.device,
    group_size: int,
    *,
    quantized_weight_keys: frozenset[str] = KREA2_QUANTIZED_WEIGHT_KEYS,
    profile_name: str = "Krea2",
    metadata_updates: dict[str, str] | None = None,
) -> None:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("Source and output checkpoints must be different files")
    if output_path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite a checkpoint: {output_path}")

    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")
    if partial_path.exists():
        raise FileExistsError(
            f"Partial output already exists; it was left intact for inspection: {partial_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resource_report = validate_conversion_resources(
        source_path, output_path, profile_name=profile_name
    )
    logger.info(
        "Resource preflight: RAM %.2f/%.2f GiB available/required, disk %.2f/%.2f GiB free/required",
        resource_report["available_memory_bytes"] / GIB,
        resource_report["required_available_memory_bytes"] / GIB,
        resource_report["free_disk_bytes"] / GIB,
        resource_report["required_free_disk_bytes"] / GIB,
    )

    if device.type != "cuda":
        raise ValueError(f"{profile_name} ConvRot conversion requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable; cannot quantize {profile_name} ConvRot weights"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise RuntimeError(f"INT8 tensor cores require SM 7.5 or newer, found SM {capability[0]}.{capability[1]}")

    source_keys, source_metadata = source_layout(
        source_path,
        group_size,
        quantized_weight_keys=quantized_weight_keys,
        profile_name=profile_name,
    )
    metadata = output_metadata(
        source_path,
        output_path,
        source_metadata,
        group_size,
        quantized_weight_keys=quantized_weight_keys,
        metadata_updates=metadata_updates,
    )
    config_tensor = quant_config_tensor(group_size)
    specs = output_tensor_specs(
        source_path,
        source_keys,
        group_size,
        quantized_weight_keys=quantized_weight_keys,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    header, expected_data_bytes = build_safetensors_header(specs, metadata)
    started = time.perf_counter()

    logger.info("Source: %s", source_path)
    logger.info("Output: %s", output_path)
    logger.info(
        f"Device: {torch.cuda.get_device_name(device)} (SM {capability[0]}.{capability[1]})",
    )
    logger.info(
        f"Profile: int8_tensorwise + ConvRot, group size {group_size}, {len(quantized_weight_keys)} weights",
    )

    logger.info("Streaming temporary checkpoint: %s", partial_path)
    with partial_path.open("xb", buffering=0) as destination:
        destination.write(header)
        data_start = destination.tell()
        with safe_open(source_path, framework="pt", device="cpu") as source:
            quantized_index = 0
            for key in source_keys:
                if key not in quantized_weight_keys:
                    source_tensor = source.get_tensor(key)
                    write_tensor_bytes(
                        destination,
                        source_tensor,
                        specs_by_name[key].nbytes,
                        key,
                    )
                    del source_tensor
                    continue

                quantized_index += 1
                weight = source.get_tensor(key).to(device=device, non_blocking=False)
                qdata, params = TensorWiseINT8Layout.quantize(
                    weight,
                    per_channel=True,
                    convrot=True,
                    convrot_groupsize=group_size,
                    stochastic_rounding=0,
                )
                scale_key, config_key = companion_keys(key)
                qdata_cpu = qdata.cpu().contiguous()
                scale_cpu = params.scale.cpu().float().contiguous()
                write_tensor_bytes(
                    destination, qdata_cpu, specs_by_name[key].nbytes, key
                )
                write_tensor_bytes(
                    destination,
                    scale_cpu,
                    specs_by_name[scale_key].nbytes,
                    scale_key,
                )
                write_tensor_bytes(
                    destination,
                    config_tensor,
                    specs_by_name[config_key].nbytes,
                    config_key,
                )
                del weight, qdata, params, qdata_cpu, scale_cpu
                torch.cuda.empty_cache()
                elapsed = time.perf_counter() - started
                logger.info(
                    f"[{quantized_index:03d}/{len(quantized_weight_keys)}] {key} ({elapsed:.1f}s)",
                )

        written_data_bytes = destination.tell() - data_start
        if written_data_bytes != expected_data_bytes:
            raise IOError(
                "Checkpoint data byte count mismatch: "
                f"wrote {written_data_bytes}, expected {expected_data_bytes}"
            )
        destination.flush()
        os.fsync(destination.fileno())

    logger.info("Validating all tensor headers and ConvRot metadata...")
    validate_checkpoint(
        source_path,
        partial_path,
        group_size,
        quantized_weight_keys=quantized_weight_keys,
        profile_name=profile_name,
    )
    os.rename(partial_path, output_path)
    elapsed = time.perf_counter() - started
    logger.info("Completed: %s", output_path)
    logger.info("Bytes: %d", output_path.stat().st_size)
    logger.info("Elapsed seconds: %.1f", elapsed)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Krea2 BF16 merged checkpoint to native INT8 tensorwise + ConvRot safetensors."
    )
    parser.add_argument("source", type=Path, help="Krea2 BF16 merged .safetensors checkpoint")
    parser.add_argument("output", type=Path, help="New INT8 ConvRot .safetensors checkpoint")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used for quantization (default: cuda:0)")
    parser.add_argument("--group-size", type=int, default=256, help="ConvRot group size (default: 256)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        convert_checkpoint(args.source, args.output, torch.device(args.device), args.group_size)
    except Exception as error:
        logger.error("ERROR: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
