from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import torch

from .convert_krea2_int8_convrot import (
    convert_checkpoint as convert_streaming_checkpoint,
)
from .convert_krea2_int8_convrot import (
    source_layout,
)

logger = logging.getLogger(__name__)

ANIMA38_BLOCK_COUNT = 52
ANIMA38_QUANTIZED_PROJECTIONS = (
    "cross_attn.k_proj",
    "cross_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.v_proj",
    "mlp.layer1",
    "mlp.layer2",
    "self_attn.k_proj",
    "self_attn.output_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)
ANIMA38_QUANTIZED_WEIGHT_KEYS = frozenset(
    f"net.blocks.{block}.{projection}.weight"
    for block in range(ANIMA38_BLOCK_COUNT)
    for projection in ANIMA38_QUANTIZED_PROJECTIONS
)
ANIMA38_REQUIRED_METADATA = {
    "old_block_count": "40",
    "new_block_count": "52",
    "architecture_expansion": "LLaMA-Pro interleaved identity blocks",
    "train_only_inserted_blocks": "1",
}
ANIMA38_PROFILE = "anima38_main_attention_mlp_v1"
ANIMA38_V11_BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
ANIMA38_V11_BUNDLE_FORMAT = "1"
ANIMA38_V11_CONNECTOR_PREFIX = "net.anima_v2_connector."
ANIMA38_V11_PROFILE = "anima38_v11_main_attention_mlp_v1"


def write_sha256_sidecar(output_path: Path) -> Path:
    sidecar = Path(f"{output_path}.sha256")
    partial = Path(f"{sidecar}.part")
    if sidecar.exists():
        raise FileExistsError(f"Checksum sidecar already exists: {sidecar}")
    if partial.exists():
        raise FileExistsError(f"Partial checksum sidecar already exists: {partial}")

    digest = hashlib.sha256()
    with output_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    payload = f"{digest.hexdigest()}  {output_path.name}\n"
    with partial.open("x", encoding="ascii", newline="\n") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(partial, sidecar)
    return sidecar


def validate_anima38_source(
    source_path: Path, group_size: int
) -> tuple[list[str], dict[str, str]]:
    keys, metadata = source_layout(
        source_path,
        group_size,
        quantized_weight_keys=ANIMA38_QUANTIZED_WEIGHT_KEYS,
        profile_name="Anima 3.8B",
    )
    mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in ANIMA38_REQUIRED_METADATA.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            "Source is not the pinned Anima 3.8B Pro52 layout: " + details
        )
    return keys, metadata


def conversion_profile(
    keys: list[str], metadata: dict[str, str]
) -> tuple[str, str]:
    is_v11 = metadata.get("architecture") == ANIMA38_V11_BUNDLE_ARCHITECTURE
    has_v11_connector = any(
        key.startswith(ANIMA38_V11_CONNECTOR_PREFIX) for key in keys
    )
    if is_v11:
        if metadata.get("anima_v2_bundle_format") != ANIMA38_V11_BUNDLE_FORMAT:
            raise ValueError("Anima 3.8B v1.1 has an unsupported bundle format.")
        if not has_v11_connector:
            raise ValueError("Anima 3.8B v1.1 bundle has no Semantic Connector v2.")
        return (
            ANIMA38_V11_PROFILE,
            "adaln,embeddings,input_output,norms,semantic_connector_v2",
        )
    if has_v11_connector:
        raise ValueError(
            "Semantic Connector v2 tensors require the official v1.1 bundle metadata."
        )
    return ANIMA38_PROFILE, "adaln,embeddings,input_output,norms"


def convert_checkpoint(
    source_path: Path,
    output_path: Path,
    device: torch.device,
    group_size: int,
) -> None:
    sidecar = Path(f"{output_path}.sha256")
    partial_sidecar = Path(f"{sidecar}.part")
    if sidecar.exists() or partial_sidecar.exists():
        raise FileExistsError(
            "Remove the existing checksum record before creating a new checkpoint: "
            f"{sidecar if sidecar.exists() else partial_sidecar}"
        )
    keys, metadata = validate_anima38_source(source_path, group_size)
    profile, preserved = conversion_profile(keys, metadata)
    convert_streaming_checkpoint(
        source_path,
        output_path,
        device,
        group_size,
        quantized_weight_keys=ANIMA38_QUANTIZED_WEIGHT_KEYS,
        profile_name="Anima 3.8B",
        metadata_updates={
            "forge.quantization.profile": profile,
            "forge.quantization.preserved": preserved,
        },
    )
    checksum_path = write_sha256_sidecar(output_path.resolve())
    logger.info("SHA256: %s", checksum_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the BF16 Anima 3.8B Pro52 checkpoint to Forge-native "
            "INT8 tensorwise + ConvRot while preserving AdaLN, boundary layers, "
            "and the bundled v1.1 Semantic Connector v2 when present."
        )
    )
    parser.add_argument(
        "source", type=Path, help="BF16 Anima 3.8B Pro52 checkpoint"
    )
    parser.add_argument(
        "output", type=Path, help="New Anima 3.8B INT8 ConvRot checkpoint"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for quantization (default: cuda:0)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=256,
        help="ConvRot group size (default: 256)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        convert_checkpoint(
            args.source,
            args.output,
            torch.device(args.device),
            args.group_size,
        )
    except Exception as error:
        logger.error("ERROR: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
