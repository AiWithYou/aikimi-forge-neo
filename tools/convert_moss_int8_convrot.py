"""Convert an unquantized, merged MOSS DiT into native Comfy INT8 ConvRot."""

import argparse
import logging
from pathlib import Path

import torch

from modules_forge import local_assets
from modules_forge.moss_image.checkpoints import QUANTIZED_KEYS, header_precision
from tools.convert_anima38_int8_convrot import write_sha256_sidecar
from tools.convert_krea2_int8_convrot import convert_checkpoint

MOSS_PROFILE = "moss_sft_main_252_v1"


def quantized_keys():
    return QUANTIZED_KEYS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=256)
    args = parser.parse_args(argv)
    if header_precision(local_assets.read_header(args.source)) != "bf16":
        raise ValueError("MOSS conversion requires an unquantized merged source")
    source = args.source.resolve()
    before = local_assets.file_version(source)
    source_sha = local_assets._digest(str(source), *before)
    convert_checkpoint(
        args.source,
        args.output,
        torch.device(args.device),
        args.group_size,
        quantized_weight_keys=QUANTIZED_KEYS,
        profile_name="MOSS Image",
        metadata_updates={
            "forge.quantization.profile": MOSS_PROFILE,
            "modelspec.architecture": "moss-image",
            "forge.quantization.source_sha256": source_sha,
        },
    )
    if before != local_assets.file_version(source):
        raise ValueError("The merged source changed during MOSS quantization")
    write_sha256_sidecar(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
