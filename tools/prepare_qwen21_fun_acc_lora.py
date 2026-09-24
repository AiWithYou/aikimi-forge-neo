"""Download or verify the pinned Qwen Image 2.1 Fun Acc PDD adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.qwen_image21.core import atomic_json, runtime_lock, runtime_manifest  # noqa: E402
from modules_forge.qwen_image21.fun_acc_lora import (  # noqa: E402
    CONFIG,
    CONFIG_SHA256,
    REPOSITORY,
    REVISION,
    WEIGHTS,
    WEIGHTS_SHA256,
    WEIGHTS_SIZE,
    adapter_dir,
    installed,
)
from modules_forge.qwen_image21.quantized_cache import file_hash  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    runtime = ROOT / "models/Qwen-Image-2.1"
    runtime_manifest(runtime, "int8")
    with runtime_lock(runtime):
        directory = adapter_dir(runtime)
        if args.download:
            from huggingface_hub import hf_hub_download

            for filename in (WEIGHTS, CONFIG, "LICENSE"):
                hf_hub_download(REPOSITORY, filename=filename, revision=REVISION, local_dir=directory)
        weights = directory / WEIGHTS
        config = directory / CONFIG
        license_file = directory / "LICENSE"
        if not weights.is_file() or weights.stat().st_size != WEIGHTS_SIZE or file_hash(weights) != WEIGHTS_SHA256:
            raise ValueError("Fun Accの重みのサイズまたはSHA-256が配布元と一致しません。")
        if not config.is_file() or file_hash(config) != CONFIG_SHA256 or not license_file.is_file():
            raise ValueError("Fun Accの設定または利用条件が配布元と一致しません。")
        atomic_json(directory / "manifest.json", {
            "repository": REPOSITORY, "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256,
        })
        installed(runtime)
        sys.stdout.write(f"Fun Acc 4-step LoRA verified: {weights}\n")


if __name__ == "__main__":
    main()
