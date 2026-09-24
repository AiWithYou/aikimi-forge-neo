"""Download or verify the pinned Qwen Image 2.1 Fun Union INT8 ConvRot patch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.qwen_image21.core import atomic_json, runtime_lock, runtime_manifest  # noqa: E402
from modules_forge.qwen_image21.fun_controlnet import (  # noqa: E402
    FILENAME,
    REPOSITORY,
    REVISION,
    SHA256,
    SIZE,
    checkpoint_path,
    inspect_checkpoint,
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
        path = checkpoint_path(runtime)
        if args.download:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(
                REPOSITORY, filename=FILENAME, revision=REVISION,
                local_dir=runtime / "controlnet",
            ))
        details = inspect_checkpoint(path)
        if path.stat().st_size != SIZE or file_hash(path) != SHA256:
            raise ValueError("Fun ControlNetのサイズまたはSHA-256が配布元と一致しません。")
        receipt = {"repository": REPOSITORY, "revision": REVISION, "filename": FILENAME,
                   "size": SIZE, "sha256": SHA256, **details}
        atomic_json(path.with_suffix(".json"), receipt)
        sys.stdout.write(f"Fun ControlNet INT8 verified: {path}\n")


if __name__ == "__main__":
    main()
