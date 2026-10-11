"""Download or verify the pinned Iris-3B Turbo LoRA.

Run with models/Iris-3B/worker-env/Scripts/python.exe.
Only the selected safetensors weights are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "pentacoxian-dev/iris-3b-turbo-lora"
REVISION = "fe16b3496eb440585993c92d95641c08c6b08263"
DEFAULT_OUTPUT_DIR = ROOT / "models/Iris-3B/loras/pentacoxian-turbo"


@dataclass(frozen=True)
class Adapter:
    filename: str
    local_filename: str
    size: int
    sha256: str


ADAPTERS = MappingProxyType(
    {
        "r2-1000": Adapter(
            filename="r2-1000/student.safetensors",
            local_filename="iris-3b-turbo-r2-1000.safetensors",
            size=459487256,
            sha256="de8e8f24075d7de12251782f36ac99d2aafc2c58dec7c85297b017a6527028a1",
        ),
        "r1-1500": Adapter(
            filename="r1-1500/student.safetensors",
            local_filename="iris-3b-turbo-r1-1500.safetensors",
            size=459487256,
            sha256="1e937c0091f5d9f772587d577cc6b7af240b832412a00c9722a2daa3d5a5f6b7",
        ),
    }
)


def download_plan(adapter: str = "r2-1000", output_dir: Path | None = None) -> dict:
    specification = ADAPTERS[adapter]
    directory = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "file": specification.filename,
        "size": specification.size,
        "sha256": specification.sha256,
        "output_path": str(directory / specification.local_filename),
    }


def verify(path: Path, specification: Adapter) -> None:
    if not path.is_file() or path.stat().st_size != specification.size:
        raise ValueError("Iris Turbo file size does not match the pinned weights.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != specification.sha256:
        raise ValueError("Iris Turbo SHA-256 does not match the pinned weights.")


def install(adapter: str = "r2-1000", output_dir: Path | None = None) -> dict:
    plan = download_plan(adapter, output_dir)
    specification = ADAPTERS[adapter]
    destination = Path(plan["output_path"])
    if destination.is_file():
        try:
            verify(destination, specification)
        except ValueError:
            pass
        else:
            return {**plan, "cached": True}

    # The dry run and a verified local file do not require Hugging Face or network access.
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the same filesystem, then publish only fully verified weights.
    with tempfile.TemporaryDirectory(prefix=".iris-turbo-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        hf_hub_download(
            REPOSITORY,
            filename=specification.filename,
            revision=REVISION,
            local_dir=staging,
        )
        downloaded = staging / specification.filename
        verify(downloaded, specification)
        os.replace(downloaded, destination)
    return {**plan, "cached": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=tuple(ADAPTERS), default="r2-1000")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the pinned download without saving or network access"
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        result = download_plan(args.adapter, args.output_dir)
    else:
        try:
            result = install(args.adapter, args.output_dir)
        except Exception as error:
            parser.exit(1, f"Iris Turbo preparation failed: {error}\n")
    sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
