"""Export reloadable rowwise INT8 Iris weights; publication is a separate gate."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.iris.core import (  # noqa: E402
    CODE_REVISION,
    RUNTIME,
    SOURCE_REPO,
    SOURCE_REVISION,
    atomic_json,
    model_directory,
    sha256,
)
from modules_forge.iris.quantization import convert_linears  # noqa: E402
from modules_forge.iris.runtime import load_model  # noqa: E402


def export(root, task):
    source = model_directory(root, "normal", task)
    destination = model_directory(root, "int8", task)
    destination.mkdir(parents=True, exist_ok=True)
    model, _, _ = load_model(root, "normal", task)
    report = convert_linears(model)
    target = destination / "model.safetensors"
    save_file(
        model.state_dict(),
        target,
        metadata={
            "format": "iris-rowwise-int8-v1",
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "code_revision": CODE_REVISION,
        },
    )
    files = [target]
    for name in ("config.yaml", "empty_prompt.safetensors"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
            files.append(destination / name)
    manifest = {
        **report,
        "task": task,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_sha256": sha256(source / "model.safetensors"),
        "code_revision": CODE_REVISION,
        "files": [{"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    atomic_json(destination / "manifest.json", manifest)
    print(f"{task}: {report['linear_count']} Linear, {target.stat().st_size:,} bytes", flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--task", choices=["generate", "depth", "upscale", "all"], default="all")
    args = parser.parse_args()
    for task in ["generate", "depth", "upscale"] if args.task == "all" else [args.task]:
        export(args.root, task)
