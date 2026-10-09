"""Export reloadable INT8 or packed W4A8 weights; validate before publication."""

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


def export(root, task, precision="int8"):
    source = model_directory(root, "normal", task)
    destination = model_directory(root, precision, task)
    destination.mkdir(parents=True, exist_ok=True)
    model, _, _ = load_model(root, "normal", task)
    if precision == "w4a8":
        from modules_forge.iris.w4a8 import convert_linears as convert_w4a8

        report = convert_w4a8(model)
    else:
        report = convert_linears(model)
    if (
        precision == "w4a8"
        and report["bytes"] >= (model_directory(root, "int8", task) / "model.safetensors").stat().st_size
    ):
        raise ValueError("W4A8の重みがINT8より小さくなっていません。対象層を確認してください。")
    target = destination / "model.safetensors"
    save_file(
        model.state_dict(),
        target,
        metadata={
            "format": report["format"],
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


def export_text(root):
    from iris3b.config import inference_config
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
    from omegaconf import OmegaConf

    from modules_forge.iris.core import TEXT_REPO, TEXT_REVISION
    from modules_forge.iris.w4a8 import convert_linears as convert_w4a8

    root = Path(root)
    raw = OmegaConf.to_container(OmegaConf.load(root / "official/config.yaml"))
    raw.pop("task", None)
    cfg = inference_config(raw, []).text_encoder
    cfg.pretrained = str(root / "text-encoder")
    encoder = Qwen3VLTextEncoder(cfg, device="cpu")
    report = convert_w4a8(encoder.decoder)
    destination = root / "w4a8/text-encoder"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "model.safetensors"
    save_file(encoder.decoder.state_dict(), target, metadata={"format": report["format"]})
    files = [target]
    for name in (
        "config.json",
        "chat_template.json",
        "merges.txt",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    ):
        shutil.copyfile(root / "text-encoder" / name, destination / name)
        files.append(destination / name)
    atomic_json(
        destination / "manifest.json",
        {
            **report,
            "task": "text",
            "source_repo": TEXT_REPO,
            "source_revision": TEXT_REVISION,
            "code_revision": CODE_REVISION,
            "files": [{"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)} for path in files],
        },
    )
    print(f"text: {report['linear_count']} Linear, {target.stat().st_size:,} bytes", flush=True)  # noqa: T201


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--task", choices=["generate", "depth", "upscale", "text", "all"], default="all")
    parser.add_argument("--precision", choices=["int8", "w4a8"], default="int8")
    args = parser.parse_args()
    if args.task == "text" and args.precision != "w4a8":
        parser.error("textには--precision w4a8を指定してください。")
    for task in ["generate", "depth", "upscale"] if args.task == "all" else [] if args.task == "text" else [args.task]:
        export(args.root, task, args.precision)
    if args.precision == "w4a8" and args.task in ("generate", "text", "all"):
        export_text(args.root)
