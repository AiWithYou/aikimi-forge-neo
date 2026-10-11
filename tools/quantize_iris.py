"""Export reloadable INT8 or packed W4A8 weights; validate before publication."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge import local_assets  # noqa: E402
from modules_forge.iris.core import (  # noqa: E402
    CODE_REVISION,
    RUNTIME,
    SOURCE_REPO,
    SOURCE_REVISION,
    atomic_json,
    model_directory,
    model_files,
    sha256,
)
from modules_forge.iris.quantization import convert_linears  # noqa: E402
from modules_forge.iris.runtime import load_model  # noqa: E402


def _local_output(output_dir):
    value = str(output_dir).strip()
    if not value or "://" in value:
        raise ValueError("このPCの出力フォルダーを--output-dirで指定してください。")
    destination = Path(value.strip('"')).expanduser()
    if not destination.is_absolute():
        destination = local_assets.ROOT / destination
    destination = destination.resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError("出力先にはフォルダーを指定してください。")
    return destination


def _protect_output(destination, weights, config):
    if destination == weights.parent or destination == config.parent:
        raise ValueError("変換元と別の出力フォルダーを指定してください。元ファイルは上書きできません。")
    if any((destination / name).exists() for name in ("model.safetensors", "config.yaml", "manifest.json")):
        raise ValueError("出力モデル・設定が既にあります。別の出力フォルダーを指定してください。上書きはできません。")


def export(root, task, precision="int8", checkpoint="base", output_dir=None):
    if precision not in ("int8", "w4a8"):
        raise ValueError("変換先の保存形式にはint8またはw4a8を指定してください。")
    custom = checkpoint != "base"
    if custom and task != "generate":
        raise ValueError("ローカルのチェックポイントには画像生成（generate）を指定してください。")
    if custom and output_dir is None:
        raise ValueError("ローカル重みの変換先を--output-dirで指定してください。")
    if not custom and output_dir is not None:
        raise ValueError("--output-dirはローカルのチェックポイントの出力に使用してください。")
    weights, config = (path.resolve() for path in model_files(root, "normal", task, checkpoint))
    destination = _local_output(output_dir) if custom else model_directory(root, precision, task)
    if custom:
        _protect_output(destination, weights, config)
    source_identity = {"weights": local_assets.file_identity(weights), "config": local_assets.file_identity(config)}
    model, _, _ = load_model(root, "normal", task, str(weights) if custom else "base")
    if precision == "w4a8":
        from modules_forge.iris.w4a8 import convert_linears as convert_w4a8

        report = convert_w4a8(model)
    else:
        report = convert_linears(model)
    if precision == "w4a8":
        reference = weights if custom else model_directory(root, "int8", task) / "model.safetensors"
        if report["bytes"] >= reference.stat().st_size:
            raise ValueError("W4A8の重みが比較元より小さくなっていません。対象層を確認してください。")
    if source_identity != {
        "weights": local_assets.file_identity(weights),
        "config": local_assets.file_identity(config),
    }:
        raise ValueError("変換中に変換元の重み・設定が変更されました。再実行してください。")
    if custom:
        _protect_output(destination, weights, config)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_sha256": source_identity["weights"]["sha256"],
        "source_config_sha256": source_identity["config"]["sha256"],
    }
    if custom:
        provenance.update(source_path=str(weights), source_config_path=str(config))
    else:
        provenance.update(source_repo=SOURCE_REPO, source_revision=SOURCE_REVISION)
    target = destination / "model.safetensors"
    save_file(
        model.state_dict(),
        target,
        metadata={
            "format": report["format"],
            **provenance,
            "code_revision": CODE_REVISION,
            "checkpoint": checkpoint,
        },
    )
    files = [target]
    shutil.copyfile(config, destination / "config.yaml")
    files.append(destination / "config.yaml")
    empty_prompt = weights.with_name("empty_prompt.safetensors")
    if not custom and empty_prompt.is_file():
        shutil.copyfile(empty_prompt, destination / empty_prompt.name)
        files.append(destination / empty_prompt.name)
    manifest = {
        **report,
        "task": task,
        "checkpoint": checkpoint,
        **provenance,
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


def main():
    import torch

    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--task", choices=["generate", "depth", "upscale", "text", "all"], default="all")
    parser.add_argument("--precision", choices=["int8", "w4a8"], default="int8")
    parser.add_argument(
        "--checkpoint", default="base", help="画像生成用のローカルsafetensorsファイルまたは単一重みフォルダー"
    )
    parser.add_argument("--output-dir", type=Path, help="ローカル重みの変換先フォルダー（既存モデルは上書きしません）")
    args = parser.parse_args()
    if args.task == "text" and args.precision != "w4a8":
        parser.error("textには--precision w4a8を指定してください。")
    if args.checkpoint != "base" and args.task != "generate":
        parser.error("ローカルのチェックポイントには--task generateを指定してください。")
    if args.checkpoint != "base" and args.output_dir is None:
        parser.error("ローカル重みの変換先を--output-dirで指定してください。")
    if args.checkpoint == "base" and args.output_dir is not None:
        parser.error("--output-dirはローカルのチェックポイントの出力に使用してください。")
    try:
        for task in (
            ["generate", "depth", "upscale"] if args.task == "all" else [] if args.task == "text" else [args.task]
        ):
            export(args.root, task, args.precision, args.checkpoint, output_dir=args.output_dir)
        if args.checkpoint == "base" and args.precision == "w4a8" and args.task in ("generate", "text", "all"):
            export_text(args.root)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
