"""Install the pinned Qwen Image 2.1 runtime without changing Forge dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
RUNTIME = ROOT / "models" / "Qwen-Image-2.1"
REQUIREMENTS = ROOT / "tools" / "requirements-qwen-image21.txt"
TERMS = f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE"


def execute(arguments, **kwargs):
    print("+", subprocess.list2cmdline([str(item) for item in arguments]), flush=True)  # noqa: T201
    return subprocess.run([str(item) for item in arguments], check=True, **kwargs)  # noqa: S603


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_environment(root):
    environment = root / "worker-env"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    uv = shutil.which("uv")
    prefix = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
    execute([*prefix, "torch==2.11.0", "torchvision==0.26.0", "--index-url", "https://download.pytorch.org/whl/cu130"])
    execute([*prefix, "-r", REQUIREMENTS])
    execute([python, "-m", "pip", "check"])
    # Import the actual new classes. A version string alone cannot detect an older checkout.
    execute(
        [
            python,
            "-c",
            "from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21; from transformers import Qwen3VLForConditionalGeneration; import bitsandbytes; print('Qwen Image 2.1 / INT8 imports OK')",
        ]
    )
    frozen = execute([python, "-m", "pip", "freeze"], capture_output=True, text=True, encoding="utf-8").stdout
    (root / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
    return python.absolute()


def download_model(root, python):
    # Download through the dedicated interpreter: the main environment is untouched.
    execute([python, Path(__file__).resolve(), "--download-only", "--root", root])


def fetch_and_verify(root):
    from huggingface_hub import HfApi, snapshot_download

    destination = root / "model"
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError("モデルの固定リビジョンを確認できません。")
    files = [item for item in info.siblings if item.rfilename != ".gitattributes"]
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[item.rfilename for item in files],
        max_workers=4,
    )
    records = []
    for item in files:
        path = destination / item.rfilename
        if not path.resolve().is_relative_to(destination.resolve()) or not path.is_file():
            raise RuntimeError(f"モデルファイルが不足しています: {item.rfilename}")
        if item.size is not None and path.stat().st_size != item.size:
            raise RuntimeError(f"モデルサイズが一致しません: {item.rfilename}")
        actual_hash = sha256(path)
        expected_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"モデルのSHA-256が一致しません: {item.rfilename}")
        records.append({"path": item.rfilename, "size": path.stat().st_size, "sha256": actual_hash})
    from modules_forge.qwen_image21.core import atomic_json

    atomic_json(root / "model-files.json", {"revision": MODEL_REVISION, "files": records})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Image 2.1 / INT8専用環境と公式モデルを準備")
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--runtime-only", action="store_true", help="依存環境のみ導入。モデルは取得しません")
    parser.add_argument("--verify", action="store_true", help="既存ファイルのサイズとSHA-256を再検証")
    parser.add_argument("--dry-run", action="store_true", help="導入内容のみ表示")
    parser.add_argument("--download-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.expanduser().absolute()
    if args.dry_run:
        print(  # noqa: T201 -- Explicit installer dry-run output.
            json.dumps(
                {
                    "runtime": str(root),
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "diffusers_revision": DIFFUSERS_REVISION,
                    "model_bytes": 33_131_616_240,
                    "precision": ["int8", "bf16"],
                    "license": TERMS,
                },
                indent=2,
            )
        )
        return 0
    if args.download_only:
        fetch_and_verify(root)
        return 0
    from modules_forge.qwen_image21.core import atomic_json, read_json, runtime_lock, runtime_manifest

    root.mkdir(parents=True, exist_ok=True)
    lock = runtime_lock(root)
    try:
        if args.verify:
            runtime_manifest(root)
            record = read_json(root / "model-files.json")
            if record.get("revision") != MODEL_REVISION or not record.get("files"):
                raise RuntimeError("固定モデルの検証記録がありません。再セットアップしてください。")
            for item in record["files"]:
                path = root / "model" / item["path"]
                if not path.resolve().is_relative_to((root / "model").resolve()):
                    raise ValueError("検証記録に不正なパスがあります。")
                if not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
                    raise RuntimeError(f"モデル検証失敗: {item['path']}")
            print("Qwen Image 2.1: 全モデルファイルのSHA-256が一致しました。")  # noqa: T201
            return 0
        print(f"Qwen Image 2.1の利用条件: {TERMS}")  # noqa: T201
        python = install_environment(root)
        if not args.runtime_only:
            download_model(root, python)
            atomic_json(
                root / "runtime.json",
                {
                    "schema": 1,
                    "python": str(python),
                    "model": str(root / "model"),
                    "model_revision": MODEL_REVISION,
                    "diffusers_revision": DIFFUSERS_REVISION,
                },
            )
            print("準備完了。Neoを起動してQwen Image 2.1を開いてください。")  # noqa: T201
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
