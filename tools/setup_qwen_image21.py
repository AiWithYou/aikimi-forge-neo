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
from tempfile import TemporaryDirectory

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


def prepare_rewriter(root):
    """Build a reusable NF4 checkpoint from the pinned official weights."""
    from huggingface_hub import HfApi, snapshot_download

    from modules_forge.qwen_image21 import prompt_rewriter as rewriter
    from modules_forge.qwen_image21.core import atomic_json, inside
    from modules_forge.qwen_image21_environment import validate_running_versions

    validate_running_versions()
    destination = inside(root, root / rewriter.DIRECTORY)
    if destination != root.resolve() / rewriter.DIRECTORY:
        raise RuntimeError("書き換えモデルの保存先が別のフォルダーを指しています。リンク先を確認してください。")
    try:
        rewriter.rewriter_manifest(root, verify_hashes=True)
    except ValueError:
        pass
    else:
        print("プロンプト書き換え: 導入済みNF4モデルのSHA-256を確認しました。")  # noqa: T201
        return
    # Keep verified/resumable source downloads until conversion succeeds. Only
    # this installer's marked staging directory is reclaimed after publication.
    source = inside(root, root / "prompt-rewriter-source")
    if source != root.resolve() / "prompt-rewriter-source":
        raise RuntimeError("一時モデルの保存先が別のフォルダーを指しています。リンク先を確認してください。")
    marker = source / "aikimi-source.json"
    expected = {"model": rewriter.MODEL_ID, "revision": rewriter.MODEL_REVISION}
    if source.exists() and (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != expected):
        raise RuntimeError(f"未登録の作業フォルダーがあります。内容を確認してください: {source}")
    source.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, expected)
    info = HfApi().model_info(rewriter.MODEL_ID, revision=rewriter.MODEL_REVISION, files_metadata=True)
    if info.sha != rewriter.MODEL_REVISION:
        raise RuntimeError("書き換えモデルの固定リビジョンが一致しません。")
    files = [item for item in info.siblings if item.rfilename != ".gitattributes"]
    print("公式の書き換えモデルを取得します。量子化完了後は4bit版だけを保持します。", flush=True)  # noqa: T201
    snapshot_download(
        rewriter.MODEL_ID,
        revision=rewriter.MODEL_REVISION,
        local_dir=source,
        allow_patterns=[item.rfilename for item in files],
        max_workers=4,
    )
    for item in files:
        path = inside(source, source / item.rfilename)
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise RuntimeError(f"書き換えモデルのサイズ不一致: {item.rfilename}")
        expected_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected_hash and sha256(path) != expected_hash:
            raise RuntimeError(f"書き換えモデルのSHA-256不一致: {item.rfilename}")
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("4bit量子化にはBF16対応のCUDA GPUが必要です。")
    print("テキスト部分と出力ヘッドを4bit NF4・二重量子化へ変換中", flush=True)  # noqa: T201
    with TemporaryDirectory(prefix=".prompt-rewriter-", dir=root) as temporary:
        staging = inside(root, Path(temporary))
        build = staging / "model"
        model = AutoModelForCausalLM.from_pretrained(
            str(source),
            dtype=torch.bfloat16,
            quantization_config=rewriter.quantization_config(),
            device_map={"": "cuda:0"},
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        ).eval()
        layers = rewriter.check_quantized_model(model)
        footprint = model.get_memory_footprint()
        model.save_pretrained(build, safe_serialization=True, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False).save_pretrained(
            build
        )
        for name in ("system_prompt.txt", "LICENSE", "README.md"):
            shutil.copy2(source / name, build / name)
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        records = [
            {"path": str(path.relative_to(build)), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(build.rglob("*"))
            if path.is_file()
        ]
        atomic_json(
            build / "rewriter-files.json",
            {
                "schema": 1,
                "model": rewriter.MODEL_ID,
                "revision": rewriter.MODEL_REVISION,
                "quantization": "nf4-double",
                "linear4bit_layers": layers,
                "model_footprint_bytes": footprint,
                "files": records,
            },
        )
        previous = staging / "previous"
        if destination.exists():
            destination.rename(previous)
        try:
            build.rename(destination)
            rewriter.rewriter_manifest(root, verify_hashes=True)
        except BaseException:
            if destination.exists():
                destination.rename(staging / "failed")
            if previous.exists():
                previous.rename(destination)
            raise
    # Both paths were resolved and checked inside the dedicated runtime. Never
    # remove an existing Hub cache or any source outside our marked directory.
    if inside(root, source) != root.resolve() / "prompt-rewriter-source":
        raise RuntimeError("一時モデルの保存先が変わったため削除を中止しました。")
    shutil.rmtree(source)
    print(f"書き換えNF4モデルの準備完了: {footprint / 2**30:.2f} GiB / {layers}層", flush=True)  # noqa: T201


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Image 2.1 / INT8専用環境と公式モデルを準備")
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--runtime-only", action="store_true", help="依存環境のみ導入。モデルは取得しません")
    parser.add_argument("--verify", action="store_true", help="既存ファイルのサイズとSHA-256を再検証")
    parser.add_argument("--dry-run", action="store_true", help="導入内容のみ表示")
    parser.add_argument("--download-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prompt-rewriter-only", action="store_true", help="任意のプロンプト書き換えモデルだけを4bitで追加"
    )
    parser.add_argument(
        "--with-prompt-rewriter", action="store_true", help="画像モデルに加えて4bit書き換えモデルも導入"
    )
    parser.add_argument("--prepare-rewriter", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runtime_only and (args.prompt_rewriter_only or args.with_prompt_rewriter):
        parser.error("--runtime-only と書き換えモデルの導入は同時に指定できません。")
    root = args.root.expanduser().absolute()
    if args.dry_run:
        from modules_forge.qwen_image21 import prompt_rewriter as rewriter

        include_rewriter = args.prompt_rewriter_only or args.with_prompt_rewriter
        print(  # noqa: T201 -- Explicit installer dry-run output.
            json.dumps(
                {
                    "runtime": str(root),
                    "model": rewriter.MODEL_ID if args.prompt_rewriter_only else MODEL_ID,
                    "model_revision": rewriter.MODEL_REVISION if args.prompt_rewriter_only else MODEL_REVISION,
                    "diffusers_revision": DIFFUSERS_REVISION,
                    "model_bytes": None if args.prompt_rewriter_only else 33_131_616_240,
                    "precision": ["nf4-double"] if args.prompt_rewriter_only else ["int8", "bf16"],
                    "license": f"https://huggingface.co/{rewriter.MODEL_ID}/blob/{rewriter.MODEL_REVISION}/LICENSE"
                    if args.prompt_rewriter_only
                    else TERMS,
                    "prompt_rewriter": {
                        "model": rewriter.MODEL_ID,
                        "revision": rewriter.MODEL_REVISION,
                        "precision": "4bit NF4 + double quantization",
                        "source_download_gb_approx": 19,
                        "text_only": True,
                        "quantize_output_head": True,
                    }
                    if include_rewriter
                    else None,
                },
                indent=2,
            )
        )
        return 0
    if args.download_only:
        fetch_and_verify(root)
        return 0
    if args.prepare_rewriter:
        prepare_rewriter(root)
        return 0
    from modules_forge.qwen_image21.core import atomic_json, read_json, runtime_lock, runtime_manifest

    root.mkdir(parents=True, exist_ok=True)
    lock = runtime_lock(root)
    try:
        if args.prompt_rewriter_only:
            from modules_forge.qwen_image21.prompt_rewriter import rewriter_manifest
            from modules_forge.qwen_image21_environment import environment_status

            if args.verify:
                rewriter_manifest(root, verify_hashes=True)
                print("書き換え4bitモデルのSHA-256を確認しました。")  # noqa: T201
                return 0
            ready, _ = environment_status(root)
            python = root / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not ready:
                python = install_environment(root)
            execute([python, "-X", "utf8", Path(__file__).resolve(), "--prepare-rewriter", "--root", root])
            return 0
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
            if args.with_prompt_rewriter:
                from modules_forge.qwen_image21.prompt_rewriter import rewriter_manifest

                rewriter_manifest(root, verify_hashes=True)
                print("書き換え4bitモデルのSHA-256を確認しました。")  # noqa: T201
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
        if args.with_prompt_rewriter:
            execute([python, "-X", "utf8", Path(__file__).resolve(), "--prepare-rewriter", "--root", root])
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
