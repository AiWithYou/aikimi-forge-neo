"""Install complete quantized Clef releases in a separate environment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.clef.bundle import bundle_directory, read_bundle  # noqa: E402
from modules_forge.clef.cache import cache_directory  # noqa: E402
from modules_forge.clef.core import MODELS, RUNTIME, atomic_json, sha256, source_manifest  # noqa: E402

RELEASES = json.loads((ROOT / "tools/clef-releases.json").read_text(encoding="utf-8"))
QUANTIZED_PROFILES = {"clef-flash": "flash-int8", "clef": "clef-24gb"}


def execute(args, **kwargs):
    print("+", subprocess.list2cmdline([str(x) for x in args]), flush=True)  # noqa: T201
    kwargs.setdefault("cwd", ROOT)
    return subprocess.run([str(x) for x in args], check=True, **kwargs)  # noqa: S603 -- Fixed runtime, requirements and model identifiers; shell=False.


def install_environment(root):
    directory = root / "worker-env"
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(directory)
    uv = shutil.which("uv")
    prefix = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
    execute([*prefix, "pip==26.2.1", "setuptools==83.0.0"])
    execute([*prefix, "torch==2.13.0", "torchvision==0.28.0", "--index-url", "https://download.pytorch.org/whl/cu130"])
    execute([*prefix, "-r", ROOT / "tools/requirements-clef.txt"])
    execute([python, "-m", "pip", "check"])
    execute(
        [
            python,
            "-c",
            'from transformers import Qwen3_5Model, AutoProcessor; import torch,bitsandbytes,accelerate; assert torch.cuda.is_available(); print("Clef CUDA imports OK")',
        ]
    )
    frozen = execute([python, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (root / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
    return python


def fetch_source(root, model, verify=False):
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    if verify:
        source_manifest(root, model, verify_hashes=True)
        print(f"{model}: SHA-256検証済み", flush=True)  # noqa: T201
        return
    settings = MODELS[model]
    directory = root / "source" / model
    info = HfApi().model_info(settings["repo"], revision=settings["revision"], files_metadata=True)
    if info.sha != settings["revision"]:
        raise RuntimeError("固定リビジョンが一致しません。")
    files = [item for item in info.siblings if item.rfilename != ".gitattributes"]
    snapshot_download(
        settings["repo"],
        revision=settings["revision"],
        local_dir=directory,
        allow_patterns=[item.rfilename for item in files],
        max_workers=4,
    )
    records = []
    for item in files:
        path = directory / item.rfilename
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
            or path.stat().st_size != item.size
        ):
            raise RuntimeError(f"サイズ不一致: {item.rfilename}")
        digest = sha256(path)
        expected = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected and digest != expected:
            raise RuntimeError(f"SHA-256不一致: {item.rfilename}")
        records.append({"path": item.rfilename, "size": item.size, "sha256": digest})
        print(f"検証済み: {item.rfilename}", flush=True)  # noqa: T201
    atomic_json(
        directory / "release.json", {"repo": settings["repo"], "revision": settings["revision"], "files": records}
    )


def fetch_quantized(root, model, verify=False):
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    profile = QUANTIZED_PROFILES[model]
    target = bundle_directory(root, profile)
    if verify:
        read_bundle(target, profile, verify_hashes=True)
        print(f"{model}: 量子化配布のSHA-256検証済み", flush=True)  # noqa: T201
        return target
    settings = RELEASES[model]
    if not settings.get("revision") or len(settings["revision"]) != 40:
        raise RuntimeError("量子化配布の公開commitがまだ設定されていません。")
    info = HfApi().model_info(settings["repo"], revision=settings["revision"], files_metadata=True)
    if info.sha != settings["revision"]:
        raise RuntimeError("量子化配布の固定commitが一致しません。")
    files = [item for item in info.siblings if item.rfilename != ".gitattributes"]
    staging = target.with_name(".download-" + target.name)
    staging.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _, local = read_bundle(target, profile)
        marker = hf_hub_download(settings["repo"], "complete.json", revision=settings["revision"], local_dir=staging)
        remote = json.loads(Path(marker).read_text(encoding="utf-8"))
        if local != remote:
            raise RuntimeError(
                f"保存済み配布と公開commitが一致しません。保存物を別の場所へ移して再実行してください: {target}"
            )
        local_hashes = {x["path"]: x["sha256"] for x in local["files"]}
        for item in files:
            if item.lfs and item.rfilename != "complete.json" and local_hashes.get(item.rfilename) != item.lfs.sha256:
                raise RuntimeError(f"公開配布のSHA-256が一致しません: {item.rfilename}")
    else:
        snapshot_download(
            settings["repo"],
            revision=settings["revision"],
            local_dir=staging,
            allow_patterns=[item.rfilename for item in files],
            max_workers=4,
        )
        _, downloaded = read_bundle(staging, profile, verify_hashes=True)
        digests = {record["path"]: record["sha256"] for record in downloaded["files"]}
        digests["complete.json"] = sha256(staging / "complete.json")
        for item in files:
            path = staging / item.rfilename
            if (
                not path.resolve().is_relative_to(staging.resolve())
                or not path.is_file()
                or path.stat().st_size != item.size
            ):
                raise RuntimeError(f"サイズ不一致: {item.rfilename}")
            if item.lfs and digests.get(item.rfilename) != item.lfs.sha256:
                raise RuntimeError(f"SHA-256不一致: {item.rfilename}")
        staging.rename(target)
    atomic_json(target / "download.json", settings)
    print(f"{model}: {settings['repo']}@{settings['revision']} 導入済み", flush=True)  # noqa: T201
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--model", choices=["all", "clef", "clef-flash"], default="all")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--source", action="store_true", help="比較・再変換用の公式BF16重みを明示して取得する")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-dir", type=Path, help="量子化済みモデルを置くディレクトリ")
    parser.add_argument("--configure-only", action="store_true", help="保存先の設定のみ変更する")
    args = parser.parse_args()
    if args.configure_only and not args.cache_dir:
        parser.error("--configure-onlyには--cache-dirが必要です。")
    models = list(MODELS) if args.model == "all" else [args.model]
    if args.dry_run:
        print(  # noqa: T201
            json.dumps(
                {
                    "runtime": str(args.root),
                    "cache_dir": str(
                        args.cache_dir.resolve() if args.cache_dir else cache_directory(args.root, "flash-int8").parent
                    ),
                    "models": {x: (MODELS if args.source else RELEASES)[x] for x in models},
                    "source": args.source,
                    "torch": "2.13.0+cu130",
                    "requirements": str(ROOT / "tools/requirements-clef.txt"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    args.root.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        destination = args.cache_dir.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(args.root / "storage.json", {"quantized": str(destination)})
        print("量子化キャッシュ保存先:", destination, flush=True)  # noqa: T201
    if args.configure_only:
        return
    if args.download_only or args.verify:
        for model in models:
            (fetch_source if args.source else fetch_quantized)(args.root, model, args.verify)
        return
    python = install_environment(args.root)
    if not args.runtime_only:
        execute(
            [
                python,
                Path(__file__).resolve(),
                "--download-only",
                "--root",
                args.root,
                "--model",
                args.model,
                *(["--source"] if args.source else []),
            ]
        )


if __name__ == "__main__":
    main()
