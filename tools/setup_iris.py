"""Prepare Iris inference and download one selected, revision-pinned task."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.iris.core import (  # noqa: E402
    CODE_REVISION,
    INT8_REPO,
    INT8_REVISION,
    PACKAGING_REVISION,
    RUNTIME,
    SOURCE_REPO,
    SOURCE_REVISION,
    TEXT_REPO,
    TEXT_REVISION,
    atomic_json,
    environment_ready,
    model_directory,
    python_path,
    sha256,
)


def download_plan(precision, task):
    folder = {"generate": "", "depth": "depth/", "upscale": "upscaler/"}[task]
    files = [folder + "config.yaml", folder + "model.safetensors"]
    if task != "generate":
        files.append(folder + "empty_prompt.safetensors")
    if precision == "int8":
        files.append(folder + "manifest.json")
    return {
        "repo": INT8_REPO if precision == "int8" else SOURCE_REPO,
        "revision": INT8_REVISION if precision == "int8" else SOURCE_REVISION,
        "files": files,
        "text_encoder": task == "generate",
    }


def execute(arguments, **kwargs):
    print("+", subprocess.list2cmdline([str(x) for x in arguments]), flush=True)  # noqa: T201
    return subprocess.run([str(x) for x in arguments], check=True, cwd=ROOT, **kwargs)  # noqa: S603


def inference_project(project):
    runtime, separator, optional = project.partition("[project.optional-dependencies]\n")
    if not separator:
        raise ValueError("Irisのパッケージ定義にoptional-dependenciesがありません。")
    lines = runtime.splitlines(keepends=True)
    training = [line for line in lines if line.strip().startswith(('"wandb>=', '"dion @'))]
    if not training:
        return project
    extra = "training = [\n" + "".join(training) + "]\n"
    return "".join(line for line in lines if line not in training) + separator + extra + optional


def install_source(root, prefix):
    source = Path(root) / ("iris-source-" + CODE_REVISION[:12])
    if not source.is_dir():
        execute(["git", "clone", "--filter=blob:none", "https://github.com/speridlabs/iris-3b.git", source])
        execute(["git", "-C", source, "checkout", "--detach", CODE_REVISION])
    revision = execute(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if revision != CODE_REVISION:
        raise ValueError("Iris Pythonソースのcommitが一致しません。")
    project = source / "pyproject.toml"
    project.write_text(inference_project(project.read_text(encoding="utf-8")), encoding="utf-8")
    execute([*prefix, "--no-deps", source])


def install_environment(root):
    python = python_path(root)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(python.parent.parent)
    uv = shutil.which("uv")
    prefix = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
    execute([*prefix, "torch==2.13.0", "torchvision==0.28.0", "--index-url", "https://download.pytorch.org/whl/cu130"])
    execute([*prefix, "-r", ROOT / "tools/requirements-iris.txt"])
    install_source(root, prefix)
    execute(
        [
            python,
            "-c",
            "from iris3b.models.dit import IrisDiT; from iris3b.downstream.depth import DepthPredictor; from transformers import Qwen3VLForConditionalGeneration; import torch; assert torch.cuda.is_available(); print('Iris CUDA imports OK')",
        ]
    )
    frozen = execute([python, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    execute([python, "-m", "pip", "check"])
    (Path(root) / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
    atomic_json(Path(root) / "runtime.json", {"code_revision": CODE_REVISION, "packaging": PACKAGING_REVISION})


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = directory / record["path"]
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"モデル配布のファイルがありません: {record['path']}")
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"モデル配布のサイズまたはSHA-256不一致: {record['path']}")
    return manifest


def fetch_model(root, precision, task):
    from huggingface_hub import HfApi, snapshot_download

    plan = download_plan(precision, task)
    if len(plan["revision"]) != 40:
        raise ValueError("INT8配布の検証済み公開commitがまだ設定されていません。")
    destination = Path(root) / ("int8" if precision == "int8" else "official")
    info = HfApi().model_info(plan["repo"], revision=plan["revision"], files_metadata=True)
    snapshot_download(
        plan["repo"], revision=plan["revision"], local_dir=destination, allow_patterns=plan["files"], max_workers=3
    )
    records = []
    for name in plan["files"]:
        remote = next(x for x in info.siblings if x.rfilename == name)
        path = destination / name
        digest = sha256(path)
        if path.stat().st_size != remote.size or (remote.lfs and digest != remote.lfs.sha256):
            raise ValueError(f"取得した重みの検証に失敗しました: {name}")
        if path.name != "manifest.json":
            records.append({"path": path.name, "size": remote.size, "sha256": digest})
        print(f"検証済み: {name}", flush=True)  # noqa: T201
    directory = model_directory(root, precision, task)
    if precision == "normal":
        atomic_json(
            directory / "manifest.json",
            {
                "format": "official-fp32",
                "task": task,
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "files": records,
            },
        )
    else:
        verify_manifest(directory)
    if plan["text_encoder"]:
        patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
        info = HfApi().model_info(TEXT_REPO, revision=TEXT_REVISION, files_metadata=True)
        encoder = Path(root) / "text-encoder"
        snapshot_download(
            TEXT_REPO,
            revision=TEXT_REVISION,
            local_dir=encoder,
            allow_patterns=patterns,
            max_workers=3,
        )
        records = []
        for remote in info.siblings:
            if not any(fnmatch.fnmatch(remote.rfilename, pattern) for pattern in patterns):
                continue
            path = encoder / remote.rfilename
            digest = sha256(path)
            if path.stat().st_size != remote.size or (remote.lfs and digest != remote.lfs.sha256):
                raise ValueError(f"テキストエンコーダーの検証に失敗しました: {remote.rfilename}")
            records.append({"path": remote.rfilename, "size": remote.size, "sha256": digest})
        atomic_json(encoder / "download.json", {"repo": TEXT_REPO, "revision": TEXT_REVISION, "files": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--precision", choices=["normal", "int8"], default="int8")
    parser.add_argument("--task", choices=["generate", "depth", "upscale", "all"], default="generate")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parent-guard", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.parent_guard:
        from modules_forge.yue2_studio.worker import parent_guard

        parent_guard()
    tasks = ["generate", "depth", "upscale"] if args.task == "all" else [args.task]
    if args.dry_run:
        print(  # noqa: T201
            json.dumps(
                {
                    "root": str(args.root),
                    "code_revision": CODE_REVISION,  # noqa: T201
                    "plans": [download_plan(args.precision, t) for t in tasks],
                },
                indent=2,
            )
        )  # noqa: T201
        return
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.download_only and not environment_ready(args.root):
        install_environment(args.root)
    if not args.runtime_only:
        if args.download_only:
            for task in tasks:
                fetch_model(args.root, args.precision, task)
        else:
            execute(
                [
                    python_path(args.root),
                    Path(__file__).resolve(),
                    "--download-only",
                    "--root",
                    args.root,
                    "--precision",
                    args.precision,
                    "--task",
                    args.task,
                ]
            )
    print("Irisの準備が完了しました。", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
