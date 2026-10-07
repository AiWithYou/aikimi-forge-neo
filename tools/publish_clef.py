"""Publish complete quantized bundles and pin only verified Hub commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.clef.bundle import bundle_manifest  # noqa: E402
from modules_forge.clef.core import RUNTIME, atomic_json, sha256  # noqa: E402

REGISTRY = ROOT / "tools/clef-releases.json"
PROFILES = {"clef-flash": "flash-int8", "clef": "clef-24gb"}


def git_blob_id(path):
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_remote(directory, info):
    directory = Path(directory)
    manifest = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    expected = {x["path"]: x for x in manifest["files"]}
    marker = directory / "complete.json"
    expected[marker.name] = {"size": marker.stat().st_size, "sha256": sha256(marker)}
    actual = {x.rfilename: x for x in info.siblings if x.rfilename != ".gitattributes"}
    if expected.keys() != actual.keys():
        raise RuntimeError(
            f"Hub配布の構成が不足・不一致です: missing={expected.keys() - actual.keys()}, extra={actual.keys() - expected.keys()}"
        )
    for name, record in expected.items():
        item = actual[name]
        if item.size != record["size"]:
            raise RuntimeError(f"Hubのサイズ不一致: {name}")
        if item.lfs:
            if item.lfs.sha256 != record["sha256"]:
                raise RuntimeError(f"HubのSHA-256不一致: {name}")
        elif item.blob_id != git_blob_id(directory / name):
            raise RuntimeError(f"HubのGit blob ID不一致: {name}")
    return info.sha


def publish(root, model):
    # Set this before importing Hub constants, which read the environment once.
    os.environ.setdefault("HF_XET_CACHE", str(Path(root) / "hub-xet"))
    from huggingface_hub import HfApi

    releases = json.loads(REGISTRY.read_text(encoding="utf-8"))
    settings = releases[model]
    directory, manifest = bundle_manifest(root, PROFILES[model])
    files = [x["path"] for x in manifest["files"]] + ["complete.json"]
    api = HfApi()
    print(f"公開アップロード: {settings['repo']} ({len(files)} files)", flush=True)  # noqa: T201
    api.create_repo(settings["repo"], repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(
        settings["repo"],
        directory,
        repo_type="model",
        private=False,
        allow_patterns=files,
        num_workers=2,
        print_report_every=30,
    )
    info = api.model_info(settings["repo"], files_metadata=True)
    revision = verify_remote(directory, info)
    if info.private:
        raise RuntimeError("配布が公開設定になっていません。")
    releases[model]["revision"] = revision
    atomic_json(REGISTRY, releases)
    print(f"公開・全ファイル照合済み: https://huggingface.co/{settings['repo']}/tree/{revision}", flush=True)  # noqa: T201
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--model", choices=["all", *PROFILES], default="all")
    args = parser.parse_args()
    for model in PROFILES if args.model == "all" else [args.model]:
        publish(args.root, model)


if __name__ == "__main__":
    main()
