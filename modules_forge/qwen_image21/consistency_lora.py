"""Pinned Consistency adapters for the ordinary optional LoRA library."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from .core import atomic_json, runtime_lock

REPOSITORY = "ausboss/Qwen-Image-2.1-Consistency-LoRA"
REVISION = "8f05b0fa027d517fa396fb31b71e0eaf48110e89"
SIZE = 159_436_496
WEIGHTS = {
    "1500": "qwen-image-2.1-consistency.safetensors",
    "2000": "qwen-image-2.1-consistency-2000.safetensors",
}
HASHES = {
    "1500": "4f44ada1be2189cc23b3d010f9603543403f48454e2f76842a3d30109b20bd63",
    "2000": "a0bf043edc4695b1661a0e768e49a7fdff9a656b7d4313203d590e0262b64beb",
}
SOURCE = f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md"


def adapter_path(runtime: Path, version: str = "1500") -> Path:
    if not isinstance(version, str) or version not in WEIGHTS:
        raise ValueError("Consistency LoRAは1500または2000を選んでください。")
    return Path(runtime) / "loras" / REPOSITORY.replace("/", "--") / WEIGHTS[version]


def _receipt(version: str) -> dict:
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "file": WEIGHTS[version],
        "sha256": HASHES[version],
        "size": SIZE,
        "version": version,
    }


def _check_links(path: Path) -> None:
    if any(p.is_symlink() for p in (path.parent.parent, path.parent, path, path.with_suffix(".source.json"))):
        raise ValueError("Consistency LoRAの保存先にシンボリックリンクは使用できません。")


def installed(runtime: Path, version: str = "1500", *, verify: bool = False) -> dict:
    path = adapter_path(runtime, version)
    _check_links(path)
    receipt = path.with_suffix(".source.json")
    expected = _receipt(version)
    if (
        not path.is_file()
        or path.stat().st_size != SIZE
        or not receipt.is_file()
        or json.loads(receipt.read_text(encoding="utf-8")) != expected
    ):
        raise ValueError(f"Consistency {version}が未導入です。「Consistency LoRAを準備」を押してください。")
    if verify:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != HASHES[version]:
            raise ValueError("Consistency LoRAのSHA-256が一致しません。再取得してください。")
    return {**expected, "name": path.relative_to(Path(runtime) / "loras").as_posix(), "path": str(path.resolve())}


def install(runtime: Path, version: str = "1500", progress=None) -> dict:
    path = adapter_path(runtime, version)
    _check_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_lock(path.parent):
        try:
            return installed(runtime, version, verify=True)
        except (OSError, ValueError):
            pass
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{path.name}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                digest, total = hashlib.sha256(), 0
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > SIZE:
                            raise ValueError("Consistency LoRAのサイズが配布版と一致しません。")
                        digest.update(block)
                        output.write(block)
                        if progress:
                            progress(total / SIZE)
            if total != SIZE or digest.hexdigest() != HASHES[version]:
                raise ValueError("Consistency LoRAのサイズまたはSHA-256が配布版と一致しません。")
            os.replace(temporary, path)
            atomic_json(path.with_suffix(".source.json"), _receipt(version))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return installed(runtime, version)


def identify(path: Path, metadata: dict) -> str | None:
    if metadata.get("ss_output_name") != "qwen21_consistency_v1":
        return None
    from modules_forge.local_assets import file_identity

    digest = file_identity(path)["sha256"]
    return next((version for version, expected in HASHES.items() if digest == expected), None)
