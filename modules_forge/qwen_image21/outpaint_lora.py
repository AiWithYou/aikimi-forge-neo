"""Pinned optional Outpaint adapters; downloading is an explicit setup action."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from .core import atomic_json, runtime_lock
from .outpaint import WEIGHTS

REPOSITORY = "ausboss/Qwen-Image-2.1-Outpaint-LoRA"
REVISION = "449336db42ff074aee970ba0facc0ac0feb77863"
SIZE = 159_436_576
HASHES = {
    "v1": "201ab351b54c3e5d208c91a129011e7e490efdd13e40d89befc4c1f0ca7eb177",
    "v2": "3611a5ccae5f790405eeacf1dfb2c89285c1e4f2987b9abc88cd0161a60eca28",
}


def adapter_path(runtime: Path, version: str) -> Path:
    if not isinstance(version, str) or version not in WEIGHTS:
        raise ValueError("Outpaint LoRAはv1またはv2を選んでください。")
    return Path(runtime) / "outpaint" / WEIGHTS[version]


def installed(runtime: Path, version: str, *, verify: bool = False) -> dict:
    path = adapter_path(runtime, version)
    receipt = path.with_suffix(".json")
    expected = {"repository": REPOSITORY, "revision": REVISION, "sha256": HASHES[version], "version": version}
    if (
        any(p.is_symlink() for p in (path.parent, path, receipt))
        or not path.is_file()
        or path.stat().st_size != SIZE
        or not receipt.is_file()
        or json.loads(receipt.read_text(encoding="utf-8")) != expected
    ):
        raise ValueError(f"Outpaint {version}が未導入です。「Outpaintを準備」を押してください（約160 MB）。")
    if verify:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != HASHES[version]:
            raise ValueError("Outpaint LoRAのSHA-256が一致しません。「Outpaintを準備」で再取得してください。")
    return {**expected, "path": str(path.resolve())}


def install(runtime: Path, version: str, progress=None) -> dict:
    path = adapter_path(runtime, version)
    if any(p.is_symlink() for p in (path.parent, path, path.with_suffix(".json"))):
        raise ValueError("Outpaintの保存先にシンボリックリンクは使用できません。")
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
                # Scheme, host, revision and filenames are fixed above.
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > SIZE:
                            raise ValueError("Outpaint LoRAのサイズが配布版と一致しません。")
                        digest.update(block)
                        output.write(block)
                        if progress:
                            progress(total / SIZE)
            if total != SIZE or digest.hexdigest() != HASHES[version]:
                raise ValueError("Outpaint LoRAのサイズまたはSHA-256が配布版と一致しません。")
            os.replace(temporary, path)
            atomic_json(
                path.with_suffix(".json"),
                {
                    "repository": REPOSITORY,
                    "revision": REVISION,
                    "sha256": HASHES[version],
                    "version": version,
                },
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return installed(runtime, version)
