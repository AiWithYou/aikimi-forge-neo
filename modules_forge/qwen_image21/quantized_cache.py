"""Persistent, versioned quantized components; incomplete writes are never loaded."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

FORMAT_VERSION = 1


class InvalidQuantizedCheckpoint(ValueError, RuntimeError):
    """A structurally invalid checkpoint, distinct from resource/device failures."""


def file_hash(path, check_cancel=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            check_cancel()
            digest.update(block)
    return digest.hexdigest()


def component_identity(model_path, component, precision, *, skip_modules=()):
    """Bind artifacts to the source revision, actual files, recipe and runtime."""
    model_path = Path(model_path).resolve()
    folder = model_path / component
    inventory = model_path.parent / "model-files.json"
    record = json.loads(inventory.read_text(encoding="utf-8")) if inventory.is_file() else {}
    versions = {}
    for name in ("torch", "diffusers", "transformers", "bitsandbytes", "accelerate", "safetensors"):
        versions[name] = importlib.metadata.version(name)
    if precision == "w4a8":
        versions["comfy-kitchen"] = importlib.metadata.version("comfy-kitchen")
    return {
        "schema": FORMAT_VERSION,
        "revision": record.get("revision"),
        "component": component,
        "precision": precision,
        "recipe": {"version": 1, "dtype": "bfloat16", "skip_modules": list(skip_modules)},
        "versions": versions,
        "source_inventory": [item for item in record.get("files", []) if Path(item["path"]).parts[0] == component],
        "source_files": [
            [
                str(path.relative_to(folder)),
                path.stat().st_size,
                path.stat().st_mtime_ns,
                file_hash(path) if path.suffix == ".json" else None,
            ]
            for path in sorted(folder.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(folder).parts
        ],
    }


def cache_path(model_path, identity):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return Path(model_path).resolve().parent / "quantized" / identity["precision"] / key / identity["component"]


def manifest(path, identity, *, verify_hashes=False, check_cancel=lambda: None):
    path = Path(path).resolve()
    record = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("identity") != identity
        or not isinstance(record.get("files"), list)
        or not record["files"]
    ):
        raise ValueError("Quantized checkpoint identity mismatch")
    seen = set()
    for entry in record["files"]:
        check_cancel()
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Invalid quantized checkpoint file record")
        relative = Path(entry["path"])
        target = (path / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(path) or relative in seen:
            raise ValueError("Invalid quantized checkpoint file path")
        seen.add(relative)
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise ValueError(f"Incomplete quantized checkpoint: {relative}")
        # Unchanged artifacts avoid a second full disk read at every startup.
        # A modified file is verified against the original digest before use.
        if verify_hashes or target.stat().st_mtime_ns != entry["mtime_ns"]:
            if file_hash(target, check_cancel) != entry["sha256"]:
                raise ValueError(f"Corrupt quantized checkpoint: {relative}")
    return record


def quarantine(path):
    """Preserve an invalid artifact for inspection instead of overwriting it."""
    path = Path(path)
    if path.exists():
        path.rename(path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex[:8]}"))


@contextmanager
def staging(path, identity, *, check_cancel=lambda: None):
    """Publish only after serialization and verification finish on this volume.

    The Qwen runtime lock serializes writers. Interrupted staging directories are
    retained and ignored; a future job always starts from an immutable source.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.building-{uuid.uuid4().hex[:8]}")
    temporary.mkdir()
    yield temporary
    check_cancel()
    files = []
    for target in sorted(temporary.rglob("*")):
        if target.is_file():
            check_cancel()
            files.append(
                {
                    "path": str(target.relative_to(temporary)),
                    "size": target.stat().st_size,
                    "mtime_ns": target.stat().st_mtime_ns,
                    "sha256": file_hash(target, check_cancel),
                }
            )
    if not files:
        raise ValueError("Cannot publish an empty quantized checkpoint")
    record = {"identity": identity, "files": files}
    with (temporary / "complete.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    check_cancel()
    manifest(temporary, identity, check_cancel=check_cancel)
    # No deletion or replacement of a previously valid checkpoint.
    temporary.rename(path)


def load_or_create(model_path, identity, load, create, save, *, check_cancel=lambda: None, progress=lambda _: None):
    from safetensors import SafetensorError

    path = cache_path(model_path, identity)
    if path.exists():
        try:
            manifest(path, identity, check_cancel=check_cancel)
            progress(f"保存済み {identity['component']} {identity['precision'].upper()} を読み込み中")
            result = load(path)
        except (FileNotFoundError, ValueError, KeyError, TypeError, SafetensorError) as exc:
            # A user's cancellation must never quarantine a healthy checkpoint.
            # Resource/device/permission errors must also propagate unchanged:
            # converting again cannot repair them and would discard a valid hit.
            check_cancel()
            quarantine(path)
            progress(f"保存済みモデルを再作成します（{type(exc).__name__}）")
        else:
            return result, {"status": "hit", "path": str(path)}
    check_cancel()
    result = create()
    check_cancel()
    progress(f"{identity['component']} {identity['precision'].upper()} を次回用に保存中")
    with staging(path, identity, check_cancel=check_cancel) as temporary:
        save(result, temporary)
    return result, {"status": "created", "path": str(path)}
