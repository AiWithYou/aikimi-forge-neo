# SPDX-License-Identifier: AGPL-3.0-only
"""Stage and import Neo's saved Qwen Image 2.1 quantized components.

Staging never changes the local quantized cache. Importing requires the pinned
official model and the same isolated runtime versions used for conversion.

Modified 2026-10-10 by Aikimi for official Turbo provenance, portable
export/import, and per-file modification notices. Importer code: AGPLv3
(see CODE_LICENSE in a release); model weights: Qwen Research (LICENSE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import uuid
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "Qwen-Image-2.1"
BASE_MODEL = "Qwen/Qwen-Image-2.1"
COMPONENTS = ("transformer", "text_encoder")
PROFILES = ("int8", "w4a8", "turbo_official_int8", "turbo_official_w4a8")
TENSOR_BYTES = {
    **dict.fromkeys(("BOOL", "U8", "I8", "F8_E4M3", "F8_E5M2", "F8_E8M0"), 1),
    **dict.fromkeys(("U16", "I16", "F16", "BF16"), 2),
    **dict.fromkeys(("U32", "I32", "F32"), 4),
    **dict.fromkeys(("U64", "I64", "F64"), 8),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root: Path, name: str) -> Path:
    relative = Path(name)
    target = (root / relative).resolve()
    if relative.is_absolute() or PureWindowsPath(name).is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe release path: {name}")
    return target


def _assert_portable(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_portable(key)
            _assert_portable(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str):
        name = value.strip()
        if name.startswith(("/", "\\\\")) or PureWindowsPath(name).drive:
            raise ValueError("Local filesystem path in release metadata")


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _identity(model_root: Path, component: str, profile: str) -> dict:
    from modules_forge.qwen_image21.capabilities import quantization_precision
    from modules_forge.qwen_image21.quantized_cache import INT8_SKIP_MODULES, component_identity
    from modules_forge.qwen_image21.turbo import OFFICIAL_PRECISIONS, transformer_directory, turbo_manifest

    precision = quantization_precision(profile)
    source = {}
    if component == "transformer" and profile in OFFICIAL_PRECISIONS:
        source = {
            "source_path": transformer_directory(model_root, profile),
            "source_record": turbo_manifest(model_root, profile),
        }
    return component_identity(
        model_root / "model",
        component,
        precision,
        skip_modules=INT8_SKIP_MODULES if precision == "int8" and component == "transformer" else (),
        **source,
    )


def _inventory(identity: dict) -> list[dict]:
    return [{key: item[key] for key in ("path", "size", "sha256")} for item in identity["source_inventory"]]


def _source(identity: dict) -> dict:
    record = identity.get("source_record")
    files = (
        {
            name.removeprefix("official/"): {key: item[key] for key in ("size", "sha256")}
            for name, item in record["files"].items()
        }
        if record
        else {item["path"]: {key: item[key] for key in ("size", "sha256")} for item in _inventory(identity)}
    )
    return {"model": identity.get("source_model", BASE_MODEL), "revision": identity["revision"], "files": files}


def _notice(identity: dict) -> str:
    source = _source(identity)
    return (
        f"Built with Qwen. Modified by Aikimi: {identity['precision'].upper()} quantization. "
        f"Source {source['model']} revision {source['revision']}. No additional training."
    )


def _export_safetensors(source: Path, destination: Path, notice: str, expected_hash: str) -> dict:
    """Add a notice while hashing the source, export and unchanged payload once."""
    with source.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        original_header = stream.read(length)
        if len(original_header) != length:
            raise ValueError("Truncated safetensors header")
        header = json.loads(original_header)
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors header")
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
            raise ValueError("Invalid safetensors metadata")
        _assert_portable(metadata)
        payload_bytes = source.stat().st_size - 8 - length
        intervals = []
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(spec, dict):
                raise ValueError("Invalid safetensors descriptor")
            offsets, shape = spec.get("data_offsets"), spec.get("shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or spec.get("dtype") not in TENSOR_BYTES
            ):
                raise ValueError("Invalid safetensors descriptor or offsets")
            elements = 1
            for dimension in shape:
                elements *= dimension
            if offsets[1] - offsets[0] != elements * TENSOR_BYTES[spec["dtype"]]:
                raise ValueError("Safetensors descriptor size mismatch")
            if offsets[1] > offsets[0]:
                intervals.append(tuple(offsets))
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError("Safetensors payload has a hole or overlapping tensors")
            cursor = end
        if cursor != payload_bytes:
            raise ValueError("Safetensors payload size mismatch")
        metadata = dict(metadata)
        existing = metadata.get("aikimi_modification_notice")
        metadata["aikimi_modification_notice"] = (
            notice if not existing or existing == notice else existing + "\n" + notice
        )
        header["__metadata__"] = metadata
        exported_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        exported_header += b" " * (-len(exported_header) % 8)
        exported_prefix = struct.pack("<Q", len(exported_header))
        source_digest, export_digest, payload_digest = (hashlib.sha256() for _ in range(3))
        source_digest.update(prefix + original_header)
        export_digest.update(exported_prefix + exported_header)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with destination.open("xb") as target:
            target.write(exported_prefix)
            target.write(exported_header)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                source_digest.update(block)
                export_digest.update(block)
                payload_digest.update(block)
                target.write(block)
                copied += len(block)
        if copied != payload_bytes or source_digest.hexdigest() != expected_hash:
            raise ValueError(f"Source file missing or changed: {source}")
    return {"sha256": export_digest.hexdigest(), "tensor_payload_sha256": payload_digest.hexdigest()}


def stage(model_root: Path, precision: str, output: Path) -> None:
    from modules_forge.qwen_image21.quantized_cache import cache_path, manifest

    if output.exists():
        raise FileExistsError(f"Release directory already exists: {output}")
    profile = precision
    identities = {component: _identity(model_root, component, profile) for component in COMPONENTS}
    transformer_source, encoder_source = (_source(identities[component]) for component in COMPONENTS)
    precision = identities["transformer"]["precision"]
    release = {
        "schema": 2,
        "base_model": transformer_source["model"],
        "base_revision": transformer_source["revision"],
        "shared_base_model": encoder_source["model"],
        "shared_base_revision": encoder_source["revision"],
        "precision": precision,
        "profile": profile,
        "components": {},
    }
    output.mkdir(parents=True)
    try:
        for component in COMPONENTS:
            identity = identities[component]
            source_dir = cache_path(model_root / "model", identity)
            try:
                record = json.loads((source_dir / "complete.json").read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                raise ValueError(f"Matching saved {profile}/{component} is missing") from error
            if record.get("identity") != identity or not record.get("files"):
                raise ValueError(f"Source identity mismatch: {source_dir}")
            manifest(source_dir, identity)
            files = []
            for entry in record["files"]:
                source = checked_path(source_dir, entry["path"])
                if not source.is_file() or source.stat().st_size != entry["size"]:
                    raise ValueError(f"Source file missing or changed: {source}")
                dest_name = f"{component}/{entry['path']}"
                destination = checked_path(output, dest_name)
                extra = {"quantized_source_sha256": entry["sha256"]}
                if destination.suffix == ".safetensors":
                    extra.update(_export_safetensors(source, destination, _notice(identity), entry["sha256"]))
                    digest = extra.pop("sha256")
                elif destination.suffix == ".json":
                    config_bytes = source.read_bytes()
                    if hashlib.sha256(config_bytes).hexdigest() != entry["sha256"]:
                        raise ValueError(f"Source file missing or changed: {source}")
                    config = json.loads(config_bytes.decode("utf-8"))
                    if destination.name == "config.json" and precision == "int8" and "_name_or_path" in config:
                        config["_name_or_path"] = _source(identity)["model"]
                    notice = _notice(identity)
                    existing = config.get("_aikimi_modification_notice")
                    config["_aikimi_modification_notice"] = (
                        notice if not existing or existing == notice else existing + "\n" + notice
                    )
                    _assert_portable(config)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    exported = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                    destination.write_bytes(exported)
                    digest = hashlib.sha256(exported).hexdigest()
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    digest = sha256(destination)
                    if destination.stat().st_size != entry["size"] or digest != entry["sha256"]:
                        raise ValueError(f"Staged file changed or corrupt: {destination}")
                files.append(
                    {
                        "path": dest_name,
                        "size": destination.stat().st_size,
                        "sha256": digest,
                        **extra,
                    }
                )
            release["components"][component] = {
                "recipe": identity["recipe"],
                "versions": identity["versions"],
                "source_inventory": _inventory(identity),
                "source": _source(identity),
                "files": files,
            }
        source_docs = ROOT / "docs" / "hf-qwen-release" / profile
        for name in ("README.md", "NOTICE.md"):
            shutil.copyfile(source_docs / name, output / name)
        shutil.copyfile(model_root / "model" / "LICENSE", output / "LICENSE")
        shutil.copyfile(ROOT / "LICENSE", output / "CODE_LICENSE")
        shutil.copyfile(Path(__file__), output / "install.py")
        _assert_portable(release)
        (output / "release_manifest.json").write_text(
            json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "release": str(output),
                    "precision": precision,
                    "files": sum(len(c["files"]) for c in release["components"].values()),
                    "bytes": sum(f["size"] for c in release["components"].values() for f in c["files"]),
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        print(f"Incomplete release staging retained for inspection: {output}", file=sys.stderr)
        raise


def install(model_root: Path, release_dir: Path, precision: str) -> None:
    from modules_forge.qwen_image21.quantized_cache import (
        cache_path,
        manifest,
    )

    record = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    profile = precision
    identities = {component: _identity(model_root, component, profile) for component in COMPONENTS}
    transformer_source, encoder_source = (_source(identities[component]) for component in COMPONENTS)
    expected = {
        "schema": 2,
        "profile": profile,
        "precision": identities["transformer"]["precision"],
        "base_model": transformer_source["model"],
        "base_revision": transformer_source["revision"],
        "shared_base_model": encoder_source["model"],
        "shared_base_revision": encoder_source["revision"],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected release format, source revision, or precision")
    if set(record.get("components", {})) != set(COMPONENTS):
        raise ValueError("Release must contain both quantized components")
    for component in COMPONENTS:
        component_record = record["components"][component]
        identity = identities[component]
        for key in ("recipe", "versions", "source_inventory"):
            local = _inventory(identity) if key == "source_inventory" else identity[key]
            if local != component_record.get(key):
                raise ValueError(f"{component}: local {key} does not match this release")
        if _source(identity) != component_record.get("source"):
            raise ValueError(f"{component}: source does not match this release")
    for component in COMPONENTS:
        component_record = record["components"][component]
        identity = identities[component]
        destination = cache_path(model_root / "model", identity)
        if destination.exists():
            manifest(destination, identity, verify_hashes=True)
            print(f"Already installed and verified: {destination}")
            continue
        temporary = destination.with_name(f"{destination.name}.building-import-{uuid.uuid4().hex[:8]}")
        temporary.mkdir(parents=True)
        entries = []
        try:
            for item in component_record["files"]:
                name = item["path"]
                if not name.startswith(f"{component}/"):
                    raise ValueError(f"Wrong component path: {name}")
                source = checked_path(release_dir, name)
                relative = name[len(component) + 1 :]
                target = checked_path(temporary, relative)
                if not source.is_file() or source.stat().st_size != item["size"] or sha256(source) != item["sha256"]:
                    raise ValueError(f"Release file missing or corrupt: {name}")
                link_or_copy(source, target)
                if target.stat().st_size != item["size"]:
                    raise ValueError(f"Copied file has the wrong size: {name}")
                entries.append(
                    {
                        "path": relative,
                        "size": item["size"],
                        "mtime_ns": target.stat().st_mtime_ns,
                        "sha256": item["sha256"],
                    }
                )
            (temporary / "complete.json").write_text(
                json.dumps({"identity": identity, "files": entries}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest(temporary, identity)
            if destination.exists():
                raise FileExistsError(f"Target was created during import: {destination}")
            temporary.rename(destination)
            print(f"Installed: {destination}")
        except Exception:
            print(f"Incomplete import retained for inspection: {temporary}", file=sys.stderr)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "install"))
    parser.add_argument("--precision", choices=PROFILES, required=True)
    parser.add_argument("--neo-root", type=Path, help="Forge Neo checkout root (for a downloaded release)")
    parser.add_argument("--model-root", type=Path, help="Installed Qwen Image 2.1 model root")
    parser.add_argument("--output", type=Path, help="New local release directory for stage")
    parser.add_argument("--release-dir", type=Path, help="Downloaded Hub repository directory for install")
    args = parser.parse_args(argv)
    neo_root = args.neo_root.resolve() if args.neo_root else ROOT
    if str(neo_root) not in sys.path:
        sys.path.insert(0, str(neo_root))
    model_root = args.model_root.resolve() if args.model_root else neo_root / "models" / "Qwen-Image-2.1"
    if args.action == "stage":
        if args.output is None:
            parser.error("stage requires --output")
        stage(model_root, args.precision, args.output.resolve())
    else:
        release_dir = args.release_dir.resolve() if args.release_dir else Path(__file__).resolve().parent
        install(model_root, release_dir, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
