"""Stage and import Neo's saved Qwen Image 2.1 quantized components.

Staging never changes the local quantized cache. Importing requires the pinned
official model and the same isolated runtime versions used for conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "Qwen-Image-2.1"
BASE_MODEL = "Qwen/Qwen-Image-2.1"
COMPONENTS = ("transformer", "text_encoder")
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:\\\\|/home/|/Users/)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root: Path, name: str) -> Path:
    relative = Path(name)
    target = (root / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe release path: {name}")
    return target


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def stage(model_root: Path, precision: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Release directory already exists: {output}")
    model_inventory = json.loads((model_root / "model-files.json").read_text(encoding="utf-8"))
    release = {
        "schema": 1,
        "base_model": BASE_MODEL,
        "base_revision": model_inventory["revision"],
        "precision": precision,
        "components": {},
    }
    output.mkdir(parents=True)
    try:
        for component in COMPONENTS:
            candidates = list((model_root / "quantized" / precision).glob(f"*/{component}/complete.json"))
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one saved {precision}/{component}, found {len(candidates)}")
            source_dir = candidates[0].parent
            record = json.loads(candidates[0].read_text(encoding="utf-8"))
            identity = record["identity"]
            if (identity["revision"], identity["precision"], identity["component"]) != (
                release["base_revision"],
                precision,
                component,
            ):
                raise ValueError(f"Source identity mismatch: {source_dir}")
            files = []
            for entry in record["files"]:
                source = checked_path(source_dir, entry["path"])
                if not source.is_file() or source.stat().st_size != entry["size"]:
                    raise ValueError(f"Source file missing or changed: {source}")
                dest_name = f"{component}/{entry['path']}"
                destination = checked_path(output, dest_name)
                if destination.suffix == ".safetensors":
                    link_or_copy(source, destination)
                elif destination.name == "config.json" and precision == "int8":
                    config = json.loads(source.read_text(encoding="utf-8"))
                    if "_name_or_path" in config:
                        config["_name_or_path"] = BASE_MODEL
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                if destination.suffix == ".json" and LOCAL_PATH.search(destination.read_text(encoding="utf-8")):
                    raise ValueError(f"Local filesystem path in release metadata: {destination}")
                files.append(
                    {
                        "path": dest_name,
                        "size": destination.stat().st_size,
                        "sha256": sha256(destination) if destination.suffix != ".safetensors" else entry["sha256"],
                    }
                )
            release["components"][component] = {
                "recipe": identity["recipe"],
                "versions": identity["versions"],
                "source_inventory": identity["source_inventory"],
                "files": files,
            }
        source_docs = ROOT / "docs" / "hf-qwen-release" / precision
        for name in ("README.md", "NOTICE.md"):
            shutil.copyfile(source_docs / name, output / name)
        shutil.copyfile(model_root / "model" / "LICENSE", output / "LICENSE")
        shutil.copyfile(Path(__file__), output / "install.py")
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
        INT8_SKIP_MODULES,
        cache_path,
        component_identity,
        manifest,
    )

    record = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    if (record.get("schema"), record.get("base_model"), record.get("precision")) != (1, BASE_MODEL, precision):
        raise ValueError("Unexpected release format, base model, or precision")
    inventory = json.loads((model_root / "model-files.json").read_text(encoding="utf-8"))
    if record["base_revision"] != inventory["revision"]:
        raise ValueError("Official base model revision does not match this release")
    for component in COMPONENTS:
        component_record = record["components"][component]
        identity = component_identity(
            model_root / "model",
            component,
            precision,
            skip_modules=INT8_SKIP_MODULES if precision == "int8" and component == "transformer" else (),
        )
        for key in ("recipe", "versions", "source_inventory"):
            if identity[key] != component_record[key]:
                raise ValueError(f"{component}: local {key} does not match this release")
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
    parser.add_argument("--precision", choices=("int8", "w4a8"), required=True)
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
