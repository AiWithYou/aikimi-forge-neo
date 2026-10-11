from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path

from safetensors import SafetensorError, safe_open

from modules_forge.anima_lora import (
    detect_anima_lora_block_count_from_keys,
    has_anima_lora_signature,
)

logger = logging.getLogger(__name__)

ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"
V2_ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"
BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
BUNDLE_FORMAT = "1"
CONNECTOR_PREFIX = "net.anima_v2_connector."


def forge_root() -> Path:
    return Path(__file__).resolve().parents[3]


def text_encoder_roots() -> list[Path]:
    roots = [forge_root() / "models" / "text_encoder"]
    try:
        from modules_forge.main_entry import module_list

        roots.extend(Path(path).parent for path in module_list.values())
    except Exception:
        logger.debug("Falling back to the local text encoder directory.", exc_info=True)
    return list(dict.fromkeys(path.resolve() for path in roots if path.is_dir()))


def lora_roots() -> list[Path]:
    roots = [forge_root() / "models" / "Lora"]
    try:
        from modules import shared

        roots = [
            Path(shared.cmd_opts.lora_dir),
            *(Path(path) for path in shared.cmd_opts.lora_dirs),
        ]
    except Exception:
        logger.debug("Falling back to the local LoRA directory.", exc_info=True)
    return list(dict.fromkeys(path.resolve() for path in roots if path.is_dir()))


def qwen35_models() -> dict[str, str]:
    found: dict[str, str] = {}
    markers = ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b")
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            if any(marker in path.name.lower() for marker in markers):
                found.setdefault(path.name, str(path))
    return dict(sorted(found.items()))


def bundle_metadata(path: str | os.PathLike) -> dict[str, str] | None:
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
    except (OSError, ValueError, SafetensorError):
        return None
    if (
        metadata.get("architecture") != BUNDLE_ARCHITECTURE
        or metadata.get("anima_v2_bundle_format") != BUNDLE_FORMAT
    ):
        return None
    return metadata


def adapters() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    metadata = checkpoint.metadata() or {}
                    keys = set(checkpoint.keys())
                if metadata.get("architecture") != ARCHITECTURE:
                    continue
                if any(key.startswith(("timestep_gates.", "anchor_deviation")) for key in keys):
                    continue
            except (OSError, ValueError, SafetensorError):
                continue
            label = os.path.relpath(path, root).replace("\\", "/")
            found.setdefault(label, str(path))
    return dict(sorted(found.items()))


def _lora_files(root: Path):
    shared = getattr(sys.modules.get("modules"), "shared", None)
    walker = getattr(shared, "walk_files", None)
    if walker is not None:
        yield from (Path(path) for path in walker(str(root), allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
        return

    def natural_key(value):
        return [
            [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", part)]
            for part in os.path.splitext(str(value))
        ]

    for directory, _, filenames in sorted(os.walk(root, followlinks=True), key=lambda item: natural_key(item[0])):
        if "/." in directory or "\\." in directory:
            continue
        yield from (Path(directory) / name for name in sorted(filenames, key=natural_key))


def standard_anima_loras() -> dict[str, str]:
    """Find complete 28-, 40-, or 52-block Anima LoRAs from their headers."""

    candidates: dict[str, Path] = {}
    seen: set[str] = set()
    for root in lora_roots():
        for path in _lora_files(root):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".ckpt", ".safetensors"}:
                continue
            canonical = os.path.normcase(os.path.realpath(path))
            if canonical in seen:
                continue
            seen.add(canonical)
            name = os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, "/")
            if name in candidates:
                name = f"{root.name}/{name}"
                if name in candidates:
                    name = "local_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
            candidates[name] = path

    found: dict[str, str] = {}
    for name, path in candidates.items():
        if path.suffix.lower() != ".safetensors":
            continue
        try:
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                metadata = checkpoint.metadata() or {}
                keys = checkpoint.keys()
                block_count = detect_anima_lora_block_count_from_keys(keys)
                is_anima = has_anima_lora_signature(keys, metadata)
            if block_count is None or not is_anima:
                continue
        except (OSError, ValueError, SafetensorError):
            continue
        found[name] = str(path)
    return dict(sorted(found.items()))


def tokenizer_dir() -> Path:
    bundled = Path(__file__).resolve().parents[1] / "qwen35_tokenizer"
    if (bundled / "tokenizer.json").is_file():
        return bundled
    for candidate in (
        root / "qwen35_tokenizer" for root in text_encoder_roots()
    ):
        if (candidate / "tokenizer.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Qwen3.5 tokenizer files are missing from the extension's "
        "qwen35_tokenizer directory. Reinstall the extension."
    )
