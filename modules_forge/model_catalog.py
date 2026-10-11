"""Small, CPU-only checkpoint discovery shared by the Studio selectors."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_IGNORED = {".cache", ".git", "__pycache__", "worker-env", "venv", ".venv"}


def _path(value) -> Path:
    path = Path(str(value).strip().strip('"')).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _unique(paths):
    result = {}
    for value in paths:
        path = _path(value)
        result.setdefault(os.path.normcase(str(path)), path)
    return tuple(result.values())


def checkpoint_roots(family: str, roots=(), search_directory: str = "") -> tuple[Path, ...]:
    """Keep official storage in place; add managed and optional external checkpoint roots."""
    if not family or family in {".", ".."} or any(char in family for char in "/\\:"):
        raise ValueError("モデル系列の名前が不正です。")
    managed = ROOT / "models" / family / "checkpoints"
    legacy = ROOT / "model" / family
    paths = [managed]
    if legacy.is_dir():
        paths.append(legacy)
    paths.extend(roots)
    if search_directory and search_directory.strip():
        paths.append(search_directory)
    return _unique(paths)


def discover_files(roots, *, suffixes=(".safetensors", ".gguf"), folders=False, validator=None) -> list[str]:
    """Adapters decide compatibility; accepted bundles hide their internal components."""
    suffixes = {suffix.lower() for suffix in suffixes}
    found = []

    def accepted(path):
        try:
            if validator is not None:
                return bool(validator(path))
            return (path / "model_index.json").is_file() or (
                path.name == "transformer" and (path / "config.json").is_file()
            )
        except (OSError, ValueError):
            return False

    for root in _unique(roots):
        if not root.is_dir():
            continue
        for directory, children, filenames in os.walk(root, followlinks=False):
            path = Path(directory)
            children[:] = sorted(name for name in children if name not in _IGNORED and not name.startswith("."))
            if folders and accepted(path):
                found.append(path)
                children.clear()
                continue
            for name in sorted(filenames):
                candidate = path / name
                if candidate.suffix.lower() in suffixes and (validator is None or accepted(candidate)):
                    found.append(candidate)
    bundles = [path for path in found if path.is_dir()]
    return [
        str(path)
        for path in sorted(_unique(found))
        if not any(path != bundle and path.is_relative_to(bundle) for bundle in bundles)
    ]


def choice_labels(paths, roots=(), current: str = "") -> list[tuple[str, str]]:
    """Use relative display labels and canonical path values, including a missing selection."""
    candidates = list(paths)
    if current and current.strip():
        candidates.append(current)
    roots = sorted(_unique(roots), key=lambda path: len(path.parts), reverse=True)
    records = []
    for path in _unique(candidates):
        parent = path.parent
        label = path.name
        for root in roots:
            if path.is_relative_to(root):
                label = path.relative_to(root).as_posix() if path != root else path.name
                parent = root
                break
        records.append((label, path, parent))
    counts = Counter(label.casefold() for label, _, _ in records)
    labels = []
    for label, path, parent in records:
        if counts[label.casefold()] > 1:
            peers = [p for other, _, p in records if other.casefold() == label.casefold()]
            depth = 1
            while depth < len(parent.parts) and len({tuple(p.parts[-depth:]) for p in peers}) < len(peers):
                depth += 1
            label += " · " + "/".join(parent.parts[-depth:])
        if not path.exists():
            label += " (見つかりません)"
        labels.append((label, str(path)))
    return sorted(labels, key=lambda item: (item[0].casefold(), item[1]))


def configure_component(component):
    """Studio selections are restored from the library, never overwritten by UiLoadsave."""
    component.do_not_save_to_config = True
    return component
