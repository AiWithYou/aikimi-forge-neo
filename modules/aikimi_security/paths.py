"""Minimal file-serving paths for Gradio."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class UnsafeAllowedPathError(ValueError):
    """Raised when a CLI path would expose data outside managed directories."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _deduplicate(paths: Iterable[Path]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(str(path))
    return result


def _relative_parts(path: Path, root: Path) -> tuple[str, ...] | None:
    try:
        return tuple(part.casefold() for part in path.relative_to(root).parts)
    except ValueError:
        return None


def _create_managed_directory(root: Path, name: str) -> Path:
    requested = root / name
    candidate = _resolved(requested)
    if not _within(candidate, root):
        raise UnsafeAllowedPathError(f"The managed {name} path resolves outside its data root.")

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise UnsafeAllowedPathError(f"The managed {name} directory could not be prepared.") from error
    created = _resolved(requested)
    if created != candidate or not created.is_dir() or not _within(created, root):
        raise UnsafeAllowedPathError(f"The managed {name} path changed while it was being prepared.")
    return created


def _is_javascript_asset(path: Path, script_root: Path, data_root: Path) -> bool:
    if path.suffix.casefold() not in {".js", ".mjs"}:
        return False
    script_parts = _relative_parts(path, script_root)
    if script_parts is not None and len(script_parts) == 2 and script_parts[0] == "javascript":
        return True
    return any(
        parts is not None
        and (len(parts) == 4 and parts[0] in {"extensions", "extensions-builtin"} and parts[2] == "javascript")
        for parts in (_relative_parts(path, root) for root in (script_root, data_root))
    )


def _is_stylesheet_asset(path: Path, script_root: Path, data_root: Path) -> bool:
    if path == _resolved(script_root / "style.css"):
        return True
    return any(
        parts is not None
        and len(parts) == 3
        and parts[0] in {"extensions", "extensions-builtin"}
        and parts[2] == "style.css"
        for parts in (_relative_parts(path, root) for root in (script_root, data_root))
    )


_TAG_AUTOCOMPLETE_TEMP_FILES = (
    "emb.txt",
    "hyp.txt",
    "lora.txt",
    "lyco.txt",
    "styles.txt",
    "umi_tags.txt",
    "wc.txt",
    "wce.txt",
    "wc_yaml.json",
)


def _tag_autocomplete_data_files(asset: Path, script_root: Path, data_root: Path) -> set[Path]:
    """Expose only data read by an active Tag Autocomplete JavaScript asset."""

    if asset.name.casefold() != "tagautocomplete.js":
        return set()
    if not any(
        parts is not None
        and len(parts) == 4
        and parts[0] == "extensions"
        and parts[2:] == ("javascript", "tagautocomplete.js")
        for parts in (_relative_parts(asset, root) for root in (script_root, data_root))
    ):
        return set()
    extension_root = asset.parent.parent
    if not (extension_root / "scripts" / "tag_autocomplete_helper.py").is_file():
        return set()
    tags_root = _resolved(extension_root / "tags")
    if not _within(tags_root, extension_root):
        raise UnsafeAllowedPathError("Tag Autocomplete data resolves outside its extension directory.")
    if not tags_root.is_dir():
        return set()

    candidates = [path for pattern in ("*.csv", "*.json") for path in tags_root.glob(pattern)]
    temp_root = _resolved(tags_root / "temp")
    if not _within(temp_root, tags_root):
        raise UnsafeAllowedPathError("Tag Autocomplete temporary data resolves outside its tags directory.")
    candidates.extend(temp_root / name for name in _TAG_AUTOCOMPLETE_TEMP_FILES)
    result: set[Path] = set()
    for path in candidates:
        resolved = _resolved(path)
        if not _within(resolved, tags_root):
            raise UnsafeAllowedPathError("Tag Autocomplete data file resolves outside its tags directory.")
        if resolved.is_file():
            result.add(resolved)
    return result


def build_gradio_allowed_paths(
    script_path: str | Path,
    data_path: str | Path,
    *,
    canvas_root: str | Path | None = None,
    javascript_paths: Iterable[str | Path] = (),
    stylesheet_paths: Iterable[str | Path] = (),
    notification_audio: str | Path | None = None,
    requested_paths: Iterable[str | Path] = (),
) -> list[str]:
    """Return managed output/temp directories and exact UI asset files.

    JavaScript and stylesheet paths must come from the active ``modules.scripts``
    listings. Only the individual files used by the UI are exposed; their parent
    repository and extension directories remain outside Gradio's allowlist.
    """

    script_root = _resolved(script_path)
    data_root = _resolved(data_path)
    directory_roots = {_create_managed_directory(data_root, name) for name in ("output", "outputs", "tmp")}
    if script_root != data_root:
        for name in ("output", "outputs", "tmp"):
            candidate = _resolved(script_root / name)
            if not _within(candidate, script_root):
                raise UnsafeAllowedPathError(f"The legacy managed {name} path resolves outside the repository.")
            if candidate.is_dir():
                directory_roots.add(candidate)
    exact_files: set[Path] = set()

    for name in ("script.js", "style.css"):
        candidate = _resolved(script_root / name)
        if not _within(candidate, script_root):
            raise UnsafeAllowedPathError(f"The root UI asset {name} resolves outside the repository.")
        if candidate.is_file():
            exact_files.add(candidate)

    user_stylesheet = _resolved(data_root / "user.css")
    if not _within(user_stylesheet, data_root):
        raise UnsafeAllowedPathError("The user.css stylesheet resolves outside the data directory.")
    if user_stylesheet.is_file():
        exact_files.add(user_stylesheet)

    card_placeholder = _resolved(script_root / "html" / "card-no-preview.jpg")
    if not _within(card_placeholder, script_root):
        raise UnsafeAllowedPathError("The card placeholder resolves outside the repository.")
    if card_placeholder.is_file():
        exact_files.add(card_placeholder)

    for path in javascript_paths:
        candidate = _resolved(path)
        if not _is_javascript_asset(candidate, script_root, data_root):
            raise UnsafeAllowedPathError(
                "A Gradio JavaScript asset must be an exact .js or .mjs file from an active UI javascript directory."
            )
        if candidate.is_file():
            exact_files.add(candidate)
            exact_files.update(_tag_autocomplete_data_files(candidate, script_root, data_root))

    for path in stylesheet_paths:
        candidate = _resolved(path)
        if not _is_stylesheet_asset(candidate, script_root, data_root):
            raise UnsafeAllowedPathError(
                "A Gradio stylesheet asset must be an exact root or active-extension style.css file."
            )
        if candidate.is_file():
            exact_files.add(candidate)

    if notification_audio is not None:
        candidate = _resolved(notification_audio)
        expected = _resolved(script_root / "notification.mp3")
        if candidate != expected or not _within(candidate, script_root):
            raise UnsafeAllowedPathError("Gradio may only expose the repository notification.mp3 audio file.")
        if candidate.is_file():
            exact_files.add(candidate)

    if canvas_root is not None:
        root = _resolved(canvas_root)
        for name in ("canvas.js", "canvas.css"):
            candidate = _resolved(root / name)
            if not _within(candidate, root):
                raise UnsafeAllowedPathError(f"The Forge Canvas {name} asset resolves outside its asset root.")
            exact_files.add(candidate)

    for requested in requested_paths:
        candidate = _resolved(requested)
        if candidate in exact_files or any(candidate == root or _within(candidate, root) for root in directory_roots):
            continue
        raise UnsafeAllowedPathError("--gradio-allowed-path may only select a managed output or temporary path.")

    existing_directories = sorted(directory_roots, key=lambda item: str(item))
    existing_files = sorted((path for path in exact_files if path.is_file()), key=lambda item: str(item))
    return _deduplicate([*existing_directories, *existing_files])


def build_gradio_blocked_paths(script_path: str | Path, data_path: str | Path) -> list[str]:
    """Add defense-in-depth blocks for credentials, code, models, and state."""

    script_root = _resolved(script_path)
    data_root = _resolved(data_path)
    relative_targets = (
        ".git",
        ".env",
        ".credentials.json",
        "config.json",
        "config_states",
        "ui-config.json",
        "forge_neo_model_paths.yaml",
        "cache",
        "models",
        "repositories",
        "venv",
        ".venv",
        "logs",
        "secrets",
        "api-auth.txt",
        "gradio-auth.txt",
        "params.txt",
        "styles.csv",
        "sysinfo.json",
        "webui-user.bat",
        "webui-user.local.bat",
    )
    targets = {_resolved(root / relative) for root in (script_root, data_root) for relative in relative_targets}
    # Block existing variable-name credential/support files too. Gradio does not
    # interpret globs in ``blocked_paths``, so expand only the narrow root-level
    # patterns instead of scanning model/output trees recursively.
    for root in (script_root, data_root):
        for pattern in (".env.*", "sysinfo*.json", "*.pem", "*.key", "*.p12", "*.pfx"):
            targets.update(_resolved(path) for path in root.glob(pattern))
    return _deduplicate(sorted(targets, key=lambda item: str(item)))
