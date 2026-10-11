"""CPU-only Clef bundle discovery and identities for explicit checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

from modules_forge import local_assets
from modules_forge.model_catalog import checkpoint_roots, choice_labels, discover_files

from .bundle import SUPPORT_FILES, bundle_directory, read_bundle
from .core import PROFILES, RUNTIME, ClefError, canonical_hash, read_source

OFFICIAL = ("flash-int8", "clef-24gb", "flash-bf16")
OFFICIAL_LABELS = {
    "flash-int8": "公式 · Clef Flash · INT8",
    "clef-24gb": "公式 · Clef · NF4",
    "flash-bf16": "公式 · Clef Flash · BF16",
}
DEFAULT_CHECKPOINT = "official:flash-int8"


def checkpoint_manifest(root, profile, checkpoint="", *, verify_hashes=False, verify_code=True):
    settings = PROFILES[profile]
    checkpoint = str(checkpoint or "official:" + profile).strip()
    if checkpoint.startswith("official:"):
        selected = checkpoint.removeprefix("official:")
        if selected not in PROFILES:
            raise ClefError("チェックポイントを選んでください。")
        selected_settings = PROFILES[selected]
        if any(settings[key] != selected_settings[key] for key in ("model", "precision")):
            raise ClefError("選択したチェックポイントと実行設定の版・精度が一致しません。")
        directory = bundle_directory(root, selected)
        if directory is None:
            directory = Path(root) / "source" / selected_settings["model"]
    else:
        try:
            directory = local_assets.local_path(checkpoint, directory=True)
        except ValueError as exc:
            raise ClefError(str(exc)) from exc
    reader = read_source if settings["precision"] == "bf16" else read_bundle
    argument = settings["model"] if settings["precision"] == "bf16" else profile
    try:
        directory, manifest = reader(
            directory.resolve(), argument, verify_hashes=verify_hashes, verify_code=verify_code
        )
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_5" or config.get("architectures") not in (
            ["Qwen3_5Model"],
            ["Qwen3_5ForConditionalGeneration"],
        ):
            raise ClefError("対応する Clef モデル一式の architecture が一致しません。")
        if settings["precision"] == "bf16":
            names = {item["path"] for item in manifest["files"]}
            if not {*SUPPORT_FILES, "config.json", "model.safetensors.index.json"} <= names:
                raise ClefError("Clef BF16 モデル一式の構成が不足しています。")
            index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
            if (
                not isinstance(index, dict)
                or "lm_head.weight" not in index
                or not any(name.endswith("language_model.embed_tokens.weight") for name in index)
                or not set(index.values()) <= names
            ):
                raise ClefError("Clef BF16 の語彙・本体の構成が一致しません。")
        return directory, manifest
    except ClefError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ClefError("対応する Clef モデル一式の manifest・構成を確認してください。") from exc


def compatible_profiles(checkpoint, root=RUNTIME):
    if str(checkpoint).startswith("official:"):
        selected = PROFILES.get(str(checkpoint).removeprefix("official:"), {})
        return [
            key
            for key, value in PROFILES.items()
            if all(value[field] == selected.get(field) for field in ("model", "precision"))
        ]
    found = []
    for profile in PROFILES:
        try:
            checkpoint_manifest(root, profile, str(checkpoint), verify_code=False)
            found.append(profile)
        except (ClefError, OSError, ValueError, KeyError, TypeError):
            pass
    return found


def checkpoint_choices(root=RUNTIME, search_directory="", current=""):
    roots = checkpoint_roots("Clef", roots=(root,), search_directory=search_directory)
    paths = discover_files(
        roots, suffixes=(), folders=True, validator=lambda path: path.is_dir() and bool(compatible_profiles(path, root))
    )
    official_paths = set()
    for profile in OFFICIAL:
        directory = bundle_directory(root, profile) or Path(root) / "source" / PROFILES[profile]["model"]
        official_paths.add(str(directory.resolve()))
    paths = [path for path in paths if path not in official_paths]
    return [(OFFICIAL_LABELS[profile], "official:" + profile) for profile in OFFICIAL] + choice_labels(
        paths, roots=roots, current=current if current and not current.startswith("official:") else ""
    )


def checkpoint_identity(root, profile, checkpoint=""):
    directory, manifest = checkpoint_manifest(root, profile, checkpoint)
    records = [
        {**local_assets.file_identity(directory / item["path"]), "path": item["path"]} for item in manifest["files"]
    ]
    manifest_name = "release.json" if PROFILES[profile]["precision"] == "bf16" else "complete.json"
    records.append({**local_assets.file_identity(directory / manifest_name), "path": manifest_name})
    records.sort(key=lambda item: item["path"])
    return {
        "path": str(directory.resolve()),
        "sha256": canonical_hash(records),
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "precision": PROFILES[profile]["precision"],
        "files": records,
    }


def load_selection():
    value = local_assets.selection("clef")
    checkpoint = str(value.get("checkpoint") or DEFAULT_CHECKPOINT)
    search_directory = str(value.get("search_directory") or "")
    profiles = compatible_profiles(checkpoint)
    profile = value.get("profile", "flash-int8")
    if profiles and profile not in profiles:
        profile = profiles[0]
    return checkpoint, search_directory, profile if profile in PROFILES else "flash-int8"


def save_selection(checkpoint, search_directory, profile):
    local_assets.save_selection(
        "clef", {"checkpoint": checkpoint, "search_directory": search_directory, "profile": profile}
    )
