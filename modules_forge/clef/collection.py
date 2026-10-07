"""Local collections, record parsing and saved ComfyUI metadata; no GPU work."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image, ImageOps

from .core import ClefError, canonical_hash, sha256

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def inspect_image(path, name=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ClefError("ファイルがないか、32MiBを超えています。")
    with Image.open(path) as image:
        if image.format not in {"PNG", "JPEG", "WEBP", "BMP"} or image.width * image.height > 40_000_000:
            raise ClefError("PNG/JPEG/WebP/BMP、40メガピクセルまで対応します。")
        size = list(image.size)
        image.verify()
    return {
        "kind": "image",
        "name": name or path.name,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "source_size": size,
    }


def scan_folder(value, recursive=False):
    raw = (value or "").strip().strip('"')
    root = Path(raw)
    if not raw or not root.is_absolute():
        raise ClefError("フォルダの絶対パスを入力してください。")
    root = root.resolve()
    if not root.is_dir():
        raise ClefError("指定されたフォルダがありません。")
    paths, skipped = [], []

    def walk_error(exc):
        skipped.append({"name": str(exc.filename), "reason": "読み取り不可"})

    visited = set()
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        here = Path(directory).resolve()
        if here in visited or not here.is_relative_to(root):
            dirs[:] = []
            continue
        visited.add(here)
        safe_dirs = []
        for name in sorted(dirs):
            child = Path(directory) / name
            resolved = child.resolve()
            if not resolved.is_relative_to(root):
                skipped.append({"name": child.relative_to(root).as_posix(), "reason": "フォルダ外へのリンク"})
            elif recursive and resolved not in visited:
                safe_dirs.append(name)
        dirs[:] = safe_dirs
        paths.extend(Path(directory) / name for name in files)
    items = []
    for path in sorted(paths, key=lambda p: (p.relative_to(root).as_posix().casefold(), str(p))):
        name = path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root):
                raise ClefError("フォルダ外へのリンク")
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ClefError("対象外の形式")
            item = inspect_image(path, name)
            item["id"] = str(len(items))
            items.append(item)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            skipped.append({"name": name, "reason": str(exc)})
    return {"root": str(root), "items": items, "skipped": skipped, "bytes": sum(x["bytes"] for x in items)}


def scan_files(paths):
    items = []
    for index, path in enumerate(paths or []):
        try:
            item = inspect_image(path)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ClefError(f"{Path(path).name}: {exc}") from exc
        item["id"] = str(index)
        items.append(item)
    return {"root": "", "items": items, "skipped": [], "bytes": sum(x["bytes"] for x in items)}


def merge_images(folder, files):
    """Keep folder input when individual files change; one item per source path."""
    items, seen = [], set()
    for item in [*folder["items"], *files["items"]]:
        path = os.path.normcase(str(Path(item["path"]).resolve()))
        if path not in seen:
            seen.add(path)
            items.append({**item, "id": str(len(items))})
    return {**folder, "items": items, "bytes": sum(item["bytes"] for item in items)}


def thumbnails(collection, page=1, per_page=12):
    values = []
    start = (int(page) - 1) * per_page
    for item in (collection or {}).get("items", [])[start : start + per_page]:
        with Image.open(item["path"]) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((300, 300))
        values.append((preview, item["name"]))
    return values


def parse_records(value, format):
    if not isinstance(value, str) or not value.strip():
        raise ClefError("判定する文章・JSONデータを入力してください。")
    if len(value) > 2_000_000:
        raise ClefError("一括入力は2,000,000文字以内にしてください。")
    try:
        if format == "text":
            values = [value]
        elif format == "paragraphs":
            values = [text.strip() for text in re.split(r"\n[ \t\r]*\n", value.replace("\r\n", "\n")) if text.strip()]
        elif format in {"json", "array"}:
            values = json.loads(value)
            if format == "json":
                values = [values]
            elif not isinstance(values, list):
                raise ClefError("JSON配列を指定してください。")
        elif format in {"lines", "jsonl"}:
            values = []
            for line, text in enumerate(value.splitlines(), 1):
                if not text.strip():
                    continue
                try:
                    values.append(json.loads(text) if format == "jsonl" else text.strip())
                except json.JSONDecodeError as exc:
                    raise ClefError(f"JSONLの{line}行目を確認してください。") from exc
        else:
            raise ClefError("入力の区切り方を選択してください。")
    except json.JSONDecodeError as exc:
        raise ClefError(f"JSONの{exc.lineno}行目を確認してください。") from exc
    if not values:
        raise ClefError("入力にレコードがありません。")
    items = []
    for index, state in enumerate(values):
        try:
            encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        except ValueError as exc:
            raise ClefError(f"{index + 1}件目: JSONの数値を確認してください。") from exc
        if len(encoded) > 60_000:
            raise ClefError(f"{index + 1}件目: 1レコードは60,000文字以内にしてください。")
        name = f"{'JSON' if format in {'json', 'array', 'jsonl'} else '文章'} {index + 1}"
        if isinstance(state, dict) and "id" in state:
            name = str(state["id"])
        items.append(
            {"id": str(index), "kind": "record", "name": name, "state": state, "sha256": canonical_hash(state)}
        )
    return items


def generation_metadata(path):
    values = {}
    with Image.open(path) as image:
        for key in ("prompt", "workflow"):
            if isinstance(image.info.get(key), str):
                values[key] = image.info[key]
        for raw in image.getexif().values():
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if isinstance(raw, str):
                prefix, separator, payload = raw.partition(":")
                if separator and prefix.strip().lower() in {"prompt", "workflow"}:
                    values[prefix.strip().lower()] = payload
    result = {"prompt": None, "workflow": None, "settings": {}, "errors": []}
    for key, raw in values.items():
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("JSONオブジェクトではありません。")
            result[key] = value
        except ValueError:
            result["errors"].append(f"{key}のJSONを読み込めません。")
    prompt = result["prompt"] or {}

    def scalar(value, depth=0):
        if isinstance(value, (str, int, float, bool)):
            return str(value)[:500]
        if isinstance(value, list) and len(value) == 2 and depth < 4:
            node = prompt.get(str(value[0]), {})
            inputs = node.get("inputs", {})
            for key in ("value", "seed", "noise_seed", "int", "float"):
                if key in inputs:
                    return scalar(inputs[key], depth + 1)
        return None

    wanted = {
        "seed",
        "noise_seed",
        "steps",
        "cfg",
        "guidance",
        "sampler_name",
        "scheduler",
        "denoise",
        "ckpt_name",
        "unet_name",
        "lora_name",
        "width",
        "height",
    }
    settings = {}
    for node in prompt.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        for key, value in node["inputs"].items():
            if key in wanted and (resolved := scalar(value)) is not None:
                normalized = "seed" if key == "noise_seed" else key
                settings.setdefault(normalized, []).append(resolved)
    result["settings"] = {key: " / ".join(dict.fromkeys(values)) for key, values in settings.items()}
    return result
