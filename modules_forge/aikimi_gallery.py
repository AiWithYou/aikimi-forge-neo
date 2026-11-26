"""Browse managed media without exposing server file names outside the output roots."""

import heapq
import json
import os
import re
import uuid
import zlib
from functools import lru_cache
from pathlib import Path

from modules.aikimi_security.redaction import redact_text
from modules_forge.p_h3_image import H3_IMAGE_OUTPUTS
from modules_forge.workspace_store import (
    STATE_DIRECTORY,
    managed_path,
    open_regular,
    read_json,
    state_transaction,
    write_json,
)

MAX_SCAN_ENTRIES = 8192

MEDIA = {
    suffix: kind
    for kind, suffixes in {
        "image": (".png", ".jpg", ".jpeg", ".webp"),
        "video": (".mp4", ".webm", ".mov"),
        "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    }.items()
    for suffix in suffixes
}
STUDIOS = {
    "moss-image": "moss_image",
    "qwen-image-2.1": "qwen_image21",
    "minimax_h3": "minimax_h3",
    "p-h3-image": "minimax_h3_image",
    "minimax-h3-image": "minimax_h3_image",
    "flux3-klein": "flux3",
    "irodori": "irodori",
    "yue2": "yue2",
}
FIELDS = {
    "prompt",
    "negative_prompt",
    "seed",
    "model",
    "checkpoint",
    "steps",
    "width",
    "height",
    "sampler",
    "model_family",
    "model_id",
    "cfg",
    "cfg_scale",
    "scheduler",
    "denoise",
    "lyrics",
    "text",
    "precision",
}
ABSOLUTE = re.compile(r"(?:[a-zA-Z]:[/\\]|\\\\|(?:^|[\s\"'=:(\[])/[^\s/])")


def _public(value):
    text = str(value)[:16384]
    return "" if ABSOLUTE.search(text) else redact_text(text)


def _collect(data, fields, depth=0):
    if not isinstance(data, dict) or depth > 8:
        return
    for key, value in data.items():
        name = key.lower().replace(" ", "_")
        if name in FIELDS and type(value) in (str, int, float):
            fields[name] = _public(value)
        elif isinstance(value, dict):
            _collect(value, fields, depth + 1)


def _comfy(data, fields):
    if not isinstance(data, dict):
        return
    for node in data.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, dest in (("positive", "prompt"), ("negative", "negative_prompt")):
            ref = inputs.get(key)
            if isinstance(ref, list) and ref:
                source = data.get(str(ref[0]))
                if isinstance(source, dict) and isinstance(source.get("inputs"), dict):
                    value = source["inputs"].get("text")
                    if isinstance(value, str):
                        fields[dest] = _public(value)
        if isinstance(inputs.get("ckpt_name"), str):
            fields["model"] = _public(inputs["ckpt_name"])
        _collect({key: inputs[key] for key in ("steps", "seed") if key in inputs}, fields)


def _png_text(stream):
    """Read bounded text chunks, skipping pixel data without decoding it."""
    if stream.read(8) != bytes.fromhex("89504e470d0a1a0a"):
        return
    remaining = 262144
    for _ in range(16384):
        header = stream.read(8)
        if len(header) != 8 or header[4:] == b"IEND":
            return
        size = int.from_bytes(header[:4], "big")
        kind = header[4:]
        if kind not in (b"iTXt", b"tEXt", b"zTXt") or size > remaining:
            stream.seek(size + 4, 1)
            continue
        chunk = stream.read(size)
        checksum = stream.read(4)
        if len(chunk) != size or zlib.crc32(kind + chunk) != int.from_bytes(checksum, "big"):
            continue
        remaining -= size
        key, separator, value = chunk.partition(bytes(1))
        if not separator or len(key) > 79:
            continue
        compressed = kind == b"zTXt"
        if compressed:
            if value[:1] != bytes(1):
                continue
            value = value[1:]
        elif kind == b"iTXt":
            if value[:2] not in (bytes(2), bytes((1, 0))):
                continue
            compressed = bool(value[0])
            fields = value[2:].split(bytes(1), 2)
            if len(fields) != 3:
                continue
            value = fields[-1]
        if compressed:
            decoder = zlib.decompressobj()
            try:
                value = decoder.decompress(value, remaining + 1)
            except zlib.error:
                continue
            if not decoder.eof or len(value) > remaining:
                continue
            remaining -= len(value)
        yield key.decode("latin1"), value.decode("utf-8" if kind == b"iTXt" else "latin1", errors="replace")


@lru_cache(maxsize=256)
def _png_fields(path, size, modified):
    fields = {}
    with open_regular(path) as stream:
        for key, data in _png_text(stream):
            if key == "parameters":
                text, _, options = data.rpartition("\nSteps:")
                prompt, _, negative = (text if options else data).partition("\nNegative prompt:")
                fields.update(prompt=_public(prompt.strip()), negative_prompt=_public(negative.strip()))
                if options:
                    _collect(
                        dict(pair.split(": ", 1) for pair in ("Steps:" + options).strip().split(", ") if ": " in pair),
                        fields,
                    )
            elif key in {"hyperweave", "moss_image", "qwen", "qwen_image21", "prompt", "flux3_klein"}:
                try:
                    parsed = json.loads(data)
                except (ValueError, RecursionError):
                    continue
                if key == "prompt":
                    _comfy(parsed, fields)
                else:
                    _collect(parsed, fields)
    return fields


@lru_cache(maxsize=256)
def _sidecar_fields(path, identity):
    fields = {}
    _collect(read_json(path, 262144), fields)
    return fields


class Gallery:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def media_path(self, identifier):
        path = managed_path(self.root, identifier)
        if (
            identifier.split("/")[0] not in {"output", "outputs"}
            or path.suffix.lower() not in MEDIA
            or not path.is_file()
        ):
            raise ValueError("作品が見つかりません。")
        return path

    @property
    def favorites_path(self):
        return managed_path(self.root, STATE_DIRECTORY + "/gallery_favorites.json")

    def _favorites(self):
        try:
            values = read_json(self.favorites_path)
        except FileNotFoundError:
            return set()
        if not isinstance(values, list) or len(values) > 4096:
            raise ValueError("お気に入りの形式が不正です。")
        for key in values:
            path = managed_path(self.root, key)
            if key.split("/")[0] not in {"outputs", "output"} or path.suffix.lower() not in MEDIA:
                raise ValueError("お気に入りの ID が不正です。")
        return set(values)

    def favorite(self, identifier, enabled):
        with state_transaction(self.root):
            self.media_path(identifier)
            values = self._favorites()
            (values.add if enabled else values.discard)(identifier)
            if len(values) > 4096:
                raise ValueError("お気に入りの件数上限です。")
            write_json(self.favorites_path, sorted(values))

    def quarantine(self, identifier):
        with state_transaction(self.root):
            path = self.media_path(identifier)
            destination = managed_path(
                self.root, "tmp/aikimi-gallery/quarantine/" + uuid.uuid4().hex + "/" + identifier
            )
            destination.parent.mkdir(parents=True)
            values = self._favorites()
            path.rename(destination)
            if identifier in values:
                try:
                    write_json(self.favorites_path, sorted(values - {identifier}))
                except Exception:
                    destination.rename(path)
                    raise

    def _metadata(self, path, info):
        fields = {}
        for sidecar in (
            path.parent / "project.json",
            path.parent / "request.json",
            path.parent / "parameters.json",
            path.parent / "studio-result.json",
            path.parent / "result.json",
            path.with_suffix(".json"),
        ):
            try:
                safe = managed_path(self.root, sidecar.relative_to(self.root).as_posix())
                info_json = safe.stat()
                fields.update(_sidecar_fields(safe, (info_json.st_mtime_ns, info_json.st_ctime_ns, info_json.st_size, info_json.st_ino)))
            except (OSError, ValueError):
                continue
        if path.suffix.lower() == ".png":
            try:
                fields.update(_png_fields(path, info.st_size, info.st_mtime_ns))
            except (OSError, ValueError):
                pass
        if not fields.get("prompt"):
            fields["prompt"] = fields.get("lyrics") or fields.get("text", "")
        return fields

    def _scan(self):
        # Count every directory entry, including ignored files and directories.
        remaining = MAX_SCAN_ENTRIES
        directories = ["output", "outputs"]
        while directories:
            try:
                parent = managed_path(self.root, directories.pop())
                with os.scandir(parent) as entries:
                    for entry in entries:
                        if remaining == 0:
                            yield None
                            return
                        remaining -= 1
                        path = parent / entry.name
                        if entry.is_dir(follow_symlinks=False):
                            directories.append(path.relative_to(self.root).as_posix())
                        elif path.suffix.lower() in MEDIA:
                            yield path
            except (OSError, ValueError):
                continue

    def list_items(self, studio="", media="", query="", favorites=False, order="newest", offset=0, limit=40):
        if (
            not 1 <= limit <= 100
            or offset < 0
            or len(query) > 512
            or order not in {"newest", "oldest"}
            or media not in {"", "image", "video", "audio"}
            or studio not in {"", "forge", *STUDIOS.values()}
        ):
            raise ValueError("検索条件が不正です。")
        starred = self._favorites()
        total = 0
        truncated = False
        keyword = query.casefold()

        def matches():
            nonlocal total, truncated
            for path in self._scan():
                if path is None:
                    truncated = True
                    break
                relative = path.relative_to(self.root)
                identifier = relative.as_posix()
                kind = MEDIA[path.suffix.lower()]
                owner = STUDIOS.get(relative.parts[1], "forge")
                if "/".join(relative.parts[1:3]) in H3_IMAGE_OUTPUTS:
                    owner = "minimax_h3_image"
                if (studio and studio != owner) or (media and media != kind) or (favorites and identifier not in starred):
                    continue
                try:
                    info = self.media_path(identifier).stat()
                    if keyword and keyword not in path.name.casefold():
                        fields = self._metadata(path, info)
                        if keyword not in " ".join((path.name, *fields.values())).casefold():
                            continue
                except (OSError, ValueError):
                    continue
                total += 1
                yield info.st_mtime_ns, identifier, kind, owner

        select = heapq.nlargest if order == "newest" else heapq.nsmallest
        candidates = select(min(offset + limit, MAX_SCAN_ENTRIES), matches())
        page = []
        for _, identifier, kind, owner in candidates[offset:]:
            try:
                path = self.media_path(identifier)
                info = path.stat()
                fields = self._metadata(path, info)
            except (OSError, ValueError):
                continue
            page.append(dict(id=identifier, studio=owner, media=kind, metadata=fields,
                             favorite=identifier in starred, modified=info.st_mtime))
        return {"items": page, "total": total, "truncated": truncated}
