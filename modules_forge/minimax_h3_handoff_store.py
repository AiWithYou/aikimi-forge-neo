"""Durable H3 handoff snapshots; stdlib only, no network or GPU side effects.

An execution prompt is NOT an editable ComfyUI workflow. The frontend bridge
imports this prompt using ComfyUI itself, verifies the round trip, and offers
its actual UI workflow JSON for download.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

PACK = "Aikimi-H3-WorkflowBridge"
STORE = "aikimi_h3_workflows"
FORMAT = "aikimi-h3-handoff-v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_NODES = 4096
TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
FILE_INPUTS = {"LoadImage": "image", "LoadVideo": "file", "LoadAudio": "audio"}


class HandoffError(ValueError):
    """A handoff is incomplete or unsafe; do not silently reconstruct it."""


def json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise HandoffError("ワークフローにJSON化できない値があります。") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def token_value(token: str) -> str:
    if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
        raise HandoffError("ワークフローIDの形式が不正です。")
    return token


def checked_path(root: Path, relative: str, *, exists: bool = True) -> Path:
    """Reject traversal, Windows paths, symlinks and junctions, including parents."""
    root = Path(root).resolve(strict=True)
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\x00" in relative:
        raise HandoffError("素材パスが不正です。")
    parts = PurePosixPath(relative).parts
    if relative.startswith("/") or any(p in {".", ".."} for p in parts):
        raise HandoffError("素材パスが保存領域の外を指しています。")
    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise HandoffError("リンクされた素材・保存先は使用しません。")
    try:
        resolved = candidate.resolve(strict=exists)
    except OSError as exc:
        raise HandoffError("再編集用の素材が見つかりません。") from exc
    if not resolved.is_relative_to(root) or resolved == root:
        raise HandoffError("素材パスが保存領域の外を指しています。")
    if exists and not resolved.is_file():
        raise HandoffError("再編集用の素材が見つかりません。")
    return resolved


def _read_json(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise HandoffError("ワークフローJSONが上限を超えています。")
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise HandoffError("ワークフローJSONを読み込めません。") from exc
    if not isinstance(value, dict):
        raise HandoffError("ワークフローJSONの形式が不正です。")
    return value


def validate_prompt(prompt: Any) -> None:
    if not isinstance(prompt, dict) or not 0 < len(prompt) <= MAX_NODES:
        raise HandoffError("実行グラフが空、または大きすぎます。")
    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not node_id.isdecimal():
            raise HandoffError("H3のノードIDは数値文字列である必要があります。")
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("class_type"), str)
            or not isinstance(node.get("inputs"), dict)
        ):
            raise HandoffError("実行ノードの形式が不正です。")
        for name, value in node["inputs"].items():
            if not isinstance(name, str):
                raise HandoffError("入力名が不正です。")
            # Existing H3 builds encode every array input as a connection.
            if isinstance(value, list):
                if (
                    len(value) != 2
                    or not isinstance(value[0], str)
                    or value[0] not in prompt
                    or type(value[1]) is not int
                    or value[1] < 0
                ):
                    raise HandoffError("ノード接続が不正です。")
    if len(json_bytes(prompt)) > MAX_JSON_BYTES:
        raise HandoffError("実行グラフが大きすぎます。")


def prepared_names(prepared: dict) -> set[str]:
    names = {prepared[k] for k in ("first_frame", "last_frame", "control_video") if prepared.get(k)}
    names.update(prepared.get("images") or [])
    names.update(prepared.get("audios") or [])
    names.update(v["name"] for v in prepared.get("videos") or [] if isinstance(v, dict) and v.get("name"))
    if not all(isinstance(n, str) for n in names):
        raise HandoffError("準備済み素材の形式が不正です。")
    return names


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_snapshot(
    input_root: Path, prompt: dict, prepared: dict, metadata: dict, *, token: str | None = None
) -> dict:
    """Copy prepared bytes once and publish an immutable directory atomically.

    Original graph is preserved as submitted_prompt. Only known loader file
    inputs are remapped in prompt; free text is NEVER globally replaced.
    The prepared source is copied. A top-level editor alias links to that
    snapshot copy; in-place edits through either name fail hash verification.
    """
    validate_prompt(prompt)
    token = token_value(token or uuid.uuid4().hex)
    root = Path(input_root).resolve(strict=True)
    store = checked_path(root, STORE, exists=False)
    store.mkdir(exist_ok=True)
    destination = checked_path(root, f"{STORE}/{token}", exists=False)
    if destination.exists():
        raise HandoffError("このワークフローIDはすでに存在します。上書きしません。")
    source_names = prepared_names(prepared)
    graph = copy.deepcopy(prompt)
    assets = []
    visible_links = []
    published = False
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=store))
    try:
        names = {}
        for index, old_name in enumerate(sorted(source_names)):
            if not old_name.startswith("forge_h3/"):
                raise HandoffError("H3が準備した素材以外はコピーしません。")
            source = checked_path(root, old_name)
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", source.name)[-150:] or "media"
            leaf = f"{index:03d}_{safe_name}"
            target = stage / leaf
            before = source.stat()
            shutil.copyfile(source, target)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise HandoffError("コピー中に入力素材が変更されました。やり直してください。")
            relative = f"{STORE}/{token}/{leaf}"
            # ComfyUI's upload selectors list top-level input files only.
            # Link our copied snapshot into that list without another media copy.
            ui_file = f"aikimi_h3_{token}_{leaf}"
            ui_path = checked_path(root, ui_file, exists=False)
            os.link(target, ui_path)
            visible_links.append(ui_path)
            names[old_name] = ui_file
            assets.append(
                {"file": relative, "ui_file": ui_file, "bytes": target.stat().st_size, "sha256": _hash_file(target)}
            )
        referenced = set()
        for node in graph.values():
            field = FILE_INPUTS.get(node["class_type"])
            if field and isinstance(node["inputs"].get(field), str):
                name = node["inputs"][field]
                if name not in names:
                    raise HandoffError("グラフが参照する素材を保存できません。")
                node["inputs"][field] = names[name]
                referenced.add(name)
        if referenced != source_names:
            raise HandoffError("準備素材とグラフの素材接続が一致しません。")
        result = {
            "format": FORMAT,
            "token": token,
            "created_at": datetime.now(UTC).isoformat(),
            "prompt": graph,
            "submitted_prompt": copy.deepcopy(prompt),
            "submitted_prompt_sha256": digest(prompt),
            "prompt_sha256": digest(graph),
            "required_nodes": sorted({n["class_type"] for n in graph.values()}),
            "assets": assets,
            "metadata": copy.deepcopy(metadata),
        }
        result["snapshot_sha256"] = digest(result)
        encoded = json_bytes(result)
        if len(encoded) > MAX_JSON_BYTES:
            raise HandoffError("ワークフローJSONが大きすぎます。")
        with (stage / "manifest.json").open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Destination token is random and exclusive. Never overwrite a manifest.
        if destination.exists():
            raise HandoffError("同じIDへの保存が競合しました。")
        os.rename(stage, destination)
        published = True
        return result
    finally:
        if not published:
            for link in visible_links:
                link.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)


def read_snapshot(input_root: Path, token: str, *, verify_hashes: bool = True) -> dict:
    root = Path(input_root).resolve(strict=True)
    token = token_value(token)
    record = _read_json(checked_path(root, f"{STORE}/{token}/manifest.json"))
    if record.get("format") != FORMAT or record.get("token") != token:
        raise HandoffError("保存ワークフローのバージョンまたはIDが一致しません。")
    if record.get("snapshot_sha256") != digest({k: v for k, v in record.items() if k != "snapshot_sha256"}):
        raise HandoffError("保存ワークフローが変更・破損しています。")
    validate_prompt(record.get("prompt"))
    validate_prompt(record.get("submitted_prompt"))
    for name in ("prompt", "submitted_prompt"):
        if record.get(f"{name}_sha256") != digest(record[name]):
            raise HandoffError("保存ワークフローが変更・破損しています。")
    assets = record.get("assets")
    if not isinstance(assets, list) or len(assets) > 1024:
        raise HandoffError("保存素材の一覧が不正です。")
    listed = set()
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("file"), str)
            or not asset["file"].startswith(f"{STORE}/{token}/")
        ):
            raise HandoffError("保存素材が別のワークフローを指しています。")
        path = checked_path(root, asset["file"])
        if path.stat().st_size != asset.get("bytes") or (verify_hashes and _hash_file(path) != asset.get("sha256")):
            raise HandoffError("再編集用の素材が変更・破損しています。")
        ui_file = asset.get("ui_file", asset["file"])
        if ui_file != asset["file"]:
            if not isinstance(ui_file, str) or not ui_file.startswith(f"aikimi_h3_{token}_") or "/" in ui_file:
                raise HandoffError("編集用素材名が不正です。")
            ui_path = checked_path(root, ui_file)
            if ui_path.stat().st_size != asset["bytes"] or (
                verify_hashes and not ui_path.samefile(path) and _hash_file(ui_path) != asset["sha256"]
            ):
                raise HandoffError("編集用素材が変更・破損しています。")
        listed.add(ui_file)
    referenced = {
        n["inputs"][field]
        for n in record["prompt"].values()
        if (field := FILE_INPUTS.get(n["class_type"])) and isinstance(n["inputs"].get(field), str)
    }
    if referenced != listed:
        raise HandoffError("保存素材とノード接続が一致しません。")
    return record


def handoff_url(server_url: str, token: str) -> str:
    token_value(token)
    try:
        url = urlsplit(server_url)
        port = url.port
    except (ValueError, TypeError) as exc:
        raise HandoffError("ComfyUI URLが不正です。") from exc
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
        or not port
    ):
        raise HandoffError("この機能はポートを明示したローカルComfyUI専用です。")
    return f"http://{url.hostname}:{port}/#aikimi-h3={token}"


def write_receipt(video: Path, record: dict) -> Path:
    """Tie a successful output to its own exact snapshot, not a global latest ID."""
    token_value(record["token"])
    video = Path(video)
    if not video.is_file() or video.is_symlink():
        raise HandoffError("生成結果の保存先が不正です。")
    target = video.with_suffix(video.suffix + ".comfy.json")
    if target.is_symlink() or getattr(target, "is_junction", lambda: False)():
        raise HandoffError("リンクされたワークフロー記録は使用しません。")
    payload = {
        "format": FORMAT,
        "token": record["token"],
        "snapshot_sha256": record["snapshot_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "submitted_prompt_sha256": record["submitted_prompt_sha256"],
    }
    if target.exists():
        if _read_json(target) != payload:
            raise HandoffError("異なる生成のワークフロー記録は上書きしません。")
        return target
    fd, temporary = tempfile.mkstemp(prefix=".h3-receipt-", dir=video.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise HandoffError("ワークフロー記録の保存が競合しました。")
        os.link(temporary, target)  # Atomic no-clobber publication on the same filesystem.
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def read_receipt(video: Path) -> dict:
    target = Path(video).with_suffix(Path(video).suffix + ".comfy.json")
    if not target.is_file() or target.is_symlink():
        raise HandoffError(
            "この履歴には再編集用スナップショットがありません。旧履歴を現在の設定で置き換えることはしません。"
        )
    record = _read_json(target)
    if record.get("format") != FORMAT:
        raise HandoffError("ワークフロー記録の形式が不正です。")
    token_value(record.get("token"))
    return record
