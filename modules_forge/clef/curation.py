"""Probability filters, human decisions and byte-preserving selected exports."""

from __future__ import annotations

import json
import math
import shutil
import threading
import uuid
from pathlib import Path

from .collection import generation_metadata
from .core import ClefError, atomic_json, sha256

REVIEW_LABELS = {"unreviewed": "未選択", "accepted": "採用", "hold": "保留", "rejected": "見送り"}
_REVIEW_LOCK = threading.RLock()


def latest_export(directory, run, decisions):
    directory = Path(directory)
    try:
        receipt = json.loads((directory / "last-export.json").read_text(encoding="utf-8"))
        exported = (directory / receipt["directory"]).resolve()
        if not exported.is_relative_to((directory / "exports").resolve()):
            return None
        manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
        accepted = {item["id"]: item["sha256"] for item in run["items"] if decisions.get(item["id"]) == "accepted"}
        copied = {item["id"]: item["sha256"] for item in manifest["items"]}
        archive = exported.with_suffix(".zip")
        if (
            not accepted
            or copied != accepted
            or manifest["run_id"] != run["id"]
            or manifest.get("fingerprint") != run.get("fingerprint")
            or not archive.is_file()
        ):
            return None
        return str(exported), str(archive)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def condition_probability(answer, target):
    if not answer:
        return None
    kind = answer.get("type")
    if kind == "noul":
        value = answer.get("noul")
        if value is None or target not in {"true", "false"}:
            return None
        value = float(value) if target == "true" else 1 - float(value)
    else:
        probabilities = answer.get("probabilities", {})
        if kind == "score":
            try:
                lower = int(str(target).removesuffix("+"))
                selected = [key for key in probabilities if int(key) >= lower]
            except ValueError:
                return None
        elif kind == "choice":
            selected = target if isinstance(target, list) else [target]
        else:
            return None
        if not selected or any(key not in probabilities for key in selected):
            return None
        selected = list(dict.fromkeys(selected))
        values = [float(probabilities[key]) for key in selected]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            return None
        value = math.fsum(values)
        # The official answer serializer rounds each option to four decimals.
        tolerance = len(values) * 0.00005 + 1e-12
        if abs(value - 1) <= tolerance and (set(selected) == set(probabilities) or value > 1):
            value = 1.0
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def matching_ids(run, rules, mode="すべて"):
    result = []
    for item in (run or {}).get("items", []):
        if item["status"] != "done":
            continue
        values = [condition_probability(item.get("answers", {}).get(rule["qid"]), rule["target"]) for rule in rules]
        if any(value is None for value in values):
            continue
        tests = [value >= float(rule["minimum"]) for value, rule in zip(values, rules, strict=True)]
        if not rules or (all(tests) if mode == "すべて" else any(tests)):
            result.append(item["id"])
    return result


def read_review(directory, run):
    path = Path(directory) / "review.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("run_id") != run["id"]:
            raise ClefError("採用記録と実行IDが一致しません。")
        return value
    return {"run_id": run["id"], "decisions": {}}


def set_decision(directory, run, token, decision):
    if not token or token[0] != run["id"] or decision not in REVIEW_LABELS:
        raise ClefError("表示中の記録から画像を選択してください。")
    item = next((item for item in run["items"] if item["id"] == token[1]), None)
    if item is None:
        raise ClefError("保存した入力から項目を選択してください。")
    with _REVIEW_LOCK:
        review = read_review(directory, run)
        if decision == "unreviewed":
            review["decisions"].pop(item["id"], None)
        else:
            review["decisions"][item["id"]] = decision
        atomic_json(Path(directory) / "review.json", review)
    return review


def verify_image_snapshot(directory, item):
    path = Path(item["path"]).resolve()
    if not path.is_relative_to((Path(directory) / "inputs").resolve()) or not path.is_file():
        raise ClefError("保存画像がありません、または保存先が不正です。")
    if sha256(path) != item["sha256"]:
        raise ClefError("保存画像が変更されています。元画像から自動で補充はしません。")
    return path


def copy_accepted(directory, run, with_settings=False):
    directory = Path(directory).resolve()
    with _REVIEW_LOCK:
        accepted = dict(read_review(directory, run)["decisions"])
    items = [item for item in run["items"] if accepted.get(item["id"]) == "accepted"]
    if not items:
        raise ClefError("画像を採用してからコピーしてください。")
    sources = []
    for item in items:
        if item["status"] != "done" or item.get("kind", "image") != "image":
            raise ClefError("コピーできるのは判定済みの画像だけです。")
        sources.append(verify_image_snapshot(directory, item))
    identifier = uuid.uuid4().hex[:12]
    parent = directory / "exports"
    staging = parent / f"preparing-{identifier}"
    target = parent / f"selected-{identifier}"
    staging.mkdir(parents=True)
    manifest = {"run_id": run["id"], "fingerprint": run.get("fingerprint"), "items": []}
    for index, (item, source) in enumerate(zip(items, sources, strict=True)):
        name = f"{index:06d}{source.suffix.lower()}"
        copied = staging / name
        shutil.copyfile(source, copied)
        if sha256(copied) != item["sha256"]:
            raise ClefError("コピーした画像のhashが一致しません。書き出しは未完了です。")
        row = {"id": item["id"], "name": item["name"], "file": name, "sha256": item["sha256"]}
        if with_settings:
            metadata = generation_metadata(source)
            for kind in ("workflow", "prompt"):
                if metadata[kind] is not None:
                    filename = f"{index:06d}.{kind}.json"
                    atomic_json(staging / filename, metadata[kind])
                    row[kind] = filename
        manifest["items"].append(row)
    atomic_json(staging / "manifest.json", manifest)
    staging.rename(target)
    return target
