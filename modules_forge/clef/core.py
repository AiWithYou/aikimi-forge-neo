"""Small, torch-free contracts for images, schemas and immutable decision runs."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "models" / "Clef"
OUTPUTS = ROOT / "outputs" / "clef"
PREPROCESSING_VERSION = "exif-rgb-size-v2"
MODELS = {
    "clef-flash": {"repo": "Cloudflare/clef-flash", "revision": "17f0b0ad64efb65d273590632833508766b2aae6"},
    "clef": {"repo": "Cloudflare/clef", "revision": "2f3de3dd85f379784083b0814d997ab627200f0c"},
}
PROFILES = {
    "flash-int8": {
        "label": "Flash INT8 · 軽量",
        "model": "clef-flash",
        "precision": "int8",
        "cpu_embeddings": False,
        "min_vram_gib": 12,
    },
    "clef-24gb": {
        "label": "Clef NF4 · 24GB",
        "model": "clef",
        "precision": "nf4",
        "cpu_embeddings": False,
        "min_vram_gib": 20,
    },
    "clef-16gb": {
        "label": "Clef NF4 · 16GB",
        "model": "clef",
        "precision": "nf4",
        "cpu_embeddings": True,
        "min_vram_gib": 14,
    },
    "flash-bf16": {
        "label": "Flash BF16 · 比較用",
        "model": "clef-flash",
        "precision": "bf16",
        "cpu_embeddings": False,
        "min_vram_gib": 20,
    },
}
TEMPLATES = {
    "顔・手の見直し（参考）": {
        "style": {
            "type": "choice",
            "instructions": "人物の主な写り方を分類してください。",
            "criteria": {
                "closeup": "顔のアップ",
                "upper": "上半身中心",
                "full": "全身または足まで見える",
                "other": "人物なし・その他",
            },
        },
        "face": {"type": "noul", "instructions": "人物の顔に、目や口の数・位置の異常や崩れがある。"},
        "hands": {"type": "noul", "instructions": "見えている手や指に、本数や形の異常がある。"},
        "marks": {"type": "noul", "instructions": "画像に文字、ロゴ、署名、透かしのいずれかがある。"},
        "background": {
            "type": "choice",
            "instructions": "背景を分類してください。",
            "criteria": {
                "simple": "単色・単純な背景",
                "indoor": "屋内の背景",
                "outdoor": "屋外の背景",
                "other": "その他",
            },
        },
    },
    "画像の評価": {
        "style": {
            "type": "choice",
            "instructions": "画像の主な表現形式を分類してください。",
            "criteria": {
                "photo": "写真または写実的な画像",
                "illustration": "イラスト・アニメ・絵画",
                "design": "図・ポスター・グラフィックデザイン",
            },
        },
        "quality": {
            "type": "score",
            "instructions": "画像の見た目の完成度を評価してください。",
            "criteria": [
                "大きな破綻や欠落がある",
                "目立つ問題があり修正が必要",
                "おおむね整っている",
                "細部まで整い完成度が高い",
            ],
        },
        "text": {"type": "noul", "instructions": "画像に読める文字が含まれている。"},
    },
    "生成画像の確認": {
        "artifact": {"type": "noul", "instructions": "画像に不自然な形、重複、形状の破綻が見られる。"},
        "composition": {
            "type": "score",
            "instructions": "構図の明瞭さと主題の伝わりやすさを評価してください。",
            "criteria": ["主題が不明瞭", "主題は分かるが構図に問題がある", "主題が明瞭で構図が整っている"],
        },
    },
    "文章・JSONの分類": {
        "department": {
            "type": "choice",
            "instructions": "問い合わせを担当するチームはどれですか。",
            "criteria": {"billing": "請求や支払い", "technical": "技術的な問題や障害", "other": "その他"},
        },
        "outage": {"type": "noul", "instructions": "サービスが停止している。"},
    },
}
TEMPLATES["人物イラストの仕分け"] = {
    key: copy.deepcopy(TEMPLATES["顔・手の見直し（参考）"][key]) for key in ("style", "marks", "background")
}


class ClefError(ValueError):
    """An input or installation problem to show directly in the Studio."""


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        deadline = time.monotonic() + 2
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                delay = 0.03 * 2**attempt
                if (
                    getattr(exc, "winerror", None) not in {5, 32, 33}
                    or attempt == 6
                    or time.monotonic() + delay > deadline
                ):
                    raise
                time.sleep(delay)
    except BaseException as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup:
            exc.add_note(f"一時ファイルを削除できません: {temporary}: {cleanup}")
        raise


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ClefError(f"IDが重複しています: {key}")
        result[key] = value
    return result


def parse_schema(value):
    if isinstance(value, str):
        if len(value) > 100_000:
            raise ClefError("判断項目のJSONは100,000文字以内にしてください。")
        try:
            value = json.loads(value, object_pairs_hook=_unique)
        except json.JSONDecodeError as exc:
            raise ClefError(f"JSONの{exc.lineno}行目・{exc.colno}文字目を確認してください。") from exc
    if not isinstance(value, dict) or not 1 <= len(value) <= 32:
        raise ClefError("判断項目は1〜32件のオブジェクトにしてください。")
    for qid, question in value.items():
        if not isinstance(qid, str) or not qid.strip() or len(qid) > 80:
            raise ClefError("項目IDは1〜80文字にしてください。")
        if not isinstance(question, dict) or question.get("type") not in {"choice", "score", "noul"}:
            raise ClefError(f"{qid}: 種類はchoice / score / noulを指定してください。")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ClefError(f"{qid}: 未対応の設定があります。")
        if not isinstance(question.get("instructions", ""), str):
            raise ClefError(f"{qid}: 判断することは文章で指定してください。")
        criteria, kind = question.get("criteria"), question["type"]
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 64:
                raise ClefError(f"{qid}: 選択肢はIDと説明の組を2〜64件指定してください。")
            if any(
                not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                for k, v in criteria.items()
            ):
                raise ClefError(f"{qid}: 選択肢のIDと説明を入力してください。")
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 32
                or any(not isinstance(x, str) or not x.strip() for x in criteria)
            ):
                raise ClefError(f"{qid}: 段階の説明を低い順に2〜32件指定してください。")
        elif criteria is not None and (
            not isinstance(criteria, dict)
            or set(criteria) - {"true", "false"}
            or any(not isinstance(x, str) for x in criteria.values())
        ):
            raise ClefError(f"{qid}: 真偽の説明はtrue / falseで指定してください。")
    return copy.deepcopy(value)


def edit_question(schema, previous, qid, kind, instructions, options):
    result = copy.deepcopy(schema)
    qid = (qid or "").strip()
    if qid in result and qid != previous:
        raise ClefError(f"IDが重複しています: {qid}")
    question = {"type": kind, "instructions": (instructions or "").strip()}
    lines = [x.strip() for x in (options or "").splitlines() if x.strip()]
    if kind == "choice":
        pairs = []
        for line in lines:
            if ":" not in line:
                raise ClefError("選択肢は1行に「ID: 説明」の形で指定してください。")
            key, description = line.split(":", 1)
            pairs.append((key.strip(), description.strip()))
        question["criteria"] = _unique(pairs)
    elif kind == "score":
        question["criteria"] = lines
    if previous and previous != qid:
        result.pop(previous, None)
    result[qid] = question
    return parse_schema(result)


def integer(value, label, lower, upper):
    try:
        number = int(value)
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) != number
            or not lower <= number <= upper
        ):
            raise ValueError
        return number
    except (ValueError, TypeError, OverflowError) as exc:
        raise ClefError(f"{label}は{lower}〜{upper}の整数にしてください。") from exc


def validate_request(request):
    profile = request.get("profile")
    if profile not in PROFILES:
        raise ClefError("実行プロファイルを選んでください。")
    schema = parse_schema(request.get("questions"))
    state = request.get("state", "")
    if not isinstance(state, (str, dict, list, int, float, bool, type(None))):
        raise ClefError("補足入力は文章またはJSONにしてください。")
    if len(json.dumps(state, ensure_ascii=False)) > 60_000:
        raise ClefError("補足入力は60,000文字以内にしてください。")
    return {
        "profile": profile,
        "questions": schema,
        "state": state,
        "max_pixels": integer(request.get("max_pixels", 262144), "画像の最大画素数", 65536, 1048576),
        "max_length": integer(request.get("max_length", 4096), "入力トークン上限", 512, 8192),
    }


def fingerprint(request, items):
    profile = PROFILES[request["profile"]]
    return canonical_hash(
        {
            "request": request,
            "model": MODELS[profile["model"]],
            "profile": profile,
            "inputs": [{"kind": item.get("kind", "image"), "sha256": item["sha256"]} for item in items],
            "preprocessing": PREPROCESSING_VERSION,
        }
    )


def summary(answer, question):
    kind = answer["type"]
    if kind == "noul":
        return f"P(真) {answer['noul']:.3f}"
    if kind == "choice":
        choice = answer["choice"]
        return f"{choice} · p={answer['confidence']:.3f}"
    return f"期待値 {answer['score']:.2f} (0–{len(question['criteria']) - 1})"


def ambiguity(answer):
    if answer["type"] == "noul":
        return 1 - 2 * abs(answer["noul"] - 0.5)
    probabilities = list(answer["probabilities"].values())
    return -sum(p * math.log(p) for p in probabilities if p > 0) / math.log(len(probabilities))


def compare_runs(left, right):
    if left.get("preprocessing") != right.get("preprocessing"):
        raise ClefError("比較には画像の前処理の一致が必要です。異なる前処理の結果は比較できません。")
    for key in ("questions", "state", "max_pixels"):
        if canonical_hash(left["request"].get(key)) != canonical_hash(right["request"].get(key)):
            raise ClefError("比較には判断項目・補足入力・処理解像度の一致が必要です。")
    other = defaultdict(deque)
    for item in right["items"]:
        if item["status"] == "done":
            other[(item.get("kind", "image"), item["sha256"])].append(item)
    rows = []
    for item in left["items"]:
        key = (item.get("kind", "image"), item["sha256"])
        if item["status"] != "done" or not other[key]:
            continue
        matched = other[key].popleft()
        differences = {}
        for qid, answer in item["answers"].items():
            target = matched["answers"][qid]
            if answer["type"] == "noul":
                delta = target["noul"] - answer["noul"]
                differences[qid] = {
                    "delta": delta,
                    "changed": (answer["noul"] >= 0.5) != (target["noul"] >= 0.5),
                    "max_probability_delta": abs(delta),
                }
            else:
                delta = max(abs(p - target["probabilities"][key]) for key, p in answer["probabilities"].items())
                changed = (
                    answer.get("choice") != target.get("choice")
                    if answer["type"] == "choice"
                    else max(answer["probabilities"], key=answer["probabilities"].get)
                    != max(target["probabilities"], key=target["probabilities"].get)
                )
                differences[qid] = {"changed": changed, "max_probability_delta": delta}
                if answer["type"] == "score":
                    differences[qid]["delta"] = target["score"] - answer["score"]
        rows.append(
            {
                "name": item["name"],
                "sha256": item["sha256"],
                "differences": differences,
                "changed": any(x["changed"] for x in differences.values()),
                "max_probability_delta": max(x["max_probability_delta"] for x in differences.values()),
            }
        )
    return sorted(rows, key=lambda x: x["max_probability_delta"], reverse=True)


def export_csv(run):
    schema = run["request"]["questions"]
    columns = ["name", "sha256", "status", "error"] + [
        col for qid in schema for col in ("answer:" + qid, "confidence:" + qid)
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for item in run["items"]:
        row = {key: item.get(key, "") for key in columns[:4]}
        for qid, answer in item.get("answers", {}).items():
            row["answer:" + qid] = answer.get("choice", answer.get("score", answer.get("noul", "")))
            row["confidence:" + qid] = answer.get("confidence", "")
        for key, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                row[key] = "'" + value
        writer.writerow(row)
    return stream.getvalue()


def source_manifest(root, model, verify_hashes=False):
    directory = Path(root) / "source" / model
    try:
        manifest = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClefError(
            f"{model}のBF16元重みは未導入です。aikimi-clef-setup.bat --source --model {model}を実行してください。"
        ) from exc
    if manifest.get("revision") != MODELS[model]["revision"] or manifest.get("repo") != MODELS[model]["repo"]:
        raise ClefError("Clefの固定リビジョンが一致しません。セットアップを再実行してください。")
    for item in manifest["files"]:
        path = directory / item["path"]
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
            or path.stat().st_size != item["size"]
        ):
            raise ClefError(f"Clef配布物が不足・変更されています: {item['path']}")
        if (verify_hashes or path.suffix == ".py") and sha256(path) != item["sha256"]:
            raise ClefError(f"ClefのSHA-256が一致しません: {item['path']}")
    return directory, manifest
