"""Request and filesystem contracts shared by the UI, worker and setup."""

from __future__ import annotations

import json
import math
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "models" / "Iris-3B"
SOURCE_REPO = "speridlabs/iris-3b"
SOURCE_REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
CODE_REVISION = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
PACKAGING_REVISION = "inference-training-extra-v1"
INT8_REPO = "Aikimi/iris-3b-int8"
INT8_REVISION = "6231647d02bb538b401a675061c11c7a64bba81f"
TEXT_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
TASKS = {"generate": "画像生成", "depth": "深度推定", "upscale": "復元・4倍拡大"}
PRECISIONS = {"int8": "INT8", "normal": "通常版"}
GENERATION_SIZES = {
    "1024×1024": (1024, 1024),
    "1344×768": (1344, 768),
    "1280×832": (1280, 832),
    "1152×896": (1152, 896),
    "896×1152": (896, 1152),
    "832×1280": (832, 1280),
    "768×1344": (768, 1344),
}


class IrisError(ValueError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_request(value):
    request = dict(value)
    task = request.get("task", "generate")
    precision = request.get("precision", "int8")
    if task not in TASKS or precision not in PRECISIONS:
        raise IrisError("処理またはモデルを選び直してください。")
    request.update(task=task, precision=precision)
    if task != "generate":
        if not request.get("image"):
            raise IrisError("入力画像を選んでください。")
        return {"task": task, "precision": precision, "image": request["image"]}
    request.pop("image", None)
    prompt = str(request.get("prompt", "")).strip()
    if not prompt or len(prompt) > 12000:
        raise IrisError("プロンプトを1〜12000文字で入力してください。")
    request["prompt"] = prompt
    request["negative_prompt"] = str(request.get("negative_prompt", ""))
    for name, default, low, high in (
        ("width", 1024, 768, 1344),
        ("height", 1024, 768, 1344),
        ("steps", 100, 1, 200),
        ("seed", -1, -1, 2**32 - 1),
    ):
        number = request.get(name, default)
        try:
            numeric = float(number)
            if not math.isfinite(numeric) or numeric != int(numeric) or not low <= numeric <= high:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise IrisError(f"{name}は{low}〜{high}の整数を指定してください。") from exc
        request[name] = int(numeric)
    if (request["width"], request["height"]) not in GENERATION_SIZES.values():
        raise IrisError("画像生成のサイズを選び直してください。")
    try:
        cfg = float(request.get("cfg", 3.0))
        if not math.isfinite(cfg) or not 1 <= cfg <= 15:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise IrisError("CFGは1〜15で指定してください。") from exc
    request["cfg"] = cfg
    if request["seed"] == -1:
        request["seed"] = secrets.randbelow(2**32)
    return request


def model_directory(root, precision, task):
    directory = Path(root) / ("int8" if precision == "int8" else "official")
    return directory / {"generate": "", "depth": "depth", "upscale": "upscaler"}[task]


def python_path(root=RUNTIME):
    return Path(root) / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment_ready(root=RUNTIME):
    try:
        marker = json.loads((Path(root) / "runtime.json").read_text(encoding="utf-8"))
        return (
            python_path(root).is_file()
            and marker["code_revision"] == CODE_REVISION
            and marker["packaging"] == PACKAGING_REVISION
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def files_ready(directory, manifest):
    directory = Path(directory)
    try:
        records = manifest["files"]
        return bool(records) and all(
            (path := directory / record["path"]).resolve().is_relative_to(directory.resolve())
            and path.is_file()
            and path.stat().st_size == record["size"]
            for record in records
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def model_ready(root, precision, task):
    if not environment_ready(root):
        return False
    directory = model_directory(root, precision, task)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        required = {"model.safetensors", "config.yaml"}
        if task != "generate":
            required.add("empty_prompt.safetensors")
        if not required.issubset({record["path"] for record in manifest["files"]}) or not files_ready(
            directory, manifest
        ):
            return False
        if task == "generate":
            encoder = Path(root) / "text-encoder"
            marker = json.loads((encoder / "download.json").read_text(encoding="utf-8"))
            return (
                marker["revision"] == TEXT_REVISION
                and (encoder / "config.json").is_file()
                and files_ready(encoder, marker)
            )
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def runtime_status(root=RUNTIME):
    ready = environment_ready(root)
    models = {p: [TASKS[t] for t in TASKS if model_ready(root, p, t)] for p in PRECISIONS}
    if not ready:
        return "実行環境を準備してください。"
    text = " · ".join(f"{PRECISIONS[p]}: {', '.join(tasks)}" for p, tasks in models.items() if tasks)
    return text or "使うモデルを準備してください。"
