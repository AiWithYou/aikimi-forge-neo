"""FLUX 3 requests and pinned model assets; no ML imports or downloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import time
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "models/Flux-3-Klein"
COMFY_ROOT = ROOT / "repositories/flux3-klein/ComfyUI"
COMFY_REVISION = "20b46065ef7f34fa21e8abebe8d0cca90a904d6d"
MODEL_REVISION = "bfec4ea10ad389929f0f483d54508fc21e6c2ede"
PUP_PROMPTS_REVISION = "a03efc6b895ae4220aebb3a958488b939df9caf0"
MANIFEST = ROOT / "tools/flux3_klein_manifest.json"
_KEYS = {"model", "precision", "prompt", "images", "boxes", "width", "height", "steps", "cfg", "seed", "pup"}
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class Flux3Error(ValueError):
    """An invalid request, incompatible asset, or unavailable runtime."""


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise Flux3Error(f"{name}は{lower}〜{upper}の整数で指定してください。")
    return value


def _box(value):
    if not isinstance(value, list) or len(value) != 4:
        raise Flux3Error("配置座標は[上, 左, 下, 右]の4要素の配列で指定してください。")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value):
        raise Flux3Error("配置座標は0〜1の有限の数値で指定してください。")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise Flux3Error("配置ボックスの下・右は上・左より大きくしてください。")
    return list(value)


def _boxes(value, image_count):
    if not isinstance(value, list) or len(value) > 256:
        raise Flux3Error("配置ボックスは256個以内の配列で指定してください。")
    result, identifiers = [], set()
    for row in value:
        if not isinstance(row, dict):
            raise Flux3Error("配置ボックスの各要素はオブジェクトで指定してください。")
        identifier, description = row.get("id"), row.get("desc")
        if (
            not isinstance(identifier, str)
            or len(identifier) > 128
            or not _ID.fullmatch(identifier)
            or identifier in identifiers
        ):
            raise Flux3Error("配置のidは重複しない英字始まりの英数字・アンダースコアで指定してください。")
        if not isinstance(description, str) or not description.strip() or len(description) > 2000:
            raise Flux3Error("配置のdescは1〜2000文字の説明で指定してください。")
        identifiers.add(identifier)
        if "bbox" in row:
            if set(row) != {"id", "desc", "bbox"}:
                raise Flux3Error("bbox形式に編集用の指定や未知のキーを混在させないでください。")
            result.append({"id": identifier, "desc": description, "bbox": _box(row["bbox"])})
            continue
        if set(row) != {"id", "desc", "from", "src_bbox", "tgt_bbox"}:
            raise Flux3Error("編集配置はid・desc・from・src_bbox・tgt_bboxで指定してください。")
        origin = row["from"]
        source, target = row["src_bbox"], row["tgt_bbox"]
        if origin == "target":
            if source is not None or target is None:
                raise Flux3Error("targetから追加する要素はsrc_bboxをnull、tgt_bboxを座標にしてください。")
        elif isinstance(origin, str) and re.fullmatch(r"ref_image_(0|[1-9][0-9]*)", origin):
            if int(origin.removeprefix("ref_image_")) >= image_count or source is None:
                raise Flux3Error("fromは入力済み参照画像を指定し、src_bboxを指定してください。")
        else:
            raise Flux3Error("fromはtargetまたはref_image_0から始まる入力済み参照画像を指定してください。")
        result.append(
            {
                "id": identifier,
                "desc": description,
                "from": origin,
                "src_bbox": None if source is None else _box(source),
                "tgt_bbox": None if target is None else _box(target),
            }
        )
    return result


def validate_request(value) -> dict:
    if not isinstance(value, dict) or set(value) - _KEYS:
        raise Flux3Error("生成要求は対応するキーだけを持つオブジェクトで指定してください。")
    model, precision = value.get("model", "distilled"), value.get("precision", "int8")
    if not isinstance(model, str) or model not in {"distilled", "base"}:
        raise Flux3Error("モデルはdistilledまたはbaseを指定してください。")
    if not isinstance(precision, str) or precision not in {"int8", "bf16", "nvfp4"}:
        raise Flux3Error("保存形式はint8・bf16・nvfp4を指定してください。")
    prompt = value.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise Flux3Error("プロンプトは1〜12000文字で入力してください。")
    images = value.get("images", [])
    if not isinstance(images, list) or len(images) > 10:
        raise Flux3Error("参照画像は入力順の配列で最大10枚です。")
    width, height = value.get("width"), value.get("height")
    if (width is None) != (height is None):
        raise Flux3Error("幅と高さは両方指定するか、両方を自動にしてください。")
    if width is not None:
        for name, number in (("幅", width), ("高さ", height)):
            _integer(number, name, 64, 4096)
            if number % 16:
                raise Flux3Error(f"{name}は16の倍数で指定してください。")
    elif not images:
        width = height = 1024
    steps = _integer(value.get("steps", 16 if model == "distilled" else 50), "Steps", 1, 200)
    cfg = value.get("cfg", 1.0 if model == "distilled" else 3.5)
    if type(cfg) not in (int, float) or not math.isfinite(cfg) or not 0 < cfg <= 100:
        raise Flux3Error("CFGは0より大きく100以下の有限の数値で指定してください。")
    if model == "distilled" and cfg != 1:
        raise Flux3Error("蒸留版のCFGは1です。")
    seed = _integer(value.get("seed", -1), "Seed", -1, 2**32 - 1)
    if seed == -1:
        seed = secrets.randbits(32)
    pup = value.get("pup", False)
    if type(pup) is not bool:
        raise Flux3Error("PUPはtrueまたはfalseで指定してください。")
    return {
        "model": model,
        "precision": precision,
        "prompt": prompt,
        "images": list(images),
        "boxes": _boxes(value.get("boxes", []), len(images)),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": float(cfg),
        "seed": seed,
        "pup": pup,
    }


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 1
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def python_path(root: Path = RUNTIME) -> Path:
    return Path(root) / "worker-env/Scripts/python.exe"


def comfy_path(root: Path = RUNTIME) -> Path:
    return Path(root).resolve().parent.parent / "repositories/flux3-klein/ComfyUI"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def model_specs(request: dict) -> dict[str, dict]:
    model, precision = request.get("model", "distilled"), request.get("precision", "int8")
    if (
        not isinstance(model, str)
        or model not in {"distilled", "base"}
        or not isinstance(precision, str)
        or precision not in {"int8", "bf16", "nvfp4"}
    ):
        raise Flux3Error("モデルまたは保存形式の指定が不正です。")
    suffix = "int8_convrot" if precision == "int8" else precision
    variant = "_sd" if model == "distilled" else ""
    names = {
        "dit": f"diffusion_models/flux3_image_klein{variant}_{suffix}.safetensors",
        "text_encoder": f"text_encoders/qwen3vl_4b_{'bf16' if precision == 'bf16' else 'int8_convrot'}.safetensors",
        "vae": "vae/flux2-vae.safetensors",
        "pup": "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
        "pup_prompts": "text_encoders/system_prompts.json",
    }
    files = manifest()["files"]
    return {name: dict(files[path], path=path) for name, path in names.items()}


def model_paths(root: Path, request: dict) -> dict[str, Path]:
    return {name: Path(root) / spec["path"] for name, spec in model_specs(request).items()}


@lru_cache(maxsize=64)
def _file_hash(path: str, size: int, mtime: int, ctime: int) -> str:
    del size, mtime, ctime
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset_ready(root: Path, spec: dict) -> bool:
    try:
        root = Path(root).resolve()
        path = root / spec["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            return False
        stat = path.stat()
        return (
            stat.st_size == spec["size"]
            and _file_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == spec["sha256"]
        )
    except (OSError, KeyError, ValueError):
        return False


def model_ready(root: Path, request: dict) -> bool:
    return runtime_ready(root) and all(
        asset_ready(root, spec)
        for name, spec in model_specs(request).items()
        if request.get("pup", False) or name not in {"pup", "pup_prompts"}
    )


def runtime_fingerprint() -> str:
    lock = ROOT / "tools/requirements-flux3-klein.lock"
    return hashlib.sha256(MANIFEST.read_bytes() + lock.read_bytes() + b"flux3-runtime-v1").hexdigest()


def runtime_ready(root: Path = RUNTIME) -> bool:
    root = Path(root)
    runtime = comfy_path(root)
    if not python_path(root).is_file() or not (runtime / "main.py").is_file():
        return False
    try:
        receipt = json.loads((root / ".runtime.json").read_text(encoding="utf-8"))
        if receipt != {"schema_version": 1, "fingerprint": runtime_fingerprint(), "comfy_revision": COMFY_REVISION}:
            return False
        options = {
            "capture_output": True,
            "text": True,
            "check": False,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        git = shutil.which("git")
        if git is None:
            return False
        result = subprocess.run([git, "-C", str(runtime), "rev-parse", "HEAD"], **options)  # noqa: S603 -- fixed read-only Git arguments; no shell.
        if result.returncode or result.stdout.strip() != COMFY_REVISION:
            return False
        changed = subprocess.run([git, "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"], **options)  # noqa: S603 -- fixed read-only Git arguments; no shell.
        return changed.returncode == 0 and not changed.stdout.strip()
    except (OSError, ValueError, TypeError):
        return False
