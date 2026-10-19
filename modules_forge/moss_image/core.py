"""Validated MOSS Image requests and pinned local assets; no ML imports."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from modules_forge.yue2_studio.core import atomic_json as atomic_json

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models/MOSS-Image"
HF_ROOT = MODEL_ROOT / "checkpoints/MOSS-Image-SFT"
MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"
MODEL_REVISION = "daf706d2e1c7a6369c17b6d1658671aee6db1ebe"
MANIFEST = ROOT / "tools/moss_image_manifest.json"
RECEIPT = ".forge-moss-image.json"
LOCK = ROOT / "tools/requirements-moss-image.lock"
_KEYS = {
    "prompt",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "cfg",
    "shift",
    "seed",
    "skip_layer",
    "precision",
    "text_precision",
    "loras",
    "checkpoint",
    "mode",
    "images",
    "edit_method",
    "denoise",
    "kv_cache",
    "hires",
    "rope",
}


class MossImageError(ValueError):
    """An invalid request or an unavailable verified MOSS Image installation."""


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise MossImageError(f"{name}は{lower}〜{upper}の整数で指定してください。")
    return value


def _number(value, name, lower, upper, *, inclusive=False):
    if (
        type(value) not in (int, float)
        or not (lower <= value <= upper if inclusive else lower < value <= upper)
        or not math.isfinite(value)
    ):
        relation = "以上" if inclusive else "より大きく"
        raise MossImageError(f"{name}は{lower}{relation}{upper}以下の有限の数値で指定してください。")
    return float(value)


def image_size(width, height):
    width = _integer(width, "幅", 512, 4096)
    height = _integer(height, "高さ", 512, 4096)
    if width % 16 or height % 16 or width * height > 16777216:
        raise MossImageError("幅と高さは16の倍数、合計16777216画素以内で指定してください。")
    return width, height


def hires_output_size(width, height, scale):
    width, height = image_size(width, height)
    scale = _number(scale, "倍率", 1.05, 4, inclusive=True)
    return image_size(int(width * scale / 16 + 0.5) * 16, int(height * scale / 16 + 0.5) * 16)


def validate_request(value, *, internal=False) -> dict:
    allowed = _KEYS | ({"lora_identities", "checkpoint_identity", "reference_sha256"} if internal else set())
    if not isinstance(value, dict) or set(value) - allowed:
        raise MossImageError("生成要求は対応するキーだけを持つオブジェクトで指定してください。")
    prompt, negative = value.get("prompt"), value.get("negative_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        raise MossImageError("プロンプトは1〜32000文字で入力してください。")
    if not isinstance(negative, str) or len(negative) > 32000:
        raise MossImageError("負のプロンプトは32000文字以内の文字列で指定してください。")
    width, height = image_size(value.get("width", 1024), value.get("height", 1024))
    mode = value.get("mode", "txt2img")
    method = value.get("edit_method", "redraw")
    if mode not in ("txt2img", "img2img"):
        raise MossImageError("生成方式または画像モードを選び直してください。")
    if method not in ("redraw", "reference"):
        raise MossImageError("編集方法を選び直してください。")
    from . import references

    images = references.validate_inputs(value.get("images", []), internal=internal)
    if mode == "txt2img" and images:
        raise MossImageError("参照画像を使う場合はimg2imgを選択してください。")
    if mode == "img2img" and (not images or (method == "redraw" and len(images) != 1)):
        raise MossImageError("再描画は参照画像1枚、複数参照は1〜3枚を入力してください。")
    kv_cache = value.get("kv_cache", True)
    if type(kv_cache) is not bool:
        raise MossImageError("参照キャッシュはtrueまたはfalseで指定してください。")
    denoise = _number(
        value.get("denoise", 0.65 if mode == "img2img" and method == "redraw" else 1),
        "変更量", 0.01, 1, inclusive=True,
    )
    if mode == "txt2img" or method == "reference":
        denoise = 1.0
    seed = _integer(value.get("seed", -1), "Seed", -1, 2**32 - 1)
    skip_layer = value.get("skip_layer", True)
    if type(skip_layer) is not bool:
        raise MossImageError("負条件側の層スキップはtrueまたはfalseで指定してください。")
    precision, text_precision = value.get("precision", "int8"), value.get("text_precision", "int8")
    if not isinstance(precision, str) or precision not in {"int8", "bf16"}:
        raise MossImageError("本体の保存形式はINT8またはBF16を選択してください。")
    if not isinstance(text_precision, str) or text_precision not in {"int8", "w4a8", "bf16"}:
        raise MossImageError("テキストの保存形式はINT8・W4A8・BF16を選択してください。")
    loras, names = value.get("loras", []), set()
    if not isinstance(loras, list) or len(loras) > 16:
        raise MossImageError("LoRAは最大16件の配列で指定してください。")
    normalized = []
    for item in loras:
        if not isinstance(item, dict) or set(item) != {"name", "strength"}:
            raise MossImageError("LoRAは名前と強度を指定してください。")
        name = item["name"]
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 512
            or "\\" in name
            or ":" in name
            or name.startswith("/")
            or any(not part or part.startswith(".") for part in name.split("/"))
            or Path(name).suffix.lower() != ".safetensors"
            or name.casefold() in names
        ):
            raise MossImageError("LoRAは専用フォルダー内の重複しないsafetensors名で指定してください。")
        names.add(name.casefold())
        normalized.append({"name": name, "strength": _number(item["strength"], "LoRA強度", -2, 2, inclusive=True)})
    if mode == "img2img" and method == "reference" and not any(item["strength"] for item in normalized):
        raise MossImageError("複数参照ではMOSSの編集用LoRAを選択してください。")
    result = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "steps": _integer(value.get("steps", 50), "Steps", 1, 200),
        "cfg": _number(value.get("cfg", 3.0), "CFG", 0, 100, inclusive=True),
        "shift": _number(value.get("shift", 5.0), "Shift", 0, 100),
        "seed": secrets.randbits(32) if seed == -1 else seed,
        "skip_layer": skip_layer,
        "precision": precision,
        "text_precision": text_precision,
        "loras": normalized,
        "mode": mode,
        "images": images,
        "edit_method": method,
        "denoise": denoise,
        "kv_cache": kv_cache,
    }
    if "hires" in value:
        if mode != "txt2img":
            raise MossImageError("Hires.fixはtxt2imgで使用してください。")
        hires = value["hires"]
        if not isinstance(hires, dict) or set(hires) - {"base_width", "base_height", "steps", "denoise"}:
            raise MossImageError("Hires.fixには最初の幅・高さ、仕上げSteps・変更量を指定してください。")
        base_width, base_height = image_size(hires.get("base_width"), hires.get("base_height"))
        if base_width > width or base_height > height or (base_width, base_height) == (width, height):
            raise MossImageError("Hires.fixの最終サイズは最初のサイズより大きく指定してください。")
        result["hires"] = {
            "base_width": base_width,
            "base_height": base_height,
            "steps": _integer(hires.get("steps", 20), "仕上げSteps", 1, 200),
            "denoise": _number(hires.get("denoise", 0.35), "仕上げの変更量", 0.01, 1, inclusive=True),
        }
    rope = value.get("rope", {})
    if not isinstance(rope, dict) or set(rope) - {"scale_x", "scale_y"}:
        raise MossImageError("RoPEには横・縦のスケールを指定してください。")
    result["rope"] = {
        key: _number(rope.get(key, 1), "RoPEスケール", 0.25, 4, inclusive=True)
        for key in ("scale_x", "scale_y")
    }
    if internal and "reference_sha256" in value:
        result["reference_sha256"] = references.validate_hashes(images, value["reference_sha256"])
    if internal and "lora_identities" in value:
        result["lora_identities"] = value["lora_identities"]
    checkpoint = value.get("checkpoint", "official")
    if checkpoint != "official":
        from .checkpoints import validate_name

        result["checkpoint"] = validate_name(checkpoint)
        if internal and "checkpoint_identity" in value:
            result["checkpoint_identity"] = value["checkpoint_identity"]
    return result


def python_path(root=MODEL_ROOT) -> Path:
    return runtime_paths(root)["python"]


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def safe_model_root(root: Path) -> Path:
    path = Path(root).absolute()
    if any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)):
        raise MossImageError("MOSS Imageの保存先にリンクやジャンクションは使用できません。")
    return path.resolve()


@lru_cache(maxsize=64)
def _file_hash(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    del size, mtime_ns, ctime_ns
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_specs(request=None) -> list[dict]:
    value = manifest()
    request = request or {}
    precision, text_precision = request.get("precision", "int8"), request.get("text_precision", "int8")
    if precision not in {"int8", "bf16"} or text_precision not in {"int8", "w4a8", "bf16"}:
        raise MossImageError("モデルの保存形式が不正です。")
    prefixes = ["checkpoints/MOSS-Image-SFT/vae/"]
    custom = request.get("checkpoint", "official") != "official"
    if precision == "bf16" and not custom:
        prefixes.append("checkpoints/MOSS-Image-SFT/transformer/")
    if text_precision == "bf16":
        prefixes.extend(("checkpoints/MOSS-Image-SFT/text_encoder/", "checkpoints/MOSS-Image-SFT/tokenizer/"))
    selected = [
        dict(spec)
        for spec in value["files"]
        if spec["path"].startswith(tuple(prefixes)) or spec["path"].endswith("/LICENSE")
    ]
    if precision == "int8" and not custom:
        selected.append(dict(value["assets"]["dit_int8"]))
    if text_precision != "bf16":
        selected.append(dict(value["assets"]["text_" + text_precision]))
    return selected


def asset_paths(root, request) -> dict:
    from . import checkpoints

    root = safe_model_root(root)
    bundle = root / "checkpoints/MOSS-Image-SFT"
    assets = manifest()["assets"]
    paths = {
        "dit": root / assets["dit_int8"]["path"] if request["precision"] == "int8" else bundle,
        "text_encoder": (
            root / assets["text_" + request["text_precision"]]["path"]
            if request["text_precision"] != "bf16"
            else bundle
        ),
        "vae": bundle / "vae/diffusion_pytorch_model.safetensors",
    }
    if request.get("checkpoint", "official") != "official":
        paths["dit"] = checkpoints.resolve(root, request["checkpoint"])
    return paths


def runtime_paths(root=MODEL_ROOT) -> dict:
    base = safe_model_root(root).parent.parent / "repositories/moss-image"
    return {"base": base, "comfy": base / "ComfyUI", "python": base / "venv/Scripts/python.exe"}


def runtime_fingerprint() -> str:
    return hashlib.sha256(MANIFEST.read_bytes() + LOCK.read_bytes() + b"moss-comfy-runtime-v1").hexdigest()


def runtime_ready(root=MODEL_ROOT) -> bool:
    try:
        paths = runtime_paths(root)
        for path in paths.values():
            safe_model_root(path)
        value = manifest()["runtime"]
        receipt = json.loads((paths["base"] / ".runtime.json").read_text(encoding="utf-8"))
        if receipt != {
            "schema_version": 1,
            "fingerprint": runtime_fingerprint(),
            "comfy_revision": value["comfy_revision"],
        }:
            return False
        if not paths["python"].is_file() or not (paths["comfy"] / "comfy/sd.py").is_file():
            return False
        git = shutil.which("git")
        if not git:
            return False
        options = {
            "capture_output": True,
            "text": True,
            "check": False,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        head = subprocess.run([git, "-C", str(paths["comfy"]), "rev-parse", "HEAD"], **options)  # noqa: S603
        changed = subprocess.run(  # noqa: S603 -- Resolved Git executable and fixed read-only arguments.
            [git, "-C", str(paths["comfy"]), "status", "--porcelain", "--untracked-files=no"], **options
        )
        return (
            head.returncode == changed.returncode == 0
            and head.stdout.strip() == value["comfy_revision"]
            and not changed.stdout.strip()
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def model_ready(root: Path = MODEL_ROOT, request=None) -> bool:
    try:
        from . import checkpoints

        root = safe_model_root(root)
        if request is not None:
            checkpoints.bind(root, request)
        if not runtime_ready(root):
            return False
        receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != 2
            or receipt.get("model_id") != MODEL_ID
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("manifest_sha256") != hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        ):
            return False
        specifications = model_specs(request)
        if not specifications or not {spec["path"] for spec in specifications} <= set(receipt["files"]):
            return False
        for spec in specifications:
            relative = Path(spec["path"])
            path = root / relative
            if relative.is_absolute() or not safe_model_root(path).is_relative_to(root):
                return False
            stat = path.stat()
            recorded = receipt["files"][spec["path"]]
            if recorded != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": spec["sha256"]}:
                return False
            if stat.st_size != spec["size"]:
                return False
            # Executable source is always hashed, including after stat-preserving edits.
            digest = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.suffix == ".py"
                else _file_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            )
            if digest != spec["sha256"]:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False
