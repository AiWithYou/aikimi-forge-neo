"""Validated Ming Image requests and pinned local assets; no ML imports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import stat
import subprocess
from pathlib import Path

from modules_forge import local_assets
from modules_forge.yue2_studio.core import atomic_json as atomic_json

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models/Ming-Image"
MODEL_ID = "inclusionAI/Ming-Image-0.2-Design"
MODEL_REVISION = "3538f4c311cf8f5256d631ddb7f47793f741aa2e"
MANIFEST = ROOT / "tools/ming_image_manifest.json"
RECEIPT = ".forge-ming-image.json"
LOCK = ROOT / "tools/requirements-ming-image.lock"
PROFILES = ("vram24g_hq", "vram16g_hq", "comfy_org_w4a8", "comfy_org_int8")
_KEYS = {
    "prompt",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "shift",
    "cfg",
    "seed",
    "profile",
    "mode",
    "images",
    "transparent",
    "checkpoint",
    "text_checkpoint",
}


class MingImageError(ValueError):
    """An invalid request or an unavailable verified Ming Image installation."""


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise MingImageError(f"{name}は{lower}〜{upper}の整数で指定してください。")
    return value


def _number(value, name, lower, upper, *, inclusive=False):
    if (
        type(value) not in (int, float)
        or not (lower <= value <= upper if inclusive else lower < value <= upper)
        or not math.isfinite(value)
    ):
        relation = "以上" if inclusive else "より大きく"
        raise MingImageError(f"{name}は{lower}{relation}{upper}以下の有限の数値で指定してください。")
    return float(value)


def image_size(width, height):
    from modules_forge.studio_dimensions import custom_dimensions

    return custom_dimensions(width, height, minimum=512, maximum=3072, grid=16)


def validate_request(value, *, internal=False) -> dict:
    allowed = _KEYS | ({"text_checkpoint_identity", "checkpoint_identity", "reference_sha256"} if internal else set())
    if not isinstance(value, dict) or set(value) - allowed:
        raise MingImageError("生成要求は対応するキーだけを持つオブジェクトで指定してください。")
    prompt, negative = value.get("prompt"), value.get("negative_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        raise MingImageError("プロンプトは1〜32000文字で入力してください。")
    if not isinstance(negative, str) or len(negative) > 32000:
        raise MingImageError("負のプロンプトは32000文字以内の文字列で指定してください。")
    from . import checkpoints, references

    width, height = image_size(value.get("width", 2048), value.get("height", 2048))
    mode = value.get("mode", "txt2img")
    images = references.validate_inputs(value.get("images", []), internal=internal)
    if mode != ("img2img" if images else "txt2img"):
        raise MingImageError("編集には1〜8枚の参照画像を指定してください。")
    profile = value.get("profile", PROFILES[0])
    if profile not in PROFILES or type(value.get("transparent", False)) is not bool:
        raise MingImageError("プロファイルまたは透過設定が不正です。")
    seed = _integer(value.get("seed", -1), "Seed", -1, 2**63 - 1)
    cfg = _number(value.get("cfg", 1), "CFG", 1, 20, inclusive=True)
    result = dict(
        prompt=prompt,
        negative_prompt=negative if cfg > 1 else "",
        width=width,
        height=height,
        mode=mode,
        images=images,
        transparent=value.get("transparent", False),
        profile=profile,
        seed=secrets.randbits(63) if seed == -1 else seed,
        cfg=cfg,
        shift=_number(value.get("shift", 1.15), "Max shift", 0, 100, inclusive=True),
        steps=_integer(value.get("steps", 12), "Steps", 1, 100),
    )
    for key in ("checkpoint", "text_checkpoint"):
        name = value.get(key, "official")
        result[key] = name if name == "official" else checkpoints.validate_name(name)
    if internal:
        result.update(
            {
                key: value[key]
                for key in ("checkpoint_identity", "text_checkpoint_identity", "reference_sha256")
                if key in value
            }
        )
    return result


def png_metadata(record):
    from modules.aikimi_security.redaction import redact_text

    fields = (
        "width",
        "height",
        "seed",
        "steps",
        "cfg",
        "shift",
        "mode",
        "transparent",
        "profile",
        "reference_sha256",
        "checkpoint_identity",
        "text_checkpoint_identity",
        "model_id",
        "model_revision",
        "comfy_revision",
        "runtime_backend",
        "elapsed_seconds",
        "quant_revision",
        "sampler",
        "scheduler",
    )
    result = {key: record[key] for key in fields if key in record}
    for key in ("prompt", "negative_prompt"):
        if key in record:
            result[key] = redact_text(record[key])
    return result


def python_path(root=MODEL_ROOT) -> Path:
    return runtime_paths(root)["python"]


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _linked(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def safe_model_root(root: Path) -> Path:
    path = Path(root).absolute()
    if any(_linked(parent) for parent in (path, *path.parents)):
        raise MingImageError("Ming Imageの保存先にリンクやジャンクションは使用できません。")
    return path.resolve()


def model_specs(request=None):
    profile = (request or {}).get("profile", PROFILES[0])
    if profile not in PROFILES:
        raise MingImageError("モデルプロファイルを選択してください。")
    value = manifest()["profiles"][profile]
    if value["status"] != "published":
        raise MingImageError("HQモデルの公開後に --sync-hf-manifest を実行するか comfy_org_w4a8 を選択してください。")
    specs = []
    for item in value["files"]:
        filename = item.get("filename", item["path"])
        path = "hq/" + filename if value["repo_id"].startswith("Aikimi/") else filename
        specs.append(dict(item, path=path, filename=filename, repo_id=value["repo_id"], revision=value["revision"]))
    return specs


def asset_paths(root, request):
    from . import checkpoints

    paths = {
        key: safe_model_root(root / spec["path"])
        for key, spec in zip(("dit", "text", "vae"), model_specs(request), strict=True)
    }
    for key, role in (("checkpoint", "dit"), ("text_checkpoint", "text")):
        if request[key] != "official":
            paths[role] = checkpoints.resolve(root, request[key])
    return paths


def runtime_paths(root=MODEL_ROOT) -> dict:
    base = safe_model_root(root).parent.parent / "repositories/ming-image"
    return {"base": base, "comfy": base / "ComfyUI-0.2", "python": base / "venv313/Scripts/python.exe"}


def runtime_fingerprint() -> str:
    return hashlib.sha256(
        manifest()["runtime"]["comfy_revision"].encode() + LOCK.read_bytes() + b"ming-runtime-313-v1"
    ).hexdigest()


CHECKOUT_STATUS = ("status", "--porcelain", "-z", "--ignored", "--untracked-files=all")


def check_checkout_paths(directory):
    safe_model_root(directory / ".git")
    if not (directory / ".git").is_dir():
        raise MingImageError("ComfyUI の Git チェックアウトが見つかりません。")

    def raise_walk_error(error):
        raise error

    for folder, dirs, files in os.walk(directory, onerror=raise_walk_error):
        if any(_linked(Path(folder) / name) for name in (*dirs, *files)):
            raise MingImageError("ComfyUI にリンクやジャンクションがあります。")


def checkout_changed(status):
    for entry in status.split("\0"):
        if not entry:
            continue
        path = Path(entry[3:])
        if entry[:3] in ("!! ", "?? ") and path.parent.name == "__pycache__" and path.suffix == ".pyc":
            continue
        return True
    return False


def runtime_ready(root=MODEL_ROOT, *, explain=False) -> bool:
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
            "encoding": "utf-8",
            "errors": "replace",
            "check": False,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        probe = subprocess.run(  # noqa: S603
            [
                str(paths["python"]),
                "-I",
                "-c",
                "import sys,importlib.metadata as m;print(sys.version_info[:2],m.version('comfy-kitchen'))",
            ],
            **options,
        )
        if probe.returncode or probe.stdout.strip() != "(3, 13) 0.2.42":
            return False
        check_checkout_paths(paths["comfy"])
        head = subprocess.run([git, "-C", str(paths["comfy"]), "rev-parse", "HEAD"], **options)  # noqa: S603
        changed = subprocess.run(  # noqa: S603 -- Resolved Git executable and fixed read-only arguments.
            [git, "-C", str(paths["comfy"]), *CHECKOUT_STATUS], **options
        )
        if checkout_changed(changed.stdout):
            raise MingImageError("ComfyUI に変更または未追跡ファイルがあります。専用チェックアウトを確認してください。")
        return head.returncode == changed.returncode == 0 and head.stdout.strip() == value["comfy_revision"]
    except MingImageError:
        if explain:
            raise
        return False
    except (OSError, ValueError, TypeError, KeyError):
        return False


def model_ready(root: Path = MODEL_ROOT, request=None, *, explain=False) -> bool:
    try:
        from . import checkpoints

        root = safe_model_root(root)
        if request is not None:
            checkpoints.bind(root, request)
        if not runtime_ready(root, explain=explain):
            return False
        receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != 2
            or receipt.get("model_id") != MODEL_ID
            or receipt.get("model_revision") != MODEL_REVISION
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
            digest = local_assets.file_identity(path)["sha256"]
            if digest != spec["sha256"]:
                return False
        return True
    except MingImageError:
        if explain:
            raise
        return False
    except (OSError, ValueError, TypeError, KeyError):
        return False
