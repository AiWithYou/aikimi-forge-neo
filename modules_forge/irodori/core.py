"""CPU-only request, model readiness and audio artifact boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import wave
from pathlib import Path

from modules_forge.yue2_studio.core import atomic_json as atomic_json

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "repositories/irodori"
MODELS = ROOT / "models/Irodori-TTS"
REQUIRED_FILES = (
    "v4-Large/model.safetensors",
    "v4-Large/tokenizer/tokenizer.json",
    "v4-Large/tokenizer/tokenizer_config.json",
    "codec/weights.pth",
    "watermark/44_1_khz/73999_iteration/enc_c.ckpt",
    "watermark/44_1_khz/73999_iteration/dec_c.ckpt",
    "watermark/44_1_khz/73999_iteration/dec_m_0.ckpt",
    "watermark/44_1_khz/73999_iteration/hparams.yaml",
)
CODE_REVISION = "89f9d8fbd4d51ea019867ee1197725ede1df13c5"
MODEL_REPO = "Aratako/Irodori-TTS-v4-Large"
MODEL_REVISION = "2e0c55428ce97268a507f1feeb2478f8d9148e8b"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
WATERMARK_REPO = "sony/silentcipher"
WATERMARK_REVISION = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1"


class IrodoriError(ValueError):
    """An actionable setup, input or generation failure."""


def python_path(root=RUNTIME):
    return Path(root) / "source/.venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def child_environment(*, offline):
    from modules_forge.yue2_studio.core import safe_environment

    environment = safe_environment() if offline else dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        environment.pop(key, None)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        if key in os.environ:
            environment[key] = os.environ[key]
    environment.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return environment


def ready(root=RUNTIME, models=MODELS):
    try:
        manifest = json.loads((Path(root) / "runtime.json").read_text(encoding="utf-8"))
        records = {x["file"]: x for x in json.loads((Path(root) / "model-files.json").read_text(encoding="utf-8"))}
        return (
            python_path(root).is_file()
            and manifest.get("code_revision") == CODE_REVISION
            and manifest.get("model_revision") == MODEL_REVISION
            and manifest.get("codec_revision") == CODEC_REVISION
            and manifest.get("watermark_revision") == WATERMARK_REVISION
            and all(
                (Path(models) / x).is_file()
                and records[x]["size"] > 0
                and (Path(models) / x).stat().st_size == records[x]["size"]
                and (Path(models) / x).stat().st_mtime_ns == records[x]["mtime_ns"]
                for x in REQUIRED_FILES
            )
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _number(value, name, low, high, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IrodoriError(f"{name}は数値で指定してください。") from exc
    if not math.isfinite(number) or not low <= number <= high or (integer and not number.is_integer()):
        raise IrodoriError(f"{name}は{low}〜{high}の{'整数' if integer else '数値'}で指定してください。")
    return int(number) if integer else number


def validate_request(value):
    text = str(value.get("text", "")).strip()
    caption = str(value.get("caption", "") or "").strip()
    if not text or len(text) > 2000:
        raise IrodoriError("読み上げ文は1〜2000文字で入力してください。長文は分けて生成してください。")
    if len(caption) > 1000:
        raise IrodoriError("声の説明は1000文字以内で入力してください。")
    precision = value.get("precision", "bf16")
    if precision not in {"bf16", "fp32"}:
        raise IrodoriError("モデルの精度はBF16またはFP32を選んでください。")
    seed = _number(value.get("seed", -1), "seed", -1, 2**32 - 1, True)
    seconds = _number(value.get("seconds", 0) or 0, "長さ", 0, 30)
    if 0 < seconds < 0.5:
        raise IrodoriError("長さは0（自動）または0.5〜30秒にしてください。")
    return {
        "text": text,
        "caption": caption or None,
        "precision": precision,
        "seed": secrets.randbelow(2**32) if seed == -1 else seed,
        "seconds": seconds or None,
        "duration_scale": _number(value.get("duration_scale", 1), "長さ倍率", 0.5, 2),
        "steps": _number(value.get("steps", 40), "ステップ", 1, 100, True),
        **{
            name: _number(value.get(name, default), label, 0, 15)
            for name, default, label in (
                ("cfg_text", 3, "テキストCFG"),
                ("cfg_caption", 3, "声の説明CFG"),
                ("cfg_speaker", 5, "参照音声CFG"),
            )
        },
        "references": [],
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_references(paths, directory):
    if len(paths) > 12:
        raise IrodoriError("参照音声は12ファイル、合計120秒までです。")
    records, total = [], 0.0
    for index, source in enumerate(paths):
        source = Path(source)
        if not source.is_file() or source.stat().st_size > 100_000_000:
            raise IrodoriError("参照音声を確認してください。1ファイル100MBまでです。")
        name = f"reference-{index + 1:02d}{source.suffix.lower()}"
        target = Path(directory) / name
        shutil.copyfile(source, target)
        try:
            with wave.open(str(target), "rb") as stream:
                frames, rate = stream.getnframes(), stream.getframerate()
                channels, width = stream.getnchannels(), stream.getsampwidth()
                if frames > 120 * rate:
                    raise IrodoriError("参照音声は合計120秒までです。")
                samples = stream.readframes(frames)
            seconds = frames / rate
            total += seconds
            if frames <= 0 or channels not in {1, 2} or width not in {2, 3, 4} or total > 120:
                raise IrodoriError("参照音声はモノラルまたはステレオ、合計120秒までです。")
            if len(samples) != frames * channels * width or not any(samples):
                raise IrodoriError("参照音声が無音、またはファイルが不完全です。")
        except (wave.Error, EOFError, OSError, ValueError) as exc:
            raise IrodoriError(f"参照音声を読み込めません。16/24/32bit PCM WAVを使用してください: {exc}") from exc
        records.append({"file": name, "seconds": seconds, "sha256": sha256(target)})
    return records


def verify_result(directory):
    directory = Path(directory)
    try:
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        with wave.open(str(directory / "audio.wav"), "rb") as stream:
            frames, rate = stream.getnframes(), stream.getframerate()
            channels, width = stream.getnchannels(), stream.getsampwidth()
            if frames > 31 * rate:
                raise IrodoriError("生成音声の長さが上限を超えています。")
            samples = stream.readframes(frames)
        if (
            rate != 48000
            or frames <= 0
            or channels != 1
            or width != 2
            or not any(samples)
            or len(samples) != frames * width
            or result.get("sample_rate") != rate
            or result.get("samples") != frames
            or result.get("seconds") != frames / rate
            or result.get("seed") != request["seed"]
            or result.get("job_id") != directory.name
            or result.get("model_revision") != MODEL_REVISION
            or result.get("code_revision") != CODE_REVISION
            or not isinstance(result.get("watermarked"), bool)
        ):
            raise IrodoriError("生成音声と結果情報が一致しません。")
        return result
    except (OSError, wave.Error, EOFError, RuntimeError, ValueError, KeyError) as exc:
        raise IrodoriError(f"完成した音声を確認できません: {exc}") from exc
