"""Validated requests and local artifact boundaries (standard library only)."""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_REVISION = "4d53bd5fc7e96a53cb907d3eb407a65df67a8b79"
MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"
GGUF_ID = "audio-cpp/Yue2-3B-GGUF"
GGUF_MODELS = {
    "q8_0": "yue2-3b-q8_0.gguf",
    "q4_0": "yue2-3b-q4_0.gguf",
    "bf16": "yue2-3b-bf16.gguf",
}
SIDECARS = (
    "yue2-model-config.json", "yue2-generation-config.json",
    "yue2-qwen.tiktoken", "yue2-vae-config.json",
)
MAX_JSON_BYTES = 512 * 1024


class YuE2Error(ValueError):
    """An actionable setup, input, or artifact error."""


def integer(value, label: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise YuE2Error(f"{label}は整数で指定してください。")
    if isinstance(value, str) and not re.fullmatch(r"-?[0-9]+", value.strip()):
        raise YuE2Error(f"{label}は整数で指定してください。")
    try:
        number = int(value)
        if not isinstance(value, str) and number != value:
            raise ValueError()
    except (ValueError, OverflowError):
        raise YuE2Error(f"{label}は有限の整数で指定してください。") from None
    if not lower <= number <= upper:
        raise YuE2Error(f"{label}は{lower}〜{upper}の範囲です。")
    return number


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, limit: int = MAX_JSON_BYTES):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise YuE2Error("JSONファイルが大きすぎます。")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=lambda _: _invalid_json())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise YuE2Error("UTF-8のJSONを指定してください。") from exc


def _invalid_json():
    raise YuE2Error("NaNやInfinityは使用できません。")


@dataclass(frozen=True)
class Request:
    title: str = ""
    style: str = ""
    lyrics: str = ""
    abc: str = ""
    cot: str = "full"
    engine: str = "official"
    seed: int = -1
    candidates: int = 1
    steps: int = 32
    max_tokens: int = 9000
    memory_gib: int = 0
    offload: bool = True
    fp8: bool = False
    gguf: str = "q8_0"

    def validate(self) -> Request:
        for label, value, limit in (
            ("曲名", self.title, 160), ("曲調", self.style, 2000),
            ("歌詞", self.lyrics, 8000), ("楽譜", self.abc, 100000),
        ):
            if not isinstance(value, str) or len(value) > limit or "\x00" in value:
                raise YuE2Error(f"{label}は{limit}文字以内のテキストを指定してください。")
        if not self.style.strip():
            raise YuE2Error("曲調を入力してください。")
        if not self.lyrics.strip():
            raise YuE2Error("歌詞、またはインスト用の [instrumental] を入力してください。")
        if self.cot not in {"full", "melody", "off"} or self.engine not in {"official", "cpp"}:
            raise YuE2Error("生成方式の指定が不正です。")
        if self.abc.strip() and self.cot == "off":
            raise YuE2Error("楽譜を使う場合は「メロディ＋コード」または「メロディ」を選んでください。")
        if self.gguf not in GGUF_MODELS:
            raise YuE2Error("GGUF形式の指定が不正です。")
        for name, value, lo, hi in (
            ("Seed", self.seed, -1, 2**63 - 1), ("候補数", self.candidates, 1, 8),
            ("Steps", self.steps, 1, 128), ("音声トークン上限", self.max_tokens, 200, 9000),
            ("VRAM予算", self.memory_gib, 0, 192),
        ):
            integer(value, name, lo, hi)
            if type(value) is not int:
                raise YuE2Error(f"{name}は整数で指定してください。")
        if 0 < self.memory_gib < 8:
            raise YuE2Error("VRAM予算は0（自動）、または8 GiB以上を指定してください。")
        if type(self.offload) is not bool or type(self.fp8) is not bool:
            raise YuE2Error("メモリ設定の形式が不正です。")
        if self.engine == "cpp" and self.fp8:
            raise YuE2Error("FP8 ARは公式Pythonエンジン専用です。audio.cppではGGUFを選択してください。")
        return self

    def resolved(self) -> Request:
        data = asdict(self.validate())
        data["seed"] = secrets.randbelow(2**63) if self.seed == -1 else self.seed
        return Request(**data)

    @classmethod
    def from_dict(cls, value) -> Request:
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise YuE2Error("YuE2の入力設定だけを読み込めます。実行パスや任意オプションは受け付けません。")
        try:
            return cls(**value).validate()
        except TypeError as exc:
            raise YuE2Error("プロジェクトの形式が不正です。") from exc

    def song(self, index: int = 0) -> dict:
        if self.seed < 0:
            raise YuE2Error("実行前にSeedを確定してください。")
        return {
            "style": self.style, "lyrics": self.lyrics, "cot": self.cot,
            "seed": (self.seed + index) % 2**63, "abc": self.abc.strip() or None,
        }


def import_project(path: Path) -> Request:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != 1 or "request" not in value:
        raise YuE2Error("YuE2 Studioのproject.jsonを指定してください。")
    return Request.from_dict(value["request"])


def inside(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root):
        raise YuE2Error("保存先の外側は参照できません。")
    return path


def runtime_manifest(root: Path, engine: str) -> dict:
    path = root / "runtime.json"
    if not path.is_file():
        raise YuE2Error("YuE2環境が未導入です。「実行環境」のセットアップを実行してください。")
    config = read_json(path)
    if not isinstance(config, dict) or config.get("schema") != 1:
        raise YuE2Error("runtime.jsonの形式が不正です。再セットアップしてください。")
    entry = config.get(engine)
    if not isinstance(entry, dict):
        raise YuE2Error("選択したエンジンは未導入です。セットアップ後に再確認してください。")
    return entry


def required_cpp_files(model_root: Path, quant: str) -> list[Path]:
    return [model_root / GGUF_MODELS[quant], model_root / "yue2-vae-f16.gguf",
            *(model_root / "sidecars" / name for name in SIDECARS)]


def cpp_command(binary: Path, models: Path, request: Request, out: Path, index: int) -> list[str]:
    request.validate()
    if not binary.is_file() or binary.suffix.lower() in {".bat", ".cmd", ".ps1"}:
        raise YuE2Error("audio.cppのネイティブ実行ファイルを登録してください。")
    missing = [p.name for p in required_cpp_files(models, request.gguf) if not p.is_file()]
    if missing:
        raise YuE2Error("GGUFの不足ファイル: " + ", ".join(missing))
    args = [str(binary), "--task", "gen", "--family", "yue2", "--model", str(models),
            "--backend", "cuda", "--threads", "8", "--lyrics", request.lyrics,
            "--session-option", f"yue2.model_gguf={GGUF_MODELS[request.gguf]}",
            "--session-option", "yue2.vae_gguf=yue2-vae-f16.gguf",
            "--request-option", f"style={request.style}",
            "--request-option", f"cot={request.cot}",
            "--request-option", f"num_inference_steps={request.steps}",
            "--request-option", f"semantic_max_tokens={request.max_tokens}",
            "--seed", str(request.song(index)["seed"]),
            "--out", str(out / "audio.wav"), "--out-dir", str(out), "--log"]
    if request.abc.strip():
        score = out / "input.abc"
        score.write_text(request.abc, encoding="utf-8")
        args += ["--request-option", f"abc_file={score}"]
    if len(subprocess.list2cmdline(args).encode("utf-16-le")) // 2 >= 30000:
        raise YuE2Error("Windowsのコマンド長上限に近いため、歌詞・曲調を短くしてください。")
    return args


def safe_environment() -> dict[str, str]:
    # Never pass API keys, proxy credentials, PYTHONPATH or user-site packages to inference.
    keys = ("PATH", "SystemRoot", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
            "HOME", "USERNAME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LANG", "LC_ALL",
            "CUDA_PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
    result = {k: os.environ[k] for k in keys if k in os.environ}
    result.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    return result


def runtime_lock(root: Path):
    """One OS-held lock shared by setup and inference; automatically released on crash."""
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "setup.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, 2) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock
    except OSError:
        lock.close()
        raise YuE2Error("YuE2のセットアップまたは生成が実行中です。終了後に再実行してください。") from None
