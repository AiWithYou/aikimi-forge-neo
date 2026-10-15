"""Read completed, validated FLUX 3 artifacts without starting a worker."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from .core import ROOT, Flux3Error, validate_request
from .service import Service

MAX_RESULTS = 50


def _linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _json(path):
    if path.stat().st_size > 2_000_000:
        raise Flux3Error("履歴の生成条件が大きすぎます。")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Flux3Error("履歴の生成条件を読み取れません。")
    return value


class History:
    def __init__(self, root=ROOT / "outputs" / "flux3-klein"):
        self.root = Path(root).absolute()

    def _root(self):
        if not self.root.is_dir() or self.root.resolve() != self.root or _linked(self.root):
            raise Flux3Error("履歴の保存先がありません。")
        return self.root

    def load(self, identifier):
        try:
            if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
                raise Flux3Error("履歴を選択してください。")
            directory = self._root() / identifier
            if _linked(directory) or not directory.is_dir() or directory.resolve().parent != self.root:
                raise Flux3Error("履歴の保存場所が一致しません。")
            for filename in ("output.png", "request.json", "result.json", "progress.json"):
                path = directory / filename
                if _linked(path) or not path.is_file():
                    raise Flux3Error("履歴のファイルがありません。")
            if (directory / "cancel").exists() or _linked(directory / "cancel"):
                raise Flux3Error("中断された履歴です。")
            if _json(directory / "progress.json").get("stage") != "complete":
                raise Flux3Error("生成が完了していません。")
            record = _json(directory / "result.json")
            if record.get("status", "complete") != "complete":
                raise Flux3Error("生成が完了していません。")
            if (
                Path(record["image_path"]) != directory / "output.png"
                or Path(record["result_path"]) != directory / "result.json"
            ):
                raise Flux3Error("履歴の出力パスが一致しません。")
            request = _json(directory / "request.json")
            validate_request({key: value for key, value in request.items() if key != "reference_sha256"})
            reference_hashes = []
            for value in request.get("images", []):
                path = Path(value)
                if (
                    _linked(path)
                    or not path.is_file()
                    or path.parent != directory
                    or path.resolve().parent != directory
                ):
                    raise Flux3Error("履歴の参照画像の保存場所が一致しません。")
                with path.open("rb") as source:
                    reference_hashes.append(hashlib.file_digest(source, "sha256").hexdigest())
            if reference_hashes != request.get("reference_sha256", []) or reference_hashes != record.get(
                "reference_hashes", []
            ):
                raise Flux3Error("履歴の参照画像が元の入力と一致しません。")
            result = Service.validate_result(directory)
            return {
                **result,
                "id": identifier,
                "files": [str(directory / "output.png"), str(directory / "result.json")],
            }
        except (OSError, ValueError, TypeError, KeyError, IndexError, Image.DecompressionBombError) as error:
            raise Flux3Error("この履歴は未完了か、保存した画像・生成条件が一致しません。") from error

    def list(self):
        try:
            candidates = []
            for directory in self._root().iterdir():
                if re.fullmatch(r"[0-9a-f]{32}", directory.name) and not _linked(directory) and directory.is_dir():
                    candidates.append((directory.stat().st_mtime, directory.name))
        except (OSError, ValueError):
            return []
        choices = []
        for modified, identifier in sorted(candidates, reverse=True)[:MAX_RESULTS]:
            try:
                result = self.load(identifier)
            except ValueError:
                continue
            stamp = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")
            model = "Base" if result["model"] == "base" else "Klein"
            prompt = " ".join(result["input_prompt"].split())[:60]
            choices.append((f"{stamp} · {model} · seed {result['seed']} · {prompt}", identifier))
        return choices
