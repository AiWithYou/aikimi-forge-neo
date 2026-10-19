"""MOSS jobs retain the shared GPU lease until their worker is idle or stopped."""

from __future__ import annotations

import json
import math
import os
import stat
import threading
import time
import traceback
import uuid
from pathlib import Path

from PIL import Image

from . import checkpoints, lora, references
from .core import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_ROOT,
    ROOT,
    MossImageError,
    atomic_json,
    model_ready,
    python_path,
    validate_request,
)


def _managed_path(value):
    path = Path(value).absolute()
    for entry in (path, *path.parents):
        try:
            attributes = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_reparse_tag", 0) & 0x20000000:
            raise MossImageError("MOSS Imageの保存先にリンクは使用できません。")
    return path.resolve(strict=False)


def _log_error(directory, error):
    try:
        with (directory / "run.log").open("a", encoding="utf-8") as stream:
            traceback.print_exception(error, file=stream)
    except OSError:
        pass


class Service:
    def __init__(self, root=MODEL_ROOT, *, outputs=None):
        from modules_forge.resident_worker import ResidentWorker

        self.root = Path(root)
        self.outputs = Path(outputs) if outputs is not None else ROOT / "outputs" / "moss-image"
        self.worker = ResidentWorker("moss-image", ROOT / "cache" / "moss-image-worker")
        self.jobs = {}
        self._guard = threading.RLock()

    def start(self, value):
        from modules_forge.gpu_ownership import GPUOwnership

        request = validate_request(value)
        checkpoints.bind(self.root, request)
        lora.bind(self.root, request)
        outputs = _managed_path(self.outputs)
        if not model_ready(self.root, request):
            raise MossImageError(
                "選択した形式をセットアップしてください: aikimi-moss-image-setup.bat "
                f"--precision {request['precision']} --text-precision {request['text_precision']}"
            )
        lease = GPUOwnership()
        lease.engine = "moss-image"
        if not lease.acquire(False):
            raise MossImageError("GPUを使用中です。処理の終了後に実行してください。")
        identifier = uuid.uuid4().hex
        directory = outputs / identifier
        try:
            directory.mkdir(parents=True)
            references.snapshot(directory, request)
            atomic_json(directory / "request.json", request)
            job = {"id": identifier, "directory": str(directory), "status": "running", "error": ""}
            with self._guard:
                self.jobs[identifier] = job
            threading.Thread(target=self.execute, args=(job, lease), daemon=True).start()
        except BaseException:
            with self._guard:
                self.jobs.pop(identifier, None)
            lease.release()
            raise
        return identifier

    @staticmethod
    def validate_result(directory):
        directory = _managed_path(directory)
        result = json.loads(_managed_path(directory / "result.json").read_text(encoding="utf-8"))
        request = json.loads(_managed_path(directory / "request.json").read_text(encoding="utf-8"))
        references.load(directory, request)
        path = _managed_path(result["image_path"])
        if path != directory / "output.png" or not path.is_file():
            raise MossImageError("生成結果の出力パスが一致しません。")
        if any(result.get(key) != value for key, value in request.items()):
            raise MossImageError("生成条件が元の要求と一致しません。")
        if result.get("model_id") != MODEL_ID or result.get("model_revision") != MODEL_REVISION:
            raise MossImageError("生成モデルが元の要求と一致しません。")
        expected_lora = lora.record(request)
        if result.get("lora") != expected_lora or result.get("lora_applied_targets") != sum(
            item["applied_targets"] for item in expected_lora
        ):
            raise MossImageError("LoRAの適用結果が元の要求と一致しません。")
        elapsed = result.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise MossImageError("生成時間の記録が不正です。")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise MossImageError("生成結果がPNGではありません。")
            if image.size != (request["width"], request["height"]):
                raise MossImageError("生成結果のサイズが生成条件と一致しません。")
            embedded = image.info.get("moss_image")
        if not embedded:
            raise MossImageError("PNG内の生成条件がありません。")
        metadata = json.loads(embedded)
        required = {*request, "model_id", "model_revision", "elapsed_seconds", "lora", "lora_applied_targets"}
        if (
            not isinstance(metadata, dict)
            or not required.issubset(metadata)
            or any(result.get(key) != value for key, value in metadata.items())
            or any(key in {"root", "directory", "files"} or key.endswith("_path") for key in metadata)
        ):
            raise MossImageError("PNG内の生成条件が保存記録と一致しません。")
        return result

    def execute(self, job, lease):
        from modules_forge import gpu_residency
        from modules_forge.gpu_ownership import release_forge_vram

        directory = Path(job["directory"])
        environment = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONIOENCODING="utf-8",
            PYTHONDONTWRITEBYTECODE="1",
        )
        status = "error"
        try:
            if (directory / "cancel").exists():
                status = "cancelled"
                return
            release_forge_vram()
            self.worker.start(
                python_path(self.root),
                ROOT / "tools" / "moss_image_worker.py",
                environment,
                {"root": str(self.root), "job_dir": str(directory)},
                directory / "run.log",
            )
            gpu_residency.register("moss-image", self.worker.close, "moss-image")
            while True:
                if (directory / "cancel").exists():
                    status = "cancelled"
                    break
                response = self.worker.result()
                if response is not None:
                    if not response["ok"]:
                        raise RuntimeError(response.get("error", "MOSS Imageの生成に失敗しました。"))
                    self.validate_result(directory)
                    with self._guard:
                        status = "cancelled" if (directory / "cancel").exists() else "complete"
                        job["_finishing"] = True
                    break
                time.sleep(0.1)
        except Exception as exc:
            _log_error(directory, exc)
            job["error"] = (
                str(exc)
                if isinstance(exc, MossImageError)
                else "生成に失敗しました。Forge Neoのログを確認してください。"
            )
        finally:
            with self._guard:
                job["_finishing"] = True
            if status != "complete":
                while True:
                    try:
                        self.worker.close()
                        break
                    except Exception as exc:
                        _log_error(directory, exc)
                        with self._guard:
                            job.update(status="stopping", error="停止を確認中…")
                        time.sleep(0.5)
                for filename in ("output.png", ".output.png.tmp"):
                    try:
                        (directory / filename).unlink(missing_ok=True)
                    except OSError as exc:
                        _log_error(directory, exc)
            try:
                if status != "complete":
                    atomic_json(directory / "result.json", {"status": status, "error": job["error"]})
            finally:
                try:
                    lease.release()
                finally:
                    with self._guard:
                        job["status"] = status

    def poll(self, identifier):
        with self._guard:
            job = self.jobs.get(identifier)
            if job is None:
                return {"status": "idle"}
            result = {key: value for key, value in job.items() if not key.startswith("_")}
        status = result["status"]
        directory = Path(result["directory"])
        terminal = status in {"complete", "error", "cancelled"}
        try:
            record = json.loads(
                (directory / ("result.json" if terminal else "progress.json")).read_text(encoding="utf-8")
            )
            result.update(
                record if terminal else {key: record[key] for key in ("stage", "current", "total") if key in record}
            )
        except (OSError, ValueError):
            pass
        result["status"] = status
        if status == "complete":
            result["files"] = [str(directory / "output.png"), str(directory / "result.json")]
        return result

    def cancel(self, identifier):
        with self._guard:
            job = self.jobs.get(identifier)
            if job and job["status"] in {"running", "stopping"} and not job.get("_finishing"):
                (Path(job["directory"]) / "cancel").touch()
                return "中断処理中…"
        return "実行中の処理はありません。"


SERVICE = Service()
