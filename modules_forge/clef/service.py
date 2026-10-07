"""Run fixed decision jobs independently of a browser, with Forge's GPU lease."""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .bundle import bundle_manifest
from .core import (
    MODELS,
    OUTPUTS,
    PREPROCESSING_VERSION,
    PROFILES,
    ROOT,
    RUNTIME,
    ClefError,
    atomic_json,
    canonical_hash,
    export_csv,
    fingerprint,
    sha256,
    validate_request,
)


def snapshot_inputs(paths, directory):
    directory = Path(directory)
    items = []
    for index, source in enumerate(paths or []):
        record = source if isinstance(source, dict) else {}
        if record.get("kind") == "record":
            target = directory / "inputs" / f"{index:06d}.json"
            atomic_json(target, {"state": record["state"]})
            items.append(
                {
                    "id": str(index),
                    "kind": "record",
                    "name": record["name"],
                    "path": str(target.resolve()),
                    "sha256": canonical_hash(record["state"]),
                    "status": "pending",
                }
            )
            continue
        path = Path(record.get("path", source))
        if not path.is_file() or path.stat().st_size > 32 * 1024**2:
            raise ClefError(f"画像がないか、32MBを超えています: {path.name}")
        target = directory / "inputs" / f"{index:03d}{path.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        try:
            with Image.open(target) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "BMP"} or image.width * image.height > 40_000_000:
                    raise ValueError("PNG/JPEG/WebP/BMP、40メガピクセルまで対応します。")
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ClefError(f"{path.name}: 画像を読み込めません。{exc}") from exc
        digest = sha256(target)
        if record.get("sha256") and digest != record["sha256"]:
            raise ClefError(
                f"{record.get('name', path.name)}: 読み込み後に画像が変更されました。フォルダを読み直してください。"
            )
        items.append(
            {
                "id": str(index),
                "kind": "image",
                "name": record.get("name", path.name),
                "source": str(path.resolve()),
                "path": str(target.resolve()),
                "sha256": digest,
                "status": "pending",
            }
        )
    if not items:
        items.append(
            {"id": "0", "name": "文章・JSON", "path": "", "sha256": sha256_text("text-only"), "status": "pending"}
        )
    return items


def sha256_text(value):
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def process_identity():
    import psutil

    process = psutil.Process()
    return {"pid": process.pid, "created": process.create_time()}


def process_stopped(identity):
    import psutil

    if (
        not isinstance(identity, dict)
        or type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or type(identity.get("created")) not in (int, float)
        or not math.isfinite(identity["created"])
        or identity["created"] <= 0
    ):
        return False
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["created"]
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, OSError, ValueError):
        return False


def execution_stopped(directory):
    try:
        return all(
            process_stopped(json.loads((Path(directory) / name).read_text(encoding="utf-8")))
            for name in ("execution.json", "worker-process.json")
        )
    except (OSError, ValueError):
        return False


@dataclass
class Job:
    owner: str
    directory: Path
    lease: object
    done: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)
    state: dict = field(default_factory=lambda: {"status": "running", "message": "モデルを読み込み中"})
    thread: threading.Thread | None = None
    lease_held: bool = True


class Studio:
    def __init__(
        self,
        runtime=RUNTIME,
        outputs=OUTPUTS,
        *,
        ownership_factory=None,
        worker_factory=None,
        release_vram=None,
        residency=None,
    ):
        self.runtime, self.outputs = Path(runtime), Path(outputs)
        self.ownership_factory, self.worker_factory = ownership_factory, worker_factory
        self.release_vram, self.residency = release_vram, residency
        self.jobs = {}
        self.worker = None
        self.guard = threading.RLock()
        self.closed = False
        atexit.register(self.shutdown)

    def _installed(self, profile):
        bundle_manifest(self.runtime, profile)
        python = self.runtime / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise ClefError("Clefの専用Python環境がありません。aikimi-clef-setup.batを実行してください。")
        return str(python)

    def _dependencies(self):
        if self.ownership_factory is None:
            from modules_forge import gpu_residency
            from modules_forge.gpu_ownership import GPUOwnership, release_forge_vram

            self.ownership_factory, self.release_vram, self.residency = GPUOwnership, release_forge_vram, gpu_residency
        if self.worker_factory is None:
            from modules_forge.resident_worker import ResidentWorker

            self.worker_factory = ResidentWorker

    def start(self, request, paths, owner):
        request = validate_request(request)
        if not owner:
            raise ClefError("Clefタブから操作してください。")
        if not paths and (request["state"] == "" or request["state"] is None):
            raise ClefError("画像または補足入力を指定してください。")
        python = self._installed(request["profile"])
        self._dependencies()
        with self.guard:
            self._idle()
            identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            directory = self.outputs / identifier
            required = sum(
                Path(x.get("path", "")).stat().st_size
                if isinstance(x, dict) and x.get("kind") != "record"
                else Path(x).stat().st_size
                if not isinstance(x, dict)
                else 0
                for x in paths or []
            )
            self.outputs.mkdir(parents=True, exist_ok=True)
            if required + 100 * 1024**2 > shutil.disk_usage(self.outputs).free:
                raise ClefError("画像の保存先に十分な空き容量がありません。")
            directory.mkdir(exist_ok=False)
            try:
                items = snapshot_inputs(paths, directory)
                profile = PROFILES[request["profile"]]
                result = {
                    "id": identifier,
                    "created_at": datetime.now(UTC).isoformat(),
                    "status": "running",
                    "request": request,
                    "model": MODELS[profile["model"]],
                    "profile": profile,
                    "preprocessing": PREPROCESSING_VERSION,
                    "fingerprint": fingerprint(request, items),
                    "items": items,
                }
                return self._launch(result, directory, python, owner)
            except BaseException:
                target = directory.resolve()
                if (
                    identifier not in self.jobs
                    and target.parent == self.outputs.resolve()
                    and target.name == identifier
                ):
                    shutil.rmtree(target)
                raise

    def _idle(self):
        if self.closed:
            raise ClefError("Clefは終了処理中です。再起動してください。")
        if any(not job.done.is_set() or job.lease_held for job in self.jobs.values()):
            raise ClefError("Clefが実行中です。終了または停止を待ってください。")

    def _launch(self, result, directory, python, owner):
        lease = self.ownership_factory()
        lease.engine = "clef"
        if not lease.acquire(False):
            raise ClefError("GPUを使用中です。現在の処理が終了してから判定してください。")
        identifier = result["id"]
        job = None
        try:
            for name in ("cancel", "progress.json", "loading.json", "worker-process.json"):
                (directory / name).unlink(missing_ok=True)
            atomic_json(directory / "execution.json", process_identity())
            atomic_json(directory / "request.json", result["request"])
            atomic_json(directory / "result.json", result)
            job = Job(owner, directory, lease)
            self.jobs[identifier] = job
            job.thread = threading.Thread(target=self._run, args=(job, python), daemon=True)
            job.thread.start()
        except BaseException as exc:
            if job is not None and job.thread is not None and job.thread.is_alive():
                job.cancel.set()
            elif job is not None:
                job.state = {"status": "failed", "message": str(exc)}
                try:
                    result["status"] = "failed"
                    for item in result["items"]:
                        if item["status"] == "pending":
                            item.update(status="failed", error=str(exc))
                    atomic_json(directory / "result.json", result)
                    atomic_json(directory / "status.json", job.state)
                finally:
                    try:
                        lease.release()
                        job.lease_held = False
                    finally:
                        job.done.set()
            else:
                lease.release()
            raise
        return identifier

    def resume(self, identifier, owner, retry_errors=False):
        if not owner or not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", identifier or ""):
            raise ClefError("再開する判定記録を選択してください。")
        with self.guard:
            self._idle()
            directory = (self.outputs / identifier).resolve()
            if not directory.is_relative_to(self.outputs.resolve()):
                raise ClefError("判定記録の保存先が不正です。")
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if result.get("preprocessing") != PREPROCESSING_VERSION:
                raise ClefError("画像の前処理が更新されています。この記録を再開せず、新しい判定を開始してください。")
            if result["status"] == "running" and not execution_stopped(directory):
                raise ClefError("この記録は実行中です。停止を確認してから再開してください。")
            request = validate_request(result["request"])
            for item in result["items"]:
                if not item["path"]:
                    continue
                path = Path(item["path"]).resolve()
                if not path.is_relative_to((directory / "inputs").resolve()) or not path.is_file():
                    raise ClefError("再開に必要な保存入力がありません。")
                digest = (
                    canonical_hash(json.loads(path.read_text(encoding="utf-8"))["state"])
                    if item.get("kind") == "record"
                    else sha256(path)
                )
                if digest != item["sha256"]:
                    raise ClefError("保存入力が変更されています。再開できません。")
            if fingerprint(request, result["items"]) != result["fingerprint"]:
                raise ClefError("保存した実行条件が変更されています。再開できません。")
            targets = {"pending", "cancelled", "failed"} | ({"error"} if retry_errors else set())
            if not any(item["status"] in targets for item in result["items"]):
                raise ClefError("未完了の項目がありません。必要ならエラーの再試行を選択してください。")
            for item in result["items"]:
                if item["status"] in targets:
                    item["status"] = "pending"
                    item.pop("error", None)
            result["status"] = "running"
            python = self._installed(request["profile"])
            self._dependencies()
            return self._launch(result, directory, python, owner)

    def _run(self, job, python):
        success = False
        try:
            if job.cancel.is_set():
                raise InterruptedError("開始前に停止しました。")
            if self.release_vram:
                self.release_vram()
            environment = dict(os.environ)
            environment.update(
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1", HF_HUB_DISABLE_TELEMETRY="1"
            )
            with self.guard:
                if job.cancel.is_set():
                    raise InterruptedError("開始前に停止しました。")
                if self.worker is None:
                    self.worker = self.worker_factory("clef", self.runtime / "sessions")
                worker = self.worker
                worker.start(
                    python,
                    ROOT / "tools/clef_worker.py",
                    environment,
                    {"root": str(self.runtime.resolve()), "job_dir": str(job.directory.resolve())},
                    job.directory / "worker.log",
                )
            while True:
                if job.cancel.is_set():
                    raise InterruptedError("停止しました。完了分の結果は保存しました。")
                response = worker.result()
                if response is not None:
                    if not response.get("ok"):
                        raise ClefError(response.get("error", "workerでエラーが発生しました。"))
                    break
                time.sleep(0.15)
            result = json.loads((job.directory / "result.json").read_text(encoding="utf-8"))
            job.state = {
                "status": result["status"],
                "message": "判定完了" if result["status"] == "complete" else "一部の画像でエラーがありました。",
            }
            cache = result.get("quantized_cache", {})
            if cache.get("status") == "error":
                job.state["message"] += " 量子化重みの保存エラー: " + cache.get("error", "")
            if self.residency:
                self.residency.register("clef", self._close_worker, "clef")
            success = True
        except BaseException as exc:
            job.state = {"status": "cancelled" if job.cancel.is_set() else "failed", "message": str(exc)}
        finally:
            stopped = success
            if not success:
                try:
                    self._close_worker()
                    stopped = True
                except Exception as exc:
                    job.state["message"] += f" プロセスの停止未確認: {exc}"
            if stopped:
                try:
                    if not success:
                        path = job.directory / "result.json"
                        result = json.loads(path.read_text(encoding="utf-8"))
                        for item in result["items"]:
                            if item["status"] == "pending":
                                item["status"] = job.state["status"]
                        result["status"] = (
                            "partial" if any(x["status"] == "done" for x in result["items"]) else job.state["status"]
                        )
                        atomic_json(path, result)
                        (job.directory / "results.csv").write_text(export_csv(result), encoding="utf-8-sig")
                except (OSError, ValueError) as exc:
                    job.state["message"] += f" 結果の保存エラー: {exc}"
                try:
                    job.lease.release()
                    job.lease_held = False
                except Exception as exc:
                    job.state["message"] += f" GPU解放の確認が必要です: {exc}"
            try:
                atomic_json(job.directory / "status.json", job.state)
            finally:
                job.done.set()

    def _close_worker(self):
        with self.guard:
            if self.worker is not None:
                self.worker.close()
                self.worker = None

    def _job(self, identifier, owner):
        job = self.jobs.get(identifier)
        if job is None or job.owner != owner:
            raise ClefError("このブラウザーが所有する判定記録ではありません。")
        return job

    def status(self, identifier, owner):
        job = self._job(identifier, owner)
        done = job.done.is_set()
        result = json.loads((job.directory / "result.json").read_text(encoding="utf-8"))
        state = dict(job.state)
        if not done:
            try:
                message = json.loads((job.directory / "progress.json").read_text(encoding="utf-8")).get("message")
                if isinstance(message, str):
                    state["message"] = message
            except (OSError, ValueError):
                pass
        return {**state, "done": done, "result": result, "directory": str(job.directory)}

    def cancel(self, identifier, owner):
        job = self._job(identifier, owner)
        if not job.done.is_set():
            job.cancel.set()
            (job.directory / "cancel").touch()
            return True
        return False

    def unload(self):
        from modules_forge.gpu_ownership import queue_lock
        from modules_forge.gpu_residency import engine_scope, release_resource

        with engine_scope(None):
            if not queue_lock.acquire(False):
                return "GPUを使用中です。終了後に解放してください。"
            try:
                release_resource("clef")
                self._close_worker()
                return "Clefモデルを解放しました。"
            finally:
                queue_lock.release()

    def shutdown(self):
        with self.guard:
            self.closed = True
            jobs = list(self.jobs.values())
            for job in jobs:
                if not job.done.is_set():
                    job.cancel.set()
        for job in jobs:
            if job.thread is not None and job.thread != threading.current_thread() and job.thread.is_alive():
                job.thread.join(30)
                if job.thread.is_alive():
                    raise RuntimeError("Clefの停止処理が未完了です。GPUは保持しています。")
        self._close_worker()
        for job in jobs:
            if job.lease_held:
                job.lease.release()
                job.lease_held = False


STUDIO = Studio()
