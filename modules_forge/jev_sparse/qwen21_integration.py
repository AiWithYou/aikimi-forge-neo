"""Select the Qwen worker and credentials per job; normal generation defaults OFF."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, replace
from pathlib import Path

from .common import REPLAY_ENV, cloud_environment, replay_requested, sdk_python
from .credentials import read_saved_key
from .qwen21 import OPTIONS_ENV, Options

ROOT = Path(__file__).resolve().parents[2]
# Reviewed 2026-09-23 Turbo BF16/GGUF, persistent quantization, I2I rewrite and mask-output worker.
WORKER_BLOB = "01cfc1df769839c13b19c9a91ac1cad3640ee5e8"


def launch_defaults() -> Options:
    return Options.parse(os.environ[OPTIONS_ENV]) if os.environ.get(OPTIONS_ENV) else Options(timeout=10)


def worker_launch(request, worker: Path, environment: dict) -> tuple[Path, dict, dict]:
    options = replace(launch_defaults(), mode=request.sparse_mode, keep_percent=request.sparse_keep_percent)
    cadence = getattr(request, "sparse_jev_cadence", "legacy")
    if cadence != "legacy" and request.sparse_mode == "jev":
        options = replace(options, decision_cadence=cadence, update_interval=request.sparse_jev_interval)
    options = replace(
        options,
        job_max_calls=getattr(request, "sparse_jev_max_calls", options.job_max_calls),
        job_max_wait_seconds=getattr(request, "sparse_jev_max_wait_seconds", options.job_max_wait_seconds),
    )
    options.validate()
    environment = dict(environment)
    for key in ("TYPESAFE_API_KEY", "AIKIMI_JEV_ALLOW_CLOUD", "AIKIMI_JEV_PYTHON", OPTIONS_ENV, REPLAY_ENV):
        environment.pop(key, None)
    if options.mode == "off":
        return worker, environment, {}
    original_worker = (ROOT / "tools/qwen_image21_worker.py").resolve()
    if Path(worker).resolve() != original_worker:
        raise ValueError("Another extension has replaced the Qwen worker")
    source = original_worker.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source, usedforsecurity=False).hexdigest()
    if digest != WORKER_BLOB:
        raise ValueError("Qwen worker differs from the reviewed revision; sparse mode is unavailable")
    if options.mode == "jev" and options.max_calls:
        if replay_requested():
            environment[REPLAY_ENV] = os.environ[REPLAY_ENV]
        else:
            key = os.environ.get("TYPESAFE_API_KEY", "").strip() or read_saved_key()
            cloud = cloud_environment({**os.environ, "TYPESAFE_API_KEY": key, "AIKIMI_JEV_ALLOW_CLOUD": "1"})
            environment.update(
                TYPESAFE_API_KEY=cloud["TYPESAFE_API_KEY"],
                AIKIMI_JEV_ALLOW_CLOUD="1",
                AIKIMI_JEV_PYTHON=str(sdk_python()),
            )
    return ROOT / "tools/qwen_image21_sparse_worker.py", environment, {"sparse_experiment": asdict(options)}
