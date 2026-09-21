"""Select the Qwen worker and credentials per job; normal generation defaults OFF."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, replace
from pathlib import Path

from .common import cloud_environment, sdk_python
from .credentials import read_saved_key
from .qwen21 import OPTIONS_ENV, Options

ROOT = Path(__file__).resolve().parents[2]
# Reviewed v1.3.0 W4A8/prompt-rewriter worker; projection and cache contracts match.
WORKER_BLOB = "96c1d349d879460861ece114a93cc15446b7fca5"


def launch_defaults() -> Options:
    return Options.parse(os.environ[OPTIONS_ENV]) if os.environ.get(OPTIONS_ENV) else Options(timeout=10)


def worker_launch(request, worker: Path, environment: dict) -> tuple[Path, dict, dict]:
    options = replace(launch_defaults(), mode=request.sparse_mode, keep_percent=request.sparse_keep_percent)
    options.validate()
    environment = dict(environment)
    for key in ("TYPESAFE_API_KEY", "AIKIMI_JEV_ALLOW_CLOUD", "AIKIMI_JEV_PYTHON", OPTIONS_ENV):
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
        key = os.environ.get("TYPESAFE_API_KEY", "").strip() or read_saved_key()
        cloud = cloud_environment({**os.environ, "TYPESAFE_API_KEY": key, "AIKIMI_JEV_ALLOW_CLOUD": "1"})
        environment.update(
            TYPESAFE_API_KEY=cloud["TYPESAFE_API_KEY"], AIKIMI_JEV_ALLOW_CLOUD="1", AIKIMI_JEV_PYTHON=str(sdk_python())
        )
    return ROOT / "tools/qwen_image21_sparse_worker.py", environment, {"sparse_experiment": asdict(options)}
