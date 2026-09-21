"""Opt-in worker routing for the existing Qwen 2.1 Studio, not Forge txt2img.

Settings are immutable for one launch session. Normal launches are untouched.
The existing service still owns inputs, annotations, GPU leases and cancellation.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .common import cloud_environment, sdk_python
from .qwen21 import OPTIONS_ENV, Options

ROOT = Path(__file__).resolve().parents[2]
WORKER_BLOB = "6138218ab2ec1343bfed083b9f4efc63bbfb167f"
_RESTORES = []


def install():
    if _RESTORES or not os.environ.get(OPTIONS_ENV):
        return
    from modules_forge.qwen_image21 import service
    original_start = service.Studio.start
    state = {"error": None}
    def guarded_start(self, request, owner):
        if state["error"]:
            raise service.QwenImage21Error("Qwen 2.1 Sparse設定を適用できません: " + state["error"])
        return original_start(self, request, owner)
    # If callback errors are swallowed by Forge, selected experiments MUST NOT
    # silently run the normal model and report an apparent benchmark success.
    _replace(service.Studio, "start", guarded_start)
    try:
        options = Options.parse(os.environ[OPTIONS_ENV])
        original_worker = ROOT / "tools/qwen_image21_worker.py"
        if Path(service.WORKER).resolve() != original_worker:
            raise ValueError("Another extension has replaced the Qwen worker")
        source = original_worker.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source, usedforsecurity=False).hexdigest()
        if digest != WORKER_BLOB:
            raise ValueError("Qwen worker differs from the reviewed revision; refusing automatic patching")
        encoded = json.dumps(asdict(options), allow_nan=False)
        original_environment = service.safe_environment
        credentials = {}
        if options.mode == "jev" and options.max_calls:
            allowed = cloud_environment()
            credentials = {"TYPESAFE_API_KEY": allowed["TYPESAFE_API_KEY"], "AIKIMI_JEV_ALLOW_CLOUD": "1", "AIKIMI_JEV_PYTHON": str(sdk_python())}
        def environment():
            result = original_environment()
            # Clear any accidentally inherited controls/credentials in fixed runs.
            for key in ("TYPESAFE_API_KEY", "AIKIMI_JEV_ALLOW_CLOUD", "AIKIMI_JEV_PYTHON", OPTIONS_ENV):
                result.pop(key, None)
            result[OPTIONS_ENV] = encoded
            result.update(credentials)
            return result
        _replace(service, "safe_environment", environment)
        _replace(service, "WORKER", ROOT / "tools/qwen_image21_sparse_worker.py")
        print(f"[Qwen 2.1 Sparse] mode={options.mode}; target keep={options.keep_percent}%; backend=block-gather SDPA; restart Neo to change modes")
    except Exception as exc:
        # Only exception class plus our own validation messages; no provider body.
        state["error"] = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
        print("[Qwen 2.1 Sparse] configuration rejected; Qwen generation is blocked")


def _replace(obj, name, value):
    old = getattr(obj, name)
    _RESTORES.append((obj, name, old, value))
    setattr(obj, name, value)


def uninstall():
    while _RESTORES:
        obj, name, old, new = _RESTORES.pop()
        if getattr(obj, name) is new:
            setattr(obj, name, old)
