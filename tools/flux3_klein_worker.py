"""Isolated, managed Comfy inference without an additional HTTP server."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_runner = None
_root = None


def resident_run(payload):
    global _runner, _root

    from modules_forge.flux3_klein.inference import Runner

    root = Path(payload["root"]).resolve()
    directory = Path(payload["job_dir"]).resolve()
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if _runner is not None and root != _root:
        _runner.close()
        _runner = None
    if _runner is None:
        _runner = Runner(root)
        _root = root
    try:
        return _runner.run(directory, request)
    except BaseException:
        _runner.close()
        _runner = None
        raise
