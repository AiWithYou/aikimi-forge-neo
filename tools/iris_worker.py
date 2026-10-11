"""Resident Iris entry point; keep all CUDA/model imports out of Gradio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.iris.core import atomic_json, checkpoint_identity, model_files  # noqa: E402

_RUNNER = None
_KEY = None


def resident_run(payload):
    global _RUNNER, _KEY
    from modules_forge.iris.runtime import Runner

    directory = Path(payload["job_dir"])
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    checkpoint = request.get("checkpoint", "base")
    if checkpoint != "base":
        checkpoint = str(model_files(payload["root"], request["precision"], request["task"], checkpoint)[0])
    identity = checkpoint_identity(payload["root"], request["precision"], request["task"], checkpoint)
    key = (
        payload["root"],
        request["precision"],
        request["task"],
        checkpoint == "base",
        json.dumps(identity, sort_keys=True),
    )
    atomic_json(directory / "progress.json", {"stage": "モデルを読み込み中"})
    if key != _KEY:
        if _RUNNER is not None:
            _RUNNER.close()
        _RUNNER = None
        _KEY = None
        _RUNNER = Runner(payload["root"], request["precision"], request["task"], checkpoint)
        if checkpoint_identity(payload["root"], request["precision"], request["task"], checkpoint) != identity:
            _RUNNER.close()
            _RUNNER = None
            raise ValueError("読み込み中にモデルが更新されました。再実行してください。")
        _RUNNER.source_identity = identity
        _KEY = key
    return _RUNNER.run(directory)
