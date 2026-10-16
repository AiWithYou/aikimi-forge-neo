"""Resident Iris entry point; keep all CUDA/model imports out of Gradio."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.iris.core import atomic_json, checkpoint_identity, model_files  # noqa: E402

_STATE = SimpleNamespace(runner=None, key=None)


def resident_run(payload):
    from modules_forge.iris.runtime import Runner

    directory = Path(payload["job_dir"])
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    checkpoint = request.get("checkpoint", "base")
    if checkpoint != "base":
        checkpoint = str(model_files(payload["root"], request["precision"], request["task"], checkpoint)[0])
    identity = checkpoint_identity(payload["root"], request["precision"], request["task"], checkpoint)
    from modules_forge.iris.lora import sampling_settings, verify_base, verify_sources
    from modules_forge.iris.official_turbo import checkpoint_settings, combine_settings

    loras = request.get("loras", [])
    verify_sources(loras)
    verify_base(loras, identity, request["precision"])
    sampling = combine_settings(
        checkpoint_settings(payload["root"], checkpoint),
        sampling_settings(loras, request.get("lora_sampling", "standard"), request.get("turbo_shift", 4.0)),
    )
    if request.get("sampling", {"mode": "standard"}) != sampling:
        raise ValueError("チェックポイント・LoRAと生成方式が一致しません。再実行してください。")
    if sampling["mode"] == "official_turbo":
        if request.get("image_mode") == "img2img":
            raise ValueError("公式Turboはimg2imgに対応していません。")
        if (
            request.get("steps") != len(sampling["sigmas"])
            or request.get("cfg") != 1
            or request.get("negative_prompt") != ""
        ):
            raise ValueError("公式Turboの固定生成条件が一致しません。再実行してください。")
    if sampling["mode"] == "turbo":
        if request.get("image_mode") == "img2img":
            raise ValueError("img2imgではサンプリングを通常にしてください。コミュニティTurbo方式は未対応です。")
        steps, cfg = request.get("steps"), request.get("cfg")
        if type(steps) is not int or not 1 <= steps <= 200:
            raise ValueError("TurboのStepsは1〜200の整数で指定してください。")
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not 1 <= cfg <= 15 or not math.isfinite(cfg):
            raise ValueError("TurboのCFGは1〜15の有限値で指定してください。")
        if not isinstance(request.get("negative_prompt"), str):
            raise ValueError("TurboのNegative Promptは文字列で指定してください。")
    key = (
        payload["root"],
        request["precision"],
        request["task"],
        checkpoint == "base",
        json.dumps(identity, sort_keys=True),
        json.dumps(loras, sort_keys=True),
    )
    atomic_json(directory / "progress.json", {"stage": "モデルを読み込み中"})
    if key != _STATE.key:
        if _STATE.runner is not None:
            _STATE.runner.close()
        _STATE.runner = None
        _STATE.key = None
        _STATE.runner = Runner(payload["root"], request["precision"], request["task"], checkpoint)
        try:
            _STATE.runner.set_loras(loras)
        except BaseException:
            _STATE.runner.close()
            _STATE.runner = None
            raise
        if checkpoint_identity(payload["root"], request["precision"], request["task"], checkpoint) != identity:
            _STATE.runner.close()
            _STATE.runner = None
            raise ValueError("読み込み中にモデルが更新されました。再実行してください。")
        _STATE.runner.source_identity = identity
        _STATE.key = key
    try:
        return _STATE.runner.run(directory, request=request)
    except BaseException:
        _STATE.runner.close()
        _STATE.runner = None
        _STATE.key = None
        raise
