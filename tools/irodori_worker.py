"""Run the pinned official inference code only inside its dedicated Python."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.irodori.core import (  # noqa: E402
    CODE_REVISION,
    CODEC_REVISION,
    MODEL_REVISION,
    WATERMARK_REVISION,
    atomic_json,
    sha256,
    validate_request,
)

_STATE = {"key": None, "runtime": None}


def cached_runtime(key, factory):
    if _STATE["runtime"] is None:
        runtime = factory(key)
        _STATE.update(key=key, runtime=runtime)
    elif key != _STATE["key"]:
        raise ValueError("モデル設定の変更にはworkerの再起動が必要です。")
    return _STATE["runtime"]


@contextmanager
def local_watermark(directory):
    """Select local weights through SilentCipher's public loader arguments."""
    import silentcipher

    checkpoint = Path(directory) / "44_1_khz/73999_iteration"
    original = silentcipher.get_model
    silentcipher.get_model = partial(original, ckpt_path=str(checkpoint), config_path=str(checkpoint / "hparams.yaml"))
    try:
        yield
    finally:
        silentcipher.get_model = original


def check_token_lengths(runtime, request):
    for text, tokenizer, limit, label in (
        (request["text"], runtime.tokenizer, runtime.default_text_max_len, "読み上げ文"),
        (request["caption"], runtime.caption_tokenizer, runtime.default_caption_max_len, "声の説明"),
    ):
        if text and tokenizer is not None:
            count = tokenizer.encode(text).numel()
            if count > limit:
                raise ValueError(f"{label}が長すぎます（{count} / {limit} tokens）。短く分けてください。")


def resident_run(payload):
    import numpy as np
    import torch

    source = Path(payload["root"]) / "source"
    sys.path.insert(0, str(source))
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav

    directory, models = Path(payload["job_dir"]), Path(payload["models"])
    original = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    request = validate_request(original)
    references = original.get("references", [])
    paths = []
    for record in references:
        path = (directory / record["file"]).resolve()
        if path.parent != directory.resolve() or sha256(path) != record["sha256"]:
            raise ValueError("参照音声が生成前に変更されました。")
        paths.append(str(path))
    if request["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("このGPUはBF16に対応していません。FP32を選んでください。")
    started = time.monotonic()
    with local_watermark(models / "watermark"):
        runtime = cached_runtime(
            RuntimeKey(
                checkpoint=str(models / "v4-Large/model.safetensors"),
                model_device="cuda",
                codec_repo=str(models / "codec/weights.pth"),
                model_precision=request["precision"],
                codec_device="cuda",
                codec_precision="fp32",
                compile_model=False,
            ),
            InferenceRuntime.from_key,
        )
    check_token_lengths(runtime, request)
    sampling = SamplingRequest(
        text=request["text"],
        caption=request["caption"],
        ref_wavs=paths or None,
        no_ref=not paths,
        max_ref_seconds=120,
        seconds=request["seconds"],
        duration_scale=request["duration_scale"],
        num_steps=request["steps"],
        cfg_scale_text=request["cfg_text"],
        cfg_scale_caption=request["cfg_caption"],
        cfg_scale_speaker=request["cfg_speaker"],
        seed=request["seed"],
    )
    result = runtime.synthesize(sampling, log_fn=None)
    torch.cuda.synchronize()
    samples = result.audio.detach().float().cpu().numpy().squeeze()
    if samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError("音声が空、無音、または不正な値を含んでいます。")
    if result.sample_rate != 48000 or len(samples) > 31 * 48000:
        raise ValueError("音声のサンプルレートまたは長さが不正です。")
    temporary = directory / "audio.pending.wav"
    save_wav(temporary, result.audio, result.sample_rate)
    os.replace(temporary, directory / "audio.wav")
    import soundfile as sf

    saved, saved_rate = sf.read(directory / "audio.wav", dtype="float32")
    if saved_rate != result.sample_rate or len(saved) != len(samples):
        raise ValueError("保存した音声が生成結果と一致しません。")
    effective = dataclasses.asdict(sampling)
    effective["ref_wavs"] = [x["file"] for x in references] or None
    atomic_json(
        directory / "result.json",
        {
            "job_id": directory.name,
            "seed": result.used_seed,
            "sample_rate": result.sample_rate,
            "samples": len(samples),
            "seconds": len(samples) / result.sample_rate,
            "model_revision": MODEL_REVISION,
            "code_revision": CODE_REVISION,
            "codec_revision": CODEC_REVISION,
            "watermarked": bool(runtime.watermarker.ready),
            "watermark_revision": WATERMARK_REVISION,
            "model_precision": request["precision"],
            "codec_precision": "fp32",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "parameters": effective,
            "elapsed_seconds": time.monotonic() - started,
            "stage_timings": result.stage_timings,
            "peak": float(np.max(np.abs(saved))),
            "rms": float(np.sqrt(np.mean(saved**2))),
            "messages": result.messages,
        },
    )


if __name__ == "__main__":
    raise SystemExit("Use the Irodori TTS tab to run the managed worker.")
