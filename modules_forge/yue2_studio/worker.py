"""Pinned YuE2 runtime entry point. Invoked only by the protected supervisor."""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules_forge.yue2_studio.core import (  # noqa: E402
    Request, YuE2Error, atomic_json, cpp_command, read_json, runtime_manifest, safe_environment,
)


def parent_guard():
    if sys.stdin.buffer.readline() != b"GO\n":
        raise YuE2Error("YuE2 Studioの監督プロセスから起動してください。")

    def watch():
        # Raw reads avoid a buffered-stdin daemon lock during normal interpreter shutdown.
        while os.read(sys.stdin.fileno(), 1):
            pass
        # The supervisor owns the Windows Job Object; on POSIX we are its session leader.
        if os.name != "nt" and os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(130)

    threading.Thread(target=watch, daemon=True).start()


def native(request: Request, entry: dict, directory: Path, plan_only: bool, report, cancelled):
    import torch
    from yue2 import YuE2Pipeline
    from yue2.pipeline import SongResult
    from yue2.protocol import GenerationConfig, SongRequest
    from yue2.storage import identity

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise YuE2Error("CUDA/BF16対応GPUが見つかりません。専用PythonのPyTorchとドライバーを確認してください。")
    if request.fp8 and torch.cuda.get_device_capability() < (8, 9):
        raise YuE2Error("FP8 ARにはCompute Capability 8.9以上が必要です。RTX 3090ではオフにしてください。")
    total = math.ceil(torch.cuda.get_device_properties(0).total_memory / 2**30)
    budget = min(request.memory_gib or total, total)
    config = GenerationConfig.from_dict({"ode_steps": request.steps, "semantic": {"max_tokens": request.max_tokens}})
    report("モデルを検証・読み込み中（初回は時間がかかります）")
    with YuE2Pipeline.from_pretrained(
        entry["model"], vae=entry["vae"], device="cuda", backend="torch-eager",
        generation_config=config, memory_budget_gib=budget, offload_ar=request.offload,
        quantization="fp8" if request.fp8 else "none", local_files_only=True,
    ) as pipe:
        for index in range(request.candidates):
            if cancelled():
                raise InterruptedError("停止しました。")
            out = directory / f".take-{index + 1}.partial"
            out.mkdir(exist_ok=False)
            song_request = SongRequest(**request.song(index))
            start = time.perf_counter()
            report(f"候補 {index + 1}/{request.candidates} · 楽譜を準備中")
            plan = pipe.plan(request=song_request, cancelled=cancelled)
            plan.save(out)
            if plan_only:
                truncated = bool(plan.truncated)
            else:
                report(f"候補 {index + 1}/{request.candidates} · 歌と演奏の構成を生成中")
                semantic = pipe.generate_semantic(plan, cancelled=cancelled)
                report(f"候補 {index + 1}/{request.candidates} · 音響を合成中")
                nar_start = time.perf_counter()
                latents = pipe.synthesize(semantic, cancelled=cancelled)
                nar_seconds = time.perf_counter() - nar_start
                if cancelled():
                    raise InterruptedError("停止しました。")
                report(f"候補 {index + 1}/{request.candidates} · 音声を復号中")
                vae_start = time.perf_counter()
                audio = pipe.decode(latents)
                effective = pipe.effective_config(song_request)
                timings = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                           "vae_seconds": time.perf_counter() - vae_start, "load": dict(pipe.load_timing),
                           "e2e_seconds": time.perf_counter() - start}
                result = SongResult(audio, 48000, semantic, latents, effective, pipe.weights, timings,
                                    identity({"request": song_request.to_dict(), "config": effective, "weights": pipe.weights}))
                result.save_artifacts(out)
                result.save(out / "audio.wav")
                truncated = any(result.truncated.values())
            if cancelled():
                raise InterruptedError("停止しました。")
            finish_take(out, request, index, truncated, plan_only, entry)


def finish_take(out: Path, request: Request, index: int, truncated, plan_only: bool, entry: dict):
    effective = replace(request, seed=request.song(index)["seed"], candidates=1)
    atomic_json(out / "project.json", {"schema": 1, "request": asdict(effective)})
    atomic_json(out / "studio-result.json", {
        "state": "complete", "title": request.title, "seed": effective.seed,
        "engine": request.engine, "plan_only": plan_only, "truncated": truncated,
        "runtime": entry, "validation": "GPU quality not certified by Aikimi Studio",
    })
    final = out.parent / f"take-{index + 1}"
    if final.exists():
        raise FileExistsError("既存の候補を上書きしません。")
    out.rename(final)


def cpp(request: Request, entry: dict, directory: Path, plan_only: bool, report, cancelled):
    if plan_only:
        raise YuE2Error("audio.cppでは楽譜のみの生成に対応していません。")
    for index in range(request.candidates):
        if cancelled():
            raise InterruptedError("停止しました。")
        out = directory / f".take-{index + 1}.partial"
        out.mkdir(exist_ok=False)
        command = cpp_command(Path(entry["binary"]), Path(entry["models"]), request, out, index)
        atomic_json(out / "command.json", {"argv": command})
        report(f"候補 {index + 1}/{request.candidates} · audio.cppで生成中（詳細はログ）")
        process = subprocess.Popen(command, env=safe_environment(), cwd=Path(entry["binary"]).parent,
                                   stdin=subprocess.DEVNULL, close_fds=True)
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError("停止しました。")
                time.sleep(0.2)
            if process.returncode:
                raise YuE2Error(f"audio.cppが終了コード {process.returncode} で停止しました。worker.logを確認してください。")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        audio = out / "audio.wav"
        if not audio.is_file() or audio.stat().st_size < 44:
            raise YuE2Error("audio.cppから有効なWAV出力が得られませんでした。")
        with audio.open("rb") as stream:
            header = stream.read(12)
        if header[:4] not in {b"RIFF", b"RF64"} or header[8:12] != b"WAVE":
            raise YuE2Error("出力がWAV形式ではありません。")
        # v0.8.0 does not expose a reliable truncation flag. Never label it false.
        finish_take(out, request, index, None, False, entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    parent_guard()
    directory = args.job.resolve()

    def report(message, state="running"):
        atomic_json(directory / "status.json", {"state": state, "message": message})

    try:
        project = read_json(directory / "project.json")
        request = Request.from_dict(project["request"])
        if request.seed < 0:
            raise YuE2Error("Seedが未確定です。")
        entry = runtime_manifest(args.runtime, request.engine)
        cancelled = lambda: (directory / "cancel").exists()
        runner = native if request.engine == "official" else cpp
        runner(request, entry, directory, project.get("plan_only", False), report, cancelled)
        report("完了しました。履歴から候補を試聴・比較できます。", "complete")
        return 0
    except InterruptedError as exc:
        report(str(exc), "cancelled")
        return 130
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        if "out of memory" in message.lower():
            message = "VRAM不足です。他のGPUアプリを終了し、CPU退避を有効にしてください。16GB環境は実験対応です。"
        report(message, "failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
