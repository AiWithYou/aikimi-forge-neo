"""Run a reproducible Krea2 Local Supersample Detail API experiment.

The script keeps the image payload and full prompt out of stdout, records GPU-wide
telemetry while Forge processes the request, and saves the returned PNG bytes
without re-encoding so all Forge PNG text chunks remain intact.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from PIL import Image
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.krea2_local_supersample import (
    LINEAR_LUMA_WEIGHTS,
    LOCAL_SUPERSAMPLE_PROFILES,
    MODE_FOCUSED_ROI_REWRITE,
    MODE_FULL_IMAGE_GRID,
    MODES,
    PROFILE_SAFE_1536,
    focused_roi_difference_metrics,
    get_profile,
    parse_roi_boxes,
    plan_focused_rois,
    plan_local_tiles,
    rgb_sha256,
    select_tiles_for_rois,
    uint8_to_linear,
    validate_request,
)


DEFAULT_API = "http://127.0.0.1:7861"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "krea2_local_supersample_case_study"
SCRIPT_NAME = "krea2 local supersample detail"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def request_json(method: str, url: str, *, timeout: float, **kwargs) -> Any:
    response = requests.request(method, url, timeout=timeout, **kwargs)
    if response.status_code >= 400:
        detail = response.text[:1500].replace("\n", " ")
        raise RuntimeError(f"Forge API {response.status_code} for {url}: {detail}")
    return response.json()


def decode_image_bytes(value: str) -> bytes:
    payload = value.split(",", 1)[-1]
    return base64.b64decode(payload)


def profile_script_args(args: argparse.Namespace, profile: dict[str, int | float]) -> list[Any]:
    return [
        args.mode,
        args.profile,
        args.roi_boxes,
        int(profile["payload"]),
        int(profile["core"]),
        int(profile["overlap"]),
        str(int(profile["process_edge"])),
        int(profile["steps"]),
        float(profile["denoise"]),
        str(int(profile["candidates"])),
        float(profile["luma_cap"]),
        float(profile["chroma_cap"]),
        float(profile["low_frequency_reject_radius"]),
        float(profile["context_scale"]),
        float(profile["rewrite_feather"]),
        bool(args.strong_edge_protection),
        bool(args.append_guidance),
        bool(args.save_qa_crops),
        bool(args.allow_expensive_2048_full_grid),
        int(args.maximum_tile_count),
    ]


def _nvidia_sample() -> dict[str, float] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()[0]
    values = [part.strip() for part in first_line.split(",")]
    if len(values) != 4:
        return None
    return {
        "memory_used_mib": float(values[0]),
        "utilization_percent": float(values[1]),
        "temperature_c": float(values[2]),
        "power_w": float(values[3]),
    }


class RunMonitor:
    def __init__(self, api: str, interval: float = 1.0) -> None:
        self.api = api.rstrip("/")
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0.0

    def start(self) -> None:
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="local-supersample-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        next_report = 0.0
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self.started
            sample: dict[str, Any] = {"elapsed_seconds": round(elapsed, 3)}
            gpu = _nvidia_sample()
            if gpu is not None:
                sample.update(gpu)
            self.samples.append(sample)
            if elapsed >= next_report:
                progress_value = None
                job = ""
                try:
                    progress = requests.get(
                        f"{self.api}/sdapi/v1/progress?skip_current_image=true",
                        timeout=5,
                    ).json()
                    progress_value = float(progress.get("progress", 0.0))
                    job = str((progress.get("state") or {}).get("job") or "")
                except Exception:
                    pass
                fields = [f"elapsed={elapsed:.1f}s"]
                if progress_value is not None:
                    fields.append(f"progress={progress_value * 100:.1f}%")
                if job:
                    fields.append(f"job={job}")
                if gpu is not None:
                    fields.extend(
                        [
                            f"gpu_mem={gpu['memory_used_mib']:.0f}MiB",
                            f"gpu={gpu['utilization_percent']:.0f}%",
                            f"temp={gpu['temperature_c']:.0f}C",
                        ]
                    )
                sys.stdout.write("MONITOR " + " ".join(fields) + "\n")
                sys.stdout.flush()
                next_report = elapsed + 15.0
            self._stop.wait(self.interval)


def telemetry_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(samples), "interval_seconds": 1.0}
    for key, output_key in (
        ("memory_used_mib", "peak_memory_used_mib"),
        ("utilization_percent", "peak_utilization_percent"),
        ("temperature_c", "peak_temperature_c"),
        ("power_w", "peak_power_w"),
    ):
        values = [float(sample[key]) for sample in samples if key in sample]
        result[output_key] = max(values) if values else None
    result["scope"] = "whole GPU sampled with nvidia-smi; not process-exclusive or an instantaneous peak"
    return result


def difference_metrics(source: np.ndarray, output: np.ndarray, *, payload: int, core: int, overlap: int) -> dict[str, Any]:
    if source.shape != output.shape:
        raise ValueError(f"output shape {output.shape} does not match source {source.shape}")
    signed = output.astype(np.int16) - source.astype(np.int16)
    absolute = np.abs(signed).astype(np.float32)
    changed = np.any(signed != 0, axis=2)
    source_linear = uint8_to_linear(source)
    output_linear = uint8_to_linear(output)
    luma_delta = np.tensordot(output_linear - source_linear, LINEAR_LUMA_WEIGHTS, axes=([2], [0]))
    low_frequency = cv2.GaussianBlur(luma_delta, (0, 0), sigmaX=12.0, sigmaY=12.0)
    high_frequency = luma_delta - low_frequency

    plans = plan_local_tiles(source.shape[1], source.shape[0], payload=payload, core=core, overlap=overlap)
    x_boundaries = sorted({plan.core_x0 for plan in plans if 0 < plan.core_x0 < source.shape[1]})
    y_boundaries = sorted({plan.core_y0 for plan in plans if 0 < plan.core_y0 < source.shape[0]})
    boundary_jumps: list[np.ndarray] = []
    for x in x_boundaries:
        boundary_jumps.append(np.abs(luma_delta[:, x] - luma_delta[:, x - 1]))
    for y in y_boundaries:
        boundary_jumps.append(np.abs(luma_delta[y] - luma_delta[y - 1]))
    boundary_values = np.concatenate([values.ravel() for values in boundary_jumps]) if boundary_jumps else np.zeros(1)
    all_x_jumps = np.abs(np.diff(luma_delta, axis=1)).ravel()
    all_y_jumps = np.abs(np.diff(luma_delta, axis=0)).ravel()
    global_jump_p95 = float(np.percentile(np.concatenate((all_x_jumps, all_y_jumps)), 95))
    boundary_jump_p95 = float(np.percentile(boundary_values, 95))

    return {
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_pixel_percent": float(np.mean(changed) * 100.0),
        "mean_abs_rgb_code_delta": float(np.mean(absolute)),
        "p95_abs_rgb_code_delta": float(np.percentile(absolute, 95)),
        "p99_abs_rgb_code_delta": float(np.percentile(absolute, 99)),
        "max_abs_rgb_code_delta": int(np.max(absolute)),
        "mean_abs_linear_luma_delta": float(np.mean(np.abs(luma_delta))),
        "p95_abs_linear_luma_delta": float(np.percentile(np.abs(luma_delta), 95)),
        "mean_abs_low_frequency_luma_delta": float(np.mean(np.abs(low_frequency))),
        "p95_abs_low_frequency_luma_delta": float(np.percentile(np.abs(low_frequency), 95)),
        "mean_abs_high_frequency_luma_delta": float(np.mean(np.abs(high_frequency))),
        "p95_abs_high_frequency_luma_delta": float(np.percentile(np.abs(high_frequency), 95)),
        "clipped_channel_fraction": float(np.mean((output == 0) | (output == 255))),
        "tile_boundary_residual_jump_p95": boundary_jump_p95,
        "global_residual_jump_p95": global_jump_p95,
        "tile_boundary_to_global_jump_p95_ratio": (
            boundary_jump_p95 / global_jump_p95 if global_jump_p95 > 0 else 0.0
        ),
        "tile_boundary_count_x": len(x_boundaries),
        "tile_boundary_count_y": len(y_boundaries),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Approved RGB PNG input.")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prompt",
        required=True,
        help="Exact base prompt to send to Forge; stdout records only its SHA-256.",
    )
    parser.add_argument("--mode", choices=MODES, default=MODE_FULL_IMAGE_GRID)
    parser.add_argument("--profile", choices=tuple(LOCAL_SUPERSAMPLE_PROFILES), default=PROFILE_SAFE_1536)
    parser.add_argument("--roi-boxes", default="")
    parser.add_argument("--seed", type=int, default=3883506083)
    parser.add_argument("--maximum-tile-count", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--save-qa-crops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strong-edge-protection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-expensive-2048-full-grid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with Image.open(input_path) as opened:
        source_image = opened.convert("RGB")
    source = np.asarray(source_image, dtype=np.uint8)
    profile = get_profile(args.profile)
    rois = parse_roi_boxes(args.roi_boxes, source_image.width, source_image.height)
    validate_request(
        mode=args.mode,
        profile=args.profile,
        roi_boxes=rois,
        payload=int(profile["payload"]),
        core=int(profile["core"]),
        overlap=int(profile["overlap"]),
        process_edge=int(profile["process_edge"]),
        steps=int(profile["steps"]),
        denoise=float(profile["denoise"]),
        candidate_count=int(profile["candidates"]),
        luma_cap=float(profile["luma_cap"]),
        chroma_cap=float(profile["chroma_cap"]),
        low_frequency_reject_radius=float(profile["low_frequency_reject_radius"]),
        focused_context_scale=float(profile["context_scale"]),
        focused_rewrite_feather=float(profile["rewrite_feather"]),
        allow_expensive_2048_full_grid=bool(args.allow_expensive_2048_full_grid),
        maximum_tile_count=int(args.maximum_tile_count),
    )
    if args.mode == MODE_FOCUSED_ROI_REWRITE:
        plans = plan_focused_rois(
            source_image.width,
            source_image.height,
            rois,
            context_scale=float(profile["context_scale"]),
        )
        selected_plans = plans
        for plan in plans:
            payload_side = plan.payload_x1 - plan.payload_x0
            if payload_side >= int(profile["process_edge"]):
                raise ValueError(
                    f"focused ROI {plan.index} context is {payload_side}px and must be "
                    f"smaller than Process Edge {int(profile['process_edge'])}"
                )
    else:
        plans = plan_local_tiles(
            source_image.width,
            source_image.height,
            payload=int(profile["payload"]),
            core=int(profile["core"]),
            overlap=int(profile["overlap"]),
        )
        selected_plans = (
            plans
            if args.mode == MODE_FULL_IMAGE_GRID
            else select_tiles_for_rois(plans, rois)
        )
    if len(selected_plans) > args.maximum_tile_count:
        raise ValueError(f"planned {len(selected_plans)} tiles exceeds limit {args.maximum_tile_count}")

    prompt = str(args.prompt)
    if not prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()
    sys.stdout.write(
        f"PREFLIGHT input={source_image.width}x{source_image.height} profile={args.profile} "
        f"mode={args.mode} tiles={len(selected_plans)} candidates={int(profile['candidates'])} "
        f"process_edge={int(profile['process_edge'])} "
        f"focused_rewrite={args.mode == MODE_FOCUSED_ROI_REWRITE} "
        f"prompt_sha256={prompt_hash}\n"
    )
    sys.stdout.flush()
    if args.dry_run:
        return 0

    api = args.api.rstrip("/")
    options = request_json("GET", f"{api}/sdapi/v1/options", timeout=30)
    scripts = request_json("GET", f"{api}/sdapi/v1/scripts", timeout=30)
    if SCRIPT_NAME not in [str(name).lower() for name in scripts.get("img2img", [])]:
        raise RuntimeError(f"Forge did not register img2img script {SCRIPT_NAME!r}")

    script_args = profile_script_args(args, profile)
    payload = {
        "init_images": [base64.b64encode(input_path.read_bytes()).decode("ascii")],
        "prompt": prompt,
        "negative_prompt": "",
        "seed": int(args.seed),
        "subseed": -1,
        "sampler_name": "DPM++ 2M SDE",
        "scheduler": "Simple",
        "steps": int(profile["steps"]),
        "cfg_scale": 1.0,
        "distilled_cfg_scale": 1.15,
        "width": source_image.width,
        "height": source_image.height,
        "denoising_strength": float(profile["denoise"]),
        "n_iter": 1,
        "batch_size": 1,
        "restore_faces": False,
        "tiling": False,
        "save_images": False,
        "send_images": True,
        "include_init_images": False,
        "override_settings": {"img2img_fix_steps": True},
        "override_settings_restore_afterwards": True,
        "script_name": SCRIPT_NAME,
        "script_args": script_args,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"local_supersample_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "krea2_local_supersample.png"
    manifest_path = run_dir / "experiment_manifest.json"
    started_at = utc_now()
    started = time.perf_counter()
    monitor = RunMonitor(api)
    monitor.start()
    try:
        response = requests.post(f"{api}/sdapi/v1/img2img", json=payload, timeout=args.timeout)
        if response.status_code >= 400:
            detail = response.text[:1500].replace("\n", " ")
            raise RuntimeError(f"Forge API {response.status_code}: {detail}")
        data = response.json()
    finally:
        monitor.stop()
    duration = time.perf_counter() - started

    encoded_images = data.get("images") or []
    if len(encoded_images) != 1:
        raise RuntimeError(f"Forge returned {len(encoded_images)} images; expected one")
    output_path.write_bytes(decode_image_bytes(encoded_images[0]))
    with Image.open(output_path) as opened:
        output_image = opened.convert("RGB")
        output_info = dict(opened.info)
    if output_image.size != source_image.size:
        raise RuntimeError(f"Forge returned {output_image.size}; expected {source_image.size}")
    if "krea2_local_supersample" not in output_info:
        raise RuntimeError("output PNG is missing krea2_local_supersample metadata")
    embedded_manifest = json.loads(output_info["krea2_local_supersample"])
    output = np.asarray(output_image, dtype=np.uint8)
    focused_metrics = None
    if args.mode == MODE_FOCUSED_ROI_REWRITE:
        focused_metrics = focused_roi_difference_metrics(source, output, rois)
        if focused_metrics["changed_pixels_outside_target"] != 0:
            raise RuntimeError(
                "Focused ROI Rewrite changed pixels outside its requested targets: "
                f"{focused_metrics['changed_pixels_outside_target']}"
            )

    try:
        api_info = json.loads(data.get("info") or "{}")
    except json.JSONDecodeError:
        api_info = {}
    api_info_summary = {
        key: api_info.get(key)
        for key in (
            "seed",
            "subseed",
            "width",
            "height",
            "sampler_name",
            "cfg_scale",
            "steps",
            "batch_size",
            "sd_model_name",
            "sd_model_hash",
        )
    }
    manifest = {
        "format_version": 1,
        "algorithm": "Krea2 Local Supersample Detail case-study runner",
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": duration,
        "input": {
            "path": str(input_path),
            "size": [source_image.width, source_image.height],
            "file_sha256": file_sha256(input_path),
            "pixel_sha256": rgb_sha256(source),
        },
        "output": {
            "path": str(output_path),
            "size": [output_image.width, output_image.height],
            "file_sha256": file_sha256(output_path),
            "pixel_sha256": rgb_sha256(output),
            "metadata_keys": sorted(output_info),
        },
        "request": {
            "base_prompt": prompt,
            "base_prompt_sha256": prompt_hash,
            "negative_prompt": "",
            "seed": int(args.seed),
            "mode": args.mode,
            "profile": args.profile,
            "roi_boxes": args.roi_boxes,
            "focused_rewrite": args.mode == MODE_FOCUSED_ROI_REWRITE,
            "focused_context_scale": float(profile["context_scale"]),
            "focused_rewrite_feather": float(profile["rewrite_feather"]),
            "selected_tile_count": len(selected_plans),
            "script_args": script_args,
        },
        "backend": {
            "checkpoint": options.get("sd_model_checkpoint"),
            "checkpoint_hash": options.get("sd_checkpoint_hash"),
            "vae": options.get("sd_vae"),
            "additional_modules": options.get("forge_additional_modules"),
        },
        "api_info": api_info_summary,
        "embedded_krea2_local_supersample": embedded_manifest,
        "difference_metrics": difference_metrics(
            source,
            output,
            payload=int(profile["payload"]),
            core=int(profile["core"]),
            overlap=int(profile["overlap"]),
        ),
        "focused_roi_difference_metrics": focused_metrics,
        "telemetry": telemetry_summary(monitor.samples),
        "telemetry_samples": monitor.samples,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outside_target_changes = (
        focused_metrics["changed_pixels_outside_target"]
        if focused_metrics is not None
        else "n/a"
    )
    sys.stdout.write(
        f"COMPLETE output={output_path} duration={duration:.1f}s "
        f"processed={embedded_manifest.get('processed_tile_count')} "
        f"noop={embedded_manifest.get('rejected_noop_tile_count')} "
        f"outside_target_changes={outside_target_changes} "
        f"output_pixel_sha256={manifest['output']['pixel_sha256']}\n"
    )
    sys.stdout.write(f"MANIFEST {manifest_path}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
