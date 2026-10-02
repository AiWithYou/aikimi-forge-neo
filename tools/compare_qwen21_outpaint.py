"""Generate three paired native Outpaint examples, changing only the adapter.

Run with models/Qwen-Image-2.1/worker-env/Scripts/python.exe.
Existing successful jobs are verified and reused; no model downloads occur.
"""

from __future__ import annotations

# ruff: noqa: E402, T201, S101
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.qwen_image21.core import Request, copy_inputs, runtime_lock, runtime_manifest
from modules_forge.qwen_image21.outpaint import prepare, recipe
from modules_forge.qwen_image21.outpaint_native import finish, snapshot, validate_snapshot
from tools import qwen_image21_worker as worker

ASSETS = ROOT / "docs/assets/qwen-outpaint-comparison"
JOBS = ROOT / "outputs/qwen-outpaint-comparison"
RUNTIME = ROOT / "models/Qwen-Image-2.1"
CASES = {
    "street": ("docs/assets/qwen-native-outpaint/source.png", (736, 512), (128, 0, 128, 0), ""),
    "anime": (
        "docs/assets/qwen-image21-fun-controlnet/anime-v221-reference.png",
        (512, 768),
        (128, 0, 128, 0),
        "Anime illustration of a woman with blonde and pink hair and a colorful white hoodie.",
    ),
    "architecture": (
        "docs/assets/qwen-image21-fun-controlnet/ring-3d-control.png",
        (768, 576),
        (0, 128, 0, 128),
        "A limestone ring observatory and bridges over the sea at sunrise, cinematic 3D render.",
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    runtime_manifest(RUNTIME, "base_q4_k_m")
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        with runtime_lock(RUNTIME):
            for version in ("none", "v2"):
                for name, (input_file, size, margins, scene) in CASES.items():
                    source_path = ASSETS / f"{name}-source.png"
                    if not source_path.exists():
                        with Image.open(ROOT / input_file) as original:
                            original.convert("RGBA").resize(size, Image.Resampling.LANCZOS).save(source_path)
                    source = Image.open(source_path)
                    _, canvas, plan = prepare(source, *margins)
                    canvas.save(ASSETS / f"{name}-reference.png")
                    request = Request(
                        prompt=recipe(plan, version, scene)["prompt"],
                        width=plan.size[0],
                        height=plan.size[1],
                        steps=25,
                        seed=42,
                        precision="base_q4_k_m",
                        memory_mode="offload",
                        input_images=(str(source_path.resolve()),),
                        outpaint_version=version,
                        outpaint_margins=margins,
                        outpaint_feather=0,
                    ).resolved()
                    job = JOBS / f"{name}-{version}"
                    job.mkdir(parents=True, exist_ok=True)
                    values = request.to_dict()
                    values["runtime_root"] = str(RUNTIME.resolve())
                    values["clean_input_images"] = copy_inputs(values["input_images"], job)
                    snapshot(values, job)
                    validate_snapshot(values, job)
                    request_path = job / "request.json"
                    if (job / "result.json").exists():
                        previous = json.loads(request_path.read_text(encoding="utf-8"))
                        if previous != values:
                            raise RuntimeError(f"Existing job has different conditions: {job}")
                        result = json.loads((job / "result.json").read_text(encoding="utf-8"))
                    else:
                        request_path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"GENERATING {name} {version}", flush=True)
                        result = worker.run_request(
                            {
                                "job_dir": str(job.resolve()),
                                "model_path": str((RUNTIME / "model").resolve()),
                                "precision": request.precision,
                                "memory_mode": request.memory_mode,
                            }
                        )
                    raw = Image.open(job / "output-generated.png").convert("RGBA")
                    output = Image.open(job / "output.png").convert("RGBA")
                    assert output.size == plan.size == raw.size
                    assert output.crop(plan.box).tobytes() == source.convert("RGBA").tobytes()
                    assert output.tobytes() == finish(raw, values).tobytes()
                    meta = result["metadata"]
                    adapter = meta["outpaint"]["adapter"]
                    assert (adapter is None) if version == "none" else adapter["version"] == version
                    for suffix, filename in (("", "output.png"), ("-raw", "output-generated.png")):
                        shutil.copyfile(job / filename, ASSETS / f"{name}-{version}{suffix}.png")
                    record = {
                        "case": name,
                        "version": version,
                        "source": input_file,
                        "source_sha256": digest(source_path),
                        "reference_sha256": digest(job / "outpaint-reference.png"),
                        "request": values,
                        "metadata": meta,
                        "output_sha256": digest(job / "output.png"),
                        "raw_sha256": digest(job / "output-generated.png"),
                        "original_rgba_exact": True,
                        "composition_exact": True,
                    }
                    records.append(record)
                    (ASSETS / "measurements.json").write_text(
                        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(f"VERIFIED {name} {version} {meta['timings']}", flush=True)
            for name in CASES:
                a, b = [next(r for r in records if r["case"] == name and r["version"] == v) for v in ("none", "v2")]
                assert a["reference_sha256"] == b["reference_sha256"]
                assert {
                    k: v
                    for k, v in a["request"].items()
                    if k not in ("outpaint_version", "input_images", "clean_input_images", "outpaint")
                } == {
                    k: v
                    for k, v in b["request"].items()
                    if k not in ("outpaint_version", "input_images", "clean_input_images", "outpaint")
                }
                for key in (
                    "scheduler",
                    "scheduler_config",
                    "effective_prompt",
                    "input_resolution",
                    "steps",
                    "seed",
                    "precision",
                ):
                    assert a["metadata"][key] == b["metadata"][key], key
            print("ALL THREE PAIRS VERIFIED", flush=True)
    finally:
        worker.clear_runtime()


if __name__ == "__main__":
    main()
