"""Reproduce the six Qwen 2.1 Fun ControlNet INT8 examples on a local GPU.

Run with models/Qwen-Image-2.1/worker-env/Scripts/python.exe. The source
control maps live in docs/assets/qwen-image21-fun-controlnet/. No download or
external API call occurs during generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.qwen_image21.core import runtime_lock, runtime_manifest  # noqa: E402
from modules_forge.qwen_image21.fun_controlnet import installed  # noqa: E402
from tools import qwen_image21_worker as worker  # noqa: E402

ASSETS = ROOT / "docs/assets/qwen-image21-fun-controlnet"
OUTPUTS = ROOT / "outputs/qwen-image-2.1/fun-controlnet-examples"
CASES = {
    "pose-anime": (
        "pose", "pose-control.png", 896, 704,
        "Waist-up 2D anime character portrait of a smiling adult performer facing forward. Coral-pink hair in two high buns, expressive eyes, ivory off-shoulder ruffled dress, soft blue theater background. Her elbows angle outward and her hands rest naturally near her hips. Exactly two arms, natural anatomy, clean ink outlines, flat cel shading and vivid hand-painted anime colors. Follow the shoulder and arm placement from the pose map.",
    ),
    "pose-3d": (
        "pose", "pose-control.png", 896, 704,
        "Waist-up polished 3D animated-film portrait of a smiling adult performer facing forward. Dark hair in two high buns, satin teal off-shoulder dress, sculpted fabric folds, warm theater background and soft rim lighting. Her elbows angle outward and her hands rest naturally near her hips. Exactly two arms, natural anatomy, physically based materials and realistic depth. Follow the shoulder and arm placement from the pose map.",
    ),
    "gray-anime": (
        "gray", "gray-control.png", 768, 896,
        "Hand-drawn 2D anime background painting of seven small houseplants in round ceramic pots arranged in two rows on a sunlit shelf. Distinct broad and spiky leaves, pastel green foliage, colorful terracotta and blue pots, warm afternoon light, bold dark ink outlines, simplified flat cel-shaded colors, visible watercolor brush texture, cozy botanical studio. Stylized animation background art. Keep the two-row arrangement from the grayscale control image.",
    ),
    "gray-3d": (
        "gray", "gray-control.png", 768, 896,
        "Photorealistic 3D product render of a collection of seven small houseplants in round ceramic pots arranged in two rows on a tabletop. Ferns and succulents with distinct leaves, glazed terracotta and ivory pots, detailed pebbles, bright studio backdrop, ray-traced soft shadows, realistic materials and depth of field. Match the arrangement and silhouettes in the grayscale control image.",
    ),
    "scribble-anime": (
        "scribble", "scribble-control.png", 768, 800,
        "Full-body 2D anime sports illustration of an adult soccer player dribbling a round football near his left foot, arms angled outward for balance. Bright orange jersey, navy shorts, running shoes, green pitch and blurred stadium crowd. Detailed expressive face and hands, complete colored skin, bold clean ink outlines, dynamic cel shading. Follow the athlete and ball placement from the scribble map.",
    ),
    "scribble-3d": (
        "scribble", "scribble-control.png", 768, 800,
        "Full-body 3D animated-film render of an adult soccer player dribbling a round football near his left foot, arms angled outward for balance. Teal jersey, navy shorts, running shoes, green pitch and softly blurred stadium crowd. Detailed expressive face and hands, complete colored skin, natural anatomy, physically based fabric and skin, cinematic soft light. Follow the athlete and ball placement from the scribble map.",
    ),
}
STRENGTHS = {name: 1.0 for name in CASES}
STRENGTHS.update({"gray-anime": 0.4, "scribble-anime": 0.65, "scribble-3d": 0.65})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=list(CASES), help="Default: all six examples")
    args = parser.parse_args()
    runtime = ROOT / "models/Qwen-Image-2.1"
    runtime_manifest(runtime, "int8")
    installed(runtime)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    names = args.names or list(CASES)
    record_path = ASSETS / "measurements.json"
    records = {record["name"]: record for record in json.loads(record_path.read_text(encoding="utf-8"))} if record_path.is_file() else {}
    try:
        with runtime_lock(runtime):
            for name in names:
                kind, source, width, height, prompt = CASES[name]
                strength = STRENGTHS[name]
                job = OUTPUTS / name
                job.mkdir(parents=True, exist_ok=True)
                request = {
                    "prompt": prompt, "width": width, "height": height,
                    "steps": 40, "seed": 43, "precision": "int8", "memory_mode": "offload",
                    "input_images": [], "control_kind": kind,
                    "control_image": str((ASSETS / source).resolve()),
                    "control_strength": strength,
                }
                (job / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
                sys.stdout.write(f"Generating {name}\n")
                sys.stdout.flush()
                result = worker.run_request({
                    "job_dir": str(job.resolve()),
                    "model_path": str((runtime / "model").resolve()),
                    "precision": "int8", "memory_mode": "offload",
                })
                published = ASSETS / f"{name}.png"
                shutil.copy2(result["output_path"], published)
                with published.open("rb") as stream:
                    image_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                meta = result["metadata"]
                records[name] = {
                    "name": name, "control_kind": kind, "control_source": source,
                    "prompt": prompt, "seed": 43, "steps": 40,
                    "size": [width, height], "control_strength": strength,
                    "load_seconds": meta["timings"]["load_seconds"],
                    "sampling_seconds": meta["timings"]["sampling_seconds"],
                    "peak_allocated_mib": meta["memory"].get("peak_allocated_mib"),
                    "reused_model": meta["reused_model"],
                    "image_sha256": image_sha256,
                }
                record_path.write_text(
                    json.dumps([records[key] for key in CASES if key in records], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    finally:
        worker.clear_runtime()


if __name__ == "__main__":
    main()
