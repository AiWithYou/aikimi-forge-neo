"""Reproduce the original Qwen 2.1 Fun ControlNet INT8 examples on a local GPU.

Run with models/Qwen-Image-2.1/worker-env/Scripts/python.exe. Inputs are stored
in docs/assets/qwen-image21-fun-controlnet. Generation makes no network calls.
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
ANIME_PROMPT = (
    "Use Image 1 as the character identity and color reference: the same young adult "
    "woman with long silver-blue high ponytail, pale blue eyes, and a blue-and-white "
    "sailor dress. Create a NEW full-body 2D anime illustration of her balancing on one "
    "foot on a narrow stepping stone over a rain puddle, her other leg lifted behind. "
    "Her left arm holds an open transparent umbrella diagonally above and left of her "
    "head; her right arm reaches far right to release a tiny origami crane. Wide, airy "
    "asymmetric composition, entire body visible, delicate blue cel shading, luminous "
    "rainy twilight. Keep her identity and outfit from Image 1 while following the "
    "separate lineart guide for pose and object positions."
)
RING_PROMPT = (
    "Cinematic 3D environment render of an impossible canal-side observatory. A huge "
    "hollow vertical limestone ring supports a small dome on top; a long raised bridge "
    "passes through the ring opening and connects to a separate lighthouse at right. "
    "Several arches at staggered heights, rocky islands, water, and a single tiny boat "
    "in the foreground. Warm sunrise light, physically based stone and water, strong "
    "depth, detailed but coherent architecture. Follow the supplied lineart layout."
)
RING_COPPER_PROMPT = (
    "Cinematic 3D render of a fantastical clockwork observatory over a misty sea. "
    "A massive hollow upright copper ring with a domed viewing room on top, a long "
    "curving elevated bridge threading through its opening toward a separate beacon "
    "tower at the right, multiple arches at different heights, rocky islets, and one "
    "tiny boat at lower right. Aged copper, wet stone, amber windows, dramatic dusk "
    "lighting, physically based materials and atmospheric depth. Follow the supplied "
    "lineart layout."
)
RING_INPAINT_PROMPT = (
    "Use Image 1 as the existing scene. Change only the huge circular observatory "
    "and its small dome inside the white edit mask from pale limestone to dark "
    "aged copper with turquoise patina in the seams. Keep the bridge passing "
    "through the ring, the right lighthouse, islands, boat, sky, sunset lighting, "
    "and water in their original positions. Photorealistic 3D architectural render."
)
ANIME_V221_PROMPT = (
    "Use Image 1 as the character identity and outfit reference: the same young adult "
    "woman with long champagne-blonde hair, bubblegum-pink hair underneath and at the "
    "tips, luminous pink eyes, tiny pink and turquoise cheek doodles, and an oversized "
    "white hoodie covered in vivid cyan, pink, yellow and lime paint splashes. Create "
    "a NEW, crisp full-body 2D anime illustration of her roller-skating down a steep "
    "diagonal glass ramp high over a blue modern city. One skate reaches toward the "
    "viewer; her other knee bends up behind; her left hand reaches toward a floating "
    "paper star at upper left, and her right hand touches the metal rail at lower right. "
    "Visible glass panels, long converging metal rails and distant buildings. "
    "Preserve the recognizable face, hair and colorful hoodie from Image 1. "
    "Sharp expressive eyes, precise ink outlines, clear hands and skates, vivid clean "
    "cel shading and fine clothing details."
)
ANIME_V221_INPAINT_PROMPT = (
    "Use Image 1 as the character identity and outfit reference. Use the supplied "
    "existing roller-skating picture as the inpainting source. Redraw only the white "
    "edit-mask region on the front of her hoodie: replace the paint splashes there "
    "with one large clean hot-pink heart patch edged in cyan, sewn onto the white "
    "fabric. Preserve her recognizable face, champagne-blonde and pink hair, eyes, "
    "cheek doodles, pose, hands, roller skates, glass ramp and blue city composition. "
    "Crisp 2D anime ink lines and cel shading."
)

# The two unconditioned baselines run first to avoid unnecessary model reloads.
CASES = {
    "anime-reference-only": {
        "prompt": ANIME_PROMPT, "size": (768, 1024),
        "references": ("anime-character-reference.jpg",),
    },
    "ring-3d-prompt-only": {"prompt": RING_PROMPT, "size": (1024, 768)},
    "anime-reference-lineart": {
        "prompt": ANIME_PROMPT, "size": (768, 1024),
        "references": ("anime-character-reference.jpg",),
        "kind": "lineart", "control": "anime-pose-lineart.png", "strength": 0.8,
    },
    "ring-3d-control": {
        "prompt": RING_PROMPT, "size": (1024, 768),
        "kind": "scribble", "control": "ring-observatory-scribble.png", "strength": 0.65,
    },
    "ring-3d-copper": {
        "prompt": RING_COPPER_PROMPT, "size": (1024, 768),
        "kind": "scribble", "control": "ring-observatory-scribble.png", "strength": 0.65,
    },
    "anime-reference-pose": {
        "prompt": ANIME_PROMPT.replace("separate lineart guide", "separate pose guide"),
        "size": (768, 1024), "references": ("anime-character-reference.jpg",),
        "kind": "pose", "control": "anime-pose-guide.png", "strength": 0.8,
    },
    "ring-3d-canny": {
        "prompt": RING_PROMPT.replace("lineart layout", "Canny edge layout"),
        "size": (1024, 768), "kind": "canny", "control": "ring-canny.png", "strength": 0.8,
    },
    "ring-3d-depth": {
        "prompt": RING_PROMPT.replace("lineart layout", "depth layout"),
        "size": (1024, 768), "kind": "depth", "control": "ring-depth.png", "strength": 0.8,
    },
    "ring-3d-gray": {
        "prompt": RING_PROMPT.replace("lineart layout", "grayscale composition"),
        "size": (1024, 768), "kind": "gray", "control": "ring-gray.png", "strength": 0.65,
    },
    "ring-3d-hed": {
        "prompt": RING_PROMPT.replace("lineart layout", "soft HED edge layout"),
        "size": (1024, 768), "kind": "hed", "control": "ring-hed.png", "strength": 0.8,
    },
    "ring-3d-mlsd": {
        "prompt": RING_PROMPT.replace("lineart layout", "MLSD line segment layout"),
        "size": (1024, 768), "kind": "mlsd", "control": "ring-mlsd.png", "strength": 0.8,
    },
    "ring-3d-inpaint": {
        "prompt": RING_INPAINT_PROMPT, "size": (1024, 768),
        "references": ("ring-3d-control.png",),
        "kind": "canny", "control": "ring-inpaint-canny.png", "strength": 0.8,
        "inpaint_source": "ring-3d-control.png", "inpaint_mask": "ring-inpaint-mask.png",
    },
    "anime-v221-reference-only": {
        "prompt": ANIME_V221_PROMPT, "size": (1152, 1536),
        "references": ("anime-v221-reference.png",),
    },
    **{
        f"anime-v221-{kind}": {
            "prompt": ANIME_V221_PROMPT,
            "size": (1152, 1536),
            "references": ("anime-v221-reference.png",),
            "kind": kind,
            "control": f"anime-v221-{kind}.png",
            "output": f"anime-v221-result-{kind}.png",
            "strength": 1.0,
        }
        for kind in ("canny", "depth", "gray", "hed", "lineart", "mlsd", "pose", "scribble")
    },
    "anime-v221-inpaint": {
        "prompt": ANIME_V221_INPAINT_PROMPT,
        "size": (1152, 1536),
        "references": ("anime-v221-reference.png",),
        "kind": "pose", "control": "anime-v221-pose.png", "strength": 1.0,
        "inpaint_source": "anime-v221-result-canny.png",
        "inpaint_mask": "anime-v221-inpaint-mask.png",
        "output": "anime-v221-result-inpaint.png",
    },
}
V221_CASES = tuple(name for name in CASES if name.startswith("anime-v221-"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=list(CASES), help="Default: v2.2.1 anime examples")
    parser.add_argument("--all", action="store_true", help="Regenerate the older 2D and 3D gallery too")
    args = parser.parse_args()
    if args.all and args.names:
        parser.error("--all cannot be combined with case names")
    names = tuple(CASES) if args.all else (args.names or V221_CASES)
    runtime = ROOT / "models/Qwen-Image-2.1"
    runtime_manifest(runtime, "int8")
    if any(CASES[name].get("control") for name in names):
        installed(runtime)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    record_path = ASSETS / "measurements.json"
    records = {
        record["name"]: record
        for record in json.loads(record_path.read_text(encoding="utf-8"))
    } if record_path.is_file() else {}
    try:
        with runtime_lock(runtime):
            for name in names:
                case = CASES[name]
                width, height = case["size"]
                job = OUTPUTS / name
                job.mkdir(parents=True, exist_ok=True)
                control = case.get("control")
                request = {
                    "prompt": case["prompt"], "width": width, "height": height,
                    "steps": 40, "seed": 43, "precision": "int8",
                    "memory_mode": "offload",
                    "input_images": [
                        str((ASSETS / source).resolve())
                        for source in case.get("references", ())
                    ],
                    "control_kind": case.get("kind", "off"),
                    "control_image": str((ASSETS / control).resolve()) if control else "",
                    "control_strength": case.get("strength", 1.0),
                    "control_inpaint": bool(case.get("inpaint_mask")),
                }
                if case.get("inpaint_mask"):
                    request["edit_mask"] = {
                        "original_path": str((ASSETS / case["inpaint_source"]).resolve()),
                        "mask_path": str((ASSETS / case["inpaint_mask"]).resolve()),
                    }
                (job / "request.json").write_text(
                    json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Generating {name}", flush=True)
                result = worker.run_request({
                    "job_dir": str(job.resolve()),
                    "model_path": str((runtime / "model").resolve()),
                    "precision": "int8", "memory_mode": "offload",
                })
                published = ASSETS / case.get("output", f"{name}.png")
                shutil.copy2(result["output_path"], published)
                with published.open("rb") as stream:
                    image_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                meta = result["metadata"]
                records[name] = {
                    "name": name, "control_kind": case.get("kind", "off"),
                    "control_source": control,
                    "control_sha256": (
                        hashlib.sha256((ASSETS / control).read_bytes()).hexdigest()
                        if control else None
                    ),
                    "control_inpaint": bool(case.get("inpaint_mask")),
                    "inpaint_source": case.get("inpaint_source"),
                    "inpaint_mask": case.get("inpaint_mask"),
                    "references": list(case.get("references", ())),
                    "prompt": case["prompt"], "seed": 43, "steps": 40,
                    "size": [width, height],
                    "control_strength": case.get("strength") if control else None,
                    "load_seconds": meta["timings"]["load_seconds"],
                    "sampling_seconds": meta["timings"]["sampling_seconds"],
                    "peak_allocated_mib": meta["memory"].get("peak_allocated_mib"),
                    "reused_model": meta["reused_model"],
                    "output": published.name,
                    "image_sha256": image_sha256,
                    "scheduler": meta.get("scheduler"),
                    "scheduler_config_sha256": hashlib.sha256(
                        (runtime / "model/scheduler/scheduler_config.json").read_bytes()
                    ).hexdigest(),
                    "use_kv_cache": meta.get("use_kv_cache"),
                    "vae_tiling": meta.get("vae_tiling"),
                    "precision": "int8",
                    "memory_mode": "offload",
                }
                record_path.write_text(
                    json.dumps(
                        [records[key] for key in CASES if key in records],
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
    finally:
        worker.clear_runtime()


if __name__ == "__main__":
    main()
