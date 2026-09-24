"""Build original guides for the Qwen 2.1 Fun Union gallery.

Run with Neo's main venv, which contains OpenCV and the legacy preprocessors.
The source render is an original ImageGen work in the gallery. HED, MLSD and
Depth Anything V2 weights are fetched only for this preparation step; they are
not required by the Qwen generation worker or distributed in this repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "docs/assets/qwen-image21-fun-controlnet"
PREPROCESSORS = ROOT / "models/ControlNetPreprocessor"
LEGACY = ROOT / "extensions-builtin/forge_legacy_preprocessors"
WEIGHTS = (
    (
        "lllyasviel/Annotators", "982e7edaec38759d914a963c48c4726685de7d96",
        "ControlNetHED.pth", "hed", "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa",
    ),
    (
        "lllyasviel/ControlNet", "e78a8c4a5052a238198043ee5c0cb44e22abb9f7",
        "annotator/ckpts/mlsd_large_512_fp32.pth", "mlsd", "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
    ),
)
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


def _download_annotators() -> None:
    for repo, revision, filename, folder, expected_sha256 in WEIGHTS:
        target = PREPROCESSORS / folder / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or _sha256(target) != expected_sha256:
            cached = hf_hub_download(repo, filename, revision=revision)
            shutil.copy2(cached, target)
        if _sha256(target) != expected_sha256:
            raise RuntimeError(f"Preprocessor weight SHA-256 mismatch: {target}")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_pose() -> None:
    """Trace the original anime lineart as a deliberately sparse pose guide."""
    canvas = Image.new("RGB", (768, 1024), "black")
    draw = ImageDraw.Draw(canvas)
    points = {
        "head": (389, 272), "neck": (395, 335),
        "left_shoulder": (327, 353), "left_elbow": (226, 254), "left_wrist": (180, 209),
        "right_shoulder": (447, 352), "right_elbow": (560, 341), "right_wrist": (680, 289),
        "left_hip": (370, 480), "left_knee": (429, 770), "left_ankle": (430, 922),
        "right_hip": (455, 475), "right_knee": (525, 660), "right_ankle": (655, 610),
    }
    bones = [
        ("head", "neck", "#ff0000"), ("neck", "left_shoulder", "#ff5500"),
        ("left_shoulder", "left_elbow", "#ffaa00"),
        ("left_elbow", "left_wrist", "#ffff00"),
        ("neck", "right_shoulder", "#00ff00"),
        ("right_shoulder", "right_elbow", "#00ffaa"),
        ("right_elbow", "right_wrist", "#00ffff"),
        ("neck", "left_hip", "#5500ff"),
        ("neck", "right_hip", "#0000ff"),
        ("left_hip", "left_knee", "#ff00ff"),
        ("left_knee", "left_ankle", "#ff0088"),
        ("right_hip", "right_knee", "#0088ff"),
        ("right_knee", "right_ankle", "#8800ff"),
    ]
    for start, end, color in bones:
        draw.line((points[start], points[end]), fill=color, width=7, joint="curve")
    for point in points.values():
        draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill="white")
    canvas.save(ASSETS / "anime-pose-guide.png")


def _save_inpaint_mask() -> None:
    # White selects the ring and its dome; the bridge, lighthouse and water
    # remain black so the result demonstrates a local material replacement.
    scale = 4
    mask = Image.new("L", (1024 * scale, 768 * scale), 0)
    draw = ImageDraw.Draw(mask)

    def ellipse(box, fill):
        draw.ellipse(tuple(value * scale for value in box), fill=fill)

    ellipse((211, 114, 574, 515), 255)
    ellipse((321, 174, 511, 460), 0)
    draw.polygon([(x * scale, y * scale) for x, y in (
        (334, 153), (337, 84), (368, 27), (423, 17), (462, 60), (470, 145),
    )], fill=255)
    mask.resize((1024, 768), Image.Resampling.LANCZOS).save(ASSETS / "ring-inpaint-mask.png")


def _save_simple_maps(render: Image.Image) -> None:
    rgb = np.asarray(render.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    Image.fromarray(gray).save(ASSETS / "ring-gray.png")
    smoothed = cv2.bilateralFilter(gray, 7, 55, 55)
    canny = cv2.Canny(smoothed, 80, 160)
    Image.fromarray(canny).save(ASSETS / "ring-canny.png")
    with Image.open(ASSETS / "ring-3d-control.png") as source:
        inpaint_gray = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
    inpaint_canny = cv2.Canny(cv2.bilateralFilter(inpaint_gray, 7, 55, 55), 80, 160)
    Image.fromarray(inpaint_canny).save(ASSETS / "ring-inpaint-canny.png")


def _save_learned_maps(render: Image.Image) -> None:
    _download_annotators()
    sys.path.insert(0, str(LEGACY))
    from annotator.hed import apply_hed, unload_hed_model
    from annotator.mlsd import apply_mlsd, unload_mlsd_model

    rgb = np.asarray(render.convert("RGB"))
    # HED operates at half resolution to avoid spending GPU memory on tiny
    # stone texture while retaining the ring and bridge silhouette.
    small = cv2.resize(rgb, (512, 384), interpolation=cv2.INTER_AREA)
    hed = apply_hed(small)
    hed = cv2.resize(hed, (1024, 768), interpolation=cv2.INTER_LINEAR)
    Image.fromarray(hed).save(ASSETS / "ring-hed.png")
    unload_hed_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mlsd = apply_mlsd(rgb, 0.1, 0.1)
    if not np.any(mlsd):
        raise RuntimeError("MLSD returned no lines; inspect the source and thresholds.")
    Image.fromarray(mlsd).save(ASSETS / "ring-mlsd.png")
    unload_mlsd_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import pipeline

    depth_pipe = pipeline(
        "depth-estimation", model=DEPTH_REPO, revision=DEPTH_REVISION,
        device=0 if torch.cuda.is_available() else -1,
    )
    depth = depth_pipe(render)["depth"].convert("L")
    depth.save(ASSETS / "ring-depth.png")


def _record_sources() -> None:
    path = ASSETS / "sources.json"
    records = {item["file"]: item for item in json.loads(path.read_text(encoding="utf-8"))}
    additions = {
        "ring-observatory-render-source.png": "Original ImageGen 3D render of the ring observatory, 2026-09-24",
        "ring-canny.png": "OpenCV Canny(80,160) of ring-observatory-render-source.png after bilateral smoothing",
        "ring-gray.png": "OpenCV RGB-to-grayscale conversion of ring-observatory-render-source.png",
        "ring-hed.png": "Legacy ControlNet HED annotator at half resolution, pinned lllyasviel/Annotators weights",
        "ring-mlsd.png": "Legacy ControlNet MLSD annotator on original render, pinned lllyasviel/ControlNet weights",
        "ring-depth.png": "Depth Anything V2 Small relative-depth estimate of original render, pinned Hugging Face revision",
        "anime-pose-guide.png": "Hand traced sparse body skeleton from anime-pose-lineart.png",
        "ring-inpaint-mask.png": "Hand drawn white edit region for ring-3d-control.png; black keeps the surroundings",
        "ring-inpaint-canny.png": "OpenCV Canny(80,160) of the original Qwen render ring-3d-control.png after bilateral smoothing",
    }
    for filename, origin in additions.items():
        records[filename] = {"file": filename, "origin": origin, "sha256": _sha256(ASSETS / filename)}
    path.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    with Image.open(ASSETS / "ring-observatory-render-source.png") as source:
        render = ImageOps.fit(source.convert("RGB"), (1024, 768), method=Image.Resampling.LANCZOS)
    render.save(ASSETS / "ring-observatory-render-source.png")
    _save_simple_maps(render)
    _save_pose()
    _save_inpaint_mask()
    _save_learned_maps(render)
    _record_sources()
    print("Prepared Canny, Depth, Gray, HED, MLSD, Pose and inpaint mask.")


if __name__ == "__main__":
    main()
