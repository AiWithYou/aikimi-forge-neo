"""Prepare eight original ControlNet guides for the v2.2.1 anime gallery.

The generated layout source is an original ImageGen edit of the user-provided
character. HED, MLSD, and Depth Anything weights are installed by the earlier
gallery preparation script and are never committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "docs/assets/qwen-image21-fun-controlnet"
SIZE = (1152, 1536)
SKETCH_SIZE = (768, 1024)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_pose() -> None:
    # Hand-traced joints from the distinct full-body layout source. This is a
    # sparse body guide, deliberately omitting hair, clothing and city lines.
    canvas = Image.new("RGB", SKETCH_SIZE, "black")
    draw = ImageDraw.Draw(canvas)
    points = {
        "head": (463, 162), "neck": (452, 270),
        "left_shoulder": (331, 300), "left_elbow": (220, 180), "left_wrist": (137, 93),
        "right_shoulder": (565, 307), "right_elbow": (598, 499), "right_wrist": (657, 625),
        "left_hip": (354, 430), "left_knee": (260, 498), "left_ankle": (266, 615),
        "right_hip": (429, 447), "right_knee": (357, 674), "right_ankle": (371, 907),
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
        draw.line((points[start], points[end]), fill=color, width=8, joint="curve")
    for x, y in points.values():
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="white")
    canvas.resize(SIZE, Image.Resampling.NEAREST).save(ASSETS / "anime-v221-pose.png")


def _save_inpaint_mask() -> None:
    # Region on the front of the hoodie in the seeded Canny output. Keep face,
    # hair, hands, skates, and the surrounding ramp out of the white edit area.
    canvas = Image.new("L", SIZE, 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((510, 472, 737, 642), radius=32, fill=255)
    canvas.save(ASSETS / "anime-v221-inpaint-mask.png")


def _save_simple_maps(rgb: np.ndarray) -> None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    Image.fromarray(gray).save(ASSETS / "anime-v221-gray.png")
    smooth = cv2.bilateralFilter(gray, 7, 55, 55)
    canny = cv2.Canny(smooth, 80, 160)
    Image.fromarray(canny).save(ASSETS / "anime-v221-canny.png")

    # Retain only long contours at half resolution. The earlier ultra-sparse
    # hand sketch collapsed the body into a geometric box at strength 1.0;
    # these loose strokes still identify both legs, arms, skates and the ramp.
    small = cv2.resize(canny, (576, 768), interpolation=cv2.INTER_AREA)
    small = np.uint8(small > 70) * 255
    contours, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    long_contours = [
        contour for contour in contours if cv2.arcLength(contour, False) > 80
    ]
    scribble = cv2.drawContours(np.zeros_like(small), long_contours, -1, 255, 1)
    scribble = cv2.resize(scribble, SIZE, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(scribble).save(ASSETS / "anime-v221-scribble.png")

    # Extract black animation ink on a white page. The color remains solely in
    # the user reference and the text prompt, not this lineart guide.
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 9,
    )
    Image.fromarray(ink).save(ASSETS / "anime-v221-lineart.png")


def _save_learned_maps(render: Image.Image) -> None:
    from tools.prepare_qwen21_fun_example_controls import (
        DEPTH_REPO, DEPTH_REVISION, LEGACY, _download_annotators,
    )

    _download_annotators()
    sys.path.insert(0, str(LEGACY))
    from annotator.hed import apply_hed, unload_hed_model
    from annotator.mlsd import apply_mlsd, unload_mlsd_model

    rgb = np.asarray(render)
    small = cv2.resize(rgb, (576, 768), interpolation=cv2.INTER_AREA)
    hed = apply_hed(small)
    hed = cv2.resize(hed, SIZE, interpolation=cv2.INTER_LINEAR)
    Image.fromarray(hed).save(ASSETS / "anime-v221-hed.png")
    unload_hed_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mlsd = apply_mlsd(rgb, 0.1, 0.1)
    if not np.any(mlsd):
        raise RuntimeError("MLSD returned no straight lines.")
    Image.fromarray(mlsd).save(ASSETS / "anime-v221-mlsd.png")
    unload_mlsd_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import pipeline

    depth_pipe = pipeline(
        "depth-estimation", model=DEPTH_REPO, revision=DEPTH_REVISION,
        device=0 if torch.cuda.is_available() else -1,
    )
    depth_pipe(render)["depth"].convert("L").save(ASSETS / "anime-v221-depth.png")


def _record_sources() -> None:
    path = ASSETS / "sources.json"
    records = {item["file"]: item for item in json.loads(path.read_text(encoding="utf-8"))}
    origins = {
        "anime-v221-reference.png": "User-provided original character image, 2026-09-24",
        "anime-v221-layout-source.png": "Original ImageGen edit of anime-v221-reference.png: new glass-ramp roller-skating composition, 2026-09-24",
        "anime-v221-canny.png": "OpenCV Canny(80,160) from anime-v221-layout-source.png after bilateral smoothing",
        "anime-v221-gray.png": "OpenCV grayscale of anime-v221-layout-source.png",
        "anime-v221-scribble.png": "Long sparse Canny contours of anime-v221-layout-source.png retained at half resolution; revised after an overly sparse sketch failed",
        "anime-v221-lineart.png": "Adaptive threshold black-ink extraction of anime-v221-layout-source.png",
        "anime-v221-pose.png": "Hand-traced sparse body joints of anime-v221-layout-source.png",
        "anime-v221-hed.png": "Legacy ControlNet HED annotator applied to anime-v221-layout-source.png",
        "anime-v221-mlsd.png": "Legacy ControlNet MLSD annotator applied to anime-v221-layout-source.png",
        "anime-v221-depth.png": "Depth Anything V2 Small relative depth of anime-v221-layout-source.png",
        "anime-v221-inpaint-mask.png": "Hand-drawn white edit region on the Canny result hoodie; black preserves the other regions",
    }
    for filename, origin in origins.items():
        records[filename] = {"file": filename, "origin": origin, "sha256": _sha256(ASSETS / filename)}
    path.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ASSETS / "anime-v221-reference.png")
    parser.add_argument("--layout", type=Path, default=ASSETS / "anime-v221-layout-source.png")
    args = parser.parse_args()
    if args.reference.resolve() != (ASSETS / "anime-v221-reference.png").resolve():
        with Image.open(args.reference) as source:
            reference = ImageOps.exif_transpose(source).convert("RGB")
        reference.save(ASSETS / "anime-v221-reference.png")
    with Image.open(args.layout) as source:
        render = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("RGB"), SIZE,
            method=Image.Resampling.LANCZOS,
        )
    render.save(ASSETS / "anime-v221-layout-source.png")
    _save_simple_maps(np.asarray(render))
    _save_pose()
    _save_inpaint_mask()
    _save_learned_maps(render)
    _record_sources()
    print("Prepared v2.2.1 anime controls: Canny, Depth, Gray, HED, Lineart, MLSD, Pose, Scribble.")


if __name__ == "__main__":
    main()
