"""Apply a shared Krea2 coherent texture finish to an existing generated PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.krea2_quality import adaptive_detail_guard
from modules_forge.krea2_highres import (
    KREA2_VRAM_CANVAS_PROFILES,
    krea2_vram_canvas_profile,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def texture_finish_values(profile_name: str) -> dict[str, float]:
    profile = krea2_vram_canvas_profile(profile_name)
    return {
        "detail_strength": float(profile["finish_detail_strength"]),
        "detail_radius": float(profile["finish_detail_radius"]),
        "detail_threshold": float(profile["finish_detail_threshold"]),
        "max_detail_delta": float(profile["finish_max_detail_delta"]),
    }


def apply_texture_finish(
    image: Image.Image,
    profile_name: str,
) -> tuple[Image.Image, dict]:
    values = texture_finish_values(profile_name)
    result, detail_report = adaptive_detail_guard(
        image,
        strength=values["detail_strength"],
        radius=values["detail_radius"],
        detail_threshold=values["detail_threshold"],
        max_detail_delta=values["max_detail_delta"],
    )
    return result, {
        "profile": profile_name,
        "enabled": values["detail_strength"] > 0,
        **values,
        "report": detail_report,
    }


def pnginfo_with_finish(source_info: dict, finish: dict) -> PngImagePlugin.PngInfo:
    pnginfo = PngImagePlugin.PngInfo()
    canvas_report = None
    raw_canvas = source_info.get("vram_canvas")
    if isinstance(raw_canvas, str):
        try:
            parsed = json.loads(raw_canvas)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            parsed["texture_finish"] = finish
            canvas_report = parsed
    for key, value in source_info.items():
        if key in {"vram_canvas", "krea2_texture_finish"}:
            continue
        if isinstance(value, str):
            pnginfo.add_text(key, value)
    if canvas_report is not None:
        pnginfo.add_text(
            "vram_canvas",
            json.dumps(canvas_report, ensure_ascii=False, separators=(",", ":")),
        )
    elif isinstance(raw_canvas, str):
        pnginfo.add_text("vram_canvas", raw_canvas)
    pnginfo.add_text(
        "krea2_texture_finish",
        json.dumps(finish, ensure_ascii=False, separators=(",", ":")),
    )
    return pnginfo


def update_manifest(source: Path, output: Path, finish: dict) -> None:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    manifest["texture_finish"] = finish
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(KREA2_VRAM_CANVAS_PROFILES)),
        default="texture_rich_4k",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    if (args.manifest is None) != (args.output_manifest is None):
        raise ValueError("--manifest and --output-manifest must be passed together")
    report_path = args.report or args.output.with_suffix(".texture_finish.json")
    output_targets = [args.output, report_path]
    if args.output_manifest is not None:
        output_targets.append(args.output_manifest)
    existing = [path for path in output_targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite output: {existing[0]}")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    with Image.open(args.input) as opened:
        opened.load()
        source_info = dict(opened.info)
        source = opened.convert("RGB")
    result, finish = apply_texture_finish(source, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(
        args.output,
        format="PNG",
        pnginfo=pnginfo_with_finish(source_info, finish),
        optimize=True,
    )
    finish["input"] = {
        "path": str(args.input),
        "sha256": sha256(args.input),
        "size": list(source.size),
    }
    finish["output"] = {
        "path": str(args.output),
        "sha256": sha256(args.output),
        "size": list(result.size),
    }
    if args.manifest is not None and args.output_manifest is not None:
        update_manifest(args.manifest, args.output_manifest, finish)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(finish, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(finish, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
