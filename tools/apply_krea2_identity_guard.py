"""Restore approved identity regions into a high-detail Krea2 4K candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin


@dataclass(frozen=True)
class ProtectionEllipse:
    """One feathered ellipse restored from the approved reference."""

    label: str
    box: tuple[int, int, int, int]
    feather: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_protection_ellipse(value: str) -> ProtectionEllipse:
    """Parse ``label:x0,y0,x1,y1,feather`` from the command line."""

    raw = str(value).strip()
    if ":" in raw:
        label, coordinates = raw.split(":", 1)
        label = label.strip()
    else:
        label = "protected_region"
        coordinates = raw
    if not label:
        raise ValueError("protection ellipse label must not be empty")
    parts = [part.strip() for part in coordinates.split(",")]
    if len(parts) != 5:
        raise ValueError(
            "protection ellipse must use label:x0,y0,x1,y1,feather"
        )
    try:
        x0, y0, x1, y1, feather = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("protection ellipse coordinates must be integers") from exc
    if x1 <= x0 or y1 <= y0:
        raise ValueError("protection ellipse must have x1 > x0 and y1 > y0")
    if feather <= 0:
        raise ValueError("protection ellipse feather must be > 0")
    return ProtectionEllipse(label=label, box=(x0, y0, x1, y1), feather=feather)


def protection_mask(
    size: tuple[int, int],
    regions: list[ProtectionEllipse],
) -> tuple[Image.Image, dict]:
    """Build a smooth union mask whose core is reference-exact and edge is blended."""

    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be > 0")
    if not regions:
        raise ValueError("at least one protection ellipse is required")

    union = np.zeros((height, width), dtype=np.float32)
    region_reports = []
    for region in regions:
        x0, y0, x1, y1 = region.box
        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(width, x1)
        clip_y1 = min(height, y1)
        if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
            raise ValueError(
                f"protection ellipse {region.label!r} does not intersect the image"
            )

        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5
        radius_x = (x1 - x0) * 0.5
        radius_y = (y1 - y0) * 0.5
        feather_normalized = min(
            1.0,
            float(region.feather) / min(radius_x, radius_y),
        )
        xs = np.arange(clip_x0, clip_x1, dtype=np.float32) + 0.5
        ys = np.arange(clip_y0, clip_y1, dtype=np.float32) + 0.5
        normalized_distance = np.sqrt(
            np.square((xs[None, :] - center_x) / radius_x)
            + np.square((ys[:, None] - center_y) / radius_y)
        )
        transition = np.clip(
            (1.0 - normalized_distance) / feather_normalized,
            0.0,
            1.0,
        )
        smooth = transition * transition * (3.0 - 2.0 * transition)
        target = union[clip_y0:clip_y1, clip_x0:clip_x1]
        np.maximum(target, smooth, out=target)
        region_reports.append(
            {
                **asdict(region),
                "box": list(region.box),
                "clipped_box": [clip_x0, clip_y0, clip_x1, clip_y1],
            }
        )

    mask_values = np.rint(union * 255.0).astype(np.uint8)
    nonzero = int(np.count_nonzero(mask_values))
    exact = int(np.count_nonzero(mask_values == 255))
    total = width * height
    report = {
        "regions": region_reports,
        "nonzero_pixels": nonzero,
        "nonzero_percent": nonzero * 100.0 / total,
        "reference_exact_pixels": exact,
        "reference_exact_percent": exact * 100.0 / total,
        "transition_pixels": nonzero - exact,
    }
    return Image.fromarray(mask_values, mode="L"), report


def apply_identity_guard(
    candidate: Image.Image,
    approved_reference: Image.Image,
    regions: list[ProtectionEllipse],
) -> tuple[Image.Image, Image.Image, dict]:
    """Restore protected reference pixels while leaving all other pixels unchanged."""

    candidate = ImageOps.exif_transpose(candidate).convert("RGB")
    approved_reference = ImageOps.exif_transpose(approved_reference).convert("RGB")
    if candidate.size != approved_reference.size:
        raise ValueError(
            "candidate and approved reference must have identical dimensions; "
            f"got {candidate.size} and {approved_reference.size}"
        )

    mask, mask_report = protection_mask(candidate.size, regions)
    result = Image.composite(approved_reference, candidate, mask)
    candidate_values = np.asarray(candidate, dtype=np.int16)
    result_values = np.asarray(result, dtype=np.int16)
    absolute_delta = np.abs(result_values - candidate_values)
    changed = np.any(absolute_delta > 0, axis=2)
    changed_pixels = int(np.count_nonzero(changed))
    total_pixels = candidate.width * candidate.height
    if changed_pixels:
        changed_delta = absolute_delta[changed]
        mean_abs_delta = float(np.mean(changed_delta))
        p95_abs_delta = float(np.percentile(changed_delta, 95))
        max_abs_delta = int(np.max(changed_delta))
    else:
        mean_abs_delta = 0.0
        p95_abs_delta = 0.0
        max_abs_delta = 0
    report = {
        "format_version": 1,
        "algorithm": "Krea2 approved-reference identity guard",
        "image_size": list(candidate.size),
        "mask": mask_report,
        "result_delta_from_candidate": {
            "changed_pixels": changed_pixels,
            "changed_percent": changed_pixels * 100.0 / total_pixels,
            "mean_abs_rgb_delta_on_changed_pixels": mean_abs_delta,
            "p95_abs_rgb_delta_on_changed_pixels": p95_abs_delta,
            "max_abs_rgb_delta": max_abs_delta,
        },
    }
    return result, mask, report


def pnginfo_with_guard(source_info: dict, report: dict) -> PngImagePlugin.PngInfo:
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in source_info.items():
        if key == "krea2_identity_guard":
            continue
        if isinstance(value, str):
            pnginfo.add_text(key, value)
    pnginfo.add_text(
        "krea2_identity_guard",
        json.dumps(report, ensure_ascii=False, separators=(",", ":")),
    )
    return pnginfo


def validate_paths(
    candidate_path: Path,
    reference_path: Path,
    output_path: Path,
    report_path: Path,
    mask_path: Path | None,
) -> None:
    paths = [candidate_path, reference_path, output_path, report_path]
    if mask_path is not None:
        paths.append(mask_path)
    resolved = [path.expanduser().resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("candidate, reference, output, report, and mask paths must be distinct")
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must use a .png extension")
    if report_path.suffix.lower() != ".json":
        raise ValueError("--report must use a .json extension")
    if mask_path is not None and mask_path.suffix.lower() != ".png":
        raise ValueError("--output-mask must use a .png extension")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--approved-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-mask", type=Path)
    parser.add_argument(
        "--protect-ellipse",
        action="append",
        required=True,
        type=parse_protection_ellipse,
        metavar="LABEL:X0,Y0,X1,Y1,FEATHER",
        help="Repeat for each identity-sensitive region restored from the reference.",
    )
    args = parser.parse_args()

    report_path = args.report or args.output.with_suffix(".identity_guard.json")
    validate_paths(
        args.candidate,
        args.approved_reference,
        args.output,
        report_path,
        args.output_mask,
    )
    if not args.candidate.is_file():
        raise FileNotFoundError(args.candidate)
    if not args.approved_reference.is_file():
        raise FileNotFoundError(args.approved_reference)
    outputs = [args.output, report_path]
    if args.output_mask is not None:
        outputs.append(args.output_mask)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite output: {existing[0]}")

    with Image.open(args.candidate) as opened:
        opened.load()
        candidate_info = dict(opened.info)
        candidate = opened.convert("RGB")
    with Image.open(args.approved_reference) as opened:
        opened.load()
        reference = opened.convert("RGB")

    result, mask, report = apply_identity_guard(
        candidate,
        reference,
        list(args.protect_ellipse),
    )
    report["candidate"] = {
        "path": str(args.candidate),
        "sha256": sha256(args.candidate),
    }
    report["approved_reference"] = {
        "path": str(args.approved_reference),
        "sha256": sha256(args.approved_reference),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(
        args.output,
        format="PNG",
        pnginfo=pnginfo_with_guard(candidate_info, report),
        optimize=True,
    )
    report["output"] = {
        "path": str(args.output),
        "sha256": sha256(args.output),
    }
    if args.output_mask is not None:
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.output_mask, format="PNG", optimize=True)
        report["mask"]["path"] = str(args.output_mask)
        report["mask"]["sha256"] = sha256(args.output_mask)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
