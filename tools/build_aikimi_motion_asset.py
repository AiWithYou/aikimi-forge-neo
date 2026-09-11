"""Build one transparent Aikimi APNG from an ImageGen motion strip.

The source art is not redrawn.  This tool only performs deterministic green-screen
matting, equal-slot extraction, uniform placement, and lossless animation encoding.
It is intentionally state-agnostic so mascot art can be replaced without changing
the browser component.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
import zlib

import numpy as np
import PIL
from PIL import Image, ImageDraw
import scipy
from scipy import ndimage


def parse_durations(value: str, frame_count: int) -> list[int]:
    durations = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(durations) != frame_count:
        raise ValueError(f"expected {frame_count} source durations, got {len(durations)}")
    if any(duration < 20 or duration > 60_000 or duration % 10 for duration in durations):
        raise ValueError("durations must be 20-60000 ms and use 10 ms increments")
    return durations


def encoded_indices(frame_count: int, loop_mode: str) -> list[int]:
    if loop_mode == "ping-pong":
        return [*range(frame_count), *range(frame_count - 2, 0, -1)]
    return list(range(frame_count))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_green_screen(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border_width = min(12, max(1, min(image.size) // 8))
    border = np.concatenate(
        (
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[:, :border_width].reshape(-1, 3),
            rgb[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    green = border[:, 1]
    candidates = border[(green - border[:, 0] > 80) & (green - border[:, 2] > 80)]
    if len(candidates) < len(border) * 0.5:
        raise ValueError("source border does not contain enough green-screen pixels")
    key = np.median(candidates, axis=0)
    if key[1] - max(key[0], key[2]) < 120:
        raise ValueError(f"estimated background is not a strong green screen: {key.tolist()}")
    return key


def matte_image(image: Image.Image, key: np.ndarray, alpha_cutoff: float) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb = rgba[..., :3]
    source_alpha = rgba[..., 3] / 255.0
    background_spill = float(key[1] - max(key[0], key[2]))
    green_spill = np.maximum(rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]), 0.0)
    physical_alpha = np.clip((background_spill - green_spill) / background_spill, 0.0, 1.0)
    physical_alpha = np.where(green_spill <= 0.0, 1.0, physical_alpha)
    low_spill = 8.0
    high_spill = background_spill * 0.82
    alpha = np.clip(
        (high_spill - green_spill) / (high_spill - low_spill),
        0.0,
        1.0,
    )
    alpha = np.where(green_spill <= low_spill, 1.0, alpha) * source_alpha
    alpha = np.where(alpha <= max(alpha_cutoff, 16.0 / 255.0), 0.0, alpha)
    alpha = np.where(alpha > 0.985, 1.0, alpha)

    safe_physical_alpha = np.maximum(physical_alpha[..., None], 1.0 / 1000.0)
    foreground = (rgb - (1.0 - physical_alpha[..., None]) * key) / safe_physical_alpha
    foreground = np.clip(foreground, 0.0, 255.0)

    # Remove residual green spill only on pixels that were blended with the key.
    translucent = alpha < 0.999
    neutral_green = np.maximum(foreground[..., 0], foreground[..., 2]) + 2.0
    foreground[..., 1] = np.where(
        translucent,
        np.minimum(foreground[..., 1], neutral_green),
        foreground[..., 1],
    )

    output = np.zeros_like(rgba, dtype=np.uint8)
    output[..., :3] = np.rint(foreground).astype(np.uint8)
    output[..., 3] = np.rint(alpha * 255.0).astype(np.uint8)
    output[output[..., 3] == 0, :3] = 0
    return Image.fromarray(output)


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("slot is empty after component cleanup")
    return bbox


def extract_components(
    strip: Image.Image,
    frame_count: int,
    alpha_cutoff: float,
) -> tuple[list[dict[str, object]], list[int], dict[str, object]]:
    key = estimate_green_screen(strip)
    matted_strip = matte_image(strip, key, alpha_cutoff)
    matted_data = np.asarray(matted_strip, dtype=np.uint8).copy()
    labels, component_count = ndimage.label(
        matted_data[..., 3] > 8,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count < frame_count:
        raise ValueError(f"found only {component_count} foreground components")

    areas = np.bincount(labels.ravel())
    slices = ndimage.find_objects(labels)
    components = []
    for label in range(1, component_count + 1):
        bounds = slices[label - 1]
        if bounds is None:
            continue
        y_slice, x_slice = bounds
        area = int(areas[label])
        components.append(
            {
                "label": label,
                "area": area,
                "center_x": (x_slice.start + x_slice.stop) / 2,
                "bbox": (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop),
            }
        )
    largest_area = max(component["area"] for component in components)
    minimum_component_area = max(12, round(largest_area * 0.0005))
    components = [component for component in components if component["area"] >= minimum_component_area]

    nominal_width = strip.width / frame_count
    minimum_seed_spacing = nominal_width * 0.35
    principal = []
    for component in sorted(components, key=lambda item: item["area"], reverse=True):
        if all(abs(component["center_x"] - seed["center_x"]) >= minimum_seed_spacing for seed in principal):
            principal.append(component)
        if len(principal) == frame_count:
            break
    if len(principal) != frame_count:
        raise ValueError(f"could not identify {frame_count} spatially separated principal poses")
    if min(component["area"] for component in principal) < largest_area * 0.15:
        raise ValueError("one or more principal pose components are unexpectedly sparse")
    principal.sort(key=lambda item: item["center_x"])
    for index, seed in enumerate(principal):
        nominal_left = index * nominal_width
        nominal_right = (index + 1) * nominal_width
        if not nominal_left <= seed["center_x"] < nominal_right:
            raise ValueError(
                f"principal pose {index} is outside its nominal slot: "
                f"center_x={seed['center_x']:.3f}, "
                f"slot=({nominal_left:.3f}, {nominal_right:.3f})"
            )

    groups = {int(seed["label"]): [int(seed["label"])] for seed in principal}
    principal_labels = set(groups)
    for component in components:
        label = int(component["label"])
        if label in principal_labels:
            continue
        nearest = min(principal, key=lambda seed: abs(component["center_x"] - seed["center_x"]))
        groups[int(nearest["label"])].append(label)

    entries: list[dict[str, object]] = []
    group_component_counts = []
    for index, seed in enumerate(principal):
        group_labels = groups[int(seed["label"])]
        mask = np.isin(labels, group_labels)
        grouped_data = matted_data.copy()
        grouped_data[~mask] = 0
        grouped = Image.fromarray(grouped_data)
        full_bbox = alpha_bbox(grouped)
        if full_bbox[0] == 0 or full_bbox[1] == 0 or full_bbox[2] == strip.width or full_bbox[3] == strip.height:
            raise ValueError(f"pose {index} reaches the source image boundary: {full_bbox}")
        nominal_left = round(index * strip.width / frame_count)
        nominal_right = round((index + 1) * strip.width / frame_count)
        local_bbox = (
            full_bbox[0] - nominal_left,
            full_bbox[1],
            full_bbox[2] - nominal_left,
            full_bbox[3],
        )
        entries.append(
            {
                "image": grouped.crop(full_bbox),
                "bbox": local_bbox,
                "full_bbox": full_bbox,
                "slot_size": (nominal_right - nominal_left, strip.height),
            }
        )
        group_component_counts.append(len(group_labels))
    diagnostics = {
        "principal_centers_x": [round(float(seed["center_x"]), 3) for seed in principal],
        "principal_component_areas": [int(seed["area"]) for seed in principal],
        "group_component_counts": group_component_counts,
        "component_minimum_area": minimum_component_area,
    }
    return entries, [int(round(value)) for value in key], diagnostics


def resize_rgba(image: Image.Image, width: int, height: int) -> Image.Image:
    target = (max(1, width), max(1, height))
    resized = image.convert("RGBA").resize(target, Image.Resampling.LANCZOS)
    resized_data = np.asarray(resized, dtype=np.float32).copy()
    rgb = resized_data[..., :3]
    resized_alpha_bytes = resized_data[..., 3]
    translucent = resized_alpha_bytes < 254
    rgb[..., 1] = np.where(
        translucent,
        np.minimum(rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2]) + 2.0),
        rgb[..., 1],
    )
    transparent = resized_alpha_bytes <= 16
    rgb[transparent] = 0
    resized_alpha_bytes[transparent] = 0
    output = np.concatenate(
        (np.clip(rgb, 0, 255), resized_alpha_bytes[..., None]),
        axis=2,
    )
    return Image.fromarray(np.rint(output).astype(np.uint8))


def render_frames(
    entries: list[dict[str, object]],
    width: int,
    height: int,
    padding: int,
    anchor_mode: str,
) -> list[Image.Image]:
    inner_width, inner_height = width - padding * 2, height - padding * 2
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError("padding leaves no usable output canvas")

    if anchor_mode == "motion":
        normalized_boxes = []
        for entry in entries:
            left, top, right, bottom = entry["bbox"]
            slot_width, _ = entry["slot_size"]
            normalized_boxes.append((left - slot_width / 2, top, right - slot_width / 2, bottom))
        union_left = min(box[0] for box in normalized_boxes)
        union_top = min(box[1] for box in normalized_boxes)
        union_right = max(box[2] for box in normalized_boxes)
        union_bottom = max(box[3] for box in normalized_boxes)
        scale = min(
            inner_width / (union_right - union_left),
            inner_height / (union_bottom - union_top),
        )
        used_width = (union_right - union_left) * scale
        used_height = (union_bottom - union_top) * scale
        origin_x = padding + (inner_width - used_width) / 2
        origin_y = padding + (inner_height - used_height) / 2
    else:
        max_width = max(entry["image"].width for entry in entries)
        max_height = max(entry["image"].height for entry in entries)
        scale = min(inner_width / max_width, inner_height / max_height)

    frames: list[Image.Image] = []
    for index, entry in enumerate(entries):
        source = entry["image"]
        scaled_width = max(1, round(source.width * scale))
        scaled_height = max(1, round(source.height * scale))
        resized = resize_rgba(source, scaled_width, scaled_height)

        if anchor_mode == "baseline":
            x = round((width - scaled_width) / 2)
            y = height - padding - scaled_height
        elif anchor_mode == "center":
            x = round((width - scaled_width) / 2)
            y = round((height - scaled_height) / 2)
        else:
            norm_left, norm_top, _, _ = normalized_boxes[index]
            x = round(origin_x + (norm_left - union_left) * scale)
            y = round(origin_y + (norm_top - union_top) * scale)

        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        frame.alpha_composite(resized, (x, y))
        frames.append(frame)
    return frames


def checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    board = Image.new("RGBA", size, (238, 241, 244, 255))
    draw = ImageDraw.Draw(board)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(211, 217, 223, 255))
    return board


def save_contact_sheet(frames: list[Image.Image], path: Path) -> None:
    label_height = 28
    sheet = Image.new("RGBA", (frames[0].width * len(frames), frames[0].height + label_height))
    for index, frame in enumerate(frames):
        panel = checkerboard(frame.size)
        panel.alpha_composite(frame)
        sheet.alpha_composite(panel, (index * frame.width, 0))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, frames[0].height, sheet.width, sheet.height), fill=(25, 31, 38, 255))
    for index in range(len(frames)):
        draw.text((index * frames[0].width + 10, frames[0].height + 7), f"{index:03d}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(path, format="PNG", optimize=True)


def edge_alpha_count(frame: Image.Image) -> int:
    alpha = np.asarray(frame.getchannel("A"))
    return int(
        np.count_nonzero(alpha[0, :])
        + np.count_nonzero(alpha[-1, :])
        + np.count_nonzero(alpha[:, 0])
        + np.count_nonzero(alpha[:, -1])
    )


def chroma_adjacent_count(frame: Image.Image, green_key: list[int]) -> int:
    rgba = np.asarray(frame.convert("RGBA"), dtype=np.int32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    distance_squared = np.square(rgb - np.asarray(green_key, dtype=np.int32)).sum(axis=2)
    return int(np.count_nonzero((alpha > 16) & (distance_squared <= 108**2)))


def visible_digest(image: Image.Image) -> str:
    rgba = bytearray(image.convert("RGBA").tobytes())
    for offset in range(0, len(rgba), 4):
        if rgba[offset + 3] == 0:
            rgba[offset : offset + 3] = b"\0\0\0"
    return hashlib.sha256(rgba).hexdigest()


def padding_error(frame: Image.Image, padding: int) -> str | None:
    bbox = frame.getchannel("A").getbbox()
    if bbox is None:
        return "frame is fully transparent"
    left, top, right, bottom = bbox
    if left < padding or top < padding or right > frame.width - padding or bottom > frame.height - padding:
        return f"visible bbox {bbox} violates {padding}px padding"
    return None


def validate_apng(
    path: Path,
    expected_durations: list[int],
    expected_loop: int,
    expected_size: tuple[int, int],
    green_key: list[int],
    padding: int,
    source_frame_count: int,
    indices: list[int],
) -> dict[str, object]:
    frame_records = []
    errors: list[str] = []
    with Image.open(path) as animation:
        if animation.format != "PNG" or not animation.is_animated:
            errors.append("output is not an animated PNG")
        if animation.size != expected_size:
            errors.append(f"output size is {animation.size}, expected {expected_size}")
        if animation.n_frames != len(expected_durations):
            errors.append(f"output has {animation.n_frames} frames, expected {len(expected_durations)}")
        actual_loop = animation.info.get("loop")
        if actual_loop != expected_loop:
            errors.append(f"loop is {actual_loop}, expected {expected_loop}")

        actual_durations = []
        frame_hashes = []
        for index in range(animation.n_frames):
            animation.seek(index)
            animation.load()
            actual_durations.append(int(round(animation.info.get("duration", 0))))
            frame = animation.convert("RGBA")
            alpha_extrema = frame.getchannel("A").getextrema()
            edge_pixels = edge_alpha_count(frame)
            green_spill = chroma_adjacent_count(frame, green_key)
            frame_hash = visible_digest(frame)
            frame_hashes.append(frame_hash)
            frame_padding_error = padding_error(frame, padding)
            frame_records.append(
                {
                    "index": index,
                    "alpha_extrema": list(alpha_extrema),
                    "edge_alpha_pixels": edge_pixels,
                    "chroma_adjacent_pixels": green_spill,
                    "padding_error": frame_padding_error,
                    "sha256": frame_hash,
                }
            )
            if alpha_extrema != (0, 255):
                errors.append(f"frame {index} does not contain full transparency and opacity")
            if edge_pixels:
                errors.append(f"frame {index} has {edge_pixels} nontransparent edge pixels")
            if green_spill:
                errors.append(f"frame {index} has {green_spill} chroma-adjacent pixels")
            if frame_padding_error:
                errors.append(f"frame {index}: {frame_padding_error}")

        if actual_durations != expected_durations:
            errors.append(f"durations are {actual_durations}, expected {expected_durations}")
        if len(set(frame_hashes[:source_frame_count])) != source_frame_count:
            errors.append("two or more distinct source poses collapsed to identical frames")
        if len(indices) != len(frame_hashes):
            errors.append(f"encoded index count is {len(indices)}, decoded frame count is {len(frame_hashes)}")
        else:
            for frame_index, source_index in enumerate(indices):
                if frame_hashes[frame_index] != frame_hashes[source_index]:
                    errors.append(
                        f"frame {frame_index} does not match declared source pose {source_index}"
                    )

    return {
        "ok": not errors,
        "errors": errors,
        "durations_ms": actual_durations,
        "loop": actual_loop,
        "expected_loop": expected_loop,
        "frames": frame_records,
    }


def validate_webp_preview(
    path: Path,
    expected_frames: list[Image.Image],
    expected_durations: list[int],
    expected_loop: int,
) -> list[str]:
    errors = []
    expected_hashes = [visible_digest(frame) for frame in expected_frames]
    with Image.open(path) as animation:
        if animation.format != "WEBP" or not animation.is_animated:
            errors.append("preview is not an animated WebP")
        if animation.n_frames != len(expected_frames):
            errors.append(f"preview has {animation.n_frames} frames, expected {len(expected_frames)}")
        if animation.size != expected_frames[0].size:
            errors.append(f"preview size is {animation.size}, expected {expected_frames[0].size}")
        if animation.info.get("loop") != expected_loop:
            errors.append(f"preview loop is {animation.info.get('loop')}, expected {expected_loop}")
        durations = []
        hashes = []
        for index in range(animation.n_frames):
            animation.seek(index)
            animation.load()
            durations.append(int(round(animation.info.get("duration", 0))))
            hashes.append(visible_digest(animation.convert("RGBA")))
        if durations != expected_durations:
            errors.append(f"preview durations are {durations}, expected {expected_durations}")
        if hashes != expected_hashes:
            errors.append("preview decoded pixels differ from the APNG frame masters")
    return errors


def validate_still(path: Path, expected_frame: Image.Image, padding: int) -> list[str]:
    errors = []
    with Image.open(path) as still:
        if still.format != "WEBP" or still.n_frames != 1:
            errors.append("still is not a single-frame WebP")
        if still.size != expected_frame.size:
            errors.append(f"still size is {still.size}, expected {expected_frame.size}")
        rgba = still.convert("RGBA")
        if visible_digest(rgba) != visible_digest(expected_frame):
            errors.append("still decoded pixels differ from the selected APNG frame master")
        still_padding_error = padding_error(rgba, padding)
        if still_padding_error:
            errors.append(f"still: {still_padding_error}")
    return errors


def validate_contact_sheet(
    path: Path,
    frame_count: int,
    frame_size: tuple[int, int],
) -> list[str]:
    errors = []
    expected_size = (frame_size[0] * frame_count, frame_size[1] + 28)
    with Image.open(path) as sheet:
        if sheet.format != "PNG":
            errors.append("contact sheet is not PNG")
        if sheet.size != expected_size:
            errors.append(f"contact sheet size is {sheet.size}, expected {expected_size}")
    return errors


def save_animation(
    frames: list[Image.Image],
    durations: list[int],
    loop: int,
    apng_path: Path,
    preview_path: Path,
) -> None:
    apng_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        apng_path,
        format="PNG",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        disposal=0,
        blend=0,
        optimize=True,
        compress_level=9,
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        preview_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        lossless=True,
        quality=100,
        method=6,
    )


def temporary_sibling(path: Path, build_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.stem}.tmp-{build_id}{path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-apng", required=True, type=Path)
    parser.add_argument("--output-still", required=True, type=Path)
    parser.add_argument("--qa-dir", required=True, type=Path)
    parser.add_argument("--frame-count", required=True, type=int)
    parser.add_argument("--durations-ms", required=True)
    parser.add_argument("--loop-mode", required=True, choices=("ping-pong", "once"))
    parser.add_argument("--anchor-mode", required=True, choices=("baseline", "center", "motion"))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--alpha-cutoff", type=float, default=16.0 / 255.0)
    parser.add_argument("--expected-source-sha256", default="")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    if not 4 <= args.frame_count <= 8:
        parser.error("frame-count must be between 4 and 8")
    if not 32 <= args.width <= 1024 or not 32 <= args.height <= 1024:
        parser.error("width and height must each be between 32 and 1024")
    if args.padding < 0 or args.padding * 2 >= min(args.width, args.height):
        parser.error("padding must leave a positive interior")
    if not 16.0 / 255.0 <= args.alpha_cutoff < 0.5:
        parser.error("alpha-cutoff must be between 16/255 and 0.5")
    if args.expected_source_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_source_sha256):
        parser.error("expected-source-sha256 must be exactly 64 hexadecimal characters")
    if args.output_apng.suffix.lower() != ".png":
        parser.error("output-apng must use the .png extension for image/png delivery")
    if args.output_still.suffix.lower() != ".webp":
        parser.error("output-still must use the .webp extension")

    source_bytes = args.source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if args.expected_source_sha256 and source_hash.lower() != args.expected_source_sha256.lower():
        parser.error(
            f"source SHA-256 is {source_hash}, expected {args.expected_source_sha256.lower()}"
        )
    qa_outputs = {
        args.qa_dir / "preview.webp",
        args.qa_dir / "contact-sheet.png",
        args.qa_dir / "validation.json",
    }
    resolved_outputs = [args.output_apng.resolve(), args.output_still.resolve()]
    resolved_outputs.extend(path.resolve() for path in qa_outputs)
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("output and QA paths must all be distinct")
    if args.source.resolve() in set(resolved_outputs):
        parser.error("source path must not collide with an output or QA path")

    try:
        source_durations = parse_durations(args.durations_ms, args.frame_count)
    except ValueError as error:
        parser.error(str(error))
    indices = encoded_indices(args.frame_count, args.loop_mode)
    encoded_durations = [source_durations[index] for index in indices]
    expected_loop = 1 if args.loop_mode == "once" else 0

    with Image.open(BytesIO(source_bytes)) as opened:
        strip = opened.convert("RGBA")
    if strip.width < args.frame_count * 32:
        parser.error("source strip is too narrow for its declared frame count")
    entries, green_key, extraction_diagnostics = extract_components(
        strip, args.frame_count, args.alpha_cutoff
    )
    source_frames = render_frames(
        entries,
        width=args.width,
        height=args.height,
        padding=args.padding,
        anchor_mode=args.anchor_mode,
    )
    encoded_frames = [source_frames[index] for index in indices]

    args.qa_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.qa_dir / "preview.webp"
    contact_sheet_path = args.qa_dir / "contact-sheet.png"
    validation_path = args.qa_dir / "validation.json"
    build_id = uuid.uuid4().hex
    temporary_apng = temporary_sibling(args.output_apng, build_id)
    temporary_still = temporary_sibling(args.output_still, build_id)
    temporary_preview = temporary_sibling(preview_path, build_id)
    temporary_contact_sheet = temporary_sibling(contact_sheet_path, build_id)
    temporary_validation = temporary_sibling(validation_path, build_id)
    temporary_paths = [
        temporary_apng,
        temporary_still,
        temporary_preview,
        temporary_contact_sheet,
        temporary_validation,
    ]
    still_index = -1 if args.loop_mode == "once" else 0
    try:
        save_animation(
            encoded_frames,
            encoded_durations,
            expected_loop,
            temporary_apng,
            temporary_preview,
        )
        save_contact_sheet(source_frames, temporary_contact_sheet)
        source_frames[still_index].save(
            temporary_still,
            format="WEBP",
            lossless=True,
            quality=100,
            method=6,
        )

        validation = validate_apng(
            temporary_apng,
            expected_durations=encoded_durations,
            expected_loop=expected_loop,
            expected_size=(args.width, args.height),
            green_key=green_key,
            padding=args.padding,
            source_frame_count=args.frame_count,
            indices=indices,
        )
        secondary_errors = []
        secondary_errors.extend(
            validate_webp_preview(
                temporary_preview,
                encoded_frames,
                encoded_durations,
                expected_loop,
            )
        )
        secondary_errors.extend(
            validate_still(temporary_still, source_frames[still_index], args.padding)
        )
        secondary_errors.extend(
            validate_contact_sheet(
                temporary_contact_sheet,
                args.frame_count,
                (args.width, args.height),
            )
        )
        validation["errors"].extend(secondary_errors)
        validation["ok"] = not validation["errors"]
        validation.update(
            {
                "source": str(args.source.resolve()),
                "algorithm_version": 2,
                "source_sha256": source_hash,
                "estimated_green_key": green_key,
                "foreground_plateau": 8.0,
                "high_spill_ratio": 0.82,
                "extraction_method": "connected-components",
                "source_bboxes": [list(entry["full_bbox"]) for entry in entries],
                "component_minimum_ratio": 0.0005,
                **extraction_diagnostics,
                "canvas": {"width": args.width, "height": args.height, "padding": args.padding},
                "source_durations_ms": source_durations,
                "loop_mode": args.loop_mode,
                "runtime": {
                    "python": sys.version.split()[0],
                    "pillow": PIL.__version__,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "zlib": zlib.ZLIB_VERSION,
                },
                "source_frame_count": args.frame_count,
                "encoded_indices": indices,
                "anchor_mode": args.anchor_mode,
                "alpha_cutoff": args.alpha_cutoff,
                "apng": str(args.output_apng.resolve()),
                "apng_sha256": sha256(temporary_apng),
                "apng_bytes": temporary_apng.stat().st_size,
                "still": str(args.output_still.resolve()),
                "still_sha256": sha256(temporary_still),
                "still_bytes": temporary_still.stat().st_size,
                "contact_sheet": str(contact_sheet_path.resolve()),
                "contact_sheet_sha256": sha256(temporary_contact_sheet),
                "preview": str(preview_path.resolve()),
                "preview_sha256": sha256(temporary_preview),
            }
        )
        validation_json = json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
        temporary_validation.write_text(validation_json, encoding="utf-8")
        if not validation["ok"]:
            failed_validation = args.qa_dir / f"validation-failed-{build_id}.json"
            failed_validation.write_text(validation_json, encoding="utf-8")
            print(validation_json, end="")
            return 1

        for temporary, final in (
            (temporary_apng, args.output_apng),
            (temporary_still, args.output_still),
            (temporary_preview, preview_path),
            (temporary_contact_sheet, contact_sheet_path),
            (temporary_validation, validation_path),
        ):
            os.replace(temporary, final)
        print(validation_json, end="")
        return 0
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
