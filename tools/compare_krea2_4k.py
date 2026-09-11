"""Compare a conservative Krea2 4K render with a texture-rich 4K render.

The report separates high-frequency energy from low-frequency drift and measures
residual jumps at the actual VRAM-Canvas core boundaries.  The image sheets are
diagnostics only; they never alter either generated image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CROP_BOXES = {
    "A": ((650, 400, 1550, 1300), "青髪・角・瞳"),
    "B": ((1600, 900, 2500, 1800), "白髪・表情・衣装"),
    "C": ((900, 2500, 1900, 3500), "スライム・透明感"),
    "D": ((180, 2500, 1180, 3500), "石段・苔・裾"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def luminance(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    return values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722


def _moving_average(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    window = radius * 2 + 1
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zeros_shape = list(cumulative.shape)
    zeros_shape[axis] = 1
    cumulative = np.concatenate((np.zeros(zeros_shape, dtype=np.float64), cumulative), axis=axis)
    high = [slice(None)] * values.ndim
    low = [slice(None)] * values.ndim
    high[axis] = slice(window, None)
    low[axis] = slice(None, -window)
    return ((cumulative[tuple(high)] - cumulative[tuple(low)]) / window).astype(np.float32)


def box_lowpass(values: np.ndarray, radius: int) -> np.ndarray:
    return _moving_average(
        _moving_average(np.asarray(values, dtype=np.float32), radius, 1),
        radius,
        0,
    )


def box_mean_2d(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    size = radius * 2 + 1
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(
        np.cumsum(integral, axis=0, dtype=np.float64),
        axis=1,
        dtype=np.float64,
    )
    summed = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return (summed / float(size * size)).astype(np.float32)


def local_luma_ssim(first: np.ndarray, second: np.ndarray, radius: int = 5) -> float:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    mu_first = box_mean_2d(first, radius)
    mu_second = box_mean_2d(second, radius)
    var_first = np.maximum(box_mean_2d(first * first, radius) - mu_first * mu_first, 0.0)
    var_second = np.maximum(box_mean_2d(second * second, radius) - mu_second * mu_second, 0.0)
    covariance = box_mean_2d(first * second, radius) - mu_first * mu_second
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_first * mu_second + c1) * (2.0 * covariance + c2)
    denominator = (mu_first * mu_first + mu_second * mu_second + c1) * (var_first + var_second + c2)
    values = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )
    return float(np.mean(values, dtype=np.float64))


def full_rgb_error(first: Image.Image, second: Image.Image) -> dict[str, float | int]:
    if first.size != second.size:
        raise ValueError("full RGB error inputs must have the same size")
    absolute_chunks: list[np.ndarray] = []
    sum_abs = 0.0
    sum_square = 0.0
    sample_count = 0
    changed_count = 0
    maximum = 0
    for y0 in range(0, first.height, 256):
        y1 = min(first.height, y0 + 256)
        before = np.asarray(first.crop((0, y0, first.width, y1)), dtype=np.int16)
        after = np.asarray(second.crop((0, y0, second.width, y1)), dtype=np.int16)
        delta = after - before
        absolute = np.abs(delta).astype(np.uint8)
        sum_abs += float(np.sum(absolute, dtype=np.float64))
        sum_square += float(np.sum(np.square(delta.astype(np.float32)), dtype=np.float64))
        sample_count += int(absolute.size)
        changed_count += int(np.count_nonzero(np.any(delta != 0, axis=2)))
        maximum = max(maximum, int(np.max(absolute)))
        absolute_chunks.append(absolute.reshape(-1)[::8])
    sampled_absolute = np.concatenate(absolute_chunks)
    mae = sum_abs / sample_count
    mse = sum_square / sample_count
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "mae_8bit": mae,
        "mse_8bit": mse,
        "psnr_db": psnr,
        "p95_abs_rgb_delta_sampled": float(np.percentile(sampled_absolute, 95.0)),
        "p99_abs_rgb_delta_sampled": float(np.percentile(sampled_absolute, 99.0)),
        "max_abs_rgb_delta": maximum,
        "changed_pixel_count": changed_count,
        "changed_pixel_percent": changed_count / float(first.width * first.height) * 100.0,
    }


def structure_masks(baseline_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    dx = np.pad(np.abs(np.diff(baseline_y, axis=1)), ((0, 0), (0, 1)))
    dy = np.pad(np.abs(np.diff(baseline_y, axis=0)), ((0, 1), (0, 0)))
    magnitude = np.maximum(dx, dy)
    low = float(np.percentile(magnitude, 30.0))
    high = float(np.percentile(magnitude, 70.0))
    flat = magnitude <= low
    structured = magnitude >= high
    return (
        flat,
        structured,
        {
            "flat_gradient_threshold": low,
            "structured_gradient_threshold": high,
            "flat_fraction": float(np.mean(flat)),
            "structured_fraction": float(np.mean(structured)),
        },
    )


def seam_metrics(
    baseline: Image.Image,
    candidate: Image.Image,
    stage: dict,
) -> dict[str, float | int]:
    before = np.asarray(baseline, dtype=np.float32)
    after = np.asarray(candidate, dtype=np.float32)
    residual_y = luminance(after - before)
    global_jumps = np.concatenate(
        (
            np.abs(np.diff(residual_y, axis=1)).reshape(-1),
            np.abs(np.diff(residual_y, axis=0)).reshape(-1),
        )
    )
    tiles = stage.get("tiles") or []
    x_boundaries = sorted({int(value) for tile in tiles for value in (tile["core_x0"], tile["core_x1"]) if 0 < int(value) < residual_y.shape[1]})
    y_boundaries = sorted({int(value) for tile in tiles for value in (tile["core_y0"], tile["core_y1"]) if 0 < int(value) < residual_y.shape[0]})
    boundary_values = [np.abs(residual_y[:, x] - residual_y[:, x - 1]).reshape(-1) for x in x_boundaries]
    boundary_values.extend(np.abs(residual_y[y, :] - residual_y[y - 1, :]).reshape(-1) for y in y_boundaries)
    boundary_jumps = np.concatenate(boundary_values) if boundary_values else np.zeros(1, dtype=np.float32)
    global_p95 = float(np.percentile(global_jumps, 95.0))
    boundary_p95 = float(np.percentile(boundary_jumps, 95.0))
    return {
        "residual_global_jump_p95": global_p95,
        "residual_tile_boundary_jump_p95": boundary_p95,
        "residual_boundary_to_global_p95_ratio": boundary_p95 / max(global_p95, 1e-9),
        "x_boundary_count": len(x_boundaries),
        "y_boundary_count": len(y_boundaries),
    }


def candidate_metrics(
    baseline: Image.Image,
    candidate: Image.Image,
    stage: dict,
) -> dict:
    if baseline.size != candidate.size:
        raise ValueError("candidate size differs from the baseline")
    metrics = full_rgb_error(baseline, candidate)
    eval_size = (baseline.width // 2, baseline.height // 2)
    before = np.asarray(baseline.resize(eval_size, Image.Resampling.LANCZOS))
    after = np.asarray(candidate.resize(eval_size, Image.Resampling.LANCZOS))
    before_y = luminance(before)
    after_y = luminance(after)
    flat, structured, mask_info = structure_masks(before_y)
    before_hp1 = np.abs(before_y - box_lowpass(before_y, 1))
    after_hp1 = np.abs(after_y - box_lowpass(after_y, 1))
    low_drift = np.abs(box_lowpass(after_y, 12) - box_lowpass(before_y, 12))
    before_grad = np.concatenate(
        (
            np.abs(np.diff(before_y, axis=1)).reshape(-1),
            np.abs(np.diff(before_y, axis=0)).reshape(-1),
        )
    )
    after_grad = np.concatenate(
        (
            np.abs(np.diff(after_y, axis=1)).reshape(-1),
            np.abs(np.diff(after_y, axis=0)).reshape(-1),
        )
    )
    frequency = []
    for radius in (1, 2, 4, 8, 16):
        before_hp = np.abs(before_y - box_lowpass(before_y, radius))
        after_hp = np.abs(after_y - box_lowpass(after_y, radius))
        before_mean = float(np.mean(before_hp, dtype=np.float64))
        after_mean = float(np.mean(after_hp, dtype=np.float64))
        frequency.append(
            {
                "radius_eval_px": radius,
                "radius_final_px": radius * 2,
                "baseline_mean_abs": before_mean,
                "candidate_mean_abs": after_mean,
                "ratio": after_mean / max(before_mean, 1e-9),
            }
        )
    metrics.update(
        {
            "luma_ssim_half_resolution": local_luma_ssim(before_y, after_y),
            "baseline_highpass_mean": float(np.mean(before_hp1, dtype=np.float64)),
            "candidate_highpass_mean": float(np.mean(after_hp1, dtype=np.float64)),
            "highpass_mean_ratio": float(np.mean(after_hp1, dtype=np.float64) / max(np.mean(before_hp1, dtype=np.float64), 1e-9)),
            "baseline_highpass_p95": float(np.percentile(before_hp1, 95.0)),
            "candidate_highpass_p95": float(np.percentile(after_hp1, 95.0)),
            "highpass_p95_ratio": float(np.percentile(after_hp1, 95.0) / max(np.percentile(before_hp1, 95.0), 1e-9)),
            "flat_region_highpass_mean": float(np.mean(after_hp1[flat], dtype=np.float64)),
            "structured_region_highpass_mean": float(np.mean(after_hp1[structured], dtype=np.float64)),
            "low_frequency_luma_drift_mean": float(np.mean(low_drift, dtype=np.float64)),
            "low_frequency_luma_drift_p95": float(np.percentile(low_drift, 95.0)),
            "baseline_gradient_p95": float(np.percentile(before_grad, 95.0)),
            "candidate_gradient_p95": float(np.percentile(after_grad, 95.0)),
            "gradient_p95_ratio": float(np.percentile(after_grad, 95.0) / max(np.percentile(before_grad, 95.0), 1e-9)),
            "frequency_energy": frequency,
            "metric_resolution": list(eval_size),
            "structure_mask": mask_info,
        }
    )
    metrics.update(seam_metrics(baseline, candidate, stage))
    return metrics


def crop_metrics(baseline: Image.Image, old: Image.Image, new: Image.Image) -> dict:
    result = {}
    for name, (box, label) in CROP_BOXES.items():
        base_crop = baseline.crop(box)
        old_crop = old.crop(box)
        new_crop = new.crop(box)
        eval_size = (box[2] - box[0], box[3] - box[1])
        base_y = luminance(np.asarray(base_crop))
        old_y = luminance(np.asarray(old_crop))
        new_y = luminance(np.asarray(new_crop))
        old_hp = np.abs(old_y - box_lowpass(old_y, 2))
        new_hp = np.abs(new_y - box_lowpass(new_y, 2))
        result[name] = {
            "label": label,
            "box": list(box),
            "size": list(eval_size),
            "old_highpass_mean": float(np.mean(old_hp, dtype=np.float64)),
            "new_highpass_mean": float(np.mean(new_hp, dtype=np.float64)),
            "new_to_old_highpass_ratio": float(np.mean(new_hp, dtype=np.float64) / max(np.mean(old_hp, dtype=np.float64), 1e-9)),
            "old_luma_ssim_to_baseline": local_luma_ssim(base_y, old_y),
            "new_luma_ssim_to_baseline": local_luma_ssim(base_y, new_y),
            "old_to_new_luma_ssim": local_luma_ssim(old_y, new_y),
        }
    return result


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothR.ttc"),
        Path("C:/Windows/Fonts/meiryob.ttc" if bold else "C:/Windows/Fonts/meiryo.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size, index=0)
    return ImageFont.load_default()


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, "white")
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def make_overview(
    baseline: Image.Image,
    old: Image.Image,
    new: Image.Image,
    output: Path,
) -> None:
    canvas = Image.new("RGB", (2100, 1240), "#F4F6F8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(42, bold=True)
    label_font = load_font(28, bold=True)
    note_font = load_font(22)
    draw.text((1050, 32), "同一1K入力からの4K比較", font=title_font, fill="#17202A", anchor="ma")
    panels = [
        ("Lanczos 4K基準", "生成なし・補間のみ", baseline),
        ("従来4K", "構造安全寄り", old),
        ("Texture Rich 4K", "書き込み・高周波寄り", new),
    ]
    for index, (label, note, image) in enumerate(panels):
        x = 60 + index * 680
        panel = fit(image, (600, 980))
        canvas.paste(panel, (x, 110))
        draw.rectangle((x, 110, x + 600, 1090), outline="#78838D", width=3)
        draw.text((x + 300, 1125), label, font=label_font, fill="#176B8A", anchor="ma")
        draw.text((x + 300, 1170), note, font=note_font, fill="#59646F", anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_crop_strip(
    name: str,
    label: str,
    box: tuple[int, int, int, int],
    baseline: Image.Image,
    old: Image.Image,
    new: Image.Image,
    output: Path,
) -> None:
    crops = [baseline.crop(box), old.crop(box), new.crop(box)]
    panel_width, panel_height = crops[0].size
    header = 130
    gap = 24
    canvas = Image.new("RGB", (panel_width * 3 + gap * 4, panel_height + header + 55), "#F4F6F8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(32, bold=True)
    label_font = load_font(23, bold=True)
    draw.text(
        (gap, 20),
        f"{name}  {label} — 原寸crop",
        font=title_font,
        fill="#17202A",
        anchor="la",
    )
    labels = ["Lanczos基準", "従来4K", "Texture Rich 4K"]
    colors = ["#59646F", "#376C82", "#A34D1F"]
    for index, (crop, panel_label, color) in enumerate(zip(crops, labels, colors)):
        x = gap + index * (panel_width + gap)
        canvas.paste(crop, (x, header))
        draw.rectangle((x, header, x + panel_width, header + panel_height), outline=color, width=4)
        draw.text(
            (x + panel_width // 2, 88),
            panel_label,
            font=label_font,
            fill=color,
            anchor="mm",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_contact_sheet(
    baseline: Image.Image,
    old: Image.Image,
    new: Image.Image,
    output: Path,
) -> None:
    canvas = Image.new("RGB", (1740, 2100), "#F4F6F8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    label_font = load_font(24, bold=True)
    region_font = load_font(25, bold=True)
    draw.text((870, 28), "4K局所書き込み比較", font=title_font, fill="#17202A", anchor="ma")
    labels = ["Lanczos基準", "従来4K", "Texture Rich 4K"]
    for index, label in enumerate(labels):
        draw.text((350 + index * 520, 82), label, font=label_font, fill="#176B8A", anchor="ma")
    for row, (name, (box, region)) in enumerate(CROP_BOXES.items()):
        y = 125 + row * 485
        draw.text((35, y + 205), f"{name}\n{region}", font=region_font, fill="#27323C", anchor="lm", spacing=8)
        for col, image in enumerate((baseline, old, new)):
            x = 155 + col * 520
            panel = fit(image.crop(box), (470, 440))
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + 470, y + 440), outline="#818B94", width=2)
    draw.text(
        (870, 2070),
        "各cropは同じ4K座標。表示用に等寸法へ縮小。原寸版も同フォルダに保存。",
        font=load_font(18),
        fill="#5B6670",
        anchor="ms",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_difference_heatmap(old: Image.Image, new: Image.Image, output: Path) -> float:
    size = (old.width // 2, old.height // 2)
    before = np.asarray(old.resize(size, Image.Resampling.LANCZOS), dtype=np.int16)
    after = np.asarray(new.resize(size, Image.Resampling.LANCZOS), dtype=np.int16)
    magnitude = np.mean(np.abs(after - before), axis=2, dtype=np.float32)
    scale = float(np.percentile(magnitude, 99.0)) or 1.0
    t = np.clip(magnitude / scale, 0.0, 1.0)
    rgb = np.stack(
        (
            np.clip(2.2 * t, 0.0, 1.0),
            np.clip(2.0 * (t - 0.22), 0.0, 1.0),
            np.clip(1.3 - 2.0 * t, 0.0, 1.0),
        ),
        axis=2,
    )
    heatmap = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")
    canvas = Image.new("RGB", (1200, 1850), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (600, 30),
        "従来4K → Texture Rich 4K 差分強度",
        font=load_font(31, bold=True),
        fill="#17202A",
        anchor="ma",
    )
    panel = fit(heatmap, (1080, 1620))
    canvas.paste(panel, (60, 90))
    draw.rectangle((60, 90, 1140, 1710), outline="#818B94", width=2)
    draw.text(
        (600, 1765),
        f"青=小、黄〜赤=大（半解像度RGB差分、p99={scale:.2f} code）",
        font=load_font(22),
        fill="#59646F",
        anchor="mm",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return scale


def final_stage(manifest: dict, expected_size: tuple[int, int]) -> dict:
    reports = manifest.get("stage_reports") or []
    if not reports:
        raise ValueError("manifest has no stage reports")
    stage = reports[-1]
    if tuple(stage.get("size") or ()) != expected_size:
        raise ValueError(f"final manifest stage size {stage.get('size')} differs from {expected_size}")
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--new-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = load_rgb(args.source)
    old = load_rgb(args.old)
    new = load_rgb(args.new)
    if old.size != new.size:
        raise ValueError(f"old/new dimensions differ: {old.size} versus {new.size}")
    baseline = source.resize(new.size, Image.Resampling.LANCZOS)
    old_manifest = read_json(args.old_manifest)
    new_manifest = read_json(args.new_manifest)
    old_stage = final_stage(old_manifest, old.size)
    new_stage = final_stage(new_manifest, new.size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = args.output_dir / "overview_lanczos_old_texture_rich.png"
    contact_path = args.output_dir / "crop_contact_sheet.png"
    heatmap_path = args.output_dir / "old_vs_texture_rich_difference_heatmap.png"
    make_overview(baseline, old, new, overview_path)
    make_contact_sheet(baseline, old, new, contact_path)
    for name, (box, label) in CROP_BOXES.items():
        make_crop_strip(
            name,
            label,
            box,
            baseline,
            old,
            new,
            args.output_dir / f"crop_{name}_original_pixels.png",
        )
    heatmap_scale = make_difference_heatmap(old, new, heatmap_path)

    old_metrics = candidate_metrics(baseline, old, old_stage)
    new_metrics = candidate_metrics(baseline, new, new_stage)
    pairwise = full_rgb_error(old, new)
    pair_eval_size = (old.width // 2, old.height // 2)
    old_y = luminance(np.asarray(old.resize(pair_eval_size, Image.Resampling.LANCZOS)))
    new_y = luminance(np.asarray(new.resize(pair_eval_size, Image.Resampling.LANCZOS)))
    pairwise["luma_ssim_half_resolution"] = local_luma_ssim(old_y, new_y)
    regions = crop_metrics(baseline, old, new)
    new_to_old_hp = new_metrics["candidate_highpass_mean"] / max(old_metrics["candidate_highpass_mean"], 1e-9)
    new_to_old_flat_hp = new_metrics["flat_region_highpass_mean"] / max(old_metrics["flat_region_highpass_mean"], 1e-9)
    new_to_old_structured_hp = new_metrics["structured_region_highpass_mean"] / max(old_metrics["structured_region_highpass_mean"], 1e-9)
    report = {
        "format_version": 1,
        "method": {
            "baseline": "same-size sRGB Lanczos upscale of the exact 1K source",
            "ssim": "11x11 uniform-window luminance SSIM at half resolution",
            "highpass": "absolute luminance residual from a separable box low-pass",
            "seam": "p95 jump of generated residual at planned core boundaries divided by global residual-jump p95",
            "interpretation": "High-frequency energy includes coherent drawing, sharpening, and noise; it is not a semantic-detail score by itself.",
        },
        "artifacts": {
            "source": {"path": str(args.source), "size": list(source.size), "sha256": sha256(args.source)},
            "old": {"path": str(args.old), "size": list(old.size), "sha256": sha256(args.old)},
            "new": {"path": str(args.new), "size": list(new.size), "sha256": sha256(args.new)},
            "old_manifest": str(args.old_manifest),
            "new_manifest": str(args.new_manifest),
            "overview": str(overview_path),
            "contact_sheet": str(contact_path),
            "difference_heatmap": str(heatmap_path),
        },
        "old_4k": old_metrics,
        "texture_rich_4k": new_metrics,
        "old_vs_texture_rich": pairwise,
        "texture_rich_to_old_ratios": {
            "whole_image_highpass_mean": new_to_old_hp,
            "flat_region_highpass_mean": new_to_old_flat_hp,
            "structured_region_highpass_mean": new_to_old_structured_hp,
            "low_frequency_drift_mean": new_metrics["low_frequency_luma_drift_mean"] / max(old_metrics["low_frequency_luma_drift_mean"], 1e-9),
        },
        "regions": regions,
        "difference_heatmap_p99_code_value": heatmap_scale,
        "checks": {
            "texture_energy_increased": new_to_old_hp > 1.0,
            "structured_texture_energy_increased": new_to_old_structured_hp > 1.0,
            "all_four_regions_increased": all(region["new_to_old_highpass_ratio"] > 1.0 for region in regions.values()),
            "new_residual_boundary_ratio_below_reference_1_5": new_metrics["residual_boundary_to_global_p95_ratio"] < 1.5,
            "subject_and_seam_visual_review_required": True,
        },
    }
    report_path = args.output_dir / "texture_rich_4k_qa_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report["checks"], ensure_ascii=False, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                "texture_rich_to_old_ratios": report["texture_rich_to_old_ratios"],
                "old_ssim": old_metrics["luma_ssim_half_resolution"],
                "new_ssim": new_metrics["luma_ssim_half_resolution"],
                "old_seam_ratio": old_metrics["residual_boundary_to_global_p95_ratio"],
                "new_seam_ratio": new_metrics["residual_boundary_to_global_p95_ratio"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
