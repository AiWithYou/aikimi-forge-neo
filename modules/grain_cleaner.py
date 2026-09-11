"""原寸の多重スケール残差を、構造を保護しながら部分減衰する。"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import asdict, dataclass
from threading import RLock

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps


@dataclass(frozen=True)
class GrainSettings:
    strength: float = 0.50
    preserve_detail: float = 0.75
    chroma_strength: float = 0.60
    grain_scale: float = 1.0
    mode: str = "auto"

    def validate(self):
        for key in ("strength", "preserve_detail", "chroma_strength", "grain_scale"):
            value = getattr(self, key)
            low, high = (0.5, 2.0) if key == "grain_scale" else (0.0, 1.0)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} は {low}～{high} で指定してください")
        if self.mode not in ("auto", "sample"):
            raise ValueError("mode は auto または sample を指定してください")


def smoothstep(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def gaussian(value, sigma):
    size = 2 * int(np.ceil(4 * sigma)) + 1
    return cv2.GaussianBlur(value, (size, size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def gradient(value, scale):
    return tuple(
        cv2.Scharr(value, cv2.CV_32F, dx, dy, borderType=cv2.BORDER_REFLECT_101) * (scale / 32)
        for dx, dy in ((1, 0), (0, 1))
    )


def gradient_norms(value, scale):
    gx, gy = gradient(value, scale)
    square = gx * gx + gy * gy
    return np.sqrt(square[..., 0]), np.sqrt(square[..., 1:].sum(axis=2))


def prepare_image(image):
    if getattr(image, "is_animated", False) or image.mode not in ("RGB", "RGBA"):
        raise ValueError("Grain Cleaner は静止した8bit RGB / 完全不透明RGBA画像に対応します")
    if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
        raise ValueError("Grain Cleaner は透明・半透明画像に対応していません")
    image = ImageOps.exif_transpose(image)
    if min(image.size) < 64:
        raise ValueError("Grain Cleaner は縦横64画素以上の画像が必要です")
    profile = image.info.get("icc_profile")
    if profile:
        image = ImageCms.profileToProfile(
            image, ImageCms.ImageCmsProfile(io.BytesIO(profile)), ImageCms.createProfile("sRGB"), outputMode=image.mode
        )
    image.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return image, "ICC converted to sRGB" if profile else "assumed sRGB (no ICC)"


def mask_array(mask, size):
    if mask is None:
        return None
    if mask.mode != "L" or mask.size != size:
        raise ValueError("マスクは向き補正後の画像と同寸法の8bitグレースケールで指定してください")
    return np.asarray(mask, dtype=np.float32) / 255


def _grid_coordinates(shape):
    """64pxパッチ中心(31.5)に合わせ、端では最近傍の値を延長する。"""
    height, width = shape
    xx, yy = np.meshgrid(
        (np.arange(width, dtype=np.float32) - 31.5) / 32, (np.arange(height, dtype=np.float32) - 31.5) / 32
    )
    return xx, yy


def _reference(bands, features, g_mid, r_mid, mode, roi):
    height, width = g_mid.shape
    if mode == "sample":
        x, y, w, h = roi
        patches = [(0, 0, (slice(y, y + h), slice(x, x + w)))]
        grid_shape = (1, 1)
    else:
        ys, xs = range(0, height - 63, 32), range(0, width - 63, 32)
        grid_shape = (len(ys), len(xs))
        patches = [(iy, ix, (slice(y, y + 64), slice(x, x + 64))) for iy, y in enumerate(ys) for ix, x in enumerate(xs)]
    q = np.zeros(grid_shape, np.float32)
    values = np.zeros((*grid_shape, 6 + len(features)), np.float32)
    for iy, ix, area in patches:
        if mode == "auto" and (np.percentile(g_mid[area], 90) >= 0.012 or np.percentile(r_mid[area], 90) >= 0.008):
            continue
        q[iy, ix] = 1
        for j, band in enumerate(bands):
            z = band[area].reshape(-1, 3)
            values[iy, ix, j * 3 : j * 3 + 3] = 1.4826 * np.median(abs(z - np.median(z, axis=0)), axis=0)
        values[iy, ix, 6:] = [np.percentile(f[area], 90) for f in features]
    count = int(q.sum())
    if mode == "auto" and count < 4:
        return None, None, count
    if mode == "sample":
        return values[0, 0], np.ones((height, width), np.float32), count
    support = gaussian(q, 1.0)
    coordinates = _grid_coordinates((height, width))
    # 各面を最終配列へ書き込み、原寸11枚をstackで再コピーしない。
    maps = np.empty((values.shape[2], height, width), np.float32)
    for i in range(values.shape[2]):
        v = values[..., i]
        smoothed = gaussian(q * (v * v if i < 6 else v), 1.0) / np.maximum(support, 1e-12)
        cv2.remap(
            np.sqrt(np.maximum(smoothed, 0)) if i < 6 else smoothed,
            *coordinates,
            cv2.INTER_LINEAR,
            dst=maps[i],
            borderMode=cv2.BORDER_REPLICATE,
        )
    return (
        np.moveaxis(maps, 0, -1),
        cv2.remap(smoothstep(0.10, 0.50, support), *coordinates, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
        count,
    )


class GrainAnalysisCache:
    """直近の1画像だけ保持する。強度・マスク・診断表示は解析キーに含めない。"""

    def __init__(self):
        self.lock = RLock()
        self.key = None
        self.analysis = None

    def clear(self):
        with self.lock:
            self.key = None
            self.analysis = None

    def get(self, baseline, settings, sample_roi):
        key = (
            baseline.mode,
            baseline.size,
            hashlib.sha256(baseline.tobytes()).digest(),
            settings.grain_scale,
            settings.mode,
            sample_roi,
        )
        with self.lock:
            if key == self.key:
                return self.analysis, True
            self.clear()
            self.analysis = analyze_grain(baseline, settings, sample_roi)
            self.key = key
            return self.analysis, False


def analyze_grain(baseline, settings, sample_roi):
    original = np.asarray(baseline.convert("RGB"))
    rgb = original.astype(np.float32) / 255
    units = np.array([100, 128, 128], np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab) / units
    r = settings.grain_scale
    b1, b2, b3 = (gaussian(lab, sigma * r) for sigma in (0.8, 1.6, 3.2))
    bands = (lab - b1, b1 - b2)
    d2 = b2 - b3
    del b1, b3
    gl, gc = gradient_norms(b2, r)
    g_mid = np.maximum(gl, 0.5 * gc)
    r_mid = np.sqrt(np.maximum(gaussian(d2[..., 0] ** 2 + 0.5 * np.sum(d2[..., 1:] ** 2, axis=2), 2 * r), 0))
    del b2, d2, gl, gc
    features = []
    for sigma in (0.6, 1.2):
        blurred = gaussian(lab, sigma * r)
        features.extend(gradient_norms(blurred, r))
    gx, gy = gradient(blurred[..., 0], r)
    del blurred
    jxx, jyy, jxy = (gaussian(v, 2 * r) for v in (gx * gx, gy * gy, gx * gy))
    energy = np.sqrt(np.maximum(jxx + jyy, 0))
    coherence = np.clip(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (energy**2 + 1e-12), 0, 1)
    del gx, gy, jxx, jyy, jxy
    features.append(energy)
    refs, support, count = _reference(bands, features, g_mid, r_mid, settings.mode, sample_roi)
    if refs is None:
        return {"reference_count": count}
    tau = [refs[..., i : i + 3] for i in (0, 3)]
    protection = np.zeros(original.shape[:2], np.float32)
    for i, feature in enumerate(features[:4]):
        eta = refs[..., 6 + i]
        low, high = (0.003, 0.012) if i % 2 == 0 else (0.002, 0.008)
        np.maximum(protection, smoothstep(np.maximum(low, 2 * eta), np.maximum(high, 4 * eta), feature), out=protection)
    eta_e = refs[..., 10]
    np.maximum(
        protection,
        smoothstep(0.45, 0.80, coherence) * smoothstep(np.maximum(0.001, eta_e), np.maximum(0.006, 2 * eta_e), energy),
        out=protection,
    )
    for z, t in (
        (abs(bands[0][..., 0]), tau[0][..., 0]),
        (np.linalg.norm(bands[0][..., 1:], axis=-1), np.linalg.norm(tau[0][..., 1:], axis=-1)),
    ):
        np.maximum(protection, smoothstep(np.maximum(0.003, 3 * t), np.maximum(0.006, 6 * t), z), out=protection)
    radius = int(np.ceil(r))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)
    protection = np.maximum(protection, gaussian(cv2.dilate(protection, kernel), 0.6 * r))
    # eta用の5面をキャッシュへ持ち越さない。
    tau = tuple(t.copy() for t in tau)
    return dict(
        reference_count=count,
        lab=lab,
        bands=bands,
        tau=tau,
        protection=protection,
        support=support,
        flat=1 - smoothstep(0.003, 0.015, r_mid),
        amplitude_median=[np.median(t.reshape(-1, 3), axis=0).tolist() for t in tau],
    )


def clean_grain(
    image, settings=None, *, sample_roi=None, apply_mask=None, protect_mask=None, diagnostics=False, cache=None
):
    settings = settings or GrainSettings()
    settings.validate()
    started = time.perf_counter()
    baseline, color_status = prepare_image(image)
    u = mask_array(apply_mask, baseline.size)
    h = mask_array(protect_mask, baseline.size)
    if settings.mode == "sample":
        if sample_roi is None or len(sample_roi) != 4 or any(int(v) != v for v in sample_roi):
            raise ValueError("sample は整数の x, y, width, height が必要です")
        x, y, w, height = sample_roi = tuple(int(v) for v in sample_roi)
        if min(x, y) < 0 or min(w, height) < 32 or x + w > baseline.width or y + height > baseline.height:
            raise ValueError("見本範囲は画像内で縦横32画素以上にしてください")
    report = {
        "algorithm_version": "0.2",
        "settings": asdict(settings),
        "size": list(baseline.size),
        "color": color_status,
        "sample_roi": sample_roi,
        "apply_mask": u is not None,
        "protect_mask": h is not None,
    }
    views = {}

    def finish(output, status):
        report.update(status=status, elapsed_seconds=round(time.perf_counter() - started, 4))
        if diagnostics and not views:
            report["diagnostics_omitted"] = "変更がないため診断画像を省略しました"
        output.info.update(baseline.info)
        return output, report, views

    if settings.strength == 0 or (u is not None and not u.any()) or (h is not None and np.all(h == 1)):
        report["unchanged_reason"] = (
            "強度が0です"
            if settings.strength == 0
            else "処理マスクが全面黒です"
            if u is not None and not u.any()
            else "全面が保護されています"
        )
        return finish(baseline, "UNCHANGED")
    analysis_started = time.perf_counter()
    if cache is None:
        analysis, hit = analyze_grain(baseline, settings, sample_roi), False
    else:
        analysis, hit = cache.get(baseline, settings, sample_roi)
    report.update(
        reference_count=analysis["reference_count"],
        analysis_cache_hit=hit,
        analysis_seconds=round(time.perf_counter() - analysis_started, 4),
    )
    if analysis["reference_count"] < 4 and settings.mode == "auto":
        return finish(baseline, "INSUFFICIENT_REFERENCE")
    original = np.asarray(baseline.convert("RGB"))
    rgb = original.astype(np.float32) / 255
    units = np.array([100, 128, 128], np.float32)
    lab, bands, tau = analysis["lab"], analysis["bands"], analysis["tau"]
    support, protection = analysis["support"], analysis["protection"]
    mask = support * analysis["flat"] * (1 - protection) ** (1 + 2 * settings.preserve_detail)
    if u is not None:
        mask *= u
    if h is not None:
        mask *= 1 - h
    delta = np.zeros_like(lab)
    suppression = np.zeros(original.shape[:2], np.float32)
    for j, band in enumerate(bands):
        for channels, amplitude, cap, strength in (
            (0, tau[j][..., 0], (0.8, 0.3)[j], settings.strength),
            (
                slice(1, 3),
                np.linalg.norm(tau[j][..., 1:], axis=-1),
                (0.75, 0.25)[j],
                settings.strength * settings.chroma_strength,
            ),
        ):
            if strength == 0:
                continue
            z = band[..., channels]
            magnitude = abs(z) if channels == 0 else np.linalg.norm(z, axis=-1)
            safe = np.maximum(amplitude, 0.001)
            alpha = (
                strength
                * cap
                * mask
                * smoothstep(0.001, 0.003, amplitude)
                * (1 - smoothstep(1.5 * safe, 3 * safe, magnitude))
            )
            delta[..., channels] -= (alpha if channels == 0 else alpha[..., None]) * z
            np.maximum(suppression, alpha, out=suppression)
    chroma_length = np.linalg.norm(delta[..., 1:], axis=-1)
    report["clipped_fraction"] = float(np.mean((abs(delta[..., 0]) > 0.02) | (chroma_length > 2 / 128)))
    delta[..., 0] = np.clip(delta[..., 0], -0.02, 0.02)
    delta[..., 1:] *= np.minimum(1, (2 / 128) / np.maximum(chroma_length, 1e-12))[..., None]
    result = rgb + cv2.cvtColor((lab + delta) * units, cv2.COLOR_Lab2RGB) - cv2.cvtColor(lab * units, cv2.COLOR_Lab2RGB)
    result = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
    result[suppression == 0] = original[suppression == 0]
    report["changed_fraction"] = float(np.mean(np.any(result != original, axis=-1)))
    report["amplitude_median"] = analysis["amplitude_median"]
    output = Image.fromarray(result)
    if baseline.mode == "RGBA":
        output.putalpha(baseline.getchannel("A"))
    if diagnostics:
        views["removed_x8"] = Image.fromarray(
            np.rint(np.clip(128 + 8 * (original.astype(np.float32) - result), 0, 255)).astype(np.uint8)
        )
        combined = protection if h is None else np.maximum(protection, h)
        for name, value in (("suppression", suppression), ("protection", combined), ("support", support)):
            views[name] = Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8))
    return finish(output, "APPLIED" if report["changed_fraction"] else "LOW_ACTIVITY")
