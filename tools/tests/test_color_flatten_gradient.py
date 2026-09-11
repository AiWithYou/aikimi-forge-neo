import unittest

import cv2
import numpy as np
from PIL import Image

from modules.color_flatten import smooth_gradient_pil


def gradient_fixture(width: int = 320, height: int = 256, *, noisy: bool) -> tuple[Image.Image, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    top = np.array([150.0, 125.0, 119.0], dtype=np.float32)
    bottom = np.array([221.0, 155.0, 100.0], dtype=np.float32)
    clean = top + (bottom - top) * y
    clean = clean + x * np.array([1.5, 0.5, -0.5], dtype=np.float32)
    clean = np.broadcast_to(clean, (height, width, 3)).copy()
    clean_u8 = np.clip(np.rint(clean), 0, 255).astype(np.uint8)
    if not noisy:
        return Image.fromarray(clean_u8, mode="RGB"), clean_u8

    rng = np.random.default_rng(20260710)
    luma_noise = rng.normal(0.0, 1.55, size=(height, width, 1)).astype(np.float32)
    color_noise = rng.normal(0.0, 0.25, size=(height, width, 3)).astype(np.float32)
    checker = ((np.indices((height, width)).sum(axis=0) % 2) * 2 - 1).astype(np.float32)
    field = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=5.0, sigmaY=5.0)
    field /= max(float(np.std(field)), 1e-6)
    low_frequency = field[..., None] * np.array([1.3, 0.8, 0.5], dtype=np.float32)
    noisy_rgb = clean + luma_noise + color_noise + checker[..., None] * 0.65 + low_frequency
    noisy_u8 = np.clip(np.rint(noisy_rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_u8, mode="RGB"), clean_u8


def lab_highpass_rms(rgb: np.ndarray, sigma: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    highpass = lab - cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    highpass = highpass[8:-8, 8:-8]
    return np.sqrt(np.mean(highpass * highpass, axis=(0, 1)))


def rgb_rmse(first: np.ndarray, second: np.ndarray) -> float:
    difference = first.astype(np.float32) - second.astype(np.float32)
    return float(np.sqrt(np.mean(difference * difference)))


class SmoothGradientTests(unittest.TestCase):
    def test_reconstructs_smooth_surface_from_ai_style_noise(self):
        image, clean = gradient_fixture(noisy=True)
        source = np.asarray(image)

        result = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=12.0,
            detail_threshold=8.0,
            edge_protect=True,
        )
        output = np.asarray(result)

        source_hp1 = lab_highpass_rms(source, 1.0)
        output_hp1 = lab_highpass_rms(output, 1.0)
        source_hp8 = lab_highpass_rms(source, 8.0)
        output_hp8 = lab_highpass_rms(output, 8.0)
        radius_one = np.asarray(
            smooth_gradient_pil(
                image,
                strength=1.0,
                radius=1.0,
                detail_threshold=8.0,
                edge_protect=True,
            )
        )
        radius_one_hp8 = lab_highpass_rms(radius_one, 8.0)
        self.assertLess(output_hp1[0], source_hp1[0] * 0.40)
        self.assertLess(output_hp1[1], source_hp1[1] * 0.80)
        self.assertLess(output_hp1[2], source_hp1[2] * 0.80)
        self.assertLess(output_hp8[0], source_hp8[0] * 0.50)
        self.assertLess(rgb_rmse(output, clean), rgb_rmse(source, clean) * 0.60)
        self.assertLess(output_hp8[0], radius_one_hp8[0] * 0.50)
        self.assertLess(rgb_rmse(output, clean), rgb_rmse(radius_one, clean) * 0.75)

    def test_strength_zero_is_bit_identical_for_rgb_and_rgba(self):
        rgb, _ = gradient_fixture(noisy=True)
        rgba = rgb.convert("RGBA")
        alpha = np.linspace(0, 255, rgb.width, dtype=np.uint8)[None, :]
        rgba.putalpha(Image.fromarray(np.repeat(alpha, rgb.height, axis=0), mode="L"))

        rgb_result = smooth_gradient_pil(rgb, strength=0.0)
        rgba_result = smooth_gradient_pil(rgba, strength=0.0)

        np.testing.assert_array_equal(np.asarray(rgb_result), np.asarray(rgb))
        np.testing.assert_array_equal(np.asarray(rgba_result), np.asarray(rgba))

    def test_clean_gradient_is_not_distorted(self):
        image, clean = gradient_fixture(noisy=False)

        result = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=12.0,
            detail_threshold=8.0,
            edge_protect=True,
        )
        output = np.asarray(result)
        difference = np.abs(output.astype(np.int16) - clean.astype(np.int16))

        self.assertLessEqual(float(np.mean(difference)), 0.5)
        self.assertLessEqual(int(np.max(difference)), 3)

    def test_hard_color_boundary_remains_sharp(self):
        source = np.empty((192, 320, 3), dtype=np.uint8)
        source[:, :160] = [55, 95, 135]
        source[:, 160:] = [225, 165, 75]
        image = Image.fromarray(source, mode="RGB")

        result = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=12.0,
            detail_threshold=8.0,
            edge_protect=True,
        )
        output = np.asarray(result)

        np.testing.assert_array_equal(output[:, 158:162], source[:, 158:162])
        left_mean = output[:, 156:160].astype(np.float32).mean(axis=(0, 1))
        right_mean = output[:, 160:164].astype(np.float32).mean(axis=(0, 1))
        source_contrast = np.linalg.norm(source[0, 160].astype(np.float32) - source[0, 159].astype(np.float32))
        result_contrast = np.linalg.norm(right_mean - left_mean)
        self.assertGreaterEqual(result_contrast, source_contrast * 0.95)

    def test_thin_high_contrast_line_does_not_create_halo(self):
        source = np.full((192, 320, 3), 220, dtype=np.uint8)
        source[:, 160] = 12
        image = Image.fromarray(source, mode="RGB")

        result = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=12.0,
            detail_threshold=8.0,
            edge_protect=True,
        )
        output = np.asarray(result)

        np.testing.assert_array_equal(output[:, 160], source[:, 160])
        background = np.delete(output, 160, axis=1)
        self.assertLessEqual(
            int(np.max(np.abs(background.astype(np.int16) - 220))), 1
        )

    def test_alpha_and_transparent_hidden_rgb_are_preserved(self):
        visible, _ = gradient_fixture(width=320, height=192, noisy=True)
        rgba = np.asarray(visible.convert("RGBA")).copy()
        rgba[:, :96, :3] = [255, 0, 255]
        rgba[:, :96, 3] = 0
        rgba[:, 96:112, 3] = np.linspace(0, 255, 16, dtype=np.uint8)[None, :]
        image = Image.fromarray(rgba, mode="RGBA")

        result = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=8.0,
            detail_threshold=8.0,
            edge_protect=True,
        )
        output = np.asarray(result)

        np.testing.assert_array_equal(output[..., 3], rgba[..., 3])
        np.testing.assert_array_equal(output[:, :96, :3], rgba[:, :96, :3])
        np.testing.assert_array_equal(output[:, 96:114], rgba[:, 96:114])

    def test_tiled_processing_has_no_visible_seam(self):
        image, _ = gradient_fixture(width=384, height=256, noisy=True)

        tiled = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=8.0,
            detail_threshold=8.0,
            tile_size=128,
        )
        single_tile = smooth_gradient_pil(
            image,
            strength=1.0,
            radius=8.0,
            detail_threshold=8.0,
            tile_size=1024,
        )
        difference = np.abs(np.asarray(tiled).astype(np.int16) - np.asarray(single_tile).astype(np.int16))

        self.assertLessEqual(int(np.max(difference)), 1)

    def test_invalid_parameters_fail_clearly(self):
        image = Image.new("RGB", (16, 16), (120, 100, 90))

        for strength in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    smooth_gradient_pil(image, strength=strength)
        for radius in (0.0, -1.0, 64.5, float("nan"), float("inf")):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    smooth_gradient_pil(image, radius=radius)
        for threshold in (0.0, -1.0, 24.5, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    smooth_gradient_pil(image, detail_threshold=threshold)


if __name__ == "__main__":
    unittest.main()
