"""CPU/Philox seed resize keeps leading axes and fills the centered crop."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from modules import rng_philox
from tools.tests.test_runtime_efficiency import load_definitions


class SeedResizeNoiseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        self.cpu = torch.device("cpu")

    def rng(self, source):
        namespace = load_definitions(
            "modules/rng.py",
            {"randn", "randn_local", "randn_without_seed", "manual_seed", "create_generator", "slerp", "ImageRNG"},
            {
                "torch": torch,
                "devices": SimpleNamespace(device=self.cpu, cpu=self.cpu),
                "shared": SimpleNamespace(
                    opts=SimpleNamespace(randn_source=source, eta_noise_seed_delta=0), device=self.cpu
                ),
                "rng_philox": rng_philox,
            },
        )
        return SimpleNamespace(**namespace)

    @staticmethod
    def centered_noise(generator, shape, seed, source_height, source_width, variation_seed=None):
        source_shape = (*shape[:-2], source_height // 8, source_width // 8)
        original = generator.randn_local(seed if variation_seed is None else variation_seed, source_shape)
        target = generator.randn_local(seed, shape)
        # Preserve the existing centering convention: an odd excess is cropped
        # from the top/left. Every target pixel inside the source is replaced.
        height, width = shape[-2:]
        source_h, source_w = source_shape[-2:]
        offset_y, offset_x = (height - source_h) // 2, (width - source_w) // 2
        source_y, source_x = max(-offset_y, 0), max(-offset_x, 0)
        target_y, target_x = max(offset_y, 0), max(offset_x, 0)
        copy_h, copy_w = min(height, source_h), min(width, source_w)
        target[..., target_y : target_y + copy_h, target_x : target_x + copy_w] = original[
            ..., source_y : source_y + copy_h, source_x : source_x + copy_w
        ]
        return target

    def test_oversized_seed_with_odd_latent_difference_fills_every_target_row_and_column(self):
        shape = (2, 8, 8)
        seeds = [21, 32]
        for source in ("CPU", "NV"):
            generator = self.rng(source)
            public = load_definitions("modules/processing.py", {"create_random_tensors"}, {"rng": generator})[
                "create_random_tensors"
            ]
            for height, width in ((64, 72), (72, 64), (72, 72), (80, 88)):
                with self.subTest(source=source, height=height, width=width):
                    expected = torch.stack(
                        [self.centered_noise(generator, shape, seed, height, width) for seed in seeds]
                    )
                    actual = public(shape, seeds, seed_resize_from_h=height, seed_resize_from_w=width)
                    self.assertEqual(actual.shape, (2, *shape))
                    torch.testing.assert_close(actual, expected)

    def test_video_seed_resize_preserves_the_channel_and_time_axes(self):
        shape = (2, 3, 8, 8)
        seeds = [21, 32]
        for source in ("CPU", "NV"):
            generator = self.rng(source)
            for height, width in ((64, 64), (56, 72), (72, 80)):
                with self.subTest(source=source, height=height, width=width):
                    stream = generator.ImageRNG(shape, seeds, seed_resize_from_h=height, seed_resize_from_w=width)
                    expected = torch.stack(
                        [self.centered_noise(generator, shape, seed, height, width) for seed in seeds]
                    )
                    actual = stream.next()
                    self.assertEqual(actual.shape, (2, *shape))
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(stream.next().shape, actual.shape)

    def test_unresized_and_even_crop_noise_preserve_existing_sequences(self):
        shape = (2, 8, 8)
        for source in ("CPU", "NV"):
            generator = self.rng(source)
            for height, width in ((0, 0), (64, 64), (80, 96), (48, 32), (48, 96)):
                with self.subTest(source=source, height=height, width=width):
                    stream = generator.ImageRNG(shape, [21], seed_resize_from_h=height, seed_resize_from_w=width)
                    expected = (
                        generator.randn_local(21, shape)
                        if height == 0
                        else self.centered_noise(generator, shape, 21, height, width)
                    )
                    torch.testing.assert_close(stream.next()[0], expected)
                    followup = generator.randn_without_seed(shape, generator=stream.generators[0])
                    # The independent generator has already consumed the first target noise.
                    control_generator = generator.create_generator(21)
                    generator.randn_without_seed(shape, generator=control_generator)
                    torch.testing.assert_close(
                        followup, generator.randn_without_seed(shape, generator=control_generator)
                    )

    def test_full_variation_and_eta_noise_delta_keep_batch_seed_order(self):
        seeds, subseeds = [21, 32], [43, 54]
        for source in ("CPU", "NV"):
            generator = self.rng(source)
            generator.shared.opts.eta_noise_seed_delta = 17
            for shape in ((2, 8, 8), (2, 3, 8, 8)):
                with self.subTest(source=source, shape=shape):
                    stream = generator.ImageRNG(
                        shape,
                        seeds,
                        subseeds=subseeds,
                        subseed_strength=1.0,
                        seed_resize_from_h=72,
                        seed_resize_from_w=80,
                    )
                    expected = torch.stack(
                        [
                            self.centered_noise(generator, shape, seed, 72, 80, subseed)
                            for seed, subseed in zip(seeds, subseeds, strict=True)
                        ]
                    )
                    torch.testing.assert_close(stream.next(), expected)
                    expected_followup = torch.stack([generator.randn_local(seed + 17, shape) for seed in seeds])
                    torch.testing.assert_close(stream.next(), expected_followup)


if __name__ == "__main__":
    unittest.main()
