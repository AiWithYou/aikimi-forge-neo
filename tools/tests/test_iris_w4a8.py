"""Packed W4 weights, dynamic A8 execution and exact checkpoint roundtrip."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

from modules_forge.iris.core import model_directory, validate_request
from modules_forge.iris.w4a8 import FORMAT, W4A8Linear, convert_linears, load_w4a8, require_native_cuda


class W4A8Tests(unittest.TestCase):
    def test_worker_rejects_internal_bf16_fallback_and_unsupported_gpu(self):
        fake_backend = types.ModuleType("comfy_kitchen.backends.cuda")
        fake_backend.eager_w4a8_int8_linear = lambda *args, **kwargs: "bf16 fallback"
        fake_kitchen = types.SimpleNamespace(list_backends=lambda: {"cuda": {"available": True}})
        require_native_cuda.cache_clear()
        with (
            patch("modules_forge.iris.w4a8.dependency", return_value=fake_kitchen),
            patch.dict("sys.modules", {"comfy_kitchen.backends.cuda": fake_backend}),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            require_native_cuda()
            with self.assertRaisesRegex(RuntimeError, "W4A8"):
                fake_backend.eager_w4a8_int8_linear()
        require_native_cuda.cache_clear()
        with (
            patch("modules_forge.iris.w4a8.dependency", return_value=fake_kitchen),
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            self.assertRaisesRegex(RuntimeError, "Ampere"),
        ):
            require_native_cuda()
        require_native_cuda.cache_clear()

    def test_w4a8_requests_and_folders_are_distinct_from_int8_and_normal(self):
        for task in ("generate", "depth", "upscale"):
            request = validate_request({"task": task, "precision": "w4a8", "prompt": "a fox", "image": "input.png"})
            self.assertEqual(request["precision"], "w4a8")
            folder = model_directory("models", "w4a8", task)
            self.assertTrue(folder.is_relative_to(Path("models/w4a8")))
            self.assertNotEqual(folder, model_directory("models", "int8", task))

    def test_preserves_shared_modulation_and_unsupported_projections(self):
        model = nn.Module()
        shared = nn.Linear(256, 256)
        model.modulation_cores = nn.ModuleDict({"adaln_img": shared})
        model.unregistered_core_reference = (shared,)
        model.projection = nn.Linear(256, 256)
        model.small = nn.Linear(64, 32)
        model.odd = nn.Linear(300, 256)
        report = convert_linears(model, device="cpu")
        self.assertEqual(report["linear_count"], 2)
        self.assertIs(model.modulation_cores["adaln_img"], shared)
        self.assertIs(model.unregistered_core_reference[0], shared)
        self.assertIsInstance(model.projection, W4A8Linear)
        self.assertIsInstance(model.small, nn.Linear)
        self.assertIsInstance(model.odd, W4A8Linear)

    def test_padding_preserves_non_aligned_input_output_and_reload(self):
        torch.manual_seed(37)
        model = nn.Sequential(nn.Linear(300, 129)).eval()
        inputs = torch.randn(2, 300)
        expected = model(inputs)
        convert_linears(model, device="cpu")
        actual = model(inputs)
        self.assertEqual(actual.shape, (2, 129))
        self.assertEqual(model[0].qweight.shape, (144, 256))
        self.assertLess(((actual - expected).square().mean() / expected.square().mean()).sqrt().item(), 0.12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(model.state_dict(), path, metadata={"format": FORMAT})
            loaded = nn.Sequential(nn.Linear(300, 129))
            load_w4a8(loaded, path)
            torch.testing.assert_close(loaded(inputs), actual, atol=0, rtol=0)

    def test_packed_storage_numerics_and_exact_reload(self):
        torch.manual_seed(17)
        model = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 128)).eval()
        inputs = torch.randn(2, 3, 256)
        expected = model(inputs)
        report = convert_linears(model, device="cpu")
        self.assertEqual(report["format"], FORMAT)
        self.assertEqual(report["linear_count"], 2)
        self.assertEqual(model[0].qweight.shape, (256, 128))
        self.assertEqual(model[0].qweight.dtype, torch.int8)
        self.assertLess(report["bytes"], 256 * 384)
        actual = model(inputs)
        self.assertTrue(torch.isfinite(actual).all())
        relative_rmse = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        # Two quantized projections compound error; image quality is evaluated
        # separately at 100 steps, rather than inferred from this toy network.
        self.assertLess(relative_rmse.item(), 0.12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(model.state_dict(), path, metadata={"format": FORMAT})
            with torch.device("meta"):
                loaded = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 128))
            load_w4a8(loaded, path)
            torch.testing.assert_close(loaded(inputs), actual, atol=0, rtol=0)
            self.assertFalse(any(t.is_meta for t in loaded.state_dict().values()))

    def test_rejects_missing_scales_wrong_storage_and_wrong_format(self):
        model = nn.Sequential(nn.Linear(256, 128))
        convert_linears(model, device="cpu")
        original = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            for state, metadata in (
                ({k: v for k, v in original.items() if k != "0.scale"}, {"format": FORMAT}),
                (dict(original, **{"0.qweight": torch.zeros(128, 128, dtype=torch.uint8)}), {"format": FORMAT}),
                (original, {"format": "weight-only-int4"}),
            ):
                with self.subTest(metadata=metadata):
                    save_file(state, path, metadata=metadata)
                    with self.assertRaises(ValueError):
                        load_w4a8(nn.Sequential(nn.Linear(256, 128)), path)

    def test_zero_weights_are_finite_and_cuda_routes_to_native_backend(self):
        import comfy_kitchen as ck

        layer = nn.Linear(256, 128, bias=False)
        layer.weight.data.zero_()
        packed = W4A8Linear.from_linear(layer, device="cpu")
        result = packed(torch.zeros(3, 256))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, torch.zeros_like(result))
        fake_input = torch.empty(3, 256)
        with (
            patch.object(torch.Tensor, "is_cuda", new_callable=unittest.mock.PropertyMock, return_value=True),
            patch.object(ck, "use_backend") as backend,
            patch.object(ck, "w4a8_int8_linear", return_value=torch.empty(3, 128)) as kernel,
        ):
            packed(fake_input)
        backend.assert_called_once_with("cuda")
        kernel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
