"""Iris request boundaries and reloadable weight-only INT8 contracts."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from modules_forge.iris.core import (
    CODE_REVISION,
    PACKAGING_REVISION,
    TEXT_REVISION,
    IrisError,
    atomic_json,
    model_ready,
    python_path,
    validate_request,
)
from modules_forge.iris.quantization import Int8Linear, convert_linears, load_int8, quantize_rows


class RequestTests(unittest.TestCase):
    def test_official_generation_defaults(self):
        request = validate_request({"task": "generate", "precision": "int8", "prompt": "a red fox"})
        self.assertEqual(
            (request["width"], request["height"], request["steps"], request["cfg"]), (1024, 1024, 100, 3.0)
        )
        self.assertEqual(request["negative_prompt"], "")
        self.assertGreaterEqual(request["seed"], 0)

    def test_reject_bad_values_before_worker_or_download(self):
        for update in (
            {"task": "edit"},
            {"precision": "fp4"},
            {"width": 1000},
            {"steps": 0},
            {"cfg": float("nan")},
            {"width": 4096},
            {"seed": -2},
            {"prompt": " "},
        ):
            with self.subTest(update=update), self.assertRaises(IrisError):
                validate_request({"task": "generate", "precision": "int8", "prompt": "a fox", **update})

    def test_generation_uses_the_official_trained_size_buckets(self):
        for width, height in (
            (1024, 1024),
            (1344, 768),
            (1280, 832),
            (1152, 896),
            (896, 1152),
            (832, 1280),
            (768, 1344),
        ):
            result = validate_request({"prompt": "a fox", "width": width, "height": height})
            self.assertEqual((result["width"], result["height"]), (width, height))
        for width, height in ((256, 256), (512, 512), (768, 1024), (1536, 1024)):
            with self.subTest(size=(width, height)), self.assertRaises(IrisError):
                validate_request({"prompt": "a fox", "width": width, "height": height})

    def test_image_tasks_require_input_and_ignore_generation_settings(self):
        for task in ("depth", "upscale"):
            with self.assertRaises(IrisError):
                validate_request({"task": task, "precision": "normal"})
            result = validate_request(
                {
                    "task": task,
                    "precision": "normal",
                    "image": "input.png",
                    "prompt": "",
                    "seed": -1,
                    "cfg": float("nan"),
                }
            )
            self.assertEqual(result["task"], task)
            self.assertEqual(result["image"], "input.png")
            self.assertNotIn("seed", result)
            self.assertNotIn("cfg", result)


class QuantizationTests(unittest.TestCase):
    def test_shared_modulation_core_keeps_its_object_identity(self):
        model = nn.Module()
        core = nn.Linear(64, 384)
        model.modulation_cores = nn.ModuleDict({"adaln_img": core})
        model.projection = nn.Linear(64, 64)
        model.unregistered_core_reference = (core,)
        convert_linears(model)
        self.assertIs(model.modulation_cores["adaln_img"], model.unregistered_core_reference[0])
        self.assertIsInstance(model.projection, Int8Linear)

    def test_zero_rows_and_finite_row_scales(self):
        weights = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.4, 2.0]])
        quantized, scales = quantize_rows(weights)
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales > 0).all())
        torch.testing.assert_close(quantized.float() * scales, weights, atol=2 / 127, rtol=0)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                quantize_rows(torch.tensor([[value]]))

    def test_reload_forward_and_storage(self):
        torch.manual_seed(5)
        original = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 128)).eval()
        inputs = torch.randn(2, 3, 128)
        expected = original(inputs)
        original_bytes = sum(x.numel() * x.element_size() for x in original.state_dict().values())
        report = convert_linears(original)
        self.assertEqual(report["linear_count"], 2)
        self.assertIsInstance(original[0], Int8Linear)
        torch.testing.assert_close(original(inputs), expected, atol=0.02, rtol=0.03)
        state = original.state_dict()
        self.assertLess(sum(x.numel() * x.element_size() for x in state.values()), original_bytes / 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            save_file(state, path)
            with torch.device("meta"):
                reloaded = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 128))
            load_int8(reloaded, path)
            torch.testing.assert_close(reloaded(inputs), original(inputs), atol=0, rtol=0)
            self.assertFalse(any(t.is_meta for t in reloaded.state_dict().values()))

    def test_missing_scale_and_shape_mismatch_are_rejected(self):
        layer = nn.Sequential(nn.Linear(64, 64))
        convert_linears(layer)
        state = layer.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            save_file({k: v for k, v in state.items() if k != "0.scale"}, path)
            with self.assertRaises((RuntimeError, ValueError)):
                load_int8(nn.Sequential(nn.Linear(64, 64)), path)
            corrupt = dict(state, **{"0.qweight": torch.zeros(63, 64, dtype=torch.int8)})
            save_file(corrupt, path)
            with self.assertRaises(ValueError):
                load_int8(nn.Sequential(nn.Linear(64, 64)), path)


class PreparationMarkerTests(unittest.TestCase):
    def test_partial_model_and_encoder_are_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = python_path(root)
            python.parent.mkdir(parents=True)
            python.touch()
            atomic_json(root / "runtime.json", {"code_revision": CODE_REVISION, "packaging": PACKAGING_REVISION})
            directory = root / "official"
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(b"test")
            (directory / "config.yaml").write_text("test", encoding="utf-8")
            self.assertFalse(model_ready(root, "normal", "generate"))
            atomic_json(
                directory / "manifest.json",
                {"files": [{"path": "model.safetensors", "size": 4}, {"path": "config.yaml", "size": 4}]},
            )
            encoder = root / "text-encoder"
            encoder.mkdir()
            (encoder / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(model_ready(root, "normal", "generate"))
            atomic_json(
                encoder / "download.json",
                {"revision": TEXT_REVISION, "files": [{"path": "model-1.safetensors", "size": 4}]},
            )
            self.assertFalse(model_ready(root, "normal", "generate"))
            (encoder / "model-1.safetensors").write_bytes(b"test")
            self.assertTrue(model_ready(root, "normal", "generate"))
            (directory / "model.safetensors").write_bytes(b"truncated")
            self.assertFalse(model_ready(root, "normal", "generate"))

    def test_wrong_or_malformed_runtime_marker_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = python_path(root)
            python.parent.mkdir(parents=True)
            python.touch()
            atomic_json(root / "runtime.json", {"code_revision": "old"})
            self.assertFalse(model_ready(root, "int8", "depth"))
            (root / "runtime.json").write_text("broken", encoding="utf-8")
            self.assertFalse(model_ready(root, "int8", "depth"))


if __name__ == "__main__":
    unittest.main()
