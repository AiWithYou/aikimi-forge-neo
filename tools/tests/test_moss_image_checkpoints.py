"""Custom MOSS models must retain their content identity and exact architecture."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.moss_image import checkpoints, core


class CheckpointTests(unittest.TestCase):
    def header(self, precision="bf16"):
        shapes = checkpoints.expected_shapes()
        result = {key: {"shape": list(shape), "dtype": "BF16"} for key, shape in shapes.items()}
        if precision == "int8":
            for key in checkpoints.QUANTIZED_KEYS:
                result[key]["dtype"] = "I8"
                prefix = key.removesuffix(".weight")
                result[prefix + ".weight_scale"] = {"shape": [shapes[key][0], 1], "dtype": "F32"}
                result[prefix + ".comfy_quant"] = {"shape": [80], "dtype": "U8"}
        return result

    def test_complete_bf16_and_int8_layouts_are_distinguished(self):
        self.assertEqual(len(checkpoints.expected_shapes()), 463)
        self.assertEqual(len(checkpoints.QUANTIZED_KEYS), 252)
        for precision in ("bf16", "int8"):
            self.assertEqual(checkpoints.header_precision(self.header(precision)), precision)

    def test_incomplete_wrong_shape_and_mixed_quantization_are_rejected(self):
        for header in (self.header(), self.header("int8")):
            for key in ("video_embedder.weight", "layers.41.proj.q.weight"):
                invalid = copy.deepcopy(header)
                invalid.pop(key)
                with self.assertRaises(ValueError):
                    checkpoints.header_precision(invalid)
            invalid = copy.deepcopy(header)
            invalid["layers.41.proj.q.weight"]["shape"] = [1, 1]
            with self.assertRaises(ValueError):
                checkpoints.header_precision(invalid)

    def test_fp16_and_partial_int8_are_not_reported_as_bf16(self):
        for dtype in ("F16", "F32", "I8"):
            invalid = self.header()
            invalid["layers.41.proj.q.weight"]["dtype"] = dtype
            with self.assertRaises(ValueError):
                checkpoints.header_precision(invalid)

    def test_request_accepts_managed_checkpoint_but_rejects_escapes(self):
        value = core.validate_request({"prompt": "cat", "checkpoint": "nested/Pastel.safetensors"})
        self.assertEqual(value["checkpoint"], "nested/Pastel.safetensors")
        for name in ("../x.safetensors", "C:/x.safetensors", "/x.safetensors", "x.txt", "MOSS-Image-SFT/x.safetensors"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", "checkpoint": name})

    def test_change_after_binding_and_precision_mismatch_are_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "checkpoints/Pastel.safetensors"
            path.parent.mkdir()
            path.write_bytes(b"first model")
            with patch.object(checkpoints.local_assets, "read_header", return_value=self.header()):
                request = {"checkpoint": path.name, "precision": "bf16"}
                checkpoints.bind(root, request)
                self.assertEqual(len(request["checkpoint_identity"]["sha256"]), 64)
                self.assertEqual(checkpoints.resolve(root, path.name), path)
                path.write_bytes(b"changed model")
                with self.assertRaises(ValueError):
                    checkpoints.bind(root, request)
                with self.assertRaises(ValueError):
                    checkpoints.bind(root, {"checkpoint": path.name, "precision": "int8"})

    def test_custom_model_uses_only_shared_frozen_official_assets(self):
        value = core.validate_request({"prompt": "cat", "checkpoint": "Pastel.safetensors", "precision": "bf16"})
        specs = core.model_specs(value)
        self.assertFalse(any("/transformer/" in spec["path"] or "dit-int8" in spec["path"] for spec in specs))
        self.assertTrue(any("/vae/" in spec["path"] for spec in specs))
