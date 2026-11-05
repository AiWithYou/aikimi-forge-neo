"""Custom MOSS models must retain their content identity and exact architecture."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.moss_image import checkpoints, core


class CheckpointTests(unittest.TestCase):
    def header(self, precision="bf16"):
        shapes = checkpoints.expected_shapes()
        result = {key: {"shape": list(shape), "dtype": "BF16"} for key, shape in shapes.items()}
        if precision in ("int8", "w4a8"):
            for key in checkpoints.QUANTIZED_KEYS:
                result[key]["dtype"] = "I8"
                prefix = key.removesuffix(".weight")
                if precision == "w4a8":
                    rows, columns = shapes[key]
                    result[key]["shape"] = [rows, columns // 2]
                    result[prefix + ".weight_s_rel"] = {"shape": [rows, columns // 16], "dtype": "F8_E4M3"}
                    result[prefix + ".weight_s_channel"] = {"shape": [rows], "dtype": "F32"}
                    result[prefix + ".weight_codebook"] = {"shape": [16], "dtype": "F32"}
                else:
                    result[prefix + ".weight_scale"] = {"shape": [shapes[key][0], 1], "dtype": "F32"}
                result[prefix + ".comfy_quant"] = {"shape": [80], "dtype": "U8"}
        return result

    def test_complete_bf16_and_int8_layouts_are_distinguished(self):
        self.assertEqual(len(checkpoints.expected_shapes()), 463)
        self.assertEqual(len(checkpoints.QUANTIZED_KEYS), 252)
        for precision in ("bf16", "int8"):
            self.assertEqual(checkpoints.header_precision(self.header(precision)), precision)

    def test_w4a8_requires_complete_packed_projection_layout(self):
        header = self.header("w4a8")
        self.assertEqual(checkpoints.header_precision(header), "w4a8")
        for key in ("layers.41.proj.q.weight", "layers.41.proj.q.weight_s_rel",
                    "layers.41.proj.q.weight_s_channel", "layers.41.proj.q.comfy_quant"):
            invalid = copy.deepcopy(header)
            invalid.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoints.header_precision(invalid)
        for key, shape in (("layers.0.proj.q.weight", [3072, 3072]),
                           ("layers.0.proj.q.weight_s_rel", [3072, 1])):
            invalid = copy.deepcopy(header)
            invalid[key]["shape"] = shape
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoints.header_precision(invalid)
        invalid = copy.deepcopy(header)
        invalid["layers.0.proj.q.weight_correction"] = {"shape": [192, 3072], "dtype": "F32"}
        with self.assertRaises(ValueError):
            checkpoints.header_precision(invalid)

    def test_custom_w4a8_is_selected_and_official_w4a8_is_rejected(self):
        value = core.validate_request({"prompt": "cat", "checkpoint": "Custom.safetensors", "precision": "w4a8"})
        self.assertEqual(value["precision"], "w4a8")
        self.assertFalse(any("/transformer/" in spec["path"] or "dit-int8" in spec["path"]
                             for spec in core.model_specs(value)))
        with self.assertRaises(ValueError):
            core.validate_request({"prompt": "cat", "precision": "w4a8"})
        with self.assertRaises(ValueError):
            core.model_specs({"checkpoint": "official", "precision": "w4a8"})
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            core.asset_paths(Path(folder), {"checkpoint": "official", "precision": "w4a8", "text_precision": "int8"})

    def test_w4a8_layer_descriptors_must_match_the_packed_format(self):
        import torch
        from safetensors.torch import save_file

        prefix = "layers.0.proj.q"
        shapes = {prefix + ".weight": (2, 16), "rope.bands": (1,)}
        descriptor = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(checkpoints, "expected_shapes", return_value=shapes),
            patch.object(checkpoints, "QUANTIZED_KEYS", frozenset({prefix + ".weight"})),
        ):
            path = Path(folder) / "checkpoints" / "anything.safetensors"
            path.parent.mkdir()
            for config, metadata, valid in (
                (descriptor, None, True),
                ({**descriptor, "format": "nvfp4"}, None, False),
                ({**descriptor, "group_size": 32}, None, False),
                ({}, None, False),
                (descriptor, {"_quantization_metadata": '{"layers": {}}'}, False),
            ):
                payload = json.dumps(config).encode("utf-8")
                save_file({
                    prefix + ".weight": torch.zeros(2, 8, dtype=torch.int8),
                    prefix + ".weight_s_rel": torch.ones(2, 1).to(torch.float8_e4m3fn),
                    prefix + ".weight_s_channel": torch.ones(2),
                    prefix + ".weight_codebook": torch.arange(15, -1, -1).float(),
                    prefix + ".comfy_quant": torch.tensor(list(payload), dtype=torch.uint8),
                    "rope.bands": torch.zeros(1),
                }, str(path), metadata=metadata)
                if valid:
                    self.assertEqual(checkpoints.precision(path), "w4a8")
                    self.assertIn(("anything · W4A8", path.name), checkpoints.inventory(folder))
                    request = {"checkpoint": path.name, "precision": "w4a8"}
                    self.assertEqual(checkpoints.bind(folder, request)["precision"], "w4a8")
                else:
                    with self.assertRaises(ValueError):
                        checkpoints.precision(path)

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
                self.assertTrue(checkpoints.resolve(root, path.name).samefile(path))
                path.write_bytes(b"changed model")
                with self.assertRaises(ValueError):
                    checkpoints.bind(root, request)
                with self.assertRaises(ValueError):
                    checkpoints.bind(root, {"checkpoint": path.name, "precision": "int8"})

    def test_inspection_rehashes_when_file_change_counter_is_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoints" / "custom.safetensors"
            path.parent.mkdir()
            path.write_bytes(b"first")
            with (
                patch.object(checkpoints, "precision", return_value="bf16"),
                patch.object(checkpoints.local_assets, "file_version", return_value=(5, 0, None, 0)),
            ):
                first = checkpoints.inspect(folder, path.name)
                path.write_bytes(b"other")
                second = checkpoints.inspect(folder, path.name)
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_custom_model_uses_only_shared_frozen_official_assets(self):
        value = core.validate_request({"prompt": "cat", "checkpoint": "Pastel.safetensors", "precision": "bf16"})
        specs = core.model_specs(value)
        self.assertFalse(any("/transformer/" in spec["path"] or "dit-int8" in spec["path"] for spec in specs))
        self.assertTrue(any("/vae/" in spec["path"] for spec in specs))
