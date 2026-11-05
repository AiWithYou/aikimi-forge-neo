"""FLUX 3 W4A8 accepts the pinned Comfy layout without changing its source."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from modules_forge.flux3_klein import checkpoints, core, lora
from modules_forge.flux3_klein.inference import Runner
from tools.tests import test_flux3_klein_inference as inference_tests
from tools.tests import test_flux3_klein_lora as lora_tests

BLOCKS = ["single_blocks.0", "single_blocks.1"]
FUSED = {f"{block}.{layer}.weight" for block in BLOCKS for layer in ("qkv", "attn_out", "mlp_in", "mlp_out")}
CONFIG = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}


def shapes(model):
    result = {"txt_in.weight": (16, 256)}
    for block in BLOCKS:
        result.update({f"{block}.qkv.weight": (48, 256)})
        result.update({f"{block}.{layer}.weight": (16, 256) for layer in ("attn_out", "mlp_in", "mlp_out")})
    if model == "distilled":
        result["ref_time_in.in_layer.weight"] = (16, 256)
    return result


def layout_tensors(*, model="base", fused=True):
    source = shapes(model)
    if not fused:
        for block in BLOCKS:
            source.pop(f"{block}.qkv.weight")
            source.update({f"{block}.{part}_proj.weight": (16, 256) for part in ("q", "k", "v")})
    tensors = {}
    for name, (rows, columns) in source.items():
        quantized = name.startswith("single_blocks.")
        if not quantized:
            tensors[name] = torch.ones((rows, columns), dtype=torch.bfloat16)
            continue
        prefix = name.removesuffix(".weight")
        tensors[name] = torch.zeros((rows, columns // 2), dtype=torch.int8)
        tensors[prefix + ".weight_s_rel"] = torch.ones((rows, columns // 16)).to(torch.float8_e4m3fn)
        tensors[prefix + ".weight_s_channel"] = torch.ones(rows)
        tensors[prefix + ".weight_codebook"] = torch.linspace(-1, 1, 16)
        payload = json.dumps(CONFIG, separators=(",", ":")).encode()
        tensors[prefix + ".comfy_quant"] = torch.tensor(list(payload), dtype=torch.uint8)
    return tensors


class Flux3W4A8Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "checkpoints" / "custom.safetensors"
        self.path.parent.mkdir()
        self.shape_patch = patch.object(checkpoints, "expected_shapes", side_effect=shapes)
        self.key_patch = patch.object(checkpoints, "quantized_keys", return_value=FUSED)
        self.shape_patch.start()
        self.key_patch.start()
        self.addCleanup(self.shape_patch.stop)
        self.addCleanup(self.key_patch.stop)

    def save(self, tensors):
        save_file(tensors, str(self.path))
        return self.path

    def test_both_variants_accept_fused_or_separate_w4a8(self):
        for model in ("base", "distilled"):
            for fused in (True, False):
                with self.subTest(model=model, fused=fused):
                    self.save(layout_tensors(model=model, fused=fused))
                    self.assertEqual(checkpoints.header_info(self.path), {"model": model, "precision": "w4a8"})
                    self.assertEqual(checkpoints.inspect(self.root, self.path.name)["precision"], "w4a8")

    def test_w4a8_accepts_finite_signed_channel_and_relative_scales(self):
        for fused in (True, False):
            for case in ("signed_zero", "negative_rel", "negative_channel"):
                with self.subTest(fused=fused, case=case):
                    tensors = layout_tensors(fused=fused)
                    prefix = "single_blocks.0.qkv" if fused else "single_blocks.0.q_proj"
                    relative = tensors[prefix + ".weight_s_rel"].view(torch.uint8)
                    channel = tensors[prefix + ".weight_s_channel"]
                    if case == "signed_zero":
                        relative[0, 0], relative[0, 1] = 0x00, 0x80
                        channel[0], channel[1] = 0.0, -0.0
                    elif case == "negative_rel":
                        relative[0, 0] = 0xB8  # E4M3FN -1.0
                    else:
                        channel[0] = -1.0
                    self.save(tensors)
                    self.assertEqual(checkpoints.header_info(self.path), {"model": "base", "precision": "w4a8"})

    def test_w4a8_rejects_nonfinite_values_with_either_sign(self):
        for fused in (True, False):
            prefix = "single_blocks.0.qkv" if fused else "single_blocks.0.q_proj"
            for field, bits in (
                ("weight_s_rel", 0x7F),
                ("weight_s_rel", 0xFF),
                *((field, bits) for field in ("weight_s_channel", "weight_codebook")
                  for bits in (0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000)),
            ):
                with self.subTest(fused=fused, field=field, bits=bits):
                    tensors = layout_tensors(fused=fused)
                    tensor = tensors[prefix + "." + field]
                    if field == "weight_s_rel":
                        tensor.view(torch.uint8)[0, 0] = bits
                    else:
                        tensor.view(torch.int32)[0] = struct.unpack("<i", struct.pack("<I", bits))[0]
                    self.save(tensors)
                    with self.assertRaises(ValueError):
                        checkpoints.header_info(self.path)

    def test_rejects_missing_mixed_duplicate_bad_config_scales_and_correction(self):
        for broken in (
            "missing",
            "mixed",
            "duplicate",
            "config",
            "shape",
            "nan_rel",
            "float32_rel",
            "nan_channel",
            "inf_channel",
            "nan_codebook",
            "correction",
        ):
            with self.subTest(broken=broken):
                tensors = layout_tensors(fused=False)
                prefix = "single_blocks.1.q_proj"
                if broken == "missing":
                    tensors.pop(prefix + ".weight_s_rel")
                if broken == "mixed":
                    for part in ("q", "k", "v"):
                        for suffix in (
                            ".weight",
                            ".weight_s_rel",
                            ".weight_s_channel",
                            ".weight_codebook",
                            ".comfy_quant",
                        ):
                            tensors.pop(f"single_blocks.1.{part}_proj{suffix}")
                    tensors.update(
                        {k: v for k, v in layout_tensors(fused=True).items() if k.startswith("single_blocks.1.qkv.")}
                    )
                if broken == "duplicate":
                    tensors["single_blocks.1.qkv.weight"] = torch.zeros((48, 128), dtype=torch.int8)
                if broken == "config":
                    tensors[prefix + ".comfy_quant"] = torch.tensor(
                        list(json.dumps({**CONFIG, "group_size": 32}).encode()), dtype=torch.uint8
                    )
                if broken == "shape":
                    tensors[prefix + ".weight_s_rel"] = torch.ones((16, 8)).to(torch.float8_e4m3fn)
                if broken == "nan_rel":
                    tensors[prefix + ".weight_s_rel"][0, 0] = float("nan")
                if broken == "float32_rel":
                    tensors[prefix + ".weight_s_rel"] = tensors[prefix + ".weight_s_rel"].float()
                if broken == "nan_channel":
                    tensors[prefix + ".weight_s_channel"][0] = float("nan")
                if broken == "inf_channel":
                    tensors[prefix + ".weight_s_channel"][0] = float("inf")
                if broken == "nan_codebook":
                    tensors[prefix + ".weight_codebook"][0] = float("nan")
                if broken == "correction":
                    tensors[prefix + ".weight_correction"] = torch.zeros((16, 16))
                self.save(tensors)
                with self.assertRaises(ValueError):
                    checkpoints.header_info(self.path)

    def test_rejects_overlapping_tensor_offsets_or_trailing_payload(self):
        for broken in ("overlap", "trailing"):
            with self.subTest(broken=broken):
                self.save(layout_tensors())
                raw = self.path.read_bytes()
                header_size = struct.unpack("<Q", raw[:8])[0]
                header = json.loads(raw[8 : 8 + header_size])
                payload = raw[8 + header_size :]
                if broken == "overlap":
                    header["single_blocks.1.attn_out.weight"]["data_offsets"] = header[
                        "single_blocks.0.attn_out.weight"
                    ]["data_offsets"]
                else:
                    payload += b"unexpected bytes"
                encoded = json.dumps(header, separators=(",", ":")).encode()
                self.path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
                with self.assertRaises(ValueError):
                    checkpoints.header_info(self.path)

    def test_legacy_metadata_cannot_override_the_validated_layer_descriptors(self):
        overrides = {"layers": {"single_blocks.0.qkv": {"format": "nvfp4"}}}
        for precision in ("bf16", "w4a8"):
            with self.subTest(precision=precision):
                tensors = layout_tensors() if precision == "w4a8" else {
                    key: torch.ones(shape, dtype=torch.bfloat16) for key, shape in shapes("base").items()
                }
                save_file(tensors, str(self.path), metadata={"_quantization_metadata": json.dumps(overrides)})
                with self.assertRaisesRegex(ValueError, "量子化"):
                    checkpoints.header_info(self.path)

    def test_w4a8_is_custom_only_and_uses_int8_text_encoder(self):
        self.save(layout_tensors())
        with self.assertRaisesRegex(ValueError, "独自|本体"):
            core.validate_request({"prompt": "cat", "precision": "w4a8"})
        request = core.validate_request(
            {"prompt": "cat", "precision": "w4a8", "checkpoint": self.path.name, "model": "base"}
        )
        identity = checkpoints.bind(self.root, request)
        self.assertEqual(identity["precision"], "w4a8")
        specs = core.model_specs(request)
        self.assertNotIn("dit", specs)
        self.assertIn("int8_convrot", specs["text_encoder"]["path"])
        self.assertEqual(checkpoints.record(request)["source"], "custom")
        self.assertEqual(core.model_paths(self.root, request)["dit"], self.path.resolve())

    def test_lora_validates_packed_logical_shape_fused_slice_and_target_codebook(self):
        prefix = "diffusion_model.single_blocks.0.q_proj"
        key = "diffusion_model.single_blocks.0.qkv.weight"
        target = (key, (0, 16, 16))
        state = {prefix + ".lora_A.weight": torch.ones(2, 256), prefix + ".lora_B.weight": torch.ones(16, 2)}
        layer = key.removesuffix(".weight")
        model_state = {
            key: torch.zeros(48, 128, dtype=torch.int8),
            layer + ".comfy_quant": torch.zeros(4, dtype=torch.uint8),
            layer + ".weight_s_rel": torch.ones(48, 16),
            layer + ".weight_s_channel": torch.ones(48),
            layer + ".weight_codebook": torch.arange(16).float(),
        }
        model = SimpleNamespace(get_submodule=lambda name: SimpleNamespace(in_features=256, out_features=48))
        self.assertEqual(lora.validate_tensors(state, {prefix: target}, model_state, torch, model=model), 1)
        model_state[layer + ".weight_codebook"] = torch.arange(15, -1, -1).float()
        with self.assertRaisesRegex(ValueError, "コードブック"):
            lora.validate_tensors(state, {prefix: target}, model_state, torch, model=model)
        with self.assertRaises(ValueError):
            lora.validate_tensors(state, {prefix: (key, (0, 40, 16))}, model_state, torch, model=model)

    def test_lora_ignores_codebook_on_untouched_layer_and_accepts_sorted_duplicates(self):
        prefix = "diffusion_model.single_blocks.0.q_proj"
        key = "diffusion_model.single_blocks.0.qkv.weight"
        target = (key, (0, 0, 16))
        state = {prefix + ".lora_A.weight": torch.ones(2, 256), prefix + ".lora_B.weight": torch.ones(16, 2)}
        layer = key.removesuffix(".weight")
        other = "diffusion_model.single_blocks.1.qkv"
        values = {
            key: torch.zeros(48, 128, dtype=torch.int8),
            layer + ".comfy_quant": torch.zeros(4, dtype=torch.uint8),
            layer + ".weight_s_rel": torch.ones(48, 16),
            layer + ".weight_s_channel": torch.ones(48),
            layer + ".weight_codebook": torch.zeros(16),
            other + ".weight_s_rel": torch.ones(48, 16),
            other + ".weight_s_channel": torch.ones(48),
            other + ".weight_codebook": torch.arange(15, -1, -1).float(),
        }
        model = SimpleNamespace(get_submodule=lambda name: SimpleNamespace(in_features=256, out_features=48))
        self.assertEqual(lora.validate_tensors(state, {prefix: target}, values, torch, model=model), 1)
        values.pop(layer + ".weight_codebook")
        self.assertEqual(lora.validate_tensors(state, {prefix: target}, values, torch, model=model), 1)

    def test_w4a8_result_records_custom_identity_and_zero_strength_without_applying_lora(self):
        from PIL import Image

        tensors = layout_tensors()
        tensors["single_blocks.1.qkv.weight_codebook"] = torch.arange(15, -1, -1).float()
        self.save(tensors)
        lora_path = lora_tests.adapter(self.root / "loras/zero.safetensors")
        request = core.validate_request(
            {
                "prompt": "cat",
                "precision": "w4a8",
                "checkpoint": self.path.name,
                "model": "base",
                "width": 64,
                "height": 64,
                "seed": 7,
                "lora": lora_path.name,
                "lora_strength": 0,
            }
        )
        checkpoints.bind(self.root, request)
        lora.bind(self.root, request)
        request["reference_sha256"] = []
        directory = self.root / ("a" * 32)
        directory.mkdir()
        core.atomic_json(directory / "request.json", request)
        runtime = inference_tests.FakeRuntime()
        runtime.apply_lora = Mock(side_effect=AssertionError("strength zero must not apply a LoRA"))
        paths = inference_tests.InferenceTests().make_paths(self.root)
        paths["dit"] = self.path
        with patch("modules_forge.flux3_klein.inference.model_paths", return_value=paths):
            result = Runner(self.root, runtime).run(directory, request)
        runtime.apply_lora.assert_not_called()
        self.assertIn(("precision", "w4a8"), runtime.events)
        self.assertEqual(
            (result["precision"], result["checkpoint"]["source"], result["lora"]["applied_targets"]),
            ("w4a8", "custom", 0),
        )
        self.assertEqual(result["checkpoint"]["sha256"], request["checkpoint_identity"]["sha256"])
        with Image.open(result["image_path"]) as image:
            self.assertEqual(json.loads(image.info["flux3_klein"])["checkpoint"], result["checkpoint"])
        from modules_forge.flux3_klein.history import History
        from modules_forge.flux3_klein.service import Service

        self.assertEqual(Service.validate_result(directory)["checkpoint"], result["checkpoint"])
        self.assertEqual(History(self.root).load("a" * 32)["checkpoint"], result["checkpoint"])


class ArchitectureTests(unittest.TestCase):
    def test_full_w4a8_architectures_quantize_every_projection(self):
        self.assertEqual(len(checkpoints.layout_quantized_keys(True)), 172)
        self.assertEqual(len(checkpoints.layout_quantized_keys(False)), 258)


if __name__ == "__main__":
    unittest.main()
