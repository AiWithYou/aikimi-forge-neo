"""FLUX 3 adapters must be selected safely, applied, and recorded exactly."""

import importlib
import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge.flux3_klein import core
from modules_forge.flux3_klein.inference import ComfyRuntime, Runner
from tools.tests import test_flux3_klein_inference as inference_tests

FakeRuntime = inference_tests.FakeRuntime

PREFIX = "diffusion_model.single_blocks.0.attn_out"


def adapter(path, value=1.0, metadata=None, pair_suffixes=("lora_down", "lora_up")):
    tensors = {
        PREFIX + f".{pair_suffixes[0]}.weight": ([1, 2], [value, value]),
        PREFIX + f".{pair_suffixes[1]}.weight": ([3, 1], [value, value, value]),
        PREFIX + ".alpha": ([], [1.0]),
    }
    if metadata is None:
        metadata = {"base_model": "black-forest-labs/flux-3-image-klein-base"}
    header, data = {"__metadata__": metadata}, b""
    for name, (shape, values) in tensors.items():
        payload = struct.pack("<" + "f" * len(values), *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(data), len(data) + len(payload)]}
        data += payload
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return path


class LoraTests(unittest.TestCase):
    def lora(self):
        return importlib.import_module("modules_forge.flux3_klein.lora")

    def test_request_accepts_only_managed_names_and_finite_strengths(self):
        request = core.validate_request({"prompt": "cat", "model": "base", "lora": "AikimiLoRA.safetensors"})
        self.assertEqual((request["lora"], request["lora_strength"]), ("AikimiLoRA.safetensors", 1.0))
        self.assertIsNone(core.validate_request({"prompt": "cat"})["lora"])
        for name in (
            "../outside.safetensors",
            "C:/private/a.safetensors",
            "a\\b.safetensors",
            "https://a/x",
            ".a.safetensors",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", "lora": name})
        for strength in (True, "1", -0.1, 2.1, math.nan, math.inf):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", "lora_strength": strength})

    def test_inventory_header_and_content_identity_change_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = adapter(root / "loras/AikimiLoRA.safetensors")
            adapter(root / "loras/.hidden.safetensors")
            self.assertEqual(self.lora().inventory(root), ["AikimiLoRA.safetensors"])
            before = self.lora().inspect(root, path.name)
            self.assertEqual((before["rank"], before["targets"], before["base_model"]), (1, 1, "base"))
            stamp = path.stat().st_mtime_ns
            adapter(path, value=2.0)
            os.utime(path, ns=(stamp, stamp))
            self.assertNotEqual(before["sha256"], self.lora().inspect(root, path.name)["sha256"])
            self.assertNotIn("path", before)

    def test_linked_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = adapter(root / "outside.safetensors")
            (root / "loras").mkdir()
            try:
                (root / "loras/linked.safetensors").symlink_to(target)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                self.lora().inspect(root, "linked.safetensors")
            self.assertEqual(self.lora().inventory(root), [])

    def test_training_source_does_not_change_the_selected_model(self):
        sources = (
            ({"base_model": "black-forest-labs/flux-3-image-klein-base"}, "base"),
            ({"base_model": "black-forest-labs/flux-3-image-klein"}, "distilled"),
            ({"base_model": "another/training-checkpoint"}, None),
            ({"training_note": "compatible adapter"}, None),
            ({}, None),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for metadata, training_source in sources:
                path = adapter(root / "loras/AikimiLoRA.safetensors", metadata=metadata)
                for selected_model in ("base", "distilled"):
                    with self.subTest(metadata=metadata, model=selected_model):
                        request = {"model": selected_model, "lora": path.name, "lora_strength": 0.75}
                        identity = self.lora().bind(root, request)
                        self.assertEqual(request["model"], selected_model)
                        self.assertEqual(identity["base_model"], training_source)
                        self.assertEqual(request["lora_identity"], identity)

    def test_changed_adapter_is_rejected_after_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = adapter(root / "loras/AikimiLoRA.safetensors")
            request = {"model": "base", "lora": path.name, "lora_strength": 1.0}
            self.lora().bind(root, request)
            adapter(path, value=2.0)
            with self.assertRaisesRegex(ValueError, "変更"):
                self.lora().bind(root, request)

    def test_header_and_worker_reject_mixed_pair_formats(self):
        import torch
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temp:
            for suffixes in (("lora_down", "lora_B"), ("lora_A", "lora_up")):
                path = adapter(Path(temp) / "mixed.safetensors", pair_suffixes=suffixes)
                with self.subTest(suffixes=suffixes, validation="header"), self.assertRaisesRegex(ValueError, "形式"):
                    self.lora().header_info(path)
                with self.subTest(suffixes=suffixes, validation="worker"), self.assertRaisesRegex(ValueError, "形式"):
                    self.lora().validate_tensors(
                        load_file(str(path)), {PREFIX: "target.weight"}, {"target.weight": torch.zeros(3, 2)}, torch
                    )

    def test_each_complete_pair_format_is_supported(self):
        import torch
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temp:
            for suffixes in (("lora_down", "lora_up"), ("lora_A", "lora_B")):
                with self.subTest(suffixes=suffixes):
                    path = adapter(Path(temp) / "complete.safetensors", pair_suffixes=suffixes)
                    self.assertEqual(self.lora().header_info(path)["targets"], 1)
                    self.assertEqual(
                        self.lora().validate_tensors(
                            load_file(str(path)), {PREFIX: "target.weight"}, {"target.weight": torch.zeros(3, 2)}, torch
                        ),
                        1,
                    )

    def test_distilled_runner_applies_base_trained_lora_and_records_both_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = adapter(root / "loras/AikimiLoRA.safetensors")
            runtime = FakeRuntime()
            patched_model = object()
            runtime.apply_lora = Mock(return_value=(patched_model, 1))
            original = runtime.sample

            def sampled(model, *args):
                self.assertIs(model, patched_model)
                return original(model, *args)

            runtime.sample = sampled
            request = core.validate_request(
                dict(prompt="cat", model="distilled", width=64, height=64, seed=7, lora=path.name, lora_strength=0.75)
            )
            self.lora().bind(root, request)
            with patch(
                "modules_forge.flux3_klein.inference.model_paths",
                return_value=inference_tests.InferenceTests().make_paths(root),
            ):
                result = Runner(root, runtime).run(root / "job", request)
            runtime.apply_lora.assert_called_once()
            self.assertEqual(runtime.apply_lora.call_args.args[1:], (path.resolve(), 0.75))
            self.assertIn(("verify", "distilled"), runtime.events)
            self.assertIn(("dit", "distilled"), runtime.events)
            sample = next(event for event in runtime.events if event[0] == "sample")
            self.assertIsNone(sample[1])
            self.assertEqual((len(sample[2]), sample[3]), (17, 1.0))
            self.assertEqual(result["model"], "distilled")
            self.assertEqual(result["lora"]["base_model"], "base")
            self.assertEqual(result["lora"], {**request["lora_identity"], "strength": 0.75, "applied_targets": 1})
            self.assertEqual(json.loads((root / "job/result.json").read_text(encoding="utf-8")), result)
            with Image.open(result["image_path"]) as image:
                saved = json.loads(image.info["flux3_klein"])
            self.assertEqual(saved["model"], "distilled")
            self.assertEqual(saved["lora"], result["lora"])

    def test_zero_strength_does_not_load_adapter_or_leave_it_for_next_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = adapter(root / "loras/AikimiLoRA.safetensors")
            runtime = FakeRuntime()
            runtime.apply_lora = Mock(side_effect=AssertionError("must not apply"))
            paths = inference_tests.InferenceTests().make_paths(root)
            with patch("modules_forge.flux3_klein.inference.model_paths", return_value=paths):
                runner = Runner(root, runtime)
                zero = runner.run(
                    root / "zero",
                    dict(prompt="cat", model="base", lora=path.name, lora_strength=0, width=64, height=64),
                )
                plain = runner.run(root / "plain", dict(prompt="cat", width=64, height=64))
            self.assertEqual(zero["lora"]["applied_targets"], 0)
            self.assertIsNone(plain["lora"])
            runtime.apply_lora.assert_not_called()

    def test_inflight_replacement_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = adapter(root / "loras/AikimiLoRA.safetensors")
            runtime = FakeRuntime()
            runtime.apply_lora = Mock(return_value=(object(), 1))
            original = runtime.sample

            def sampled(*args):
                adapter(path, value=2.0)
                return original(*args)

            runtime.sample = sampled
            with patch(
                "modules_forge.flux3_klein.inference.model_paths",
                return_value=inference_tests.InferenceTests().make_paths(root),
            ):
                with self.assertRaisesRegex(ValueError, "変更"):
                    Runner(root, runtime).run(
                        root / "job",
                        dict(prompt="cat", model="base", lora=path.name, lora_strength=1, width=64, height=64),
                    )
            self.assertFalse((root / "job/result.json").exists())
            self.assertEqual(runtime.events[-1], ("offload",))

    def test_worker_requires_all_targets_shapes_and_finite_values(self):
        import torch

        with tempfile.TemporaryDirectory() as temp:
            path = adapter(Path(temp) / "AikimiLoRA.safetensors")
            from safetensors.torch import load_file

            state = load_file(str(path))
            runtime = ComfyRuntime.__new__(ComfyRuntime)
            runtime.torch = torch
            runtime.utils = SimpleNamespace(load_torch_file=lambda *args, **kwargs: (state, {}))
            key = PREFIX + ".weight"
            patches = {key: object()}
            applied = SimpleNamespace(add_patches=Mock(return_value=[key]))
            model = SimpleNamespace(
                model=SimpleNamespace(state_dict=lambda: {key: torch.zeros(3, 2)}), clone=Mock(return_value=applied)
            )
            comfy_lora = SimpleNamespace(
                model_lora_keys_unet=lambda *args: {PREFIX: key}, load_lora=Mock(return_value=patches)
            )
            with patch("modules_forge.flux3_klein.inference.importlib.import_module", return_value=comfy_lora):
                self.assertEqual(runtime.apply_lora(model, path, 0.5), (applied, 1))
                comfy_lora.load_lora.assert_called_once_with(state, {PREFIX: key})
                model.clone.assert_called_once_with()
                applied.add_patches.assert_called_once_with(patches, 0.5)
                state[PREFIX + ".lora_up.weight"][0, 0] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite|有限"):
                    runtime.apply_lora(model, path, 0.5)
                comfy_lora.load_lora.assert_called_once()

    def test_worker_rejects_missing_or_unexpected_loaded_and_registered_targets(self):
        import torch

        key = PREFIX + ".weight"
        state = {PREFIX + ".lora_down.weight": torch.ones(1, 2), PREFIX + ".lora_up.weight": torch.ones(3, 1)}
        for loaded, accepted in (
            ({}, []),
            ({key: object(), "unexpected.weight": object()}, [key]),
            ({key: object()}, []),
            ({key: object()}, [key, "unexpected.weight"]),
        ):
            with self.subTest(loaded=list(loaded), accepted=accepted):
                applied = SimpleNamespace(add_patches=Mock(return_value=accepted))
                model = SimpleNamespace(
                    model=SimpleNamespace(state_dict=lambda: {key: torch.zeros(3, 2)}),
                    clone=Mock(return_value=applied),
                )
                runtime = ComfyRuntime.__new__(ComfyRuntime)
                runtime.torch = torch
                runtime.utils = SimpleNamespace(load_torch_file=lambda *args, **kwargs: (state, {}))
                mapper = SimpleNamespace(
                    model_lora_keys_unet=lambda *args: {PREFIX: key}, load_lora=Mock(return_value=loaded)
                )
                with (
                    patch("modules_forge.flux3_klein.inference.importlib.import_module", return_value=mapper),
                    self.assertRaisesRegex(ValueError, "LoRA"),
                ):
                    runtime.apply_lora(model, Path("selected.safetensors"), 0.75)
                if set(loaded) != {key}:
                    model.clone.assert_not_called()
                    applied.add_patches.assert_not_called()

    def test_worker_registers_fused_slice_targets_without_losing_the_offset(self):
        import torch

        prefix = "diffusion_model.single_blocks.0.q_proj"
        key = "diffusion_model.single_blocks.0.qkv.weight"
        target = (key, (0, 0, 3))
        state = {prefix + ".lora_A.weight": torch.ones(1, 2), prefix + ".lora_B.weight": torch.ones(3, 1)}
        patches = {target: object()}
        applied = SimpleNamespace(add_patches=Mock(return_value=[target]))
        model = SimpleNamespace(
            model=SimpleNamespace(state_dict=lambda: {key: torch.zeros(9, 2)}), clone=Mock(return_value=applied)
        )
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.utils = SimpleNamespace(load_torch_file=lambda *args, **kwargs: (state, {}))
        mapper = SimpleNamespace(
            model_lora_keys_unet=lambda *args: {prefix: target}, load_lora=Mock(return_value=patches)
        )
        with patch("modules_forge.flux3_klein.inference.importlib.import_module", return_value=mapper):
            self.assertEqual(runtime.apply_lora(model, Path("selected.safetensors"), 0.75), (applied, 1))
        mapper.load_lora.assert_called_once_with(state, {prefix: target})
        model.clone.assert_called_once_with()
        applied.add_patches.assert_called_once_with(patches, 0.75)

    def test_fused_qkv_targets_use_slice_shapes_and_reject_partial_adapters(self):
        import torch

        prefix = "diffusion_model.single_blocks.0.q_proj"
        state = {prefix + ".lora_down.weight": torch.ones(1, 2), prefix + ".lora_up.weight": torch.ones(3, 1)}
        key = "single_blocks.0.qkv.weight"
        mapping, weights = {prefix: (key, (0, 3, 3))}, {key: torch.zeros(9, 2)}
        self.assertEqual(self.lora().validate_tensors(state, mapping, weights, torch), 1)
        for invalid in ({}, {prefix: (key, (0, 8, 3))}, {prefix: (key, (0, 3, 2))}):
            with self.subTest(mapping=invalid), self.assertRaises(ValueError):
                self.lora().validate_tensors(state, invalid, weights, torch)
        for invalid in (
            {prefix + ".lora_down.weight": torch.ones(1, 2)},
            {**state, "unexpected.weight": torch.ones(1)},
            {**state, prefix + ".alpha": torch.tensor(0.0)},
        ):
            with self.subTest(keys=list(invalid)), self.assertRaises(ValueError):
                self.lora().validate_tensors(invalid, mapping, weights, torch)

    def test_saved_lora_identity_survives_history_and_png_tampering_is_rejected(self):
        from PIL import PngImagePlugin

        from modules_forge.flux3_klein.core import atomic_json
        from modules_forge.flux3_klein.history import History
        from modules_forge.flux3_klein.service import Service

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = adapter(root / "loras/AikimiLoRA.safetensors")
            request = core.validate_request(
                dict(prompt="cat", model="distilled", width=64, height=64, seed=7, lora=path.name, lora_strength=0.75)
            )
            directory = root / "outputs" / ("a" * 32)
            directory.mkdir(parents=True)
            self.lora().bind(root, request)
            atomic_json(directory / "request.json", request)
            runtime = FakeRuntime()
            runtime.apply_lora = Mock(return_value=(object(), 1))
            with patch(
                "modules_forge.flux3_klein.inference.model_paths",
                return_value=inference_tests.InferenceTests().make_paths(root),
            ):
                result = Runner(root, runtime).run(directory, request)
            history = History(root / "outputs")
            self.assertIn("AikimiLoRA ×0.75", history.list()[0][0])
            restored = history.load("a" * 32)
            self.assertEqual(restored["model"], "distilled")
            self.assertEqual(restored["lora"], result["lora"])
            self.assertEqual(restored["lora"]["base_model"], "base")
            with Image.open(directory / "output.png") as image:
                record = json.loads(image.info["flux3_klein"])
                pixels = image.copy()
            record["lora"]["strength"] = 1.0
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("flux3_klein", json.dumps(record))
            pixels.save(directory / "output.png", pnginfo=metadata)
            with self.assertRaisesRegex(ValueError, "PNG.*LoRA"):
                Service.validate_result(directory)
            self.assertEqual(history.list(), [])


if __name__ == "__main__":
    unittest.main()
