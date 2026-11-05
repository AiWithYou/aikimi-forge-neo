"""Managed LoRA identities and exact Comfy patches never rewrite base weights."""

import importlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

PREFIX = "layers.0.proj.q"
TARGET = "diffusion_model." + PREFIX + ".weight"


def adapter(path, *, prefix=PREFIX, suffixes=("lora_A", "lora_B"), value=1.0, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            prefix + f".{suffixes[0]}.weight": torch.full((1, 2), value),
            prefix + f".{suffixes[1]}.weight": torch.ones(3, 1),
            prefix + ".alpha": torch.tensor(1.0),
        },
        str(path),
        metadata=metadata or {"base_model": "unrecognized/training-name"},
    )
    return path


class Patcher:
    def __init__(self):
        self.weight = torch.zeros(3, 2, dtype=torch.int8)
        self.model = SimpleNamespace(state_dict=lambda: {TARGET: self.weight})
        self.calls = []

    def clone(self):
        cloned = Patcher()
        cloned.weight = self.weight
        return cloned

    def add_patches(self, patches, strength):
        self.calls.append((patches, strength))
        return list(patches)


def mapper():
    return SimpleNamespace(
        model_lora_keys_unet=Mock(return_value={"lora_unet_layers_0_proj_q": TARGET}),
        load_lora=Mock(side_effect=lambda state, mapping: {target: ("lora", state) for target in mapping.values()}),
    )


class LoraTests(unittest.TestCase):
    def lora(self):
        return importlib.import_module("modules_forge.moss_image.lora")

    def test_generation_avoids_duplicate_hashing_and_still_detects_changed_lora(self):
        from PIL import Image
        from safetensors.torch import load_file

        from modules_forge.moss_image import core
        from modules_forge.moss_image.inference import ComfyRuntime

        for changed_at in (None, "loading", "sampling"):
            with self.subTest(changed_at=changed_at), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = adapter(root / "loras/style.safetensors")
                request = core.validate_request({
                    "prompt": "a fox", "width": 512, "height": 512,
                    "loras": [{"name": path.name, "strength": 0.75}],
                })
                runtime = ComfyRuntime.__new__(ComfyRuntime)
                runtime.root, runtime.torch, runtime.mapper = root, torch, mapper()
                runtime.load_clip = Mock(return_value=object())
                runtime.encode_text = Mock(return_value=([[torch.zeros(1, 4, 16), {}]], 4))
                runtime.load_dit = Mock(return_value=Patcher())
                runtime.load_vae = Mock(return_value=object())
                runtime.decode = Mock(return_value=Image.new("RGB", (512, 512)))
                runtime.offload = Mock()

                def changed(path=path):
                    stamp = path.stat()
                    adapter(path, value=2.0)
                    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

                def read(*args, changed_at=changed_at, changed=changed, **kwargs):
                    state = load_file(*args, **kwargs)
                    if changed_at == "loading":
                        changed()
                    return state

                def sample(*args, changed_at=changed_at, changed=changed, **kwargs):
                    if changed_at == "sampling":
                        changed()
                    return torch.zeros(1, 128, 32, 32)

                runtime.sample = Mock(side_effect=sample)
                with (
                    patch.object(core, "asset_paths", return_value={"dit": root, "text_encoder": root, "vae": root}),
                    patch("safetensors.torch.load_file", side_effect=read),
                    patch.object(self.lora(), "inspect", wraps=self.lora().inspect) as inspect,
                ):
                    if changed_at:
                        with self.assertRaisesRegex(ValueError, "LoRAが変更"):
                            runtime.generate(request, lambda *a: None, lambda: None)
                    else:
                        runtime.generate(request, lambda *a: None, lambda: None)
                        self.assertEqual(inspect.call_count, 3)
                        self.assertEqual(runtime.last_metadata["lora_applied_targets"], 1)
                        self.assertEqual(runtime.sample.call_args.args[0].calls[0][1], 0.75)

    def test_inventory_nested_names_and_path_free_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root / "loras/portraits/style.safetensors")
            adapter(root / "loras/.hidden.safetensors")
            self.assertEqual(self.lora().inventory(root), ["portraits/style.safetensors"])
            identity = self.lora().inspect(root, "portraits/style.safetensors")
            self.assertEqual(identity["targets"], 1)
            self.assertEqual(identity["name"], "portraits/style.safetensors")
            self.assertEqual(len(identity["sha256"]), 64)
            self.assertGreater(identity["mtime_ns"], 0)
            self.assertNotIn("path", identity)
            self.assertNotIn(str(root), json.dumps(identity))

    def test_paths_and_redirecting_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "../private.safetensors",
                "C:/private.safetensors",
                "a\\b.safetensors",
                ".hidden.safetensors",
                "a//b.safetensors",
            ):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.lora().resolve(root, name)
            target = adapter(root / "private.safetensors")
            (root / "loras").mkdir()
            try:
                (root / "loras/link.safetensors").symlink_to(target)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                self.lora().resolve(root, "link.safetensors")

    def test_bind_detects_same_size_same_timestamp_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/style.safetensors")
            request = {"precision": "int8", "steps": 23, "cfg": 4.5, "loras": [{"name": path.name, "strength": 0.75}]}
            self.lora().bind(root, request)
            before = dict(request["lora_identities"][0])
            stamp = path.stat().st_mtime_ns
            adapter(path, value=2.0)
            os.utime(path, ns=(stamp, stamp))
            with self.assertRaisesRegex(ValueError, "変更"):
                self.lora().bind(root, request)
            self.assertEqual((request["precision"], request["steps"], request["cfg"]), ("int8", 23, 4.5))
            self.assertNotEqual(before["sha256"], self.lora().inspect(root, path.name)["sha256"])

    def test_disappearing_adapter_is_an_operational_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/style.safetensors")
            with patch.object(self.lora().local_assets, "file_version", side_effect=FileNotFoundError("private/path")):
                with self.assertRaisesRegex(ValueError, "LoRA"):
                    self.lora().inspect(root, path.name)

    def test_multiple_loras_apply_to_a_clone_and_preserve_quantized_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root / "loras/first.safetensors")
            adapter(root / "loras/second.safetensors", suffixes=("lora_down", "lora_up"))
            request = {
                "loras": [
                    {"name": "first.safetensors", "strength": 0.5},
                    {"name": "second.safetensors", "strength": -1.0},
                ]
            }
            original = Patcher()
            patched, count = self.lora().apply(original, root, request, mapper(), torch)
            self.assertIsNot(patched, original)
            self.assertEqual(count, 2)
            self.assertEqual([strength for _, strength in patched.calls], [0.5, -1.0])
            self.assertEqual(original.calls, [])
            self.assertEqual(original.weight.dtype, torch.int8)
            self.assertTrue(torch.equal(original.weight, torch.zeros(3, 2, dtype=torch.int8)))
            self.assertEqual([item["applied_targets"] for item in self.lora().record(request)], [1, 1])

    def test_zero_strength_is_recorded_without_loading_tensor_patches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root / "loras/style.safetensors")
            request = {"loras": [{"name": "style.safetensors", "strength": 0}]}
            original, mapping = Patcher(), mapper()
            patched, count = self.lora().apply(original, root, request, mapping, torch)
            self.assertIs(patched, original)
            self.assertEqual(count, 0)
            mapping.load_lora.assert_not_called()
            self.assertEqual(self.lora().record(request)[0]["applied_targets"], 0)

    def test_finite_zero_and_negative_alpha_are_valid_linear_adapters(self):
        for alpha in (0.0, -2.0):
            with self.subTest(alpha=alpha):
                state = {
                    PREFIX + ".lora_A.weight": torch.ones(1, 2),
                    PREFIX + ".lora_B.weight": torch.ones(3, 1),
                    PREFIX + ".alpha": torch.tensor(alpha),
                }
                self.assertEqual(
                    self.lora().validate_tensors(state, {PREFIX: TARGET}, {TARGET: torch.zeros(3, 2)}, torch), {TARGET}
                )

    def test_unknown_targets_shapes_rank_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "loras/style.safetensors"
            for kind in ("target", "shape", "rank", "nan", "alpha", "mixed", "unknown"):
                with self.subTest(kind=kind):
                    prefix = "other.layers.0" if kind == "target" else PREFIX
                    tensors = {
                        prefix + ".lora_A.weight": torch.ones(2 if kind == "rank" else 1, 2),
                        prefix + ".lora_B.weight": torch.ones(4 if kind == "shape" else 3, 1),
                        prefix + ".alpha": torch.tensor(math.inf if kind == "alpha" else 1.0),
                    }
                    if kind == "nan":
                        tensors[prefix + ".lora_A.weight"][0, 0] = math.nan
                    if kind == "mixed":
                        tensors[prefix + ".lora_up.weight"] = tensors.pop(prefix + ".lora_B.weight")
                    if kind == "unknown":
                        tensors[prefix + ".dora_scale"] = torch.ones(3)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(path))
                    with self.assertRaises(ValueError):
                        self.lora().apply(
                            Patcher(), root, {"loras": [{"name": path.name, "strength": 1.0}]}, mapper(), torch
                        )

    def test_all_targets_must_survive_loader_and_patcher_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/style.safetensors")
            request = {"loras": [{"name": path.name, "strength": 1.0}]}
            mapping = mapper()
            mapping.load_lora.return_value = {}
            mapping.load_lora.side_effect = None
            with self.assertRaisesRegex(ValueError, "全対象"):
                self.lora().apply(Patcher(), root, request, mapping, torch)
            original, clone = Patcher(), Patcher()
            original.clone = lambda: clone
            clone.add_patches = Mock(return_value=[])
            with self.assertRaisesRegex(ValueError, "全対象"):
                self.lora().apply(original, root, request, mapper(), torch)

    def test_native_separate_qkv_aliases_target_the_correct_fused_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/query.safetensors")
            original = Patcher()
            fused = "diffusion_model.layers.0.proj.qkv.weight"
            original.model = SimpleNamespace(state_dict=lambda: {fused: torch.zeros(9, 2, dtype=torch.int8)})
            mapping = mapper()
            target = (fused, (0, 0, 3))
            mapping.model_lora_keys_unet.return_value = {"diffusion_model." + PREFIX: target}
            patched, count = self.lora().apply(
                original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapping, torch
            )
            self.assertEqual(count, 1)
            self.assertEqual(set(patched.calls[0][0]), {target})
            mapping.model_lora_keys_unet.return_value = {"diffusion_model." + PREFIX: (fused, (0, 8, 3))}
            with self.assertRaises(ValueError):
                self.lora().apply(original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapping, torch)

    def test_packed_w4a8_uses_original_linear_dimensions_for_narrow_lora(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/query.safetensors")
            original = Patcher()
            fused = "diffusion_model.layers.0.proj.qkv.weight"
            original.model = SimpleNamespace(
                state_dict=lambda: {fused: torch.zeros(9, 1, dtype=torch.int8)},
                get_submodule=lambda name: SimpleNamespace(in_features=2, out_features=9),
            )
            mapping = mapper()
            target = (fused, (0, 3, 3))
            mapping.model_lora_keys_unet.return_value = {"diffusion_model." + PREFIX: target}
            patched, count = self.lora().apply(
                original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapping, torch
            )
            self.assertEqual(count, 1)
            self.assertEqual(set(patched.calls[0][0]), {target})

    def test_w4a8_unsorted_target_codebook_is_rejected_before_any_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/query.safetensors")
            for fused in (False, True):
                with self.subTest(fused=fused):
                    key = "diffusion_model.layers.0.proj.qkv.weight" if fused else TARGET
                    rows = 9 if fused else 3
                    target = (key, (0, 3, 3)) if fused else key
                    codebook = torch.arange(16, dtype=torch.float32)
                    codebook[4], codebook[5] = 5, 4
                    original_codebook = codebook.clone()
                    original = Patcher()
                    original.clone = Mock(side_effect=AssertionError("must reject before cloning"))
                    state = {
                        key: torch.zeros(rows, 1, dtype=torch.int8),
                        key.removesuffix(".weight") + ".comfy_quant": torch.zeros(5, dtype=torch.uint8),
                        key.removesuffix(".weight") + ".weight_s_rel": torch.zeros(rows, 1),
                        key.removesuffix(".weight") + ".weight_s_channel": torch.zeros(rows),
                        key.removesuffix(".weight") + ".weight_codebook": codebook,
                    }
                    original.model = SimpleNamespace(
                        state_dict=Mock(return_value=state),
                        get_submodule=lambda name, rows=rows: SimpleNamespace(in_features=2, out_features=rows),
                    )
                    mapping = mapper()
                    mapping.model_lora_keys_unet.return_value = {"diffusion_model." + PREFIX: target}
                    with self.assertRaisesRegex(ValueError, "コードブック"):
                        self.lora().apply(
                            original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapping, torch
                        )
                    mapping.load_lora.assert_not_called()
                    original.clone.assert_not_called()
                    self.assertTrue(torch.equal(codebook, original_codebook))

    def test_w4a8_permits_sorted_duplicate_zero_or_omitted_codebooks(self):
        codebooks = (
            None,
            torch.zeros(16),
            torch.tensor([-3.0, -1.0, -1.0, -0.5, -0.25, 0.0, 0.0, 0.1,
                          0.2, 0.4, 0.7, 0.9, 1.1, 2.0, 4.0, 9.0]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/query.safetensors")
            for codebook in codebooks:
                with self.subTest(codebook=codebook):
                    prefix = TARGET.removesuffix(".weight")
                    state = {
                        TARGET: torch.zeros(3, 1, dtype=torch.int8),
                        prefix + ".comfy_quant": torch.zeros(5, dtype=torch.uint8),
                        prefix + ".weight_s_rel": torch.zeros(3, 1),
                        prefix + ".weight_s_channel": torch.zeros(3),
                        "diffusion_model.layers.1.proj.q.weight": torch.zeros(3, 1, dtype=torch.int8),
                        "diffusion_model.layers.1.proj.q.comfy_quant": torch.zeros(5, dtype=torch.uint8),
                        "diffusion_model.layers.1.proj.q.weight_s_rel": torch.zeros(3, 1),
                        "diffusion_model.layers.1.proj.q.weight_s_channel": torch.zeros(3),
                        "diffusion_model.layers.1.proj.q.weight_codebook": torch.arange(15, -1, -1).float(),
                    }
                    if codebook is not None:
                        state[prefix + ".weight_codebook"] = codebook
                    original = Patcher()
                    original.model = SimpleNamespace(
                        state_dict=Mock(return_value=state),
                        get_submodule=lambda name: SimpleNamespace(in_features=2, out_features=3),
                    )
                    patched, count = self.lora().apply(
                        original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapper(), torch
                    )
                    self.assertIsNot(patched, original)
                    self.assertEqual(count, 1)

    def test_non_w4a8_target_does_not_check_unrelated_codebook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/query.safetensors")
            original = Patcher()
            original.model = SimpleNamespace(state_dict=lambda: {
                TARGET: torch.zeros(3, 2),
                TARGET.removesuffix(".weight") + ".weight_codebook": torch.arange(15, -1, -1).float(),
            })
            patched, count = self.lora().apply(
                original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapper(), torch
            )
            self.assertIsNot(patched, original)
            self.assertEqual(count, 1)

    def test_w4a8_nonmonotonic_codebook_is_ignored_when_no_lora_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = adapter(root / "loras/query.safetensors")
            original = Patcher()
            original.model = SimpleNamespace(state_dict=Mock(return_value={
                TARGET: torch.zeros(3, 1, dtype=torch.int8),
                TARGET.removesuffix(".weight") + ".weight_s_rel": torch.zeros(3, 1),
                TARGET.removesuffix(".weight") + ".weight_s_channel": torch.zeros(3),
                TARGET.removesuffix(".weight") + ".weight_codebook": torch.arange(15, -1, -1).float(),
            }))
            for selections in ([], [{"name": path.name, "strength": 0}]):
                with self.subTest(selections=selections):
                    mapping = mapper()
                    result, count = self.lora().apply(original, root, {"loras": selections}, mapping, torch)
                    self.assertIs(result, original)
                    self.assertEqual(count, 0)
                    mapping.load_lora.assert_not_called()
            original.model.state_dict.assert_not_called()

    def test_quantized_lora_is_rejected_when_logical_dimensions_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "loras/query.safetensors"
            path.parent.mkdir()
            save_file({PREFIX + ".lora_A.weight": torch.ones(1, 1),
                       PREFIX + ".lora_B.weight": torch.ones(3, 1)}, str(path))
            original = Patcher()
            original.model = SimpleNamespace(state_dict=lambda: {
                TARGET: torch.zeros(3, 1, dtype=torch.int8),
                TARGET.removesuffix(".weight") + ".comfy_quant": torch.zeros(5, dtype=torch.uint8),
            })
            with self.assertRaisesRegex(ValueError, "寸法"):
                self.lora().apply(
                    original, root, {"loras": [{"name": path.name, "strength": 1}]}, mapper(), torch
                )

    def test_normalized_aliases_cannot_duplicate_an_adapter_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "loras/duplicate.safetensors"
            path.parent.mkdir()
            save_file(
                {
                    PREFIX + ".lora_A.weight": torch.ones(1, 2),
                    PREFIX + ".lora_B.weight": torch.ones(3, 1),
                    "diffusion_model." + PREFIX + ".lora_A.weight": torch.ones(1, 2),
                    "diffusion_model." + PREFIX + ".lora_B.weight": torch.ones(3, 1),
                },
                str(path),
            )
            with self.assertRaisesRegex(ValueError, "重複"):
                self.lora().apply(Patcher(), root, {"loras": [{"name": path.name, "strength": 1}]}, mapper(), torch)


if __name__ == "__main__":
    unittest.main()
