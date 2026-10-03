"""Local asset contracts with no official denoiser, network, or CUDA requirement."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from modules_forge import local_assets, ming_image_studio, ming_local
from modules_forge.qwen_image21 import core, local_source, quantized_cache, style_lora
from tools.tests.test_additional_module_identity import load_main_entry, stub_modules


def tensor_file(path, keys=None, value=1, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or {"weight": [1]}
    header = {"__metadata__": metadata or {}}
    payload = b""
    for key, shape in keys.items():
        count = 1
        for n in shape:
            count *= n
        data = struct.pack("<f", value) * count
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def qwen_fixture(root, *, transformer=True):
    json_file(root / "model_index.json", {"_class_name": "QwenImage21Pipeline", **local_source.COMPONENTS})
    json_file(root / "text_encoder/config.json", {"model_type": "qwen3_vl"})
    json_file(root / "vae/config.json", {"_class_name": "AutoencoderKLQwenImage21"})
    json_file(root / "processor/processor_config.json", {"test": True})
    json_file(root / "scheduler/scheduler_config.json", {"_class_name": "FlowMatchEulerDiscreteScheduler"})
    for name in ("text_encoder", "vae"):
        tensor_file(root / name / "model.safetensors")
    if transformer:
        qwen_transformer(root / "transformer")
    return root


def qwen_transformer(root):
    json_file(root / "config.json", local_source.TRANSFORMER_CONFIG)
    keys = {f"transformer_blocks.{i}.attn.to_q.weight": [1, 1] for i in range(32)}
    keys.update({"img_in.weight": [1, 1], "txt_in.in_layer.weight": [1, 1]})
    tensor_file(root / "diffusion_pytorch_model.safetensors", keys)
    return root


class LocalAssetsTests(unittest.TestCase):
    def test_disk_hash_cache_survives_process_cache_reset_but_not_overwrite(self):
        with TemporaryDirectory() as folder, patch.object(local_assets, "HASH_CACHE", Path(folder) / "hashes.json"):
            path = tensor_file(Path(folder) / "weights.safetensors")
            first = local_assets.file_identity(path)
            local_assets._digest.cache_clear()
            with patch.object(local_assets.hashlib, "file_digest", side_effect=AssertionError("unexpected rehash")):
                self.assertEqual(local_assets.file_identity(path), first)
            tensor_file(path, value=9)
            self.assertNotEqual(local_assets.file_identity(path), first)

    def test_equal_header_and_same_size_content_change_never_share_identity(self):
        with TemporaryDirectory() as folder:
            first = tensor_file(Path(folder) / "one.safetensors", value=1)
            second = tensor_file(Path(folder) / "two.safetensors", value=2)
            self.assertEqual(local_assets.read_header(first), local_assets.read_header(second))
            self.assertNotEqual(
                local_assets.file_identity(first)["sha256"], local_assets.file_identity(second)["sha256"]
            )
            old = local_assets.file_identity(first)
            stamp = first.stat().st_mtime_ns
            tensor_file(first, value=3)
            os.utime(first, ns=(stamp, stamp))
            self.assertNotEqual(old["sha256"], local_assets.file_identity(first)["sha256"])

    def test_lora_rows_keep_strength_by_identity_and_reject_bad_values(self):
        self.assertEqual(
            local_assets.lora_rows(["b", "c", "a"], [["a", 0.5], ["b", 0]]), [["b", 0], ["c", 1], ["a", 0.5]]
        )
        for value in (True, float("nan"), float("inf"), 2.1, "oops"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                local_assets.lora_settings(["a"], [["a", value]])

    def test_filename_first_labels_keep_distinct_paths_and_edited_strengths(self):
        a = str((local_assets.ROOT / "a/style.safetensors").resolve())
        b = str((local_assets.ROOT / "b/style.safetensors").resolve())
        rows = local_assets.lora_rows([a, b], [[a, 0.25], [b, -0.5]])
        self.assertTrue(rows[0][0].startswith("style.safetensors · "))
        self.assertNotEqual(rows[0][0], rows[1][0])
        self.assertEqual(
            local_assets.lora_settings([b, a], rows), ({"name": b, "strength": -0.5}, {"name": a, "strength": 0.25})
        )

    def test_library_keeps_missing_selections_and_external_paths_without_copy(self):
        with TemporaryDirectory() as folder, patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"):
            local_assets.remember("model", ["X:/missing/model.safetensors"])
            local_assets.save_selection("engine", {"model": "X:/missing/model.safetensors"})
            self.assertEqual(local_assets.selection("engine")["model"], "X:/missing/model.safetensors")
            self.assertEqual(len(local_assets.choices("model")), 1)


class QwenLocalSourceTests(unittest.TestCase):
    def test_folder_loader_rejects_missing_and_wrong_shaped_tensors(self):
        fake = SimpleNamespace(
            from_config=lambda _: SimpleNamespace(state_dict=lambda: {"weight": SimpleNamespace(shape=(2, 3))})
        )
        with (
            TemporaryDirectory() as folder,
            stub_modules({"accelerate": SimpleNamespace(init_empty_weights=nullcontext)}),
        ):
            path = Path(folder)
            json_file(path / "config.json", local_source.TRANSFORMER_CONFIG)
            for keys in ({"weight": [2, 2]}, {"other": [2, 3]}, {"weight": [2, 3], "extra": [1]}):
                tensor_file(path / "model.safetensors", keys)
                with self.assertRaisesRegex(ValueError, "部分読み込み"):
                    local_source.validate_folder_tensors(path, fake)
            tensor_file(path / "model.safetensors", {"weight": [2, 3]})
            local_source.validate_folder_tensors(path, fake)

    def test_custom_transformer_uses_shared_components_without_official_denoiser(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            shared = qwen_fixture(root / "runtime/model", transformer=False)
            custom = qwen_transformer(root / "outside/fine-tune")
            result = local_source.resolve(root / "runtime", str(custom), "", "int8")
            self.assertEqual(result["transformer"], str(custom.resolve()))
            self.assertEqual(result["model"], str(shared.resolve()))
            self.assertFalse((shared / "transformer").exists())
            self.assertEqual(
                sorted(p.name for p in custom.iterdir()), ["config.json", "diffusion_pytorch_model.safetensors"]
            )

    def test_full_pipeline_does_not_require_managed_model_or_inventory(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pipeline = qwen_fixture(root / "external")
            runtime = root / "runtime"
            python = runtime / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.write_bytes(b"test")
            json_file(runtime / "runtime.json", {"schema": 1, "diffusers_revision": core.DIFFUSERS_REVISION})
            result = core.runtime_manifest(runtime, "bf16", str(pipeline))
            self.assertEqual(result["model"], str(pipeline.resolve()))
            self.assertFalse((runtime / "model").exists())

    def test_wrong_family_missing_shard_and_profile_rejected(self):
        with TemporaryDirectory() as folder:
            root = qwen_fixture(Path(folder))
            for precision in ("turbo_bf16", "base_q4_k_m"):
                with self.subTest(precision=precision), self.assertRaises(ValueError):
                    local_source.resolve(root.parent, str(root), "", precision)
            json_file(root / "transformer/config.json", {"_class_name": "QwenImageTransformer2DModel"})
            with self.assertRaisesRegex(ValueError, "旧Qwen"):
                local_source.resolve(root.parent, str(root), "", "int8")
            json_file(root / "transformer/config.json", local_source.TRANSFORMER_CONFIG)
            json_file(
                root / "transformer/model.safetensors.index.json", {"weight_map": {"weight": "../outside.safetensors"}}
            )
            with self.assertRaises(ValueError):
                local_source.resolve(root.parent, str(root), "", "int8")

    def test_quantized_cache_binds_source_content_not_display_name(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            a = qwen_transformer(root / "a")
            b = qwen_transformer(root / "b")
            versions = dict.fromkeys(
                ("torch", "diffusers", "transformers", "bitsandbytes", "accelerate", "safetensors"), "test"
            )
            first = quantized_cache.component_identity(root, "transformer", "int8", source_path=a, versions=versions)
            second = quantized_cache.component_identity(root, "transformer", "int8", source_path=b, versions=versions)
            self.assertNotEqual(quantized_cache.cache_path(root, first), quantized_cache.cache_path(root, second))

    def test_absolute_loras_and_duplicate_real_file(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = tensor_file(
                root / "loras/style.safetensors",
                {
                    "transformer_blocks.0.attn.to_q.lora_A.weight": [1, 2],
                    "transformer_blocks.0.attn.to_q.lora_B.weight": [2, 1],
                },
            )
            self.assertEqual(style_lora.resolve(root, str(path)), path.resolve())
            self.assertEqual(style_lora.resolve(root, f'"{path}"'), path.resolve())
            request = {
                "style_loras": [{"name": "style.safetensors", "strength": 1}, {"name": str(path), "strength": 0.5}]
            }
            with self.assertRaisesRegex(ValueError, "二重"):
                style_lora.validate_installed(root, request)
            request["style_loras"][1]["strength"] = 0
            self.assertEqual(len(style_lora.validate_installed(root, request)), 1)


class MingLocalSourceTests(unittest.TestCase):
    def test_all_custom_files_need_no_managed_weights_and_graph_chains_two_loras(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = tensor_file(
                root / "custom/model.safetensors",
                {"layers.0.attention.qkv.weight": [3, 1]},
                metadata={"config": json.dumps({"transformer": {"image_model": "ming_image"}})},
            )
            encoder = tensor_file(root / "custom/encoder.safetensors")
            vae = tensor_file(root / "custom/vae.safetensors")
            a = tensor_file(root / "a/style.safetensors")
            b = tensor_file(root / "b/style.safetensors")
            request = ming_image_studio.MingImageRequest(
                "poster",
                model_path=str(model),
                text_encoder_path=str(encoder),
                vae_path=str(vae),
                loras=(
                    {"name": str(a), "strength": 0.5},
                    {"name": str(b), "strength": -1},
                    {"name": "missing", "strength": 0},
                ),
            )
            assets = ming_local.selected_assets(request, root / "empty-runtime")
            graph = ming_local.apply_graph(ming_image_studio.build_workflow(request, 1), request, assets)
            self.assertEqual(graph["sampling"]["inputs"]["model"], ["local_lora_1", 0])
            self.assertEqual(graph["local_lora_1"]["inputs"]["model"], ["local_lora_0", 0])
            self.assertEqual(graph["local_lora_0"]["inputs"]["strength"], 0.5)
            self.assertFalse((root / "empty-runtime").exists())
            self.assertEqual(len(assets["loras"]), 2)
            with self.assertRaises(ValueError):
                ming_local.selected_assets(
                    replace(request, loras=({"name": str(a), "strength": 1}, {"name": str(a), "strength": 0.2})), root
                )

    def test_setup_modes_do_not_plan_a_default_denoiser(self):
        from tools import setup_ming_image

        with TemporaryDirectory() as folder, patch.object(setup_ming_image, "runtime_ready", return_value=False):
            empty = setup_ming_image.run(root=Path(folder), dry_run=True, runtime_only=True)
            self.assertEqual(empty["plans"], [])
            shared = setup_ming_image.run(root=Path(folder), dry_run=True, components_only=True)
            self.assertNotIn("diffusion_models", json.dumps(shared))

    def test_ming_lora_node_refuses_mutated_file(self):
        location = ming_local.NODE_SOURCE / "__init__.py"
        spec = importlib.util.spec_from_file_location("ming_local_nodes_test", location)
        nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodes)
        with TemporaryDirectory() as folder:
            path = tensor_file(Path(folder) / "lora.safetensors")
            digest = local_assets.file_identity(path)["sha256"]
            self.assertEqual(nodes.checked_file(str(path), digest), str(path))
            tensor_file(path, value=2)
            with self.assertRaises(ValueError):
                nodes.checked_file(str(path), digest)

    def test_ming_lora_rejects_partial_pairs_unknown_targets_and_wrong_shapes(self):
        import torch

        spec = importlib.util.spec_from_file_location("ming_local_math_test", ming_local.NODE_SOURCE / "__init__.py")
        nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodes)
        comfy = ModuleType("comfy")
        comfy.lora = SimpleNamespace(
            model_lora_keys_unet=lambda *_: {},
            load_lora=lambda weights, mapping: {target: object() for target in mapping.values()},
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                state_dict=lambda: {"diffusion_model.layers.0.attention.qkv.weight": torch.ones(4, 3)}
            )
        )
        down = "layers.0.attention.qkv.lora_A.weight"
        up = "layers.0.attention.qkv.lora_B.weight"
        good = {down: torch.ones(1, 3), up: torch.ones(4, 1)}
        with stub_modules({"comfy": comfy, "comfy.lora": comfy.lora}):
            self.assertEqual(len(nodes.linear_patches(model, good)), 1)
            for weights in (
                {down: good[down]},
                {**good, "unknown.weight": torch.ones(1)},
                {**good, up: torch.ones(5, 1)},
            ):
                with self.assertRaises(ValueError):
                    nodes.linear_patches(model, weights)
            model.model.state_dict = lambda: {"diffusion_model.layers.0.attention.qkv.weight": torch.ones(4, 2)}
            model.model.get_submodule = lambda _: SimpleNamespace(out_features=4, in_features=3)
            self.assertEqual(
                len(nodes.linear_patches(model, good)), 1, "Packed W4A8 storage must use logical layer dimensions"
            )


class ForgeLocalSelectionTests(unittest.TestCase):
    def test_external_module_with_same_basename_stays_distinct(self):
        ui, _, _ = load_main_entry()
        with TemporaryDirectory() as folder, patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"):
            root = Path(folder)
            installed = tensor_file(root / "vae/shared.safetensors")
            external = tensor_file(root / "external/shared.safetensors")
            local_assets.remember("forge_module", [str(external)])
            ui._rebuild_module_registry([("VAE", str(installed.parent))])
            self.assertEqual(set(ui.module_list.values()), {str(installed.resolve()), str(external.resolve())})
            self.assertEqual(len(ui.module_list), 2)

    def test_multiple_external_loras_keep_path_identity_and_zero_skips_missing_file(self):
        from tools.tests.test_ui_loadsave_selection import load_add_component

        package = ModuleType("modules")
        package.scripts = SimpleNamespace(Script=object, AlwaysVisible=True)
        package.shared = SimpleNamespace()
        spec = importlib.util.spec_from_file_location(
            "local_stack_test", local_assets.ROOT / "scripts/local_lora_stack.py"
        )
        picker = importlib.util.module_from_spec(spec)
        with stub_modules({"modules": package}):
            spec.loader.exec_module(picker)
        networks = SimpleNamespace(available_networks={}, available_network_aliases={})
        network = SimpleNamespace(NetworkOnDisk=lambda name, filename: SimpleNamespace(name=name, filename=filename))
        with (
            TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
            stub_modules({"network": network, "networks": networks}),
        ):
            a = str(tensor_file(Path(folder) / "a/style.safetensors"))
            b = str(tensor_file(Path(folder) / "b/style.safetensors"))
            p = SimpleNamespace(prompt=["one", "two"], all_prompts=["one", "two"], extra_generation_params={})
            picker.apply_selections(p, [a, b, "missing"], [[a, 0.5], [b, -0.7], ["missing", 0]])
            self.assertEqual(len(networks.available_networks), 2)
            self.assertEqual(p.prompt[0].count("<lora:"), 2)
            self.assertEqual(p.prompt, p.all_prompts)
            with self.assertRaisesRegex(ValueError, "同じファイル"):
                picker.apply_selections(p, [a], [[a, 1]])

            with patch.object(picker, "available", return_value=[(name, name) for name in (a, b, "missing")]):
                with picker.gr.Blocks(analytics_enabled=False):
                    names, _rows = picker.Script().ui(False)
            defaults = SimpleNamespace(
                finalized_ui=False,
                ui_settings={"stack/value": []},
                component_mapping={},
            )
            load_add_component()(defaults, "stack", names)
            self.assertEqual(names.value, [a, b, "missing"], "Legacy UI defaults must not erase the asset selection")


if __name__ == "__main__":
    unittest.main()
