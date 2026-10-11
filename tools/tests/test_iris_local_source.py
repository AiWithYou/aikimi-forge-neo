"""Header-only Iris checkpoint selection; no model, network, or GPU imports."""

from __future__ import annotations

import importlib
import json
import math
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from modules_forge import local_assets
from modules_forge.iris import local_source

NORMAL = {
    "s_embedder.proj.weight": ("BF16", [2, 2]),
    "t_embedder.mlp.0.weight": ("F32", [2, 2]),
    "final_layer.linear.weight": ("F32", [3, 16]),
    "blocks.0.adaln_img.bias": ("F32", [2]),
}
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
BYTE_WIDTH = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "I8": 1, "U8": 1}


def tensor_file(path, tensors=None, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    header, payload = {}, bytearray()
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dtype, shape) in (tensors if tensors is not None else NORMAL).items():
        length = math.prod(shape) * BYTE_WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + length]}
        payload.extend(bytes(length))
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def checkpoint(directory, name="model.safetensors", *, raw=None, tensors=None, metadata=None):
    path = tensor_file(directory / name, tensors, metadata)
    raw = {"model": {"hidden_size": 2}} if raw is None else raw
    (directory / "config.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def quantized(kind):
    tensors = dict(NORMAL)
    for name in ("s_embedder.proj", "t_embedder.mlp.0"):
        tensors.pop(name + ".weight")
        if kind == "int8":
            tensors[name + ".qweight"] = ("I8", [2, 4])
            tensors[name + ".scale"] = ("F32", [2, 1])
        else:
            tensors[name + ".qweight"] = ("I8", [16, 128])
            tensors[name + ".scale"] = ("F8_E4M3", [16, 16])
            tensors[name + ".s_channel"] = ("F32", [16])
            tensors[name + ".codebook"] = ("F32", [16])
    return tensors


class IrisLocalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / "models/Iris-3B"
        self.root.mkdir(parents=True)
        self.patch = patch.object(local_assets, "ROOT", self.project)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_unicode_spaces_absolute_quoted_and_project_relative_paths(self):
        path = checkpoint(self.project / "model/学習 重み/子 フォルダー", "水彩 300.safetensors")
        for value in (str(path), f'"{path}"', f"'{path}'", str(path.relative_to(self.project))):
            with self.subTest(value=value):
                result = local_source.resolve(self.root, value)
                self.assertEqual(result["weights"], path.resolve())
                self.assertEqual(result["config"], path.with_name("config.yaml").resolve())
                self.assertEqual(result["precision"], "normal")
                self.assertEqual(result["raw"]["model"], {"hidden_size": 2})

    def test_single_weight_folder_resolves_arbitrary_filename(self):
        path = checkpoint(self.project / "outside", "my-full-ft.SAFETENSORS")
        self.assertEqual(local_source.resolve(self.root, str(path.parent))["weights"], path)

    def test_multiple_weights_in_folder_require_explicit_file(self):
        first = checkpoint(self.project / "outside", "first.safetensors")
        second = tensor_file(first.with_name("second.safetensors"))
        with self.assertRaisesRegex(ValueError, "複数.*ファイル"):
            local_source.resolve(self.root, str(first.parent))
        self.assertEqual(local_source.resolve(self.root, str(second))["weights"], second)

    def test_empty_or_invalid_paths_fail_without_remote_lookup(self):
        for value in ("", "https://example.test/model.safetensors", "missing.safetensors", str(self.root)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                local_source.resolve(self.root, value)

    def test_discovery_defaults_missing_folder_to_empty(self):
        self.assertEqual(local_source.discover(self.root, ""), [])
        path = checkpoint(self.root / "checkpoints/学習/step 300", "best.safetensors")
        self.assertEqual([value for _, value in local_source.discover(self.root, "")], [str(path)])

    def test_discovery_rejects_explicit_invalid_directory(self):
        path = checkpoint(self.project / "outside")
        for directory in ("missing", str(path), "https://example.test/models"):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "フォルダー|このPC"):
                local_source.discover(self.root, directory)

    def test_discovery_recursive_duplicate_basenames_and_same_directory_files(self):
        first = checkpoint(self.project / "model/和風 A", "same.safetensors")
        second = checkpoint(self.project / "model/和風 B/子", "same.safetensors")
        third = tensor_file(first.with_name("other.safetensors"))
        choices = local_source.discover(self.root, "model")
        self.assertEqual({value for _, value in choices}, {str(first), str(second), str(third)})
        self.assertEqual(len({label for label, _ in choices}), 3)
        for label, value in choices:
            self.assertTrue(label.startswith(Path(value).name + " · "))
            self.assertIn(str(Path(value).parent), label)

    def test_no_manifest_environment_encoder_or_hash_needed(self):
        path = checkpoint(self.project / "outside")
        with patch.object(local_assets, "file_identity", side_effect=AssertionError("discovery must not hash weights")):
            self.assertEqual(local_source.resolve(self.root, str(path))["precision"], "normal")
            self.assertEqual(local_source.discover(self.root, str(path.parent))[0][1], str(path))
        self.assertFalse((self.root / "worker-env").exists())
        self.assertFalse((self.root / "text-encoder").exists())
        self.assertFalse((path.parent / "manifest.json").exists())

    def test_discovery_prunes_runtime_dependencies_and_does_not_follow_symlinks(self):
        path = checkpoint(self.root / "checkpoints/custom")
        for directory in (".git", ".cache", "worker-env", "iris-source-a8d15239dea4"):
            checkpoint(self.root / directory / "nested", "irrelevant.safetensors")
        link = self.root / "checkpoints/loop"
        try:
            link.symlink_to(self.root / "checkpoints", target_is_directory=True)
        except OSError:
            pass  # Windows may not grant symlink creation; dependency pruning still runs.
        choices = local_source.discover(self.root, str(self.project / "models"))
        self.assertEqual([value for _, value in choices], [str(path)])

    def test_config_required_explicit_nonempty_model_and_safe_yaml(self):
        path = checkpoint(self.project / "outside")
        config = path.with_name("config.yaml")
        config.unlink()
        with self.assertRaisesRegex(ValueError, "config.yaml"):
            local_source.resolve(self.root, str(path))
        for content in (
            "",
            "[]",
            "{}",
            "model: {}",
            "model: null",
            "model: []",
            "!!python/object:os.system {}",
            "model: [",
        ):
            with self.subTest(content=content):
                config.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "設定|model|config.yaml"):
                    local_source.resolve(self.root, str(path))

    def test_training_extras_and_machine_local_null_embed_override_are_allowed(self):
        raw = {
            "model": {"hidden_size": 2},
            "train": {"max_steps": 300},
            "data": {"dirs": ["日本語"]},
            "text_encoder": {"null_embed_dir": "different/computer/cache"},
        }
        path = checkpoint(self.project / "outside", raw=raw)
        self.assertEqual(local_source.resolve(self.root, str(path))["raw"], raw)

    def test_tasks_other_than_unmarked_generation_are_rejected(self):
        for name in ("depth", "restoration", "upscale", "unknown", "generate", ""):
            with self.subTest(name=name):
                path = checkpoint(self.project / "outside", raw={"model": {"hidden_size": 2}, "task": {"name": name}})
                with self.assertRaisesRegex(ValueError, "画像生成|task"):
                    local_source.resolve(self.root, str(path))
        for task in (None, {}, [], "depth"):
            with self.subTest(task=task):
                path = checkpoint(self.project / "outside", raw={"model": {"hidden_size": 2}, "task": task})
                with self.assertRaises(ValueError):
                    local_source.resolve(self.root, str(path))

    def test_shared_encoder_defaults_and_local_reference_are_allowed(self):
        for pretrained in ("Qwen/Qwen3-VL-4B-Instruct", str(self.root / "text-encoder"), "models/Iris-3B/text-encoder"):
            with self.subTest(pretrained=pretrained):
                raw = {
                    "model": {"text_dim": 2560, "text_len": 300},
                    "text_encoder": {
                        "name": "qwen3_vl",
                        "pretrained": pretrained,
                        "dim": 2560,
                        "max_length": 300,
                        "hidden_layers": LAYERS,
                    },
                }
                path = checkpoint(self.project / "outside", raw=raw)
                self.assertEqual(local_source.resolve(self.root, str(path))["precision"], "normal")

    def test_shared_encoder_accepts_default_and_explicit_layerwise_adapter(self):
        for settings in ({}, {"text_adapter": "lap_blocks2", "text_lap_num_layers": len(LAYERS)}):
            with self.subTest(settings=settings):
                path = checkpoint(self.project / "outside", raw={"model": {"hidden_size": 2, **settings}})
                self.assertEqual(local_source.resolve(self.root, str(path))["precision"], "normal")

    def test_shared_encoder_rejects_incompatible_adapter_or_layer_count(self):
        changes = (
            {"text_adapter": "linear"},
            {"text_adapter": "blocks2"},
            {"text_adapter": None},
            {"text_lap_num_layers": 8},
            {"text_lap_num_layers": True},
            {"text_lap_num_layers": 12.0},
            {"text_lap_num_layers": "12"},
        )
        for settings in changes:
            with self.subTest(settings=settings):
                path = checkpoint(self.project / "outside", raw={"model": {"hidden_size": 2, **settings}})
                key = next(iter(settings))
                with self.assertRaisesRegex(ValueError, f"model\\.{key}"):
                    local_source.resolve(self.root, str(path))

    def test_foreign_encoder_or_text_dimensions_are_rejected(self):
        changes = (
            {"name": "clip"},
            {"pretrained": "Qwen/Qwen3-VL-8B-Instruct"},
            {"pretrained": str(self.project / "other-encoder")},
            {"dim": 2048},
            {"max_length": 256},
            {"hidden_layers": [1, 2]},
            {"hidden_layers": None},
        )
        for encoder in changes:
            with self.subTest(encoder=encoder):
                path = checkpoint(self.project / "outside", raw={"model": {"hidden_size": 2}, "text_encoder": encoder})
                with self.assertRaisesRegex(ValueError, "エンコーダー|text_encoder"):
                    local_source.resolve(self.root, str(path))
        for model in ({"text_dim": 2048}, {"text_len": 256}):
            with self.subTest(model=model):
                path = checkpoint(self.project / "outside", raw={"model": model})
                with self.assertRaisesRegex(ValueError, "エンコーダー|model.text"):
                    local_source.resolve(self.root, str(path))
        for encoder in (None, "qwen3_vl", []):
            path = checkpoint(self.project / "outside", raw={"model": {"hidden_size": 2}, "text_encoder": encoder})
            with self.assertRaises(ValueError):
                local_source.resolve(self.root, str(path))

    def test_wrong_family_lora_only_and_partial_fingerprint_are_rejected(self):
        alternatives = (
            {"transformer_blocks.0.attn.to_q.weight": ("F32", [2, 2])},
            {"blocks.0.attn.lora_A.weight": ("F32", [2, 2])},
            {key: tensor for key, tensor in NORMAL.items() if key != "final_layer.linear.weight"},
            {key: tensor for key, tensor in NORMAL.items() if not key.startswith("blocks.0.")},
        )
        for tensors in alternatives:
            with self.subTest(keys=list(tensors)):
                path = checkpoint(self.project / "outside", tensors=tensors)
                with self.assertRaisesRegex(ValueError, "Iris|Full FT"):
                    local_source.resolve(self.root, str(path))

    def test_incomplete_tensor_data_and_invalid_descriptors_are_rejected(self):
        path = checkpoint(self.project / "outside")
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "不完全"):
            local_source.resolve(self.root, str(path))
        tensors = dict(NORMAL, **{"final_layer.linear.weight": ("I8", [3, 16])})
        tensor_file(path, tensors)
        with self.assertRaises(ValueError):
            local_source.resolve(self.root, str(path))
        raw = json.dumps(
            {key: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]} for key, (dtype, shape) in NORMAL.items()}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaises(ValueError):
            local_source.resolve(self.root, str(path))

    def test_misleading_int8_folder_uses_float_contents_and_accepts_pt_metadata(self):
        path = checkpoint(self.project / "outside/int8", metadata={"format": "pt"})
        self.assertEqual(local_source.resolve(self.root, str(path))["precision"], "normal")

    def test_valid_rowwise_int8_with_or_without_format_marker(self):
        for metadata in (
            None,
            {"format": "iris-rowwise-int8-v1"},
            {"format": "iris-rowwise-int8-v1", "source_repo": "speridlabs/iris-3b"},
        ):
            with self.subTest(metadata=metadata):
                path = checkpoint(self.project / "outside/normal", tensors=quantized("int8"), metadata=metadata)
                self.assertEqual(local_source.resolve(self.root, str(path))["precision"], "int8")

    def test_rowwise_int8_requires_signed_weights_and_matching_float_scales(self):
        for replacement in (
            {"s_embedder.proj.qweight": ("U8", [2, 4])},
            {"s_embedder.proj.scale": ("F32", [3, 1])},
            {"s_embedder.proj.scale": ("BF16", [2, 1])},
            {"s_embedder.proj.scale": ("F32", [2])},
            {"s_embedder.proj.weight": ("F32", [2, 4])},
        ):
            with self.subTest(replacement=replacement):
                path = checkpoint(self.project / "outside", tensors={**quantized("int8"), **replacement})
                with self.assertRaisesRegex(ValueError, "量子化|INT8"):
                    local_source.resolve(self.root, str(path))
        tensors = quantized("int8")
        tensors.pop("s_embedder.proj.scale")
        path = checkpoint(self.project / "outside", tensors=tensors)
        with self.assertRaises(ValueError):
            local_source.resolve(self.root, str(path))

    def test_valid_w4a8_requires_packed_shapes_and_marker(self):
        path = checkpoint(
            self.project / "outside", tensors=quantized("w4a8"), metadata={"format": "iris-convrot-w4a8-v1"}
        )
        self.assertEqual(local_source.resolve(self.root, str(path))["precision"], "w4a8")
        for replacement in (
            {"s_embedder.proj.qweight": ("U8", [16, 128])},
            {"s_embedder.proj.qweight": ("I8", [15, 128])},
            {"s_embedder.proj.qweight": ("I8", [16, 127])},
            {"s_embedder.proj.scale": ("F32", [16, 16])},
            {"s_embedder.proj.scale": ("F8_E4M3", [16, 15])},
            {"s_embedder.proj.s_channel": ("F32", [15])},
            {"s_embedder.proj.codebook": ("F32", [15])},
        ):
            with self.subTest(replacement=replacement):
                path = checkpoint(
                    self.project / "outside",
                    tensors={**quantized("w4a8"), **replacement},
                    metadata={"format": "iris-convrot-w4a8-v1"},
                )
                with self.assertRaisesRegex(ValueError, "W4A8|量子化"):
                    local_source.resolve(self.root, str(path))
        tensors = quantized("w4a8")
        tensors.pop("s_embedder.proj.s_channel")
        path = checkpoint(self.project / "outside", tensors=tensors, metadata={"format": "iris-convrot-w4a8-v1"})
        with self.assertRaises(ValueError):
            local_source.resolve(self.root, str(path))

    def test_cross_format_unknown_metadata_and_content_disagreement_are_rejected(self):
        cases = (
            (NORMAL, {"format": "iris-rowwise-int8-v1"}),
            (NORMAL, {"format": "iris-convrot-w4a8-v1"}),
            (NORMAL, {"format": "some-foreign-format"}),
            (NORMAL, {"quantization": "int8"}),
            (NORMAL, {"quantization_config": "nf4"}),
            (quantized("int8"), {"format": "pt"}),
            (quantized("int8"), {"format": "iris-convrot-w4a8-v1"}),
            (quantized("w4a8"), {"format": "iris-rowwise-int8-v1"}),
            (quantized("w4a8"), None),
        )
        for tensors, metadata in cases:
            with self.subTest(metadata=metadata):
                path = checkpoint(self.project / "outside", tensors=tensors, metadata=metadata)
                with self.assertRaisesRegex(ValueError, "保存形式|量子化|INT8|W4A8"):
                    local_source.resolve(self.root, str(path))

    def test_discovery_excludes_invalid_assets_but_lists_each_valid_file(self):
        expected = checkpoint(self.project / "models/custom/valid")
        checkpoint(self.project / "models/custom/foreign", tensors={"foreign.weight": ("F32", [1])})
        checkpoint(self.project / "models/custom/depth", raw={"model": {"hidden_size": 2}, "task": {"name": "depth"}})
        checkpoint(
            self.project / "models/custom/encoder", raw={"model": {"hidden_size": 2}, "text_encoder": {"name": "clip"}}
        )
        bad = checkpoint(self.project / "models/custom/truncated")
        bad.write_bytes(bad.read_bytes()[:-1])
        tensor_file(self.project / "models/custom/no-config/weights.safetensors")
        self.assertEqual([value for _, value in local_source.discover(self.root, "models")], [str(expected)])

    def test_module_import_keeps_heavy_model_dependencies_unloaded(self):
        original = __import__

        def lightweight_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "iris3b", "safetensors", "omegaconf"}:
                raise AssertionError("UI discovery must not import " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=lightweight_import):
            importlib.reload(local_source)


if __name__ == "__main__":
    unittest.main()
