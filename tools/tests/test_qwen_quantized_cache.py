"""Persistence must bypass quantization and preserve every packed scale."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from modules_forge.qwen_image21 import quantized_cache as cache

VERSIONS = dict.fromkeys(
    ("torch", "diffusers", "transformers", "bitsandbytes", "accelerate", "safetensors", "comfy-kitchen"), "test"
)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / "model"
        self.model.mkdir()
        self.identity = {"component": "transformer", "precision": "int8", "revision": "abc"}

    def save(self, value, folder):
        (folder / "model.json").write_text(json.dumps(value), encoding="utf-8")

    def test_next_process_reuses_saved_checkpoint_without_conversion(self):
        create = mock.Mock(return_value={"value": 42})

        def load(path):
            return json.loads((path / "model.json").read_text())

        first, created = cache.load_or_create(self.model, self.identity, load, create, self.save)
        second, hit = cache.load_or_create(self.model, self.identity, load, create, self.save)
        self.assertEqual(first, second)
        self.assertEqual(created["status"], "created")
        self.assertEqual(hit["status"], "hit")
        create.assert_called_once()
        cache.manifest(hit["path"], self.identity, verify_hashes=True)

    def test_cancelled_write_is_not_published_or_loaded(self):
        def fail(value, path):
            self.save(value, path)
            raise InterruptedError("cancel")

        with self.assertRaises(InterruptedError):
            cache.load_or_create(self.model, self.identity, mock.Mock(), lambda: {}, fail)
        self.assertFalse(cache.cache_path(self.model, self.identity).exists())
        self.assertEqual(len(list((self.model.parent / "quantized").rglob("*.building-*"))), 1)

    def test_corrupted_artifact_is_preserved_and_rebuilt(self):
        _, first = cache.load_or_create(self.model, self.identity, mock.Mock(), lambda: {}, self.save)
        (Path(first["path"]) / "model.json").write_text("broken", encoding="utf-8")
        loader = mock.Mock()
        _, second = cache.load_or_create(self.model, self.identity, loader, lambda: {}, self.save)
        loader.assert_not_called()
        self.assertEqual(second["status"], "created")
        self.assertEqual(len(list(Path(first["path"]).parent.glob("transformer.invalid-*"))), 1)

    def test_changed_recipe_or_revision_has_separate_path(self):
        for key in ("precision", "revision"):
            changed = {**self.identity, key: "different"}
            self.assertNotEqual(cache.cache_path(self.model, self.identity), cache.cache_path(self.model, changed))

    def test_official_transformer_identity_retains_source_manifest_and_separates_base(self):
        base = self.model / "transformer"
        official = self.model.parent / "turbo" / "official" / "transformer"
        for folder in (base, official):
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            (folder / "weights.safetensors").write_bytes(b"same-fixture-weights")
        record = {
            "model": "Qwen/Qwen-Image-2.1-Turbo",
            "revision": "d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
            "files": {"official/transformer/weights.safetensors": {"size": 20, "sha256": "pinned"}},
        }
        normal = cache.component_identity(self.model, "transformer", "int8", versions=VERSIONS)
        identity = cache.component_identity(
            self.model, "transformer", "int8", source_path=official, source_record=record, versions=VERSIONS
        )
        self.assertEqual(identity["revision"], record["revision"])
        self.assertEqual(identity["source_record"], record)
        self.assertNotEqual(cache.cache_path(self.model, normal), cache.cache_path(self.model, identity))
        changed = cache.component_identity(
            self.model,
            "transformer",
            "int8",
            source_path=official,
            source_record={**record, "revision": "different"},
            versions=VERSIONS,
        )
        self.assertNotEqual(cache.cache_path(self.model, identity), cache.cache_path(self.model, changed))

    def test_official_saved_components_reuses_normal_encoder_identity(self):
        record = {"model": "Qwen/Qwen-Image-2.1-Turbo", "revision": "pinned", "files": {}}
        for selected, underlying in (("turbo_official_int8", "int8"), ("turbo_official_w4a8", "w4a8")):
            with (
                self.subTest(precision=selected),
                mock.patch("modules_forge.qwen_image21.turbo.turbo_manifest", return_value=record),
                mock.patch.object(
                    cache, "component_identity", side_effect=lambda _p, n, p, **_k: {"component": n, "precision": p}
                ) as identity,
                mock.patch.object(cache, "manifest") as validate,
            ):
                result = cache.saved_components(self.model, selected, versions=VERSIONS)
                transformer, encoder = identity.call_args_list
                self.assertEqual(transformer.args[2], underlying)
                self.assertEqual(
                    transformer.kwargs["source_path"], self.model.parent / "turbo" / "official" / "transformer"
                )
                self.assertEqual(transformer.kwargs["source_record"], record)
                self.assertEqual(encoder.args[2], underlying)
                self.assertNotIn("source_path", encoder.kwargs)
                self.assertNotIn("source_record", encoder.kwargs)
                self.assertEqual(validate.call_count, 2)
                self.assertEqual(set(result), {"transformer", "text_encoder"})

    def test_prepare_cli_accepts_official_quantized_profiles_without_loading_for_dry_run(self):
        from tools.prepare_qwen21_quantized import main

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "--root",
                        str(self.model.parent),
                        "--precision",
                        "turbo_official_int8",
                        "turbo_official_w4a8",
                        "--dry-run",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(output.getvalue())["precision"], ["turbo_official_int8", "turbo_official_w4a8"])

    def test_prepare_cli_validates_selected_source_for_each_precision(self):
        from tools.prepare_qwen21_quantized import main

        root = self.model.parent.resolve()
        with (
            redirect_stdout(io.StringIO()),
            mock.patch("modules_forge.qwen_image21.core.runtime_lock"),
            mock.patch("modules_forge.qwen_image21.core.runtime_manifest") as validate,
            mock.patch.object(cache, "saved_components", return_value={"transformer": {"path": "saved"}}) as saved,
            mock.patch("tools.qwen_image21_worker.clear_runtime"),
        ):
            self.assertEqual(main(["--root", str(root), "--precision", "int8", "turbo_official_w4a8", "--verify"]), 0)
        self.assertEqual(validate.call_args_list, [mock.call(root, "int8"), mock.call(root, "turbo_official_w4a8")])
        self.assertEqual(
            saved.call_args_list,
            [
                mock.call(root / "model", "int8", verify_hashes=True),
                mock.call(root / "model", "turbo_official_w4a8", verify_hashes=True),
            ],
        )

    def test_official_saved_status_checks_quantized_cache_instead_of_only_installation(self):
        for selected in ("turbo_official_int8", "turbo_official_w4a8"):
            with self.subTest(precision=selected), mock.patch.object(cache, "saved_components") as saved:
                saved.side_effect = FileNotFoundError("missing converted weights")
                self.assertIn("未保存", cache.saved_status(self.model.parent, selected))
                saved.side_effect = None
                self.assertIn("保存済み", cache.saved_status(self.model.parent, selected))
                self.assertEqual(saved.call_args.args, (self.model, selected))

    def test_device_or_memory_failure_keeps_healthy_checkpoint(self):
        _, info = cache.load_or_create(self.model, self.identity, mock.Mock(), lambda: {}, self.save)
        for error in (MemoryError("RAM"), RuntimeError("CUDA out of memory"), PermissionError("sharing")):
            create = mock.Mock()
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                cache.load_or_create(self.model, self.identity, mock.Mock(side_effect=error), create, self.save)
            create.assert_not_called()
            self.assertTrue(Path(info["path"]).is_dir())

    def test_manifest_rejects_escape(self):
        _, info = cache.load_or_create(self.model, self.identity, mock.Mock(), lambda: {}, self.save)
        path = Path(info["path"]) / "complete.json"
        record = json.loads(path.read_text())
        record["files"][0]["path"] = "../../outside"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "path"):
            cache.manifest(path.parent, self.identity)

    def test_same_size_modified_content_is_hash_checked(self):
        _, info = cache.load_or_create(self.model, self.identity, mock.Mock(), lambda: {}, self.save)
        path = Path(info["path"]) / "model.json"
        previous = path.stat()
        path.write_text("[]", encoding="utf-8")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            cache.manifest(path.parent, self.identity)

    def test_w4a8_sharded_roundtrip_bypasses_packing(self):
        import torch

        from modules_forge.qwen_image21.w4a8 import convert_model, load_saved_model, save_model
        from tools.tests.test_qwen_image21_w4a8 import tiny_model

        model = tiny_model().to(dtype=torch.bfloat16)
        stats = convert_model(model, "transformer", device="cpu")
        folder = self.model / "saved"
        save_model(model, folder, stats, max_shard_bytes=100000)
        with mock.patch("modules_forge.qwen_image21.w4a8._pack_weight", side_effect=AssertionError("repacked")):
            restored, restored_stats = load_saved_model(folder, "transformer", tiny_model)
        x = torch.randn(2, 4, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(model.transformer_blocks[0](x), restored.transformer_blocks[0](x), rtol=0, atol=0)
        self.assertEqual(stats, restored_stats)

    @unittest.skipUnless(os.environ.get("QWEN_QUANTIZED_CACHE_GPU_TEST") == "1", "opt-in saved INT8/CUDA check")
    def test_int8_save_reload_offload_is_exact(self):
        import torch
        from diffusers import BitsAndBytesConfig, QwenImage21Transformer2DModel

        from tools.qwen_image21_worker import INT8_SKIP_MODULES, _install_int8_offload_fix, _int8_layers

        source = self.model / "source"
        saved = self.model / "int8"
        model = QwenImage21Transformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=16,
            context_in_dim=256,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        ).to(dtype=torch.bfloat16)
        model.save_pretrained(source)
        del model
        model = QwenImage21Transformer2DModel.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=list(INT8_SKIP_MODULES)),
            local_files_only=True,
        )
        self.assertGreater(_int8_layers(model), 0)
        _install_int8_offload_fix(model)
        layer_name = next(name for name, layer in model.named_modules() if layer.__class__.__name__ == "Linear8bitLt")
        layer = model.get_submodule(layer_name)
        x = torch.randn(1, 4, layer.in_features, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            expected = layer(x)
        model.to("cpu")
        model.save_pretrained(saved, safe_serialization=True)
        model = None
        reloaded = QwenImage21Transformer2DModel.from_pretrained(
            saved, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True
        )
        _install_int8_offload_fix(reloaded)
        for _ in range(2):
            with torch.no_grad():
                actual = reloaded.get_submodule(layer_name)(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            reloaded.to("cpu").to("cuda:0")


if __name__ == "__main__":
    unittest.main()
