"""Forge must validate actual checkpoint files before reusing loaded weights."""

from __future__ import annotations

import ast
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from modules_forge import local_assets
from tools.tests.test_additional_module_identity import load_main_entry, stub_modules

ROOT = Path(__file__).resolve().parents[2]


class ForgeCheckpointReloadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.enterContext(patch.object(local_assets, "LIBRARY", self.root / "library.json"))
        self.enterContext(patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json"))
        self.ui, self.opts, self.shared = load_main_entry()
        self.source = self.root / "checkpoint.ckpt"
        self.source.write_bytes(bytes(3 * 1024 * 1024))
        self.opts.sd_model_checkpoint = str(self.source)
        self.opts.forge_unet_storage_dtype = None
        self.opts.forge_checkpoint_sd = str(self.source)
        self.data = SimpleNamespace(
            forge_loading_parameters={},
            forge_hash="",
            sd_model=object(),
            forge_loading=False,
            last_load_seconds=None,
            last_load_error=None,
        )
        tree = ast.parse((ROOT / "modules/sd_models.py").read_text(encoding="utf-8"))
        names = {
            "CheckpointInfo",
            "replace_key",
            "get_closet_checkpoint_match",
            "model_hash",
            "select_checkpoint",
            "loading_file_identity",
            "forge_model_reload",
        }
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        self.namespace = {
            "os": os,
            "re_strip_checksum": re.compile(r"\s*\[[^]]+]\s*$"),
            "cmd_opts": SimpleNamespace(ckpt_dirs=[]),
            "model_path": str(self.root),
            "hashes": SimpleNamespace(sha256_from_cache=lambda *_args: None),
            "checkpoints_list": {},
            "checkpoint_aliases": {},
            "shared": self.shared,
            "model_data": self.data,
            "torch": SimpleNamespace(inference_mode=lambda: lambda fn: fn),
            "_forge_model_reload": self.fake_load,
        }
        exec(  # noqa: S102 - exercise repository loader entry with a CPU-only model loader
            compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "modules/sd_models.py"), "exec"),
            self.namespace,
        )
        self.calls = []
        self.info = self.namespace["CheckpointInfo"](str(self.source))
        self.info.register()
        self.data.forge_loading_parameters = {
            "checkpoint_info": self.info,
            "additional_modules": [],
            "unet_storage_dtype": None,
        }
        self.sd_models = ModuleType("modules.sd_models")
        for name, value in self.namespace.items():
            setattr(self.sd_models, name, value)
        self.ui.sd_models = self.sd_models

    def fake_load(self, key):
        self.calls.append(key)
        self.data.sd_model = SimpleNamespace(load_number=len(self.calls))
        self.data.forge_hash = key
        return self.data.sd_model, True

    def rewrite_outside_legacy_hash(self, path):
        before = path.stat()
        with path.open("r+b") as stream:
            stream.seek(2 * 1024 * 1024)
            stream.write(b"updated")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(path.stat().st_size, before.st_size)
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)

    def reload(self):
        modules = ModuleType("modules")
        modules.sd_models = self.sd_models
        with (
            stub_modules(
                {"modules": modules, "modules.sd_models": self.sd_models, "modules_forge.main_entry": self.ui}
            ),
            patch.object(self.ui, "print_json"),
        ):
            return self.namespace["forge_model_reload"]()

    def configure_encoder(self):
        component = self.root / "encoder.ckpt"
        component.write_bytes(b"encoder")
        self.opts.data["forge_additional_modules"] = [str(component)]
        self.opts.forge_unet_storage_dtype = "float8-e4m3fn"
        dtype = self.ui.forge_unet_storage_dtype_options[self.opts.forge_unet_storage_dtype][0]
        self.data.forge_loading_parameters.update(additional_modules=[str(component)], unet_storage_dtype=dtype)
        return component, dtype

    def refresh(self):
        tree = ast.parse((ROOT / "modules_forge/main_entry.py").read_text(encoding="utf-8"))
        manager = next(node for node in tree.body if getattr(node, "name", None) == "make_checkpoint_manager_ui")
        callback = next(node for node in manager.body if getattr(node, "name", None) == "refresh_model_list")

        def rebuild():
            self.namespace["checkpoint_aliases"].clear()
            self.namespace["checkpoints_list"].clear()
            if self.source.is_file():
                self.namespace["CheckpointInfo"](str(self.source)).register()
            return [], []

        scope = {
            "refresh_models": rebuild,
            "shared": self.shared,
            "gr": self.ui.gr,
            "module_values_to_ui_selectors": self.ui.module_values_to_ui_selectors,
            "preset_checkpoint_selection": lambda _preset: {"choices": [], "value": str(self.source)},
            "refresh_model_loading_parameters": self.ui.refresh_model_loading_parameters,
        }
        exec(  # noqa: S102 - invoke the real refresh callback without constructing GPU components
            compile(ast.Module(body=[callback], type_ignores=[]), str(ROOT / "modules_forge/main_entry.py"), "exec"),
            scope,
        )
        modules = ModuleType("modules")
        modules.sd_models = self.sd_models
        with (
            stub_modules({"modules": modules, "modules.sd_models": self.sd_models}),
            patch.object(self.ui, "print_json"),
        ):
            return scope["refresh_model_list"]()

    def test_unchanged_file_reuses_the_loaded_model(self):
        loaded, first = self.reload()
        reused, second = self.reload()
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertIs(reused, loaded)
        self.assertEqual(len(self.calls), 1)

    def test_same_size_overwrite_outside_partial_hash_reloads_without_ui_refresh(self):
        self.reload()
        old_hash = self.namespace["model_hash"](str(self.source))
        self.rewrite_outside_legacy_hash(self.source)
        self.assertEqual(self.namespace["model_hash"](str(self.source)), old_hash)
        loaded, changed = self.reload()
        self.assertTrue(changed)
        self.assertEqual(loaded.load_number, 2)
        self.assertNotEqual(self.calls[0], self.calls[1])

    def test_additional_component_overwrite_also_reloads(self):
        component = self.root / "encoder.ckpt"
        component.write_bytes(bytes(3 * 1024 * 1024))
        self.data.forge_loading_parameters["additional_modules"] = [str(component)]
        self.reload()
        self.rewrite_outside_legacy_hash(component)
        self.assertTrue(self.reload()[1])
        self.assertEqual(len(self.calls), 2)

    def test_missing_encoder_fails_on_consecutive_attempts(self):
        component, _ = self.configure_encoder()
        self.reload()
        component.unlink()
        for _ in range(2):
            with self.assertRaises(FileNotFoundError):
                self.reload()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.opts.forge_additional_modules, [str(component)])
        self.assertEqual(self.opts.forge_unet_storage_dtype, "float8-e4m3fn")

    def test_restored_encoder_keeps_selected_components_and_dtype(self):
        component, dtype = self.configure_encoder()
        self.reload()
        component.unlink()
        with self.assertRaises(FileNotFoundError):
            self.reload()
        component.write_bytes(b"restored encoder")
        self.assertTrue(self.reload()[1])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.data.forge_loading_parameters["additional_modules"], [str(component)])
        self.assertIs(self.data.forge_loading_parameters["unet_storage_dtype"], dtype)

    def test_deleted_selected_file_never_reuses_loaded_weights_before_refresh(self):
        self.reload()
        self.source.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "selected checkpoint"):
            self.reload()
        self.assertEqual(self.data.forge_loading_parameters, {})
        self.assertEqual(self.data.forge_hash, "")
        self.assertIn(str(self.source), self.data.last_load_error)
        self.assertEqual(len(self.calls), 1)

    def test_deleted_selected_file_refresh_clears_state_and_load_is_explicit_error(self):
        self.reload()
        self.source.unlink()
        self.refresh()
        self.assertEqual(self.data.forge_loading_parameters, {})
        self.assertEqual(self.data.forge_hash, "")
        with self.assertRaisesRegex(FileNotFoundError, "selected checkpoint"):
            self.reload()
        self.assertEqual(len(self.calls), 1)

    def test_refresh_replaces_stale_checkpoint_record(self):
        self.refresh()
        refreshed = self.data.forge_loading_parameters["checkpoint_info"]
        self.assertIsNot(refreshed, self.info)
        self.assertEqual(refreshed.filename, str(self.source))

    def test_selecting_same_canonical_path_refreshes_loading_parameters(self):
        with patch.object(self.ui, "refresh_model_loading_parameters") as refresh:
            changed = self.ui.checkpoint_change(str(self.source), "sd", save=False)
        self.assertFalse(changed)
        refresh.assert_called_once_with(refresh=True)


if __name__ == "__main__":
    unittest.main()
