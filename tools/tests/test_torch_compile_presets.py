from __future__ import annotations

import ast
import logging
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_script():
    path = Path(__file__).resolve().parents[2] / "extensions-builtin/sd_forge_compile/scripts/compile.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {
        "torch": torch,
        "wraps": wraps,
        "scripts": SimpleNamespace(Script=object),
        "cmd_args": SimpleNamespace(cuda_malloc=False),
        "logger": logging.getLogger("compile-test"),
        "get_attr": getattr,
        "set_attr_raw": setattr,
        "_COMPILE_CONFIG_KEY": "_torch_compile_config",
        "_ORIG_APPLY_KEY": "_orig_apply_model",
        "_COMPILE_WRAPPER_KEY": "_torch_compile_wrapper",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- execute repository code with isolated host dependencies
    return namespace["TorchCompileForForge"]


class TorchCompilePresetTests(unittest.TestCase):
    def setUp(self):
        self.script = object.__new__(load_script())
        self.model = SimpleNamespace(diffusion_model=torch.nn.Identity())
        self.original = lambda: self.model.diffusion_model
        self.model.apply_model = self.original
        self.processing = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(
            unet=SimpleNamespace(model=self.model)
        )))

    def test_all_presets_reach_real_torch_compile_without_conflicting_arguments(self):
        for preset in ("guard_filter_fn", "dynamic", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"):
            with self.subTest(preset=preset):
                self.script.process_batch(self.processing, preset)
                original_model = self.model.diffusion_model
                with patch.object(torch, "compile", wraps=torch.compile) as compiler:
                    compiled = self.model.apply_model()
                    self.assertEqual(compiler.call_count, 1)
                    self.assertIs(self.model.apply_model(), compiled)
                    self.assertEqual(compiler.call_count, 1)
                self.assertIs(self.model.diffusion_model, original_model)
                self.script.process_batch(self.processing, "Disable")
                self.assertIs(self.model.apply_model, self.original)
                self.assertFalse(hasattr(self.model, "_forge_compiled_model"))

    def test_unknown_preset_preserves_existing_wrapper_and_cached_model(self):
        self.script.process_batch(self.processing, "dynamic")
        wrapper = self.model.apply_model
        cached = object()
        self.model._forge_compiled_model = cached
        with self.assertRaisesRegex(ValueError, "Unknown.*preset"):
            self.script.process_batch(self.processing, "invalid")
        self.assertIs(self.model.apply_model, wrapper)
        self.assertIs(self.model._forge_compiled_model, cached)
        self.assertEqual(self.model._torch_compile_config, "dynamic")

    def test_forward_failure_restores_original_model(self):
        def fail():
            raise RuntimeError("forward failed")

        self.model.apply_model = fail
        original_model = self.model.diffusion_model
        self.script.process_batch(self.processing, "dynamic")
        with patch.object(torch, "compile", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                self.model.apply_model()
        self.assertIs(self.model.diffusion_model, original_model)
        self.script.process_batch(self.processing, "Disable")
        self.assertIs(self.model.apply_model, fail)

    def test_automatic_and_same_preset_keep_compiled_model(self):
        self.script.process_batch(self.processing, "dynamic")
        wrapper = self.model.apply_model
        cached = object()
        self.model._forge_compiled_model = cached
        for preset in ("Automatic", "dynamic"):
            self.script.process_batch(self.processing, preset)
            self.assertIs(self.model.apply_model, wrapper)
            self.assertIs(self.model._forge_compiled_model, cached)

    def test_compile_failure_can_retry_without_replacing_original_model(self):
        original_model = self.model.diffusion_model
        self.script.process_batch(self.processing, "dynamic")
        with patch.object(torch, "compile", side_effect=RuntimeError("compile failed")):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                self.model.apply_model()
        self.assertIs(self.model.diffusion_model, original_model)
        self.assertFalse(hasattr(self.model, "_forge_compiled_model"))
        compiled = object()
        with patch.object(torch, "compile", return_value=compiled):
            self.assertIs(self.model.apply_model(), compiled)
        self.assertIs(self.model.diffusion_model, original_model)

    def test_switching_preset_discards_cache_without_nesting_wrappers(self):
        self.script.process_batch(self.processing, "dynamic")
        old_wrapper = self.model.apply_model
        self.model._forge_compiled_model = object()
        self.script.process_batch(self.processing, "reduce-overhead")
        self.assertIsNot(self.model.apply_model, old_wrapper)
        self.assertIs(self.model._orig_apply_model, self.original)
        self.assertFalse(hasattr(self.model, "_forge_compiled_model"))
        self.assertEqual(self.model._torch_compile_config, "reduce-overhead")
        self.script.process_batch(self.processing, "Disable")
        self.assertIs(self.model.apply_model, self.original)
        for attribute in ("_forge_compiled_model", "_orig_apply_model", "_torch_compile_config"):
            self.assertFalse(hasattr(self.model, attribute))


if __name__ == "__main__":
    unittest.main()
