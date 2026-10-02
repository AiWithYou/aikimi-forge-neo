"""Legacy VAE overrides must report and restore the identity of loaded weights."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class VaeOverrideIdentityTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "modules/sd_vae.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {
            "_load_vae_dict",
            "get_loaded_vae_name",
            "get_loaded_vae_hash",
            "store_base_vae",
            "delete_base_vae",
            "restore_base_vae",
            "reload_vae_weights",
            "restore_vae_weights",
        }
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.filename = str(Path(temp.name) / "override.vae.pt")
        Path(self.filename).touch()
        self.original_state = {"weight": 1}
        self.current_state = deepcopy(self.original_state)

        def load_state_dict(value):
            self.current_state = deepcopy(value)

        self.model = SimpleNamespace(
            first_stage_model=SimpleNamespace(
                state_dict=lambda: self.current_state,
                load_state_dict=load_state_dict,
            ),
            sd_checkpoint_info=object(),
        )
        self.load_file = Mock(return_value={"weight": 2, "loss.metric": 99})
        self.hash_file = Mock(return_value="a" * 64)
        self.namespace = {
            "os": os,
            "torch": SimpleNamespace(Tensor=object, inference_mode=lambda: lambda fn: fn),
            "deepcopy": deepcopy,
            "memory_management": SimpleNamespace(logger=SimpleNamespace(debug=Mock()), soft_empty_cache=Mock()),
            "utils": SimpleNamespace(load_torch_file=self.load_file),
            "hashes": SimpleNamespace(sha256=self.hash_file),
            "shared": SimpleNamespace(sd_model=self.model),
            "vae_ignore_keys": {"model_ema.decay", "model_ema.num_updates"},
            "base_vae": None,
            "checkpoint_info": None,
            "loaded_vae_file": None,
        }
        exec(  # noqa: S102 - execute only repository VAE helpers with a CPU-only fake model
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.namespace
        )

    def test_success_reports_loaded_override_name_and_hash(self):
        self.assertTrue(self.namespace["reload_vae_weights"](self.filename))
        self.assertEqual(self.current_state, {"weight": 2})
        self.assertEqual(self.namespace["get_loaded_vae_name"](), "override.vae.pt")
        self.assertEqual(self.namespace["get_loaded_vae_hash"](), "a" * 10)
        self.hash_file.assert_called_once_with(self.filename, "vae")

    def test_restoring_base_clears_override_identity_and_recovers_weights(self):
        self.namespace["reload_vae_weights"](self.filename)
        self.namespace["restore_vae_weights"]()
        self.assertEqual(self.current_state, self.original_state)
        self.assertIsNone(self.namespace["get_loaded_vae_name"]())
        self.assertIsNone(self.namespace["get_loaded_vae_hash"]())
        self.assertIsNone(self.namespace["base_vae"])
        self.assertIsNone(self.namespace["checkpoint_info"])

    def test_failed_load_keeps_unloaded_identity_and_can_restore(self):
        self.load_file.side_effect = ValueError("invalid weights")
        with self.assertRaisesRegex(ValueError, "invalid weights"):
            self.namespace["reload_vae_weights"](self.filename)
        self.assertIsNone(self.namespace["get_loaded_vae_name"]())
        self.assertIsNone(self.namespace["get_loaded_vae_hash"]())
        self.namespace["restore_vae_weights"]()
        self.assertEqual(self.current_state, self.original_state)
        self.assertIsNone(self.namespace["base_vae"])


if __name__ == "__main__":
    unittest.main()
