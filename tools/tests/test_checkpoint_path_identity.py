"""Checkpoint names must follow directory boundaries and platform path rules."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class CheckpointPathIdentityTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "modules/sd_models.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if getattr(node, "name", None) in {"CheckpointInfo", "replace_key"}]
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.cmd_opts = SimpleNamespace(ckpt_dirs=[])
        self.namespace = {
            "os": os,
            "cmd_opts": self.cmd_opts,
            "model_path": str(self.models),
            "model_hash": lambda _filename: "fixture-hash",
            "hashes": SimpleNamespace(sha256_from_cache=lambda *_args: None),
            "checkpoints_list": {},
            "checkpoint_aliases": {},
        }
        exec(  # noqa: S102 - execute only the extracted repository checkpoint class and helper
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.namespace
        )
        self.checkpoint_info = self.namespace["CheckpointInfo"]

    def make_file(self, directory, name="model.ckpt"):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_sibling_with_common_prefix_keeps_external_path(self):
        filename = self.make_file(self.root / "models-extra")
        info = self.checkpoint_info(str(filename))
        expected = os.path.abspath(filename).strip("/").strip("\\")
        self.assertEqual(info.name, expected)
        info.register()
        self.assertIs(self.namespace["checkpoint_aliases"][expected], info)

    def test_nested_checkpoint_keeps_relative_name(self):
        filename = self.make_file(self.models / "nested")
        info = self.checkpoint_info(str(filename))
        self.assertEqual(info.name, os.path.join("nested", "model.ckpt"))
        self.assertEqual(info.name_for_extra, "model")

    def test_relative_custom_root_is_resolved_before_naming(self):
        custom = self.root / "custom"
        filename = self.make_file(custom / "nested")
        self.cmd_opts.ckpt_dirs = [os.path.relpath(custom)]
        info = self.checkpoint_info(str(filename))
        self.assertEqual(info.name, os.path.join("nested", "model.ckpt"))

    @unittest.skipUnless(os.name == "nt", "Windows checkpoint roots are case-insensitive")
    def test_windows_root_case_does_not_change_checkpoint_name(self):
        filename = self.make_file(self.models / "nested")
        self.namespace["model_path"] = str(self.models).upper()
        info = self.checkpoint_info(str(filename))
        self.assertEqual(info.name, os.path.join("nested", "model.ckpt"))

    @unittest.skipUnless(os.name == "nt", "Windows relpath can reject another drive")
    def test_other_drive_root_does_not_block_matching_default_root(self):
        filename = self.make_file(self.models / "nested")
        other_drive = "Z:" if filename.drive.upper() != "Z:" else "Y:"
        self.cmd_opts.ckpt_dirs = [other_drive + "\\checkpoints"]
        info = self.checkpoint_info(str(filename))
        self.assertEqual(info.name, os.path.join("nested", "model.ckpt"))


if __name__ == "__main__":
    unittest.main()
