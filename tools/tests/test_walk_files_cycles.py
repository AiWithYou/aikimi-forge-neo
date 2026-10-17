"""Model discovery follows directory links without following cycles."""

import ast
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class WalkFilesCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "modules/util.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"natural_sort_key", "walk_files"}]
        namespace = dict(os=os, re=re, shared=SimpleNamespace(opts=SimpleNamespace(list_hidden_files=False)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        cls.walk = staticmethod(namespace["walk_files"])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.models = self.root / "models"
        self.models.mkdir()
        real_walk = os.walk

        def bounded_walk(*args, **kwargs):
            for count, entry in enumerate(real_walk(*args, **kwargs)):
                if count >= 32:
                    self.fail("directory cycles were not pruned")
                yield entry

        self.enterContext(patch("os.walk", side_effect=bounded_walk))

    def link(self, path, target):
        try:
            path.symlink_to(target, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks unavailable: {error}")

    def test_self_and_parent_cycles_do_not_repeat_models(self):
        (self.models / "base.safetensors").touch()
        child = self.models / "child"
        child.mkdir()
        (child / "child.safetensors").touch()
        for link in (self.models / "self", child / "parent"):
            with self.subTest(link=link.name):
                self.link(link, self.models)
                try:
                    found = list(self.walk(str(self.models), [".safetensors"]))
                    self.assertEqual(found, [str(self.models / "base.safetensors"), str(child / "child.safetensors")])
                finally:
                    link.unlink()

    def test_external_link_and_independent_aliases_remain_available(self):
        external = self.root / "external"
        external.mkdir()
        (external / "model2.safetensors").touch()
        (external / "model10.safetensors").touch()
        (external / "notes.txt").touch()
        for alias in ("one", "two"):
            self.link(self.models / alias, external)
        found = list(self.walk(str(self.models), [".safetensors"]))
        expected = [str(self.models / alias / name) for alias in ("one", "two") for name in ("model2.safetensors", "model10.safetensors")]
        self.assertEqual(found, expected)

    def test_multiple_cycles_are_pruned_before_descent(self):
        (self.models / "model.safetensors").touch()
        for name in ("cycle1", "cycle2"):
            self.link(self.models / name, self.models)
        self.assertEqual(list(self.walk(str(self.models))), [str(self.models / "model.safetensors")])

    def test_unicode_numeric_names_do_not_break_sorting(self):
        names = ["model１０.safetensors", "model２.safetensors", "model①.safetensors", "².safetensors", "１２.safetensors", "12.safetensors", "model10.safetensors", "model2.safetensors"]
        for name in names:
            (self.models / name).touch()
        result = [Path(path).name for path in self.walk(str(self.models))]
        self.assertCountEqual(result, names)
        self.assertLess(result.index("model２.safetensors"), result.index("model１０.safetensors"))
        self.assertLess(result.index("model2.safetensors"), result.index("model10.safetensors"))

    def test_filesystem_without_inode_still_prunes_cycles(self):
        (self.models / "model.safetensors").touch()
        self.link(self.models / "cycle", self.models)
        real_stat = os.stat

        def no_inode(*args, **kwargs):
            result = real_stat(*args, **kwargs)
            return SimpleNamespace(st_dev=result.st_dev, st_ino=0)

        with patch("os.stat", side_effect=no_inode):
            found = list(self.walk(str(self.models)))
        self.assertEqual(found, [str(self.models / "model.safetensors")])

    def test_directory_removed_after_enumeration_does_not_abort_other_files(self):
        (self.models / "model.safetensors").touch()
        missing = self.models / "vanished"
        missing.mkdir()
        (missing / "stale.safetensors").touch()
        real_stat = os.stat

        def disappeared(path, *args, **kwargs):
            if os.fspath(path) == str(missing):
                raise FileNotFoundError(path)
            return real_stat(path, *args, **kwargs)

        with patch("os.stat", side_effect=disappeared):
            found = list(self.walk(str(self.models)))
        self.assertEqual(found, [str(self.models / "model.safetensors")])


if __name__ == "__main__":
    unittest.main()
