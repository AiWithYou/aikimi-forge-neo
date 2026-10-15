"""Canonical aliases are accepted while links and escaped paths stay rejected."""

import ctypes
import os
import stat
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge.flux3_klein import core, inference
from modules_forge.flux3_klein.history import History
from modules_forge.flux3_klein.inference import verify_module_origin
from tools import flux3_klein_worker
from tools.setup_flux3_klein import _target


@contextmanager
def canonical_alias(alias, canonical):
    """Model Windows realpath expanding an existing short-name directory."""
    resolve, is_dir = Path.resolve, Path.is_dir

    def expanded(path, *args, **kwargs):
        absolute = path.absolute()
        if absolute.is_relative_to(alias):
            return resolve(canonical / absolute.relative_to(alias), *args, **kwargs)
        return resolve(path, *args, **kwargs)

    def directory(path):
        return True if path.absolute() == alias else is_dir(path)

    with patch.object(Path, "resolve", expanded), patch.object(Path, "is_dir", directory):
        yield


class ManagedPathTests(unittest.TestCase):
    def test_constructor_inspects_model_and_checkout_ancestors_before_resolving(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, repository = base / "models/Flux-3-Klein", base / "repositories/flux3-klein/ComfyUI"
            root.mkdir(parents=True)
            model = repository / "comfy/ldm/flux3/model.py"
            model.parent.mkdir(parents=True)
            model.touch()
            original = Path.lstat
            for redirect in (root.parent, repository.parent):

                def linked(path, redirect=redirect):
                    return (
                        SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=0xA0000003)
                        if path == redirect
                        else original(path)
                    )

                with (
                    self.subTest(redirect=redirect),
                    patch.object(Path, "lstat", linked),
                    patch.object(core, "comfy_path", return_value=repository),
                    patch.object(sys, "path", list(sys.path)),
                    patch.object(inference, "verify_module_origin"),
                    patch.object(
                        inference.importlib,
                        "import_module",
                        return_value=SimpleNamespace(add_model_folder_path=lambda *args, **kwargs: None),
                    ) as imports,
                ):
                    with self.assertRaises(ValueError):
                        inference.ComfyRuntime(root)
                    imports.assert_not_called()

    def test_worker_preserves_root_boundary_before_constructing_the_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, directory = base / "models/Flux-3-Klein", base / "job"
            root.mkdir(parents=True)
            directory.mkdir()
            (directory / "request.json").write_text("{}", encoding="utf-8")
            original = Path.lstat

            def linked(path):
                return (
                    SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=0xA0000003)
                    if path == root.parent
                    else original(path)
                )

            with (
                patch.object(Path, "lstat", linked),
                patch.object(flux3_klein_worker, "_runner", None),
                patch.object(flux3_klein_worker, "_root", None),
                patch.object(inference, "Runner") as runner,
                self.assertRaises(ValueError),
            ):
                flux3_klein_worker.resident_run({"root": str(root), "job_dir": str(directory)})
            runner.assert_not_called()

    def test_worker_forwards_a_canonical_alias_root_to_the_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, directory = base / "models/Flux-3-Klein", base / "job"
            root.mkdir(parents=True)
            directory.mkdir()
            (directory / "request.json").write_text("{}", encoding="utf-8")
            alias = base / "MODEL~1"
            with (
                canonical_alias(alias, root),
                patch.object(flux3_klein_worker, "_runner", None),
                patch.object(flux3_klein_worker, "_root", None),
                patch.object(inference, "Runner") as runner,
            ):
                runner.return_value.run.return_value = {"ok": True}
                self.assertEqual(
                    flux3_klein_worker.resident_run({"root": str(alias), "job_dir": str(directory)}), {"ok": True}
                )
                runner.assert_called_once_with(root)

    def test_alias_root_accepts_uncreated_leaf_history_and_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            canonical = Path(temporary).resolve()
            alias = canonical.parent / (canonical.name + "~1")
            with canonical_alias(alias, canonical):
                self.assertEqual(_target(alias, "new/weights.bin"), canonical / "new/weights.bin")
                self.assertEqual(History(alias)._root(), canonical)
                verify_module_origin(SimpleNamespace(__file__=None, __path__=[str(alias / "comfy")]), alias)

    @unittest.skipUnless(os.name == "nt", "Windows GetShortPathNameW boundary")
    def test_real_windows_short_name_is_not_a_link(self):
        canonical = Path(os.environ["ProgramFiles"]).resolve()
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_short.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        self.assertGreater(get_short(str(canonical), buffer, len(buffer)), 0)
        alias = Path(buffer.value)
        if alias == canonical:
            self.skipTest("The volume does not expose a distinct 8.3 alias")
        self.assertEqual(alias.resolve(), canonical)
        self.assertFalse(alias.is_symlink() or alias.is_junction())
        self.assertEqual(
            _target(alias, "nonexistent-fixture/weights.bin"), canonical / "nonexistent-fixture/weights.bin"
        )
        self.assertEqual(History(alias)._root(), canonical)
        verify_module_origin(SimpleNamespace(__file__=str(alias / "nonexistent-fixture.py")), alias)

    def test_linked_root_ancestors_and_internal_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "linked-parent/runtime"
            root.mkdir(parents=True)
            original = Path.lstat
            for mode, tag in ((stat.S_IFLNK, 0), (stat.S_IFDIR, 0xA0000003)):

                def linked(path, mode=mode, tag=tag):
                    return SimpleNamespace(st_mode=mode, st_reparse_tag=tag) if path == root.parent else original(path)

                with self.subTest(mode=mode, tag=tag), patch.object(Path, "lstat", linked):
                    with self.assertRaises(ValueError):
                        _target(root, "weights.bin")
                    self.assertEqual(History(root).list(), [])
                    with self.assertRaises(RuntimeError):
                        verify_module_origin(SimpleNamespace(__file__=str(root / "nodes.py")), root)
            module = root / "internal/nodes.py"
            for mode, tag in ((stat.S_IFLNK, 0), (stat.S_IFDIR, 0xA0000003)):

                def internal(path, mode=mode, tag=tag):
                    return (
                        SimpleNamespace(st_mode=mode, st_reparse_tag=tag) if path == module.parent else original(path)
                    )

                with self.subTest(internal_tag=tag), patch.object(Path, "lstat", internal):
                    with self.assertRaises(ValueError):
                        _target(root, "internal/weights.bin")
                    with self.assertRaises(RuntimeError):
                        verify_module_origin(SimpleNamespace(__file__=str(module)), root)

    def test_nonredirecting_reparse_tag_is_allowed_but_access_denied_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = Path.lstat

            def cloud(path):
                return (
                    SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=0x9000001A) if path == root else original(path)
                )

            with patch.object(Path, "lstat", cloud):
                self.assertEqual(_target(root, "new.bin"), root / "new.bin")

            def denied(path):
                if path == root:
                    raise PermissionError("not inspectable")
                return original(path)

            with patch.object(Path, "lstat", denied), self.assertRaises(PermissionError):
                _target(root, "new.bin")

    def test_canonical_sibling_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "runtime"
            root.mkdir()
            outside = base / "runtime-other"
            alias = base / "OUTSID~1"
            with canonical_alias(alias, outside):
                with self.assertRaises(RuntimeError):
                    verify_module_origin(SimpleNamespace(__file__=str(alias / "nodes.py")), root)
            with self.assertRaises(ValueError):
                _target(root, "../runtime-other/weights.bin")

    @unittest.skipUnless(os.name == "nt", "Windows junction boundary")
    def test_real_internal_junction_is_rejected_without_touching_its_target(self):
        import _winapi

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target, link = root / "target", root / "inside"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            _winapi.CreateJunction(str(target), str(link))
            try:
                self.assertTrue(link.is_junction())
                with self.assertRaises(ValueError):
                    _target(root, "inside/new.bin")
                with self.assertRaises(ValueError):
                    _target(link / "pending", "new.bin")
                with self.assertRaises(RuntimeError):
                    verify_module_origin(SimpleNamespace(__file__=str(link / "nodes.py")), root)
                self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
            finally:
                link.rmdir()


if __name__ == "__main__":
    unittest.main()
