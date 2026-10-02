"""Extension metadata follows real Git worktrees and ordinary repositories."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import git

from modules.file_identity import cache_file_identity
from modules.gitpython_hack import Repo
from modules.metadata_cache import MetadataCache
from tools.tests.test_runtime_efficiency import load_definitions


class ExtensionGitCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="extension-git-日本語-")))
        self.source_path = self.root / "source"
        self.source = self.enterContext(git.Repo.init(self.source_path, initial_branch="main"))
        self.actor = git.Actor("CPU Fixture", "cpu-fixture@example.invalid")
        (self.source_path / "payload.txt").write_text("initial", encoding="utf-8")
        self.source.index.add(["payload.txt"])
        self.first = self.source.index.commit("initial", author=self.actor, committer=self.actor)
        self.source.create_remote("origin", self.source_path.as_uri())
        self.worktree = self.root / "extension worktree"
        self.source.git.worktree("add", "-b", "worktree-one", str(self.worktree))
        self.repo = self.enterContext(Repo(self.worktree))
        self.metadata = MetadataCache(self.root / "metadata")
        self.addCleanup(lambda: self.metadata.close())
        cached = load_definitions(
            "modules/cache.py",
            {"cached_data_for_file"},
            {"os": os, "cache": lambda _: self.metadata, "dump_cache": lambda: None},
        )["cached_data_for_file"]
        self.cache_reader = Mock(wraps=cached)
        self.errors = Mock()
        self.Extension = load_definitions(
            "modules/extensions.py",
            {"Extension"},
            {
                "os": os,
                "threading": threading,
                "Repo": Repo,
                "cache": SimpleNamespace(cached_data_for_file=self.cache_reader),
                "errors": self.errors,
            },
        )["Extension"]
        self.pointer = self.worktree / ".git"
        self.assertTrue(self.pointer.is_file())
        self.identity = cache_file_identity(self.pointer)
        self.initial = self.extension()
        self.initial.read_info_from_repo()
        self.assertEqual(self.initial.commit_hash, self.first.hexsha)
        self.assertEqual(self.initial.branch, "worktree-one")

    def extension(self, path=None, name="fixture-extension", **kwargs):
        return self.Extension(name, str(path or self.worktree), metadata=SimpleNamespace(canonical_name=name), **kwargs)

    def reopen(self):
        self.metadata.close()
        self.metadata = MetadataCache(self.root / "metadata")

    def test_on_branch_commit_refreshes_after_persistent_cache_reopen(self):
        (self.worktree / "payload.txt").write_text("second", encoding="utf-8")
        self.repo.index.add(["payload.txt"])
        second = self.repo.index.commit("second", author=self.actor, committer=self.actor)
        self.assertNotEqual(second.hexsha, self.first.hexsha)
        self.assertEqual(cache_file_identity(self.pointer), self.identity)
        self.reopen()
        actual = self.extension()
        actual.read_info_from_repo()
        self.assertEqual(actual.commit_hash, second.hexsha)
        self.assertEqual(actual.version, second.hexsha[:8])
        self.assertEqual(actual.branch, "worktree-one")
        self.errors.report.assert_not_called()

    def test_branch_switch_refreshes_after_persistent_cache_reopen(self):
        self.repo.git.checkout("-b", "worktree-two")
        self.assertEqual(cache_file_identity(self.pointer), self.identity)
        self.reopen()
        actual = self.extension()
        actual.read_info_from_repo()
        self.assertEqual(actual.branch, "worktree-two")
        self.assertEqual(actual.commit_hash, self.repo.head.commit.hexsha)
        self.errors.report.assert_not_called()

    def test_detached_worktree_keeps_commit_identity_without_a_branch(self):
        self.repo.git.checkout("--detach", self.first.hexsha)
        self.assertTrue(self.repo.head.is_detached)
        actual = self.extension(name="detached-extension")
        actual.read_info_from_repo()
        self.assertEqual(actual.commit_hash, self.first.hexsha)
        self.assertEqual(actual.version, self.first.hexsha[:8])
        self.assertIsNone(actual.branch)
        self.errors.report.assert_not_called()

    def test_ordinary_repository_retains_persistent_cache_reuse(self):
        actual = self.extension(self.source_path, "ordinary-extension")
        reader = Mock(wraps=actual.do_read_info_from_repo)
        actual.do_read_info_from_repo = reader
        actual.read_info_from_repo()
        reader.assert_called_once()
        self.reopen()
        cached = self.extension(self.source_path, "ordinary-extension")
        cached.do_read_info_from_repo = Mock(side_effect=AssertionError("warm repository cache was ignored"))
        cached.read_info_from_repo()
        self.assertEqual(cached.commit_hash, self.first.hexsha)
        self.assertEqual(cached.branch, "main")
        cached.do_read_info_from_repo.assert_not_called()

    def test_builtin_and_initialized_instances_do_not_read_again(self):
        self.initial.do_read_info_from_repo = Mock(side_effect=AssertionError("initialized instance was reread"))
        builtin = self.extension(is_builtin=True)
        builtin.do_read_info_from_repo = Mock(side_effect=AssertionError("builtin extension was read"))
        self.cache_reader.reset_mock()
        self.initial.read_info_from_repo()
        builtin.read_info_from_repo()
        self.cache_reader.assert_not_called()
        self.initial.do_read_info_from_repo.assert_not_called()
        builtin.do_read_info_from_repo.assert_not_called()

    def test_missing_repository_keeps_unknown_status(self):
        actual = self.extension(self.root / "missing", "missing-extension")
        actual.read_info_from_repo()
        self.assertEqual(actual.status, "unknown")
        self.assertEqual(actual.commit_hash, "")
        self.errors.report.assert_not_called()

    def test_concurrent_first_read_uses_the_finished_instance(self):
        class CoordinatedLock:
            def __init__(self):
                self.arrivals = threading.Barrier(2)
                self.lock = threading.Lock()

            def __enter__(self):
                self.arrivals.wait(timeout=5)
                self.lock.acquire()

            def __exit__(self, *_args):
                self.lock.release()

        actual = self.extension(name="concurrent-extension")
        actual.lock = CoordinatedLock()
        actual.do_read_info_from_repo = Mock(wraps=actual.do_read_info_from_repo)
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: actual.read_info_from_repo(), range(2)))
        self.assertEqual(actual.commit_hash, self.first.hexsha)
        self.assertEqual(actual.branch, "worktree-one")
        actual.do_read_info_from_repo.assert_called_once()


if __name__ == "__main__":
    unittest.main()
