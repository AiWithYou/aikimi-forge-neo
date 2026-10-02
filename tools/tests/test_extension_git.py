"""Exercise extension Git operations against disposable local repositories."""

import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import git

import modules.shared  # noqa: F401 -- initialize Forge before UI imports
from modules import extensions, launch_utils, ui_extensions
from modules.gitpython_hack import Repo


class ExtensionGitTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="forge git ")))
        # File transport is allowed only for these disposable fixtures, including submodules.
        stack.enter_context(patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}))
        self.source_path = self.root / "source repo 日本語"
        self.source = stack.enter_context(git.Repo.init(self.source_path, initial_branch="main"))
        with self.source.config_writer() as config:
            config.set_value("user", "name", "Forge Git Test")
            config.set_value("user", "email", "forge-test@example.invalid")
        self.first = self.commit("initial")
        self.installed = self.root / "extensions 日本語"
        self.installed.mkdir()
        stack.enter_context(patch.object(extensions, "extensions_dir", str(self.installed)))
        stack.enter_context(patch.object(extensions, "extensions", []))
        stack.enter_context(patch.object(ui_extensions.paths, "data_path", str(self.root)))
        stack.enter_context(patch.object(ui_extensions, "check_access"))
        stack.enter_context(patch.object(extensions, "list_extensions"))
        stack.enter_context(patch.object(ui_extensions, "extension_table", return_value="table"))
        import launch

        self.installer = stack.enter_context(patch.object(launch, "run_extension_installer"))

    def commit(self, content):
        (self.source_path / "payload.txt").write_text(content, encoding="utf-8")
        self.source.index.add(["payload.txt"])
        return self.source.index.commit(content)

    def install(self, name="extension repo", branch=None):
        result = ui_extensions.install_extension_from_url(name, self.source_path.as_uri(), branch)
        target = self.installed / name
        self.assertEqual(result[0], "table")
        self.installer.assert_called_once_with(str(target))
        self.assertFalse((self.root / "tmp" / name).exists())
        return target

    def extension(self, target):
        metadata = extensions.ExtensionMetadata(target, target.name)
        extension = extensions.Extension(target.name, str(target), metadata=metadata)
        extension.do_read_info_from_repo()
        self.assertEqual(extension.branch, "main")
        self.assertEqual(extension.remote, self.source_path.as_uri())
        self.assertEqual(extension.commit_hash, self.first.hexsha)
        self.assertEqual(extension.version, self.first.hexsha[:8])
        return extension

    def test_default_branch_clone_reads_metadata_and_reports_latest(self):
        target = self.install()
        self.assertEqual((target / "payload.txt").read_text(encoding="utf-8"), "initial")
        extension = self.extension(target)
        extension.check_updates()
        self.assertFalse(extension.can_update)
        self.assertEqual(extension.status, "latest")

    def test_explicit_branch_clone_selects_requested_commit(self):
        self.source.git.checkout("-b", "feature")
        selected = self.commit("feature branch")
        self.source.git.checkout("main")
        target = self.install(branch="feature")
        with Repo(target) as repo:
            self.assertEqual(repo.active_branch.name, "feature")
            self.assertEqual(repo.head.commit.hexsha, selected.hexsha)

    def test_submodule_checkout_survives_move_to_extension_directory(self):
        sub_path = self.root / "submodule source"
        with git.Repo.init(sub_path, initial_branch="main") as sub:
            (sub_path / "module.txt").write_text("submodule payload", encoding="utf-8")
            sub.index.add(["module.txt"])
            sub.index.commit(
                "submodule",
                author=git.Actor("Test", "test@example.invalid"),
                committer=git.Actor("Test", "test@example.invalid"),
            )
        self.source.create_submodule("nested", "nested/module", url=sub_path.as_uri(), branch="main")
        self.source.index.commit("add submodule")
        target = self.install()
        self.assertEqual((target / "nested/module/module.txt").read_text(encoding="utf-8"), "submodule payload")
        with Repo(target) as repo, repo.submodules[0].module() as sub:
            self.assertEqual(sub.head.commit.message.strip(), "submodule")

    def test_check_updates_and_fetch_reset_advance_to_remote_commit(self):
        target = self.install()
        extension = self.extension(target)
        second = self.commit("updated")
        extension.check_updates()
        self.assertTrue(extension.can_update)
        self.assertEqual(extension.status, "new commits")
        with Repo(target) as repo:
            self.assertEqual(repo.head.commit.hexsha, self.first.hexsha)
        (target / "payload.txt").write_text("local edit", encoding="utf-8")
        extension.fetch_and_reset_hard()
        self.assertFalse(extension.have_info_from_repo)
        extension.do_read_info_from_repo()
        self.assertEqual(extension.commit_hash, second.hexsha)
        self.assertEqual((target / "payload.txt").read_text(encoding="utf-8"), "updated")
        extension.check_updates()
        self.assertEqual(extension.status, "latest")

    def test_recursive_pull_preserves_unrelated_local_edits(self):
        target = self.install()
        (self.source_path / "local.txt").write_text("original", encoding="utf-8")
        self.source.index.add(["local.txt"])
        self.source.index.commit("add independent file")
        with redirect_stdout(StringIO()):
            launch_utils.git_pull_recursive(str(self.installed))
        (target / "local.txt").write_text("local edit", encoding="utf-8")
        second = self.commit("pull update")
        with redirect_stdout(StringIO()):
            launch_utils.git_pull_recursive(str(self.installed))
        with Repo(target) as repo:
            self.assertEqual(repo.head.commit.hexsha, second.hexsha)
        self.assertEqual((target / "local.txt").read_text(encoding="utf-8"), "local edit")

    def test_forge_object_reader_works_without_persistent_git_processes(self):
        target = self.install()
        with (
            Repo(target) as repo,
            patch.object(repo.git, "_get_persistent_cmd", side_effect=AssertionError("persistent process")),
        ):
            commit = repo.head.commit
            self.assertEqual(commit.hexsha, self.first.hexsha)
            self.assertEqual(commit.message.strip(), "initial")
            sha, kind, size = repo.git.get_object_header(commit.hexsha)
            self.assertEqual((sha, kind), (self.first.hexsha.encode("ascii"), b"commit"))
            stream_sha, stream_kind, stream_size, stream = repo.git.stream_object_data(commit.hexsha)
            self.assertEqual((stream_sha, stream_kind, stream_size), (sha, kind, size))
            self.assertEqual(len(stream.read()), size)


if __name__ == "__main__":
    unittest.main()
