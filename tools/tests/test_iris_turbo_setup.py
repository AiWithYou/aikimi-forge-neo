"""Pinned Iris Turbo downloads verify staging before publishing local weights."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from tools import prepare_iris_turbo


class IrisTurboSetupTests(unittest.TestCase):
    def specification(self, adapter, payload):
        return replace(
            prepare_iris_turbo.ADAPTERS[adapter],
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def hub(self, download):
        module = ModuleType("huggingface_hub")
        module.hf_hub_download = Mock(side_effect=download)
        return module

    def test_both_adapters_are_pinned_to_published_files_and_hashes(self):
        self.assertEqual(prepare_iris_turbo.REPOSITORY, "pentacoxian-dev/iris-3b-turbo-lora")
        self.assertEqual(prepare_iris_turbo.REVISION, "fe16b3496eb440585993c92d95641c08c6b08263")
        for adapter, digest in (
            ("r2-1000", "de8e8f24075d7de12251782f36ac99d2aafc2c58dec7c85297b017a6527028a1"),
            ("r1-1500", "1e937c0091f5d9f772587d577cc6b7af240b832412a00c9722a2daa3d5a5f6b7"),
        ):
            with self.subTest(adapter=adapter):
                plan = prepare_iris_turbo.download_plan(adapter)
                self.assertEqual(plan["file"], adapter + "/student.safetensors")
                self.assertEqual(plan["size"], 459487256)
                self.assertEqual(plan["sha256"], digest)
                destination = Path(plan["output_path"])
                self.assertIn(adapter, destination.name)
                self.assertEqual(destination.suffix, ".safetensors")
                self.assertEqual(destination.parent, prepare_iris_turbo.ROOT / "models/Iris-3B/loras/pentacoxian-turbo")

    def test_dry_run_defaults_never_import_dependencies_write_or_download(self):
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
            patch.object(Path, "mkdir", side_effect=AssertionError("dry run wrote a directory")),
            patch.object(prepare_iris_turbo, "install") as install,
            redirect_stdout(output),
        ):
            self.assertEqual(prepare_iris_turbo.main(["--dry-run"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["file"], "r2-1000/student.safetensors")
        self.assertEqual(plan["revision"], "fe16b3496eb440585993c92d95641c08c6b08263")
        install.assert_not_called()

    def test_dry_run_selects_alternate_and_output_directory_without_creating_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "unused"
            output = io.StringIO()
            with patch.dict(sys.modules, {"huggingface_hub": None}), redirect_stdout(output):
                prepare_iris_turbo.main(["--adapter", "r1-1500", "--output-dir", str(directory), "--dry-run"])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["file"], "r1-1500/student.safetensors")
            self.assertEqual(Path(plan["output_path"]).parent, directory)
            self.assertFalse(directory.exists())

    def test_download_checks_staging_then_atomically_replaces_bad_existing_weights(self):
        for adapter in ("r2-1000", "r1-1500"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                payload = b"verified Turbo weights"
                specification = self.specification(adapter, payload)
                target = directory / specification.local_filename
                target.write_bytes(b"bad cached weights")
                stages = []

                def download(
                    repo_id,
                    *,
                    filename,
                    revision,
                    local_dir,
                    adapter=adapter,
                    stages=stages,
                    directory=directory,
                    target=target,
                    payload=payload,
                ):
                    self.assertEqual(repo_id, "pentacoxian-dev/iris-3b-turbo-lora")
                    self.assertEqual(revision, "fe16b3496eb440585993c92d95641c08c6b08263")
                    self.assertEqual(filename, adapter + "/student.safetensors")
                    stage = Path(local_dir)
                    stages.append(stage)
                    self.assertEqual(stage.parent, directory)
                    self.assertNotEqual(stage, directory)
                    self.assertEqual(target.read_bytes(), b"bad cached weights")
                    path = stage / filename
                    path.parent.mkdir(parents=True)
                    path.write_bytes(payload)
                    return str(path)

                hub = self.hub(download)
                with (
                    patch.object(prepare_iris_turbo, "ADAPTERS", {adapter: specification}),
                    patch.dict(sys.modules, {"huggingface_hub": hub}),
                    patch.object(prepare_iris_turbo.os, "replace", wraps=prepare_iris_turbo.os.replace) as replace_file,
                ):
                    receipt = prepare_iris_turbo.install(adapter, directory)
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(receipt["size"], len(payload))
                self.assertFalse(receipt["cached"])
                hub.hf_hub_download.assert_called_once()
                replace_file.assert_called_once()
                self.assertEqual(Path(replace_file.call_args.args[1]), target)
                self.assertFalse(stages[0].exists())
                self.assertEqual(list(directory.iterdir()), [target])

    def test_correct_existing_weights_work_without_hub_and_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = b"verified Turbo weights"
            specification = self.specification("r2-1000", payload)
            target = directory / specification.local_filename
            target.write_bytes(payload)
            with (
                patch.object(prepare_iris_turbo, "ADAPTERS", {"r2-1000": specification}),
                patch.dict(sys.modules, {"huggingface_hub": None}),
                patch.object(prepare_iris_turbo.os, "replace") as replace_file,
            ):
                receipt = prepare_iris_turbo.install("r2-1000", directory)
            self.assertTrue(receipt["cached"])
            self.assertEqual(Path(receipt["output_path"]), target)
            replace_file.assert_not_called()

    def test_wrong_size_or_hash_never_replaces_existing_file_and_cleans_staging(self):
        for payload, reason in ((b"partial", "size"), (b"bad weights!", "SHA-256")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                specification = self.specification("r2-1000", b"good weights")
                target = directory / specification.local_filename
                target.write_bytes(b"previous file")

                def download(repo_id, *, filename, revision, local_dir, payload=payload):
                    path = Path(local_dir) / filename
                    path.parent.mkdir(parents=True)
                    path.write_bytes(payload)
                    return str(path)

                with (
                    patch.object(prepare_iris_turbo, "ADAPTERS", {"r2-1000": specification}),
                    patch.dict(sys.modules, {"huggingface_hub": self.hub(download)}),
                    patch.object(prepare_iris_turbo.os, "replace") as replace_file,
                    self.assertRaisesRegex(ValueError, reason),
                ):
                    prepare_iris_turbo.install("r2-1000", directory)
                self.assertEqual(target.read_bytes(), b"previous file")
                self.assertEqual(list(directory.iterdir()), [target])
                replace_file.assert_not_called()

    def test_failed_download_removes_partial_staging_and_leaves_no_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def download(repo_id, *, filename, revision, local_dir):
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True)
                path.write_bytes(b"partial")
                raise OSError("network interrupted")

            with (
                patch.dict(sys.modules, {"huggingface_hub": self.hub(download)}),
                self.assertRaisesRegex(OSError, "network interrupted"),
            ):
                prepare_iris_turbo.install("r2-1000", directory)
            self.assertEqual(list(directory.iterdir()), [])

    def test_cli_rejects_unknown_adapter_before_installing(self):
        with (
            patch.object(prepare_iris_turbo, "install") as install,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            prepare_iris_turbo.main(["--adapter", "main"])
        self.assertEqual(error.exception.code, 2)
        install.assert_not_called()

    def test_cli_failed_verification_never_reports_success(self):
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch.object(prepare_iris_turbo, "install", side_effect=ValueError("SHA-256 mismatch")),
            redirect_stdout(output),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as error,
        ):
            prepare_iris_turbo.main([])
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("SHA-256 mismatch", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
