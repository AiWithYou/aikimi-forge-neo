"""Pinned model installation verifies content before recording readiness."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import setup_moss_image as setup


class MossImageSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "model"
        self.data = b"verified model"
        self.spec = {
            "path": "transformer/model.safetensors",
            "size": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }
        self.manifest = {
            "schema_version": 2,
            "repo_id": "OpenMOSS-Team/MOSS-Image-SFT",
            "revision": "a" * 40,
            "files": [self.spec],
            "assets": {},
        }
        self.spec.update(
            repo_id=self.manifest["repo_id"], revision=self.manifest["revision"], filename=self.spec["path"]
        )
        self.enterContext(patch.object(setup, "prepare_runtime"))
        self.enterContext(patch.object(setup.core, "runtime_ready", return_value=True))
        self.plan = self.enterContext(patch.object(setup, "download_plan", return_value=[self.spec]))

    def populate(self, data=None):
        path = self.root / self.spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data if data is None else data)
        return path

    def test_dry_run_has_no_download_or_filesystem_side_effect(self):
        download = Mock()
        with patch.object(setup, "manifest", return_value=self.manifest):
            result = setup.run(root=self.root, dry_run=True, download=download)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["bytes"], len(self.data))
        self.assertFalse(self.root.exists())
        download.assert_not_called()

    def test_valid_local_asset_is_reused_and_receipt_records_verified_content(self):
        path = self.populate()
        download = Mock()
        with patch.object(setup, "manifest", return_value=self.manifest):
            result = setup.run(root=self.root, download=download)
        download.assert_not_called()
        self.assertTrue(result["ok"])
        receipt = json.loads((self.root / ".forge-moss-image.json").read_text())
        self.assertEqual(receipt["model_revision"], self.manifest["revision"])
        self.assertEqual(receipt["files"][self.spec["path"]]["sha256"], self.spec["sha256"])
        self.assertEqual(receipt["files"][self.spec["path"]]["mtime_ns"], path.stat().st_mtime_ns)

    def test_tampered_same_size_asset_never_produces_a_ready_receipt(self):
        self.populate(b"x" * len(self.data))
        download = Mock(return_value=str(self.root / self.spec["path"]))
        with patch.object(setup, "manifest", return_value=self.manifest):
            with self.assertRaises(setup.MossImageSetupError):
                setup.run(root=self.root, download=download)
        self.assertFalse((self.root / ".forge-moss-image.json").exists())
        self.assertTrue(download.call_args.kwargs["force_download"])
        self.assertEqual(download.call_args.kwargs["revision"], self.manifest["revision"])

    def test_download_is_pinned_and_verified_before_success(self):
        def download(**kwargs):
            self.assertEqual(kwargs["revision"], self.manifest["revision"])
            self.assertEqual(kwargs["filename"], self.spec["path"])
            return str(self.populate())

        with patch.object(setup, "manifest", return_value=self.manifest):
            self.assertTrue(setup.run(root=self.root, download=download)["ok"])

    def test_unsafe_manifest_path_rejected_before_downloading(self):
        self.manifest["files"][0]["path"] = "../outside.safetensors"
        download = Mock()
        with patch.object(setup, "manifest", return_value=self.manifest):
            with self.assertRaises(setup.MossImageSetupError):
                setup.run(root=self.root, download=download)
        download.assert_not_called()

    def test_download_returning_same_file_with_a_path_alias_is_verified(self):
        def download(**kwargs):
            actual = self.populate()
            return str(actual.parent / ".." / actual.parent.name / actual.name)

        with patch.object(setup, "manifest", return_value=self.manifest):
            self.assertTrue(setup.run(root=self.root, download=download)["ok"])
        self.assertTrue((self.root / ".forge-moss-image.json").is_file())

    def test_failure_invalidates_an_old_ready_receipt(self):
        self.populate(b"x" * len(self.data))
        receipt = self.root / ".forge-moss-image.json"
        receipt.write_text("{}")
        with patch.object(setup, "manifest", return_value=self.manifest):
            with self.assertRaises(setup.MossImageSetupError):
                setup.run(root=self.root, download=Mock(side_effect=OSError("network unavailable")))
        self.assertFalse(receipt.exists())

    def test_checked_in_manifest_pins_all_code_and_weight_files(self):
        value = setup.manifest()
        self.assertEqual(value["repo_id"], "OpenMOSS-Team/MOSS-Image-SFT")
        self.assertRegex(value["revision"], "^[0-9a-f]{40}$")
        files = {item["path"] for item in value["files"]}
        self.assertIn("checkpoints/MOSS-Image-SFT/LICENSE", files)
        self.assertIn("checkpoints/MOSS-Image-SFT/tokenizer/chat_template.jinja", files)
        for item in [*value["files"], *value["assets"].values()]:
            self.assertRegex(item["sha256"], "^[0-9a-f]{64}$")
            self.assertGreater(item["size"], 0)

    def test_runtime_only_skips_weights_without_reporting_the_model_ready(self):
        with patch.object(setup.core, "model_ready", return_value=False):
            value = setup.run(root=self.root, runtime_only=True, download=Mock())
        self.assertTrue(value["runtime_ready"])
        self.assertFalse(value["model_ready"])
        self.assertFalse((self.root / ".forge-moss-image.json").exists())

    def test_selected_precision_is_passed_to_download_plan(self):
        with patch.object(setup, "manifest", return_value=self.manifest):
            setup.run(root=self.root, dry_run=True, precision="bf16", text_precision="w4a8")
        self.assertEqual(self.plan.call_args.kwargs, {"precision": "bf16", "text_precision": "w4a8", "with_dmd": False})

    def test_fresh_checkout_is_pinned_before_dirty_worktree_check(self):
        calls = []
        runtime = self.root / "ComfyUI"

        def command(args, **kwargs):
            calls.append(args)
            if "clone" in args:
                (runtime / ".git").mkdir(parents=True)
            return ""

        setup.prepare_comfy_checkout(runtime, command, "git")
        self.assertTrue(any("checkout" in args for args in calls))
        self.assertFalse(any("status" in args for args in calls))

    def test_dirty_comfy_checkout_is_preserved(self):
        runtime = self.root / "ComfyUI"
        (runtime / ".git").mkdir(parents=True)
        (runtime / ".git/index").touch()
        command = Mock(return_value=" M comfy/sd.py")
        with self.assertRaises(setup.MossImageSetupError):
            setup.prepare_comfy_checkout(runtime, command, "git")
        self.assertEqual(command.call_count, 1)
