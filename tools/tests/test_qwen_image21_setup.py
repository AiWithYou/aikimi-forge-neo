"""Installer and isolated dependency provenance contracts without downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge import qwen_image21_environment as environment
from modules_forge.qwen_image21.core import DIFFUSERS_REVISION, MODEL_REVISION, atomic_json
from tools import setup_qwen_image21 as setup


class FakeDistribution:
    def __init__(self, name, version, commit=None):
        self.metadata = {"Name": name}
        self.version = version
        self.commit = commit

    def read_text(self, _name):
        return json.dumps({"vcs_info": {"commit_id": self.commit}})


class QwenSetupTests(unittest.TestCase):
    def distributions(self):
        return [
            *(FakeDistribution(name, version) for name, version in environment.VERSIONS.items()),
            FakeDistribution("diffusers", "0.41.0.dev0", DIFFUSERS_REVISION),
        ]

    def test_dependency_validation_requires_new_offload_versions_and_pinned_source(self):
        records = self.distributions()
        environment._validate(records)
        next(item for item in records if item.metadata["Name"] == "bitsandbytes").version = "0.47.0"
        with self.assertRaisesRegex(RuntimeError, "bitsandbytes"):
            environment._validate(records)
        records = self.distributions()
        records[-1].commit = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "Diffusers"):
            environment._validate(records)

    def test_installer_and_runtime_use_the_same_pins(self):
        self.assertEqual(setup.MODEL_REVISION, MODEL_REVISION)
        self.assertEqual(setup.DIFFUSERS_REVISION, DIFFUSERS_REVISION)
        self.assertEqual(environment.DIFFUSERS_REVISION, DIFFUSERS_REVISION)

    def test_dry_run_creates_no_environment_or_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created"
            with mock.patch("builtins.print"), mock.patch.object(setup, "execute") as execute:
                self.assertEqual(setup.main(["--root", str(target), "--dry-run"]), 0)
            execute.assert_not_called()
            self.assertFalse(target.exists())

    def test_runtime_only_never_registers_a_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            lock = mock.Mock()
            with (
                mock.patch("builtins.print"),
                mock.patch("modules_forge.qwen_image21.core.runtime_lock", return_value=lock),
                mock.patch.object(setup, "install_environment", return_value=target / "python"),
                mock.patch.object(setup, "download_model") as download,
            ):
                self.assertEqual(setup.main(["--root", str(target), "--runtime-only"]), 0)
            download.assert_not_called()
            self.assertFalse((target / "runtime.json").exists())
            lock.close.assert_called_once()

    def test_corrupt_download_does_not_publish_an_install_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"bad")
            info = SimpleNamespace(
                sha=MODEL_REVISION,
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=3, lfs=SimpleNamespace(sha256="0" * 64))],
            )
            hub = SimpleNamespace(
                HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: info), snapshot_download=mock.Mock()
            )
            with mock.patch.dict("sys.modules", {"huggingface_hub": hub}):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    setup.fetch_and_verify(root)
            self.assertFalse((root / "model-files.json").exists())
            self.assertFalse((root / "runtime.json").exists())

    def test_verification_rejects_modified_or_escaping_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            path = root / "model" / "weight"
            path.write_bytes(b"abc")
            for record in (
                {"path": "weight", "size": 3, "sha256": "0" * 64},
                {"path": "../outside", "size": 3, "sha256": "0" * 64},
            ):
                atomic_json(root / "model-files.json", {"revision": MODEL_REVISION, "files": [record]})
                lock = mock.Mock()
                with (
                    mock.patch("modules_forge.qwen_image21.core.runtime_lock", return_value=lock),
                    mock.patch("modules_forge.qwen_image21.core.runtime_manifest", return_value={}),
                ):
                    with self.assertRaises((RuntimeError, ValueError)):
                        setup.main(["--root", str(root), "--verify"])
                lock.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
