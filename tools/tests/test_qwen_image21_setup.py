"""Installer and isolated dependency provenance contracts without downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge import qwen_image21_environment as environment
from modules_forge.qwen_image21.core import DIFFUSERS_REVISION, MODEL_REVISION, QwenImage21Error, atomic_json
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

    def test_default_dry_run_uses_unsloth_and_full_flag_keeps_official_source(self):
        from modules_forge.qwen_image21 import regular_gguf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            for flags, model, precision in (
                ([], regular_gguf.MODEL_ID, ["base_q4_k_m"]),
                (["--official-full"], setup.MODEL_ID, ["int8", "w4a8", "bf16"]),
            ):
                with self.subTest(flags=flags), mock.patch("builtins.print") as printed:
                    self.assertEqual(setup.main(["--root", str(root), *flags, "--dry-run"]), 0)
                    plan = json.loads(printed.call_args.args[0])
                    self.assertEqual(plan["model"], model)
                    self.assertEqual(plan["precision"], precision)
            self.assertFalse(root.exists())

    def test_turbo_dry_run_points_to_pinned_models_without_writing_files(self):
        from modules_forge.qwen_image21.turbo import GGUF_ID, GGUF_REVISION, VIGGLE_ID, VIGGLE_REVISION

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            for flag, model, revision in (
                ("--turbo-q4-only", GGUF_ID, GGUF_REVISION),
                ("--turbo-bf16-only", VIGGLE_ID, VIGGLE_REVISION),
            ):
                with self.subTest(flag=flag), mock.patch("builtins.print") as printed:
                    self.assertEqual(setup.main(["--root", str(root), flag, "--dry-run"]), 0)
                    plan = json.loads(printed.call_args.args[0])
                    self.assertEqual((plan["model"], plan["model_revision"]), (model, revision))
                    self.assertGreater(plan["model_bytes"], 4_000_000_000)
                    self.assertGreater(plan["shared_model_bytes"], 18_000_000_000)
            self.assertFalse(root.exists())

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

    def test_default_install_downloads_shared_assets_and_unsloth_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "worker-env/Scripts/python.exe"
            with (
                mock.patch("builtins.print"),
                mock.patch("modules_forge.qwen_image21.core.runtime_lock") as lock,
                mock.patch(
                    "modules_forge.qwen_image21.core.runtime_manifest",
                    side_effect=[*[QwenImage21Error("missing")] * 4, {}],
                ) as manifest,
                mock.patch.object(setup, "install_environment", return_value=python),
                mock.patch.object(setup, "download_model") as download,
                mock.patch.object(setup, "execute") as execute,
            ):
                self.assertEqual(setup.main(["--root", str(root)]), 0)
            download.assert_called_once_with(root, python, shared_only=True)
            self.assertIn("--download-regular-gguf", execute.call_args.args[0])
            self.assertEqual(manifest.call_args.args, (root, "base_q4_k_m"))
            lock.return_value.close.assert_called_once()

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
