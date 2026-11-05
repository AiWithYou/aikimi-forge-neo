"""Installer and isolated dependency provenance contracts without downloads."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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

    def test_pinned_runtime_is_accepted_and_old_direct_dependencies_are_rejected(self):
        records = self.distributions()
        next(item for item in records if item.metadata["Name"] == "Pillow").version = "13.0.0"
        environment._validate(records)
        for name, old in (("Pillow", "12.3.0"), ("setuptools", "83.0.0"), ("certifi", "2026.7.22"), ("gguf", "0.18.0")):
            with self.subTest(name=name):
                records = self.distributions()
                next(item for item in records if item.metadata["Name"] == name).version = old
                with self.assertRaisesRegex(RuntimeError, name):
                    environment._validate(records)

    def test_installer_and_runtime_use_the_same_pins(self):
        self.assertEqual(setup.MODEL_REVISION, MODEL_REVISION)
        self.assertEqual(setup.DIFFUSERS_REVISION, DIFFUSERS_REVISION)
        self.assertEqual(environment.DIFFUSERS_REVISION, DIFFUSERS_REVISION)

    def test_direct_installation_pins_match_every_runtime_version_check(self):
        pins, urls = {}, {}
        lines = setup.REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines.count("./vendor/accelerate"), 1)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line == "./vendor/accelerate":
                continue
            requirement = Requirement(line)
            name = canonicalize_name(requirement.name)
            self.assertNotIn(name, pins)
            self.assertNotIn(name, urls)
            self.assertIsNone(requirement.marker)
            if requirement.url:
                urls[name] = requirement.url
                continue
            versions = list(requirement.specifier)
            self.assertEqual(len(versions), 1, line)
            self.assertEqual(versions[0].operator, "==", line)
            self.assertNotIn("*", versions[0].version, line)
            pins[name] = versions[0].version

        checked = {
            canonicalize_name(name): version for name, version in environment.VERSIONS.items() if name != "accelerate"
        }
        self.assertEqual(pins, checked)
        self.assertEqual(
            urls, {"diffusers": f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_REVISION}"}
        )
        vendor = setup.ROOT / "vendor/accelerate"
        setup_tree = ast.parse((vendor / "setup.py").read_text(encoding="utf-8"))
        setup_calls = [
            node for node in ast.walk(setup_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
        ]
        self.assertEqual(len(setup_calls), 1)
        setup_versions = [item.value for item in setup_calls[0].keywords if item.arg == "version"]
        init_tree = ast.parse((vendor / "src/accelerate/__init__.py").read_text(encoding="utf-8"))
        init_versions = [
            node.value for node in init_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        ]
        self.assertEqual(len(setup_versions), 1)
        self.assertEqual(len(init_versions), 1)
        for value in (*setup_versions, *init_versions):
            self.assertEqual(ast.literal_eval(value), environment.VERSIONS["accelerate"])
        installed = [FakeDistribution(name, version) for name, version in pins.items()]
        installed.extend([
            FakeDistribution("accelerate", "1.15.0+aikimi.1"),
            FakeDistribution("diffusers", "0.41.0.dev0", DIFFUSERS_REVISION),
        ])
        environment._validate(installed)

    def test_environment_status_reads_installed_metadata_without_importing_model_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker-env"
            if environment.os.name == "nt":
                python, site = worker / "Scripts/python.exe", worker / "Lib/site-packages"
            else:
                python, site = worker / "bin/python", worker / "lib/python3.13/site-packages"
            python.parent.mkdir(parents=True)
            python.touch()
            for name, version in {**environment.VERSIONS, "diffusers": "0.41.0.dev0"}.items():
                distribution = site / f"{name.replace('-', '_')}-{version}.dist-info"
                distribution.mkdir(parents=True)
                (distribution / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
                )
                if name == "diffusers":
                    (distribution / "direct_url.json").write_text(
                        json.dumps({"vcs_info": {"commit_id": DIFFUSERS_REVISION}}), encoding="utf-8"
                    )
            self.assertTrue(environment.environment_status(root)[0])
            pillow = site / "Pillow-13.0.0.dist-info/METADATA"
            pillow.write_text("Metadata-Version: 2.1\nName: Pillow\nVersion: 12.3.0\n", encoding="utf-8")
            ready, message = environment.environment_status(root)
            self.assertFalse(ready)
            self.assertIn("Pillow==13.0.0", message)

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

    def test_official_turbo_dry_run_reports_one_pinned_install_for_three_precisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            with mock.patch("builtins.print") as printed:
                self.assertEqual(setup.main(["--root", str(root), "--official-turbo-only", "--dry-run"]), 0)
                plan = json.loads(printed.call_args.args[0])
            self.assertEqual(plan["model"], "Qwen/Qwen-Image-2.1-Turbo")
            self.assertEqual(plan["model_revision"], "d65dbc9a7e8f6b5479e33dee6030eaab2a906509")
            self.assertEqual(plan["model_bytes"], 14_230_316_125)
            self.assertEqual(plan["precision"], ["turbo_official_int8", "turbo_official_w4a8", "turbo_official_bf16"])
            self.assertIn(
                "Qwen/Qwen-Image-2.1-Turbo/blob/d65dbc9a7e8f6b5479e33dee6030eaab2a906509/LICENSE", plan["license"]
            )
            self.assertFalse(root.exists())

    def test_official_turbo_install_reuses_ready_environment_and_shared_components(self):
        for platform in ("nt", "posix"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                python = root / "worker-env" / ("Scripts/python.exe" if platform == "nt" else "bin/python")
                with (
                    mock.patch("builtins.print"),
                    mock.patch.object(setup, "os", SimpleNamespace(name=platform)),
                    mock.patch("modules_forge.qwen_image21.core.runtime_lock"),
                    mock.patch("modules_forge.qwen_image21.core.runtime_manifest", return_value={}),
                    mock.patch.object(environment, "environment_status", return_value=(True, "ready")),
                    mock.patch.object(setup, "install_environment") as install,
                    mock.patch.object(setup, "download_model") as download,
                    mock.patch.object(setup, "execute") as execute,
                ):
                    self.assertEqual(setup.main(["--root", str(root), "--official-turbo-only"]), 0)
                install.assert_not_called()
                download.assert_not_called()
                execute.assert_called_once()
                arguments = execute.call_args.args[0]
                self.assertEqual(arguments[0], python)
                self.assertIn("--download-turbo-profile", arguments)
                self.assertIn("turbo_official_int8", arguments)

    def test_runtime_only_registers_environment_without_downloading_model(self):
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
            record = json.loads((target / "runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(record["python"], str(target / "python"))
            self.assertFalse((target / "model-files.json").exists())
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
                    side_effect=[*[QwenImage21Error("missing")] * 7, {}],
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

    def test_regular_gguf_addition_reuses_official_turbo_shared_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "worker-env/Scripts/python.exe"
            regular_added = False

            def installed(_root, precision):
                if precision == "turbo_official_int8" or precision == "base_q4_k_m" and regular_added:
                    return {}
                raise QwenImage21Error("not installed")

            def execute(arguments):
                nonlocal regular_added
                self.assertIn("--download-regular-gguf", arguments)
                regular_added = True

            with (
                mock.patch("builtins.print"),
                mock.patch("modules_forge.qwen_image21.core.runtime_lock"),
                mock.patch("modules_forge.qwen_image21.core.runtime_manifest", side_effect=installed),
                mock.patch.object(setup, "install_environment", return_value=python),
                mock.patch.object(setup, "download_model") as download,
                mock.patch.object(setup, "execute", side_effect=execute) as run,
            ):
                self.assertEqual(setup.main(["--root", str(root)]), 0)
            download.assert_not_called()
            run.assert_called_once()
            self.assertTrue(regular_added)

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
