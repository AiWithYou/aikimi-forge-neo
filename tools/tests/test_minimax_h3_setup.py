"""H3の初回導入、モデル共有、実行中の競合を小さなローカルfixtureで検証する。"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_runtime as runtime
from tools import setup_minimax_h3 as setup
from tools.aikimi_setup import ArtifactSpec, ProfileSpec, SetupError


class ModelsOnlyCliTests(unittest.TestCase):
    def test_models_only_never_initializes_runtime(self):
        destination = Path("fixture/models/MiniMax-H3")
        with (
            mock.patch.object(setup.sys, "argv", ["setup_minimax_h3.py", "--models-only"]),
            mock.patch.object(setup, "plan", return_value={"models": str(destination)}),
            mock.patch.object(setup, "verify_models") as download,
            mock.patch.object(setup, "RuntimeInstaller") as installer,
            mock.patch.object(setup, "emit"),
        ):
            self.assertEqual(setup.main(), 0)
        download.assert_called_once_with(destination, shared=False)
        installer.assert_not_called()

    def test_models_only_reports_download_failure(self):
        with (
            mock.patch.object(setup.sys, "argv", ["setup_minimax_h3.py", "--models-only"]),
            mock.patch.object(setup, "plan", return_value={"models": "fixture/models"}),
            mock.patch.object(setup, "verify_models", side_effect=SetupError("download failed")),
            mock.patch.object(setup, "RuntimeInstaller") as installer,
            mock.patch.object(setup, "emit"),
        ):
            self.assertEqual(setup.main(), 1)
        installer.assert_not_called()

    def test_models_only_dry_run_does_not_download(self):
        with (
            mock.patch.object(setup.sys, "argv", ["setup_minimax_h3.py", "--models-only", "--dry-run"]),
            mock.patch.object(setup, "plan", return_value={"models": "fixture/models"}),
            mock.patch.object(setup, "verify_models") as download,
            mock.patch.object(setup, "RuntimeInstaller") as installer,
            mock.patch.object(setup, "emit"),
        ):
            self.assertEqual(setup.main(), 0)
        download.assert_not_called()
        installer.assert_not_called()


class ManagedH3RuntimeTests(unittest.TestCase):
    def test_external_model_yaml_cannot_select_an_external_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            other = root / "other/ComfyUI"
            other.mkdir(parents=True)
            (other / "main.py").write_text("external", encoding="utf-8")
            (root / "forge_neo_model_paths.yaml").write_text(
                f"external:\n  base_path: '{other.as_posix()}/models'\n",
                encoding="utf-8",
            )
            self.assertIsNone(runtime.installed_runtime_root(root))
            self.assertEqual(runtime.managed_runtime_root(root), root.resolve() / "repositories/minimax-h3/ComfyUI")

    def test_incomplete_or_outdated_setup_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            managed = runtime.managed_runtime_root(root)
            managed.mkdir(parents=True)
            python = managed.parent / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            (managed / "main.py").touch()
            (managed / "extra_model_paths.yaml").write_text(runtime.model_config(root / "models"), encoding="utf-8")
            record = managed.parent / "setup.json"
            self.assertIsNone(runtime.installed_runtime_root(root))
            with mock.patch.object(runtime, "setup_fingerprint", return_value="current"):
                record.write_text(json.dumps({"schema_version": 1, "fingerprint": "old"}), encoding="utf-8")
                self.assertIsNone(runtime.installed_runtime_root(root))
                record.write_text(json.dumps({"schema_version": 1, "fingerprint": "current"}), encoding="utf-8")
                self.assertEqual(runtime.installed_runtime_root(root), managed)

    def test_shared_model_settings_are_used_without_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            managed = runtime.managed_runtime_root(root)
            managed.mkdir(parents=True)
            models = root / "shared"
            models.mkdir()
            (managed / "extra_model_paths.yaml").write_text(runtime.model_config(models), encoding="utf-8")
            self.assertEqual(runtime.model_root(managed), models.resolve())
            self.assertFalse((managed / "models").exists())

    def test_invalid_model_configuration_does_not_use_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "extra_model_paths.yaml"
            for content in ("broken", "{}", '{"aikimi_h3":{"base_path":"https://example.com"}}'):
                config.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    runtime.model_root(root)

    def test_network_model_paths_are_rejected(self):
        for value in (r"\\server\models", "//server/models", "https://example.com/models"):
            with (
                self.subTest(value=value),
                mock.patch.object(Path, "resolve", side_effect=AssertionError("network lookup")),
                self.assertRaises(ValueError),
            ):
                runtime.local_directory(value)

    def test_setup_lock_prevents_other_installs_and_backend_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ComfyUI"
            root.mkdir()
            (root / "main.py").touch()
            (root / "models").mkdir()
            with runtime.setup_lock(root), mock.patch.object(bridge, "_start_runtime_locked") as start:
                with self.assertRaises(ValueError):
                    with runtime.setup_lock(root):
                        self.fail("lock acquired twice")
                with self.assertRaises(bridge.H3BridgeError):
                    bridge.start_runtime(root, runtime.SERVER_URL, Path(temporary))
                start.assert_not_called()


class H3ModelSetupTests(unittest.TestCase):
    @staticmethod
    def profile():
        artifact = ArtifactSpec(
            "test",
            "diffusion_models/test.bin",
            "https://example.com/test.bin",
            3,
            hashlib.sha256(b"abc").hexdigest(),
            "https://example.com/LICENSE",
        )
        return ProfileSpec("h3", "fixture", (artifact,), (artifact.license_url,), 3)

    def test_sharing_verifies_existing_files_without_writes_or_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "diffusion_models/test.bin"
            model.parent.mkdir()
            model.write_bytes(b"abc")
            before = {p.relative_to(root): p.stat().st_mtime_ns for p in root.rglob("*")}
            with (
                mock.patch.object(setup, "model_profile", return_value=self.profile()),
                mock.patch.object(
                    setup.Installer,
                    "_download",
                    side_effect=AssertionError("shared models must not be downloaded"),
                ),
            ):
                self.assertTrue(setup.verify_models(root, shared=True)["ok"])
            self.assertEqual(before, {p.relative_to(root): p.stat().st_mtime_ns for p in root.rglob("*")})

    def test_shared_corruption_is_reported_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "diffusion_models/test.bin"
            model.parent.mkdir()
            model.write_bytes(b"bad")
            with mock.patch.object(setup, "model_profile", return_value=self.profile()), self.assertRaises(SetupError):
                setup.verify_models(root, shared=True)
            self.assertEqual(model.read_bytes(), b"bad")

    def test_plan_does_not_create_directories_and_keeps_downloads_inside_neo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Neo"
            with mock.patch.object(setup, "model_profile", return_value=self.profile()):
                result = setup.plan(root)
            self.assertEqual(result["model_download_bytes"], 3)
            self.assertEqual(Path(result["models"]), root.resolve() / "models/MiniMax-H3")
            self.assertFalse(root.exists())

    def test_missing_shared_models_fail_before_core_or_python_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installer = setup.RuntimeInstaller(root)
            with (
                mock.patch.object(setup.sys, "platform", "win32"),
                mock.patch.object(setup.shutil, "which", return_value="git"),
                mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=100 * 1024**3)),
                mock.patch.object(setup.socket, "socket") as connection,
                mock.patch.object(setup, "model_profile", return_value=self.profile()),
                mock.patch.object(installer, "install_core") as core,
                mock.patch.object(setup, "emit"),
            ):
                connection.return_value.__enter__.return_value.connect_ex.return_value = 1
                with self.assertRaises(SetupError):
                    installer.install(root / "missing-models")
                core.assert_not_called()
                self.assertFalse((installer.base / "setup.json").exists())

    def test_interrupted_install_invalidates_old_completion_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            installer = setup.RuntimeInstaller(Path(temporary))
            installer.base.mkdir(parents=True)
            record = installer.base / "setup.json"
            record.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(setup.sys, "platform", "win32"),
                mock.patch.object(setup.shutil, "which", return_value="git"),
                mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=100 * 1024**3)),
                mock.patch.object(setup.socket, "socket") as connection,
                mock.patch.object(installer, "install_core", side_effect=SetupError("interrupted")),
                mock.patch.object(setup, "emit"),
            ):
                connection.return_value.__enter__.return_value.connect_ex.return_value = 1
                with self.assertRaises(SetupError):
                    installer.install()
                self.assertFalse(record.exists())


class H3SetupLifecycleTests(unittest.TestCase):
    def test_windows_python_child_is_owned_but_an_unrelated_listener_is_not(self):
        identity = (Path("runtime"), runtime.SERVER_URL)
        parent = mock.Mock(pid=100)
        parent.poll.return_value = None
        listener = mock.Mock(pid=101)
        listener.parents.return_value = [mock.Mock(pid=100)]
        with (
            mock.patch.object(bridge, "_MANAGED_PROCESS", parent),
            mock.patch.object(
                bridge,
                "_MANAGED_PROCESS_IDENTITY",
                identity,
            ),
        ):
            self.assertTrue(bridge._owns_runtime_process(listener, identity))
            listener.parents.return_value = [mock.Mock(pid=200)]
            self.assertFalse(bridge._owns_runtime_process(listener, identity))
            listener.parents.return_value = [parent]
            self.assertFalse(bridge._owns_runtime_process(listener, (Path("other"), runtime.SERVER_URL)))

    def test_stop_terminates_the_owned_python_child_before_releasing_its_launcher(self):
        identity = (Path("runtime"), runtime.SERVER_URL)
        parent = mock.Mock(pid=100)
        parent.poll.return_value = None
        listener = mock.Mock(pid=101)
        listener.parents.return_value = [parent]
        events = []
        listener.terminate.side_effect = lambda: events.append("listener stopped")
        parent.wait.side_effect = lambda **_: events.append("launcher exited")
        with (
            mock.patch.object(bridge, "_MANAGED_PROCESS", parent),
            mock.patch.object(
                bridge,
                "_MANAGED_PROCESS_IDENTITY",
                identity,
            ),
        ):
            bridge._stop_managed_runtime(listener)
            self.assertIsNone(bridge._MANAGED_PROCESS)
            self.assertIsNone(bridge._MANAGED_PROCESS_IDENTITY)
        self.assertEqual(events, ["listener stopped", "launcher exited"])
        parent.terminate.assert_not_called()

    def test_setup_rejects_active_generation_before_stopping_any_process(self):
        with (
            mock.patch.object(bridge, "_active_generation_count", return_value=1),
            mock.patch.object(
                bridge,
                "_stop_managed_runtime",
            ) as stop,
        ):
            with self.assertRaises(bridge.H3BridgeError), bridge.runtime_setup_session(Path("runtime")):
                self.fail("setup must not start")
            stop.assert_not_called()

    def test_setup_rejects_an_external_process(self):
        with (
            mock.patch.object(bridge, "server_runtime_root", return_value=Path("other")),
            mock.patch.object(
                bridge,
                "_loopback_server_process",
                return_value=mock.Mock(pid=123),
            ),
            mock.patch.object(bridge, "_stop_managed_runtime") as stop,
        ):
            with self.assertRaises(bridge.H3BridgeError), bridge.runtime_setup_session(Path("runtime")):
                self.fail("external process must not stop")
            stop.assert_not_called()

    def test_setup_blocks_generation_and_can_finish_on_a_different_worker(self):
        with mock.patch.object(bridge, "server_runtime_root", return_value=None):
            context = bridge.runtime_setup_session(Path("runtime"))
            context.__enter__()
            try:
                with self.assertRaises(bridge.H3BridgeError):
                    bridge.ensure_ready(Path("runtime"), runtime.SERVER_URL, Path("logs"))
                failures = []

                def finish():
                    try:
                        context.__exit__(None, None, None)
                    except Exception as error:
                        failures.append(error)

                worker = threading.Thread(target=finish)
                worker.start()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(failures, [])
                self.assertFalse(bridge._RUNTIME_SETUP_ACTIVE)
            finally:
                bridge._RUNTIME_SETUP_ACTIVE = False

    def test_setup_callback_reenables_controls_after_cli_failure(self):
        from contextlib import nullcontext

        from modules_forge import minimax_h3_setup_ui as ui

        process = mock.Mock(stdout=io.StringIO("セットアップ失敗\n"))
        process.wait.return_value = 1
        process.poll.return_value = 1
        with (
            mock.patch.object(ui, "runtime_setup_session", return_value=nullcontext()),
            mock.patch.object(
                ui.subprocess,
                "Popen",
                return_value=process,
            ),
        ):
            updates = list(ui.setup_updates("", Path(__file__).resolve().parents[2]))
        self.assertIn("失敗", updates[-1][1])
        self.assertTrue(updates[-1][2]["interactive"])
        self.assertFalse(bridge._RUNTIME_SETUP_ACTIVE)


if __name__ == "__main__":
    unittest.main()
