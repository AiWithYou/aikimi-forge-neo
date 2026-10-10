"""Pinned Orbit download and local verification with small offline fixtures."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules_forge import local_assets
from modules_forge import minimax_h3_orbit_assets as orbit
from modules_forge import minimax_h3_runtime as runtime
from tools import prepare_minimax_h3_orbit as cli


class OrbitManifestTests(unittest.TestCase):
    def test_release_is_fixed_to_the_authors_published_bytes(self):
        self.assertEqual(orbit.MODEL_NAME, "minimax_h3_flf2v_lora_v1.safetensors")
        self.assertEqual(orbit.REVISION, "5ddbc2dbbe95edbbdaf5017c3e934b1d01791697")
        self.assertEqual(orbit.MODEL_BYTES, 155_111_424)
        self.assertEqual(orbit.MODEL_SHA256, "14f13e3effaf3e729fdc0c97680344aa63f473be0c55963f963d718b3db2a4d4")
        self.assertEqual(
            orbit.MODEL_URL,
            "https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/resolve/"
            "5ddbc2dbbe95edbbdaf5017c3e934b1d01791697/minimax_h3_flf2v_lora_v1.safetensors",
        )

    def test_imports_do_not_download_or_make_directories(self):
        with (
            mock.patch("urllib.request.urlopen", side_effect=AssertionError("network during import")),
            mock.patch.object(Path, "mkdir", side_effect=AssertionError("directory created during import")),
        ):
            importlib.reload(orbit)
            importlib.reload(cli)


class TrackingResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.received = 0
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        chunk = super().read(size)
        self.received += len(chunk)
        return chunk


class OrbitInstallerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="h3-orbit-assets-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.payload = b"orbit-fixture-video-model"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.enterContext(mock.patch.object(orbit, "MODEL_BYTES", len(self.payload)))
        self.enterContext(mock.patch.object(orbit, "MODEL_SHA256", self.digest))
        self.enterContext(mock.patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json"))
        self.download = self.enterContext(
            mock.patch("urllib.request.urlopen", side_effect=AssertionError("unexpected external download"))
        )

    def path(self):
        return runtime.configured_model_root(self.root) / "loras" / orbit.MODEL_NAME

    def seed(self, payload=None):
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload if payload is None else payload)
        return path

    def serve(self, payload=None):
        response = TrackingResponse(self.payload if payload is None else payload)
        self.download.side_effect = None
        self.download.return_value = response
        return response

    def assert_no_parts(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_missing_is_not_installed_and_verify_never_downloads(self):
        self.assertFalse(orbit.installed(self.root))
        with self.assertRaises(ValueError):
            orbit.validate_model(runtime.managed_runtime_root(self.root))
        self.download.assert_not_called()
        self.assertFalse(self.path().parent.exists())

    def test_installer_streams_verifies_and_publishes_only_complete_bytes(self):
        response = self.serve()
        path = orbit.install(self.root)
        self.assertEqual(path, self.path())
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertTrue(orbit.installed(self.root))
        self.download.assert_called_once_with(orbit.MODEL_URL, timeout=90)
        self.assertTrue(all(0 < size <= len(self.payload) + 1 for size in response.read_sizes))
        self.assert_no_parts()

    def test_install_is_idempotent_and_uses_cached_local_identity(self):
        path = self.seed()
        before = path.stat().st_mtime_ns
        with mock.patch.object(orbit, "file_identity", wraps=local_assets.file_identity) as identify:
            self.assertTrue(orbit.installed(self.root))
            self.assertEqual(orbit.install(self.root), path)
        self.assertEqual(identify.call_count, 2)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.download.assert_not_called()
        self.assert_no_parts()

    def test_same_size_tampering_with_restored_mtime_is_detected(self):
        path = self.seed()
        self.assertTrue(orbit.installed(self.root))
        before = path.stat()
        path.write_bytes(b"!" * len(self.payload))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertFalse(orbit.installed(self.root))
        with self.assertRaises(ValueError):
            orbit.install(self.root)
        self.assertEqual(path.read_bytes(), b"!" * len(self.payload))
        self.download.assert_not_called()

    def test_configured_shared_directory_is_used_by_install_and_runtime_validation(self):
        managed = runtime.managed_runtime_root(self.root)
        managed.mkdir(parents=True)
        shared = self.root / "shared-h3-models"
        (managed / "extra_model_paths.yaml").write_text(runtime.model_config(shared), encoding="utf-8")
        self.serve()
        expected = shared.resolve() / "loras" / orbit.MODEL_NAME
        self.assertEqual(orbit.install(self.root), expected)
        self.assertEqual(orbit.validate_model(managed), expected)
        self.assertTrue(orbit.installed(self.root))
        self.assertFalse((self.root / "models/MiniMax-H3").exists())

    def test_selected_external_runtime_is_used_without_managed_fallback(self):
        selected = self.root / "external/ComfyUI"
        path = selected / "models/loras" / orbit.MODEL_NAME
        self.serve()
        self.assertEqual(orbit.install(self.root, runtime_root=selected), path)
        self.assertTrue(orbit.installed(self.root, runtime_root=selected))
        self.assertFalse(orbit.installed(self.root))
        self.assertFalse(self.path().exists())
        path.write_bytes(b"!" * len(self.payload))
        self.assertFalse(orbit.installed(self.root, runtime_root=selected))

    def test_runtime_without_config_uses_its_own_models_directory(self):
        managed = self.root / "runtime/ComfyUI"
        path = managed / "models/loras" / orbit.MODEL_NAME
        path.parent.mkdir(parents=True)
        path.write_bytes(self.payload)
        self.assertEqual(orbit.validate_model(managed), path)
        path.write_bytes(b"!" * len(self.payload))
        with self.assertRaises(ValueError):
            orbit.validate_model(managed)
        self.download.assert_not_called()

    def test_short_hash_mismatched_and_oversized_downloads_leave_no_final_or_partial_file(self):
        for payload in (self.payload[:-1], b"!" * len(self.payload), self.payload + b"extra-data" * 100):
            with self.subTest(size=len(payload)):
                response = self.serve(payload)
                with self.assertRaises(ValueError):
                    orbit.install(self.root)
                self.assertLessEqual(response.received, len(self.payload) + 1)
                self.assertFalse(self.path().exists())
                self.assert_no_parts()

    def test_network_failure_cleans_partial_bytes(self):
        response = self.serve()
        original_read = response.read
        reads = 0

        def failed_read(size):
            nonlocal reads
            reads += 1
            if reads == 1:
                return original_read(3)
            raise OSError("connection lost")

        response.read = failed_read
        with self.assertRaises(OSError):
            orbit.install(self.root)
        self.assertFalse(self.path().exists())
        self.assert_no_parts()

    def test_different_existing_file_is_retained_without_network(self):
        path = self.seed(b"existing-different-model")
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            orbit.install(self.root)
        self.assertEqual(path.read_bytes(), before)
        self.download.assert_not_called()
        self.assert_no_parts()

    def test_competing_file_is_never_clobbered_during_atomic_publish(self):
        self.serve()
        link = os.link

        def competing_publish(source, target):
            Path(target).write_bytes(b"competing-model")
            return link(source, target)

        with mock.patch.object(orbit.os, "link", side_effect=competing_publish):
            with self.assertRaises(ValueError):
                orbit.install(self.root)
        self.assertEqual(self.path().read_bytes(), b"competing-model")
        self.assert_no_parts()

    def test_setup_lock_blocks_install_before_network(self):
        with runtime.setup_lock(runtime.managed_runtime_root(self.root)):
            with self.assertRaises(ValueError):
                orbit.install(self.root)
        self.download.assert_not_called()
        self.assertFalse(self.path().parent.exists())

    def test_invalid_shared_settings_do_not_fall_back_or_download(self):
        managed = runtime.managed_runtime_root(self.root)
        managed.mkdir(parents=True)
        (managed / "extra_model_paths.yaml").write_text("broken", encoding="utf-8")
        self.assertFalse(orbit.installed(self.root))
        with self.assertRaises(ValueError):
            orbit.install(self.root)
        self.download.assert_not_called()


class OrbitCliTests(unittest.TestCase):
    def test_verify_reports_missing_without_installer(self):
        with (
            mock.patch.object(cli, "installed", return_value=False),
            mock.patch.object(cli, "install") as install,
            mock.patch.object(cli.sys, "stderr", io.StringIO()),
        ):
            self.assertEqual(cli.main(["--verify"]), 1)
        install.assert_not_called()

    def test_default_invokes_fixed_install(self):
        with (
            mock.patch.object(cli, "install", return_value=Path("fixture/model.safetensors")) as install,
            mock.patch.object(cli.sys, "stdout", io.StringIO()) as output,
        ):
            self.assertEqual(cli.main([]), 0)
            self.assertIn("fixture", output.getvalue())
        install.assert_called_once_with(cli.ROOT)


if __name__ == "__main__":
    unittest.main()
