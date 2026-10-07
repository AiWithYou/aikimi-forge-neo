"""Offline contracts for the pinned optional Qwen Image 2.1 Consistency LoRA."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.qwen_image21 import consistency_lora, style_lora


class StreamingResponse(io.BytesIO):
    """Serve tiny chunks and optionally fail after a partially written download."""

    def __init__(self, data: bytes, *, fail_after: int | None = None, observe=None):
        super().__init__(data)
        self.reads = 0
        self.fail_after = fail_after
        self.observe = observe

    def read(self, size=-1):
        if size <= 0:
            raise AssertionError("The adapter must be downloaded with bounded reads.")
        if self.observe:
            self.observe()
        if self.fail_after is not None and self.reads >= self.fail_after:
            raise OSError("injected connection interruption")
        self.reads += 1
        return super().read(min(size, 4))


class ConsistencyLoraReleaseTests(unittest.TestCase):
    def test_release_is_pinned_to_documented_repository_files_and_hashes(self):
        self.assertEqual(consistency_lora.REPOSITORY, "ausboss/Qwen-Image-2.1-Consistency-LoRA")
        self.assertEqual(consistency_lora.REVISION, "8f05b0fa027d517fa396fb31b71e0eaf48110e89")
        self.assertEqual(consistency_lora.SIZE, 159_436_496)
        self.assertEqual(
            consistency_lora.WEIGHTS,
            {
                "1500": "qwen-image-2.1-consistency.safetensors",
                "2000": "qwen-image-2.1-consistency-2000.safetensors",
            },
        )
        self.assertEqual(
            consistency_lora.HASHES,
            {
                "1500": "4f44ada1be2189cc23b3d010f9603543403f48454e2f76842a3d30109b20bd63",
                "2000": "a0bf043edc4695b1661a0e768e49a7fdff9a656b7d4313203d590e0262b64beb",
            },
        )


class ConsistencyLoraInstallerTests(unittest.TestCase):
    def setUp(self):
        from tools.tests.test_qwen_image21_service import isolate_local_assets

        temporary = tempfile.TemporaryDirectory(prefix="qwen21-consistency-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        isolate_local_assets(self, self.root)
        self.payloads = {"1500": b"consistency-1500-fixture", "2000": b"consistency-2000-fixture"}
        self.size = len(self.payloads["1500"])
        self.hashes = {version: hashlib.sha256(data).hexdigest() for version, data in self.payloads.items()}
        self.enterContext(patch.object(consistency_lora, "SIZE", self.size))
        self.enterContext(patch.object(consistency_lora, "HASHES", self.hashes))
        self.download = self.enterContext(
            patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected external model download"))
        )

    def path(self, version="1500", runtime=None):
        return consistency_lora.adapter_path(self.root if runtime is None else runtime, version)

    def receipt(self, version="1500"):
        return {
            "repository": consistency_lora.REPOSITORY,
            "revision": consistency_lora.REVISION,
            "file": consistency_lora.WEIGHTS[version],
            "sha256": self.hashes[version],
            "size": self.size,
            "version": version,
        }

    def seed_installed(self, version="1500", *, receipt=None, runtime=None):
        path = self.path(version, runtime)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payloads[version])
        path.with_suffix(".source.json").write_text(
            json.dumps(self.receipt(version) if receipt is None else receipt), encoding="utf-8"
        )
        return path

    def serve(self, data, **options):
        response = StreamingResponse(data, **options)
        self.download.side_effect = None
        self.download.return_value = response
        return response

    def assert_no_partial_files(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_default_and_both_versions_are_under_the_normal_lora_inventory(self):
        folder = "ausboss--Qwen-Image-2.1-Consistency-LoRA"
        self.assertEqual(consistency_lora.adapter_path(self.root), self.path("1500"))
        for version in self.payloads:
            with self.subTest(version=version):
                expected = self.root / "loras" / folder / consistency_lora.WEIGHTS[version]
                self.assertEqual(self.path(version), expected)
                self.serve(self.payloads[version])
                info = consistency_lora.install(self.root, version)
                self.assertEqual(info["name"], expected.relative_to(self.root / "loras").as_posix())
        self.assertEqual(
            style_lora.inventory(self.root),
            sorted(self.path(version).relative_to(self.root / "loras").as_posix() for version in self.payloads),
        )

    def test_install_streams_pinned_download_and_records_verified_source(self):
        progress = []
        response = self.serve(self.payloads["1500"])
        info = consistency_lora.install(self.root, progress=progress.append)
        path = self.path()
        expected = self.receipt()
        self.assertEqual(path.read_bytes(), self.payloads["1500"])
        self.assertEqual(json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8")), expected)
        self.assertEqual(
            info,
            {
                **expected,
                "name": path.relative_to(self.root / "loras").as_posix(),
                "path": str(path.resolve()),
            },
        )
        self.download.assert_called_once()
        args, kwargs = self.download.call_args
        self.assertEqual(
            args[0],
            f"https://huggingface.co/{consistency_lora.REPOSITORY}/resolve/"
            f"{consistency_lora.REVISION}/{consistency_lora.WEIGHTS['1500']}",
        )
        self.assertGreater(kwargs["timeout"], 0)
        self.assertGreater(response.reads, 2)
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(progress, sorted(progress))
        self.assertTrue(all(0 < value <= 1 for value in progress))
        self.assert_no_partial_files()

    def test_install_reuses_fully_verified_existing_file_without_network(self):
        path = self.seed_installed()
        before = path.stat().st_mtime_ns
        result = consistency_lora.install(self.root)
        self.assertEqual(result, consistency_lora.installed(self.root, verify=True))
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.download.assert_not_called()
        self.assert_no_partial_files()

    def test_verify_detects_same_size_corruption_without_network_and_install_repairs_it(self):
        path = self.seed_installed()
        path.write_bytes(b"!" * self.size)
        with self.assertRaises(ValueError):
            consistency_lora.installed(self.root, verify=True)
        self.download.assert_not_called()
        self.serve(self.payloads["1500"])
        consistency_lora.install(self.root)
        self.assertEqual(path.read_bytes(), self.payloads["1500"])
        self.assertEqual(consistency_lora.installed(self.root, verify=True)["sha256"], self.hashes["1500"])

    def test_verification_requires_complete_pinned_receipt_and_never_downloads(self):
        with self.assertRaises(ValueError):
            consistency_lora.installed(self.root, verify=True)
        for field in self.receipt():
            with self.subTest(field=field):
                bad_receipt = self.receipt()
                del bad_receipt[field]
                self.seed_installed(receipt=bad_receipt)
                with self.assertRaises(ValueError):
                    consistency_lora.installed(self.root, verify=True)
                bad_receipt[field] = "unexpected"
                self.seed_installed(receipt=bad_receipt)
                with self.assertRaises(ValueError):
                    consistency_lora.installed(self.root, verify=True)
        self.download.assert_not_called()

    def test_invalid_versions_fail_before_download_or_directory_creation(self):
        for version in ("1000", "../1500", "", 1500, None, True):
            with self.subTest(version=version):
                for operation in (
                    consistency_lora.adapter_path,
                    consistency_lora.installed,
                    consistency_lora.install,
                ):
                    with self.assertRaises(ValueError):
                        operation(self.root, version)
        self.download.assert_not_called()
        self.assertFalse((self.root / "loras").exists())

    def test_incomplete_oversized_and_hash_mismatched_downloads_are_not_installed(self):
        for data in (self.payloads["1500"][:-1], self.payloads["1500"] + b"!", b"!" * self.size):
            with self.subTest(size=len(data), digest=hashlib.sha256(data).hexdigest()):
                self.serve(data)
                with self.assertRaises(ValueError):
                    consistency_lora.install(self.root)
                self.assertFalse(self.path().exists())
                self.assertFalse(self.path().with_suffix(".source.json").exists())
                self.assert_no_partial_files()

    def test_failed_refresh_keeps_existing_good_weights_and_receipt(self):
        stale_receipt = {**self.receipt(), "revision": "old-revision"}
        path = self.seed_installed(receipt=stale_receipt)
        before = path.with_suffix(".source.json").read_bytes()
        for data in (self.payloads["1500"][:-1], self.payloads["1500"] + b"!", b"!" * self.size):
            with self.subTest(size=len(data), digest=hashlib.sha256(data).hexdigest()):
                self.serve(data)
                with self.assertRaises(ValueError):
                    consistency_lora.install(self.root)
                self.assertEqual(path.read_bytes(), self.payloads["1500"])
                self.assertEqual(path.with_suffix(".source.json").read_bytes(), before)
                self.assert_no_partial_files()

    def test_stream_interruption_keeps_existing_file_and_cleans_partial_download(self):
        path = self.seed_installed(receipt={**self.receipt(), "revision": "old-revision"})
        before = path.with_suffix(".source.json").read_bytes()

        def observe():
            self.assertEqual(path.read_bytes(), self.payloads["1500"])
            self.assertEqual(path.with_suffix(".source.json").read_bytes(), before)

        self.serve(self.payloads["1500"], fail_after=1, observe=observe)
        with self.assertRaisesRegex(OSError, "interruption"):
            consistency_lora.install(self.root)
        observe()
        self.assert_no_partial_files()

    def test_atomic_replacement_failure_preserves_existing_file_and_cleans_partial(self):
        path = self.seed_installed(receipt={**self.receipt(), "revision": "old-revision"})
        before = path.with_suffix(".source.json").read_bytes()
        self.serve(self.payloads["1500"])
        with patch.object(consistency_lora.os, "replace", side_effect=OSError("injected replacement failure")):
            with self.assertRaisesRegex(OSError, "replacement"):
                consistency_lora.install(self.root)
        self.assertEqual(path.read_bytes(), self.payloads["1500"])
        self.assertEqual(path.with_suffix(".source.json").read_bytes(), before)
        self.assert_no_partial_files()

    def test_busy_runtime_lock_prevents_download(self):
        with patch.object(consistency_lora, "runtime_lock", side_effect=ValueError("busy runtime")):
            with self.assertRaisesRegex(ValueError, "busy"):
                consistency_lora.install(self.root)
        self.download.assert_not_called()
        self.assertFalse(self.path().exists())
        self.assert_no_partial_files()

    def test_symlinked_directory_weights_or_receipt_are_rejected_without_network(self):
        for kind in ("directory", "weights", "receipt"):
            with self.subTest(kind=kind):
                runtime = self.root / kind
                path = self.path(runtime=runtime)
                outside = self.root / (kind + "-outside")
                outside.mkdir()
                if kind == "directory":
                    path.parent.parent.mkdir(parents=True)
                    target = outside
                    link = path.parent
                    target_file = target / path.name
                    target_file.write_bytes(self.payloads["1500"])
                    target_file.with_suffix(".source.json").write_text(json.dumps(self.receipt()), encoding="utf-8")
                else:
                    self.seed_installed(runtime=runtime)
                    link = path if kind == "weights" else path.with_suffix(".source.json")
                    target = outside / link.name
                    target.write_bytes(link.read_bytes())
                    link.unlink()
                try:
                    link.symlink_to(target, target_is_directory=kind == "directory")
                except OSError as exc:
                    self.skipTest(f"Symbolic links are unavailable on this Windows host: {exc}")
                for operation in (consistency_lora.installed, consistency_lora.install):
                    with self.assertRaises(ValueError):
                        operation(runtime)
                self.assertTrue(link.is_symlink())
                self.download.assert_not_called()
                self.assert_no_partial_files()


if __name__ == "__main__":
    unittest.main()
