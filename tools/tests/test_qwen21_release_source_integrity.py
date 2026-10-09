"""Release staging checks the saved component bytes against their completion receipt."""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from modules_forge.qwen_image21.quantized_cache import cache_path
from tools import qwen21_hub_release as release
from tools.tests import test_qwen21_hub_release as fixtures


class QwenReleaseSourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Qwen21HubReleaseTests()
        self.addCleanup(self.fixture.doCleanups)
        with redirect_stdout(io.StringIO()):
            self.fixture.setUp()

    def source_file(self, component, filename):
        return cache_path(self.fixture.source / "model", self.fixture.identities[component]) / filename

    def stage(self, output):
        with patch.object(
            release, "_identity", side_effect=lambda _root, component, _profile: self.fixture.identities[component]
        ):
            release.stage(self.fixture.source, "int8", output)

    def test_stage_rejects_same_size_tensor_mutation_after_completion(self):
        source = self.source_file("transformer", "model.safetensors")
        original = source.read_bytes()
        stat = source.stat()
        source.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(source.stat().st_size, len(original))
        output = self.fixture.root / "corrupt-release"
        with (
            patch.object(release, "_export_safetensors", wraps=release._export_safetensors) as export,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(ValueError, "Source file missing or changed"),
        ):
            self.stage(output)
        export.assert_called_once()
        self.assertFalse((output / "release_manifest.json").exists())

    def test_staged_manifest_hashes_match_the_actual_files(self):
        manifest = json.loads((self.fixture.staged / "release_manifest.json").read_text(encoding="utf-8"))
        for component in manifest["components"].values():
            for record in component["files"]:
                path = self.fixture.staged / record["path"]
                self.assertEqual(path.stat().st_size, record["size"])
                self.assertEqual(release.sha256(path), record["sha256"])

    def test_stage_rejects_same_size_config_mutation_after_completion(self):
        source = self.source_file("text_encoder", "config.json")
        original = source.read_bytes()
        stat = source.stat()
        source.write_bytes(original.replace(b"test", b"best"))
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(source.stat().st_size, len(original))
        output = self.fixture.root / "corrupt-config-release"
        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(ValueError, "Source file missing or changed"),
        ):
            self.stage(output)
        self.assertFalse((output / "release_manifest.json").exists())

    def test_stage_rejects_tensor_mutation_between_inventory_check_and_stream(self):
        original_export = release._export_safetensors

        def changed_source(source, destination, notice, expected_hash):
            data = source.read_bytes()
            source.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            return original_export(source, destination, notice, expected_hash)

        output = self.fixture.root / "changed-copy-release"
        with (
            patch.object(release, "_export_safetensors", side_effect=changed_source) as export,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(ValueError, "Source file missing or changed"),
        ):
            self.stage(output)
        export.assert_called_once()
        self.assertFalse((output / "release_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
