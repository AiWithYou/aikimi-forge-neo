"""Release staging checks the saved component bytes against their completion receipt."""

from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tools import qwen21_hub_release as release
from tools.tests import test_qwen21_hub_release as fixtures


class QwenReleaseSourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Qwen21HubReleaseTests()
        self.addCleanup(self.fixture.doCleanups)
        with redirect_stdout(io.StringIO()):
            self.fixture.setUp()

    def test_stage_rejects_same_size_tensor_mutation_after_completion(self):
        source = self.fixture.source / "quantized/int8/test-key/transformer/model.safetensors"
        original = source.read_bytes()
        source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self.assertEqual(source.stat().st_size, len(original))
        output = self.fixture.root / "corrupt-release"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
            release.stage(self.fixture.source, "int8", output)
        self.assertFalse((output / "release_manifest.json").exists())

    def test_staged_manifest_hashes_match_the_actual_files(self):
        manifest = json.loads((self.fixture.staged / "release_manifest.json").read_text(encoding="utf-8"))
        for component in manifest["components"].values():
            for record in component["files"]:
                path = self.fixture.staged / record["path"]
                self.assertEqual(path.stat().st_size, record["size"])
                self.assertEqual(release.sha256(path), record["sha256"])

    def test_stage_rejects_same_size_config_mutation_after_completion(self):
        source = self.fixture.source / "quantized/int8/test-key/text_encoder/config.json"
        original = source.read_bytes()
        source.write_bytes(original.replace(b"test", b"best"))
        self.assertEqual(source.stat().st_size, len(original))
        output = self.fixture.root / "corrupt-config-release"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
            release.stage(self.fixture.source, "int8", output)
        self.assertFalse((output / "release_manifest.json").exists())

    def test_stage_rechecks_tensor_bytes_after_copy(self):
        def changed_copy(source, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            data = destination.read_bytes()
            destination.write_bytes(bytes([data[0] ^ 1]) + data[1:])

        output = self.fixture.root / "changed-copy-release"
        with (
            patch.object(release, "link_or_copy", side_effect=changed_copy),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
            self.assertRaises(ValueError),
        ):
            release.stage(self.fixture.source, "int8", output)
        self.assertFalse((output / "release_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
