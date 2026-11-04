"""Model installers must not restore uv versions with compromised CA trust."""

import json
import re
import unittest
from pathlib import Path


class BootstrapUvCaSecurityTests(unittest.TestCase):
    def test_bootstrap_manifests_require_fixed_certificate_trust(self):
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "flux3_klein_manifest.json",
            "minimax_h3_runtime_manifest.json",
            "moss_image_manifest.json",
        ):
            with self.subTest(manifest=filename):
                uv = json.loads((root / filename).read_text(encoding="utf-8"))["uv"]
                match = re.search(r"/uv-(\d+)\.(\d+)\.(\d+)-", uv["url"])
                self.assertIsNotNone(match)
                version = tuple(map(int, match.groups()))
                self.assertGreaterEqual(version, (0, 13, 4))
                self.assertIn("/" + ".".join(map(str, version)) + "/", uv["license_url"])


if __name__ == "__main__":
    unittest.main()
