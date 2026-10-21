"""The Gradio compatibility wheel must preserve upstream runtime and assets."""

import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class GradioDependencyBuildTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "vendor/gradio/build_backend.py"
        spec = importlib.util.spec_from_file_location("aikimi_gradio_build", path)
        self.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.backend)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source_files = {
            "gradio/__init__.py": b'__version__ = "6.17.3"\n',
            "gradio/templates/frontend/assets/original.js": b"const source = 'unchanged';\n",
            "gradio-6.17.3.dist-info/METADATA": (path.parent / "UPSTREAM-METADATA").read_bytes(),
            "gradio-6.17.3.dist-info/WHEEL": (path.parent / "UPSTREAM-WHEEL").read_bytes(),
            "gradio-6.17.3.dist-info/licenses/LICENSE": (path.parent / "LICENSE").read_bytes(),
            "gradio-6.17.3.dist-info/RECORD": b"old record\n",
        }
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as output:
            for name, content in self.source_files.items():
                output.writestr(name, content)
        self.source = stream.getvalue()
        manifest = dict(self.backend.UPSTREAM, sha256=hashlib.sha256(self.source).hexdigest(), size=len(self.source))
        self.enterContext(patch.object(self.backend, "UPSTREAM", manifest))

    def test_only_distribution_metadata_changes(self):
        filename = self.backend.build_from_upstream(self.source, self.directory)
        self.assertEqual(filename, "gradio-6.17.3+aikimi.1-py3-none-any.whl")
        with zipfile.ZipFile(self.directory / filename) as wheel:
            for name, content in self.source_files.items():
                if name.endswith(("/METADATA", "/RECORD")):
                    continue
                target = name.replace("gradio-6.17.3.dist-info/", "gradio-6.17.3+aikimi.1.dist-info/")
                self.assertEqual(wheel.read(target), content)
            metadata = wheel.read("gradio-6.17.3+aikimi.1.dist-info/METADATA").decode("utf-8")
            original = self.source_files["gradio-6.17.3.dist-info/METADATA"].decode("utf-8")
            self.assertEqual(
                metadata,
                original.replace("Version: 6.17.3\n", "Version: 6.17.3+aikimi.1\n").replace(
                    "Requires-Dist: pillow<13.0,>=8.0\n", "Requires-Dist: pillow<14.0,>=8.0\n"
                ),
            )
            self.assertIsNone(wheel.testzip())

    def test_corrupt_or_truncated_upstream_wheel_is_rejected(self):
        for source in (self.source[:-1], b"x" + self.source[1:]):
            with self.subTest(size=len(source)), self.assertRaises(ValueError):
                self.backend.build_from_upstream(source, self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_metadata_preparation_does_not_download_runtime_code(self):
        with patch.object(self.backend, "download_upstream", side_effect=AssertionError("No network for metadata")):
            name = self.backend.prepare_metadata_for_build_wheel(str(self.directory))
        self.assertEqual(name, "gradio-6.17.3+aikimi.1.dist-info")
        self.assertIn("pillow<14.0", (self.directory / name / "METADATA").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
