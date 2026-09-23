import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qwen21_hub_release as release


class Qwen21HubReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "model").mkdir(parents=True)
        (self.source / "model" / "LICENSE").write_text("Test license\n", encoding="utf-8")
        (self.source / "model-files.json").write_text(json.dumps({"revision": "test-revision"}), encoding="utf-8")
        self.identities = {}
        for component in release.COMPONENTS:
            folder = self.source / "quantized" / "int8" / "test-key" / component
            folder.mkdir(parents=True)
            config = {"_name_or_path": r"H:\private\source"} if component == "transformer" else {"model_type": "test"}
            (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (folder / "model.safetensors").write_bytes((component + "-weights").encode())
            files = [
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": release.sha256(path),
                }
                for path in folder.iterdir()
                if path.name != "complete.json"
            ]
            identity = {
                "schema": 1,
                "revision": "test-revision",
                "component": component,
                "precision": "int8",
                "recipe": {"version": 1, "dtype": "bfloat16", "skip_modules": []},
                "versions": {"test": "1"},
                "source_inventory": [],
                "source_files": [],
            }
            self.identities[component] = identity
            (folder / "complete.json").write_text(json.dumps({"identity": identity, "files": files}), encoding="utf-8")
        self.staged = self.root / "staged"
        release.stage(self.source, "int8", self.staged)

    def _target(self, name):
        target = self.root / name
        (target / "model").mkdir(parents=True)
        shutil.copyfile(self.source / "model-files.json", target / "model-files.json")
        return target

    def _install(self, target):
        with patch(
            "modules_forge.qwen_image21.quantized_cache.component_identity",
            side_effect=lambda _model, component, _precision, **_kw: self.identities[component],
        ):
            release.install(target, self.staged, "int8")

    def test_stage_removes_local_path_and_install_rebinds_cache(self):
        config = json.loads((self.staged / "transformer" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["_name_or_path"], release.BASE_MODEL)
        self.assertNotIn("complete.json", [p.name for p in self.staged.rglob("complete.json")])
        target = self._target("target")
        self._install(target)
        self._install(target)
        for component in release.COMPONENTS:
            caches = list((target / "quantized" / "int8").glob(f"*/{component}/complete.json"))
            self.assertEqual(len(caches), 1)
            self.assertEqual(json.loads(caches[0].read_text(encoding="utf-8"))["identity"], self.identities[component])

    def test_install_rejects_corrupt_release_file(self):
        (self.staged / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self._install(self._target("corrupt-target"))
        self.assertFalse(
            list((self.root / "corrupt-target" / "quantized" / "int8").glob("*/text_encoder/complete.json"))
        )


if __name__ == "__main__":
    unittest.main()
