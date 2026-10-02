from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from modules import launch_utils


class LaunchModelPathTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory(prefix="Neo モデル "))
        self.root = Path(temporary)
        self.config = self.root / "設定 置場" / "extra_model_paths.yaml"
        self.config.parent.mkdir()
        self.enterContext(patch.object(sys, "argv", ["launch.py"]))

    def configure(self, data):
        self.config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        unrelated = self.root / "another-working-directory"
        unrelated.mkdir(exist_ok=True)
        with patch.object(os, "getcwd", return_value=str(unrelated)):
            launch_utils.configure_comfy_yaml(self.config)

    def test_standard_multiline_paths_are_registered_separately(self):
        models = self.root / "共有 モデル"
        paths = [models / "画像 本体", models / "別の 本体"]
        for path in paths:
            path.mkdir(parents=True)

        self.configure({"comfy": {"base_path": str(models), "diffusion_models": "画像 本体\n別の 本体\n"}})

        self.assertEqual(sys.argv, ["launch.py", "--ckpt-dirs", str(paths[0]), "--ckpt-dirs", str(paths[1])])

    def test_relative_paths_are_based_on_the_yaml_location(self):
        for base in (None, "../共有 モデル"):
            with self.subTest(base=base):
                sys.argv[:] = ["launch.py"]
                models = self.config.parent if base is None else self.root / "共有 モデル"
                path = models / "画像 本体"
                path.mkdir(parents=True, exist_ok=True)
                data = {"checkpoints": "画像 本体"}
                if base is not None:
                    data["base_path"] = base

                self.configure({"comfy": data})

                self.assertEqual(sys.argv, ["launch.py", "--ckpt-dirs", str(path)])

    def test_environment_variable_base_is_expanded(self):
        models = self.root / "共有 モデル"
        path = models / "loras"
        path.mkdir(parents=True)
        with patch.dict(os.environ, {"FORGE_TEST_MODEL_HOME": str(models)}):
            self.configure({"comfy": {"base_path": "${FORGE_TEST_MODEL_HOME}", "loras": "loras"}})

        self.assertEqual(sys.argv, ["launch.py", "--lora-dirs", str(path)])

    def test_empty_optional_config_section_does_not_block_other_paths(self):
        path = self.config.parent / "vae"
        path.mkdir()

        self.configure({"disabled": None, "comfy": {"base_path": str(self.config.parent), "vae": "vae"}})

        self.assertEqual(sys.argv, ["launch.py", "--vae-dirs", str(path)])


if __name__ == "__main__":
    unittest.main()
