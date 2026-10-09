"""Download planning never converts a user's model or fetches other tasks."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import setup_iris
from tools.setup_iris import download_plan, inference_project


class SetupTests(unittest.TestCase):
    def test_bootstrap_download_runs_in_isolated_python_without_caller_hub(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for ready in (False, True):
                with (
                    self.subTest(environment_ready=ready),
                    patch.object(sys, "argv", ["setup_iris.py", "--root", str(root), "--task", "all"]),
                    patch.object(setup_iris, "environment_ready", return_value=ready),
                    patch.object(setup_iris, "install_environment") as install,
                    patch.object(setup_iris, "execute") as execute,
                    patch.object(setup_iris, "fetch_model", side_effect=ModuleNotFoundError("caller has no Hub")),
                ):
                    setup_iris.main()
                    self.assertEqual(install.call_count, 0 if ready else 1)
                    arguments = execute.call_args.args[0]
                    self.assertEqual(arguments[0], setup_iris.python_path(root))
                    self.assertEqual(arguments[1], Path(setup_iris.__file__).resolve())
                    self.assertEqual(
                        arguments[2:], ["--download-only", "--root", root, "--precision", "int8", "--task", "all"]
                    )

    def test_download_only_fetches_selected_task_without_recursive_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "setup_iris.py",
                        "--root",
                        str(root),
                        "--task",
                        "depth",
                        "--precision",
                        "normal",
                        "--download-only",
                    ],
                ),
                patch.object(setup_iris, "install_environment") as install,
                patch.object(setup_iris, "execute") as execute,
                patch.object(setup_iris, "fetch_model") as fetch,
            ):
                setup_iris.main()
                install.assert_not_called()
                execute.assert_not_called()
                fetch.assert_called_once_with(root, "normal", "depth")

    def test_training_dependencies_are_optional_without_changing_inference_requirements(self):
        original = """[project]
dependencies = [
    "torch>=2.7.1",
    "wandb>=0.18",
    "dion @ git+https://github.com/microsoft/dion.git@fixed",
]
[project.optional-dependencies]
ui = ["gradio>=5.49,<6", "spaces"]
"""
        result = inference_project(original)
        runtime, optional = result.split("[project.optional-dependencies]", 1)
        self.assertIn('"torch>=2.7.1"', runtime)
        self.assertNotIn('"wandb', runtime)
        self.assertNotIn('"dion', runtime)
        self.assertIn("training = [", optional)
        self.assertIn('"wandb>=0.18"', optional)
        self.assertIn('"dion @ git+', optional)
        self.assertIn('ui = ["gradio>=5.49,<6", "spaces"]', optional)
        self.assertEqual(inference_project(result), result)

    def test_int8_comes_from_public_bundle(self):
        plan = download_plan("int8", "generate")
        self.assertEqual(plan["repo"], "Aikimi/iris-3b-int8")
        self.assertRegex(plan["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(plan["files"], ["config.yaml", "model.safetensors", "manifest.json"])
        self.assertNotIn("convert", plan)

    def test_normal_uses_pinned_official_task(self):
        plan = download_plan("normal", "depth")
        self.assertEqual(plan["repo"], "speridlabs/iris-3b")
        self.assertEqual(len(plan["revision"]), 40)
        self.assertEqual(
            plan["files"], ["depth/config.yaml", "depth/model.safetensors", "depth/empty_prompt.safetensors"]
        )

    def test_upscale_includes_empty_prompt_and_never_encoder(self):
        for precision in ("normal", "int8"):
            plan = download_plan(precision, "upscale")
            self.assertIn("upscaler/empty_prompt.safetensors", plan["files"])
            self.assertFalse(plan["text_encoder"])
        self.assertTrue(download_plan("normal", "generate")["text_encoder"])


if __name__ == "__main__":
    unittest.main()
