"""Download planning never converts a user's model or fetches other tasks."""

import fnmatch
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tools import setup_iris
from tools.setup_iris import download_plan, inference_project


class SetupTests(unittest.TestCase):
    def test_components_only_plans_contain_only_the_corresponding_shared_encoder(self):
        for precision in ("normal", "int8", "w4a8"):
            with self.subTest(precision=precision):
                plan = download_plan(precision, "generate", components_only=True)
                self.assertRegex(plan["revision"], r"^[0-9a-f]{40}$")
                self.assertNotIn("model.safetensors", plan["files"])
                self.assertNotIn("config.yaml", plan["files"])
                self.assertNotIn("manifest.json", plan["files"])
                if precision == "w4a8":
                    self.assertEqual(plan["repo"], setup_iris.W4A8_REPO)
                    self.assertEqual(plan["revision"], setup_iris.W4A8_REVISION)
                    self.assertEqual(plan["files"], ["text-encoder/" + name for name in setup_iris.W4A8_TEXT_FILES])
                    self.assertFalse(plan["text_encoder"])
                else:
                    self.assertEqual(plan["repo"], setup_iris.TEXT_REPO)
                    self.assertEqual(plan["revision"], setup_iris.TEXT_REVISION)
                    self.assertIn("*.safetensors", plan["files"])
                    self.assertTrue(plan["text_encoder"])

    def test_components_only_rejects_other_tasks_before_fetching(self):
        for task in ("depth", "upscale", "all"):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "画像生成|generate"):
                download_plan("int8", task, components_only=True)
            with self.subTest(fetch_task=task), self.assertRaisesRegex(ValueError, "画像生成|generate"):
                setup_iris.fetch_model(Path("unused"), "int8", task, components_only=True)

    def test_components_only_fetch_never_requests_or_creates_a_standard_denoiser(self):
        for precision in ("normal", "int8", "w4a8"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if precision == "w4a8":
                    payloads = {
                        "text-encoder/" + name: name.encode()
                        for name in setup_iris.W4A8_TEXT_FILES
                        if name != "manifest.json"
                    }
                    records = [
                        {
                            "path": name.removeprefix("text-encoder/"),
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                        for name, data in payloads.items()
                    ]
                    payloads["text-encoder/manifest.json"] = json.dumps({"files": records}).encode()
                    expected_repo = setup_iris.W4A8_REPO
                    expected_revision = setup_iris.W4A8_REVISION
                    encoder = root / "w4a8/text-encoder"
                else:
                    payloads = {"config.json": b"{}", "model.safetensors": b"encoder", "tokenizer.json": b"{}"}
                    expected_repo = setup_iris.TEXT_REPO
                    expected_revision = setup_iris.TEXT_REVISION
                    encoder = root / "text-encoder"
                snapshots = []

                def model_info(
                    repo,
                    *,
                    revision,
                    files_metadata,
                    expected_repo=expected_repo,
                    expected_revision=expected_revision,
                    payloads=payloads,
                ):
                    self.assertEqual((repo, revision, files_metadata), (expected_repo, expected_revision, True))
                    return SimpleNamespace(
                        siblings=[
                            SimpleNamespace(
                                rfilename=name,
                                size=len(data),
                                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()),
                            )
                            for name, data in payloads.items()
                        ]
                    )

                def snapshot(
                    repo,
                    *,
                    revision,
                    local_dir,
                    allow_patterns,
                    max_workers,
                    expected_repo=expected_repo,
                    expected_revision=expected_revision,
                    snapshots=snapshots,
                    precision=precision,
                    payloads=payloads,
                ):
                    self.assertEqual((repo, revision), (expected_repo, expected_revision))
                    snapshots.append((repo, list(allow_patterns)))
                    if precision == "w4a8":
                        self.assertTrue(all(name.startswith("text-encoder/") for name in allow_patterns))
                    for name, data in payloads.items():
                        if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
                            path = Path(local_dir) / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(data)

                hub = ModuleType("huggingface_hub")
                hub.HfApi = lambda: SimpleNamespace(model_info=model_info)
                hub.snapshot_download = snapshot
                with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(io.StringIO()):
                    setup_iris.fetch_model(root, precision, "generate", components_only=True)
                self.assertEqual(len(snapshots), 1)
                self.assertTrue((encoder / "model.safetensors").is_file())
                self.assertTrue((encoder / ("manifest.json" if precision == "w4a8" else "download.json")).is_file())
                for directory in (root / "official", root / "int8", root / "w4a8"):
                    for name in ("model.safetensors", "config.yaml", "manifest.json"):
                        self.assertFalse((directory / name).exists())

    def test_components_only_cli_installs_runtime_and_forwards_flag_to_isolated_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(
                    sys, "argv", ["setup_iris.py", "--root", str(root), "--precision", "w4a8", "--components-only"]
                ),
                patch.object(setup_iris, "environment_ready", return_value=False),
                patch.object(setup_iris, "install_environment") as install,
                patch.object(setup_iris, "execute") as execute,
                patch.object(setup_iris, "fetch_model") as fetch,
                redirect_stdout(io.StringIO()),
            ):
                setup_iris.main()
                install.assert_called_once_with(root)
                fetch.assert_not_called()
                arguments = execute.call_args.args[0]
                self.assertEqual(arguments[0], setup_iris.python_path(root))
                self.assertEqual(
                    arguments[2:],
                    [
                        "--download-only",
                        "--root",
                        root,
                        "--precision",
                        "w4a8",
                        "--task",
                        "generate",
                        "--components-only",
                    ],
                )

    def test_components_only_download_child_forwards_mode_to_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(
                    sys, "argv", ["setup_iris.py", "--root", str(root), "--components-only", "--download-only"]
                ),
                patch.object(setup_iris, "install_environment") as install,
                patch.object(setup_iris, "execute") as execute,
                patch.object(setup_iris, "fetch_model") as fetch,
                redirect_stdout(io.StringIO()),
            ):
                setup_iris.main()
                install.assert_not_called()
                execute.assert_not_called()
                fetch.assert_called_once_with(root, "int8", "generate", components_only=True)

    def test_components_only_cli_rejects_runtime_only_and_non_generation_before_actions(self):
        for extra in (["--runtime-only"], ["--task", "depth"], ["--task", "upscale"], ["--task", "all"]):
            with (
                self.subTest(extra=extra),
                patch.object(sys, "argv", ["setup_iris.py", "--components-only", *extra]),
                patch.object(setup_iris, "install_environment") as install,
                patch.object(setup_iris, "execute") as execute,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                setup_iris.main()
            self.assertEqual(error.exception.code, 2)
            install.assert_not_called()
            execute.assert_not_called()

    def test_dry_run_uses_component_plan_and_runtime_only_has_no_downloads(self):
        for extra in (["--components-only"], ["--runtime-only"]):
            output = io.StringIO()
            with patch.object(sys, "argv", ["setup_iris.py", "--dry-run", *extra]), redirect_stdout(output):
                setup_iris.main()
            result = json.loads(output.getvalue())
            if extra == ["--runtime-only"]:
                self.assertEqual(
                    result["plans"], [], "runtime-only performs no download, so dry-run must show no model plan"
                )
            else:
                self.assertEqual(result["plans"], [download_plan("int8", "generate", components_only=True)])

    def test_w4a8_fetches_packed_text_with_generation_only(self):
        plan = download_plan("w4a8", "generate")
        self.assertEqual(plan["repo"], "Aikimi/iris-3b-w4a8")
        self.assertRegex(plan["revision"], r"^[0-9a-f]{40}$", "Use the actual validated public W4A8 commit")
        self.assertIn("text-encoder/model.safetensors", plan["files"])
        self.assertIn("text-encoder/manifest.json", plan["files"])
        self.assertFalse(plan["text_encoder"], "W4A8 must not download a BF16 encoder")
        depth = download_plan("w4a8", "depth")
        self.assertEqual(depth["repo"], plan["repo"])
        self.assertFalse(any(name.startswith("text-encoder/") for name in depth["files"]))

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
                fetch.assert_called_once_with(root, "normal", "depth", components_only=False)

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
