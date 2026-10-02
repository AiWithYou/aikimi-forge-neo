"""CPU Extras routes preserve requests and handle cancelled output safely."""

import ast
import dataclasses
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from tools.tests.test_gpu_ownership import load_function

ROOT = Path(__file__).resolve().parents[2]


def load_classes(path, names, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return SimpleNamespace(**{name: namespace[name] for name in names})


def extras_models():
    return load_classes(
        "modules/api/models.py",
        {
            "ExtrasBaseRequest",
            "ExtraBaseResponse",
            "ExtrasSingleImageRequest",
            "ExtrasSingleImageResponse",
            "FileData",
            "ExtrasBatchImagesRequest",
            "ExtrasBatchImagesResponse",
        },
        {"BaseModel": BaseModel, "Field": Field, "Literal": Literal, "sd_upscalers": []},
    )


class ExtrasApiTests(unittest.TestCase):
    def setUp(self):
        self.models = extras_models()
        self.api = SimpleNamespace(queue_lock=Lock())
        self.state = Mock(interrupted=False, stopping_generation=False, skipped=False)
        self.image = Image.new("RGB", (2, 2), "red")
        self.addCleanup(self.image.close)
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        pp = load_classes(
            "modules/scripts_postprocessing.py",
            {"PostprocessedImage", "PostprocessedImageSharedInfo"},
            {"dataclasses": dataclasses, "__name__": __name__},
        )
        self.runner = Mock()
        self.runner.create_args_for_run.return_value = []
        self.images = Mock()
        self.images.read_info_from_image.return_value = ("", {})
        self.images.fix_image.side_effect = lambda image: image
        self.images.save_image.side_effect = AssertionError("API must not save files")
        self.devices = Mock()
        namespace = {
            "contextmanager": contextmanager,
            "devices": self.devices,
            "shared": SimpleNamespace(state=self.state),
            "Image": Image,
            "images": self.images,
            "scripts": SimpleNamespace(scripts_postproc=self.runner),
            "scripts_postprocessing": pp,
            "opts": SimpleNamespace(
                outdir_samples="",
                outdir_extras_samples=self.directory,
                use_original_name_batch=False,
                enable_pnginfo=False,
            ),
            "result_html": lambda _info: "<p>result</p>",
            "infotext_utils": SimpleNamespace(quote=str),
        }
        for name in ("_extras_job", "run_postprocessing", "run_extras"):
            load_function("modules/postprocessing.py", name, namespace)
        api_namespace = {
            "models": self.models,
            "postprocessing": SimpleNamespace(run_extras=namespace["run_extras"]),
            "decode_base64_to_image": lambda _data: self.image,
            "encode_pil_to_base64": lambda _image: "encoded-output",
        }
        for name in ("setUpscalers", "extras_single_image_api", "extras_batch_images_api"):
            load_function("modules/api/api.py", name, api_namespace)
        self.single = api_namespace["extras_single_image_api"]
        self.batch = api_namespace["extras_batch_images_api"]

    def assert_finalized(self):
        self.assertFalse(self.api.queue_lock.locked())
        self.state.end.assert_called_once()
        self.assertEqual(self.devices.torch_gc.call_count, 2)
        self.images.save_image.assert_not_called()
        self.assertEqual(list(Path(self.directory).iterdir()), [])

    def test_interrupted_or_skipped_single_image_returns_nullable_output(self):
        for flag in ("interrupted", "skipped"):
            with self.subTest(flag=flag):
                self.state.reset_mock()
                self.devices.reset_mock()
                self.state.interrupted = False
                self.state.skipped = False
                self.runner.run.side_effect = lambda _pp, _args, flag=flag: setattr(self.state, flag, True)
                request = self.models.ExtrasSingleImageRequest(image="encoded-input")
                response = self.single(self.api, request)
                self.assertIsNone(response.image)
                self.assertEqual(response.html_info, "<p>result</p>")
                self.assert_finalized()

    def test_single_image_keeps_request_reusable_after_success(self):
        request = self.models.ExtrasSingleImageRequest(image="encoded-input", upscaler_1="fixture")
        original = request.model_dump()
        for _ in range(2):
            self.state.reset_mock()
            self.devices.reset_mock()
            response = self.single(self.api, request)
            self.assertEqual(response.image, "encoded-output")
            self.assertEqual(request.model_dump(), original)
            self.assert_finalized()

    def test_batch_images_keep_request_reusable_after_success(self):
        request = self.models.ExtrasBatchImagesRequest(imageList=[{"data": "encoded-input", "name": "image.png"}])
        original = request.model_dump()
        for _ in range(2):
            self.state.reset_mock()
            self.devices.reset_mock()
            response = self.batch(self.api, request)
            self.assertEqual(response.images, ["encoded-output"])
            self.assertEqual(request.model_dump(), original)
            self.assert_finalized()

    def test_processing_failure_releases_lock_and_preserves_request(self):
        self.runner.run.side_effect = RuntimeError("postprocessing failed")
        request = self.models.ExtrasSingleImageRequest(image="encoded-input")
        original = request.model_dump()
        with self.assertRaisesRegex(RuntimeError, "postprocessing failed"):
            self.single(self.api, request)
        self.assert_finalized()
        self.assertEqual(request.model_dump(), original)


class ExtrasScaleValidationTests(unittest.TestCase):
    def setUp(self):
        self.models = extras_models()
        self.firstpass = load_function(
            "scripts/postprocessing_upscale.py",
            "process_firstpass",
            {
                "scripts_postprocessing": SimpleNamespace(PostprocessedImage=object),
                "limit_size_by_one_dimension": lambda width, height, _limit: (width, height),
            },
        )
        self.calls = []
        app = FastAPI()
        models = self.models

        @app.post("/extras")
        def extras(request: models.ExtrasSingleImageRequest):
            self.calls.append(request.upscaling_resize)
            pp = SimpleNamespace(image=SimpleNamespace(width=2, height=3), shared=SimpleNamespace())
            self.firstpass(None, pp, upscale_mode=request.resize_mode, upscale_by=request.upscaling_resize)
            return {"width": pp.shared.target_width, "height": pp.shared.target_height}

        self.client = self.enterContext(TestClient(app, raise_server_exceptions=False))

    def test_nonfinite_or_nonpositive_scale_is_rejected_before_processing(self):
        for scale in ("Infinity", "-Infinity", "NaN", 0, -1):
            with self.subTest(scale=scale):
                response = self.client.post("/extras", json={"upscaling_resize": scale})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.calls, [])

    def test_nonfinite_numeric_scale_is_rejected_by_the_request_model(self):
        # Nonfinite numeric literals cannot be represented in standard JSON.
        for scale in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValidationError):
                self.models.ExtrasSingleImageRequest(upscaling_resize=scale)

    def test_positive_finite_scale_retains_existing_coercion_and_dimensions(self):
        for scale in (0.5, 2, "1.5"):
            with self.subTest(scale=scale):
                response = self.client.post("/extras", json={"upscaling_resize": scale})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"width": int(2 * float(scale)), "height": int(3 * float(scale))})


if __name__ == "__main__":
    unittest.main()
