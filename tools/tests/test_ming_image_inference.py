import unittest

import numpy as np

from modules_forge.ming_image.inference import output_image


class ContractTests(unittest.TestCase):
    def test_alpha(self):
        pixels = np.array([[[0.0, 1.0, 0.0, 0.5]]])
        assert output_image(pixels, True).getpixel((0, 0)) == (0, 255, 0, 128)
        assert output_image(pixels, False).mode == "RGB"
        with self.assertRaisesRegex(ValueError, "RGBA"):
            output_image(pixels[:, :, :3], True)
        pixels[..., 3] = 1
        self.assertEqual(output_image(pixels, True).getpixel((0, 0))[3], 255)

    def test_worker_result_round_trip(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from PIL import Image

        from modules_forge.ming_image import core, inference, references, service

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            reference = Image.new("RGBA", (16, 16), (20, 30, 40, 50))
            request = core.validate_request(
                dict(
                    profile="comfy_org_w4a8",
                    mode="img2img",
                    images=[reference],
                    prompt="design api_key=dummy-secret H:/private/sample.txt",
                    transparent=True,
                    width=512,
                    height=512,
                )
            )
            references.snapshot(directory, request)
            output = directory / "public"

            def generate(value, progress, check, images):
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0].getpixel((0, 0)), (20, 30, 40, 50))
                return Image.new("RGBA", (512, 512), (8, 10, 12, 17))

            runner = inference.Runner(directory, SimpleNamespace(generate=generate))
            with patch.object(core, "model_ready", return_value=True):
                runner.run(output, request, work_dir=directory)
            self.assertEqual(service.Service.validate_result(output, directory, request)["seed"], request["seed"])
            for path in output.iterdir():
                for private in (b"dummy-secret", b"H:/private", str(directory).encode()):
                    self.assertNotIn(private, path.read_bytes())
            self.assertEqual(set(p.name for p in output.iterdir()), {"result.json", "output.png"})
            with Image.open(output / "output.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (8, 10, 12, 17))
                self.assertNotIn(str(directory), image.info["ming_image"])
                rgb = image.convert("RGB")
            rgb.save(output / "output.png")
            with self.assertRaisesRegex(core.MingImageError, "チャンネル"):
                service.Service.validate_result(output, directory, request)

    def test_public_png_metadata_omits_paths_and_credentials(self):
        from modules_forge.ming_image import core

        public = core.png_metadata(
            dict(
                prompt="poster api_key=dummy-value",
                checkpoint="private/models/test.safetensors",
                images=["reference-0.png"],
                seed=4,
            )
        )
        self.assertEqual(public["seed"], 4)
        self.assertNotIn("dummy-value", public["prompt"])
        self.assertNotIn("checkpoint", public)
        self.assertNotIn("images", public)
