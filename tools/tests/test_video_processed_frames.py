"""Video output must contain the final postprocessed images shown in the UI."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


class VideoProcessedFrameTests(unittest.TestCase):
    def test_video_uses_final_callback_results_and_normalizes_rgb(self):
        path = Path(__file__).resolve().parents[2] / "modules/processing.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                    and [getattr(x, "id", "") for x in n.target.elts] == ["i", "x_sample"])

        def recolor(_p, pp):
            pp.image = Image.new("RGB", pp.image.size, "blue")

        def final_color(_p, pp):
            pp.image = Image.new("RGBA", pp.image.size, (0, 255, 0, 85) if pp.index == 0 else (255, 255, 0, 85))

        scripts = SimpleNamespace(
            PostprocessImageArgs=lambda image, index: SimpleNamespace(image=image, index=index),
            PostProcessMaskOverlayArgs=lambda i, mask, overlay: SimpleNamespace(mask_for_overlay=mask, overlay_image=overlay),
        )
        for video in (True, False):
            with self.subTest(video=video):
                p = SimpleNamespace(
                    restore_faces=False, iteration=0, batch_size=2, color_corrections=None, paste_to=None, pixels_after_sampling=[],
                    scripts=SimpleNamespace(postprocess_image=recolor, postprocess_maskoverlay=lambda *args: None,
                                            postprocess_image_after_composite=final_color),
                )
                scope = dict(
                    np=np, Image=Image, p=p, scripts=scripts, shared=SimpleNamespace(opts=SimpleNamespace(overlay_inpaint=False)),
                    opts=SimpleNamespace(enable_pnginfo=False), x_samples_ddim=[torch.zeros(3, 2, 2), torch.ones(3, 2, 2)],
                    _is_video=video, frames=[], save_samples=False, apply_overlay=lambda image, *args: (image, image),
                    infotext=lambda index: "fixture", infotexts=[], output_images=[],
                )
                exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
                self.assertEqual(len(scope["output_images"]), 2)
                if video:
                    self.assertEqual(len(scope["frames"]), 2)
                    for frame, image in zip(scope["frames"], scope["output_images"], strict=True):
                        np.testing.assert_array_equal(frame, np.asarray(image.convert("RGB")))
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertEqual(frame.shape, (2, 2, 3))
                else:
                    self.assertEqual(scope["frames"], [])


if __name__ == "__main__":
    unittest.main()
