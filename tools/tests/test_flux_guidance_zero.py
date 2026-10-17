"""Exercise prompt copying and FLUX guidance without loading model weights."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_krea2_reference_conditioning_cache import _load_methods

ROOT = Path(__file__).resolve().parents[2]


class FluxGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "modules/prompt_parser.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SdConditioning")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        cls.Prompt = namespace["SdConditioning"]

    def test_explicit_zero_survives_prompt_copy_and_overrides_inherited_scale(self):
        original = self.Prompt(["image"], distilled_cfg_scale=4.0)
        zero = self.Prompt(["image"], copy_from=original, distilled_cfg_scale=0.0)
        self.assertEqual(zero.distilled_cfg_scale, 0.0)
        self.assertEqual(self.Prompt(zero).distilled_cfg_scale, 0.0)
        self.assertEqual(self.Prompt(original).distilled_cfg_scale, 4.0)

    def test_flux_guidance_preserves_zero_and_defaults_only_when_missing(self):
        namespace = dict(
            torch=torch,
            memory_management=SimpleNamespace(load_model_gpu=Mock(), logger=Mock()),
            dynamic_args=SimpleNamespace(kontext=False, ref_latents=[]),
        )
        condition = _load_methods("backend/diffusion_engine/flux.py", "Flux", namespace)["get_learned_conditioning"]
        model = SimpleNamespace(
            forge_objects=SimpleNamespace(clip=SimpleNamespace(patcher=None)),
            use_distilled_cfg_scale=True,
            text_processing_engine_l=lambda _p: (None, torch.zeros(2, 1)),
            text_processing_engine_t5=lambda _p: torch.zeros(2, 1, 1),
        )
        for scale, expected in ((0.0, 0.0), (None, 3.0), (4.5, 4.5)):
            with self.subTest(scale=scale):
                prompt = self.Prompt(["a", "b"])
                prompt.distilled_cfg_scale = scale
                result = condition(model, prompt)
                self.assertEqual(result["guidance"].tolist(), [expected, expected])


if __name__ == "__main__":
    unittest.main()
