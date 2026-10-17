"""Grid metadata must keep first-image conditions when comments are saved."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from tools.tests.test_infotext_prompt_restore import RestoreFixture


class GridCommentInfotextTests(unittest.TestCase):
    def test_grid_and_sample_keep_their_own_prompts_and_seeds(self):
        fixture = RestoreFixture()
        fixture.opts.save_prompt_comments = True
        p = SimpleNamespace(
            batch_size=1, all_prompts=["first", "second"], all_negative_prompts=["bad first", "bad second"],
            _all_prompts_c=["first # first note", "second # second note"],
            _all_negative_prompts_c=["bad first # first negative", "bad second # second negative"],
            all_seeds=[101, 202], all_subseeds=[303, 404], seeds=[202], subseeds=[404],
            prompts=["second"], negative_prompts=["bad second"], main_prompt="first", main_negative_prompt="bad first",
            get_token_merging_ratio=lambda **_kwargs: 0, steps=20, sampler_name="Euler", scheduler="Simple", cfg_scale=7,
            sd_model=SimpleNamespace(use_distilled_cfg_scale=False, use_shift=False, is_sd1=False),
            restore_faces=False, width=512, height=384, subseed_strength=0.5, seed_resize_from_w=0, seed_resize_from_h=0,
            extra_generation_params={}, is_using_inpainting_conditioning=False,
        )
        path = Path(__file__).resolve().parents[2] / "modules/processing.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "infotext" and "use_main_prompt" in {a.arg for a in n.args.args})
        scope = {"p": p, "n": 1, "opts": fixture.opts, "create_infotext": fixture.create}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
        for save_comments, main, prompt, negative, seed, variation in (
            (True, True, p._all_prompts_c[0], p._all_negative_prompts_c[0], "101", "303"),
            (True, False, p._all_prompts_c[1], p._all_negative_prompts_c[1], "202", "404"),
            (False, True, "first", "bad first", "101", "303"),
            (False, False, "second", "bad second", "202", "404"),
        ):
            with self.subTest(grid=main, save_comments=save_comments):
                fixture.opts.save_prompt_comments = save_comments
                parsed = fixture.info.parse_generation_parameters(scope["infotext"](use_main_prompt=main))
                self.assertEqual(parsed["Prompt"], prompt)
                self.assertEqual(parsed["Negative prompt"], negative)
                self.assertEqual(parsed["Seed"], seed)
                self.assertEqual(parsed["Variation seed"], variation)

        video_call = next(
            n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "save_video"
        )
        for save_comments in (False, True):
            with self.subTest(video=True, save_comments=save_comments):
                fixture.opts.save_prompt_comments = save_comments
                images = SimpleNamespace(save_video=Mock(return_value="fixture.mp4"))
                scope.update(images=images, frames=[])
                exec(compile(ast.Module(body=[video_call], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
                parsed = fixture.info.parse_generation_parameters(images.save_video.call_args.kwargs["info"])
                self.assertEqual(parsed["Prompt"], p._all_prompts_c[1] if save_comments else "second")
                self.assertEqual(parsed["Seed"], "202")
                self.assertEqual(parsed["Variation seed"], "404")


if __name__ == "__main__":
    unittest.main()
