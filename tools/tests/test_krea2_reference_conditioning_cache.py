"""Exercise production processing/Krea cache methods without loading Forge models."""
# ruff: noqa: S102 - execute selected AST nodes from trusted repository source.

import ast
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class _Tensor:
    def __init__(self, value, device="cuda"):
        self.value = value
        self.device = device

    def squeeze(self, _dimension):
        return self

    def detach(self):
        return _Tensor(self.value, self.device)

    def cpu(self):
        return _Tensor(self.value, "cpu")


class _Prompt(list):
    def __init__(self, text="same prompt", *, negative=False):
        super().__init__([text])
        self.is_negative_prompt = negative


def _load_methods(path, class_name, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {
            "get_learned_conditioning",
            "get_learned_conditioning_with_image",
            "get_conditioning_cache_state",
            "restore_conditioning_cache_state",
            "cached_params",
            "get_conds_with_caching",
            "clear_prompt_cache",
        }:
            continue
        node.decorator_list = []
        node.returns = None
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            argument.annotation = None
        methods.append(node)
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), "exec"), scope)
    return {node.name: scope[node.name] for node in methods}


class Krea2ReferenceConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        self.dynamic = SimpleNamespace(ref_latents=[], last_extra_generation_params={})
        self.opts = SimpleNamespace(
            krea2_do_reference=True, CLIP_stop_at_last_layers=1, sdxl_crop_left=0, sdxl_crop_top=0, emphasis="Original"
        )
        self.model = SimpleNamespace(
            ref_latents=[],
            ini_latent=None,
            extra_generation_params={},
            forge_objects=SimpleNamespace(clip=SimpleNamespace(patcher=None)),
            set_clip_skip=Mock(),
        )
        self.model.text_processing_engine_qwen = Mock(side_effect=lambda prompt, **_kw: tuple(prompt))
        self.model.encode_vision = Mock(side_effect=lambda image: ("vision", _Tensor(image)))
        namespace = dict(
            dynamic_args=self.dynamic, opts=self.opts, memory_management=SimpleNamespace(load_model_gpu=Mock())
        )
        for name, fn in _load_methods("backend/diffusion_engine/krea.py", "Krea2", namespace).items():
            setattr(self.model, name, fn.__get__(self.model))

        class Processing:
            cached_c = [None, None, None]
            cached_uc = [None, None, None]

        self.processing_class = Processing
        namespace = dict(
            shared=SimpleNamespace(sd_model=self.model),
            opts=self.opts,
            args=SimpleNamespace(dynamic_args=self.dynamic),
            sd_models=SimpleNamespace(model_data=SimpleNamespace(forge_loading_parameters="same model")),
            uses_prompt_only_conditioning_cache=lambda _model: not self.opts.krea2_do_reference,
            StableDiffusionProcessingImg2Img=Processing,
            StableDiffusionProcessing=Processing,
            hash_tensor=lambda tensor: tensor,
        )
        for name, fn in _load_methods("modules/processing.py", "StableDiffusionProcessing", namespace).items():
            setattr(Processing, name, fn)
        self.p = Processing()
        self.p.cached_c = Processing.cached_c
        self.p.cached_uc = Processing.cached_uc
        self.p.width = self.p.height = 512
        self.p.distilled_cfg_scale = 3
        self.p.init_latent = "image A"
        self.function = Mock(side_effect=lambda model, prompt, *_args: model.get_learned_conditioning(prompt))

    def generate(self, *, image="image A", prompt=None, caches=None):
        self.p.init_latent = image
        self.model.ini_latent = image
        return self.p.get_conds_with_caching(self.function, prompt or _Prompt(), 8, caches or [self.p.cached_c], {})

    def test_positive_hit_restores_references_after_in_place_global_reset_without_reencoding(self):
        first = self.generate()
        prepared = self.dynamic.ref_latents
        self.dynamic.ref_latents.clear()
        second = self.generate()
        self.assertIs(second, first)
        self.assertEqual([ref.value for ref in self.dynamic.ref_latents], ["image A"])
        self.assertIsNot(self.dynamic.ref_latents, prepared)
        self.assertIsNone(self.model.ini_latent)
        self.model.text_processing_engine_qwen.assert_called_once()
        self.model.encode_vision.assert_called_once()
        self.assertEqual(self.function.call_count, 1)

    def test_reference_snapshot_is_cpu_only_and_restores_with_a_new_list(self):
        self.generate()
        snapshot = self.p.cached_c[3]
        self.assertIsNot(snapshot, self.dynamic.ref_latents)
        self.assertEqual([ref.device for ref in snapshot], ["cpu"])
        self.assertEqual([ref.value for ref in snapshot], ["image A"])
        self.dynamic.ref_latents.clear()
        self.assertEqual(len(snapshot), 1)
        self.generate()
        restored = self.dynamic.ref_latents
        self.assertIsNot(restored, snapshot)
        self.dynamic.ref_latents.clear()
        self.generate()
        self.assertIsNot(self.dynamic.ref_latents, restored)
        self.assertEqual(len(snapshot), 1)

    def test_negative_cache_hit_and_miss_preserve_positive_references_and_ini_latent(self):
        self.generate()
        references = self.dynamic.ref_latents
        initial = object()
        self.model.ini_latent = initial
        for _ in range(2):
            self.p.get_conds_with_caching(self.function, _Prompt("negative", negative=True), 8, [self.p.cached_uc], {})
            self.assertIs(self.dynamic.ref_latents, references)
            self.assertIs(self.model.ini_latent, initial)
        self.assertEqual(self.model.text_processing_engine_qwen.call_count, 2)
        self.model.encode_vision.assert_called_once()

    def test_clearing_prompt_cache_releases_its_reference_snapshot(self):
        self.generate()
        snapshot_reference = weakref.ref(self.p.cached_c[3][0])
        self.dynamic.ref_latents.clear()
        self.p.clear_prompt_cache()
        self.assertIsNone(snapshot_reference())
        self.assertEqual(self.p.cached_c, [None, None, None])
        self.assertEqual(self.processing_class.cached_c, [None, None, None])

    def test_changed_image_and_prompt_recompute_the_correct_reference(self):
        self.generate()
        self.dynamic.ref_latents.clear()
        self.generate(image="image B")
        self.assertEqual([ref.value for ref in self.dynamic.ref_latents], ["image B"])
        self.dynamic.ref_latents.clear()
        self.generate(image="image B", prompt=_Prompt("new prompt"))
        self.assertEqual([ref.value for ref in self.dynamic.ref_latents], ["image B"])
        self.assertEqual(self.model.text_processing_engine_qwen.call_count, 3)
        self.assertEqual(self.model.encode_vision.call_count, 3)

    def test_c_and_hr_c_restore_the_state_of_the_matching_cache(self):
        self.generate()
        hr_cache = [None, None, None]
        self.p.width = self.p.height = 1024
        self.generate(image="image B", caches=[hr_cache])
        self.dynamic.ref_latents.clear()
        self.p.width = self.p.height = 512
        self.generate(caches=[hr_cache, self.p.cached_c])
        self.assertEqual([ref.value for ref in self.dynamic.ref_latents], ["image A"])
        self.dynamic.ref_latents.clear()
        self.p.width = self.p.height = 1024
        self.generate(image="image B", caches=[self.p.cached_c, hr_cache])
        self.assertEqual([ref.value for ref in self.dynamic.ref_latents], ["image B"])
        self.assertEqual(self.function.call_count, 2)
        self.assertIsNone(self.model.ini_latent)

    def test_reference_disabled_hit_clears_stale_references(self):
        self.opts.krea2_do_reference = False
        self.generate()
        self.dynamic.ref_latents.append(_Tensor("stale"))
        self.generate()
        self.assertEqual(self.dynamic.ref_latents, [])
        self.assertIsNone(self.model.ini_latent)
        self.model.encode_vision.assert_not_called()
        self.model.text_processing_engine_qwen.assert_called_once()

    def test_model_without_state_hooks_keeps_the_existing_cache_contract(self):
        for name in ("get_conditioning_cache_state", "restore_conditioning_cache_state"):
            if hasattr(self.model, name):
                delattr(self.model, name)
        result = self.generate()
        references = self.dynamic.ref_latents
        self.assertIs(self.generate(), result)
        self.assertIs(self.dynamic.ref_latents, references)
        self.assertEqual(len(self.p.cached_c), 3)
        self.assertEqual(self.function.call_count, 1)


if __name__ == "__main__":
    unittest.main()
