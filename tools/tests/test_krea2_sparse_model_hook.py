"""Real CPU model forwards verify Krea2 hook scope and exact OFF recovery."""

import copy
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend.args import dynamic_args
from backend.nn.krea import SingleStreamDiT
from modules_forge.jev_sparse import common, krea2


class Krea2SparseModelHookTests(unittest.TestCase):
    def test_attached_step_clock_groups_cfg_forwards_in_real_model(self):
        model = SingleStreamDiT(
            features=256,
            txtdim=256,
            heads=2,
            kvheads=1,
            layers=2,
            txtlayers=1,
            txtheads=2,
            txtkvheads=2,
            multiplier=2,
        )
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.01)

        class Patcher:
            load_device = torch.device("cpu")

            def __init__(self):
                self.model_options = {}

            def clone(self):
                result = Patcher()
                result.model_options = copy.deepcopy(self.model_options)
                return result

            def get_model_object(self, _name):
                return model

            def set_model_unet_function_wrapper(self, wrapper):
                self.model_options["model_function_wrapper"] = wrapper

        original_refs = dynamic_args.ref_latents
        dynamic_args.ref_latents = []
        try:
            for cadence, expected in (("once", 1), ("interval", 2), ("step", 3)):
                with self.subTest(cadence=cadence), TemporaryDirectory() as directory:
                    client = SimpleNamespace(calls=0, wait_seconds=0, last_diagnostics={})

                    def decide(_state, allowed, _fallback, client=client):
                        client.calls += 1
                        return dict.fromkeys(allowed, 100.0)

                    client.decide = decide
                    options = krea2.Options(mode="jev", min_tokens=64, decision_cadence=cadence, update_interval=2)
                    run = krea2.KreaRun(options, 2, common.RunLog(directory, "clock", {}), client)
                    original = Patcher()
                    ck = SimpleNamespace(sol_attn_is_available=lambda _device: True)
                    with patch.object(krea2.importlib, "import_module", return_value=ck):
                        patched, _ = krea2.attach(original, options, directory, run=run)
                    clock = patched.model_options["conditioning_modifiers"][-1]
                    wrapper = patched.model_options["model_function_wrapper"]
                    context = torch.randn(1, 5, 1, 256)
                    for _step in range(4):
                        x, timestep = torch.randn(1, 16, 1, 16, 16), torch.ones(1)
                        clock(model, x, timestep, None, None, 1.0, patched.model_options, 7)
                        for _cfg_branch in range(2):
                            with torch.inference_mode():
                                result = wrapper(
                                    model,
                                    {
                                        "input": x.clone(),
                                        "timestep": timestep,
                                        "c": {
                                            "context": context.clone(),
                                            "transformer_options": patched.model_options["transformer_options"],
                                        },
                                    },
                                )
                            self.assertTrue(torch.isfinite(result).all())
                    self.assertEqual(client.calls, expected)
                    self.assertEqual(run.evaluation + 1, 8)
                    self.assertEqual(original.model_options, {})
                    run.close()
        finally:
            dynamic_args.ref_latents = original_refs

    def test_hook_only_receives_generation_blocks_and_preserves_dense_result(self):
        original_refs = dynamic_args.ref_latents
        dynamic_args.ref_latents = []
        try:
            torch.manual_seed(7)
            model = SingleStreamDiT(
                features=256,
                txtdim=256,
                heads=2,
                kvheads=1,
                layers=2,
                txtlayers=2,
                txtheads=2,
                txtkvheads=2,
                multiplier=2,
            )
            for param in model.parameters():
                torch.nn.init.normal_(param, std=0.01)
            x = torch.randn(1, 16, 1, 8, 8)
            context = torch.randn(1, 5, 2, 256)
            timestep = torch.ones(1)
            seen = []

            def hook(q, k, v, heads, mask, options, dense):
                seen.append(
                    (
                        options["krea2_block_index"],
                        options["krea2_text_tokens"],
                        options["krea2_reference_tokens"],
                        options["krea2_image_tokens"],
                        tuple(q.shape),
                        tuple(k.shape),
                    )
                )
                return dense(q, k, v, heads, mask=mask, skip_reshape=True, transformer_options=options)

            with torch.inference_mode():
                baseline = model(x.clone(), timestep, context.clone())
                routed = model(
                    x.clone(), timestep, context.clone(), transformer_options={"krea2_attention_override": hook}
                )
                restored = model(x.clone(), timestep, context.clone())
            self.assertTrue(torch.equal(baseline, routed))
            self.assertTrue(torch.equal(baseline, restored))
            self.assertEqual([0, 1], [r[0] for r in seen])
            self.assertEqual([(5, 0, 16)] * 2, [r[1:4] for r in seen])
            self.assertTrue(all(r[4] == r[5] == (1, 2, 21, 128) for r in seen))
            dynamic_args.ref_latents = [torch.randn(1, 16, 8, 8)]
            seen.clear()
            with torch.inference_mode():
                model(x.clone(), timestep, context.clone(), transformer_options={"krea2_attention_override": hook})
            self.assertEqual([(5, 16, 16)] * 2, [r[1:4] for r in seen])
        finally:
            dynamic_args.ref_latents = original_refs


if __name__ == "__main__":
    unittest.main()
