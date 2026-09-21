"""Real CPU model forwards verify Krea2 hook scope and exact OFF recovery."""

import unittest

import torch

from backend.args import dynamic_args
from backend.nn.krea import SingleStreamDiT


class Krea2SparseModelHookTests(unittest.TestCase):
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
