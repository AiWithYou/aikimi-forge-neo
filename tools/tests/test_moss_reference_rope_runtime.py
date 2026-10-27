"""Run with the isolated MOSS runtime Python to exercise the actual RoPE code."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ReferenceRopeTests(unittest.TestCase):
    def test_reference_positions_keep_trained_frequencies_with_and_without_cache(self):
        python = ROOT / "repositories/moss-image/venv/Scripts/python.exe"
        if not python.is_file():
            self.skipTest("isolated MOSS runtime is not installed")
        result = subprocess.run(  # noqa: S603
            [str(python), __file__, "--worker"], capture_output=True, text=True, timeout=90
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def verify_rope():
    from types import SimpleNamespace
    from unittest.mock import patch

    sys.path[:0] = [str(ROOT), str(ROOT / "repositories/moss-image/ComfyUI")]
    sys.argv = [sys.argv[0], "--cpu"]
    import torch

    from vendor import moss_image_ostris_edit as edit

    _image_ids, _pe, _text_ids = edit._image_ids, edit._pe, edit._text_ids

    dit = SimpleNamespace(rope_axes_dim=[8, 8, 8, 8])
    device = torch.device("cpu")
    live = torch.cat((_image_ids(0, 3, 3, 4, device), _text_ids(3, device)))
    refs = torch.cat((_image_ids(1, 3, 3, 4, device), _image_ids(2, 3, 4, 3, device)))
    options = {"rope_options": {"scale_x": 0.5, "scale_y": 0.7}}
    default_refs = _pe(dit, refs, {})
    cached_refs = _pe(dit, refs, options, reference_start=0)
    joint = _pe(dit, torch.cat((live, refs)), options, reference_start=len(live))
    default_live = _pe(dit, live, {})
    scaled_live = _pe(dit, live, options)
    assert torch.equal(cached_refs, default_refs)
    assert torch.equal(joint[:, len(live) :], default_refs)
    assert torch.equal(joint[:, : len(live)], scaled_live)
    assert not torch.equal(scaled_live, default_live)
    assert torch.equal(
        _pe(dit, torch.cat((live, refs)), {}, reference_start=len(live)), _pe(dit, torch.cat((live, refs)), {})
    )
    # Check the actual joint and precomputed-reference paths use the intended split.
    x = torch.zeros(1, 128, 3, 4)
    ref_latents = [torch.zeros(1, 128, 3, 4), torch.zeros(1, 128, 4, 3)]
    timestep = torch.zeros(1)
    context = torch.zeros(1, 3, 8)
    for cached, expected in ((False, len(live)), (True, 0)):
        with patch.object(edit, "_pe", side_effect=StopIteration) as build_pe:
            try:
                if cached:
                    edit._precompute_ref_kv(dit, x, timestep, 3, ref_latents, options)
                else:
                    edit._forward_with_refs(dit, x, timestep, context, ref_latents, options)
            except StopIteration:
                pass
            else:
                raise AssertionError("RoPE calculation was not reached")
            assert build_pe.call_args.kwargs["reference_start"] == expected

    bias = torch.tensor([[0.0, -0.7, -0.7, 0.0]])
    mask = edit._attention_mask(bias, 3, 5, torch.float32, device)
    assert mask.shape == (1, 1, 1, 12)
    assert mask.stride(-2) % 8 == 0
    assert torch.equal(mask[0, 0, 0, 3:7], bias[0])
    assert torch.count_nonzero(mask[0, 0, 0, :3]) == 0
    assert torch.count_nonzero(mask[0, 0, 0, 7:]) == 0

    class Patcher:
        def __init__(self):
            self.base = SimpleNamespace(extra_conds=lambda **kwargs: {}, extra_conds_shapes=lambda **kwargs: {},
                                        process_latent_in=lambda x: x)
            self.model = self.base
            self.dit = SimpleNamespace(latent_channels=128, forward=lambda *args, **kwargs: None)
            self.patches = {}

        def clone(self):
            return self

        def get_model_object(self, name):
            return self.base if name == "" else self.dit

        def add_object_patch(self, name, value):
            self.patches[name] = value

    patched = Patcher()
    edit.MOSSImageOstrisEditModelPatch().patch(patched)
    conditions = patched.patches["extra_conds"](context_attention_bias=bias, reference_latents=ref_latents)
    assert torch.equal(conditions["context_attention_bias"].cond, bias)
    assert len(conditions["ref_latents"].cond) == 2

    # The text refiner must use the same operation when the attention bias is zero.
    import comfy.model_management
    from comfy.ldm.moss_image.model import TextEmbedder

    previous_training = comfy.model_management.in_training
    comfy.model_management.in_training = True
    try:
        torch.manual_seed(11)
        tiny = SimpleNamespace(rope_axes_dim=[8, 8, 8, 8],
                               text_embedder=TextEmbedder(32, 32, 64, 1, operations=torch.nn))
        text = torch.randn(1, 4, 32)
        pe = _pe(tiny, _text_ids(4, device), {})
        original = edit._text_embedder(tiny, text, pe, {}, None)
        neutral = edit._text_embedder(tiny, text, pe, {}, torch.zeros(1, 4))
        weighted = edit._text_embedder(tiny, text, pe, {}, bias)
        torch.testing.assert_close(original, neutral, rtol=1e-5, atol=1e-6)
        assert not torch.allclose(neutral, weighted)
    finally:
        comfy.model_management.in_training = previous_training


if __name__ == "__main__":
    if "--worker" in sys.argv:
        verify_rope()
    else:
        unittest.main()
