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


if __name__ == "__main__":
    if "--worker" in sys.argv:
        verify_rope()
    else:
        unittest.main()
