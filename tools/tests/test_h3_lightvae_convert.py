"""Native VAE exports preserve H3 attention ordering and encoder topology."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path


class NativeExportTests(unittest.TestCase):
    def test_diffusers_encoder_names_reach_the_native_pruned_block_detector(self):
        convert = importlib.import_module("tools.h3_lightvae_convert")
        tensor = object()
        state = convert.native_state_dict({
            "encoder.down_blocks.5.resnets.0.conv_shortcut.weight": tensor,
            "encoder.down_blocks.0.resnets.0.conv1.weight": tensor,
            "encoder.down_blocks.0.downsamplers.0.conv.weight": tensor,
            "encoder.conv_in.weight": tensor,
        }, "encoder")
        self.assertIs(state["encoder.down.5.block.0.nin_shortcut.weight"], tensor)
        self.assertIs(state["encoder.down.0.block.0.conv1.weight"], tensor)
        self.assertIs(state["encoder.down.0.downsample.conv.weight"], tensor)

    def test_decoder_packs_qkv_per_head_and_swaps_the_feedforward_gate(self):
        import torch

        convert = importlib.import_module("tools.h3_lightvae_convert")
        q, k, v = [torch.arange(4) + offset for offset in (0, 10, 20)]
        state = convert.native_state_dict({
            "decoder.transformer_blocks.0.attn.to_q.bias": q,
            "decoder.transformer_blocks.0.attn.to_k.bias": k,
            "decoder.transformer_blocks.0.attn.to_v.bias": v,
            "decoder.transformer_blocks.0.ff.net.0.proj.bias": torch.arange(8),
            "decoder.proj_in.weight": torch.ones(2, 2),
            "decoder.register_tokens": torch.ones(1, 4, 2),
        }, "decoder", heads=2)
        self.assertEqual(state["decoder.transformer_blocks.0.attn.to_qkv.bias"].tolist(),
                         [0, 1, 10, 11, 20, 21, 2, 3, 12, 13, 22, 23])
        self.assertEqual(state["decoder.transformer_blocks.0.ff.w1.bias"].tolist(), [4, 5, 6, 7, 0, 1, 2, 3])
        self.assertIn("decoder.x_embedder.weight", state)
        self.assertEqual(state["decoder.mask_token"].shape, (1, 1, 2))

    def test_export_records_normalization_and_matches_fixed_integrity_before_promotion(self):
        import torch
        from safetensors.torch import load_file, save_file

        convert = importlib.import_module("tools.h3_lightvae_convert")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, expected, target = (root / name for name in ("source.safetensors", "expected.safetensors", "native.safetensors"))
            weights = {"encoder.down_blocks.5.resnets.0.conv_shortcut.weight": torch.ones(2, 2)}
            save_file(weights, source, metadata={"architecture": json.dumps({"teacher_config": {
                "latents_mean": [0.25], "latents_std": [1.5]}})})
            digest = convert.sha256(source)
            save_file({"encoder.down.5.block.0.nin_shortcut.weight": torch.ones(2, 2),
                       "latents_mean": torch.tensor([0.25]), "latents_std": torch.tensor([1.5])},
                      expected, metadata={"source_sha256": digest})
            args = (source, target, "encoder", digest, convert.sha256(expected), expected.stat().st_size)
            convert.export_native(*args)
            self.assertEqual(target.read_bytes(), expected.read_bytes())
            self.assertEqual(load_file(target)["latents_mean"].tolist(), [0.25])
            convert.export_native(*args)  # A verified existing export is reusable.
            target.unlink()
            with self.assertRaisesRegex(ValueError, "export size or SHA-256 mismatch"):
                convert.export_native(source, target, "encoder", digest, "0" * 64, expected.stat().st_size)
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob("*.tmp")))
            with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                convert.export_native(source, target, "encoder", "0" * 64, "0" * 64, 1)


if __name__ == "__main__":
    unittest.main()
