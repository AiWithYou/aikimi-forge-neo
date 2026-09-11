from types import SimpleNamespace
import unittest

from backend.nn.anima import Anima
from modules_forge.packages.huggingface_guess.detection import detect_unet_config


def anima_state_dict(block_count: int, key_prefix: str = "") -> dict:
    state_dict = {
        f"{key_prefix}llm_adapter.blocks.0.cross_attn.q_proj.weight": SimpleNamespace(
            shape=(1024, 1024)
        ),
        f"{key_prefix}x_embedder.proj.1.weight": SimpleNamespace(shape=(2048, 68)),
    }
    for block in range(block_count):
        state_dict[f"{key_prefix}blocks.{block}.mlp.layer1.weight"] = SimpleNamespace(
            shape=(8192, 2048)
        )
    return state_dict


class Anima29BDetectionTests(unittest.TestCase):
    @staticmethod
    def _tiny_model(block_count: int) -> Anima:
        return Anima(
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=48,
            crossattn_emb_channels=8,
            adaln_lora_dim=8,
            num_blocks=block_count,
            num_heads=4,
            mlp_ratio=1.0,
        )

    def test_detects_original_anima_as_28_blocks(self):
        config = detect_unet_config(anima_state_dict(28), "")

        self.assertEqual(config["image_model"], "anima")
        self.assertEqual(config["num_blocks"], 28)

    def test_detects_anima_29b_as_40_blocks(self):
        config = detect_unet_config(anima_state_dict(40), "")

        self.assertEqual(config["image_model"], "anima")
        self.assertEqual(config["num_blocks"], 40)
        self.assertEqual(config["model_channels"], 2048)
        self.assertEqual(config["in_channels"], 16)

    def test_detects_anima_38b_as_52_blocks(self):
        config = detect_unet_config(anima_state_dict(52), "")

        self.assertEqual(config["image_model"], "anima")
        self.assertEqual(config["num_blocks"], 52)
        self.assertEqual(config["model_channels"], 2048)
        self.assertEqual(config["in_channels"], 16)

    def test_tensor_fusion_is_scoped_to_the_52_block_model(self):
        for block_count in (28, 40):
            model = self._tiny_model(block_count)
            self.assertFalse(model.optimize_anima38)
            self.assertFalse(model.final_layer.optimized)
            self.assertTrue(all(not block.optimized for block in model.blocks))

        model = self._tiny_model(52)
        self.assertTrue(model.optimize_anima38)
        self.assertTrue(model.final_layer.optimized)
        self.assertTrue(all(block.optimized for block in model.blocks))

    def test_counts_blocks_below_a_checkpoint_prefix(self):
        prefix = "model.diffusion_model."

        config = detect_unet_config(anima_state_dict(40, prefix), prefix)

        self.assertEqual(config["num_blocks"], 40)


if __name__ == "__main__":
    unittest.main()
