import unittest

from modules_forge.anima_lora import (
    ANIMA_29B_INSERTED_BLOCKS,
    ANIMA_29B_TO_BASE,
    ANIMA_29B_TO_38B,
    ANIMA_38B_INSERTED_BLOCKS,
    ANIMA_38B_TO_29B,
    ANIMA_BASE_TO_29B,
    anima_lora_block_indices,
    anima_lora_block_indices_from_keys,
    convert_anima_lora_layout,
    detect_anima_lora_block_count,
    detect_anima_lora_block_count_from_keys,
    has_anima_lora_signature,
)


def kohya_lora(blocks: int) -> tuple[dict[str, object], list[object]]:
    values = [object() for _ in range(blocks)]
    lora = {
        f"lora_unet_blocks_{block}_self_attn_q_proj.lora_A.weight": value
        for block, value in enumerate(values)
    }
    lora["lora_te_layers_0_self_attn_q_proj.lora_A.weight"] = object()
    return lora, values


class AnimaLoraLayoutTests(unittest.TestCase):
    def test_published_expansion_mapping_is_complete(self):
        self.assertEqual(len(ANIMA_BASE_TO_29B), 28)
        self.assertEqual(len(ANIMA_29B_TO_BASE), 40)
        self.assertEqual(set(ANIMA_29B_INSERTED_BLOCKS), set(range(40)) - set(ANIMA_BASE_TO_29B))
        self.assertEqual(
            ANIMA_29B_TO_BASE,
            (0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 27),
        )
        self.assertEqual(len(ANIMA_29B_TO_38B), 40)
        self.assertEqual(len(ANIMA_38B_TO_29B), 52)
        self.assertEqual(
            set(ANIMA_38B_INSERTED_BLOCKS),
            set(range(52)) - set(ANIMA_29B_TO_38B),
        )
        self.assertEqual(ANIMA_38B_TO_29B[2:5], (2, 2, 3))
        self.assertEqual(ANIMA_38B_TO_29B[46:49], (35, 35, 36))

    def test_detects_complete_layouts_but_not_sparse_layouts(self):
        base, _ = kohya_lora(28)
        expanded, _ = kohya_lora(40)
        pro52, _ = kohya_lora(52)
        sparse = {key: value for key, value in base.items() if "blocks_7_" not in key}

        self.assertEqual(detect_anima_lora_block_count(base), 28)
        self.assertEqual(detect_anima_lora_block_count(expanded), 40)
        self.assertEqual(detect_anima_lora_block_count(pro52), 52)
        self.assertIsNone(detect_anima_lora_block_count(sparse))

    def test_detects_layout_from_safetensors_header_keys(self):
        lora, _ = kohya_lora(28)

        self.assertEqual(
            anima_lora_block_indices_from_keys(lora.keys()),
            tuple(range(28)),
        )
        self.assertEqual(
            detect_anima_lora_block_count_from_keys(lora.keys()),
            28,
        )

    def test_anima_signature_rejects_unrelated_block_models(self):
        anima_keys = {
            "diffusion_model.blocks.0.adaln_modulation_mlp.1.lora_A.weight",
            "diffusion_model.blocks.0.cross_attn.q_proj.lora_A.weight",
            "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight",
            "diffusion_model.blocks.0.mlp.layer1.lora_A.weight",
        }
        wan_keys = {
            "diffusion_model.blocks.0.cross_attn.q.lora_down.weight",
            "diffusion_model.blocks.0.ffn.0.lora_down.weight",
        }

        self.assertTrue(has_anima_lora_signature(anima_keys))
        self.assertFalse(has_anima_lora_signature(wan_keys))
        self.assertTrue(
            has_anima_lora_signature(
                wan_keys,
                {"ss_network_module": "networks.lora_anima"},
            )
        )

    def test_expands_28_block_kohya_lora_without_copying_tensors(self):
        lora, source_values = kohya_lora(28)
        text_encoder_value = lora["lora_te_layers_0_self_attn_q_proj.lora_A.weight"]

        converted, report = convert_anima_lora_layout(lora, 40)

        self.assertEqual(anima_lora_block_indices(converted), tuple(range(40)))
        self.assertEqual(report.direction, "28_to_40")
        self.assertEqual(report.duplicated_entries, 12)
        self.assertIs(converted["lora_unet_blocks_1_self_attn_q_proj.lora_A.weight"], source_values[1])
        self.assertIs(converted["lora_unet_blocks_2_self_attn_q_proj.lora_A.weight"], source_values[1])
        self.assertIs(converted["lora_unet_blocks_3_self_attn_q_proj.lora_A.weight"], source_values[2])
        self.assertIs(converted["lora_unet_blocks_10_self_attn_q_proj.lora_A.weight"], source_values[7])
        self.assertIs(converted["lora_te_layers_0_self_attn_q_proj.lora_A.weight"], text_encoder_value)
        self.assertEqual(anima_lora_block_indices(lora), tuple(range(28)), "input must not be mutated")

    def test_expands_generic_and_peft_style_keys(self):
        lora = {
            f"base_model.model.diffusion_model.blocks.{block}.mlp.layer1.lora_A.weight": object()
            for block in range(28)
        }

        converted, report = convert_anima_lora_layout(lora, 40)

        self.assertEqual(report.direction, "28_to_40")
        self.assertIn("base_model.model.diffusion_model.blocks.39.mlp.layer1.lora_A.weight", converted)
        self.assertEqual(anima_lora_block_indices(converted), tuple(range(40)))

    def test_collapses_40_block_lora_and_discards_inserted_blocks(self):
        lora, source_values = kohya_lora(40)

        converted, report = convert_anima_lora_layout(lora, 28)

        self.assertEqual(report.direction, "40_to_28")
        self.assertEqual(report.dropped_entries, 12)
        self.assertEqual(anima_lora_block_indices(converted), tuple(range(28)))
        self.assertIs(converted["lora_unet_blocks_1_self_attn_q_proj.lora_A.weight"], source_values[1])
        self.assertIs(converted["lora_unet_blocks_2_self_attn_q_proj.lora_A.weight"], source_values[3])
        self.assertIs(converted["lora_unet_blocks_27_self_attn_q_proj.lora_A.weight"], source_values[39])
        self.assertNotIn(source_values[2], converted.values())

    def test_28_to_40_to_28_round_trip_preserves_the_original_lora(self):
        original, _ = kohya_lora(28)

        expanded, _ = convert_anima_lora_layout(original, 40)
        collapsed, report = convert_anima_lora_layout(expanded, 28)

        self.assertEqual(report.direction, "40_to_28")
        self.assertEqual(collapsed.keys(), original.keys())
        for key, value in original.items():
            self.assertIs(collapsed[key], value)

    def test_expands_40_block_lora_to_52_without_copying_tensors(self):
        lora, source_values = kohya_lora(40)

        converted, report = convert_anima_lora_layout(lora, 52)

        self.assertEqual(report.direction, "40_to_52")
        self.assertEqual(report.duplicated_entries, 12)
        self.assertEqual(anima_lora_block_indices(converted), tuple(range(52)))
        self.assertIs(
            converted["lora_unet_blocks_2_self_attn_q_proj.lora_A.weight"],
            source_values[2],
        )
        self.assertIs(
            converted["lora_unet_blocks_3_self_attn_q_proj.lora_A.weight"],
            source_values[2],
        )

    def test_28_to_52_to_28_round_trip_preserves_the_original_lora(self):
        original, source_values = kohya_lora(28)

        expanded, expand_report = convert_anima_lora_layout(original, 52)
        collapsed, collapse_report = convert_anima_lora_layout(expanded, 28)

        self.assertEqual(expand_report.direction, "28_to_52")
        self.assertEqual(expand_report.duplicated_entries, 24)
        self.assertIs(
            expanded["lora_unet_blocks_3_self_attn_q_proj.lora_A.weight"],
            source_values[1],
        )
        self.assertEqual(collapse_report.direction, "52_to_28")
        self.assertEqual(collapse_report.dropped_entries, 24)
        self.assertEqual(collapsed.keys(), original.keys())
        for key, value in original.items():
            self.assertIs(collapsed[key], value)

    def test_collapses_52_block_lora_to_40_lossily(self):
        lora, source_values = kohya_lora(52)

        converted, report = convert_anima_lora_layout(lora, 40)

        self.assertEqual(report.direction, "52_to_40")
        self.assertEqual(report.dropped_entries, 12)
        self.assertEqual(anima_lora_block_indices(converted), tuple(range(40)))
        self.assertIs(
            converted["lora_unet_blocks_2_self_attn_q_proj.lora_A.weight"],
            source_values[2],
        )
        self.assertNotIn(source_values[3], converted.values())

    def test_native_and_ambiguous_layouts_are_left_unchanged(self):
        native, _ = kohya_lora(52)
        sparse = {
            "lora_unet_blocks_0_self_attn_q_proj.lora_A.weight": object(),
            "lora_unet_blocks_27_self_attn_q_proj.lora_A.weight": object(),
        }

        native_result, native_report = convert_anima_lora_layout(native, 52)
        sparse_result, sparse_report = convert_anima_lora_layout(sparse, 52)

        self.assertIs(native_result, native)
        self.assertEqual(native_report.direction, "native")
        self.assertIs(sparse_result, sparse)
        self.assertEqual(sparse_report.direction, "ambiguous_source")


if __name__ == "__main__":
    unittest.main()
