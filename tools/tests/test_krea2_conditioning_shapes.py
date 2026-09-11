import sys
import unittest

import torch

sys.argv = sys.argv[:1]

from backend.nn.krea import _repeat_reference_to_batch, _validate_context_shape
from backend.sampling.condition import ConditionCrossAttn


class Krea2ReferenceBatchTests(unittest.TestCase):
    def test_matching_reference_batch_is_returned_unchanged(self):
        reference = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)

        repeated = _repeat_reference_to_batch(reference, 2)

        self.assertIs(reference, repeated)

    def test_single_reference_repeats_to_sampling_batch(self):
        reference = torch.arange(3 * 2 * 2).reshape(1, 3, 2, 2)

        repeated = _repeat_reference_to_batch(reference, 4)

        self.assertEqual((4, 3, 2, 2), tuple(repeated.shape))
        self.assertTrue(torch.equal(repeated, reference.repeat(4, 1, 1, 1)))

    def test_reference_batch_repeats_by_integer_factor(self):
        reference = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)

        repeated = _repeat_reference_to_batch(reference, 4)

        self.assertTrue(torch.equal(repeated, reference.repeat(2, 1, 1, 1)))

    def test_non_divisible_reference_batch_fails_clearly(self):
        reference = torch.zeros(2, 3, 2, 2)

        with self.assertRaisesRegex(ValueError, "reference batch 2.*sampling batch 3"):
            _repeat_reference_to_batch(reference, 3)

    def test_empty_reference_batch_fails_clearly(self):
        reference = torch.empty(0, 3, 2, 2)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            _repeat_reference_to_batch(reference, 1)


class Krea2ContextShapeTests(unittest.TestCase):
    def test_canonical_context_shape_is_accepted(self):
        context = torch.zeros(2, 7, 4, 5)

        validated = _validate_context_shape(
            context,
            batch_size=2,
            text_layers=4,
            text_dim=5,
        )

        self.assertIs(context, validated)

    def test_incompatible_context_shapes_fail_clearly(self):
        cases = {
            "legacy_fused": torch.zeros(2, 1, 7, 20),
            "wrong_layers": torch.zeros(2, 7, 3, 5),
            "wrong_hidden": torch.zeros(2, 7, 4, 6),
            "wrong_batch": torch.zeros(1, 7, 4, 5),
            "missing_layer_axis": torch.zeros(2, 7, 20),
        }

        for name, context in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError,
                    r"\[batch, sequence, 4, 5\]",
                ):
                    _validate_context_shape(
                        context,
                        batch_size=2,
                        text_layers=4,
                        text_dim=5,
                    )


class ConditionCrossAttentionShapeTests(unittest.TestCase):
    def test_four_dimensional_conditioning_repeats_sequence_axis(self):
        long_condition = torch.zeros(1, 4, 3, 2)
        short_condition = torch.arange(12).reshape(1, 2, 3, 2)
        long = ConditionCrossAttn(long_condition)
        short = ConditionCrossAttn(short_condition)

        self.assertTrue(long.can_concat(short))
        combined = long.concat([short])

        self.assertEqual((2, 4, 3, 2), tuple(combined.shape))
        self.assertTrue(
            torch.equal(combined[1:], short_condition.repeat(1, 2, 1, 1))
        )

    def test_three_dimensional_conditioning_remains_supported(self):
        long = ConditionCrossAttn(torch.zeros(1, 4, 6))
        short_tensor = torch.arange(12).reshape(1, 2, 6)
        short = ConditionCrossAttn(short_tensor)

        combined = long.concat([short])

        self.assertEqual((2, 4, 6), tuple(combined.shape))
        self.assertTrue(torch.equal(combined[1:], short_tensor.repeat(1, 2, 1)))

    def test_different_layer_or_hidden_shapes_are_not_concatable(self):
        base = ConditionCrossAttn(torch.zeros(1, 4, 3, 2))

        self.assertFalse(
            base.can_concat(ConditionCrossAttn(torch.zeros(1, 2, 4, 2)))
        )
        self.assertFalse(
            base.can_concat(ConditionCrossAttn(torch.zeros(1, 2, 3, 5)))
        )


if __name__ == "__main__":
    unittest.main()
