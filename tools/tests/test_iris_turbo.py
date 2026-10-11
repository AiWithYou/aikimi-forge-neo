"""Turbo preserves the public CFG1 loop and applies optional manual guidance."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modules_forge.iris.turbo import generate_turbo


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.cfg = SimpleNamespace(in_channels=3)
        self.times = []

    def forward(self, x, t, y, y_mask=None):
        self.times.append(t.clone())
        return SimpleNamespace(x=x * self.weight)


class TurboTests(unittest.TestCase):
    def test_manual_cfg_uses_negative_condition_and_one_noise_per_step(self):
        class ConditionedModel(StubModel):
            def forward(self, x, t, y, y_mask=None):
                self.times.append(t.clone())
                return SimpleNamespace(x=x * 0.1 + y.mean())

        positive = SimpleNamespace(embeddings=torch.full((1, 2, 4), 0.02), mask=torch.ones(1, 2))
        negative = SimpleNamespace(embeddings=torch.full((1, 2, 4), -0.03), mask=torch.zeros(1, 2))
        negative_prompts = []

        def null(prompt):
            negative_prompts.append(prompt)
            return negative

        encoder = SimpleNamespace(encode=lambda prompts: positive, null=null)
        model = ConditionedModel()
        generator = torch.Generator().manual_seed(42)
        actual = generate_turbo(
            model, encoder, ["test"], 8, 8, 3, 4, generator, "cpu", cfg_scale=2, negative_prompt="dark"
        )
        reference_generator = torch.Generator().manual_seed(42)
        x = torch.randn(1, 3, 8, 8, generator=reference_generator)
        times = [1, 8 / 9, 2 / 3]
        for index, time in enumerate(times):
            conditional, unconditional = x * 0.1 + 0.02, x * 0.1 - 0.03
            x0 = x - time * (unconditional + 2 * (conditional - unconditional))
            if index < 2:
                fresh = torch.randn(x.shape, generator=reference_generator)
                x = (1 - times[index + 1]) * x0 + times[index + 1] * fresh
        torch.testing.assert_close(actual, x0.clamp(-1, 1))
        self.assertEqual(negative_prompts, ["dark"])
        self.assertEqual(len(model.times), 6)
        self.assertTrue(torch.equal(generator.get_state(), reference_generator.get_state()))

    def test_infinite_prediction_is_rejected_before_clamping(self):
        positive = SimpleNamespace(embeddings=torch.ones(1, 2, 4), mask=torch.ones(1, 2))
        encoder = SimpleNamespace(encode=lambda prompts: positive)
        model = StubModel()
        model.weight.data.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "非有限値"):
            generate_turbo(model, encoder, ["test"], 8, 8, 1, 4, torch.Generator().manual_seed(42), "cpu")

    def test_matches_reference_noise_order_times_and_four_calls(self):
        positive = SimpleNamespace(embeddings=torch.ones(1, 2, 4), mask=torch.ones(1, 2))
        encoder = SimpleNamespace(encode=lambda prompts: positive)
        model = StubModel()
        generator = torch.Generator().manual_seed(42)
        actual = generate_turbo(
            model, encoder, ["test"], 8, 8, 4, 4, generator, "cpu", cfg_scale=1, negative_prompt="unused"
        )
        times = [1, 12 / 13, 4 / 5, 4 / 7]
        expected_generator = torch.Generator().manual_seed(42)
        x = torch.randn(1, 3, 8, 8, generator=expected_generator)
        for index, time in enumerate(times):
            x0 = x - time * x * 0.25
            if index < 3:
                fresh = torch.randn(x.shape, generator=expected_generator)
                x = (1 - times[index + 1]) * x0 + times[index + 1] * fresh
        torch.testing.assert_close(actual, x0.clamp(-1, 1))
        self.assertEqual(len(model.times), 4)
        torch.testing.assert_close(torch.cat(model.times), torch.tensor(times) * 1000)
        self.assertTrue(torch.equal(generator.get_state(), expected_generator.get_state()))


if __name__ == "__main__":
    unittest.main()
