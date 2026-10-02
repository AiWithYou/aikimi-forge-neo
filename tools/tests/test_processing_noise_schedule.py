"""Temporary noise schedules must be restored on the predictor they changed."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class SamplingInterrupted(BaseException):
    """Match the sampler's InterruptedException hierarchy without importing models."""


class NoiseScheduleRestoreTests(unittest.TestCase):
    def setUp(self):
        self.source = ROOT / "modules/processing.py"
        tree = ast.parse(self.source.read_text(encoding="utf-8"))
        process = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_images_inner"
        )
        batch_loop = next(
            node
            for node in ast.walk(process)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "n"
        )
        start = next(
            index
            for index, node in enumerate(batch_loop.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "sigmas_backup" for target in node.targets)
        )
        end = next(
            index
            for index in range(start + 1, len(batch_loop.body))
            if isinstance(batch_loop.body[index], ast.If)
            and ast.unparse(batch_loop.body[index].test) == "p.scripts is not None"
        )
        self.code = compile(ast.Module(body=batch_loop.body[start:end], type_ignores=[]), str(self.source), "exec")
        self.original_sigmas = [1.0, 0.0]
        self.temporary_sigmas = [1.1, 0.0]
        self.predictor = self.make_predictor(self.original_sigmas)
        self.model = SimpleNamespace(predictor=self.predictor)
        self.p = SimpleNamespace(
            sd_model=SimpleNamespace(
                model_config=SimpleNamespace(ztsnr=False),
                forge_objects=SimpleNamespace(unet=SimpleNamespace(model=self.model)),
            ),
            extra_generation_params={},
            c=None,
            uc=None,
            seeds=[1],
            subseeds=[2],
            subseed_strength=0.0,
            prompts=["fixture"],
            latents_after_sampling=[],
            sample=lambda **_kwargs: ["latent"],
        )

    @staticmethod
    def make_predictor(sigmas):
        predictor = SimpleNamespace(sigmas=sigmas, changes=[])

        def set_sigmas(value):
            predictor.changes.append(value)
            predictor.sigmas = value

        predictor.set_sigmas = set_sigmas
        return predictor

    def run_sample(self, schedule="Zero Terminal SNR"):
        namespace = {
            "p": self.p,
            "opts": SimpleNamespace(sd_noise_schedule=schedule),
            "rescale_zero_terminal_snr_sigmas": lambda _sigmas: self.temporary_sigmas,
        }
        exec(self.code, namespace)  # noqa: S102 - execute only the extracted repository sampling block

    def test_success_restores_original_schedule(self):
        self.run_sample()
        self.assertIs(self.predictor.sigmas, self.original_sigmas)
        self.assertEqual(self.p.latents_after_sampling, ["latent"])
        self.assertEqual(self.predictor.changes, [self.temporary_sigmas, self.original_sigmas])

    def test_sampling_failure_and_interrupt_restore_original_schedule(self):
        for error in (RuntimeError("sample failed"), SamplingInterrupted("cancelled")):
            with self.subTest(error=type(error).__name__):
                self.predictor.changes.clear()
                self.predictor.sigmas = self.original_sigmas

                def fail(error=error, **_kwargs):
                    raise error

                self.p.sample = fail
                with self.assertRaises(type(error)) as caught:
                    self.run_sample()
                self.assertIs(caught.exception, error)
                self.assertIs(self.predictor.sigmas, self.original_sigmas)
                self.assertEqual(self.predictor.changes, [self.temporary_sigmas, self.original_sigmas])

    def test_model_switch_restores_only_original_predictor(self):
        for error in (None, RuntimeError("sample failed"), SamplingInterrupted("cancelled")):
            with self.subTest(error=type(error).__name__):
                self.p.sd_model.forge_objects.unet.model = self.model
                replacement_sigmas = [4.0, 0.0]
                replacement = self.make_predictor(replacement_sigmas)

                def switch_model(error=error, replacement=replacement, **_kwargs):
                    self.p.sd_model.forge_objects.unet.model = SimpleNamespace(predictor=replacement)
                    if error is not None:
                        raise error
                    return ["latent"]

                self.p.sample = switch_model
                if error is None:
                    self.run_sample()
                else:
                    with self.assertRaises(type(error)) as caught:
                        self.run_sample()
                    self.assertIs(caught.exception, error)
                self.assertIs(self.predictor.sigmas, self.original_sigmas)
                self.assertIs(replacement.sigmas, replacement_sigmas)
                self.assertEqual(replacement.changes, [])

    def test_default_schedule_does_not_change_predictor(self):
        self.run_sample("Default")
        self.assertEqual(self.predictor.changes, [])
        self.assertIs(self.predictor.sigmas, self.original_sigmas)


if __name__ == "__main__":
    unittest.main()
