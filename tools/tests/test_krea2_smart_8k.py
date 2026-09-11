import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.krea2_smart_8k import (
    FINAL_SIZE,
    NATIVE_SIZE,
    PREFLIGHT_SIZE,
    RAW_NATIVE_SIZE,
    analysis_metrics,
    detail_retention_gate,
    detail_prompt,
    generate_native,
    highres_resolution_plan,
    native_detail_prompt,
    portable_basename,
    raw_resolution_shift,
    redacted_command,
    resolve_inference_profile,
    run_command,
    save_qa_crops,
    validate_krea2_backend,
    validate_finish_report,
    validate_vram_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


class Smart8KPromptTests(unittest.TestCase):
    def test_refinement_prompt_preserves_base_and_requests_material_detail(self):
        base = "one person in a snowy railway station"

        prompt = detail_prompt(base)

        self.assertTrue(prompt.startswith(f"{base}. "))
        self.assertIn("facial proportions", prompt)
        self.assertIn("iris radial structure", prompt)
        self.assertIn("fabric weave", prompt)
        self.assertIn("stone pores", prompt)
        self.assertIn("source style", prompt)
        self.assertIn("Do not introduce random grain", prompt)
        self.assertNotIn("slime", prompt.lower())
        self.assertNotIn("horn", prompt.lower())

    def test_trailing_tag_comma_is_not_followed_by_an_extra_period(self):
        prompt = detail_prompt("purple eyes,green slime,")

        self.assertTrue(prompt.startswith("purple eyes,green slime, Preserve"))

    def test_empty_base_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            detail_prompt("  ")

    def test_refinement_prompt_is_idempotent_for_4k_resume(self):
        once = detail_prompt("purple eyes, green slime")

        self.assertEqual(detail_prompt(once), once)

    def test_native_prompt_is_subject_and_style_neutral_by_default(self):
        base = "brutalist concrete library at blue hour, architectural photograph"
        prompt = native_detail_prompt(base)

        self.assertTrue(prompt.startswith(f"{base}. Treat the preceding user prompt"))
        self.assertIn("exactly the requested subjects", prompt)
        self.assertIn("material- and style-appropriate", prompt)
        self.assertNotIn("anime", prompt.lower())
        self.assertNotIn("slime", prompt.lower())
        self.assertNotIn("horn", prompt.lower())
        self.assertNotIn("purple", prompt.lower())
        self.assertEqual(native_detail_prompt(prompt), prompt)

    def test_case_study_prompt_requires_an_explicit_profile(self):
        prompt = native_detail_prompt(
            "light blue hair,purple_eyes,green_slime,",
            profile="anime-slime-case-study",
        )

        self.assertIn("green slime is mandatory", prompt)
        self.assertIn("individually resolved layered hair locks", prompt)
        self.assertIn("layered lace", prompt)

    def test_requested_base_prompt_is_preserved_byte_for_byte_as_prefix(self):
        base = (
            "light blue hair,long_wavy_hair,devil’s_horn,purple horn,"
            "purple_eyes,green_slime,jig eyes,smile,jitome,Expressionless,"
        )

        prompt = native_detail_prompt(base)

        self.assertEqual(prompt[: len(base)], base)
        self.assertEqual(prompt.count(base), 1)


class Smart8KInferenceProfileTests(unittest.TestCase):
    def resolve(self, profile: str, **overrides):
        values = {
            "profile": profile,
            "model_profile": None,
            "native_width": None,
            "native_height": None,
            "native_steps": None,
            "sampler": None,
            "scheduler": None,
            "cfg": None,
            "distilled_cfg": None,
        }
        values.update(overrides)
        return resolve_inference_profile(**values)

    def test_turbo_fast_keeps_the_measured_local_baseline(self):
        profile = self.resolve("turbo-fast")

        self.assertEqual(profile["model_profile"], "turbo")
        self.assertEqual(profile["native_size"], NATIVE_SIZE)
        self.assertEqual(profile["steps"], 4)
        self.assertEqual(profile["sampler"], "DPM++ 2M SDE")
        self.assertEqual(profile["cfg"], 1.0)
        self.assertEqual(profile["shift"], 1.15)

    def test_turbo_official_uses_eight_step_no_cfg_forge_mapping(self):
        profile = self.resolve("turbo-official")

        self.assertEqual(profile["steps"], 8)
        self.assertEqual(profile["sampler"], "Euler")
        self.assertEqual(profile["scheduler"], "Simple")
        self.assertEqual(profile["cfg"], 1.0)
        self.assertEqual(profile["shift"], 1.15)

    def test_raw_official_uses_52_steps_and_resolution_derived_shift(self):
        profile = self.resolve("raw-official")

        self.assertEqual(profile["model_profile"], "raw")
        self.assertEqual(profile["native_size"], RAW_NATIVE_SIZE)
        self.assertEqual(profile["steps"], 52)
        self.assertEqual(profile["cfg"], 3.5)
        self.assertAlmostEqual(
            profile["shift"], raw_resolution_shift(*RAW_NATIVE_SIZE), places=12
        )

    def test_raw_shift_matches_official_interpolation_endpoints(self):
        self.assertAlmostEqual(raw_resolution_shift(256, 256), 0.5, places=12)
        self.assertAlmostEqual(raw_resolution_shift(1280, 1280), 1.15, places=12)

    def test_fixed_profile_rejects_sampling_overrides(self):
        with self.assertRaisesRegex(ValueError, "Use --inference-profile custom"):
            self.resolve("turbo-official", native_steps=7)

    def test_custom_profile_requires_every_value(self):
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            self.resolve("custom", model_profile="turbo")

    def test_custom_profile_accepts_a_fully_explicit_configuration(self):
        profile = self.resolve(
            "custom",
            model_profile="turbo",
            native_width=1024,
            native_height=1024,
            native_steps=6,
            sampler="Euler",
            scheduler="Simple",
            cfg=1.0,
            distilled_cfg=1.05,
        )

        self.assertEqual(profile["native_size"], (1024, 1024))
        self.assertEqual(profile["steps"], 6)
        self.assertEqual(profile["shift"], 1.05)


class Smart8KDryRunTests(unittest.TestCase):
    def test_dry_run_writes_complete_resolution_plan_without_api(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "runs"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "krea2_smart_8k.py"),
                    "--prompt",
                    "test base prompt",
                    "--seed",
                    "20260713",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("DRY_RUN=1", completed.stdout)
            run_dir = next(output_root.glob("smart8k_*"))
            manifest = json.loads(
                (run_dir / "smart8k_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(
                manifest["resolution_plan"]["source"], list(NATIVE_SIZE)
            )
            self.assertEqual(
                manifest["resolution_plan"]["native_generation"], list(NATIVE_SIZE)
            )
            self.assertEqual(
                manifest["resolution_plan"]["preflight_4k"], list(PREFLIGHT_SIZE)
            )
            self.assertEqual(manifest["resolution_plan"]["final_8k"], list(FINAL_SIZE))
            self.assertEqual(manifest["seed"], 20260713)
            self.assertEqual(manifest["native_prompt_profile"], "generic")
            self.assertEqual(manifest["settings"]["inference_profile"], "turbo-fast")
            self.assertEqual(manifest["settings"]["model_profile"], "turbo")
            self.assertNotIn("slime", manifest["native_prompt"].lower())
            self.assertTrue(
                manifest["refinement_prompt"].startswith("test base prompt. ")
            )


class Smart8KPlanningTests(unittest.TestCase):
    def test_native_source_aspect_is_preserved_and_8k_is_exactly_double_4k(self):
        plan = highres_resolution_plan((1024, 1472), "native")

        self.assertEqual(plan["preflight_4k"], (2849, 4096))
        self.assertEqual(plan["final_8k"], (5698, 8192))

    def test_uhd_4k_source_continues_to_exact_uhd_8k(self):
        plan = highres_resolution_plan((3840, 2160), "4k")

        self.assertEqual(plan["preflight_4k"], (3840, 2160))
        self.assertEqual(plan["final_8k"], (7680, 4320))

    def test_non_4k_resume_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "3840..4096"):
            highres_resolution_plan((2048, 2048), "4k")


class Smart8KBackendTests(unittest.TestCase):
    def test_portable_basename_normalizes_windows_and_posix_paths(self):
        self.assertEqual(portable_basename(r"C:\models\turbo.safetensors"), "turbo.safetensors")
        self.assertEqual(portable_basename("/models/turbo.safetensors"), "turbo.safetensors")

    def runtime_status(self, **overrides):
        status = {
            "loaded": True,
            "architecture": "backend.diffusion_engine.krea.Krea2",
            "configuration": "huggingface_guess.model_list.Krea2",
            "transformer": "backend.nn.krea.SingleStreamDiT",
            "checkpoint": r"C:\models\turbo_krea2_int8.safetensors",
            "checkpoint_sha256": "abc",
            "additional_modules": [
                r"C:\models\qwen_image_vae.safetensors",
                r"C:\models\qwen3vl_4b_bf16.safetensors",
            ],
            "quantization": {
                "formats": {"int8_tensorwise": 224},
                "convrot_layer_count": 224,
            },
        }
        status.update(overrides)
        return status

    def test_loaded_krea2_engine_checkpoint_and_qwen_modules_are_required(self):
        report = validate_krea2_backend(
            {
                "sd_model_checkpoint": "turbo_krea2_int8.safetensors",
                "sd_checkpoint_hash": "abc",
                "forge_additional_modules": [
                    r"C:\models\qwen_image_vae.safetensors",
                    r"C:\models\qwen3vl_4b_bf16.safetensors",
                ],
            },
            self.runtime_status(),
            "turbo",
        )

        self.assertEqual(report["checkpoint"], "turbo_krea2_int8.safetensors")
        self.assertEqual(report["architecture"], "backend.diffusion_engine.krea.Krea2")
        self.assertEqual(report["quantization"]["convrot_layer_count"], 224)
        self.assertEqual(len(report["additional_modules"]), 2)

    def test_wrong_loaded_architecture_is_rejected_before_generation(self):
        with self.assertRaisesRegex(RuntimeError, "architecture is not Krea2"):
            validate_krea2_backend(
                {
                    "sd_model_checkpoint": "turbo_krea2_int8.safetensors",
                    "forge_additional_modules": self.runtime_status()[
                        "additional_modules"
                    ],
                },
                self.runtime_status(architecture="backend.diffusion_engine.flux.Flux"),
                "turbo",
            )

    def test_raw_profile_rejects_a_loaded_turbo_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "raw inference profile"):
            validate_krea2_backend(
                {"sd_model_checkpoint": "turbo_krea2_int8.safetensors"},
                self.runtime_status(),
                "raw",
            )

    def test_prompt_values_are_redacted_from_command_log(self):
        rendered = redacted_command(
            ["python", "tool.py", "--prompt", "secret words", "--seed", "1"]
        )

        self.assertNotIn("secret words", rendered)
        self.assertIn("<redacted>", rendered)

    def test_subprocess_hard_timeout_does_not_wait_for_stdout_eof(self):
        with TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "timeout.json"
            with patch("tools.krea2_smart_8k.post_json") as interrupt:
                with self.assertRaisesRegex(TimeoutError, "hard timeout"):
                    run_command(
                        [sys.executable, "-c", "import time; time.sleep(10)"],
                        "TIMEOUT_TEST",
                        1,
                        interrupt_api="http://127.0.0.1:7861",
                        telemetry_path=telemetry_path,
                        telemetry_interval=0.25,
                    )

            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            self.assertEqual(telemetry["error"]["type"], "TimeoutError")
            self.assertIsNotNone(telemetry["subprocess_exit_code"])

        interrupt.assert_called_once_with(
            "http://127.0.0.1:7861", "/sdapi/v1/interrupt", {}, 5
        )

    def test_successful_subprocess_persists_machine_readable_telemetry(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.json"
            lines = run_command(
                [sys.executable, "-c", "print('ok')"],
                "TELEMETRY_TEST",
                10,
                telemetry_path=path,
                telemetry_interval=0.25,
            )

            telemetry = json.loads(path.read_text(encoding="utf-8"))

            self.assertIn("ok", lines)
            self.assertEqual(telemetry["subprocess_exit_code"], 0)
            self.assertGreaterEqual(telemetry["duration_seconds"], 0)
            self.assertNotIn("secret words", telemetry["command"])


class _Txt2ImgHandler(BaseHTTPRequestHandler):
    payload = None

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path != "/sdapi/v1/txt2img":
            self.send_error(404)
            return
        size = (int(type(self).payload["width"]), int(type(self).payload["height"]))
        image = Image.new("RGB", size, (90, 110, 130))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        body = json.dumps(
            {
                "images": [encoded],
                "info": json.dumps(
                    {
                        "infotexts": [
                            "test prompt\nNegative prompt: avoid\n"
                            "Steps: 4, Seed: 123, Size: 1024x1448"
                        ]
                    }
                ),
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Smart8KNativeGenerationTests(unittest.TestCase):
    def test_native_api_payload_and_png_metadata_are_recorded(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Txt2ImgHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "native.png"
                report = generate_native(
                    api=f"http://127.0.0.1:{server.server_port}",
                    output=output,
                    prompt="test prompt",
                    negative_prompt="avoid",
                    seed=123,
                    steps=4,
                    sampler="DPM++ 2M SDE",
                    scheduler="Simple",
                    cfg=1.0,
                    distilled_cfg=1.15,
                    size=NATIVE_SIZE,
                    timeout=30,
                )

                self.assertEqual(report["size"], list(NATIVE_SIZE))
                self.assertEqual(report["seed"], 123)
                self.assertEqual(_Txt2ImgHandler.payload["prompt"], "test prompt")
                self.assertEqual(_Txt2ImgHandler.payload["seed"], 123)
                with Image.open(output) as result:
                    self.assertIn("test prompt", result.info["parameters"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class Smart8KGateTests(unittest.TestCase):
    def test_vram_manifest_requires_every_tile(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "target_size": list(PREFLIGHT_SIZE),
                        "stage_reports": [
                            {
                                "tile_count": 3,
                                "processed_tile_count": 2,
                                "skipped_tile_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "tile gate failed"):
                validate_vram_manifest(path, PREFLIGHT_SIZE)

    def test_finish_report_rejects_flat_region_changes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            path.write_text(
                json.dumps(
                    {
                        "detail_guard": {
                            "accepted": True,
                            "applied": True,
                            "detail_energy_ratio": 1.01,
                            "flat_region_changed_pixels": 1,
                            "clipped_channel_fraction": 0.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "flat-region"):
                validate_finish_report(path)

    def test_finish_report_accepts_a_clean_bit_identical_noop(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            path.write_text(
                json.dumps(
                    {
                        "detail_guard": {
                            "accepted": True,
                            "applied": False,
                            "changed_pixels": 0,
                            "detail_energy_ratio": 1.0,
                            "flat_region_changed_pixels": 0,
                            "clipped_channel_fraction": 0.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = validate_finish_report(path)

            self.assertFalse(report["detail_guard"]["applied"])

    def test_analysis_metrics_detect_line_detail(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lines.png"
            rgb = np.full((192, 256, 3), 120, dtype=np.uint8)
            rgb[:, ::16] = 145
            Image.fromarray(rgb, "RGB").save(path)

            metrics = analysis_metrics(path)

            self.assertEqual(metrics["analysis_size"], [1536, 1152])
            self.assertGreater(metrics["laplacian_variance"], 0.0)
            self.assertGreater(metrics["highpass_abs_p95"], 0.0)
            self.assertEqual(metrics["detail_analysis_size"], [4096, 3072])
            self.assertEqual(
                set(metrics["normalized_multiband"]),
                {"sigma_0_1", "sigma_1_2", "sigma_2_4", "sigma_4_8"},
            )

    def test_detail_gate_rejects_noise_like_metric_explosion(self):
        source = {
            "gradient_p95": 10.0,
            "highpass_abs_mean": 2.0,
            "highpass_abs_p95": 5.0,
        }
        candidate = {
            "gradient_p95": 20.0,
            "highpass_abs_mean": 4.0,
            "highpass_abs_p95": 10.0,
        }

        with self.assertRaisesRegex(RuntimeError, "noise or oversharpening"):
            detail_retention_gate(source, candidate, minimum_ratio=0.7)

    def test_native_resolution_qa_crops_are_fixed_and_hashed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (1600, 2200), (20, 40, 60)).save(source)

            records = save_qa_crops(source, root / "qa")

            self.assertEqual(len(records), 7)
            self.assertEqual(records[0]["name"], "upper_left")
            self.assertEqual(records[0]["size"], [1024, 1024])
            self.assertEqual(len(records[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
