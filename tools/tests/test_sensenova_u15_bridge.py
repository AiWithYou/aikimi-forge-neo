import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import psutil
import torch
from PIL import Image
from safetensors.torch import save_file

from modules_forge import sensenova_u15_bridge as bridge


class SenseNovaRequestTests(unittest.TestCase):
    def test_reference_profile_limits_are_explicit(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT, prompt="Combine references", generation_profile=bridge.PROFILE_QUALITY,
            input_images=(Image.new("RGB", (32, 32)),) * 8,
            width=2048, height=2048, input_max_pixels=1024 * 1024, vram_mode="reference",
        )
        bridge.validate_request(request)
        for changes in (
            {"input_images": request.input_images + request.input_images[:1]},
            {"input_max_pixels": "auto"}, {"input_max_pixels": 2048 * 2048},
            {"width": 4096},
        ):
            with self.subTest(changes=changes.keys()), self.assertRaises(bridge.SenseNovaBridgeError):
                bridge.validate_request(bridge.SenseNovaRequest(**{**request.__dict__, **changes}))

    def test_reference_limits_are_checked_before_pixel_loading(self):
        with mock.patch.object(bridge, "_image_from_gallery_item") as decode:
            with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "最大64枚"):
                bridge.normalize_gallery_images(["unused.png"] * 65)
            decode.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.png"
            Image.new("RGB", (11, 10)).save(path)
            with (
                mock.patch.object(bridge, "MAX_SOURCE_PIXELS", 100),
                mock.patch.object(bridge, "_flatten_rgb") as flatten,
            ):
                with self.assertRaises(bridge.SenseNovaBridgeError):
                    bridge.normalize_gallery_images([path])
                flatten.assert_not_called()

    def test_official_resolution_and_edit_auto_resolution(self):
        self.assertEqual(
            bridge.parse_resolution("2720x1536", bridge.MODE_TEXT), (2720, 1536)
        )
        self.assertEqual(
            bridge.parse_resolution("auto", bridge.MODE_EDIT), (None, None)
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "画像編集"):
            bridge.parse_resolution("auto", bridge.MODE_TEXT)

    def test_multiple_gallery_images_keep_order_and_flatten_alpha(self):
        red = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
        blue = Image.new("RGB", (12, 10), (0, 0, 255))
        images = bridge.normalize_gallery_images([(red, None), (blue, "second")])
        self.assertEqual([image.size for image in images], [(16, 16), (12, 10)])
        self.assertEqual([image.mode for image in images], ["RGB", "RGB"])
        self.assertGreater(images[0].getpixel((0, 0))[0], images[0].getpixel((0, 0))[1])
        self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 255))

    def test_edit_accepts_sixty_four_ordered_images(self):
        images = tuple(
            Image.new("RGB", (32, 32), (index, 0, 0))
            for index in range(bridge.MAX_REFERENCE_IMAGES)
        )
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Use every image in order.",
            generation_profile=bridge.PROFILE_QUALITY,
            input_images=images,
            width=None,
            height=None,
            vram_mode="unrestricted",
        )
        bridge.validate_request(request)

        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "最大64枚"):
            bridge.validate_request(
                bridge.SenseNovaRequest(
                    mode=bridge.MODE_EDIT,
                    prompt="Too many references",
                    generation_profile=bridge.PROFILE_QUALITY,
                    input_images=images + (Image.new("RGB", (32, 32)),),
                )
            )

    def test_low_vram_profile_rejects_the_failed_four_megapixel_workload(self):
        images = (
            Image.new("RGB", (32, 32), (255, 0, 0)),
            Image.new("RGB", (32, 32), (0, 0, 255)),
        )
        unsafe = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Combine two references",
            generation_profile=bridge.PROFILE_QUALITY,
            input_images=images,
            width=1664,
            height=2496,
            input_max_pixels="auto",
            vram_mode="low",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "24GB安全"):
            bridge.validate_request(unsafe)

        unrestricted = bridge.SenseNovaRequest(
            **{**unsafe.__dict__, "vram_mode": "unrestricted"}
        )
        bridge.validate_request(unrestricted)

    def test_low_vram_defaults_match_the_measured_3090_profile(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="Combine two references",
            generation_profile=bridge.PROFILE_QUALITY,
            input_images=(
                Image.new("RGB", (32, 32)),
                Image.new("RGB", (32, 32)),
            ),
            width=None,
            height=None,
        )
        bridge.validate_request(request)
        self.assertEqual(request.target_pixels, 2048 * 2048)
        self.assertEqual(request.input_max_pixels, 512 * 512)
        self.assertEqual(request.vram_mode, "low")

        three_references = bridge.SenseNovaRequest(
            **{
                **request.__dict__,
                "input_images": request.input_images
                + (Image.new("RGB", (32, 32)),),
            }
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "参照3枚"):
            bridge.validate_request(three_references)

        two_k = bridge.SenseNovaRequest(
            mode=bridge.MODE_EDIT,
            prompt="2K edit",
            generation_profile=bridge.PROFILE_QUALITY,
            input_images=request.input_images,
            width=2048,
            height=2048,
        )
        bridge.validate_request(two_k)

    def test_text_mode_rejects_hidden_reference_data(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            generation_profile=bridge.PROFILE_QUALITY,
            input_images=(Image.new("RGB", (32, 32)),),
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "画像編集"):
            bridge.validate_request(request)

    def test_worker_payload_pins_final_checkpoint_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_TEXT,
                prompt="A lighthouse",
                generation_profile=bridge.PROFILE_QUALITY,
                checkpoint=str(root / "model.safetensors"),
            )
            payload = bridge._request_payload(
                request,
                input_paths=[],
                output_path=root / "output.png",
                metadata_path=root / "output.json",
            )
        self.assertEqual(payload["model_path"], bridge.DEFAULT_MODEL_ID)
        self.assertEqual(
            payload["checkpoint_revision"], bridge.CHECKPOINT_REVISION
        )
        self.assertEqual(payload["checkpoint_sha256"], bridge.CONVROT_SHA256)
        self.assertEqual(payload["quantization"], bridge.QUANT_INT8_CONVROT)
        self.assertEqual(payload["generation_profile"], bridge.PROFILE_QUALITY)
        self.assertEqual(payload["lora_path"], "")

    def test_official_8step_profile_is_fixed_to_the_verified_t2i_preset(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            generation_profile=bridge.PROFILE_OFFICIAL_8STEP,
            steps=8,
            cfg_scale=1.0,
            timestep_shift=3.0,
        )
        with mock.patch.object(bridge, "_official_lora_is_ready", return_value=True):
            bridge.validate_request(request)

        payload = bridge._request_payload(
            request,
            input_paths=[],
            output_path=Path("output.png"),
            metadata_path=Path("output.json"),
        )
        self.assertEqual(
            payload["lora_path"], str(bridge.DEFAULT_LORA_PATH.resolve())
        )
        self.assertEqual(payload["lora_revision"], bridge.OFFICIAL_LORA_REVISION)
        self.assertEqual(payload["lora_sha256"], bridge.OFFICIAL_LORA_SHA256)

        for updates, message in (
            ({"mode": bridge.MODE_EDIT}, "テキスト生成専用"),
            ({"steps": 7}, "Steps 8"),
            ({"cfg_scale": 4.0}, "CFG 1.0"),
            ({"timestep_shift": 2.0}, "Shift 3.0"),
        ):
            invalid = bridge.SenseNovaRequest(
                **{**request.__dict__, **updates}
            )
            with (
                mock.patch.object(
                    bridge, "_official_lora_is_ready", return_value=True
                ),
                self.assertRaisesRegex(bridge.SenseNovaBridgeError, message),
            ):
                bridge.validate_request(invalid)

    def test_convrot_requires_safetensors_extension(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            generation_profile=bridge.PROFILE_QUALITY,
            checkpoint="weights.gguf",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, r"\.safetensors"):
            bridge.validate_request(request)

    def test_convrot_rejects_preview_model_config(self):
        request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A lighthouse",
            generation_profile=bridge.PROFILE_QUALITY,
            model_path="sensenova/SenseNova-U1.5-8B-MoT-Preview",
        )
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "正式版"):
            bridge.validate_request(request)


class SenseNovaRuntimeTests(unittest.TestCase):
    @staticmethod
    def _make_runtime(root: Path) -> Path:
        source = root / "runtime-final"
        package = source / "SenseNova" / "src" / "sensenova_u1"
        inference = source / "SenseNova" / "examples" / "editing"
        config = source / "SenseNova-U1.5-8B-MoT"
        package.mkdir(parents=True)
        inference.mkdir(parents=True)
        config.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (inference / "inference.py").write_text("", encoding="utf-8")
        (config / "config.json").write_text("{}", encoding="utf-8")
        (source / ".sensenova_runtime_revision").write_text(
            bridge.SOURCE_REVISION + "\n", encoding="utf-8"
        )
        return source

    def test_runtime_status_checks_revision_size_and_convrot_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_runtime(root)
            checkpoint = root / "model.safetensors"
            marker = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
                dtype=torch.uint8,
            )
            save_file(
                {
                    "layer.comfy_quant": marker,
                    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": torch.ones(1),
                },
                checkpoint,
            )
            Path(str(checkpoint) + ".sha256").write_text(
                bridge.CONVROT_SHA256 + "  model.safetensors\n", encoding="utf-8"
            )

            with (
                mock.patch.object(bridge, "CONVROT_EXPECTED_BYTES", checkpoint.stat().st_size),
                mock.patch.object(bridge, "EXPECTED_CONVROT_LAYERS", 1),
                mock.patch.object(bridge, "environment_status", return_value=(True, "ready")),
            ):
                status = bridge.inspect_runtime(source, checkpoint=checkpoint)

            self.assertTrue(status.ready)
            self.assertTrue(status.source_ready)
            self.assertTrue(status.checkpoint_ready)

    def test_partial_convrot_is_reported_without_being_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_runtime(root)
            checkpoint = root / "model.safetensors"
            Path(str(checkpoint) + ".part").write_bytes(b"partial")
            status = bridge.inspect_runtime(source, checkpoint=checkpoint)
            self.assertFalse(status.ready)
            self.assertEqual(status.partial_bytes, 7)
            self.assertTrue(
                any("ダウンロード中" in message for message in status.messages)
            )

    def test_parallel_convrot_chunks_are_counted_in_download_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._make_runtime(root)
            checkpoint = root / "model.safetensors"
            chunks = Path(str(checkpoint) + ".part.chunks")
            chunks.mkdir()
            (chunks / "chunk-001.bin").write_bytes(b"1234")
            (chunks / "chunk-002.bin").write_bytes(b"56789")
            (chunks / "chunk-002.bin.resume").write_bytes(b"abc")
            status = bridge.inspect_runtime(source, checkpoint=checkpoint)
            self.assertFalse(status.ready)
            self.assertEqual(status.partial_bytes, 12)


class SenseNovaWorkerBridgeTests(unittest.TestCase):
    def setUp(self):
        worker_python = mock.patch.object(bridge, "WORKER_PYTHON", Path(sys.executable))
        worker_python.start()
        self.addCleanup(worker_python.stop)

    def test_worker_events_are_parsed_strictly(self):
        event = bridge._parse_event(
            'SENSENOVA_EVENT {"stage":"sampling","progress":0.5}'
        )
        self.assertEqual(event["stage"], "sampling")
        self.assertIsNone(bridge._parse_event("ordinary log line"))
        self.assertIsNone(bridge._parse_event("SENSENOVA_EVENT not-json"))

    def test_job_cleanup_rejects_paths_outside_cache_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "許可範囲外"):
                bridge._cleanup_job_directory(outside, cache)
            self.assertTrue(outside.is_dir())

    def test_run_generation_stages_ordered_inputs_and_verifies_outputs(self):
        fake_worker = r"""
import hashlib
import json
import pathlib
import sys
from PIL import Image

request_path = pathlib.Path(sys.argv[sys.argv.index("--request") + 1])
payload = json.loads(request_path.read_text(encoding="utf-8"))
print('SENSENOVA_EVENT ' + json.dumps({"stage": "loading", "message": "fake load", "progress": 0.2}), flush=True)
names = [pathlib.Path(path).name for path in payload["input_images"]]
output = pathlib.Path(payload["output_path"])
Image.new("RGB", (payload["width"], payload["height"]), "blue").save(output)
metadata = {
    "schema_version": 3,
    "input_image_names": names, "input_image_count": len(names),
    "width": payload["width"], "height": payload["height"],
    "mode": payload["mode"], "prompt": payload["prompt"],
    "seed": payload["seed"], "steps": payload["steps"],
    "generation_profile": payload["generation_profile"],
    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}
pathlib.Path(payload["metadata_path"]).write_text(json.dumps(metadata), encoding="utf-8")
print('SENSENOVA_EVENT ' + json.dumps({"stage": "complete", "message": "fake done", "progress": 1.0}), flush=True)
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text(fake_worker, encoding="utf-8")
            output = root / "outputs"
            cache = root / "cache"
            logs = root / "logs"
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_EDIT,
                prompt="Combine in order",
                generation_profile=bridge.PROFILE_QUALITY,
                quantization=bridge.QUANT_INT8_CONVROT,
                checkpoint="weights.safetensors",
                input_images=(
                    Image.new("RGB", (20, 20), (255, 0, 0)),
                    Image.new("RGB", (20, 20), (0, 0, 255)),
                ),
                width=1024,
                height=1024,
            )
            ready = bridge.RuntimeStatus(
                ready=True,
                source_ready=True,
                dependencies_ready=True,
                checkpoint_ready=True,
                source_path=root,
                checkpoint_path=root / "weights.safetensors",
                messages=(),
            )
            real_popen = bridge.subprocess.Popen

            def launch_dedicated_worker(command, **kwargs):
                self.assertEqual(command[0], "dedicated-worker-python")
                return real_popen([sys.executable, *command[1:]], **kwargs)

            with (
                mock.patch.object(bridge, "inspect_runtime", return_value=ready),
                mock.patch.object(bridge, "_release_forge_vram"),
                mock.patch.object(bridge, "WORKER_PYTHON", Path("dedicated-worker-python")),
                mock.patch.object(bridge.subprocess, "Popen", side_effect=launch_dedicated_worker),
            ):
                updates = list(
                    bridge.run_generation(
                        request,
                        output_directory=output,
                        cache_directory=cache,
                        log_directory=logs,
                        worker_path=worker,
                    )
                )

            completed = [update for update in updates if update["stage"] == "complete"]
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["metadata"]["input_image_count"], 2)
            self.assertEqual(
                completed[0]["metadata"]["input_image_names"],
                ["reference_01.png", "reference_02.png"],
            )
            self.assertTrue(Path(completed[0]["path"]).is_file())
            self.assertEqual(list((cache / "jobs").glob("*")), [])

    def test_cancel_before_worker_launch_cleans_job_without_spawning(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text("raise SystemExit(99)\n", encoding="utf-8")
            cache = root / "cache"
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_TEXT,
                prompt="Cancel before launch",
                generation_profile=bridge.PROFILE_QUALITY,
                quantization=bridge.QUANT_INT8_CONVROT,
                checkpoint="weights.safetensors",
                width=1024,
                height=1024,
            )
            ready = bridge.RuntimeStatus(
                ready=True,
                source_ready=True,
                dependencies_ready=True,
                checkpoint_ready=True,
                source_path=root,
                checkpoint_path=root / "weights.safetensors",
                messages=(),
            )
            updates = bridge.run_generation(
                request,
                output_directory=root / "outputs",
                cache_directory=cache,
                log_directory=root / "logs",
                worker_path=worker,
            )
            with (
                mock.patch.object(bridge, "inspect_runtime", return_value=ready),
                mock.patch.object(bridge.subprocess, "Popen") as popen,
            ):
                prepared = next(updates)
                self.assertEqual(prepared["stage"], "prepare")
                self.assertIn("worker起動前", bridge.cancel_generation(prepared["job_id"]))
                with self.assertRaises(bridge.SenseNovaGenerationCancelled):
                    next(updates)

            popen.assert_not_called()
            self.assertEqual(list((cache / "jobs").glob("*")), [])
            self.assertIsNone(bridge._ACTIVE_JOB_ID)

    def test_cancel_terminates_the_active_worker_and_cleans_the_job(self):
        fake_worker = r"""
import json
import os
import pathlib
import sys
import time

request_path = pathlib.Path(sys.argv[sys.argv.index("--request") + 1])
payload = json.loads(request_path.read_text(encoding="utf-8"))
handles = [open(path, "rb") for path in payload["input_images"]]
(request_path.parent / "worker.pid").write_text(str(os.getpid()), encoding="ascii")
print('SENSENOVA_EVENT ' + json.dumps({"stage": "loading", "message": "waiting", "progress": 0.1}), flush=True)
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text(fake_worker, encoding="utf-8")
            cache = root / "cache"
            request = bridge.SenseNovaRequest(
                mode=bridge.MODE_EDIT,
                input_images=(Image.new("RGB", (512, 512)),),
                prompt="A lighthouse",
                generation_profile=bridge.PROFILE_QUALITY,
                quantization=bridge.QUANT_INT8_CONVROT,
                checkpoint="weights.safetensors",
                width=1024,
                height=1024,
            )
            ready = bridge.RuntimeStatus(
                ready=True,
                source_ready=True,
                dependencies_ready=True,
                checkpoint_ready=True,
                source_path=root,
                checkpoint_path=root / "weights.safetensors",
                messages=(),
            )
            with (
                mock.patch.object(bridge, "inspect_runtime", return_value=ready),
                mock.patch.object(bridge, "_release_forge_vram"),
            ):
                updates = bridge.run_generation(
                    request,
                    output_directory=root / "outputs",
                    cache_directory=cache,
                    log_directory=root / "logs",
                    worker_path=worker,
                )
                loading = None
                for update in updates:
                    if update["stage"] == "loading":
                        loading = update
                        break
                self.assertIsNotNone(loading)
                pid_file = cache / "jobs" / loading["job_id"] / "worker.pid"
                worker_process = psutil.Process(int(pid_file.read_text(encoding="ascii")))
                try:
                    message = bridge.cancel_generation(loading["job_id"])
                    self.assertIn("キャンセル", message)
                    worker_process.wait(timeout=2)
                    with self.assertRaises(bridge.SenseNovaGenerationCancelled):
                        list(updates)
                finally:
                    if worker_process.is_running():
                        worker_process.kill()
                        worker_process.wait(timeout=5)
                    updates.close()
            self.assertEqual(list((cache / "jobs").glob("*")), [])
            # 次回受付まで回復し、前workerの入力ハンドルが削除を妨げない。
            with mock.patch.object(bridge, "inspect_runtime", return_value=ready):
                following = bridge.run_generation(
                    request, output_directory=root / "outputs", cache_directory=cache,
                    log_directory=root / "logs", worker_path=worker,
                )
                self.assertEqual(next(following)["stage"], "prepare")
                following.close()


if __name__ == "__main__":
    unittest.main()
