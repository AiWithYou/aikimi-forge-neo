import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "tools" / "sensenova_u15_worker.py"
SPEC = importlib.util.spec_from_file_location("sensenova_u15_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(worker)


class SenseNovaWorkerTests(unittest.TestCase):
    def validate_payload(self, payload):
        from modules_forge import local_assets
        from modules_forge import sensenova_u15_bridge as bridge

        value = {**payload, "checkpoint_identity": local_assets.file_identity(Path(payload["checkpoint"]))}
        # These tests exercise sampling/input contracts; structure validation has separate fixtures.
        with mock.patch.object(bridge, "compatible_checkpoint", return_value=True):
            worker._validate_payload(value)

    def test_auto_input_budget_obeys_final_multi_reference_contract(self):
        self.assertEqual(worker._auto_input_max_pixels(1), 2048 * 2048)
        self.assertEqual(worker._auto_input_max_pixels(2), 2048 * 2048)
        self.assertEqual(worker._auto_input_max_pixels(4), 2048 * 2048)
        self.assertEqual(worker._auto_input_max_pixels(16), 1024 * 1024)
        self.assertEqual(worker._auto_input_max_pixels(64), 512 * 512)
        self.assertEqual(worker._auto_input_max_pixels(100), 512 * 512)
        self.assertEqual(worker._effective_input_max_pixels(64, 2048 * 2048), 512 * 512)
        self.assertEqual(worker._effective_input_max_pixels(2, 1024 * 1024), 1024 * 1024)

    def test_explicit_output_size_does_not_call_resize(self):
        payload = {"width": 1024, "height": 1536}
        size = worker._resolve_output_size(payload, [], lambda **_: self.fail("unexpected resize"))
        self.assertEqual(size, (1024, 1536))

    def test_automatic_output_size_uses_original_first_reference_ratio(self):
        calls = []

        def smart_resize(**kwargs):
            calls.append(kwargs)
            return 1536, 2752

        size = worker._resolve_output_size(
            {"target_pixels": 2048 * 2048},
            [(1920, 1080)],
            smart_resize,
        )

        self.assertEqual(size, (2752, 1536))
        self.assertEqual(calls[0]["width"], 1920)
        self.assertEqual(calls[0]["height"], 1080)

    def test_aspect_fit_keeps_subject_geometry_inside_model_canvas(self):
        content_size = worker._aspect_fit_size(1600, 1200, 576, 416)
        self.assertEqual(content_size, (555, 416))
        self.assertAlmostEqual(content_size[0] / content_size[1], 4 / 3, delta=0.001)

        source = Image.new("RGB", (4, 2))
        for y in range(source.height):
            for x in range(source.width):
                source.putpixel((x, y), (x * 60, y * 120, 20))
        prepared = worker._resize_with_edge_padding(source, 4, 4)

        self.assertEqual(prepared.size, (4, 4))
        self.assertEqual(
            [prepared.getpixel((x, 0)) for x in range(prepared.width)],
            [prepared.getpixel((x, 1)) for x in range(prepared.width)],
        )
        self.assertEqual(
            [prepared.getpixel((x, 3)) for x in range(prepared.width)],
            [prepared.getpixel((x, 2)) for x in range(prepared.width)],
        )

    def test_load_images_returns_original_sizes_before_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.png"
            Image.new("RGB", (160, 120), (20, 40, 60)).save(path)
            images, original_sizes = worker._load_images(
                [str(path)],
                smart_resize=lambda **_: (416, 576),
                input_max_pixels=512 * 512,
            )

        self.assertEqual(original_sizes, [(160, 120)])
        self.assertEqual([image.size for image in images], [(576, 416)])

    def test_edit_payload_accepts_multiple_existing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.png"
            second = Path(temp) / "second.png"
            checkpoint = Path(temp) / "model.safetensors"
            Image.new("RGB", (16, 16), (255, 0, 0)).save(first)
            Image.new("RGB", (16, 16), (0, 0, 255)).save(second)
            checkpoint.write_bytes(b"test")
            self.validate_payload(
                {
                    "mode": "edit",
                    "prompt": "combine",
                    "generation_profile": worker.PROFILE_QUALITY,
                    "input_images": [str(first), str(second)],
                    "quantization": "int8_convrot",
                    "model_path": worker.FINAL_MODEL_ID,
                    "checkpoint_revision": worker.CHECKPOINT_REVISION,
                    "checkpoint": str(checkpoint),
                    "width": 1024,
                    "height": 1024,
                    "input_max_pixels": 512 * 512,
                    "vram_mode": "low",
                }
            )

    def test_official_8step_payload_keeps_fixed_asset_and_accepts_user_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model.safetensors"
            lora = root / "official-8step.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            lora.write_bytes(b"lora")
            payload = {
                "mode": "text",
                "prompt": "A lighthouse",
                "input_images": [],
                "quantization": "int8_convrot",
                "model_path": worker.FINAL_MODEL_ID,
                "checkpoint_revision": worker.CHECKPOINT_REVISION,
                "checkpoint": str(checkpoint),
                "width": 512,
                "height": 512,
                "generation_profile": worker.PROFILE_OFFICIAL_8STEP,
                "cfg_norm": "none",
                "steps": 8,
                "cfg_scale": 1.0,
                "timestep_shift": 3.0,
                "lora_path": str(lora),
                "lora_revision": worker.OFFICIAL_LORA_REVISION,
                "lora_sha256": worker.OFFICIAL_LORA_SHA256,
                "vram_mode": "low",
            }
            with (
                mock.patch.object(worker, "OFFICIAL_LORA_EXPECTED_BYTES", lora.stat().st_size),
                mock.patch.object(worker, "_sha256", return_value=worker.OFFICIAL_LORA_SHA256),
            ):
                self.validate_payload(payload)
                payload.update(cfg_norm="global", steps=7, cfg_scale=2.5, timestep_shift=2.0)
                self.validate_payload(payload)
                reference = root / "reference.png"
                Image.new("RGB", (32, 32)).save(reference)
                self.validate_payload(
                    {**payload, "mode": "edit", "input_images": [str(reference)], "input_max_pixels": 512 * 512}
                )
                for field in ("lora_revision", "lora_sha256"):
                    with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "LoRA"):
                        self.validate_payload({**payload, field: "changed"})

    def test_ui_sampling_values_reach_the_official_lora_sampling_call(self):
        from modules_forge import sensenova_u15_bridge as bridge
        from tools.tests.test_sensenova_u15_studio import load_studio_module

        studio = load_studio_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "selected.safetensors"
            lora = root / "official-8step.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            lora.write_bytes(b"lora")
            with mock.patch.object(bridge, "_official_lora_is_ready", return_value=True):
                request = studio._request_from_ui(
                    mode=bridge.MODE_TEXT,
                    prompt="A lighthouse",
                    gallery=[],
                    model_path=bridge.DEFAULT_MODEL_ID,
                    quantization=bridge.QUANT_INT8_CONVROT,
                    checkpoint_path=str(checkpoint),
                    source_path=str(root),
                    resolution="512x512",
                    input_max_pixels="auto",
                    generation_profile=bridge.PROFILE_OFFICIAL_8STEP,
                    steps=12,
                    cfg_scale=2.5,
                    img_cfg_scale=1.0,
                    timestep_shift=2.0,
                    seed=42,
                    vram_mode="low",
                    attn_backend="sdpa",
                    dtype="bfloat16",
                    cfg_norm="global",
                )
            with mock.patch.object(bridge, "DEFAULT_LORA_PATH", lora):
                payload = bridge._request_payload(
                    request, input_paths=[], output_path=root / "output.png", metadata_path=root / "output.json"
                )
            payload.update(device="cpu")
            engine = SimpleNamespace(
                generate=mock.Mock(return_value=torch.zeros(1, 3, 512, 512)),
                edit=mock.Mock(return_value=torch.zeros(1, 3, 512, 512)),
                release_variant="final",
                pruned_lm_head=True,
            )
            with (
                mock.patch.object(bridge, "compatible_checkpoint", return_value=True),
                mock.patch.object(worker, "OFFICIAL_LORA_EXPECTED_BYTES", lora.stat().st_size),
                mock.patch.object(worker, "_sha256", return_value=worker.OFFICIAL_LORA_SHA256),
                mock.patch.object(
                    worker,
                    "_runtime_for_request",
                    return_value=(
                        SimpleNamespace(),
                        lambda **kwargs: (512, 512),
                        engine,
                        1,
                        588,
                        {"targets": 294, "training_task": "t2i"},
                        0.0,
                        {},
                    ),
                ),
                mock.patch.object(worker, "emit"),
            ):
                reference = root / "reference.png"
                Image.new("RGB", (32, 32)).save(reference)
                for mode in ("text", "edit"):
                    with self.subTest(mode=mode):
                        payload.update(
                            mode=mode,
                            input_images=[str(reference)] if mode == "edit" else [],
                            input_max_pixels=512 * 512,
                        )
                        result = worker.run_request(payload)
                        sampling = (engine.generate if mode == "text" else engine.edit).call_args.kwargs
                        self.assertEqual(
                            (
                                sampling["num_steps"],
                                sampling["cfg_scale"],
                                sampling["timestep_shift"],
                                sampling["cfg_norm"],
                            ),
                            (12, 2.5, 2.0, "global"),
                        )
                        metadata = result["metadata"]
                        self.assertEqual(
                            (metadata["steps"], metadata["cfg_scale"], metadata["timestep_shift"]), (12, 2.5, 2.0)
                        )
                        self.assertEqual(metadata["cfg_norm"], "global")
                        self.assertEqual(payload["checkpoint"], str(checkpoint.resolve()))
                        self.assertEqual(metadata["checkpoint_file"], checkpoint.name)
                        self.assertEqual(metadata["mode"], mode)
                        self.assertEqual(metadata["official_8step_lora"]["training_task"], "t2i")

    def test_edit_applies_native_lora_hooks_without_changing_its_training_task(self):
        from contextlib import nullcontext

        from safetensors.torch import save_file

        runtime = ROOT / "models/SenseNova-U1/runtime-final/SenseNova/lora_runtime.py"
        if not runtime.is_file():
            self.skipTest("installed pinned SenseNova runtime is unavailable")
        spec = importlib.util.spec_from_file_location("_sensenova_native_lora_test", runtime)
        native_lora = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {spec.name: native_lora}):
            spec.loader.exec_module(native_lora)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.target = torch.nn.Linear(2, 3, bias=False)
                torch.nn.init.zeros_(self.target.weight)
                self.calls = []

            def it2i_generate(self, tokenizer, prompt, images, **kwargs):
                self.calls.append((tokenizer, prompt, images, kwargs))
                return self.target(torch.ones(1, 2))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "8step.safetensors"
            save_file(
                {
                    "target.lora_down.weight": torch.ones(1, 2),
                    "target.lora_up.weight": torch.ones(3, 1),
                    "target.alpha": torch.tensor(1.0),
                },
                str(path),
                metadata={"comfyui_sensenova_task": "t2i"},
            )
            lora = native_lora.load_sensenova_lora(str(path))
            model = Model()
            engine = SimpleNamespace(
                lora=lora,
                model=model,
                tokenizer=object(),
                _execution_lock=threading.RLock(),
                _lora_ctx=lambda: lora.apply(model, torch.float32),
                _model_ctx=mock.Mock(side_effect=lambda _: nullcontext(model)),
                edit=mock.Mock(side_effect=AssertionError("native task metadata gate must be bypassed")),
            )
            reference = Image.new("RGB", (32, 32))
            for prefetch in (None, 1):
                with self.subTest(prefetch=prefetch):
                    result = worker._edit_with_lora(
                        engine,
                        "Recolor the coat",
                        [reference],
                        streaming_prefetch_count=prefetch,
                        image_size=(512, 512),
                        num_steps=12,
                        cfg_scale=2.5,
                        cfg_norm="global",
                        timestep_shift=2.0,
                        img_cfg_scale=1.5,
                    )
                    self.assertTrue(torch.equal(result, torch.full((1, 3), 2.0)))
                    self.assertEqual(lora.task, "t2i")
                    self.assertEqual(lora.metadata["comfyui_sensenova_task"], "t2i")
                    self.assertTrue(torch.equal(model.target(torch.ones(1, 2)), torch.zeros(1, 3)))
                    self.assertFalse(model.target._forward_hooks)
                    call = model.calls[-1]
                    self.assertIs(call[0], engine.tokenizer)
                    self.assertIs(call[2][0], reference)
                    self.assertEqual((call[3]["num_steps"], call[3]["cfg_scale"]), (12, 2.5))
            engine._model_ctx.assert_called_once_with(1)

    def test_low_vram_payload_rejects_uncapped_four_megapixel_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "reference.png"
            checkpoint = root / "model.safetensors"
            Image.new("RGB", (16, 16)).save(image)
            checkpoint.write_bytes(b"test")
            payload = {
                "mode": "edit",
                "prompt": "combine",
                "generation_profile": worker.PROFILE_QUALITY,
                "input_images": [str(image), str(image)],
                "quantization": "int8_convrot",
                "model_path": worker.FINAL_MODEL_ID,
                "checkpoint_revision": worker.CHECKPOINT_REVISION,
                "checkpoint": str(checkpoint),
                "width": 1664,
                "height": 2496,
                "input_max_pixels": "auto",
                "vram_mode": "low",
            }
            with self.assertRaisesRegex(RuntimeError, "24 GB safe profile"):
                self.validate_payload(payload)

            payload["vram_mode"] = "unrestricted"
            self.validate_payload(payload)

            payload.update(
                vram_mode="reference",
                width=2048,
                height=2048,
                input_max_pixels=1024 * 1024,
                input_images=[str(image)] * 8,
            )
            self.validate_payload(payload)
            for changes in (
                {"input_images": [str(image)] * 9},
                {"input_max_pixels": "auto"},
                {"input_max_pixels": 2048 * 2048},
                {"width": 4096},
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "Reference profile"):
                    self.validate_payload({**payload, **changes})

            payload.update(
                {
                    "vram_mode": "low",
                    "width": 2048,
                    "height": 2048,
                    "input_max_pixels": 512 * 512,
                    "input_images": [str(image), str(image)],
                }
            )
            self.validate_payload(payload)

            payload.update(
                {
                    "vram_mode": "low",
                    "width": 1024,
                    "height": 1024,
                    "input_max_pixels": 512 * 512,
                    "input_images": [str(image), str(image), str(image)],
                }
            )
            with self.assertRaisesRegex(RuntimeError, "at most 2 references"):
                self.validate_payload(payload)

    def test_quality_profile_rejects_all_hidden_lora_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.safetensors"
            checkpoint.write_bytes(b"test")
            payload = {
                "mode": "text",
                "prompt": "A lighthouse",
                "generation_profile": worker.PROFILE_QUALITY,
                "input_images": [],
                "quantization": "int8_convrot",
                "model_path": worker.FINAL_MODEL_ID,
                "checkpoint_revision": worker.CHECKPOINT_REVISION,
                "checkpoint": str(checkpoint),
                "width": 512,
                "height": 512,
                "vram_mode": "low",
            }
            missing_profile = dict(payload)
            missing_profile.pop("generation_profile")
            with self.assertRaisesRegex(RuntimeError, "generation_profile is required"):
                self.validate_payload(missing_profile)
            self.validate_payload(payload)
            for field in ("lora_path", "lora_revision", "lora_sha256"):
                payload[field] = "hidden"
                with self.assertRaisesRegex(RuntimeError, "hidden LoRA provenance"):
                    self.validate_payload(payload)
                payload[field] = ""

    def test_normalized_tensor_is_converted_to_rgb_image(self):
        batch = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])
        image = worker._to_pil(batch)[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 255))

    def test_unit_range_nhwc_tensor_is_converted_to_rgb_image(self):
        batch = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        image = worker._to_pil(batch)[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 255))


if __name__ == "__main__":
    unittest.main()
