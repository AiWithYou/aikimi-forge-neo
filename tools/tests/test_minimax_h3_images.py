"""CPU/offline contracts for H3 images; no model download or GPU claim."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge import minimax_h3_images as images

MODELS = {key: f"{key}.safetensors" for key in ("fl2va", "ref2va", "clip", "vae", "audio_vae")}
SCHEMAS = {
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}}},
    "ImageFromBatch": {"input": {"required": {"image": ["IMAGE"], "batch_index": ["INT"], "length": ["INT"]}}},
}


def readiness():
    return SimpleNamespace(
        ready_for_ref2va=True, ram_free_gib=32.0, commit_free_gib=32.0,
        node_schemas={}, core_revision="a" * 40, comfy_version="test",
        package_versions={"comfy-kitchen": "test"}, runtime_profile="fast",
    )


def fake_bridge():
    class Error(RuntimeError):
        pass

    class Missing(Error):
        pass

    client = Mock()
    client.submit.return_value = "job-1"
    client.object_info.return_value = SCHEMAS
    bridge = SimpleNamespace(
        H3BridgeError=Error, H3JobNotFound=Missing,
        H3Request=lambda **kw: SimpleNamespace(**kw),
        H3_FL_MODEL=MODELS["fl2va"], H3_REF_MODEL=MODELS["ref2va"],
        H3_TEXT_ENCODER=MODELS["clip"], H3_VIDEO_VAE=MODELS["vae"],
        H3_AUDIO_VAE=MODELS["audio_vae"], RUNTIME_PROFILE_FAST="fast",
        resolve_runtime_root=lambda value: Path(value), normalize_loopback_url=lambda value: value,
        ensure_ready=Mock(return_value=readiness()), validate_readiness=Mock(),
        cleanup_stale_prepared_media=Mock(), prepare_media=Mock(return_value={"images": []}),
        cleanup_prepared_media=Mock(), _RUNTIME_LIFECYCLE_LOCK=threading.RLock(),
        ComfyH3Client=Mock(return_value=client), _mark_active_generation=Mock(),
        _clear_active_generation=Mock(), _clear_cancelled_job=Mock(),
        _is_cancelled_job=Mock(return_value=False), _schedule_deferred_cleanup=Mock(),
        _execution_error=lambda job: "generation failed",
    )
    return bridge, client


class RequestTests(unittest.TestCase):
    def test_default_request(self):
        images.H3ImageRequest("A still photograph.").validate()

    def test_invalid_inputs(self):
        cases = [
            {"prompt": " "}, {"prompt": "x" * 20001}, {"mode": "keyframes"},
            {"width": 770}, {"width": 128}, {"height": 2080}, {"width": True},
            {"steps": 0}, {"steps": 1.5}, {"steps": float("inf")}, {"steps": True},
            {"seed": -2}, {"seed": 2**63}, {"seed": float("nan")},
            {"frame_index": 5}, {"frame_index": -1}, {"frame_index": 1.5},
            {"scheduler": "unexpected"}, {"save_candidates": "yes"},
            {"reference_images": ["x.png"]}, {"reference_images": ("",)},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(images.H3ImageError):
                replace(images.H3ImageRequest("test"), **values).validate()

    def test_seed_integer_parser_preserves_64_bit_values(self):
        self.assertEqual(images.integer(str(2**63 - 1), "seed", -1, 2**63 - 1), 2**63 - 1)
        for bad in ("1.1", "1e3", float("inf"), float("nan"), True, None, "x"):
            with self.subTest(value=bad), self.assertRaises(images.H3ImageError):
                images.integer(bad, "seed", -1, 10000)

    def test_explicit_seed_is_reproducible(self):
        self.assertEqual(images.H3ImageRequest("x", seed=42).resolved_seed(), 42)
        with patch.object(images.secrets, "randbelow", return_value=123):
            self.assertEqual(images.H3ImageRequest("x").resolved_seed(), 123)

    def test_references_require_all_tags_and_at_most_nine_images(self):
        request = images.H3ImageRequest("Edit <Picture 1> using <Picture 2>.", mode="references", reference_images=("a.png", "b.png"))
        request.validate()
        for values in (
            {"prompt": "Edit <Picture 1>."}, {"prompt": "<picture 1> <Picture 2>"},
            {"prompt": "<Picture 1> <Picture 2> <Video 1>"}, {"reference_images": ()},
            {"reference_images": ("a.png",) * 10}, {"mode": "text"},
        ):
            with self.subTest(values=values), self.assertRaises(images.H3ImageError):
                replace(request, **values).validate()


class WorkflowTests(unittest.TestCase):
    def graph(self, **kwargs):
        request = images.H3ImageRequest("test", **kwargs)
        return images.build_image_workflow(request, {}, 42, models=MODELS)

    def test_text_graph_uses_native_short_av_latent_without_video_output(self):
        graph = self.graph(width=1024, height=768)
        condition = graph["condition"]
        self.assertEqual(condition["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(condition["inputs"]["length"], 5)
        self.assertEqual(condition["inputs"]["width"], 1024)
        self.assertEqual(condition["inputs"]["height"], 768)
        self.assertEqual(graph["sample"]["inputs"]["latent_image"], ["condition", 1])
        self.assertEqual(graph["save"]["class_type"], "SaveImage")
        kinds = {n["class_type"] for n in graph.values()}
        self.assertTrue(kinds.isdisjoint({"SaveVideo", "CreateVideo", "VAEDecodeAudio"}))
        self.assertNotIn("audio_vae", graph)
        self.assertEqual(graph["noise"]["inputs"]["noise_seed"], 42)

    def test_all_graph_edges_exist(self):
        graph = self.graph(save_candidates=True)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)
                    self.assertIsInstance(value[1], int)

    def test_candidates_are_opt_in_and_selection_is_explicit(self):
        self.assertNotIn("save_candidates", self.graph())
        graph = self.graph(frame_index=4, save_candidates=True)
        self.assertEqual(graph["select"]["inputs"]["batch_index"], 4)
        self.assertEqual(graph["select"]["inputs"]["length"], 1)
        self.assertEqual(graph["candidates"]["inputs"]["length"], 5)

    def test_reference_edit_uses_ref2va_not_first_frame_anchor(self):
        request = images.H3ImageRequest("Edit <Picture 1> using <Picture 2>", mode="references", reference_images=("one", "two"))
        graph = images.build_image_workflow(request, {"images": ["forge_h3/a.png", "forge_h3/b.png"]}, 42, models=MODELS)
        self.assertEqual(graph["model"]["inputs"]["unet_name"], MODELS["ref2va"])
        self.assertEqual(graph["condition"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["condition"]["inputs"]["ref_images.ref_image_0"], ["reference_0", 0])
        self.assertEqual(graph["condition"]["inputs"]["ref_images.ref_image_1"], ["reference_1", 0])
        self.assertNotIn("first_frame", graph["condition"]["inputs"])
        with self.assertRaises(images.H3ImageError):
            images.build_image_workflow(request, {"images": []}, 42, models=MODELS)

    def test_image_node_probe_does_not_rely_on_video_readiness_subset(self):
        client = Mock()
        client.object_info.return_value = SCHEMAS
        images.check_image_nodes(client)
        client.object_info.assert_called_once_with({"SaveImage", "ImageFromBatch"}, timeout=8.0)
        client.object_info.return_value = {}
        with self.assertRaises(images.H3ImageError):
            images.check_image_nodes(client)

    def test_schema_drift_is_blocked(self):
        client = Mock()
        client.object_info.return_value = {"SaveImage": {"input": {}}, "ImageFromBatch": SCHEMAS["ImageFromBatch"]}
        with self.assertRaises(images.H3ImageError):
            images.check_image_nodes(client)

    def test_runtime_guards_and_memory_are_retained(self):
        bridge, _ = fake_bridge()
        with patch.object(images, "_bridge", return_value=bridge):
            ready = readiness()
            images.validate_image_runtime(images.H3ImageRequest("x"), ready, "fast")
            bridge.validate_readiness.assert_called_once_with(ready, "fast")
            ready.ram_free_gib = 0.1
            with self.assertRaises(images.H3ImageError):
                images.validate_image_runtime(images.H3ImageRequest("x"), ready, "fast")
            ready.ram_free_gib = float("nan")
            with self.assertRaises(images.H3ImageError):
                images.validate_image_runtime(images.H3ImageRequest("x"), ready, "fast")

    def test_native_bridge_constants_when_repository_is_present(self):
        if importlib.util.find_spec("modules_forge.minimax_h3_bridge") is None:
            self.skipTest("Full Forge checkout is not present in this local workspace.")
        from modules_forge import minimax_h3_bridge as bridge
        self.assertEqual(images.model_names()["fl2va"], bridge.H3_FL_MODEL)
        graph = images.build_image_workflow(images.H3ImageRequest("test"), {}, 1)
        self.assertEqual(graph["clip"]["inputs"]["clip_name"], bridge.H3_TEXT_ENCODER)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "output" / "image").mkdir(parents=True)
        self.png = self.root / "output" / "image" / "test.png"
        Image.new("RGB", (256, 256)).save(self.png)
        self.request = images.H3ImageRequest("test", width=256, height=256)
        self.history = {"job": {"outputs": {"save": {"images": [{"filename": "test.png", "subfolder": "image", "type": "output"}]}}}}

    def test_extracts_only_the_designated_save_node(self):
        self.history["job"]["outputs"]["untrusted"] = {"images": [{"filename": "../bad.png"}]}
        primary, candidates = images.extract_image_outputs(self.history, "job", self.root, self.request)
        self.assertEqual(primary, [self.png.resolve()])
        self.assertEqual(candidates, [])

    def test_rejects_traversal_absolute_paths_and_non_output_files(self):
        for changes in (
            {"subfolder": "../"}, {"subfolder": "/etc"}, {"subfolder": "C:\\temp"},
            {"filename": "../test.png"}, {"filename": "x\\test.png"},
            {"filename": "test.jpg"}, {"type": "temp"}, {"filename": "missing.png"},
        ):
            with self.subTest(changes=changes):
                history = json.loads(json.dumps(self.history))
                history["job"]["outputs"]["save"]["images"][0].update(changes)
                with self.assertRaises(images.H3ImageError):
                    images.extract_image_outputs(history, "job", self.root, self.request)

    def test_symlink_escape_is_blocked(self):
        outside = self.root / "outside.png"
        Image.new("RGB", (256, 256)).save(outside)
        self.png.unlink()
        try:
            self.png.symlink_to(outside)
        except OSError:
            self.skipTest("Symlink creation is not permitted on this host.")
        with self.assertRaises(images.H3ImageError):
            images.extract_image_outputs(self.history, "job", self.root, self.request)

    def test_missing_candidate_outputs_are_not_success(self):
        with self.assertRaises(images.H3ImageError):
            images.extract_image_outputs(self.history, "job", self.root, replace(self.request, save_candidates=True))

    def save(self, request=None, candidates=None):
        return images.save_image_result([self.png], candidates or [], self.root / "results", request or self.request, 42, "job", readiness(), models=MODELS)

    def test_saves_png_and_parameters_without_absolute_input_paths(self):
        result = self.save()
        self.assertEqual(Path(result["path"]).read_bytes(), self.png.read_bytes())
        metadata = json.loads(Path(result["files"][-1]).read_text())
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["internal_frames"], 5)
        self.assertNotIn(str(self.root), json.dumps(metadata))
        self.assertFalse(list((self.root / "results").glob(".h3-image-*")))

    def test_candidate_batch_is_saved(self):
        candidates = []
        for i in range(5):
            path = self.root / f"candidate{i}.png"
            Image.new("RGB", (256, 256)).save(path)
            candidates.append(path)
        result = self.save(replace(self.request, save_candidates=True), candidates)
        self.assertEqual(len(result["candidates"]), 5)
        self.assertEqual(len(result["files"]), 7)

    def test_no_overwrite_on_repeated_save(self):
        first, second = self.save(), self.save()
        self.assertNotEqual(first["path"], second["path"])
        self.assertTrue(Path(first["path"]).is_file())

    def test_invalid_png_does_not_publish_result(self):
        self.png.write_bytes(b"not a PNG")
        with self.assertRaises(OSError):
            self.save()
        self.assertEqual(list((self.root / "results").iterdir()), [])

    def test_wrong_size_does_not_publish_result(self):
        Image.new("RGB", (512, 512)).save(self.png)
        with self.assertRaises(images.H3ImageError):
            self.save()
        self.assertEqual(list((self.root / "results").iterdir()), [])

    def test_failed_commit_cleans_staged_outputs(self):
        with patch.object(images.os, "replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual(list((self.root / "results").iterdir()), [])


class LifecycleTests(unittest.TestCase):
    def run_job(self, bridge):
        return images.run_image_generation(images.H3ImageRequest("x"), Path("runtime"), "http://127.0.0.1:8188", Path("logs"), Path("out"), poll_seconds=0.05)

    def test_success_releases_shared_lifecycle_and_cleans_inputs(self):
        bridge, client = fake_bridge()
        client.job.return_value = {"status": "completed"}
        with patch.object(images, "_bridge", return_value=bridge), patch.object(images, "extract_image_outputs", return_value=([Path("one.png")], [])), patch.object(images, "save_image_result", return_value={"path": "saved.png"}):
            events = list(self.run_job(bridge))
        self.assertEqual(events[-1]["stage"], "complete")
        bridge._mark_active_generation.assert_called_once_with("job-1")
        bridge._clear_active_generation.assert_called_once_with("job-1")
        bridge.cleanup_prepared_media.assert_called_once()
        client.cancel.assert_not_called()
        client.close.assert_called_once()

    def test_closing_generator_cancels_only_its_job_and_defers_cleanup(self):
        bridge, client = fake_bridge()
        with patch.object(images, "_bridge", return_value=bridge):
            generator = self.run_job(bridge)
            next(generator)
            next(generator)
            generator.close()
        client.cancel.assert_called_once_with("job-1")
        bridge._schedule_deferred_cleanup.assert_called_once()
        bridge.cleanup_prepared_media.assert_not_called()
        client.close.assert_not_called()
        bridge._clear_active_generation.assert_called_once_with("job-1")

    def test_cancelled_job_is_not_success(self):
        bridge, client = fake_bridge()
        client.job.return_value = {"status": "cancelled"}
        with patch.object(images, "_bridge", return_value=bridge), self.assertRaises(images.H3ImageCancelled):
            list(self.run_job(bridge))
        bridge.cleanup_prepared_media.assert_called_once()
        bridge._clear_cancelled_job.assert_called_once()

    def test_retries_poll_failure_but_stops_after_three(self):
        bridge, client = fake_bridge()
        client.job.side_effect = bridge.H3BridgeError("offline")
        with patch.object(images, "_bridge", return_value=bridge), patch.object(images.time, "sleep"), self.assertRaises(bridge.H3BridgeError):
            list(self.run_job(bridge))
        self.assertEqual(client.job.call_count, 3)
        client.cancel.assert_called_once_with("job-1")
        bridge._schedule_deferred_cleanup.assert_called_once()

    def test_readiness_failure_does_not_submit_or_copy(self):
        bridge, client = fake_bridge()
        bridge.validate_readiness.side_effect = bridge.H3BridgeError("wrong runtime")
        with patch.object(images, "_bridge", return_value=bridge), self.assertRaises(bridge.H3BridgeError):
            list(self.run_job(bridge))
        bridge.prepare_media.assert_not_called()
        client.submit.assert_not_called()

    def test_schema_failure_cleans_prepared_inputs_before_submission(self):
        bridge, client = fake_bridge()
        client.object_info.return_value = {}
        with patch.object(images, "_bridge", return_value=bridge), self.assertRaises(images.H3ImageError):
            list(self.run_job(bridge))
        client.submit.assert_not_called()
        bridge.cleanup_prepared_media.assert_called_once()
        client.close.assert_called_once()


class UITests(unittest.TestCase):
    def test_real_gradio_component_graph_without_starting_backend(self):
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("Gradio is not installed in this CPU test environment.")
        callbacks = SimpleNamespace(on_ui_tabs=Mock())
        modules = types.ModuleType("modules")
        modules.script_callbacks = callbacks
        paths = types.ModuleType("modules.paths")
        paths.data_path = paths.script_path = str(Path(__file__).resolve().parents[2])
        filename = Path(__file__).resolve().parents[2] / "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py"
        spec = importlib.util.spec_from_file_location("h3_image_ui_test", filename)
        ui = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules": modules, "modules.paths": paths, "modules.script_callbacks": callbacks}):
            spec.loader.exec_module(ui)
        with patch.object(ui, "_initial_runtime", return_value=""), patch.object(images, "_bridge") as bridge:
            tabs = ui.on_ui_tabs()
            bridge.assert_not_called()
        self.assertEqual(tabs[0][1:], ("H3 Image", "minimax_h3_image_studio"))
        config = tabs[0][0].get_config_file()
        self.assertTrue(config["components"])
        for dependency in config["dependencies"]:
            if dependency.get("backend_fn"):
                self.assertEqual(dependency["api_visibility"], "private")
        self.assertFalse(any(component["type"] == "video" for component in config["components"]))
        self.assertEqual(ui._preset("横長 · 1344×768"), (1344, 768))
        callbacks.on_ui_tabs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
