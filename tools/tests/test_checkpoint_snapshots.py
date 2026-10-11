"""Checkpoint transitions and immutable generation identities."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import local_assets, studio_checkpoints
from modules_forge.minimax_h3_acceleration import H3Acceleration


class CheckpointSnapshots(unittest.TestCase):
    def test_owned_restart_must_confirm_registration_for_custom_and_standard(self):
        from modules_forge import minimax_h3_bridge as bridge
        from tools.tests.test_studio_checkpoint_routes import checkpoint

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = checkpoint(
                root / "custom_fl2va.safetensors",
                ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
            )
            wrong = SimpleNamespace(
                runtime_args=("--extra-model-paths-config", str(root / ".neo-checkpoints-old.json"))
            )
            process = Mock()
            process.poll.return_value = None
            for path in (selected, ""):
                with (
                    self.subTest(checkpoint=path),
                    patch.object(bridge, "_MANAGED_PROCESS", process),
                    patch.object(bridge, "_MANAGED_PROCESS_IDENTITY", (root, "http://127.0.0.1:8189")),
                    patch.object(bridge, "_restart_runtime_locked", return_value=wrong),
                ):
                    with self.assertRaisesRegex(bridge.H3BridgeError, "登録"):
                        bridge.ensure_checkpoint_registration(
                            wrong, root, "http://127.0.0.1:8189", root, H3Acceleration(checkpoint=path)
                        )

    def test_h3_start_with_wrong_registered_root_fails_before_readiness_validation(self):
        from modules_forge import minimax_h3_bridge as bridge
        from tools.tests.test_studio_checkpoint_routes import checkpoint

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = checkpoint(
                root / "custom_fl2va.safetensors",
                ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
            )
            with (
                patch.object(bridge, "inspect_readiness", return_value=SimpleNamespace(connected=False)),
                patch.object(bridge, "start_runtime", return_value=SimpleNamespace(connected=True, runtime_args=())),
                patch.object(bridge, "validate_readiness") as validate,
            ):
                with self.assertRaisesRegex(bridge.H3BridgeError, "登録"):
                    bridge._ensure_ready_locked(
                        root, "http://127.0.0.1:8189", root, acceleration=H3Acceleration(checkpoint=selected)
                    )
                validate.assert_not_called()

    def test_nanosaur_start_with_wrong_registered_root_fails_before_node_validation(self):
        from modules_forge import minimax_h3_bridge as bridge
        from modules_forge import nanosaur2_studio as nano

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = root / "nanosaur.safetensors"
            selected.write_bytes(b"exists")
            with (
                patch("tools.setup_nanosaur2.runtime_ready", return_value=True),
                patch.object(nano, "source_ready", return_value=True),
                patch.object(nano, "model_ready", return_value=True),
                patch.object(bridge, "resolve_runtime_root", return_value=root),
                patch.object(bridge, "inspect_readiness", return_value=SimpleNamespace(connected=False)),
                patch.object(bridge, "start_runtime", return_value=SimpleNamespace(connected=True, runtime_args=())),
                patch.object(nano, "check_nodes") as check,
            ):
                with self.assertRaisesRegex(bridge.H3BridgeError, "登録"):
                    nano.ensure_runtime(selected_root=root, checkpoint=str(selected))
                check.assert_not_called()

    def test_late_verification_failure_releases_all_comfy_leases_without_submission(self):
        from contextlib import ExitStack

        from modules_forge import minimax_h3_bridge as h3
        from modules_forge import minimax_h3_images as images
        from modules_forge import nanosaur2_studio as nano
        from tools.tests.test_minimax_h3_images import fake_bridge, readiness

        for family in ("h3", "image", "nano"):
            with self.subTest(family=family), TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                bridge, client = fake_bridge()
                if family == "h3":
                    bridge = h3
                    stack.enter_context(patch.object(bridge, "_GPU_OWNERSHIPS", {}))
                    for name, value in {
                        "ensure_ready": readiness(),
                        "prepare_media": {},
                        "build_workflow": {},
                        "ComfyH3Client": client,
                        "_is_cancelled_job": False,
                    }.items():
                        stack.enter_context(patch.object(bridge, name, return_value=value))
                    for name in (
                        "release_forge_vram",
                        "_validate_request_runtime_constraints",
                        "cleanup_stale_prepared_media",
                        "cleanup_prepared_media",
                        "_schedule_deferred_cleanup",
                        "_clear_cancelled_job",
                    ):
                        stack.enter_context(patch.object(bridge, name))
                    stack.enter_context(patch.object(bridge, "pending_jobs", Mock()))
                    stack.enter_context(
                        patch.object(bridge.workflow_handoff, "capture_generation", return_value=(None, ""))
                    )
                else:
                    stack.enter_context(
                        patch.object(images if family == "image" else nano, "_bridge", return_value=bridge)
                    )
                state = {"changed": False}

                def process(_, state=state):
                    state["changed"] = True
                    return SimpleNamespace(pid=1, create_time=lambda: 1.0)

                def verify(_, state=state):
                    if state["changed"]:
                        raise ValueError("選択したチェックポイントが変更されました。")

                stack.enter_context(patch.object(bridge, "_loopback_server_process", side_effect=process))
                stack.enter_context(patch.object(studio_checkpoints, "verify_snapshot", side_effect=verify))
                stack.enter_context(patch("modules_forge.gpu_residency.register"))
                ownership = Mock()
                ownership.acquire.return_value = True
                stack.enter_context(patch.object(h3, "GPUOwnership", return_value=ownership))
                stack.enter_context(patch("modules_forge.gpu_ownership.GPUOwnership", return_value=ownership))
                if family == "h3":
                    events = h3.run_generation(
                        h3.H3Request(mode="text", prompt="test"), root, "http://127.0.0.1:8189", root, root
                    )
                elif family == "image":
                    stack.enter_context(patch.object(images, "build_image_workflow", return_value={}))
                    events = images.run_image_generation(
                        images.H3ImageRequest("test"), root, "http://127.0.0.1:8189", root, root
                    )
                else:
                    stack.enter_context(patch.object(nano, "ensure_runtime", return_value=readiness()))
                    stack.enter_context(patch.object(nano, "build_workflow", return_value={}))
                    events = nano.run_generation(nano.Nanosaur2Request("test"), output_root=root)
                with self.assertRaisesRegex(ValueError, "変更"):
                    list(events)
                client.submit.assert_not_called()
                bridge.pending_jobs.write.assert_not_called()
                bridge._schedule_deferred_cleanup.assert_not_called()
                self.assertEqual(bridge._GPU_OWNERSHIPS, {})
                ownership.release.assert_called_once()

    def test_h3_profile_accepts_selected_checkpoint_registration(self):
        from modules_forge import minimax_h3_bridge as bridge

        command = bridge._runtime_command(Path("python.exe"), 8189, "fast")
        arguments = command[1:] + ["--extra-model-paths-config", "other.yaml", ".neo-checkpoints-selected.json"]
        self.assertEqual(bridge.runtime_profile_from_args(arguments, expected_port=8189), "fast")

    def test_standard_without_runtime_arguments_keeps_fixture_compatibility(self):
        self.assertTrue(studio_checkpoints.config_loaded(SimpleNamespace(), Path("."), ""))

    def test_standard_requires_removing_only_neo_checkpoint_configuration(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            custom = root / "custom" / "same.safetensors"
            custom.parent.mkdir()
            custom.write_bytes(b"custom")
            config = studio_checkpoints.write_comfy_config(root, str(custom))
            readiness = SimpleNamespace(runtime_args=("--extra-model-paths-config", "other.yaml", str(config)))
            self.assertFalse(studio_checkpoints.config_loaded(readiness, root, ""))
            self.assertTrue(
                studio_checkpoints.config_loaded(
                    SimpleNamespace(runtime_args=("--extra-model-paths-config", "other.yaml")), root, ""
                )
            )

    def test_equal_sign_comfy_config_is_checked_and_user_config_is_preserved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = root / "same.safetensors"
            selected.write_bytes(b"custom")
            config = studio_checkpoints.write_comfy_config(root, str(selected))
            readiness = SimpleNamespace(
                runtime_args=(
                    f"--extra-model-paths-config={config}",
                    "--extra-model-paths-config=other.yaml",
                    "--other",
                    str(root / ".neo-checkpoints-unrelated.json"),
                )
            )
            self.assertTrue(studio_checkpoints.config_loaded(readiness, root, str(selected)))
            self.assertFalse(studio_checkpoints.config_loaded(readiness, root, ""))
            self.assertEqual(studio_checkpoints.user_comfy_configs(readiness), ("other.yaml",))

    def test_owned_runtime_restarts_when_returning_to_standard(self):
        from modules_forge import minimax_h3_bridge as bridge

        with TemporaryDirectory() as folder:
            root = Path(folder)
            readiness = SimpleNamespace(
                runtime_args=("--extra-model-paths-config", "other.yaml", str(root / ".neo-checkpoints-old.json"))
            )
            process = Mock()
            process.poll.return_value = None
            with (
                patch.object(bridge, "_MANAGED_PROCESS", process),
                patch.object(bridge, "_MANAGED_PROCESS_IDENTITY", (root, "http://127.0.0.1:8189")),
                patch.object(bridge, "_restart_runtime_locked", return_value="standard") as restart,
            ):
                result = bridge.ensure_checkpoint_registration(
                    readiness, root, "http://127.0.0.1:8189", root, H3Acceleration()
                )
            self.assertEqual(result, "standard")
            self.assertEqual(restart.call_args.kwargs["acceleration"].checkpoint, "")
            self.assertEqual(restart.call_args.kwargs["extra_model_configs"], ("other.yaml",))

    def test_external_runtime_requires_explicit_removal_when_returning_to_standard(self):
        from modules_forge import minimax_h3_bridge as bridge

        with TemporaryDirectory() as folder:
            root = Path(folder)
            readiness = SimpleNamespace(
                runtime_args=("--extra-model-paths-config", str(root / ".neo-checkpoints-old.json"))
            )
            with (
                patch.object(bridge, "_MANAGED_PROCESS", None),
                patch.object(bridge, "_restart_runtime_locked") as restart,
            ):
                with self.assertRaisesRegex(bridge.H3BridgeError, "外部ComfyUI"):
                    bridge.ensure_checkpoint_registration(
                        readiness, root, "http://127.0.0.1:8189", root, H3Acceleration()
                    )
                restart.assert_not_called()

    def test_snapshot_rejects_overwrite_and_retains_original_identity(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "selected.safetensors"
            path.write_bytes(b"first")
            snapshot = studio_checkpoints.snapshot_checkpoint(str(path))
            original = dict(snapshot["identity"])
            path.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "変更"):
                studio_checkpoints.verify_snapshot(snapshot)
            self.assertEqual(snapshot["identity"], original)

    def test_snapshot_rehashes_without_reliable_change_counter(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "selected.safetensors"
            path.write_bytes(b"first")
            with patch.object(local_assets, "file_version", return_value=(5, 0, None, 1)):
                snapshot = studio_checkpoints.snapshot_checkpoint(str(path))
                path.write_bytes(b"other")
                with self.assertRaisesRegex(ValueError, "変更"):
                    studio_checkpoints.verify_snapshot(snapshot)

    def test_yue_worker_uses_supervisor_entry_and_snapshot(self):
        from dataclasses import asdict

        from modules_forge.yue2_studio import worker
        from modules_forge.yue2_studio.core import Request

        with TemporaryDirectory() as folder:
            root = Path(folder)
            models = root / "models"
            models.mkdir()
            (models / "weights").write_bytes(b"first")
            snapshot = studio_checkpoints.snapshot_checkpoint(str(models), directory=True)
            entry = {"model": str(models), "vae": "shared", "checkpoint_identity": snapshot["identity"]}
            request = Request(style="piano", lyrics="lyrics", seed=42, checkpoint=str(models))
            (root / "project.json").write_text(
                json.dumps({"request": asdict(request), "runtime_entry": entry, "checkpoint_snapshot": snapshot})
            )
            with (
                patch.object(worker, "runtime_manifest", side_effect=AssertionError("must not reread registration")),
                patch.object(worker, "native") as run,
            ):
                worker.resident_run({"job": str(root), "runtime": str(root)})
            self.assertEqual(run.call_args.args[1], entry)

    def test_yue_job_keeps_start_request_and_entry_after_ui_selection_changes(self):
        from dataclasses import replace

        from modules_forge.yue2_studio import service, worker
        from modules_forge.yue2_studio.core import Request

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = root / "selected"
            selected.mkdir()
            (selected / "weights").write_bytes(b"first")
            request = Request(style="piano", lyrics="lyrics", seed=42, checkpoint=str(selected))
            lease = Mock()
            lease.acquire.return_value = True
            studio = service.Studio(root / "runtime", root / "outputs", ownership_factory=lambda: lease)
            studio._runtime_lock = Mock()
            try:
                with (
                    patch.object(
                        service,
                        "runtime_manifest",
                        return_value={"model": str(selected), "vae": "shared", "python": "dedicated-python"},
                    ) as manifest,
                    patch.object(service.threading, "Thread"),
                ):
                    identifier = studio.start(request, "owner")
                manifest.assert_called_once_with(studio.runtime, "official", str(selected))
                request = replace(request, checkpoint="")
                job = studio._jobs[identifier]
                with (
                    patch.object(worker, "runtime_manifest", side_effect=AssertionError("registration changed")),
                    patch.object(worker, "native") as run,
                ):
                    worker.resident_run({"job": str(job.directory), "runtime": str(studio.runtime)})
                self.assertEqual(request.checkpoint, "")
                self.assertEqual(run.call_args.args[0].checkpoint, str(selected))
                self.assertEqual(run.call_args.args[1], job.runtime_entry)
                self.assertEqual(run.call_args.kwargs["checkpoint_snapshot"], job.checkpoint_snapshot)
            finally:
                studio.shutdown()

    def test_h3_snapshot_is_taken_before_gpu_wait_and_rejects_overwrite(self):
        from modules_forge import minimax_h3_bridge as bridge
        from tools.tests.test_studio_checkpoint_routes import checkpoint

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = checkpoint(
                root / "custom_fl2va.safetensors",
                ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
            )
            request = bridge.H3Request(mode="text", prompt="test", acceleration=H3Acceleration(checkpoint=selected))
            ownership = Mock()
            ownership.acquire.side_effect = [False, True]
            with (
                patch.object(bridge, "GPUOwnership", return_value=ownership),
                patch.object(bridge, "release_forge_vram"),
                patch.object(bridge, "ensure_ready") as ready,
            ):
                events = bridge.run_generation(request, root, "http://127.0.0.1:8189", root, root)
                self.assertEqual(next(events)["stage"], "queued")
                Path(selected).write_bytes(b"replacement")
                with self.assertRaisesRegex(ValueError, "変更"):
                    next(events)
                ready.assert_not_called()
            ownership.release.assert_called_once()

    def test_all_comfy_result_metadata_uses_the_start_identity(self):
        from PIL import Image

        from modules_forge import minimax_h3_bridge as bridge
        from modules_forge import minimax_h3_images as images
        from modules_forge import nanosaur2_studio as nano
        from tools.tests.test_studio_checkpoint_routes import checkpoint

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = checkpoint(
                root / "custom_fl2va.safetensors",
                ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
            )
            snapshot = studio_checkpoints.snapshot_checkpoint(selected)
            readiness = SimpleNamespace(
                core_revision="test", comfy_version="test", package_versions={}, runtime_profile="fast"
            )
            source = root / "source.png"
            Image.new("RGB", (256, 256)).save(source)
            video = root / "source.mp4"
            video.write_bytes(b"fake video")
            video_request = bridge.H3Request(
                mode="text", prompt="test", acceleration=H3Acceleration(checkpoint=selected)
            )
            image_request = images.H3ImageRequest("test", width=256, height=256, checkpoint=selected)
            nano_request = nano.Nanosaur2Request("test", width=256, height=256, checkpoint=selected)
            with (
                patch.object(local_assets, "file_identity", side_effect=AssertionError("must retain start identity")),
                patch.object(bridge, "_validate_video_result", return_value={}),
            ):
                video_result = bridge.mirror_result(
                    video, root / "videos", video_request, "job", 42, readiness, checkpoint_snapshot=snapshot
                )
                image_result = images.save_image_result(
                    [source], [], root / "images", image_request, 42, "job", readiness, checkpoint_snapshot=snapshot
                )
                nano_result = nano.save_result(
                    source, nano_request, 42, "job", readiness, root / "nano", checkpoint_snapshot=snapshot
                )
            self.assertEqual(
                json.loads(video_result.with_suffix(".json").read_text(encoding="utf-8"))["checkpoint_identity"],
                snapshot["identity"],
            )
            self.assertEqual(image_result["metadata"]["checkpoint_identity"], snapshot["identity"])
            self.assertEqual(nano_result["metadata"]["checkpoint"], snapshot["identity"])
            Path(selected).write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "変更"):
                images.save_image_result(
                    [source], [], root / "rejected", image_request, 42, "job", readiness, checkpoint_snapshot=snapshot
                )
            self.assertFalse((root / "rejected").exists())

    def test_yue_native_rejects_package_overwrite_during_load(self):
        from modules_forge.yue2_studio import worker
        from modules_forge.yue2_studio.core import Request

        with TemporaryDirectory() as folder:
            root = Path(folder)
            models = root / "models"
            models.mkdir()
            weights = models / "weights"
            weights.write_bytes(b"first")
            snapshot = studio_checkpoints.snapshot_checkpoint(str(models), directory=True)
            pipe = Mock()

            def load(*args, **kwargs):
                weights.write_bytes(b"other")
                return pipe

            fake = {
                "torch": SimpleNamespace(
                    cuda=SimpleNamespace(
                        is_available=lambda: True,
                        is_bf16_supported=lambda: True,
                        get_device_properties=lambda _: SimpleNamespace(total_memory=24 * 2**30),
                    )
                ),
                "yue2": SimpleNamespace(YuE2Pipeline=SimpleNamespace(from_pretrained=load)),
                "yue2.pipeline": SimpleNamespace(SongResult=Mock()),
                "yue2.protocol": SimpleNamespace(
                    GenerationConfig=SimpleNamespace(from_dict=lambda value: value), SongRequest=Mock()
                ),
                "yue2.storage": SimpleNamespace(identity=Mock()),
            }
            with patch.dict("sys.modules", fake):
                with self.assertRaisesRegex(ValueError, "変更"):
                    worker.native(
                        Request(style="piano", lyrics="lyrics", seed=42, checkpoint=str(models)),
                        {"model": str(models), "vae": "shared", "checkpoint_identity": snapshot["identity"]},
                        root,
                        True,
                        Mock(),
                        lambda: False,
                        checkpoint_snapshot=snapshot,
                    )
            pipe.close.assert_called_once()
            pipe.plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
