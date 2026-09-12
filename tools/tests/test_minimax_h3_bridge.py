from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest import mock

import modules_forge.minimax_h3_bridge as h3_bridge
from tools.tests.media_fixtures import write_video

from modules_forge.minimax_h3_bridge import (
    ComfyH3Client,
    H3BridgeError,
    H3JobNotFound,
    H3Request,
    HistoryItem,
    H3_AUDIO_VAE,
    H3_FL_MODEL,
    H3_MINIMUM_COMFY_COMMIT,
    H3_REF_MODEL,
    H3_TEXT_ENCODER,
    H3_VIDEO_VAE,
    MODE_KEYFRAMES,
    MODE_REFERENCES,
    MODE_TEXT,
    RUNTIME_PROFILE_FAST,
    RUNTIME_PROFILE_LOW_RAM,
    RuntimeReadiness,
    _active_generation_count,
    _clear_active_generation,
    _clear_cancelled_job,
    _cleanup_after_terminal,
    _copy_to_comfy_input,
    _generation_poll_interval,
    _is_cancelled_job,
    _loopback_server_process,
    _mark_active_generation,
    _mark_cancelled_job,
    _probe_media,
    _probe_media_cached,
    _resolve_local_path,
    _runtime_command,
    _validate_request_runtime_constraints,
    append_prompt_section,
    build_workflow,
    cache_history_video,
    cleanup_prepared_media,
    cleanup_stale_prepared_media,
    dimensions_for,
    ensure_ready,
    extract_history_video,
    generation_preset_values,
    h3_core_optimization_status,
    history_html,
    history_choices,
    inspect_readiness,
    list_history,
    load_history_request,
    mirror_result,
    normalize_file_list,
    normalize_loopback_url,
    prepare_media,
    progress_html,
    prompt_template,
    reference_guide_html,
    readiness_html,
    relative_workload,
    resolve_runtime_root,
    restart_runtime,
    run_generation,
    runtime_profile_from_args,
    server_model_file_status,
    settings_summary_html,
    snap_h3_frames,
    validate_request,
)


class MiniMaxH3GeometryTests(unittest.TestCase):
    def test_duration_snaps_to_17k_plus_5_grid(self):
        self.assertEqual(snap_h3_frames(5), 124)
        self.assertEqual(snap_h3_frames(15), 362)
        for seconds in (5, 5.5, 8, 12.5, 15):
            self.assertEqual(snap_h3_frames(seconds) % 17, 5)

    def test_duration_rejects_untrained_ui_range(self):
        for seconds in (4.5, 15.5, float("inf"), float("-inf"), float("nan")):
            with self.subTest(seconds=seconds), self.assertRaises(H3BridgeError):
                snap_h3_frames(seconds)

    def test_non_finite_steps_and_seed_are_validation_errors(self):
        for field in ("steps", "seed"):
            with self.subTest(field=field), self.assertRaises(H3BridgeError):
                validate_request(
                    H3Request(
                        mode=MODE_TEXT,
                        prompt="A scene with synchronized audio.",
                        **{field: float("inf")},
                    )
                )

    def test_every_quality_dimension_is_multiple_of_32(self):
        for quality in ("draft", "preview", "balanced", "native"):
            for aspect in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
                with self.subTest(quality=quality, aspect=aspect):
                    width, height = dimensions_for(aspect, quality)
                    self.assertEqual(width % 32, 0)
                    self.assertEqual(height % 32, 0)

    def test_default_summary_shows_exact_effective_output(self):
        rendered = settings_summary_html("16:9", "preview", 5, 20)
        self.assertIn("864 × 480", rendered)
        self.assertIn("124 frames", rendered)
        self.assertIn("5.17 sec", rendered)
        self.assertIn("24fps stereo", rendered)
        self.assertIn("相対負荷 1.00×", rendered)

    def test_generation_presets_are_ordered_by_workload(self):
        quick = generation_preset_values("quick", "16:9")
        recommended = generation_preset_values("recommended", "16:9")
        final = generation_preset_values("final", "16:9")
        self.assertEqual(quick[:3], ("draft", 5.0, 20))
        self.assertEqual(recommended[:3], ("preview", 5.0, 20))
        self.assertEqual(final[:3], ("native", 5.0, 20))
        self.assertEqual(quick[3:5], ("simple", "match"))
        self.assertEqual(recommended[3:5], ("simple", "match"))
        self.assertEqual(final[3:5], ("simple", "match"))
        self.assertLess(
            relative_workload("16:9", quick[0], quick[1], quick[2]),
            relative_workload("16:9", recommended[0], recommended[1], recommended[2]),
        )
        self.assertLess(
            relative_workload("16:9", recommended[0], recommended[1], recommended[2]),
            relative_workload("16:9", final[0], final[1], final[2]),
        )

    def test_long_high_step_preview_warns_from_total_workload(self):
        rendered = settings_summary_html("16:9", "preview", 15, 40)
        self.assertIn('data-tone="warn"', rendered)
        self.assertIn("相対負荷 5.84×", rendered)

    def test_native_square_is_never_described_as_fast_preview(self):
        rendered = settings_summary_html("1:1", "native", 5, 20)
        self.assertIn('data-tone="warn"', rendered)
        self.assertIn("Native最終出力", rendered)
        self.assertNotIn("Fast Preview相当", rendered)

    def test_experimental_advanced_values_override_recommended_copy(self):
        rendered = settings_summary_html("16:9", "preview", 5, 20, "beta", "max")
        self.assertIn('data-tone="warn"', rendered)
        self.assertIn("Reference Max", rendered)
        self.assertNotIn("Fast Preview相当", rendered)

    def test_progress_exposes_accessible_numeric_value(self):
        rendered = progress_html("queued", "待機中", 0.42)
        self.assertIn('role="progressbar"', rendered)
        self.assertIn('aria-valuenow="42"', rendered)
        self.assertNotIn('aria-live=', rendered)

    def test_running_progress_is_indeterminate_and_errors_are_alerts(self):
        running = progress_html("running", "生成中", 0.42)
        self.assertIn('aria-valuetext="生成中"', running)
        self.assertNotIn("aria-valuenow", running)
        self.assertNotIn('aria-live=', running)
        error = progress_html("error", "入力を確認", 0.0)
        self.assertNotIn('role="alert"', error)
        self.assertNotIn('aria-live=', error)
        self.assertIn("エラー", error)
        validation = progress_html("validation", "入力欄を確認", 0.0)
        self.assertIn("入力修正待ち", validation)

    def test_progress_normalizes_non_finite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                rendered = progress_html("queued", "待機中", value)
                self.assertIn('aria-valuenow="0"', rendered)
                self.assertIn('style="width:0%"', rendered)
                self.assertNotIn("nan", rendered.lower())
                self.assertNotIn("inf", rendered.lower())


class MiniMaxH3ValidationTests(unittest.TestCase):
    def test_text_request_is_valid(self):
        validate_request(H3Request(mode=MODE_TEXT, prompt="A quiet room with natural stereo ambience."))

    def test_keyframe_request_requires_an_image(self):
        with self.assertRaisesRegex(H3BridgeError, "開始画像または終了画像"):
            validate_request(H3Request(mode=MODE_KEYFRAMES, prompt="A slow transition."))

    def test_reference_request_rejects_audio_only(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Audio 1>.",
            reference_audios=("voice.wav",),
        )
        with self.assertRaisesRegex(H3BridgeError, "音声だけ"):
            validate_request(request)

    def test_reference_request_enforces_each_limit_and_total(self):
        cases = (
            H3Request(mode=MODE_REFERENCES, prompt="x", reference_images=tuple(f"{i}.png" for i in range(10))),
            H3Request(
                mode=MODE_REFERENCES,
                prompt="x",
                reference_images=tuple(f"{i}.png" for i in range(9)),
                reference_videos=tuple(f"{i}.mp4" for i in range(3)),
                reference_audios=("voice.wav",),
            ),
        )
        for request in cases:
            with self.subTest(count=len(request.reference_images)), self.assertRaises(H3BridgeError):
                validate_request(request)

    def test_backend_url_is_strictly_loopback(self):
        self.assertEqual(normalize_loopback_url("http://localhost:8188/"), "http://localhost:8188")
        self.assertEqual(normalize_loopback_url("http://127.0.0.1:8188"), "http://127.0.0.1:8188")
        for url in (
            "https://127.0.0.1:8188",
            "http://example.com:8188",
            "http://127.0.0.1:8188/api",
            "http://[::1]:8188",
        ):
            with self.subTest(url=url), self.assertRaises(H3BridgeError):
                normalize_loopback_url(url)

    def test_runtime_rejects_unc_path_before_filesystem_access(self):
        with self.assertRaisesRegex(H3BridgeError, "UNC"):
            resolve_runtime_root(r"\\server\share\ComfyUI")

    @unittest.skipUnless(os.name == "nt", "Windows drive classification")
    @mock.patch("modules_forge.minimax_h3_bridge.ctypes.windll.kernel32.GetDriveTypeW", return_value=4)
    def test_runtime_rejects_mapped_network_drive(self, _drive_type):
        with self.assertRaisesRegex(H3BridgeError, "ネットワークドライブ"):
            resolve_runtime_root(r"Z:\ComfyUI")

    def test_file_normalization_accepts_gradio_style_values(self):
        class Named:
            name = "b.png"

        self.assertEqual(normalize_file_list([{"path": "a.png"}, Named(), None]), ("a.png", "b.png"))

    def test_media_probe_cache_reuses_unchanged_file_and_invalidates_on_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            media = Path(temporary) / "reference.mp4"
            media.write_bytes(b"first")
            opened = mock.MagicMock()
            container = opened.return_value.__enter__.return_value
            container.duration = 2_000_000
            container.streams.audio = [object()]
            container.streams.video = []
            _probe_media_cached.cache_clear()
            with mock.patch("av.open", opened):
                first = _probe_media(media)
                second = _probe_media(media)
                media.write_bytes(b"changed-size")
                third = _probe_media(media)
            self.assertEqual(first, (2.0, True, None))
            self.assertEqual(second, first)
            self.assertEqual(third, first)
            self.assertEqual(opened.call_count, 2)


class MiniMaxH3WorkflowTests(unittest.TestCase):
    def test_t2v_graph_matches_native_h3_contract(self):
        request = H3Request(mode=MODE_TEXT, prompt="A scene with synchronized rain audio.", seed=42)
        workflow = build_workflow(request, {}, seed=42)

        self.assertEqual(workflow["1"]["inputs"]["unet_name"], H3_FL_MODEL)
        self.assertEqual(workflow["2"]["inputs"], {"clip_name": H3_TEXT_ENCODER, "type": "minimax", "device": "default"})
        self.assertEqual(workflow["3"]["inputs"]["vae_name"], H3_VIDEO_VAE)
        self.assertEqual(workflow["4"]["inputs"]["vae_name"], H3_AUDIO_VAE)
        self.assertEqual(workflow["5"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(
            workflow["15"]["inputs"],
            {"model": ["1", 0], "attention": "comfy kitchen attention"},
        )
        self.assertEqual(workflow["9"]["inputs"]["model"], ["15", 0])
        self.assertEqual(workflow["8"]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow["5"]["inputs"]["length"], 124)
        self.assertEqual(workflow["10"]["inputs"]["latent_image"], ["5", 1])
        self.assertEqual(workflow["11"]["inputs"]["samples"], ["10", 0])
        self.assertEqual(workflow["12"]["class_type"], "VAEDecodeAudio")
        self.assertEqual(workflow["13"]["inputs"]["audio"], ["12", 0])
        self.assertEqual(workflow["14"]["inputs"]["codec"], "auto")

    def test_keyframe_graph_omits_missing_optional_input(self):
        request = H3Request(mode=MODE_KEYFRAMES, prompt="Move from day to night.", first_frame="first.png")
        workflow = build_workflow(request, {"first_frame": "forge_h3/first.png", "last_frame": None}, seed=7)
        inputs = workflow["5"]["inputs"]
        self.assertEqual(inputs["first_frame"], ["20", 0])
        self.assertNotIn("last_frame", inputs)

    def test_reference_graph_uses_ref_model_dynamic_paths_and_video_audio(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Picture 1>, <Audio 1>, <Video 1>, and <Audio 2>.",
            reference_images=("ref.png",),
            reference_videos=("ref.mp4",),
            reference_audios=("voice.wav",),
        )
        media = {
            "images": ["forge_h3/ref.png"],
            "videos": [{"name": "forge_h3/ref.mp4", "has_audio": True, "duration": 4.0}],
            "audios": ["forge_h3/voice.wav"],
        }
        workflow = build_workflow(request, media, seed=9)
        conditioning = workflow["5"]
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], H3_REF_MODEL)
        self.assertEqual(conditioning["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertIn("ref_images.ref_image_0", conditioning["inputs"])
        self.assertIn("ref_videos.ref_video_0", conditioning["inputs"])
        self.assertIn("ref_video_audios.ref_video_audio_0", conditioning["inputs"])
        self.assertIn("ref_audios.ref_audio_0", conditioning["inputs"])
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "simple")

    def test_reference_graph_does_not_connect_absent_video_audio(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Video 1>.",
            reference_videos=("ref.mp4",),
        )
        media = {"images": [], "videos": [{"name": "forge_h3/ref.mp4", "has_audio": False}], "audios": []}
        workflow = build_workflow(request, media, seed=9)
        inputs = workflow["5"]["inputs"]
        self.assertIn("ref_videos.ref_video_0", inputs)
        self.assertNotIn("ref_video_audios.ref_video_audio_0", inputs)

    def test_reference_graph_rejects_missing_material_tag(self):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Picture 1>.",
            reference_images=("ref.png",),
            reference_videos=("ref.mp4",),
        )
        media = {
            "images": ["forge_h3/ref.png"],
            "videos": [{"name": "forge_h3/ref.mp4", "has_audio": True}],
            "audios": [],
        }
        with self.assertRaisesRegex(H3BridgeError, "<Audio 1>.*<Video 1>"):
            build_workflow(request, media, seed=9)

    def test_reference_graph_rejects_unknown_or_malformed_tag(self):
        media = {"images": ["forge_h3/ref.png"], "videos": [], "audios": []}
        cases = (
            ("Use <Picture 1> and <Video 1>.", "存在しない参照タグ"),
            ("Use <picture 1>.", "形式が不正"),
        )
        for prompt, message in cases:
            request = H3Request(
                mode=MODE_REFERENCES,
                prompt=prompt,
                reference_images=("ref.png",),
            )
            with self.subTest(prompt=prompt), self.assertRaisesRegex(H3BridgeError, message):
                build_workflow(request, media, seed=9)

    @mock.patch("modules_forge.minimax_h3_bridge._copy_to_comfy_input", return_value="forge_h3/ref.mp4")
    @mock.patch("modules_forge.minimax_h3_bridge._validate_media_path", return_value=Path("ref.mp4"))
    @mock.patch("modules_forge.minimax_h3_bridge._probe_media", return_value=(4.0, True, 30.0))
    def test_reference_media_rejects_video_that_is_not_24fps(self, _probe, _validate, _copy):
        request = H3Request(
            mode=MODE_REFERENCES,
            prompt="Use <Audio 1> and <Video 1>.",
            reference_videos=("ref.mp4",),
        )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(H3BridgeError, "24fps"):
            prepare_media(request, Path(temporary))


class MiniMaxH3RuntimeTests(unittest.TestCase):
    def setUp(self):
        # 遅延後始末をmockするテストも、実ジョブと同じ所有権を保持する。
        from modules.fifo_lock import FIFOLock
        from modules_forge import gpu_ownership

        self.enterContext(mock.patch.object(gpu_ownership, "queue_lock", FIFOLock()))
        self.enterContext(mock.patch.object(h3_bridge, "_GPU_OWNERSHIPS", {}))
        self.enterContext(mock.patch.object(h3_bridge, "_ACTIVE_GENERATION_IDS", set()))
        self.enterContext(mock.patch.object(h3_bridge, "release_forge_vram"))
        self.enterContext(mock.patch.object(h3_bridge, "_CANCELLED_JOB_IDS", set()))
        self.enterContext(mock.patch.object(h3_bridge, "_CANCEL_ACK_IDS", set()))
        self.enterContext(mock.patch.object(h3_bridge.uuid, "uuid4", return_value=uuid.UUID("11111111-1111-4111-8111-111111111111")))
        self.enterContext(mock.patch.object(h3_bridge.pending_jobs, "write"))
        self.enterContext(mock.patch.object(h3_bridge.pending_jobs, "update"))
        self.enterContext(mock.patch.object(h3_bridge.pending_jobs, "remove"))
        self.enterContext(mock.patch.object(h3_bridge, "_loopback_server_process", return_value=SimpleNamespace(pid=os.getpid(), create_time=lambda: 1.0, parents=lambda: [])))

    @staticmethod
    def ready_runtime(runtime_profile: str = RUNTIME_PROFILE_FAST) -> RuntimeReadiness:
        files = {
            name: True
            for name in ("FL2VA", "Ref2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")
        }
        return RuntimeReadiness(
            runtime_root=Path("runtime").resolve(),
            server_url="http://127.0.0.1:8188",
            connected=True,
            comfy_version="0.31.0",
            package_versions={"comfy-kitchen": "0.2.30"},
            ck_attention_available=True,
            core_revision=H3_MINIMUM_COMFY_COMMIT,
            h3_core_optimized=True,
            runtime_profile=runtime_profile,
            model_files=files,
            server_model_files=files,
        )

    def test_listener_rejects_mixed_loopback_and_public_bindings(self):
        class FakePsutilError(Exception):
            pass

        listeners = [
            SimpleNamespace(
                status="LISTEN",
                laddr=SimpleNamespace(ip="127.0.0.1", port=8188),
                pid=123,
            ),
            SimpleNamespace(
                status="LISTEN",
                laddr=SimpleNamespace(ip="0.0.0.0", port=8188),
                pid=123,
            ),
        ]
        fake_psutil = SimpleNamespace(
            CONN_LISTEN="LISTEN",
            Error=FakePsutilError,
            net_connections=lambda kind: listeners,
            Process=mock.Mock(),
        )

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with self.assertRaisesRegex(H3BridgeError, "loopback専用"):
                _loopback_server_process("http://127.0.0.1:8188")
        fake_psutil.Process.assert_not_called()

    def test_listener_rejects_multiple_loopback_processes(self):
        class FakePsutilError(Exception):
            pass

        listeners = [
            SimpleNamespace(
                status="LISTEN",
                laddr=SimpleNamespace(ip="127.0.0.1", port=8188),
                pid=123,
            ),
            SimpleNamespace(
                status="LISTEN",
                laddr=SimpleNamespace(ip="::1", port=8188),
                pid=456,
            ),
        ]
        fake_psutil = SimpleNamespace(
            CONN_LISTEN="LISTEN",
            Error=FakePsutilError,
            net_connections=lambda kind: listeners,
            Process=mock.Mock(),
        )

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with self.assertRaisesRegex(H3BridgeError, "複数のprocess"):
                _loopback_server_process("http://127.0.0.1:8188")
        fake_psutil.Process.assert_not_called()

    def test_listener_wraps_psutil_access_error(self):
        class FakePsutilError(Exception):
            pass

        def denied(kind):
            raise FakePsutilError("access denied")

        fake_psutil = SimpleNamespace(
            CONN_LISTEN="LISTEN",
            Error=FakePsutilError,
            net_connections=denied,
            Process=mock.Mock(),
        )

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with self.assertRaisesRegex(H3BridgeError, "access denied"):
                _loopback_server_process("http://127.0.0.1:8188")

    def test_runtime_command_uses_3090_optimized_offload_without_usb_disk_flags(self):
        command = _runtime_command(
            Path(r"C:\Python\python.exe"),
            8188,
            RUNTIME_PROFILE_FAST,
        )
        self.assertEqual(
            command,
            [
                r"C:\Python\python.exe",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--disable-all-custom-nodes",
                "--disable-api-nodes",
                "--reserve-vram",
                "2",
                "--preview-method",
                "none",
                "--async-offload",
                "2",
            ],
        )
        for harmful_flag in (
            "--vram-headroom",
            "--fast-disk",
            "--disable-pinned-memory",
            "--disable-async-offload",
        ):
            self.assertNotIn(harmful_flag, command)

    def test_low_ram_runtime_command_disables_cache_pinning_and_async(self):
        command = _runtime_command(
            Path(r"C:\Python\python.exe"),
            8188,
            RUNTIME_PROFILE_LOW_RAM,
        )
        self.assertIn("--cache-none", command)
        self.assertIn("--disable-pinned-memory", command)
        self.assertIn("--disable-async-offload", command)
        self.assertNotIn("--async-offload", command)
        self.assertNotIn("--fast-disk", command)

    def test_runtime_profile_parser_checks_values_conflicts_and_duplicates(self):
        fast = _runtime_command(Path("python.exe"), 8188, RUNTIME_PROFILE_FAST)[1:]
        low_ram = _runtime_command(Path("python.exe"), 8188, RUNTIME_PROFILE_LOW_RAM)[1:]
        self.assertEqual(runtime_profile_from_args(fast, 8188), RUNTIME_PROFILE_FAST)
        self.assertEqual(runtime_profile_from_args(low_ram, 8188), RUNTIME_PROFILE_LOW_RAM)
        self.assertEqual(
            runtime_profile_from_args([*fast, "--auto-launch"], 8188),
            RUNTIME_PROFILE_FAST,
        )
        for invalid in (
            [*fast[:-1], "1"],
            [*fast, "--disable-async-offload"],
            [*fast, "--async-offload", "2"],
            [*fast, "--fast-disk"],
            [*fast, "--vram-headroom", "12"],
            [*fast, "--port", "8189"],
            [*fast, "--cache-ram", "1", "2"],
            [*fast, "--cpu-vae"],
            [*fast, "--force-fp32"],
            [*fast, "--use-sage-attention"],
            [*fast, "--disable-xformers"],
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(runtime_profile_from_args(invalid, 8188))

    def test_unknown_runtime_profile_fails_instead_of_falling_back(self):
        with self.assertRaisesRegex(H3BridgeError, "未対応"):
            _runtime_command(Path("python.exe"), 8188, "unknown")

    def test_h3_core_optimization_requires_minimum_commit_ancestor(self):
        revision = "a" * 40
        completed = [
            mock.Mock(returncode=0, stdout=revision + "\n"),
            mock.Mock(returncode=0, stdout=""),
        ]
        with mock.patch("modules_forge.minimax_h3_bridge.subprocess.run", side_effect=completed) as run:
            optimized, detected = h3_core_optimization_status(Path("runtime"))
        self.assertTrue(optimized)
        self.assertEqual(detected, revision)
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["git", "merge-base", "--is-ancestor", H3_MINIMUM_COMFY_COMMIT, revision],
        )

    @mock.patch("modules_forge.minimax_h3_bridge.inspect_readiness")
    def test_ensure_ready_rejects_connected_profile_mismatch(self, inspect):
        inspect.return_value = self.ready_runtime(RUNTIME_PROFILE_LOW_RAM)
        with self.assertRaisesRegex(H3BridgeError, "起動設定が一致"):
            ensure_ready(
                Path("runtime"),
                "http://127.0.0.1:8188",
                Path("logs"),
                runtime_profile=RUNTIME_PROFILE_FAST,
            )

    @mock.patch("modules_forge.minimax_h3_bridge.start_runtime")
    @mock.patch("modules_forge.minimax_h3_bridge.inspect_readiness")
    def test_ensure_ready_reuses_disconnected_snapshot_for_start(self, inspect, start):
        disconnected = RuntimeReadiness(
            runtime_root=Path("runtime"),
            server_url="http://127.0.0.1:8188",
            connected=False,
            error="not running",
        )
        inspect.return_value = disconnected
        start.return_value = self.ready_runtime()

        actual = ensure_ready(
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            runtime_profile=RUNTIME_PROFILE_FAST,
        )

        self.assertTrue(actual.connected)
        inspect.assert_called_once()
        self.assertIs(start.call_args.kwargs["initial_readiness"], disconnected)

    @mock.patch("modules_forge.minimax_h3_bridge._loopback_server_process")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.server_runtime_root")
    def test_restart_refuses_to_terminate_external_runtime(self, server_root, client_class, listener):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary).resolve()
            (runtime / "main.py").write_text("# test\n", encoding="utf-8")
            (runtime / "models").mkdir()
            server_root.return_value = runtime
            client_class.return_value.queue_counts.return_value = (0, 0)
            process = mock.Mock(pid=1234)
            listener.return_value = process
            with mock.patch("modules_forge.minimax_h3_bridge._MANAGED_PROCESS", None), mock.patch(
                "modules_forge.minimax_h3_bridge._MANAGED_PROCESS_IDENTITY",
                None,
            ):
                with self.assertRaisesRegex(H3BridgeError, "自動停止しません"):
                    restart_runtime(
                        runtime,
                        "http://127.0.0.1:8188",
                        Path("logs"),
                        runtime_profile=RUNTIME_PROFILE_FAST,
                    )
        process.terminate.assert_not_called()

    @mock.patch("modules_forge.minimax_h3_bridge.server_runtime_root")
    def test_restart_is_blocked_until_active_generation_finishes_result_processing(self, server_root):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary).resolve()
            (runtime / "main.py").write_text("# test\n", encoding="utf-8")
            (runtime / "models").mkdir()
            _mark_active_generation("result-processing-job")
            try:
                with self.assertRaisesRegex(H3BridgeError, "生成結果を処理中"):
                    restart_runtime(
                        runtime,
                        "http://127.0.0.1:8188",
                        Path("logs"),
                    )
            finally:
                _clear_active_generation("result-processing-job")
        server_root.assert_not_called()
        self.assertEqual(_active_generation_count(), 0)

    @mock.patch("modules_forge.minimax_h3_bridge.start_runtime")
    @mock.patch("modules_forge.minimax_h3_bridge._loopback_server_process")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.server_runtime_root")
    def test_restart_managed_runtime_requires_idle_queue_and_restarts_selected_profile(
        self,
        server_root,
        client_class,
        listener,
        start,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary).resolve()
            (runtime / "main.py").write_text("# test\n", encoding="utf-8")
            (runtime / "models").mkdir()
            url = "http://127.0.0.1:8188"
            server_root.return_value = runtime
            client_class.return_value.queue_counts.return_value = (0, 0)
            process = mock.Mock(pid=1234)
            listener.side_effect = [process, None]
            managed = mock.Mock(pid=1234)
            managed.poll.return_value = None
            expected = self.ready_runtime(RUNTIME_PROFILE_LOW_RAM)
            start.return_value = expected
            with mock.patch("modules_forge.minimax_h3_bridge._MANAGED_PROCESS", managed), mock.patch(
                "modules_forge.minimax_h3_bridge._MANAGED_PROCESS_IDENTITY",
                (runtime, url),
            ):
                actual = restart_runtime(
                    runtime,
                    url,
                    Path("logs"),
                    runtime_profile=RUNTIME_PROFILE_LOW_RAM,
                    wait_seconds=1,
                )
        self.assertIs(actual, expected)
        process.terminate.assert_called_once_with()
        start.assert_called_once_with(
            runtime,
            url,
            Path("logs"),
            runtime_profile=RUNTIME_PROFILE_LOW_RAM,
            wait_seconds=1,
            acceleration=None,
        )

    def test_readiness_html_never_labels_async_one_as_fast_profile(self):
        arguments = _runtime_command(Path("python.exe"), 8188, RUNTIME_PROFILE_FAST)[1:]
        arguments[-1] = "1"
        readiness = RuntimeReadiness(
            runtime_root=Path("runtime"),
            server_url="http://127.0.0.1:8188",
            connected=True,
            comfy_version="0.31.0",
            package_versions={"comfy-kitchen": "0.2.30"},
            runtime_args=tuple(arguments),
            runtime_profile=runtime_profile_from_args(arguments, 8188),
            ck_attention_available=True,
            core_revision=H3_MINIMUM_COMFY_COMMIT,
            h3_core_optimized=True,
            model_files={name: True for name in ("FL2VA", "Ref2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")},
            server_model_files={name: True for name in ("FL2VA", "Ref2VA", "Qwen3-VL 32B", "Video VAE", "Audio VAE")},
        )
        rendered = readiness_html(readiness, RUNTIME_PROFILE_FAST)
        self.assertIn("起動設定 未確認", rendered)
        self.assertNotIn("高速 · Async 2", rendered)

    def test_low_ram_readiness_says_async_is_disabled(self):
        rendered = readiness_html(
            self.ready_runtime(RUNTIME_PROFILE_LOW_RAM),
            RUNTIME_PROFILE_LOW_RAM,
        )
        self.assertIn("省RAM · Async無効", rendered)
        self.assertNotIn("省RAM · Async 2", rendered)

    def test_native_fifteen_second_request_is_blocked_when_ram_headroom_is_too_low(self):
        readiness = replace(self.ready_runtime(), ram_free_gib=7.0)
        request = H3Request(
            mode=MODE_TEXT,
            prompt="A scene with stereo ambience.",
            quality="native",
            duration_seconds=15,
        )
        with self.assertRaisesRegex(H3BridgeError, "空きRAMが不足"):
            _validate_request_runtime_constraints(request, readiness, RUNTIME_PROFILE_FAST)

    def test_draft_request_passes_ram_preflight_at_same_free_memory(self):
        readiness = replace(self.ready_runtime(), ram_free_gib=7.0)
        request = H3Request(
            mode=MODE_TEXT,
            prompt="A scene with stereo ambience.",
            quality="draft",
            duration_seconds=5,
        )
        _validate_request_runtime_constraints(request, readiness, RUNTIME_PROFILE_FAST)

    def test_commit_headroom_blocks_request_even_when_physical_ram_is_available(self):
        readiness = replace(
            self.ready_runtime(),
            ram_free_gib=12.0,
            commit_free_gib=4.4,
        )
        request = H3Request(
            mode=MODE_TEXT,
            prompt="A scene with stereo ambience.",
            quality="preview",
            duration_seconds=5,
        )
        with self.assertRaisesRegex(H3BridgeError, "OS commit余力 4.4 GiB"):
            _validate_request_runtime_constraints(request, readiness, RUNTIME_PROFILE_FAST)

    def test_readiness_html_warns_when_default_request_exceeds_commit_headroom(self):
        readiness = replace(
            self.ready_runtime(),
            ram_free_gib=12.0,
            ram_total_gib=64.0,
            commit_free_gib=4.4,
        )
        rendered = readiness_html(readiness, RUNTIME_PROFILE_FAST)
        self.assertIn("RAM余力 4.4 GiB", rendered)
        self.assertIn("標準5秒", rendered)
        self.assertIn("commit余力 4.4 GiB", rendered)
        self.assertGreaterEqual(rendered.count('data-mobile="primary"'), 3)
        self.assertIn("<dt>Runtime</dt>", rendered)

    def test_non_finite_ram_telemetry_is_rejected(self):
        readiness = replace(self.ready_runtime(), ram_free_gib=float("nan"))
        request = H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience.")
        with self.assertRaisesRegex(H3BridgeError, "空き物理RAMの情報が不正"):
            _validate_request_runtime_constraints(request, readiness, RUNTIME_PROFILE_FAST)

    def test_object_info_fetches_only_requested_nodes_with_one_deadline(self):
        client = ComfyH3Client()
        self.addCleanup(client.close)
        with mock.patch.object(
            client,
            "_request_json",
            side_effect=lambda path, timeout: {path.rsplit("/", 1)[-1]: {"input": {}}},
        ) as request:
            result = client.object_info(["VAELoader", "UNETLoader"], timeout=8.0)
        self.assertEqual(set(result), {"UNETLoader", "VAELoader"})
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            ["/object_info/UNETLoader", "/object_info/VAELoader"],
        )
        self.assertTrue(all(call.kwargs["timeout"] <= 8.0 for call in request.call_args_list))

    def test_object_info_total_deadline_fails_before_another_request(self):
        client = ComfyH3Client()
        self.addCleanup(client.close)
        with mock.patch(
            "modules_forge.minimax_h3_bridge.time.monotonic",
            side_effect=[100.0, 109.0],
        ), mock.patch.object(client, "_request_json") as request:
            with self.assertRaisesRegex(H3BridgeError, "時間切れ"):
                client.object_info(["UNETLoader"], timeout=8.0)
        request.assert_not_called()

    def test_client_uses_loopback_keepalive_without_environment_proxy(self):
        with mock.patch("modules_forge.minimax_h3_bridge.httpx.Client") as client_class:
            client = ComfyH3Client("http://127.0.0.1:8188")
            client.close()

        kwargs = client_class.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8188")
        self.assertFalse(kwargs["trust_env"])
        self.assertEqual(kwargs["limits"].max_connections, 1)
        self.assertEqual(kwargs["limits"].max_keepalive_connections, 1)
        client_class.return_value.close.assert_called_once_with()

    def test_client_preserves_job_not_found_contract(self):
        with mock.patch("modules_forge.minimax_h3_bridge.httpx.Client") as client_class:
            client_class.return_value.request.return_value = mock.Mock(
                status_code=404,
                text='{"error": {"message": "missing"}}',
                content=b'{"error": {"message": "missing"}}',
            )
            client = ComfyH3Client()
            with self.assertRaisesRegex(H3JobNotFound, "HTTP 404.*missing"):
                client.job("missing-job")
            client.close()

    def test_adaptive_generation_poll_interval_is_responsive_then_backs_off(self):
        self.assertEqual(_generation_poll_interval(False, 0.0), 1.0)
        self.assertEqual(_generation_poll_interval(True, 0.0), 2.0)
        self.assertEqual(_generation_poll_interval(True, 59.99), 2.0)
        self.assertEqual(_generation_poll_interval(True, 60.0), 5.0)

    def test_server_model_status_uses_exact_loader_choices(self):
        nodes = {
            "UNETLoader": {"input": {"required": {"unet_name": [[H3_FL_MODEL, H3_REF_MODEL]]}}},
            "CLIPLoader": {"input": {"required": {"clip_name": [[H3_TEXT_ENCODER]]}}},
            "VAELoader": {
                "input": {"required": {"vae_name": [[H3_VIDEO_VAE, H3_AUDIO_VAE]]}}
            },
        }
        self.assertTrue(all(server_model_file_status(nodes).values()))
        nodes["CLIPLoader"]["input"]["required"]["clip_name"] = [["other.safetensors"]]
        self.assertFalse(server_model_file_status(nodes)["Qwen3-VL 32B"])

    @mock.patch("modules_forge.minimax_h3_bridge.server_runtime_root")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    def test_readiness_rejects_different_runtime_on_same_port(self, client_class, connected_root):
        with tempfile.TemporaryDirectory() as temporary:
            selected = Path(temporary) / "selected"
            other = Path(temporary) / "other"
            for root in (selected, other):
                (root / "models").mkdir(parents=True)
                (root / "main.py").write_text("# test\n", encoding="utf-8")
            client_class.return_value.system_stats.return_value = {"system": {}, "devices": []}
            connected_root.return_value = other.resolve()
            readiness = inspect_readiness(selected, "http://127.0.0.1:8188")
            self.assertFalse(readiness.connected)
            self.assertIn("別のComfyUI", readiness.error or "")

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_generation_rechecks_profile_and_submits_under_runtime_lifecycle_lock(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        _schedule_cleanup,
        _cleanup,
    ):
        class RecordingLock:
            depth = 0

            def __enter__(self):
                self.depth += 1
                return self

            def __exit__(self, _type, _value, _traceback):
                self.depth -= 1

        lifecycle_lock = RecordingLock()
        ready.return_value = self.ready_runtime()
        client = client_class.return_value

        def submit(_workflow_value, _prompt_id):
            self.assertGreater(lifecycle_lock.depth, 0)
            return "11111111-1111-4111-8111-111111111111"

        client.submit.side_effect = submit
        with mock.patch(
            "modules_forge.minimax_h3_bridge._RUNTIME_LIFECYCLE_LOCK",
            lifecycle_lock,
        ):
            updates = run_generation(
                H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
                Path("runtime"),
                "http://127.0.0.1:8188",
                Path("logs"),
                Path("output"),
            )
            self.assertEqual(next(updates)["stage"], "runtime")
            self.assertEqual(next(updates)["stage"], "prepare")
            self.assertEqual(next(updates)["stage"], "queued")
            self.assertEqual(lifecycle_lock.depth, 0)
            updates.close()
        self.assertEqual(ready.call_count, 2)

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_stale_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_generation_cleans_prepared_media_without_submit_when_second_ram_check_fails(
        self,
        ready,
        prepare,
        _workflow,
        client_class,
        _stale_cleanup,
        schedule_cleanup,
        cleanup,
    ):
        prepared = {
            "first_frame": None,
            "last_frame": None,
            "images": ["forge_h3/reference.png"],
            "videos": [],
            "audios": [],
        }
        prepare.return_value = prepared
        ready.side_effect = [
            replace(self.ready_runtime(), ram_free_gib=8.0),
            replace(self.ready_runtime(), ram_free_gib=4.0),
        ]
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
        )

        self.assertEqual(next(updates)["stage"], "runtime")
        self.assertEqual(next(updates)["stage"], "prepare")
        with self.assertRaisesRegex(H3BridgeError, "空きRAMが不足"):
            next(updates)

        self.assertEqual(ready.call_count, 2)
        client_class.return_value.submit.assert_not_called()
        schedule_cleanup.assert_not_called()
        cleanup.assert_called_once_with(prepared, Path("runtime"))

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_in_progress_job_is_running_and_terminal_cancel_cleans_inputs(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        cleanup,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.return_value = {"status": "in_progress"}
        request = H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience.")
        updates = run_generation(
            request,
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
            poll_seconds=0.5,
        )
        self.assertEqual(next(updates)["stage"], "runtime")
        self.assertEqual(next(updates)["stage"], "prepare")
        self.assertEqual(next(updates)["stage"], "queued")
        running = next(updates)
        self.assertEqual(running["stage"], "running")
        self.assertIn("生成中", running["message"])
        client.job.return_value = {"status": "cancelled"}
        with self.assertRaisesRegex(H3BridgeError, "停止"):
            next(updates)
        cleanup.assert_called_once_with({"images": []}, Path("runtime"))

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_cancelled_pending_job_404_is_terminal_and_cleans_inputs(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        cleanup,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.side_effect = H3JobNotFound("HTTP 404")
        h3_bridge._CANCEL_ACK_IDS.add("11111111-1111-4111-8111-111111111111")
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
        )
        self.assertEqual(next(updates)["stage"], "runtime")
        self.assertEqual(next(updates)["stage"], "prepare")
        self.assertEqual(next(updates)["stage"], "queued")
        with self.assertRaisesRegex(H3BridgeError, "停止"):
            next(updates)
        cleanup.assert_called_once_with({"images": []}, Path("runtime"))
        self.assertFalse(_is_cancelled_job("11111111-1111-4111-8111-111111111111"))

    def test_cancel_intent_survives_lost_cancel_response(self):
        client = ComfyH3Client.__new__(ComfyH3Client)

        def assert_intent(_path, _payload):
            self.assertTrue(_is_cancelled_job("cancel-race-job"))
            return {}

        client._request_json = mock.Mock(side_effect=assert_intent)
        try:
            client.cancel("cancel-race-job")
            self.assertTrue(_is_cancelled_job("cancel-race-job"))
        finally:
            _clear_cancelled_job("cancel-race-job")

        client._request_json = mock.Mock(side_effect=H3BridgeError("cancel failed"))
        with self.assertRaisesRegex(H3BridgeError, "cancel failed"):
            client.cancel("cancel-failure-job")
        self.assertTrue(_is_cancelled_job("cancel-failure-job"))
        _clear_cancelled_job("cancel-failure-job")

    @mock.patch("modules_forge.minimax_h3_bridge.time.sleep")
    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.mirror_result", return_value=Path("result.mp4"))
    @mock.patch("modules_forge.minimax_h3_bridge.extract_history_video", return_value=Path("source.mp4"))
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_temporary_poll_failure_recovers_without_cancelling_job(
        self,
        ready,
        _prepare,
        _workflow,
        _extract,
        _mirror,
        client_class,
        _cleanup,
        sleep,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.side_effect = [
            {"status": "in_progress"},
            H3BridgeError("temporary timeout"),
            {"status": "in_progress"},
            {"status": "completed"},
        ]
        client.history.return_value = {}

        updates = list(
            run_generation(
                H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
                Path("runtime"),
                "http://127.0.0.1:8188",
                Path("logs"),
                Path("output"),
                poll_seconds=0.5,
            )
        )

        self.assertEqual(
            [update["stage"] for update in updates],
            ["runtime", "prepare", "queued", "running", "reconnecting", "running", "complete"],
        )
        client.cancel.assert_not_called()
        self.assertEqual(_active_generation_count(), 0)
        self.assertGreaterEqual(sleep.call_count, 3)

    @mock.patch("modules_forge.minimax_h3_bridge.time.sleep")
    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_cancel_intent_keeps_inputs_until_backend_confirms_terminal_status(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        schedule_cleanup,
        cleanup,
        _sleep,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.side_effect = [
            H3BridgeError("temporary timeout"),
            {"status": "cancelled"},
        ]
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
            poll_seconds=0.5,
        )
        self.assertEqual(next(updates)["stage"], "runtime")
        self.assertEqual(next(updates)["stage"], "prepare")
        self.assertEqual(next(updates)["stage"], "queued")
        _mark_cancelled_job("11111111-1111-4111-8111-111111111111")
        self.assertEqual(next(updates)["stage"], "reconnecting")
        cleanup.assert_not_called()

        with self.assertRaisesRegex(H3BridgeError, "停止"):
            next(updates)

        cleanup.assert_called_once_with({"images": []}, Path("runtime"))
        schedule_cleanup.assert_not_called()
        self.assertFalse(_is_cancelled_job("11111111-1111-4111-8111-111111111111"))

    @mock.patch("modules_forge.minimax_h3_bridge.time.sleep")
    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.mirror_result", return_value=Path("result.mp4"))
    @mock.patch("modules_forge.minimax_h3_bridge.extract_history_video", return_value=Path("source.mp4"))
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_completed_job_retries_transient_history_failure_before_mirroring(
        self,
        ready,
        _prepare,
        _workflow,
        extract,
        mirror,
        client_class,
        _cleanup,
        sleep,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.return_value = {"status": "completed"}
        client.history.side_effect = [{}, {"ok": True}]
        extract.side_effect = [H3BridgeError("result is not visible yet"), Path("source.mp4")]

        updates = list(
            run_generation(
                H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
                Path("runtime"),
                "http://127.0.0.1:8188",
                Path("logs"),
                Path("output"),
                poll_seconds=0.5,
            )
        )

        self.assertEqual(updates[-1]["stage"], "complete")
        self.assertEqual(client.history.call_count, 2)
        self.assertEqual(extract.call_count, 2)
        mirror.assert_called_once()
        client.cancel.assert_not_called()
        sleep.assert_called_once_with(0.5)

    @mock.patch("modules_forge.minimax_h3_bridge.time.sleep")
    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_repeated_poll_failures_cancel_only_after_retry_budget_is_exhausted(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        schedule_cleanup,
        _cleanup,
        _sleep,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.side_effect = H3BridgeError("temporary timeout")

        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
            poll_seconds=0.5,
        )
        stages = [next(updates)["stage"] for _ in range(5)]
        self.assertEqual(stages, ["runtime", "prepare", "queued", "reconnecting", "reconnecting"])
        with self.assertRaisesRegex(H3BridgeError, "連続して失敗"):
            next(updates)
        self.assertEqual(client.job.call_count, 3)
        client.cancel.assert_called_once_with("11111111-1111-4111-8111-111111111111")
        schedule_cleanup.assert_called_once()
        self.assertEqual(_active_generation_count(), 1)

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_abandoned_job_requests_cancel_without_deleting_live_inputs(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        schedule_cleanup,
        cleanup,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.return_value = {"status": "in_progress"}
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
        )
        next(updates)
        next(updates)
        next(updates)
        next(updates)
        updates.close()
        cleanup.assert_not_called()
        from modules_forge.gpu_ownership import GPUOwnership

        next_owner = GPUOwnership()
        self.assertFalse(next_owner.acquire())
        client.job.return_value = {"status": "failed"}
        _cleanup_after_terminal(client, "11111111-1111-4111-8111-111111111111", {}, Path("runtime"))
        self.assertTrue(next_owner.acquire())
        next_owner.release()
        client.cancel.assert_called_once_with("11111111-1111-4111-8111-111111111111")
        schedule_cleanup.assert_called_once_with(client, "11111111-1111-4111-8111-111111111111", {"images": []}, Path("runtime"))

    @mock.patch("modules_forge.minimax_h3_bridge.cleanup_prepared_media")
    @mock.patch("modules_forge.minimax_h3_bridge._schedule_deferred_cleanup")
    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media", return_value={"images": []})
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_abandoned_job_schedules_deferred_cleanup_when_cancel_request_fails(
        self,
        ready,
        _prepare,
        _workflow,
        client_class,
        schedule_cleanup,
        cleanup,
    ):
        ready.return_value = self.ready_runtime()
        client = client_class.return_value
        client.submit.return_value = "11111111-1111-4111-8111-111111111111"
        client.job.return_value = {"status": "in_progress"}
        client.cancel.side_effect = H3BridgeError("cancel request failed")
        updates = run_generation(
            H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
            Path("runtime"),
            "http://127.0.0.1:8188",
            Path("logs"),
            Path("output"),
        )
        next(updates)
        next(updates)
        next(updates)
        next(updates)
        updates.close()

        client.cancel.assert_called_once_with("11111111-1111-4111-8111-111111111111")
        schedule_cleanup.assert_called_once_with(client, "11111111-1111-4111-8111-111111111111", {"images": []}, Path("runtime"))
        cleanup.assert_not_called()

    @mock.patch("modules_forge.minimax_h3_bridge.ComfyH3Client")
    @mock.patch("modules_forge.minimax_h3_bridge.build_workflow", return_value={"graph": {}})
    @mock.patch("modules_forge.minimax_h3_bridge.prepare_media")
    @mock.patch("modules_forge.minimax_h3_bridge.ensure_ready")
    def test_completed_generation_mirrors_metadata_and_cleans_prepared_media(
        self,
        ready,
        prepare,
        _workflow,
        client_class,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            managed = runtime / "input" / "forge_h3" / "0123456789ab_reference.png"
            managed.parent.mkdir(parents=True)
            managed.write_bytes(b"reference")
            source = runtime / "output" / "video" / "Forge_Neo_H3_result.mp4"
            source.parent.mkdir(parents=True)
            write_video(source, width=864, height=480, frames=124)
            output = root / "forge-output"
            prepared = {
                "first_frame": None,
                "last_frame": None,
                "images": ["forge_h3/0123456789ab_reference.png"],
                "videos": [],
                "audios": [],
            }
            prepare.return_value = prepared
            ready.return_value = replace(
                self.ready_runtime(),
                runtime_root=runtime.resolve(),
                ram_free_gib=8.0,
            )
            client = client_class.return_value
            client.submit.return_value = "11111111-1111-4111-8111-111111111111"
            client.job.return_value = {"status": "completed"}
            client.history.return_value = {
                "11111111-1111-4111-8111-111111111111": {
                    "outputs": {
                        "14": {
                            "videos": [
                                {
                                    "filename": source.name,
                                    "subfolder": "video",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }
            request = H3Request(
                mode=MODE_TEXT,
                prompt="A scene with stereo ambience.",
                seed=314159,
            )

            updates = list(
                run_generation(
                    request,
                    runtime,
                    "http://127.0.0.1:8188",
                    root / "logs",
                    output,
                )
            )

            self.assertEqual(
                [update["stage"] for update in updates],
                ["runtime", "prepare", "queued", "complete"],
            )
            mirrored = Path(updates[-1]["path"])
            self.assertEqual(mirrored.read_bytes(), source.read_bytes())
            metadata_text = mirrored.with_suffix(".json").read_text(encoding="utf-8")
            metadata = json.loads(metadata_text)
            self.assertEqual(metadata["prompt_id"], "11111111-1111-4111-8111-111111111111")
            self.assertEqual(metadata["seed"], 314159)
            self.assertEqual(metadata["runtime_profile"], RUNTIME_PROFILE_FAST)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["acceleration"]["attention"], "dense")
            self.assertEqual(metadata["source_backend"], "comfyui")
            self.assertEqual(metadata["source_file"], source.name)
            self.assertNotIn("source", metadata)
            self.assertNotIn(os.fspath(source.resolve()), metadata_text)
            self.assertFalse(managed.exists())
            client.cancel.assert_not_called()

    def test_history_choices_use_opaque_ids_without_disclosing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "runtime" / "private-video.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            item = HistoryItem(video.resolve(), video.stat().st_mtime, "ComfyUI")

            choices = history_choices([item])

            self.assertEqual(len(choices), 1)
            self.assertTrue(choices[0][1].startswith("h3-"))
            self.assertNotIn(os.fspath(root), choices[0][1])
            cached = Path(cache_history_video(choices[0][1], [item], root / "output"))
            self.assertEqual(cached.read_bytes(), b"video")

    def test_managed_runtime_timeout_falls_back_to_kill(self):
        process = mock.MagicMock()
        process.poll.return_value = None
        def wait(timeout):
            if not process.kill.called:
                raise subprocess.TimeoutExpired("comfy", timeout)
            process.poll.return_value = 0
            return 0

        process.wait.side_effect = wait

        with (
            mock.patch.object(h3_bridge, "_MANAGED_PROCESS", process),
            mock.patch.object(
                h3_bridge,
                "_MANAGED_PROCESS_IDENTITY",
                (Path("runtime"), "http://127.0.0.1:8188"),
            ),
        ):
            h3_bridge._stop_managed_runtime()
            self.assertIsNone(h3_bridge._MANAGED_PROCESS)
            self.assertIsNone(h3_bridge._MANAGED_PROCESS_IDENTITY)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def test_runtime_uses_shared_models_without_external_core(self):
        from modules_forge.minimax_h3_runtime import model_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ComfyUI"
            root.mkdir()
            shared = Path(temporary) / "shared-models"
            model_dir = shared / "diffusion_models"
            model_dir.mkdir(parents=True)
            (root / "main.py").write_text("# test\n", encoding="utf-8")
            (model_dir / H3_FL_MODEL).write_bytes(b"test")
            (root / "extra_model_paths.yaml").write_text(model_config(shared), encoding="utf-8")
            self.assertTrue(h3_bridge.model_file_status(root)["FL2VA"])
            self.assertFalse((root / "models/diffusion_models" / H3_FL_MODEL).exists())

    def test_resolved_unc_path_is_rejected(self):
        with mock.patch.object(Path, "resolve", return_value=Path(r"\\server\share\ComfyUI")):
            with self.assertRaisesRegex(H3BridgeError, "UNC"):
                _resolve_local_path(Path("C:/local/ComfyUI"), "ComfyUI runtime")

    def test_history_result_is_contained_in_runtime_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "output" / "video" / "MiniMax_H3_00001_.mp4"
            result.parent.mkdir(parents=True)
            result.write_bytes(b"video")
            history = {
                "job": {
                    "outputs": {
                        "14": {
                            "images": [
                                {"filename": result.name, "subfolder": "video", "type": "output"}
                            ]
                        }
                    }
                }
            }
            self.assertEqual(extract_history_video(history, "job", root), result.resolve())

    def test_history_result_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.mp4"
            outside.write_bytes(b"video")
            history = {
                "job": {
                    "outputs": {
                        "14": {
                            "images": [
                                {"filename": "outside.mp4", "subfolder": "../..", "type": "output"}
                            ]
                        }
                    }
                }
            }
            with self.assertRaises(H3BridgeError):
                extract_history_video(history, "job", root)

    def test_history_html_escapes_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            video = output / "MiniMax_H3_unsafe&name.mp4"
            video.write_bytes(b"video")
            items = list_history(None, output)
            rendered = history_html(items)
            self.assertNotIn("unsafe&name", rendered)
            self.assertIn("unsafe&amp;name", rendered)

    def test_history_html_uses_captured_file_size_after_the_file_disappears(self):
        item = HistoryItem(
            Path("missing.mp4"),
            1_700_000_000.0,
            "Forge Neo",
            5 * 1024**2,
        )
        rendered = history_html([item])
        self.assertIn("5.0 MiB", rendered)
        self.assertIn("MP4", rendered)

    def test_cache_history_rejects_unlisted_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            outside = Path(temporary) / "outside.mp4"
            outside.write_bytes(b"video")
            with self.assertRaises(H3BridgeError):
                cache_history_video(str(outside), [], output)

    def test_cache_history_reports_a_listed_output_deleted_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            video = output / "MiniMax_H3_deleted.mp4"
            video.write_bytes(b"video")
            item = HistoryItem(video.resolve(), video.stat().st_mtime, "Forge Neo")
            video.unlink()
            with self.assertRaisesRegex(H3BridgeError, "削除されたか、移動"):
                cache_history_video(str(video), [item], output)

    def test_history_request_restores_valid_forge_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            video = output / "MiniMax_H3_restore.mp4"
            video.write_bytes(b"video")
            metadata = {
                "model": "MiniMax H3",
                "mode": MODE_REFERENCES,
                "prompt": "Restore this synchronized scene.",
                "aspect": "9:16",
                "quality": "balanced",
                "requested_seconds": 7.5,
                "steps": 24,
                "seed": 42,
                "scheduler": "beta",
                "ref_image_size": "max",
            }
            video.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            item = HistoryItem(video.resolve(), video.stat().st_mtime, "Forge Neo")

            request = load_history_request(str(video), [item], output)

            self.assertEqual(request.mode, MODE_REFERENCES)
            self.assertEqual(request.prompt, metadata["prompt"])
            self.assertEqual(request.aspect, "9:16")
            self.assertEqual(request.duration_seconds, 7.5)
            self.assertEqual(request.steps, 24)
            self.assertEqual(request.seed, 42)
            self.assertEqual(request.ref_image_size, "max")

    def test_history_request_rejects_external_or_incomplete_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            external = root / "runtime" / "MiniMax_H3_external.mp4"
            external.parent.mkdir()
            external.write_bytes(b"video")
            external_item = HistoryItem(external.resolve(), external.stat().st_mtime, "ComfyUI")
            with self.assertRaisesRegex(H3BridgeError, "Forge Neoで保存"):
                load_history_request(str(external), [external_item], output)

            video = output / "MiniMax_H3_incomplete.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".json").write_text(
                json.dumps({"model": "MiniMax H3", "prompt": "missing fields"}),
                encoding="utf-8",
            )
            item = HistoryItem(video.resolve(), video.stat().st_mtime, "Forge Neo")
            with self.assertRaisesRegex(H3BridgeError, "設定が不足"):
                load_history_request(str(video), [item], output)

    def test_mirror_result_writes_video_and_utf8_metadata_without_bom(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            write_video(source, width=864, height=480, frames=124)
            request = H3Request(
                mode=MODE_TEXT,
                prompt="A scene with stereo ambience.",
                aspect="16:9",
                quality="preview",
                duration_seconds=5,
                steps=20,
                seed=271828,
            )

            target = mirror_result(
                source,
                root / "output",
                request,
                "0123456789abcdef",
                271828,
                self.ready_runtime(),
            )

            self.assertEqual(target.read_bytes(), source.read_bytes())
            metadata_path = target.with_suffix(".json")
            metadata_bytes = metadata_path.read_bytes()
            self.assertFalse(metadata_bytes.startswith(b"\xef\xbb\xbf"))
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            self.assertEqual(metadata["prompt_id"], "0123456789abcdef")
            self.assertEqual(metadata["prompt"], request.prompt)
            self.assertEqual(metadata["dimensions"], [864, 480])
            self.assertEqual(metadata["frames"], 124)
            self.assertEqual(metadata["steps"], 20)
            self.assertEqual(metadata["seed"], 271828)
            self.assertEqual(metadata["ref_image_size"], "match")
            self.assertEqual(metadata["attention_backend"], "comfy-kitchen-int8")
            self.assertEqual(metadata["comfyui_version"], "0.31.0")
            self.assertEqual(metadata["comfy_kitchen_version"], "0.2.30")
            self.assertEqual(metadata["comfyui_revision"], H3_MINIMUM_COMFY_COMMIT)
            self.assertEqual(metadata["runtime_profile"], RUNTIME_PROFILE_FAST)

    def test_mirror_result_does_not_publish_final_video_when_copy_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"generated video")
            output = root / "output"

            def fail_after_partial_copy(_source, target):
                Path(target).write_bytes(b"partial video")
                raise OSError("disk full during copy")

            with mock.patch(
                "modules_forge.minimax_h3_bridge.shutil.copy2",
                side_effect=fail_after_partial_copy,
            ), self.assertRaisesRegex(H3BridgeError, "disk full"):
                mirror_result(
                    source,
                    output,
                    H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
                    "copyfail01234567",
                    1,
                    self.ready_runtime(),
                )

            self.assertEqual(list(output.glob("MiniMax_H3_*.mp4")), [])
            self.assertEqual(list(output.glob("MiniMax_H3_*.json")), [])

    def test_mirror_result_does_not_publish_final_video_when_metadata_write_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            write_video(source, width=864, height=480, frames=124)
            output = root / "output"

            with mock.patch.object(
                Path,
                "write_text",
                side_effect=OSError("disk full during metadata write"),
            ), self.assertRaisesRegex(H3BridgeError, "metadata"):
                mirror_result(
                    source,
                    output,
                    H3Request(mode=MODE_TEXT, prompt="A scene with stereo ambience."),
                    "metadatafail1234",
                    2,
                    self.ready_runtime(),
                )

            self.assertEqual(list(output.glob("MiniMax_H3_*.mp4")), [])
            self.assertEqual(list(output.glob("MiniMax_H3_*.json")), [])

    def test_history_cache_uses_distinct_targets_for_same_basename_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "runtime-a" / "video" / "same-name.mp4"
            second = root / "runtime-b" / "video" / "same-name.mp4"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first video")
            second.write_bytes(b"second video")
            now = time.time()
            os.utime(first, (now - 2, now - 2))
            os.utime(second, (now - 1, now - 1))
            items = [
                HistoryItem(first.resolve(), first.stat().st_mtime, "runtime-a"),
                HistoryItem(second.resolve(), second.stat().st_mtime, "runtime-b"),
            ]
            output = root / "output"

            first_cached = Path(cache_history_video(str(first), items, output))
            second_cached = Path(cache_history_video(str(second), items, output))

            self.assertNotEqual(first_cached, second_cached)
            self.assertEqual(first_cached.read_bytes(), b"first video")
            self.assertEqual(second_cached.read_bytes(), b"second video")


@unittest.skipUnless(os.environ.get("MINIMAX_H3_LIVE_TEST") == "1", "set MINIMAX_H3_LIVE_TEST=1")
class MiniMaxH3LiveRuntimeTests(unittest.TestCase):
    def test_ck_attention_minimal_audio_video_smoke(self):
        from modules_forge.minimax_h3_runtime import managed_runtime_root

        runtime = Path(
            os.environ.get("MINIMAX_H3_RUNTIME", managed_runtime_root())
        ).resolve()
        readiness = inspect_readiness(runtime)
        self.assertTrue(readiness.ready_for_fl2va, readiness)
        self.assertTrue(readiness.ck_attention_available)

        configured_seed = os.environ.get("MINIMAX_H3_SMOKE_SEED")
        seed = int(configured_seed) if configured_seed is not None else time.time_ns() % (2**53)
        request = H3Request(
            mode=MODE_TEXT,
            prompt=(
                "A single red paper lantern sways gently in a dark studio. "
                "Static camera. Soft cloth rustle and quiet room ambience in stereo."
            ),
            seed=seed,
            steps=2,
        )
        workflow = build_workflow(request, {}, seed=request.seed)
        workflow["5"]["inputs"].update({"width": 320, "height": 192, "length": 5})
        workflow["8"]["inputs"]["steps"] = 2
        workflow["14"]["inputs"]["filename_prefix"] = "video/Forge_Neo_H3_CK_live_smoke"

        client = ComfyH3Client()
        prompt_id = client.submit(workflow, str(uuid.uuid4()))
        started = time.monotonic()
        terminal = False
        try:
            while time.monotonic() - started < 2700:
                job = client.job(prompt_id)
                status = str(job.get("status") or "pending").lower()
                if status in {"completed", "success"}:
                    terminal = True
                    break
                if status in {"failed", "error", "cancelled", "canceled"}:
                    terminal = True
                    self.fail(f"H3 live smoke ended as {status}: {job}")
                time.sleep(5.0)
            else:
                self.fail("H3 live smoke did not finish within 45 minutes")

            history = client.history(prompt_id)
            entry = history.get(prompt_id) or {}
            self.assertIsInstance(entry, dict)
            messages = (entry.get("status") or {}).get("messages") or []
            self.assertIsInstance(messages, list)
            cached_nodes: set[str] = set()
            for message in messages:
                if (
                    isinstance(message, (list, tuple))
                    and len(message) >= 2
                    and message[0] == "execution_cached"
                    and isinstance(message[1], dict)
                ):
                    cached_nodes.update(str(node) for node in message[1].get("nodes") or [])
            critical_nodes = {"10", "11", "12", "14"}
            self.assertFalse(
                critical_nodes & cached_nodes,
                f"live smoke reused critical nodes from cache: {sorted(cached_nodes)}",
            )
            result = extract_history_video(history, prompt_id, runtime)
            import av

            with av.open(os.fspath(result)) as container:
                self.assertTrue(container.streams.video)
                self.assertTrue(container.streams.audio)
                video_frames = list(container.decode(video=0))
            self.assertGreaterEqual(len(video_frames), 5)
            first_video_frame = video_frames[0].to_ndarray().astype("int16")
            self.assertGreater(float(first_video_frame.std()), 0.0)
            self.assertGreater(
                max(
                    float(abs(frame.to_ndarray().astype("int16") - first_video_frame).mean())
                    for frame in video_frames[1:]
                ),
                0.0,
                "generated video must change across frames",
            )

            with av.open(os.fspath(result)) as container:
                audio_stream = container.streams.audio[0]
                audio_frames = list(container.decode(audio=0))
                self.assertEqual(audio_stream.codec_context.sample_rate, 32000)
                self.assertEqual(audio_stream.codec_context.channels, 2)
            self.assertTrue(audio_frames)
            self.assertGreater(
                max(float(abs(frame.to_ndarray()).max()) for frame in audio_frames),
                0.0,
                "generated audio must not be silent",
            )
            elapsed = time.monotonic() - started
            sys.stderr.write(f"H3_LIVE_SEED={seed}\n")
            sys.stderr.write(f"H3_LIVE_RESULT={result}\n")
            sys.stderr.write(f"H3_LIVE_ELAPSED_SECONDS={elapsed:.1f}\n")
        finally:
            if not terminal:
                try:
                    client.cancel(prompt_id)
                except H3BridgeError:
                    pass
            client.close()


class MiniMaxH3PromptHelperTests(unittest.TestCase):
    def test_prompt_template_preserves_user_text_and_adds_av_sections(self):
        rendered = prompt_template("A cat crosses a quiet kitchen.")
        self.assertIn("A cat crosses a quiet kitchen.", rendered)
        self.assertIn("Camera:", rendered)
        self.assertIn("Audio:", rendered)

    def test_prompt_chip_is_idempotent(self):
        first = append_prompt_section("Scene", "sfx")
        second = append_prompt_section(first, "sfx")
        self.assertEqual(first, second)

    @mock.patch("modules_forge.minimax_h3_bridge._validate_media_path", side_effect=lambda value, expected: Path(value))
    @mock.patch("modules_forge.minimax_h3_bridge._probe_media")
    def test_reference_guide_uses_paired_audio_presentation_order(self, probe, _validate):
        probe.side_effect = [(4.0, True, 24.0), (3.0, True, None)]
        rendered = reference_guide_html(["hero.png"], ["motion.mp4"], ["voice.wav"])
        picture = rendered.index("&lt;Picture 1&gt;")
        paired_audio = rendered.index("&lt;Audio 1&gt;")
        video = rendered.index("&lt;Video 1&gt;")
        standalone_audio = rendered.index("&lt;Audio 2&gt;")
        self.assertLess(picture, paired_audio)
        self.assertLess(paired_audio, video)
        self.assertLess(video, standalone_audio)
        self.assertIn("合計 3/12", rendered)

    def test_reference_guide_reports_audio_only_before_generate(self):
        with mock.patch("modules_forge.minimax_h3_bridge._validate_media_path", return_value=Path("voice.wav")), mock.patch(
            "modules_forge.minimax_h3_bridge._probe_media", return_value=(3.0, True, None)
        ):
            rendered = reference_guide_html(None, None, ["voice.wav"])
        self.assertIn("音声だけでは生成できません", rendered)
        self.assertIn('data-tone="error"', rendered)

    def test_cleanup_only_removes_managed_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            managed = runtime / "input" / "forge_h3" / "copy.png"
            outside = runtime / "input" / "keep.png"
            managed.parent.mkdir(parents=True)
            managed.write_bytes(b"copy")
            outside.write_bytes(b"original")
            cleanup_prepared_media(
                {"images": ["forge_h3/copy.png", "keep.png"], "videos": [], "audios": []},
                runtime,
            )
            self.assertFalse(managed.exists())
            self.assertTrue(outside.exists())

    def test_stale_cleanup_only_reaps_owned_old_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            managed_root = runtime / "input" / "forge_h3"
            managed_root.mkdir(parents=True)
            stale = managed_root / "0123456789ab_old.png"
            recent = managed_root / "abcdef012345_recent.png"
            foreign = managed_root / "keep.png"
            for path in (stale, recent, foreign):
                path.write_bytes(b"test")
            with mock.patch("modules_forge.minimax_h3_bridge.time.time", return_value=10_000.0):
                os.utime(stale, (1.0, 1.0))
                os.utime(recent, (9_999.0, 9_999.0))
                cleanup_stale_prepared_media(runtime, max_age_seconds=60)
            self.assertFalse(stale.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(foreign.exists())

    def test_stale_cleanup_rejects_managed_root_resolved_to_unc(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            input_root = runtime / "input"
            input_root.mkdir()
            with mock.patch.object(
                Path,
                "resolve",
                side_effect=[input_root, Path(r"\\server\share\forge_h3")],
            ):
                with self.assertRaisesRegex(H3BridgeError, "UNC"):
                    cleanup_stale_prepared_media(runtime)

    def test_stale_cleanup_rejects_managed_root_outside_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            input_root = runtime / "input"
            outside = runtime / "outside"
            input_root.mkdir()
            outside.mkdir()
            with mock.patch.object(Path, "resolve", side_effect=[input_root, outside]):
                with self.assertRaisesRegex(H3BridgeError, "外"):
                    cleanup_stale_prepared_media(runtime)

    def test_managed_copy_gets_current_mtime_instead_of_source_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "old-reference.png"
            runtime = root / "ComfyUI"
            (runtime / "input").mkdir(parents=True)
            source.write_bytes(b"reference")
            old_time = time.time() - 30 * 24 * 60 * 60
            os.utime(source, (old_time, old_time))
            copied_at = time.time()
            relative_name = _copy_to_comfy_input(source, runtime)
            copied = runtime / "input" / relative_name
            self.assertGreaterEqual(copied.stat().st_mtime, copied_at - 1.0)

    def test_deferred_cleanup_treats_known_cancel_404_as_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            managed = runtime / "input" / "forge_h3" / "copy.png"
            managed.parent.mkdir(parents=True)
            managed.write_bytes(b"copy")
            client = mock.Mock()
            client.job.side_effect = H3JobNotFound("HTTP 404")
            h3_bridge._CANCEL_ACK_IDS.add("11111111-1111-4111-8111-111111111111")
            _cleanup_after_terminal(
                client,
                "11111111-1111-4111-8111-111111111111",
                {"images": ["forge_h3/copy.png"]},
                runtime,
                wait_seconds=0.1,
            )
            self.assertFalse(managed.exists())
            self.assertFalse(_is_cancelled_job("11111111-1111-4111-8111-111111111111"))


if __name__ == "__main__":
    unittest.main()
