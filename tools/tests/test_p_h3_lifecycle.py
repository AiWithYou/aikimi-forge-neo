"""P-H3 jobs keep GPU ownership until a submitted job is reconciled."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from modules_forge import p_h3_image as ph3


def bridge_and_client():
    class BridgeError(RuntimeError):
        pass

    class Rejected(BridgeError):
        pass

    class Missing(BridgeError):
        pass

    client = Mock()
    bridge = SimpleNamespace(
        H3BridgeError=BridgeError, H3SubmissionRejected=Rejected, H3JobNotFound=Missing,
        _RUNTIME_LIFECYCLE_LOCK=threading.RLock(), _ACTIVE_GENERATION_LOCK=threading.RLock(),
        _GPU_OWNERSHIPS={}, _is_cancelled_job=Mock(return_value=False),
        _clear_cancelled_job=Mock(), _mark_active_generation=Mock(), _clear_active_generation=Mock(),
        _cancel_confirmed=Mock(return_value=False), _schedule_deferred_cleanup=Mock(),
        _loopback_server_process=Mock(return_value=SimpleNamespace(pid=1, create_time=lambda: 1)),
        pending_jobs=Mock(), release_forge_vram=Mock(), _release_retained_runtime=Mock(),
        cleanup_prepared_media=Mock(), cleanup_stale_prepared_media=Mock(),
        ComfyH3Client=Mock(return_value=client), _execution_error=lambda job: "failed",
    )

    def finish(prompt_id):
        owned = bridge._GPU_OWNERSHIPS.pop(prompt_id, None)
        if owned is not None:
            owned.release()

    bridge._finish_gpu_generation = Mock(side_effect=finish)
    return bridge, client


def run_with_bridge(tmp_path, bridge, client, ownership):
    request = ph3.PH3Request("A photograph", use_upsampler=False)
    readiness = SimpleNamespace(core_revision="a" * 40, comfy_version="test", runtime_profile="fast", package_versions={})
    with (
        patch.object(ph3, "_bridge", return_value=bridge),
        patch.object(ph3, "ensure_runtime", return_value=readiness),
        patch.object(ph3, "prepare_images", return_value={"images": []}),
        patch.object(ph3, "check_nodes"),
        patch("modules_forge.gpu_ownership.GPUOwnership", return_value=ownership),
        patch("modules_forge.gpu_residency.register"),
        patch("modules_forge.studio_checkpoints.snapshot_checkpoint", return_value=None),
        patch("modules_forge.studio_checkpoints.verify_snapshot"),
    ):
        yield from ph3.run_generation(request, Path("runtime"), "http://127.0.0.1:8189", tmp_path, tmp_path, poll_seconds=0.01)


def test_record_failure_before_submission_does_not_send_and_releases_gpu(tmp_path):
    bridge, client = bridge_and_client()
    bridge.pending_jobs.write.side_effect = OSError("record unavailable")
    ownership = Mock()
    ownership.acquire.return_value = True
    with pytest.raises(OSError, match="record unavailable"):
        list(run_with_bridge(tmp_path, bridge, client, ownership))
    client.submit.assert_not_called()
    ownership.release.assert_called_once()
    bridge.cleanup_prepared_media.assert_called_once()


def test_record_failure_after_submission_defers_cleanup_and_gpu_release(tmp_path):
    bridge, client = bridge_and_client()
    bridge.pending_jobs.update.side_effect = OSError("record unavailable")
    ownership = Mock()
    ownership.acquire.return_value = True
    with pytest.raises(OSError, match="record unavailable"):
        list(run_with_bridge(tmp_path, bridge, client, ownership))
    client.submit.assert_called_once()
    bridge._schedule_deferred_cleanup.assert_called_once()
    bridge.cleanup_prepared_media.assert_not_called()
    ownership.release.assert_not_called()
    assert ownership in bridge._GPU_OWNERSHIPS.values()


def test_cancelled_job_releases_gpu_only_after_terminal_status(tmp_path):
    bridge, client = bridge_and_client()
    client.job.return_value = {"status": "cancelled"}
    ownership = Mock()
    ownership.acquire.return_value = True
    with pytest.raises(ph3.PH3Cancelled):
        list(run_with_bridge(tmp_path, bridge, client, ownership))
    bridge._finish_gpu_generation.assert_called_once()
    bridge.cleanup_prepared_media.assert_called_once()
    ownership.release.assert_called_once()


def test_model_root_change_before_submission_is_rejected(tmp_path):
    from modules_forge import local_assets
    from modules_forge.minimax_h3_runtime import model_config

    bridge, client = bridge_and_client()
    client.job.return_value = {"status": "cancelled"}
    ownership = Mock()
    ownership.acquire.return_value = True
    runtime = tmp_path / "runtime"
    first = runtime / "models/diffusion_models" / ph3.MODEL
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second = tmp_path / "new-models/diffusion_models" / ph3.MODEL
    second.parent.mkdir(parents=True)
    second.write_bytes(b"different")
    request = ph3.PH3Request("photo", use_upsampler=False)
    with (
        patch.object(ph3, "_bridge", return_value=bridge),
        patch.object(ph3, "ensure_runtime", return_value=SimpleNamespace()) as ensure,
        patch.object(ph3, "prepare_images", return_value={"images": []}),
        patch.object(ph3, "check_nodes"),
        patch("modules_forge.gpu_ownership.GPUOwnership", return_value=ownership),
        patch("modules_forge.gpu_residency.register"),
        patch.object(local_assets, "HASH_CACHE", tmp_path / "hashes.json"),
    ):
        updates = ph3.run_generation(request, runtime, "http://127.0.0.1:8189", tmp_path, tmp_path, poll_seconds=0.01)
        assert next(updates)["stage"] == "verifying"
        assert next(updates)["stage"] == "runtime"
        (runtime / "extra_model_paths.yaml").write_text(model_config(second.parents[1]), encoding="utf-8")
        with pytest.raises(ph3.PH3Error, match="モデル保存先が変更"):
            next(updates)
        ensure.assert_not_called()
        client.submit.assert_not_called()
    assert request.checkpoint == ""


def test_standard_checkpoint_is_registered_by_absolute_path(tmp_path):
    bridge, client = bridge_and_client()
    client.job.return_value = {"status": "cancelled"}
    ownership = Mock()
    ownership.acquire.return_value = True
    request = ph3.PH3Request("photo", use_upsampler=False)
    with (
        patch.object(ph3, "_bridge", return_value=bridge),
        patch.object(ph3, "ensure_runtime", return_value=SimpleNamespace()) as ensure,
        patch.object(ph3, "prepare_images", return_value={"images": []}),
        patch.object(ph3, "check_nodes"),
        patch("modules_forge.gpu_ownership.GPUOwnership", return_value=ownership),
        patch("modules_forge.gpu_residency.register"),
        patch("modules_forge.studio_checkpoints.snapshot_checkpoint", return_value=None),
        patch("modules_forge.studio_checkpoints.verify_snapshot"),
    ):
        updates = ph3.run_generation(request, tmp_path, "http://127.0.0.1:8189", tmp_path, tmp_path, poll_seconds=0.01)
        assert next(updates)["stage"] == "verifying"
        assert next(updates)["stage"] == "runtime"
        assert next(updates)["stage"] == "queued"
        expected = tmp_path / "models/diffusion_models" / ph3.MODEL
        assert ensure.call_args.args[0].checkpoint == str(expected.resolve())
        with pytest.raises(ph3.PH3Cancelled):
            next(updates)


def test_cancel_during_initial_model_check_does_not_acquire_gpu(tmp_path):
    bridge, client = bridge_and_client()
    ownership = Mock()
    request = ph3.PH3Request("photo", use_upsampler=False)
    with (patch.object(ph3, "_bridge", return_value=bridge),
          patch("modules_forge.gpu_ownership.GPUOwnership", return_value=ownership),
          patch("modules_forge.studio_checkpoints.snapshot_checkpoint") as snapshot):
        updates = ph3.run_generation(request, tmp_path, "http://127.0.0.1:8189", tmp_path, tmp_path)
        event = next(updates)
        assert event["stage"] == "verifying"
        assert event["prompt_id"]
        bridge._is_cancelled_job.return_value = True
        with pytest.raises(ph3.PH3Cancelled):
            next(updates)
    snapshot.assert_not_called()
    ownership.acquire.assert_not_called()
    client.submit.assert_not_called()
    bridge.pending_jobs.write.assert_not_called()
    bridge._clear_cancelled_job.assert_called_once_with(event["prompt_id"])


def test_initial_cancel_uses_the_real_stop_flag_and_clears_it(tmp_path):
    from modules_forge import minimax_h3_bridge as bridge

    ownership = Mock()
    request = ph3.PH3Request("photo", use_upsampler=False)
    with (patch("modules_forge.gpu_ownership.GPUOwnership", return_value=ownership),
          patch("modules_forge.studio_checkpoints.snapshot_checkpoint") as snapshot,
          patch.object(bridge, "ComfyH3Client") as client,
          patch.object(bridge.pending_jobs, "write") as pending):
        updates = ph3.run_generation(request, tmp_path, "http://127.0.0.1:8189", tmp_path, tmp_path)
        event = next(updates)
        bridge.cancel_generation(event["prompt_id"], "http://127.0.0.1:8189")
        with pytest.raises(ph3.PH3Cancelled):
            next(updates)
    assert not bridge._is_cancelled_job(event["prompt_id"])
    assert event["prompt_id"] not in bridge._ACTIVE_GENERATION_IDS
    assert event["prompt_id"] not in bridge._GPU_OWNERSHIPS
    ownership.acquire.assert_not_called()
    snapshot.assert_not_called()
    pending.assert_not_called()
    client.assert_not_called()
