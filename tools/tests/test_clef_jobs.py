"""Cancellation, partial results and GPU ownership do not depend on the GUI."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from modules_forge.clef.core import TEMPLATES, ClefError, atomic_json
from modules_forge.clef.service import Studio, snapshot_inputs
from tools.clef_worker import process_run


def request():
    return {
        "profile": "flash-int8",
        "questions": TEMPLATES["画像の評価"],
        "state": "evaluate",
        "max_pixels": 262144,
        "max_length": 4096,
    }


def image(tmp_path, name):
    target = tmp_path / name
    Image.new("RGB", (32, 32), "red").save(target)
    return str(target)


def test_snapshots_keep_duplicate_names_distinct_and_original_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first, second = image(a, "same.png"), image(b, "same.png")
    Image.new("RGB", (32, 32), "blue").save(second)
    items = snapshot_inputs([first, second], tmp_path / "run")
    assert len(items) == 2 and items[0]["path"] != items[1]["path"]
    assert items[0]["sha256"] != items[1]["sha256"]
    assert Path(items[0]["path"]).read_bytes() == Path(first).read_bytes()


def test_corrupt_images_are_refused_before_start(tmp_path):
    corrupt = tmp_path / "image.png"
    corrupt.write_text("not an image")
    with pytest.raises(ClefError):
        snapshot_inputs([str(corrupt)], tmp_path / "run")


def test_batch_stop_keeps_first_result_and_marks_remaining(tmp_path):
    items = snapshot_inputs([image(tmp_path, "one.png"), image(tmp_path, "two.png")], tmp_path / "run")
    directory = tmp_path / "run"
    atomic_json(directory / "request.json", request())
    atomic_json(directory / "result.json", {"request": request(), "items": items, "status": "running"})

    def decide(*args):
        (directory / "cancel").touch()
        return {"answers": {"text": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 10, "output_tokens": 0}}

    result = process_run(directory, SimpleNamespace(decide=decide))
    assert result["status"] == "partial"
    assert [x["status"] for x in result["items"]] == ["done", "cancelled"]
    assert (directory / "results.csv").is_file()


def test_batch_publishes_each_item_without_serializing_the_full_batch_again(tmp_path, monkeypatch):
    import modules_forge.clef.service as service
    import tools.clef_worker as worker
    from modules_forge.clef.collection import parse_records

    directory = tmp_path / "run"
    count = 32
    items = snapshot_inputs(parse_records("\n".join(f"record {index}" for index in range(count)), "lines"), directory)
    atomic_json(directory / "request.json", request())
    atomic_json(directory / "result.json", {"request": request(), "items": items, "status": "running"})
    studio = Studio(tmp_path / "runtime", tmp_path / "outputs")
    studio.jobs["run"] = service.Job("owner", directory, Mock())
    serialized_items = []
    original = worker.atomic_json
    calls = []

    def save(path, value):
        serialized_items.append(len(value.get("items", [])))
        return original(path, value)

    def decide(value):
        visible = studio.status("run", "owner")["result"]
        assert sum(item["status"] == "done" for item in visible["items"]) == len(calls)
        calls.append(value["state"])
        return {"answers": {}, "usage": {"input_tokens": 2}}

    monkeypatch.setattr(worker, "atomic_json", save)
    result = process_run(directory, SimpleNamespace(decide=decide))
    assert result["status"] == "complete" and len(calls) == count
    assert sum(serialized_items) == count, "Only the final aggregate may serialize all items"
    assert len((directory / "results.jsonl").read_text(encoding="utf-8").splitlines()) == count
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
    studio.jobs.clear()


def test_partial_result_reader_ignores_an_unfinished_utf8_journal_tail(tmp_path):
    from modules_forge.clef.core import read_result

    directory = tmp_path / "run"
    atomic_json(directory / "result.json", {"status": "running", "items": [{"id": "0", "status": "pending"}]})
    completed = {"item": {"id": "0", "status": "done", "answers": {}}, "seconds": 0.5}
    (directory / "results.jsonl").write_bytes(json.dumps(completed).encode() + b"\n" + b'{"name":"\xe6\x97')
    result = read_result(directory)
    assert result["items"][0]["status"] == "done" and result["seconds"] == 0.5


def test_resume_recovers_journal_results_and_does_not_repeat_completed_inference(tmp_path, monkeypatch):
    import modules_forge.clef.service as service
    from modules_forge.clef.core import PREPROCESSING_VERSION, fingerprint

    identifier = "20261011T120000-12345678"
    directory = tmp_path / "outputs" / identifier
    items = snapshot_inputs([image(tmp_path, "one.png"), image(tmp_path, "two.png")], directory)
    result = {
        "id": identifier,
        "status": "running",
        "request": request(),
        "items": items,
        "preprocessing": PREPROCESSING_VERSION,
        "fingerprint": fingerprint(request(), items),
    }
    atomic_json(directory / "result.json", result)
    completed = {"item": {**items[0], "status": "done", "answers": {}, "usage": {}}, "seconds": 0.5}
    (directory / "results.jsonl").write_bytes(json.dumps(completed).encode() + b"\n" + b'{"unfinished":')
    calls = []
    runner = SimpleNamespace(decide=lambda *args: calls.append(args) or {"answers": {}, "usage": {}})
    worker = Mock(result=Mock(return_value={"ok": True}))
    worker.start.side_effect = lambda python, script, env, payload, log: process_run(payload["job_dir"], runner)
    lease = Mock(acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime", tmp_path / "outputs", ownership_factory=lambda: lease, worker_factory=lambda *a: worker
    )
    studio._installed = lambda _: "python"
    monkeypatch.setattr(service, "execution_stopped", lambda _: True)
    assert studio.resume(identifier, "owner") == identifier
    assert studio.jobs[identifier].done.wait(5)
    assert len(calls) == 1 and studio.status(identifier, "owner")["result"]["status"] == "complete"
    assert len((directory / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    studio.shutdown()


def test_worker_failure_aggregates_previously_published_items_before_releasing_gpu(tmp_path):
    count = []

    def decide(*args):
        count.append(True)
        if len(count) == 2:
            raise SystemExit("worker exited")
        return {"answers": {"text": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 2}}

    worker = Mock()
    worker.start.side_effect = lambda python, script, env, payload, log: process_run(
        payload["job_dir"], SimpleNamespace(decide=decide)
    )
    lease = Mock(acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime", tmp_path / "outputs", ownership_factory=lambda: lease, worker_factory=lambda *a: worker
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png"), image(tmp_path, "two.png")], "owner")
    assert studio.jobs[identifier].done.wait(5)
    result = studio.status(identifier, "owner")["result"]
    assert result["status"] == "partial"
    assert [item["status"] for item in result["items"]] == ["done", "failed"]
    assert result["items"][0]["answers"]["text"]["noul"] == 0.8
    assert (studio.jobs[identifier].directory / "results.csv").is_file()
    lease.release.assert_called_once()
    studio.shutdown()


def test_item_error_does_not_mark_success_or_lose_following_items(tmp_path):
    directory = tmp_path / "run"
    items = snapshot_inputs([image(tmp_path, "one.png"), image(tmp_path, "two.png")], directory)
    atomic_json(directory / "request.json", request())
    atomic_json(directory / "result.json", {"request": request(), "items": items, "status": "running"})
    decide = Mock(
        side_effect=[ValueError("bad input"), {"answers": {}, "usage": {"input_tokens": 4, "output_tokens": 0}}]
    )
    result = process_run(directory, SimpleNamespace(decide=decide))
    assert result["status"] == "partial"
    assert result["items"][0]["status"] == "error" and result["items"][1]["status"] == "done"


def test_worker_uses_each_fixed_record_and_keeps_completed_items_on_resume(tmp_path):
    from modules_forge.clef.collection import parse_records

    directory = tmp_path / "run"
    items = snapshot_inputs(parse_records("請求が違う\n接続できない", "lines"), directory)
    items[0].update(status="done", answers={"text": {"type": "noul", "noul": 0.9}}, seconds=1.0)
    items[1]["status"] = "pending"
    atomic_json(directory / "request.json", request())
    atomic_json(directory / "result.json", {"request": request(), "items": items, "status": "running"})
    decide = Mock(return_value={"answers": {}, "usage": {"input_tokens": 2, "output_tokens": 0}})
    result = process_run(directory, SimpleNamespace(decide=decide))
    assert decide.call_count == 1
    assert decide.call_args.args[0]["state"] == "接続できない"
    assert result["items"][0]["answers"]["text"]["noul"] == 0.9
    assert result["status"] == "complete"


def test_studio_refuses_other_owner_and_busy_gpu(tmp_path):
    lease = Mock(engine="", acquire=Mock(return_value=False))
    studio = Studio(tmp_path / "runtime", tmp_path / "outputs", ownership_factory=lambda: lease)
    studio._installed = lambda _: str(tmp_path / "python")
    with pytest.raises(ClefError, match="GPU"):
        studio.start(request(), [image(tmp_path, "one.png")], "owner")
    assert not list((tmp_path / "outputs").iterdir())
    with pytest.raises(ClefError):
        studio.status("unknown", "other")


def test_input_failure_removes_only_its_unstarted_directory(tmp_path):
    lease = Mock(acquire=Mock(return_value=True))
    outputs = tmp_path / "outputs"
    existing = outputs / "keep"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("saved")
    bad = tmp_path / "bad.png"
    bad.write_text("broken")
    studio = Studio(tmp_path / "runtime", outputs, ownership_factory=lambda: lease, worker_factory=Mock())
    studio._installed = lambda _: "python"
    with pytest.raises(ClefError, match="読み込めません"):
        studio.start(request(), [image(tmp_path, "one.png"), str(bad)], "owner")
    assert list(outputs.iterdir()) == [existing]
    lease.acquire.assert_not_called()


def test_interrupted_execution_needs_both_exact_processes_to_have_stopped(tmp_path, monkeypatch):
    import modules_forge.clef.service as service

    directory = tmp_path / "run"
    atomic_json(directory / "execution.json", {"pid": 101, "created": 1.0})
    atomic_json(directory / "worker-process.json", {"pid": 102, "created": 2.0})
    active = {101}
    monkeypatch.setattr(service, "process_stopped", lambda data: data["pid"] not in active)
    assert not service.execution_stopped(directory)
    active.clear()
    assert service.execution_stopped(directory)
    (directory / "worker-process.json").unlink()
    assert not service.execution_stopped(directory)


def test_final_status_cannot_freeze_a_result_read_before_completion(tmp_path, monkeypatch):
    from modules_forge.clef.service import Job

    directory = tmp_path / "run"
    atomic_json(directory / "result.json", {"status": "running", "items": []})
    job = Job("owner", directory, Mock())
    studio = Studio(tmp_path / "runtime", tmp_path / "outputs")
    studio.jobs["run"] = job
    original = Path.read_text

    def read_with_completion(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path.name == "result.json":
            atomic_json(path, {"status": "partial", "items": []})
            job.done.set()
        return value

    monkeypatch.setattr(Path, "read_text", read_with_completion)
    info = studio.status("run", "owner")
    assert not (info["done"] and info["result"]["status"] == "running")


def test_process_receipt_rejects_nonfinite_or_boolean_identity():
    import os

    from modules_forge.clef.service import process_stopped

    assert not process_stopped({"pid": os.getpid(), "created": float("nan")})
    assert not process_stopped({"pid": True, "created": 1.0})


def test_worker_failure_closes_process_before_releasing_gpu(tmp_path):
    events = []
    worker = Mock()
    worker.start.return_value = False
    worker.result.side_effect = RuntimeError("worker failed")
    worker.close.side_effect = lambda: events.append("closed")
    lease = Mock(engine="", acquire=Mock(return_value=True))
    lease.release.side_effect = lambda: events.append("released")
    studio = Studio(
        tmp_path / "runtime",
        tmp_path / "outputs",
        ownership_factory=lambda: lease,
        worker_factory=lambda *a: worker,
        release_vram=lambda: None,
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png")], "owner")
    assert studio.jobs[identifier].done.wait(5)
    assert events == ["closed", "released"]
    assert studio.status(identifier, "owner")["status"] == "failed"
    with pytest.raises(ClefError):
        studio.cancel(identifier, "other")


def test_failed_stop_keeps_gpu_lease_until_shutdown_can_confirm_exit(tmp_path):
    worker = Mock()
    worker.result.side_effect = RuntimeError("worker failed")
    worker.close.side_effect = RuntimeError("not confirmed")
    lease = Mock(engine="", acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime",
        tmp_path / "outputs",
        ownership_factory=lambda: lease,
        worker_factory=lambda *a: worker,
        release_vram=lambda: None,
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png")], "owner")
    assert studio.jobs[identifier].done.wait(5)
    lease.release.assert_not_called()
    worker.close.side_effect = None
    studio.shutdown()
    lease.release.assert_called_once()


def test_shutdown_during_startup_does_not_spawn_after_releasing_gpu(tmp_path):
    entered, proceed = threading.Event(), threading.Event()

    def release():
        entered.set()
        assert proceed.wait(3)

    worker = Mock()
    lease = Mock(engine="", acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime",
        tmp_path / "outputs",
        ownership_factory=lambda: lease,
        worker_factory=lambda *args: worker,
        release_vram=release,
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png")], "owner")
    assert entered.wait(3)
    closer = threading.Thread(target=studio.shutdown)
    closer.start()
    try:
        assert not lease.release.called
    finally:
        proceed.set()
        closer.join(5)
        studio.jobs[identifier].done.wait(5)
    assert not worker.start.called
    lease.release.assert_called_once()


def test_thread_start_failure_records_failure_and_does_not_leave_a_busy_job(tmp_path, monkeypatch):
    lease = Mock(engine="", acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime",
        tmp_path / "outputs",
        ownership_factory=lambda: lease,
        worker_factory=Mock(),
        release_vram=lambda: None,
    )
    studio._installed = lambda _: "python"
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("no thread")))
    with pytest.raises(RuntimeError, match="no thread"):
        studio.start(request(), [image(tmp_path, "one.png")], "owner")
    job = next(iter(studio.jobs.values()))
    assert job.done.is_set() and not job.lease_held
    assert json.loads((job.directory / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    lease.release.assert_called_once()


def test_worker_phase_is_visible_until_the_job_finishes(tmp_path):
    worker = Mock(result=Mock(return_value=None))
    lease = Mock(engine="", acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime",
        tmp_path / "outputs",
        ownership_factory=lambda: lease,
        worker_factory=lambda *args: worker,
        release_vram=lambda: None,
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png")], "owner")
    job = studio.jobs[identifier]
    atomic_json(job.directory / "progress.json", {"message": "量子化重みを保存中"})
    try:
        assert studio.status(identifier, "owner")["message"] == "量子化重みを保存中"
    finally:
        studio.cancel(identifier, "owner")
        assert job.done.wait(5)
    assert studio.status(identifier, "owner")["message"] != "量子化重みを保存中"


def test_resume_uses_frozen_inputs_and_never_repeats_done_items(tmp_path):
    calls = []
    runner = SimpleNamespace(decide=lambda *args: calls.append(args[0]) or {"answers": {}, "usage": {}})
    worker = Mock(result=Mock(return_value={"ok": True}))
    worker.start.side_effect = lambda python, script, env, payload, log: process_run(payload["job_dir"], runner)
    lease = Mock(acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime", tmp_path / "outputs", ownership_factory=lambda: lease, worker_factory=lambda *a: worker
    )
    studio._installed = lambda _: "python"
    sources = [image(tmp_path, "one.png"), image(tmp_path, "two.png")]
    identifier = studio.start(request(), sources, "owner")
    assert studio.jobs[identifier].done.wait(5)
    directory = studio.jobs[identifier].directory
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    result["status"] = "partial"
    result["items"][1]["status"] = "cancelled"
    atomic_json(directory / "result.json", result)
    Path(sources[1]).unlink()
    calls.clear()
    assert studio.resume(identifier, "owner") == identifier
    assert studio.jobs[identifier].done.wait(5)
    assert len(calls) == 1 and studio.status(identifier, "owner")["result"]["status"] == "complete"
    studio.shutdown()


def test_resume_rejects_modified_saved_input_before_acquiring_gpu(tmp_path):
    runner = SimpleNamespace(decide=lambda *a: {"answers": {}, "usage": {}})
    worker = Mock(result=Mock(return_value={"ok": True}))
    worker.start.side_effect = lambda python, script, env, payload, log: process_run(payload["job_dir"], runner)
    lease = Mock(acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime", tmp_path / "outputs", ownership_factory=lambda: lease, worker_factory=lambda *a: worker
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png")], "owner")
    assert studio.jobs[identifier].done.wait(5)
    directory = studio.jobs[identifier].directory
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    result["status"] = "partial"
    result["items"][0]["status"] = "cancelled"
    atomic_json(directory / "result.json", result)
    Image.new("RGB", (10, 10), "red").save(result["items"][0]["path"])
    lease.acquire.reset_mock()
    with pytest.raises(ClefError, match="変更"):
        studio.resume(identifier, "owner")
    lease.acquire.assert_not_called()
    studio.shutdown()


def test_resume_refuses_to_mix_a_previous_image_pipeline(tmp_path):
    runner = SimpleNamespace(decide=lambda *a: {"answers": {}, "usage": {}})
    worker = Mock(result=Mock(return_value={"ok": True}))
    worker.start.side_effect = lambda python, script, env, payload, log: process_run(payload["job_dir"], runner)
    lease = Mock(acquire=Mock(return_value=True))
    studio = Studio(
        tmp_path / "runtime", tmp_path / "outputs", ownership_factory=lambda: lease, worker_factory=lambda *a: worker
    )
    studio._installed = lambda _: "python"
    identifier = studio.start(request(), [image(tmp_path, "one.png")], "owner")
    assert studio.jobs[identifier].done.wait(5)
    directory = studio.jobs[identifier].directory
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    result.update(status="partial", preprocessing="exif-rgb-v1")
    result["items"][0]["status"] = "cancelled"
    atomic_json(directory / "result.json", result)
    lease.acquire.reset_mock()
    with pytest.raises(ClefError, match="前処理"):
        studio.resume(identifier, "owner")
    lease.acquire.assert_not_called()
    studio.shutdown()
