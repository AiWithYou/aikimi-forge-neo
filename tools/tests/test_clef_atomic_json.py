"""Atomic results tolerate Windows reader locks without sharing temporary paths."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules_forge.clef import core


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_lock_retries_without_truncating_the_saved_result(tmp_path, monkeypatch, winerror):
    path = tmp_path / "result.json"
    path.write_text('{"old":true}', encoding="utf-8")
    replace = os.replace
    calls = []

    def locked_once(source, target):
        calls.append(Path(source))
        if len(calls) == 1:
            assert json.loads(path.read_text()) == {"old": True}
            exc = PermissionError("reader still open")
            exc.winerror = winerror
            raise exc
        return replace(source, target)

    monkeypatch.setattr(core.os, "replace", locked_once)
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    core.atomic_json(path, {"complete": True})
    assert len(calls) == 2 and json.loads(path.read_text()) == {"complete": True}
    assert list(tmp_path.iterdir()) == [path]


def test_permanent_failure_preserves_result_and_removes_only_its_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"old":true}', encoding="utf-8")
    unrelated = tmp_path / "unrelated.tmp"
    unrelated.write_text("keep")
    calls = []

    def blocked(source, target):
        calls.append(source)
        exc = PermissionError("permanent access denial")
        exc.winerror = 5
        raise exc

    monkeypatch.setattr(core.os, "replace", blocked)
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        core.atomic_json(path, {"new": True})
    assert 1 < len(calls) <= 10
    assert json.loads(path.read_text()) == {"old": True}
    assert set(tmp_path.iterdir()) == {path, unrelated}


def test_other_io_errors_are_not_retried_or_hidden(tmp_path, monkeypatch):
    calls = []

    def fail(source, target):
        calls.append(source)
        raise OSError("disk failure")

    monkeypatch.setattr(core.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        core.atomic_json(tmp_path / "result.json", {"new": True})
    assert len(calls) == 1 and not list(tmp_path.iterdir())


def test_concurrent_saves_use_separate_temporary_files_and_publish_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    replace = os.replace
    barrier = threading.Barrier(2)
    sources = []
    thread = threading.local()

    def simultaneous(source, target):
        sources.append(Path(source))
        if not getattr(thread, "waited", False):
            thread.waited = True
            barrier.wait(timeout=5)
        return replace(source, target)

    monkeypatch.setattr(core.os, "replace", simultaneous)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(core.atomic_json, path, {"writer": index, "body": "x" * 1000}) for index in range(2)]
        for result in results:
            result.result(timeout=10)
    assert len(set(sources)) == 2
    assert json.loads(path.read_text())["body"] == "x" * 1000
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows denies replace while a standard reader is open")
def test_real_windows_reader_can_close_during_the_save_retry(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"old":true}', encoding="utf-8")
    replace = os.replace
    attempted = threading.Event()

    def observed(source, target):
        try:
            return replace(source, target)
        except PermissionError:
            attempted.set()
            raise

    monkeypatch.setattr(core.os, "replace", observed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = path.open("r", encoding="utf-8")
        try:
            result = pool.submit(core.atomic_json, path, {"complete": True})
            assert attempted.wait(timeout=2), "The reader must cause a real sharing violation first"
            assert json.load(reader) == {"old": True}
        finally:
            reader.close()
        result.result(timeout=5)
    assert json.loads(path.read_text()) == {"complete": True}


def test_cleanup_failure_does_not_hide_the_original_save_error(tmp_path, monkeypatch):
    def save_error(*args):
        raise OSError("original save failure")

    def cleanup_error(*args, **kwargs):
        raise PermissionError("temporary cleanup locked")

    monkeypatch.setattr(core.os, "replace", save_error)
    monkeypatch.setattr(Path, "unlink", cleanup_error)
    with pytest.raises(OSError, match="original save failure") as error:
        core.atomic_json(tmp_path / "result.json", {"new": True})
    assert "temporary cleanup locked" in error.value.__notes__[0]


def test_worker_saves_both_records_after_a_transient_lock_without_repeating_inference(tmp_path, monkeypatch):
    from modules_forge.clef.collection import parse_records
    from modules_forge.clef.service import snapshot_inputs
    from tools.clef_worker import process_run

    directory = tmp_path / "run"
    items = snapshot_inputs(parse_records("first\nsecond", "lines"), directory)
    request = {"questions": {"stopped": {"type": "noul", "instructions": "The service has stopped."}}}
    core.atomic_json(directory / "request.json", request)
    core.atomic_json(directory / "result.json", {"request": request, "items": items, "status": "running"})
    replace = os.replace
    states = []
    failures = []

    def locked_on_second_result(source, target):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        if payload["items"][1]["status"] == "done" and not failures:
            failures.append(True)
            exc = PermissionError("reader still open on the second result")
            exc.winerror = 32
            raise exc
        return replace(source, target)

    def decide(value):
        states.append(value["state"])
        return {"answers": {"stopped": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 2}}

    monkeypatch.setattr(core.os, "replace", locked_on_second_result)
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    result = process_run(directory, SimpleNamespace(decide=decide))
    assert failures == [True] and states == ["first", "second"]
    assert result["status"] == "complete"
    assert [item["answers"]["stopped"]["noul"] for item in result["items"]] == [0.7, 0.7]
    assert json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
    assert (directory / "results.csv").is_file()
    assert not list(directory.glob("*.tmp"))
