"""Job-level IPC lifetime, paid-call limits and strictly offline replay."""

import io
import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from modules_forge.jev_sparse import common, krea2, krea2_jobs, sdk_worker

ENV = {"AIKIMI_JEV_ALLOW_CLOUD": "1", "TYPESAFE_API_KEY": "fake-local-test-key"}


def recorded(tmp_path, client, *, state=None, target="test", close=True):
    state = state or {"target": "test", "evaluation": 1, "tensor_shapes": {"0": [1, 64, 128]}}
    allowed = {"0": (50.0, 100.0)}
    log = common.RunLog(tmp_path, target, {}, "private prompt")
    answer = client.decide(state, allowed, 100.0)
    log.write("decision", source="jev", replay=client.last_replay, keep_percent=answer)
    if close:
        log.finish("completed", api_calls=client.calls)
    return log, state, allowed


def test_worker_reuses_sdk_client_and_closes_once(monkeypatch):
    constructed, entered, exited, calls = [], [], [], []

    class Client:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def __enter__(self):
            entered.append(True)
            return self

        def __exit__(self, *args):
            exited.append(True)

        def system_one(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices={"0": SimpleNamespace(choice=50, confidence=0.9)})

    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(
            TypeSafeClient=Client,
            Choice=lambda **kw: kw,
            RetryPolicy=lambda **kw: kw,
        ),
    )
    monkeypatch.setattr(sdk_worker.importlib.metadata, "version", lambda name: "0.7.0")
    request = json.dumps({"state": {}, "allowed": {"0": [50, 100]}, "timeout": 1}) + "\n"
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO((request * 2).encode())))
    monkeypatch.setattr(sys, "stdout", output)
    assert sdk_worker.main() == 0
    assert len(constructed) == len(entered) == len(exited) == 1
    assert len(calls) == len(output.getvalue().splitlines()) == 2
    assert constructed[0]["retry"] == {"max_retries": 0}


def test_krea_attention_and_tile_share_budget_and_worker(monkeypatch, tmp_path, mock_sdk):
    child = mock_sdk(choice=None)
    monkeypatch.setattr(krea2_jobs, "cloud_source", lambda: ENV)
    monkeypatch.setattr(krea2_jobs, "cancelled", lambda: False)
    original = common.create_client
    monkeypatch.setattr(krea2_jobs, "create_client", lambda **kw: original(sys.executable, **kw))
    options = krea2.Options(
        mode="jev", tile_mode="jev", tile_cadence="stage", decision_cadence="step", job_max_calls=2, min_tokens=64
    )
    session = krea2_jobs.Session(options, tmp_path)
    allocator = session.prepare_tiles([0.003, 0.06], 2, 4, 0.035)
    client = session.get_client()
    run = krea2.new_run(options, 2, tmp_path, client=client, owns_client=False)
    session.run = run
    run.begin_sampling()
    run.begin_evaluation()
    run.observations = {str(i): {"relative_output_norm": 1.0, "cross_evaluation_drift": None} for i in range(2)}
    run.begin_evaluation()
    assert client.calls == 2 and allocator.client is run.client is client
    session.prepare_tiles([0.01], 2, 4, 0.035)
    run.begin_evaluation()
    assert client.calls == 2 and not run.wants_observations()
    assert len(child.children) == 1 and not client.closed
    session.close("completed")
    assert client.closed and child.children[0].poll() == 0
    assert session.budget.calls == 2 and session.budget.wait_seconds > 0
    replay_options = replace(options, replay_logs=json.dumps([str(run.log.path), str(allocator.log.path)]))
    replay_session = krea2_jobs.Session(replay_options, tmp_path / "replay")
    replay_allocator = replay_session.prepare_tiles([0.003, 0.06], 2, 4, 0.035)
    replay_client = replay_session.get_client()
    replay_run = krea2.new_run(replay_options, 2, tmp_path / "replay", client=replay_client, owns_client=False)
    replay_session.run = replay_run
    replay_run.begin_sampling()
    replay_run.begin_evaluation()
    replay_run.observations = {str(i): {"relative_output_norm": 1.0, "cross_evaluation_drift": None} for i in range(2)}
    replay_run.begin_evaluation()
    replay_session.prepare_tiles([0.01], 2, 4, 0.035)
    replay_run.begin_evaluation()
    assert replay_client.calls == replay_client.wait_seconds == 0
    assert replay_client.replay_calls == 2 and replay_allocator.decisions == replay_run.decisions == 1
    assert not replay_run.wants_observations() and len(child.children) == 1
    replay_session.close("completed")


def test_replay_is_offline_and_preserves_decisions(monkeypatch, tmp_path, mock_sdk):
    mock_sdk()
    with common.JevClient(sys.executable, environment=ENV) as live:
        log, state, allowed = recorded(tmp_path, live)
    monkeypatch.setattr(common, "JevClient", lambda *a, **kw: pytest.fail("Replay constructed cloud client"))
    monkeypatch.setenv(common.REPLAY_ENV, json.dumps([str(log.path)]))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    replay = common.create_client()
    assert replay.decide(state, allowed, 100.0) == {"0": 50.0}
    assert replay.calls == replay.wait_seconds == 0 and replay.replay_calls == 1
    replay.close()


@pytest.mark.parametrize("changed", ["prompt", "count", "layout", "group_order"])
def test_tile_only_replay_checks_prompt_count_and_layout(monkeypatch, tmp_path, mock_sdk, changed):
    child = mock_sdk(choice=None)
    monkeypatch.setattr(krea2_jobs, "cloud_source", lambda: ENV)
    monkeypatch.setattr(krea2_jobs, "cancelled", lambda: False)
    original = common.create_client
    monkeypatch.setattr(krea2_jobs, "create_client", lambda **kw: original(sys.executable, **kw))
    options = krea2.Options(tile_mode="jev", job_max_calls=1)
    session = krea2_jobs.Session(options, tmp_path / "live", prompt="tile source prompt alpha")
    layout = {"width": 64, "height": 64, "tiles": [[0, 0, 32, 64], [32, 0, 64, 64]]}
    allocator = session.prepare_tiles([0.003, 0.06], 2, 4, 0.035, layout=layout)
    session.close("completed")
    assert "tile source prompt alpha" not in allocator.log.path.read_text(encoding="utf-8")
    replay_options = replace(options, replay_logs=json.dumps([str(allocator.log.path)]))
    replay_session = krea2_jobs.Session(
        replay_options,
        tmp_path / "replay",
        prompt="different prompt" if changed == "prompt" else "tile source prompt alpha",
    )
    scores = (
        [0.003, 0.06, 0.003] if changed == "count" else [0.06, 0.003] if changed == "group_order" else [0.003, 0.06]
    )
    if changed == "layout":
        layout["tiles"][0][2] = 48
    with pytest.raises(common.ReplayError):
        replay_session.prepare_tiles(scores, 2, 4, 0.035, layout=layout)
    assert len(child.children) == 1
    replay_session.close("failed")


@pytest.mark.parametrize("mutation", ["shape", "schedule", "allowed", "answer", "short", "unused", "prompt"])
def test_replay_rejects_mismatch_or_incomplete_consumption(tmp_path, mock_sdk, mutation):
    mock_sdk()
    with common.JevClient(sys.executable, environment=ENV) as live:
        log, state, allowed = recorded(tmp_path, live)
    if mutation == "answer":
        records = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
        records[1]["replay"]["answer"]["decisions"]["0"]["choice"] = 17
        log.path.write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    if mutation == "prompt":
        with pytest.raises(common.ReplayError, match="prompt"):
            common.ReplayClient([log.path], prompt_sha256="wrong")
        return
    replay = common.ReplayClient([log.path])
    if mutation == "shape":
        state["tensor_shapes"]["0"][1] = 65
    elif mutation == "schedule":
        state["evaluation"] = 2
    elif mutation == "allowed":
        allowed["0"] = (25.0, 50.0, 100.0)
    elif mutation == "short":
        replay.decide(state, allowed, 100.0)
    with pytest.raises(common.ReplayError):
        if mutation == "unused":
            replay.close()
        else:
            replay.decide(state, allowed, 100.0)
    assert replay.calls == 0 and replay.failed


def test_old_logs_without_shape_contract_are_not_accepted(tmp_path):
    log = common.RunLog(tmp_path, "anima", asdict(common.AnimaOptions(mode="jev")))
    log.write("decision", source="jev", keep_percent={"0": 50})
    log.finish("completed")
    with pytest.raises(common.ReplayError, match="contract"):
        common.ReplayClient([log.path])


def test_replay_failure_is_not_dense_fallback(tmp_path, mock_sdk):
    mock_sdk()
    with common.JevClient(sys.executable, environment=ENV) as live:
        log, _, _ = recorded(tmp_path, live)
    replay = common.ReplayClient([log.path])
    run = krea2.KreaRun(krea2.Options(mode="jev", min_tokens=64), 1, common.RunLog(tmp_path, "replay", {}), replay)
    run.begin_evaluation()
    run.observations = {"0": {"relative_output_norm": 1, "cross_evaluation_drift": None}}
    with pytest.raises(common.ReplayError):
        run.begin_evaluation()
    assert not run.circuit_open
    run.close("failed")


@pytest.mark.parametrize("calls,wait", [(True, 0), (-1, 0), (1.5, 0), (0, float("nan")), (0, -1), (0, True)])
def test_job_budget_rejects_invalid_limits(calls, wait):
    with pytest.raises(ValueError):
        common.JevBudget(calls, wait)
