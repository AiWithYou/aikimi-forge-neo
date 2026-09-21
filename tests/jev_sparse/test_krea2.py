"""Krea2 routing boundaries and whole-image budgets, without provider/GPU calls."""

from __future__ import annotations

import copy
import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from modules_forge.jev_sparse import common, krea2, krea2_jobs


class Client:
    def __init__(self, value=3, fail=False):
        self.value, self.fail = value, fail
        self.calls, self.wait_seconds = 0, 0.0
        self.last_diagnostics = {}
        self.requests = []

    def decide(self, state, allowed, fallback):
        self.calls += 1
        self.requests.append((copy.deepcopy(state), copy.deepcopy(allowed)))
        if self.fail:
            raise common.JevError("fake provider failure")
        return {key: float(self.value) if self.value in values else min(values) for key, values in allowed.items()}


def make_run(tmp_path, mode="jev", **kwargs):
    options = krea2.Options(mode=mode, min_tokens=64, **kwargs)
    client = Client()
    log = common.RunLog(tmp_path, "krea2-test", {})
    return krea2.KreaRun(options, 2, log, client), client


def observe_all(run):
    for layer in range(run.layers):
        run.observe(layer, torch.ones(1, 64, 256) * (layer + 1), torch.ones(1, 64, 2, 128))


def test_one_layer_decision_is_reused_for_many_upscale_evaluations(tmp_path):
    run, client = make_run(tmp_path)
    for i in range(24):
        run.begin_evaluation()
        assert run.keep(0) == (100 if i == 0 else 3)
        observe_all(run)
        run.end_evaluation()
    assert client.calls == 1
    assert client.requests[0][1] == {"0": krea2.KEEP_CHOICES, "1": krea2.KEEP_CHOICES}
    run.close()
    records = [json.loads(line) for line in run.log.path.read_text().splitlines()]
    assert records[-1]["api_calls"] == 1 and records[-1]["model_evaluations"] == 24


def test_layer_failure_falls_back_once_and_does_not_retry(tmp_path):
    run, client = make_run(tmp_path)
    client.fail = True
    for _ in range(4):
        run.begin_evaluation()
        observe_all(run)
        run.end_evaluation()
    assert client.calls == 1 and run.circuit_open
    assert run.keeps == {"0": 100, "1": 100}


def test_short_image_sequence_does_not_trigger_jev(tmp_path):
    run, client = make_run(tmp_path)
    for _ in range(4):
        run.begin_evaluation()
        for layer in range(2):
            run.observe(layer, torch.ones(1, 32, 256), torch.ones(1, 32, 2, 128))
        run.end_evaluation()
    assert client.calls == 0


def test_conditioning_prefix_and_original_dense_path(tmp_path):
    run, _ = make_run(tmp_path, mode="fixed", keep_percent=5, warmup_evaluations=0)
    run.begin_evaluation()
    calls = []
    q = torch.randn(1, 2, 192, 128)
    expected = torch.randn(1, 192, 256)

    def kernel(q, k, v, keep, protected):
        calls.append((keep, protected, tuple(q.shape)))
        return expected

    def dense(q, k, v, heads, **kwargs):
        return torch.zeros(1, q.shape[2], heads * 128)

    options = {"krea2_block_index": 1, "krea2_text_tokens": 32, "krea2_reference_tokens": 64, "krea2_image_tokens": 96}
    override = krea2.attention_override(run, kernel)
    assert torch.count_nonzero(override(q, q, q, 2, None, options, dense)) == 0
    token = krea2._ACTIVE.set(run)
    try:
        assert override(q, q, q, 2, None, options, dense) is expected
        assert calls == [(5, 96, (1, 2, 192, 128))]
        assert run.counts["sparse"] == 1
        assert torch.count_nonzero(override(q, q, q, 2, torch.ones(192), options, dense)) == 0
        assert run.counts["dense_mask"] == 1
        with pytest.raises(ValueError, match="layout"):
            override(q, q, q, 2, None, {**options, "krea2_image_tokens": 97}, dense)
        run.close()
        assert torch.count_nonzero(override(q, q, q, 2, None, options, dense)) == 0
    finally:
        krea2._ACTIVE.reset(token)


@pytest.mark.parametrize("tokens", [64, 67, 132])
def test_head_groups_preserve_order_and_padding_masks(tokens):
    q = torch.arange(2 * 48 * tokens * 128, dtype=torch.float32).remainder(97).reshape(2, 48, tokens, 128).bfloat16()
    calls = []

    class Kernel:
        @staticmethod
        def sol_attn(q, k, v, **kwargs):
            calls.append((tuple(q.shape), kwargs))
            assert q.is_contiguous() and q.shape == k.shape == v.shape
            if tokens % 64:
                assert torch.count_nonzero(q[:, tokens:]) == 0
            return q

    output = krea2.blocked_attention(q, q, q, 3, 1, Kernel)
    assert torch.equal(output, q.transpose(1, 2).reshape(2, tokens, -1))
    assert len(calls) == 3 and all(shape[2] == 16 for shape, _ in calls)
    for _, opts in calls:
        assert opts["sink_blocks"] == opts["sink_q"] == [0, 1]
        assert opts["topk_ratio"] == 0.03
        if tokens % 64:
            assert opts["block_len"].tolist() == [64] * (tokens // 64) + [tokens % 64]
        else:
            assert opts["block_len"] is None


def test_tile_steps_have_step_units_and_one_call_across_stages(tmp_path):
    client = Client(value=0)
    planner = krea2_jobs.TileAllocator("jev", tmp_path, client=client, is_cancelled=lambda: False)
    for _ in range(3):
        planner.prepare([0, 0.002, 0.04, 0.2], 2, 4, 0.035)
        assert planner.steps(0, 2, 4, 0.035) == 0
    assert client.calls == 1
    state, choices = client.requests[0]
    assert state["decision_kind"] == "tile_steps"
    assert all(value == (0, 2, 4) for value in choices.values())
    assert "prompt" not in state and "image" not in state
    planner.close("completed")
    report = json.loads(planner.log.path.read_text().splitlines()[-1])
    assert report["selected_step_counts"] == {"0": 3}


def test_tile_failure_uses_bounded_rules_without_retry(tmp_path):
    client = Client(fail=True)
    planner = krea2_jobs.TileAllocator("jev", tmp_path, client=client, is_cancelled=lambda: False)
    planner.prepare([0, 0.003, 0.2], 2, 4, 0.035)
    planner.prepare([0.9], 2, 4, 0.035)
    assert client.calls == 1 and planner.source == "rules_fallback"
    assert planner.steps(0, 2, 4, 0.035) == 0
    assert planner.steps(0.9, 2, 4, 0.035) == 4
    assert 2 <= planner.steps(0.04, 2, 4, 0.035) <= 4


def test_tile_rules_never_construct_sdk(tmp_path, monkeypatch):
    monkeypatch.setattr(krea2_jobs, "JevClient", lambda *a, **kw: pytest.fail("Unexpected cloud call"))
    planner = krea2_jobs.TileAllocator("rules", tmp_path, is_cancelled=lambda: False)
    planner.prepare([0, 0.1], 2, 4, 0.035)
    assert planner.steps(0, 2, 4, 0.035) == 0
    assert planner.steps(0.1, 2, 4, 0.035) == 2


def test_nested_job_restores_context_on_exception(tmp_path, monkeypatch):
    options = krea2.Options(mode="jev")
    monkeypatch.setattr(krea2_jobs, "processing_options", lambda p: options)
    monkeypatch.setattr(krea2_jobs, "krea_selected", lambda p: True)
    monkeypatch.setattr(krea2_jobs, "cancelled", lambda: False)
    captured = []
    original = krea2_jobs.Session

    def new_session(options):
        session = original(options, tmp_path)
        captured.append(session)
        return session

    monkeypatch.setattr(krea2_jobs, "Session", new_session)
    with pytest.raises(RuntimeError):
        with krea2_jobs.generation_scope(object()) as outer:
            with krea2_jobs.generation_scope(object()) as nested:
                assert outer is nested and krea2_jobs.current_session() is outer
                raise RuntimeError("job failed")
    assert len(captured) == 1 and captured[0].closed
    assert krea2_jobs.current_session() is None


def test_off_scope_is_noop_and_does_not_access_credentials(monkeypatch):
    monkeypatch.setattr(krea2_jobs, "Session", lambda *a, **kw: pytest.fail("Unexpected session"))
    with krea2_jobs.generation_scope(SimpleNamespace()) as session:
        assert session is None


def test_other_model_scope_is_noop_even_with_krea_controls_on(monkeypatch):
    monkeypatch.setattr(krea2_jobs, "processing_options", lambda p: krea2.Options(mode="jev", tile_mode="jev"))
    monkeypatch.setattr(krea2_jobs, "krea_selected", lambda p: False)
    monkeypatch.setattr(krea2_jobs, "Session", lambda *a, **kw: pytest.fail("Unexpected Krea2 session"))
    with krea2_jobs.generation_scope(SimpleNamespace()) as session:
        assert session is None


def test_log_close_failure_still_clears_job_context(monkeypatch):
    monkeypatch.setattr(krea2_jobs, "processing_options", lambda p: krea2.Options(mode="jev"))
    monkeypatch.setattr(krea2_jobs, "krea_selected", lambda p: True)
    monkeypatch.setattr(krea2_jobs, "cancelled", lambda: False)

    def close(_status):
        raise OSError("log write failed")

    monkeypatch.setattr(krea2_jobs, "Session", lambda *a: SimpleNamespace(close=close))
    with pytest.raises(OSError):
        with krea2_jobs.generation_scope(SimpleNamespace()):
            assert krea2_jobs.current_session() is not None
    assert krea2_jobs.current_session() is None


@pytest.mark.parametrize("marker,expected", [("txtfusion.projector.weight", True), ("other.weight", False)])
def test_checkpoint_header_guard(tmp_path, marker, expected):
    path = tmp_path / "renamed-model.safetensors"
    header = json.dumps({marker: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
    stat = path.stat()
    assert krea2_jobs._checkpoint_is_krea(str(path), stat.st_mtime_ns, stat.st_size) is expected


@pytest.mark.parametrize("changes", [{"max_calls": 2}, {"tile_mode": "unknown"}, {"min_tokens": 32}])
def test_invalid_krea2_options(changes):
    with pytest.raises(ValueError):
        replace(krea2.Options(), **changes).validate()
