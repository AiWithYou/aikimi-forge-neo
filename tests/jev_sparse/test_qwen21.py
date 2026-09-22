"""CPU correctness + mocked integration tests. No real model, CUDA or Jev calls.

The small single-stream fixture mirrors the pinned 2.1 processor/cache contract;
it is intentionally NOT a legacy dual-stream Qwen mock or a quality benchmark.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge.jev_sparse import common
from modules_forge.jev_sparse import qwen21 as q
from modules_forge.jev_sparse import qwen21_integration as integration


class Cache:
    def __init__(self):
        self.k = self.v = None

    def store(self, k, v):
        self.k, self.v = k, v

    def get(self):
        if self.k is None:
            raise RuntimeError("Cache not initialized")
        return self.k, self.v


def prepare(attn, hidden, rope, cache, mode, write_slice):
    # Same single-stream projection/cache ordering as Diffusers' 2.1 helper.
    query, key, value = [fn(hidden).unflatten(-1, (attn.heads, -1)) for fn in (attn.to_q, attn.to_k, attn.to_v)]
    query, key = attn.norm_q(query).to(value.dtype), attn.norm_k(key).to(value.dtype)
    if rope is not None:

        def rotate(x):
            return (
                torch.view_as_real(torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * rope.unsqueeze(1))
                .flatten(3)
                .to(x.dtype)
            )

        query, key = rotate(query), rotate(key)
    if cache is not None:
        if mode == "extract" and write_slice is not None:
            cache.store(key[:, write_slice].clone(), value[:, write_slice].clone())
        elif mode == "cached":
            ck, cv = cache.get()
            key, value = torch.cat((ck, key), dim=1), torch.cat((cv, value), dim=1)
    return query, key, value, query.shape[1]


class DefaultProcessor:
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        query, key, value, n = prepare(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)

        def sdpa(start, end, key_end, mask=None):
            return F.scaled_dot_product_attention(
                query[:, start:end].transpose(1, 2),
                key[:, :key_end].transpose(1, 2),
                value[:, :key_end].transpose(1, 2),
                attn_mask=mask,
            ).transpose(1, 2)

        if segments is None:
            out = sdpa(0, n, key.shape[1], attention_mask)
        else:
            outputs = []
            for start, end, text in segments:
                mask = None
                if text:
                    mask = torch.cat(
                        (
                            torch.ones(end - start, start, dtype=torch.bool),
                            torch.ones(end - start, end - start, dtype=torch.bool).tril(),
                        ),
                        1,
                    )[None, None]
                if key_valid is not None:
                    valid = key_valid[:, None, None, :end]
                    mask = valid if mask is None else mask & valid
                outputs.append(sdpa(start, end, end, mask))
            prefix = segments[-1][1] if segments else 0
            outputs.append(sdpa(prefix, n, key.shape[1], None if key_valid is None else key_valid[:, None, None, :]))
            out = torch.cat(outputs, 1)
        return attn.to_out[1](attn.to_out[0](out.flatten(2)))


UPSTREAM = types.SimpleNamespace(QwenImage21AttnProcessor=DefaultProcessor, _qwenimage21_prepare_qkv=prepare)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.to_q = torch.nn.Linear(16, 16, bias=False)
        self.to_k = torch.nn.Linear(16, 16, bias=False)
        self.to_v = torch.nn.Linear(16, 16, bias=False)
        self.norm_q = torch.nn.RMSNorm(8)
        self.norm_k = torch.nn.RMSNorm(8)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(16, 16, bias=False), torch.nn.Identity()])
        self.processor = DefaultProcessor()

    def set_processor(self, p):
        self.processor = p

    def forward(self, x, **kw):
        return self.processor(self, x, **kw)


class QwenImage21Transformer2DModel(torch.nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": Attention()}) for _ in range(layers)]
        )
        # ModuleDict supports .attn through nn.Module's attribute lookup.
        self.caches = [Cache() for _ in range(layers)]

    def forward(self, x, mode, mask=None, rope=None):
        for i, block in enumerate(self.transformer_blocks):
            x = block.attn(
                x,
                attention_mask=mask if mode == "cached" else None,
                rotary_emb=rope,
                layer_cache=self.caches[i],
                kv_cache_mode=mode,
                cache_write_slice=slice(0, 13) if mode == "extract" else None,
                segments=[(0, 5, True), (5, 13, False)] if mode == "extract" else None,
                key_valid=mask[:, 0, 0] if mask is not None and mode == "extract" else None,
            )
        return x


class FakeClient:
    def __init__(self, fail=False):
        self.calls, self.wait_seconds, self.fail = 0, 0.0, fail
        self.states = []

    def decide(self, state, allowed, fallback):
        self.calls += 1
        self.states.append(copy.deepcopy(state))
        if self.fail:
            raise common.JevError("fake failure")
        return {i: 50.0 for i in allowed}


@pytest.mark.parametrize(
    "name,value",
    [
        ("mode", []),
        ("mode", "legacy"),
        ("mode", None),
        ("keep_percent", True),
        ("keep_percent", float("nan")),
        ("keep_percent", 0),
        ("keep_percent", 101),
        ("min_tokens", 32),
        ("min_tokens", 64.0),
        ("warmup_evaluations", 0),
        ("update_interval", 0),
        ("max_calls", True),
        ("max_calls", 9),
        ("timeout", float("inf")),
        ("block_size", True),
        ("block_size", 129),
    ],
)
def test_invalid_options(name, value):
    with pytest.raises(ValueError):
        replace(q.Options(), **{name: value}).validate()


@pytest.mark.parametrize("mode", ["off", "dense", "fixed", "rules", "jev"])
def test_options_roundtrip(mode):
    value = q.Options(mode=mode)
    assert q.Options.parse(json.dumps(asdict(value))) == value


def test_options_reject_secrets_and_unknown_fields():
    with pytest.raises(ValueError):
        q.Options.parse({"api_key": "must-not-be-a-setting"})
    with pytest.raises(ValueError):
        q.Options.parse(" " * 4097)


@pytest.mark.parametrize(
    "prefix,length,keep", [(0, 135, 50), (17, 135, 50), (137, 130, 75), (17, 192, 1), (3, 128, 100)]
)
def test_gather_matches_independent_dense_mask_reference(prefix, length, keep):
    torch.manual_seed(17)
    query = torch.randn(2, length, 2, 8)
    key, value = [torch.randn(2, prefix + length, 2, 8) for _ in range(2)]
    valid = torch.ones(2, 1, 1, prefix + length, dtype=torch.bool)
    if prefix:
        valid[0, :, :, :prefix:3] = False
    actual = q.block_gather_attention(query, key, value, keep, prefix, valid, 64)
    allowed = valid.expand(2, 2, length, -1).clone()
    if keep < 100:
        allowed[:, :, :, prefix:] = False
        for b in range(2):
            for h in range(2):
                means = torch.stack([key[b, prefix + s : prefix + s + 64, h].mean(0) for s in range(0, length, 64)])
                for start in range(0, length, 64):
                    scores = means @ query[b, start : start + 64, h].mean(0)
                    chosen = scores.topk(max(1, __import__("math").ceil(len(means) * keep / 100))).indices
                    for idx in chosen.tolist():
                        allowed[b, h, start : start + 64, prefix + idx * 64 : prefix + min(length, (idx + 1) * 64)] = (
                            True
                        )
    reference = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=allowed
    ).transpose(1, 2)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


def test_sparse_uses_shorter_key_sequences(monkeypatch):
    original = F.scaled_dot_product_attention
    sizes = []

    def record(query, key, value, **kw):
        sizes.append(key.shape[-2])
        return original(query, key, value, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    query = torch.randn(1, 256, 2, 8)
    key = torch.randn(1, 269, 2, 8)
    q.block_gather_attention(query, key, key, 50, 13, block_size=64)
    assert sizes == [13 + 128] * 4  # actual reduced matmul, not a masked 269-key dense call


def test_exact_prefix_masked_values_never_leak():
    query = torch.randn(1, 192, 2, 8)
    key, value = [torch.randn(1, 209, 2, 8) for _ in range(2)]
    mask = torch.ones(1, 1, 1, 209, dtype=torch.bool)
    mask[..., :17:2] = False
    a = q.block_gather_attention(query, key, value, 50, 17, mask, 64)
    changed = value.clone()
    changed[:, :17:2] = 1e8
    b = q.block_gather_attention(query, key, changed, 50, 17, mask, 64)
    torch.testing.assert_close(a, b)
    changed[:, 1:17:2] += 1
    assert not torch.equal(a, q.block_gather_attention(query, key, changed, 50, 17, mask, 64))


def test_unknown_mask_rejected_by_kernel():
    t = torch.randn(1, 128, 2, 8)
    with pytest.raises(ValueError):
        q.block_gather_attention(t, t, t, 50, 0, torch.zeros(128, 128), 64)


def call_prefill(model):
    data = torch.randn(1, 205, 16)
    rope = torch.polar(torch.ones(205, 4), torch.randn(205, 4))
    valid = torch.ones(1, 1, 1, 205, dtype=torch.bool)
    valid[..., 1] = False
    return model(data, "extract", valid, rope), data, rope, valid


def test_prefill_exact_cache_and_dense_off_identity(tmp_path):
    torch.manual_seed(51)
    a = QwenImage21Transformer2DModel()
    b = copy.deepcopy(a)
    data = torch.randn(1, 205, 16)
    valid = torch.ones(1, 1, 1, 205, dtype=torch.bool)
    valid[..., 1] = False
    out = a(data, "extract", valid)
    original_processors = [x.attn.processor for x in b.transformer_blocks]
    before = copy.deepcopy(b.state_dict())
    with q.experiment(
        b, q.Options(mode="fixed", keep_percent=50, min_tokens=64, block_size=64), tmp_path, upstream=UPSTREAM
    ) as run:
        actual = b(data, "extract", valid)
        assert torch.equal(out, actual)
        for ca, cb in zip(a.caches, b.caches, strict=True):
            assert torch.equal(ca.k, cb.k) and torch.equal(ca.v, cb.v)
        old = [(x.k.clone(), x.v.clone()) for x in b.caches]
        b(data[:, 13:], "cached", valid)
        assert run.total["sparse_target"] == 3
        for cache, (ck, cv) in zip(b.caches, old, strict=True):
            assert torch.equal(cache.k, ck) and torch.equal(cache.v, cv)
    assert run.summary["api_calls"] == 0
    assert [x.attn.processor for x in b.transformer_blocks] == original_processors
    assert not b._forward_hooks and not b._forward_pre_hooks
    for name, value in before.items():
        assert torch.equal(value, b.state_dict()[name])


def test_dense_only_delegates_and_records(tmp_path):
    model = QwenImage21Transformer2DModel()
    with q.experiment(model, q.Options(mode="dense", min_tokens=64, block_size=64), tmp_path, upstream=UPSTREAM) as run:
        out, data, rope, mask = call_prefill(model)
        model(data[:, 13:], "cached", mask, rope[13:])
    assert run.summary["attention_calls"] == {"dense": 6}
    assert run.summary["model_evaluations"] == 2
    assert run.summary["model_seconds"] > 0


def test_real_observations_drive_jev_from_dense_prefill(tmp_path):
    model = QwenImage21Transformer2DModel()
    client = FakeClient()
    with q.experiment(
        model,
        q.Options(mode="jev", min_tokens=64, block_size=64, max_calls=2, update_interval=2),
        tmp_path,
        upstream=UPSTREAM,
        client=client,
    ) as run:
        _, data, rope, mask = call_prefill(model)
        assert len(run.observations) == 3
        for _ in range(5):
            model(data[:, 13:], "cached", mask, rope[13:])
    assert client.calls == 2
    assert [s["evaluation"] for s in client.states] == [1, 3]
    assert run.total["sparse_target"] == 15
    serialized = json.dumps(client.states)
    assert "prompt" not in client.states[0] and "image" not in client.states[0]
    assert "raw_tensor" not in serialized


@pytest.mark.parametrize("cadence,expected", [("once", 1), ("interval", 6), ("step", 11)])
def test_qwen_cadence_continues_after_first_decision(tmp_path, cadence, expected):
    model = QwenImage21Transformer2DModel()
    client = FakeClient()
    with q.experiment(
        model,
        q.Options(mode="jev", min_tokens=64, block_size=64, decision_cadence=cadence, update_interval=2),
        tmp_path,
        upstream=UPSTREAM,
        client=client,
    ):
        _, data, rope, mask = call_prefill(model)
        for _ in range(11):
            model(data[:, 13:], "cached", mask, rope[13:])
    assert client.calls == expected


def test_rules_offline_and_client_failure_dense(tmp_path):
    for mode, client in [("rules", None), ("jev", FakeClient(fail=True))]:
        model = QwenImage21Transformer2DModel()
        with q.experiment(
            model, q.Options(mode=mode, min_tokens=64, block_size=64), tmp_path, upstream=UPSTREAM, client=client
        ) as run:
            _, data, _, mask = call_prefill(model)
            for _ in range(4):
                model(data[:, 13:], "cached", mask)
        if mode == "jev":
            assert client.calls == 1 and run.circuit_open and run.total["sparse_target"] == 0
        else:
            assert run.total["sparse_target"] > 0


def test_cleanup_on_exception_and_cancellation(tmp_path):
    for cancel in (False, True):
        model = QwenImage21Transformer2DModel()
        originals = [x.attn.processor for x in model.transformer_blocks]
        flag = [False]
        with pytest.raises((RuntimeError, common.ExperimentCancelled)):
            with q.experiment(
                model, q.Options(mode="dense"), tmp_path, upstream=UPSTREAM, cancelled=lambda flag=flag: flag[0]
            ) as run:
                call_prefill(model)
                if cancel:
                    flag[0] = True
                    model(torch.ones(1, 192, 16), "cached")
                raise RuntimeError("test interruption")
        assert [x.attn.processor for x in model.transformer_blocks] == originals
        assert not model._forward_hooks and not model._forward_pre_hooks
        assert run.summary["status"] == ("cancelled" if cancel else "failed")
        assert not run.previous and not run.pending


def test_off_does_not_import_or_mutate(tmp_path):
    with q.experiment(object(), q.Options(), tmp_path / "absent") as run:
        assert run is None
    assert not (tmp_path / "absent").exists()


def test_reject_legacy_and_custom_processors(tmp_path):
    with pytest.raises(ValueError, match="2.1 ONLY"):
        with q.experiment(object(), q.Options(mode="dense"), tmp_path, upstream=UPSTREAM):
            pass
    model = QwenImage21Transformer2DModel()
    model.transformer_blocks[0].attn.processor = object()
    with pytest.raises(ValueError, match="unmodified"):
        with q.experiment(model, q.Options(mode="dense"), tmp_path, upstream=UPSTREAM):
            pass


def test_nonfinite_statistics_break_circuit(tmp_path):
    log = common.RunLog(tmp_path, "qwen21", {})
    run = q.Run(q.Options(mode="rules"), 1, log)
    run.begin()
    run.pending[0] = torch.tensor([float("nan"), 0.0])
    run.end()
    assert run.circuit_open and run.keep(0) == 100
    run.finish("completed")


def fake_service(monkeypatch, tmp_path):
    worker = tmp_path / "tools/qwen_image21_worker.py"
    worker.parent.mkdir()
    worker.write_text("# reviewed worker\n")
    source = worker.read_bytes().replace(b"\r\n", b"\n")
    monkeypatch.setattr(integration, "ROOT", tmp_path)
    monkeypatch.setattr(
        integration,
        "WORKER_BLOB",
        hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source, usedforsecurity=False).hexdigest(),
    )
    module = types.ModuleType("modules_forge.qwen_image21.service")
    module.WORKER = worker
    module.QwenImage21Error = type("QwenImage21Error", (RuntimeError,), {})

    class Studio:
        def start(self, request, owner):
            return "original-job"

    module.Studio = Studio
    module.safe_environment = lambda: {"PATH": "ok", "TYPESAFE_API_KEY": "unwanted"}
    package = types.ModuleType("modules_forge.qwen_image21")
    package.service = module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_actual_qwen_service_routing_and_uninstall(monkeypatch, tmp_path):
    service = fake_service(monkeypatch, tmp_path)
    original_worker = service.WORKER
    request = types.SimpleNamespace(sparse_mode="fixed", sparse_keep_percent=75)
    worker, env, payload = integration.worker_launch(request, original_worker, service.safe_environment())
    assert worker.name == "qwen_image21_sparse_worker.py"
    assert "TYPESAFE_API_KEY" not in env
    assert payload["sparse_experiment"]["mode"] == "fixed"
    assert service.WORKER == original_worker
    request.sparse_mode = "off"
    worker, env, payload = integration.worker_launch(request, original_worker, service.safe_environment())
    assert worker == original_worker and not payload and "TYPESAFE_API_KEY" not in env


def test_incompatible_source_blocks_instead_of_fake_benchmark(monkeypatch, tmp_path):
    service = fake_service(monkeypatch, tmp_path)
    service.WORKER.write_text("# changed worker\n")
    request = types.SimpleNamespace(sparse_mode="fixed", sparse_keep_percent=75)
    with pytest.raises(ValueError, match="reviewed revision"):
        integration.worker_launch(request, service.WORKER, {})


def test_no_launch_option_means_no_hooks(monkeypatch):
    monkeypatch.delenv(q.OPTIONS_ENV, raising=False)
    assert integration.launch_defaults().mode == "off"


def test_windows_crlf_worker_is_accepted(monkeypatch, tmp_path):
    service = fake_service(monkeypatch, tmp_path)
    service.WORKER.write_bytes(b"# reviewed worker\r\n")
    request = types.SimpleNamespace(sparse_mode="dense", sparse_keep_percent=75)
    worker, _, _ = integration.worker_launch(request, service.WORKER, {})
    assert worker.name == "qwen_image21_sparse_worker.py"


def test_cloud_key_scoped_to_selected_qwen_worker(monkeypatch, tmp_path):
    service = fake_service(monkeypatch, tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(integration, "read_saved_key", lambda: "fake-secret")
    monkeypatch.setattr(integration, "sdk_python", lambda: Path(sys.executable))
    request = types.SimpleNamespace(
        sparse_mode="jev", sparse_keep_percent=75, sparse_jev_cadence="interval", sparse_jev_interval=2
    )
    _, env, payload = integration.worker_launch(request, service.WORKER, service.safe_environment())
    assert env["TYPESAFE_API_KEY"] == "fake-secret" and env["AIKIMI_JEV_ALLOW_CLOUD"] == "1"
    assert "fake-secret" not in json.dumps(payload)
    assert payload["sparse_experiment"]["decision_cadence"] == "interval"
    assert payload["sparse_experiment"]["update_interval"] == 2


def load_worker(monkeypatch, tmp_path):
    base = types.ModuleType("_aikimi_qwen21_base_worker")
    base.DIFFUSERS_REVISION = q.REVISION
    base.GenerationCancelled = type("GenerationCancelled", (RuntimeError,), {})
    base.cleared = 0

    def clear():
        base.cleared += 1

    base.clear_runtime = clear

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf8")

    base._atomic_json = write

    def read(payload):
        job = Path(payload["job_dir"])
        return job, tmp_path / "model", json.loads((job / "request.json").read_text())

    base._read_request = read

    class Pipe:
        def __init__(self):
            self.transformer = QwenImage21Transformer2DModel()
            self.calls = 0
            self.modes = []

        def __call__(self, **kwargs):
            self.calls += 1
            self.modes.append(type(self.transformer.transformer_blocks[0].attn.processor).__name__)
            data = torch.ones(1, 269, 16)
            self.transformer(data, "extract")
            return self.transformer(data[:, 13:], "cached")

    pipe = Pipe()
    runtime = {"pipe": pipe}
    base._runtime_for_request = lambda *a: (runtime, pipe.calls > 0)

    def run(payload):
        job, model, req = base._read_request(payload)
        r, reused = base._runtime_for_request(model, req, job)
        r["pipe"](use_kv_cache=True, true_cfg_scale=1.0)
        result = {"metadata": {"reused_model": reused}, "output_path": str(job / "output.png")}
        write(job / "result.json", result)
        return result

    base.run_request = run
    monkeypatch.setitem(sys.modules, base.__name__, base)
    spec = importlib.util.spec_from_file_location(
        "test_qwen21_sparse_worker", ROOT / "tools/qwen_image21_sparse_worker.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from functools import partial

    monkeypatch.setattr(mod, "experiment", partial(q.experiment, upstream=UPSTREAM))
    return mod, base, pipe


def make_job(tmp_path, name):
    job = tmp_path / name
    job.mkdir()
    (job / "request.json").write_text(
        json.dumps(
            {
                "prompt": "private test prompt",
                "seed": 7,
                "width": 1024,
                "height": 1024,
                "steps": 2,
                "precision": "int8",
                "memory_mode": "offload",
                "input_images": [],
            }
        ),
        encoding="utf8",
    )
    return job


def test_worker_metadata_routing_and_two_job_model_reuse(monkeypatch, tmp_path):
    mod, base, pipe = load_worker(monkeypatch, tmp_path)
    original_lookup, original_read = base._runtime_for_request, base._read_request
    monkeypatch.setenv(
        q.OPTIONS_ENV, json.dumps({"mode": "fixed", "keep_percent": 50, "block_size": 64, "min_tokens": 64})
    )
    results = []
    for name in ("first", "second"):
        job = make_job(tmp_path, name)
        result = mod.resident_run({"job_dir": str(job)})
        results.append(result)
        report = result["metadata"]["sparse_experiment"]
        assert report["attention_calls"]["sparse_target"] == 3
        assert report["api_calls"] == 0 and report["status"] == "completed"
        saved = json.loads((job / "result.json").read_text())
        assert saved == result
        log = Path(report["log_path"]).read_text()
        assert "private test prompt" not in log
        assert json.loads((job / "request.json").read_text())["sparse_experiment"]["mode"] == "fixed"
        assert base._runtime_for_request is original_lookup and base._read_request is original_read
        assert all(type(b.attn.processor) is DefaultProcessor for b in pipe.transformer.transformer_blocks)
    assert pipe.modes == ["Processor", "Processor"]
    assert results[0]["metadata"]["reused_model"] is False and results[1]["metadata"]["reused_model"] is True
    assert (
        results[0]["metadata"]["sparse_experiment"]["log_path"]
        != results[1]["metadata"]["sparse_experiment"]["log_path"]
    )


def test_worker_off_and_option_mismatch(monkeypatch, tmp_path):
    mod, base, pipe = load_worker(monkeypatch, tmp_path)
    monkeypatch.delenv(q.OPTIONS_ENV, raising=False)
    job = make_job(tmp_path, "off")
    result = mod.resident_run({"job_dir": str(job)})
    assert result["metadata"]["sparse_experiment"]["status"] == "off"
    assert pipe.modes == ["DefaultProcessor"]
    monkeypatch.setenv(q.OPTIONS_ENV, '{"mode":"fixed"}')
    with pytest.raises(ValueError, match="differ"):
        mod.resident_run({"job_dir": str(job)})
    assert not (job / "result.json").exists() and base.cleared == 1


def test_worker_post_save_error_retracts_success(monkeypatch, tmp_path):
    mod, base, pipe = load_worker(monkeypatch, tmp_path)
    monkeypatch.setenv(q.OPTIONS_ENV, '{"mode":"dense"}')
    job = make_job(tmp_path, "failure")
    original = base._atomic_json

    def fail(path, value):
        if path.name == "metadata.json":
            raise OSError("test disk full")
        original(path, value)

    monkeypatch.setattr(base, "_atomic_json", fail)
    with pytest.raises(OSError, match="disk full"):
        mod.resident_run({"job_dir": str(job)})
    assert not (job / "result.json").exists() and base.cleared == 1
    assert not pipe.transformer._forward_hooks
