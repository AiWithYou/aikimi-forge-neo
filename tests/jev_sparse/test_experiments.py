"""Offline CPU tests. ComfyUI/Forge/provider boundaries are mocked explicitly."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge.jev_sparse import anima, common, h3_node
from modules_forge.jev_sparse import h3_integration as integration


class MemoryLog:
    def __init__(self):
        self.events = []
        self.closed = False

    def write(self, event, **data):
        self.events.append({"event": event, **data})

    def finish(self, status, **data):
        self.closed = True
        self.write("end", status=status, **data)


class FakeClient:
    def __init__(self, fail_at=None, keep=None):
        self.calls, self.wait_seconds = 0, 0.0
        self.requests = []
        self.fail_at, self.keep = fail_at, keep

    def decide(self, state, allowed, fallback):
        self.calls += 1
        self.requests.append((copy.deepcopy(state), allowed, fallback))
        if self.calls == self.fail_at:
            raise common.JevError("fake failure")
        return {key: self.keep if self.keep in choices else max(choices) for key, choices in allowed.items()}


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "bogus"),
        ("keep_percent", True),
        ("keep_percent", float("nan")),
        ("keep_percent", 0),
        ("keep_percent", 101),
        ("min_tokens", 63),
        ("min_tokens", 4096.5),
        ("warmup_evaluations", -1),
        ("max_calls", True),
        ("max_calls", 9),
        ("update_interval", 0),
        ("timeout", float("inf")),
        ("timeout", 0.1),
    ],
)
def test_invalid_anima_options(field, value):
    with pytest.raises(ValueError):
        replace(common.AnimaOptions(), **{field: value}).validate()


@pytest.mark.parametrize("mode", ["off", "dense", "fixed", "rules", "jev"])
def test_valid_anima_options(mode):
    common.AnimaOptions(mode=mode).validate()


def test_decisions_confidence_and_ids():
    assert common.parse_decisions(
        {"decisions": {"0": {"choice": "50", "confidence": 0.2}, "1": {"choice": 75, "confidence": 0.9}}},
        {"0": (50, 75, 100), "1": (50, 75, 100)},
        100,
    ) == {"0": 100, "1": 75}


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        {"decisions": {}},
        {"decisions": {0: {"choice": 50, "confidence": 0.9}}},
        {"decisions": {"0": {"choice": True, "confidence": 0.9}}},
        {"decisions": {"0": {"choice": 50, "confidence": True}}},
        {"decisions": {"0": {"choice": 1, "confidence": 0.9}}},
        {"decisions": {"0": {"choice": 50, "confidence": float("nan")}}},
        {"decisions": {"0": {"choice": 50, "confidence": 1.1}}},
    ],
)
def test_decisions_reject_bad_provider_values(answer):
    with pytest.raises(common.JevError):
        common.parse_decisions(answer, {"0": (50, 75, 100)}, 100)


def test_env_explicit_permission_and_least_privilege():
    with pytest.raises(common.JevError):
        common.cloud_environment({"TYPESAFE_API_KEY": "fake-key"})
    with pytest.raises(common.JevError):
        common.cloud_environment({"AIKIMI_JEV_ALLOW_CLOUD": "1"})
    result = common.cloud_environment(
        {
            "AIKIMI_JEV_ALLOW_CLOUD": "1",
            "TYPESAFE_API_KEY": "fake-key",
            "GITHUB_TOKEN": "must-not-leak",
            "HTTP_PROXY": "bad",
            "PYTHONPATH": "bad",
            "PATH": "okay",
        }
    )
    assert result["TYPESAFE_API_KEY"] == "fake-key"
    assert result["PATH"] == "okay"
    assert not {"GITHUB_TOKEN", "HTTP_PROXY", "PYTHONPATH"} & result.keys()


def test_json_log_does_not_save_prompt_and_finishes_once(tmp_path):
    log = common.RunLog(tmp_path, "anima", {"mode": "dense"}, "private prompt example")
    log.finish("completed")
    log.finish("failed")
    text = log.path.read_text()
    assert "private prompt example" not in text
    assert [json.loads(line)["event"] for line in text.splitlines()] == ["begin", "end"]
    with pytest.raises(ValueError):
        common.dumps({"not_finite": float("nan")})


def test_sdk_process_success_and_redaction(monkeypatch):
    monkeypatch.setenv("AIKIMI_JEV_ALLOW_CLOUD", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    calls = []

    class Process:
        returncode = 0

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))

        def communicate(self, input=None, timeout=None):
            return '{"decisions":{"0":{"choice":"50.0","confidence":0.9}}}', ""

        def poll(self):
            return 0

    monkeypatch.setattr(common.subprocess, "Popen", Process)
    client = common.JevClient(Path(sys.executable), 1)
    assert client.decide({}, {"0": (50, 75, 100)}, 100) == {"0": 50}
    assert client.calls == 1
    assert "fake-key" not in str(calls[0][0])
    assert calls[0][1]["env"]["TYPESAFE_API_KEY"] == "fake-key"
    assert calls[0][0][1:3] == ["-I", "-B"]


def test_sdk_cancellation_reaps_child(monkeypatch):
    monkeypatch.setenv("AIKIMI_JEV_ALLOW_CLOUD", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    child = types.SimpleNamespace(killed=False, communicates=0)

    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, input=None, timeout=None):
            child.communicates += 1
            if not child.killed:
                raise subprocess.TimeoutExpired("sdk", timeout)
            return "", ""

        def poll(self):
            return 1 if child.killed else None

        def kill(self):
            child.killed = True

    monkeypatch.setattr(common.subprocess, "Popen", Process)
    client = common.JevClient(Path(sys.executable), 1, cancelled=lambda: child.communicates > 0)
    with pytest.raises(common.ExperimentCancelled):
        client.decide({}, {"0": (50, 100)}, 100)
    assert child.killed and child.communicates >= 2
    assert client.calls == 1


def test_sdk_timeout_has_bounded_single_attempt(monkeypatch):
    monkeypatch.setenv("AIKIMI_JEV_ALLOW_CLOUD", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    tick = [0]

    def clock():
        tick[0] += 1
        return float(tick[0])

    child = types.SimpleNamespace(killed=False)

    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, input=None, timeout=None):
            if not child.killed:
                raise subprocess.TimeoutExpired("sdk", timeout)
            return "", ""

        def poll(self):
            return 1 if child.killed else None

        def kill(self):
            child.killed = True

    monkeypatch.setattr(common.subprocess, "Popen", Process)
    monkeypatch.setattr(common.time, "perf_counter", clock)
    client = common.JevClient(Path(sys.executable), 0.5)
    with pytest.raises(common.JevError, match="timeout"):
        client.decide({}, {"0": (50, 100)}, 100)
    assert child.killed and client.calls == 1


def observations(layers):
    return {str(i): {"relative_output_norm": float(i + 1), "cross_evaluation_drift": None} for i in range(layers)}


def test_anima_rules_without_cloud():
    run = anima.AnimaRun(common.AnimaOptions(mode="rules"), 3, MemoryLog())
    run.begin_evaluation()
    assert run.keep(0) == 100
    run.observations = observations(3)
    run.begin_evaluation()
    assert run.keeps == {"0": 50, "1": 75, "2": 100}


def test_anima_jev_budget_and_interval():
    client = FakeClient(keep=50)
    run = anima.AnimaRun(common.AnimaOptions(mode="jev", max_calls=2, update_interval=4), 3, MemoryLog(), client)
    for _ in range(20):
        run.begin_evaluation()
        run.observations = observations(3)
    assert client.calls == 2
    assert [s[0]["next_model_evaluation"] for s in client.requests] == [1, 5]


def test_anima_failure_opens_circuit():
    client = FakeClient(fail_at=1)
    run = anima.AnimaRun(common.AnimaOptions(mode="jev", warmup_evaluations=0), 3, MemoryLog(), client)
    run.observations = observations(3)
    for _ in range(10):
        run.begin_evaluation()
    assert client.calls == 1 and run.circuit_open
    assert set(run.keeps.values()) == {100}


def test_real_cpu_attention_output_and_inactive_identity():
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 64, 2, 128) for _ in range(3)]
    mod = types.SimpleNamespace(output_proj=torch.nn.Linear(256, 256, bias=False), output_dropout=torch.nn.Identity())
    dense_calls, sparse_calls = [], []

    def kernel(q, k, v, keep):
        sparse_calls.append(keep)
        # CPU reference computes selected keys, not the production CUDA kernel.
        n = max(1, int(k.shape[1] * keep / 100))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k[:, :n].transpose(1, 2), v[:, :n].transpose(1, 2)
        ).transpose(1, 2)

    def original(q, k, v, transformer_options):
        dense_calls.append(True)
        return mod.output_proj(v.flatten(2))

    run = anima.AnimaRun(
        common.AnimaOptions(mode="fixed", keep_percent=50, min_tokens=64, warmup_evaluations=0), 1, MemoryLog()
    )
    patch = anima.make_attention_patch(mod, original, 0, run, kernel)
    assert torch.equal(patch(q, k, v), original(q, k, v, {}))
    assert not sparse_calls
    token = anima._ACTIVE.set(run)
    try:
        run.begin_evaluation()
        output = patch(q, k, v)
        assert output.shape == (1, 64, 256) and torch.isfinite(output).all()
        assert sparse_calls == [50]
        run.close()
        assert torch.equal(patch(q, k, v), original(q, k, v, {}))
        assert sparse_calls == [50]
    finally:
        anima._ACTIVE.reset(token)


def test_short_sequence_stays_dense():
    run = anima.AnimaRun(common.AnimaOptions(mode="fixed", min_tokens=128, warmup_evaluations=0), 1, MemoryLog())

    def original(q, k, v, transformer_options):
        return v.flatten(2)

    patch = anima.make_attention_patch(None, original, 0, run, lambda *args: pytest.fail("kernel called"))
    q = torch.randn(1, 64, 1, 128)
    token = anima._ACTIVE.set(run)
    try:
        run.begin_evaluation()
        assert torch.equal(patch(q, q, q), q.flatten(2))
        assert run.counts == {"dense_short_sequence": 1}
    finally:
        anima._ACTIVE.reset(token)


def test_real_sparse_kernel_rejects_cpu():
    q = torch.randn(1, 64, 1, 128)
    with pytest.raises(RuntimeError, match="CUDA"):
        anima.sparse_attention(q, q, q, 50, kernel=types.SimpleNamespace())


def test_nonfinite_anima_observation_falls_back():
    run = anima.AnimaRun(common.AnimaOptions(mode="rules"), 1, MemoryLog())
    run.pending[0] = torch.tensor([float("nan"), 0.0])
    run.end_evaluation()
    assert run.circuit_open and run.keeps == {"0": 100}


def graph():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "private scene"}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "8": {"class_type": "BasicScheduler", "inputs": {"steps": 4}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["15", 0]}},
    }


@pytest.mark.parametrize("mode", ["dense", "fixed5", "fixed10", "jev"])
def test_h3_graph_patch(mode):
    g = graph()
    integration.patch_workflow(g, mode, "/sdk/python")
    assert g["9"]["inputs"]["model"] == ["aikimi_h3_sparse", 0]
    inputs = g["aikimi_h3_sparse"]["inputs"]
    assert inputs["model"] == ["15", 0] and inputs["mode"] == mode
    assert inputs["prompt_context"] == "private scene"
    assert "TYPESAFE_API_KEY" not in json.dumps(g)


@pytest.mark.parametrize("bad", ["8steps", "20steps", "sampler", "control", "sparse", "collision", "image"])
def test_h3_graph_rejects_without_mutation(bad):
    g = graph()
    if bad in {"8steps", "20steps"}:
        g["8"]["inputs"]["steps"] = 8 if bad == "8steps" else 20
    elif bad == "sampler":
        g["7"]["inputs"]["sampler_name"] = "euler"
    elif bad == "control":
        g["17"] = {"class_type": "ApplyControlNet", "inputs": {}}
    elif bad == "sparse":
        g["16"] = {"class_type": "BlockSparseAttention", "inputs": {}}
    elif bad == "collision":
        g["aikimi_h3_sparse"] = {"class_type": "Other", "inputs": {}}
    else:
        g["5"]["class_type"] = "H3Image"
    before = copy.deepcopy(g)
    with pytest.raises(ValueError):
        integration.patch_workflow(g, "fixed5")
    assert g == before


@pytest.mark.parametrize("mode,keep", [("dense", 100), ("fixed5", 5), ("fixed10", 10)])
def test_h3_fixed_four_steps_no_client(mode, keep):
    c = h3_node.H3Controller(mode, MemoryLog())
    c.initialize()
    for index in range(4):
        c.done(index)
    assert c.step == 4 and set(c.keeps) == {keep}


def test_h3_speed_profile_calls_once_without_sending_prompt():
    client = FakeClient(keep=3.0)
    controller = h3_node.H3Controller("jev", MemoryLog(), client, "private prompt", max_calls=1, initial_decision=False)
    controller.initialize()
    assert client.calls == 0 and set(controller.keeps) == {5.0}
    for step in range(4):
        controller.pending = {i: torch.tensor([1.0, 0.1, 2.0, 0.2]) for i in range(50)}
        controller.done(step)
    assert client.calls == 1
    assert "private prompt" not in json.dumps(client.requests)
    assert controller.keeps == [5.0] + [3.0] * 49


def test_h3_jev_batches_50_then_49_four_calls():
    client = FakeClient()
    c = h3_node.H3Controller("jev", MemoryLog(), client, "private prompt")
    c.initialize()
    for index in range(4):
        c.pending = {i: torch.tensor([1.0, -1.0, 2.0, -1.0]) for i in range(50)}
        c.done(index)
    assert [len(r[1]) for r in client.requests] == [50, 49, 49, 49]
    assert client.calls == 4
    assert c.keeps[0] == 5
    assert c.prompt == ""
    assert "prompt" not in client.requests[1][0]


@pytest.mark.parametrize("fail_at,expected_keep", [(1, 10), (2, 5)])
def test_h3_api_failure_and_no_retry(fail_at, expected_keep):
    client = FakeClient(fail_at=fail_at)
    c = h3_node.H3Controller("jev", MemoryLog(), client, "prompt")
    c.initialize()
    for index in range(4):
        c.pending = {i: torch.tensor([1.0, -1.0, 2.0, -1.0]) for i in range(50)}
        c.done(index)
    assert client.calls == fail_at and c.disabled
    assert set(c.keeps) == {expected_keep}


def test_h3_sampler_callback_order_checked():
    with pytest.raises(RuntimeError, match="step order"):
        h3_node.H3Controller("fixed5", MemoryLog()).done(1)


@pytest.mark.parametrize("interval,maximum,expected", [(1, 1, [2]), (2, 3, [2, 4]), (1, 3, [2, 3, 4])])
def test_h3_configurable_cadence_without_prompt_or_final_step_call(interval, maximum, expected):
    client = FakeClient(keep=3)
    controller = h3_node.H3Controller(
        "jev",
        MemoryLog(),
        client,
        "private prompt",
        max_calls=maximum,
        initial_decision=False,
        update_interval=interval,
    )
    controller.initialize()
    for step in range(4):
        controller.pending = {i: torch.tensor([step + 1.0, 0.1, 2.0, 0.2]) for i in range(50)}
        controller.done(step)
    assert [state["next_step"] for state, _, _ in client.requests] == expected
    assert all(set(allowed) == {str(i) for i in range(1, 50)} for _, allowed, _ in client.requests)
    assert "private prompt" not in json.dumps(client.requests)


def test_h3_cadence_roundtrip_and_workflow():
    from modules_forge.minimax_h3_acceleration import H3Acceleration

    option = H3Acceleration(jev_cadence="interval", jev_interval=2)
    assert H3Acceleration.from_values(option.values()) == option
    assert H3Acceleration.from_dict(option.to_dict()) == option
    assert H3Acceleration.from_values(option.values()[:-2]).jev_cadence == "once"
    workflow = graph()
    integration.patch_workflow(workflow, "jev", "/sdk/python", cadence="interval", interval=2)
    inputs = workflow["aikimi_h3_sparse"]["inputs"]
    assert inputs["decision_cadence"] == "interval" and inputs["decision_interval"] == 2


@pytest.mark.parametrize("cadence,expected", [("once", 1), ("interval", 6), ("step", 11)])
def test_anima_cadence_can_continue_beyond_legacy_call_limit(cadence, expected):
    client = FakeClient()
    run = anima.AnimaRun(
        common.AnimaOptions(mode="jev", min_tokens=64, max_calls=1, decision_cadence=cadence, update_interval=2),
        2,
        MemoryLog(),
        client,
    )
    for step in range(12):
        run.begin_evaluation()
        for layer in range(2):
            run.observe(layer, torch.ones(1, 64, 16) * (step + 1), torch.ones(1, 64, 1, 16))
        run.end_evaluation()
    assert client.calls == expected


@pytest.mark.parametrize("cadence", ["legacy", "once", "interval", "step"])
def test_zero_api_budget_remains_explicit_opt_out(cadence):
    assert not common.cadence_has_budget(cadence, 0, 0)


def test_h3_samples_skip_reference_and_limit():
    x = torch.randn(1000, 512)
    layout = types.SimpleNamespace(segments=[(0, 100, "reference"), (100, 200, "audio"), (200, 1000, "video")])
    samples = h3_node._samples(x, layout)
    assert samples["audio"].shape == samples["video"].shape == (64, 16)
    assert "reference" not in samples


def test_subprocess_proxy_key_is_scoped(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    monkeypatch.setenv("AIKIMI_JEV_ALLOW_CLOUD", "1")
    calls = []
    original = types.SimpleNamespace(Popen=lambda *args, **kwargs: calls.append((args, kwargs)))
    proxy = integration._SubprocessProxy(original)
    for mode in (None, "h3_fixed5", "h3_jev"):
        token = integration._START_MODE.set(mode)
        try:
            proxy.Popen(["python", "main.py", "--whitelist-custom-nodes", integration.PACK], env={})
            proxy.Popen(["git", "status"], env={})
        finally:
            integration._START_MODE.reset(token)
    assert [bool(kwargs["env"].get("TYPESAFE_API_KEY")) for _, kwargs in calls] == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_pack_removal_preserves_unknown_flags():
    def parse(args):
        return (integration.PACK, "UnknownPack")

    args = [
        "main.py",
        "--listen",
        "0.0.0.0",
        "--whitelist-custom-nodes",
        integration.PACK,
        "UnknownPack",
        "--enable-manager",
    ]
    stripped = integration.strip_pack(args, parse)
    assert "UnknownPack" in stripped and "--enable-manager" in stripped and "0.0.0.0" in stripped
    assert integration.PACK not in stripped


def test_dropdown_choices_idempotent():
    control = types.SimpleNamespace(elem_id="h3-attention-mode", choices=[("Dense", "dense")])
    integration.after_component(control)
    integration.after_component(control)
    assert len(control.choices) == 5
    other = types.SimpleNamespace(elem_id="unrelated", choices=[])
    integration.after_component(other)
    assert not other.choices


def test_installer_dry_run_is_offline():
    before = sorted(str(p) for p in (ROOT / "repositories").glob("*"))
    result = subprocess.run(  # noqa: S603 -- fixed offline installer test
        [sys.executable, str(ROOT / "tools/setup_jev_sparse.py"), "--sdk", "--create-h3-runtime", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["api_calls"] == 0
    assert sorted(str(p) for p in (ROOT / "repositories").glob("*")) == before


def fake_anima_patcher(monkeypatch):
    backend = types.ModuleType("backend")
    args = types.ModuleType("backend.args")
    args.dynamic_args = types.SimpleNamespace(ref_latents=[])
    monkeypatch.setitem(sys.modules, "backend", backend)
    monkeypatch.setitem(sys.modules, "backend.args", args)

    class Attention:
        is_SelfAttn, head_dim = True, 128

        def __init__(self):
            self.output_proj = torch.nn.Identity()
            self.output_dropout = torch.nn.Identity()

        def compute_attention(self, q, k, v, transformer_options=None):
            return v.flatten(2)

    Model = type("Anima", (), {"__module__": "backend.nn.anima"})
    model = Model()
    model.blocks = [types.SimpleNamespace(self_attn=Attention(), cross_attn=Attention()) for _ in range(2)]

    class Patcher:
        def __init__(self):
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}
            self.load_device = torch.device("cpu")

        def get_model_object(self, name):
            if name == "diffusion_model":
                return model
            return model.blocks[int(name.split(".")[2])].self_attn.compute_attention

        def clone(self):
            other = Patcher()
            other.model_options = self.model_options.copy()
            other.object_patches = self.object_patches.copy()
            return other

        def add_object_patch(self, name, fn):
            self.object_patches[name] = fn

        def set_model_unet_function_wrapper(self, fn):
            self.model_options["model_function_wrapper"] = fn

    return Patcher(), args.dynamic_args


def test_anima_off_returns_same_object_without_io(tmp_path):
    obj = object()
    assert anima.attach(obj, common.AnimaOptions(), tmp_path / "absent") == (obj, None)
    assert not (tmp_path / "absent").exists()


def test_anima_cloned_patch_chain_and_cleanup(monkeypatch, tmp_path):
    original, dynamic = fake_anima_patcher(monkeypatch)
    chained = []

    def previous(fn, args):
        chained.append(True)
        return fn(args["input"], args["timestep"], **args["c"])

    original.model_options["model_function_wrapper"] = previous
    patched, run = anima.attach(original, common.AnimaOptions(mode="dense", min_tokens=64), tmp_path)
    assert original.object_patches == {}
    assert all(".self_attn.compute_attention" in key and ".cross_attn." not in key for key in patched.object_patches)
    q = torch.randn(1, 64, 1, 128)

    def model(x, t, **kwargs):
        result = None
        for fn in patched.object_patches.values():
            result = fn(x, x, x, transformer_options={})
        return result

    wrapper = patched.model_options["model_function_wrapper"]
    args = {"input": q, "timestep": torch.ones(1), "c": {}}
    assert torch.equal(wrapper(model, args), q.flatten(2))
    assert chained == [True] and run.evaluation == 0
    assert anima._ACTIVE.get() is None
    assert run.total_counts["dense"] == 2
    run.close()
    assert torch.equal(wrapper(model, args), q.flatten(2))
    assert run.evaluation == 0  # No old controller reactivation.
    assert anima._ACTIVE.get() is None


def test_anima_refs_added_after_attach_rejected(monkeypatch, tmp_path):
    original, dynamic = fake_anima_patcher(monkeypatch)
    patched, run = anima.attach(original, common.AnimaOptions(mode="dense"), tmp_path)
    dynamic.ref_latents = [object()]
    with pytest.raises(ValueError, match="reference"):
        patched.model_options["model_function_wrapper"](
            lambda *a: pytest.fail("model executed"), {"input": torch.ones(1), "timestep": torch.ones(1), "c": {}}
        )
    assert run.closed


def test_anima_failure_deactivates_context(monkeypatch, tmp_path):
    original, dynamic = fake_anima_patcher(monkeypatch)
    patched, run = anima.attach(original, common.AnimaOptions(mode="dense"), tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("fake GPU error")

    with pytest.raises(RuntimeError, match="fake GPU"):
        patched.model_options["model_function_wrapper"](
            fail, {"input": torch.ones(1), "timestep": torch.ones(1), "c": {}}
        )
    assert run.closed and anima._ACTIVE.get() is None


def fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.InterruptProcessingException = type("InterruptProcessingException", (Exception,), {})
    pe = types.ModuleType("comfy.patcher_extension")
    pe.CallbacksMP = types.SimpleNamespace(ON_PREPARE_STATE="prepare", ON_CLEANUP="cleanup")
    pe.WrappersMP = types.SimpleNamespace(SAMPLER_SAMPLE="sample")
    extras = types.ModuleType("comfy_extras")
    sparse = types.ModuleType("comfy_extras.nodes_sparse_attention")

    class SparsePatch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    sparse.SparseAttnPatch = SparsePatch
    sparse.install_override = lambda patch, options: None
    sparse.h3_eligible = lambda *args: False
    sparse.h3_sparse_attention = lambda *args: pytest.fail("CUDA producer should not run in this CPU integration test")
    for name, module in [
        ("comfy", comfy),
        ("comfy.model_management", mm),
        ("comfy.patcher_extension", pe),
        ("comfy_extras", extras),
        ("comfy_extras.nodes_sparse_attention", sparse),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    class Model:
        def __init__(self):
            self.model_options = {"transformer_options": {}}
            self.blocks = [types.SimpleNamespace(attn=object()) for _ in range(50)]
            self.patches, self.callbacks, self.wrappers = {}, {}, {}

        def get_model_object(self, name):
            return (
                types.SimpleNamespace(blocks=self.blocks)
                if name == "diffusion_model"
                else types.SimpleNamespace(percent_to_sigma=lambda p: 1 - p)
            )

        def clone(self):
            return Model()

        def add_callback_with_key(self, name, key, fn):
            self.callbacks[name] = fn

        def set_model_patch_replace(self, fn, kind, block, i):
            self.patches[i] = fn

        def add_wrapper_with_key(self, name, key, fn):
            self.wrappers[name] = fn

    return Model()


@pytest.mark.parametrize("mode", ["dense", "fixed5", "fixed10"])
def test_h3_mock_comfy_four_step_execution(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("AIKIMI_SPARSE_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(h3_node, "JevClient", lambda *args: pytest.fail("fixed mode constructed cloud client"))
    original = fake_comfy(monkeypatch)
    (patched,) = h3_node.AikimiH3SparseExperiment().patch(original, mode)
    assert not original.patches

    class Executor:
        class_obj = types.SimpleNamespace(sampler_function=types.SimpleNamespace(__name__="sample_res_multistep"))

        def __call__(self, model, sigmas, extra, cb, noise, *args):
            x = noise
            for step in range(4):
                for block in patched.patches.values():
                    x = block(
                        {"img": x, "rope_freqs": None, "transformer_options": {}},
                        {"original_block": lambda a: {"img": a["img"] + 1}},
                    )["img"]
                cb(step, x, x, 4)
            return x

    result = patched.wrappers["sample"](Executor(), None, torch.linspace(1, 0, 5), {}, None, torch.zeros(2, 2))
    assert torch.equal(result, torch.full((2, 2), 200.0))
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert len([r for r in records if r["event"] == "step"]) == 4
    assert records[-1]["status"] == "completed" and records[-1]["api_calls"] == 0


@pytest.mark.parametrize(
    "sigmas,sampler", [(9, "sample_res_multistep"), (21, "sample_res_multistep"), (5, "sample_euler")]
)
def test_h3_node_rejects_wrong_actual_sampler(monkeypatch, sigmas, sampler):
    (patched,) = h3_node.AikimiH3SparseExperiment().patch(fake_comfy(monkeypatch), "fixed5")
    executor = types.SimpleNamespace(
        class_obj=types.SimpleNamespace(sampler_function=types.SimpleNamespace(__name__=sampler))
    )
    with pytest.raises(ValueError, match="restricted"):
        patched.wrappers["sample"](executor, None, torch.ones(sigmas), {}, None, torch.ones(2, 2))


def load_setup():
    spec = importlib.util.spec_from_file_location("jev_setup_test", ROOT / "tools/setup_jev_sparse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_pack_atomic_and_reject_user_edits(monkeypatch, tmp_path):
    module = load_setup()
    root = tmp_path / "ComfyUI"
    (root / "models").mkdir(parents=True)
    (root / "comfy_extras").mkdir()
    (root / "main.py").write_text("# fake core")
    source = b"# fake pinned sparse API\n"
    (root / "comfy_extras/nodes_sparse_attention.py").write_bytes(source)
    monkeypatch.setattr(module, "SPARSE_BLOBS", {integration.git_blob(source)})
    module.install_pack(root)
    installed = root / "custom_nodes" / integration.PACK
    assert (installed / "jev_sparse/h3_node.py").exists()
    module.install_pack(root)
    assert not installed.with_name(integration.PACK + ".previous").exists()
    (installed / "jev_sparse/h3_node.py").write_text("# user edits")
    with pytest.raises(RuntimeError, match="Local node edits"):
        module.install_pack(root)
    assert (installed / "jev_sparse/h3_node.py").read_text() == "# user edits"


def test_runtime_create_missing_models_does_not_download(monkeypatch, tmp_path):
    module = load_setup()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", lambda *args, **kw: pytest.fail("download before input validation"))
    with pytest.raises(FileNotFoundError):
        module.create_runtime(tmp_path / "missing")
    assert not (tmp_path / "repositories").exists()


def test_policy_hooks_default_delegation_roundtrip_and_uninstall(monkeypatch):
    integration.uninstall()

    @dataclass(frozen=True)
    class Policy:
        attention: str = "dense"
        model_variant: str = "base"
        clip_cache: str = "off"
        negpip: bool = False

        def validate(self):
            if self.attention not in {"dense", "sol", "sla"}:
                raise ValueError("bad attention")
            if self.negpip and self.attention != "dense":
                raise ValueError("incompatible NegPiP")

        def runtime_packs(self):
            return ("OldPack",)

        def extra_nodes(self):
            return {"BlockSparseAttention"} if self.attention != "dense" else set()

        def apply_workflow(self, graph, base, mode):
            self.validate()
            graph["1"]["inputs"]["variant"] = self.model_variant

        def validate_nodes(self, nodes):
            self.validate()

        @classmethod
        def from_values(cls, values):
            p = cls(*values)
            p.validate()
            return p

    policy = types.ModuleType("modules_forge.minimax_h3_acceleration")
    policy.H3Acceleration = Policy
    ui = types.ModuleType("modules_forge.minimax_h3_acceleration_ui")
    ui.acceleration_note = lambda *values: "original note"
    ui._control_state = lambda *values: (ui.acceleration_note(*values), {}, {}, {}, {})
    bridge = types.ModuleType("modules_forge.minimax_h3_bridge")
    bridge.H3BridgeError = type("H3BridgeError", (Exception,), {})
    bridge.validate_request = lambda req: req.acceleration.validate()

    def parse(args):
        if "--whitelist-custom-nodes" not in args:
            return ()
        i = args.index("--whitelist-custom-nodes") + 1
        return tuple(args[i:])

    bridge.custom_node_whitelist = parse
    bridge._runtime_arguments_are_allowed = lambda args: (
        set(parse(args)) <= {"OldPack"} and "--enable-manager" not in args
    )

    def start(
        runtime_root,
        server_url,
        log_directory,
        runtime_profile="fast",
        wait_seconds=120,
        initial_readiness=None,
        acceleration=None,
    ):
        return "original-start"

    bridge._start_runtime_locked = start
    bridge.subprocess = subprocess
    import modules_forge

    for module in (policy, ui, bridge):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(modules_forge, module.__name__.rsplit(".", 1)[1], module, raising=False)
    original_validate = Policy.validate
    try:
        integration.install()
        integration.install()  # Idempotent, not nested twice.
        assert Policy().runtime_packs() == ("OldPack",)
        assert Policy(attention="sla").extra_nodes() == {"BlockSparseAttention"}
        baseline = graph()
        Policy().apply_workflow(baseline, {}, "text")
        assert "aikimi_h3_sparse" not in baseline
        option = Policy.from_values(("h3_fixed5", "fused_turbo", "off", False))
        option.validate()
        assert option.extra_nodes() == {integration.NODE}
        assert option.runtime_packs() == ("OldPack", integration.PACK)
        g = graph()
        option.apply_workflow(g, {}, "text")
        assert g["aikimi_h3_sparse"]["inputs"]["mode"] == "fixed5"
        assert g["1"]["inputs"]["variant"] == "fused_turbo"
        with pytest.raises(ValueError):
            Policy(attention="h3_fixed5", model_variant="base").validate()
        with pytest.raises(ValueError):
            Policy(attention="h3_fixed5", model_variant="fused_turbo", negpip=True).validate()
        Policy(attention="h3_fixed5", model_variant="fused_turbo", clip_cache="auto").validate()
        assert bridge._runtime_arguments_are_allowed(["main.py", "--whitelist-custom-nodes", integration.PACK])
        assert not bridge._runtime_arguments_are_allowed(
            ["main.py", "--whitelist-custom-nodes", integration.PACK, "UnknownPack"]
        )
        req = types.SimpleNamespace(acceleration=option, steps=20, control=types.SimpleNamespace(enabled=False))
        with pytest.raises(bridge.H3BridgeError):
            bridge.validate_request(req)
        req.steps = 4
        req.control.enabled = True
        with pytest.raises(bridge.H3BridgeError):
            bridge.validate_request(req)
        schema = {integration.NODE: {"input": h3_node.AikimiH3SparseExperiment.INPUT_TYPES(), "output": ["MODEL"]}}
        option.validate_nodes(schema)
        schema[integration.NODE]["output"] = ["IMAGE"]
        with pytest.raises(ValueError):
            option.validate_nodes(schema)
        assert (
            bridge._start_runtime_locked(Path("."), "localhost", Path("."), acceleration=Policy()) == "original-start"
        )
        assert bridge.subprocess is not subprocess
    finally:
        integration.uninstall()
    assert Policy.validate is original_validate
    assert bridge.subprocess is subprocess
    assert bridge._start_runtime_locked is start


def test_runtime_selector_is_exposed_without_changing_default():
    c = types.SimpleNamespace(elem_id="h3-runtime-path", value="normal/ComfyUI", visible=False, interactive=False)
    integration.after_component(c)
    assert c.visible and c.interactive
    assert c.value == "normal/ComfyUI"


def test_real_gradio_anima_ui_default_off(monkeypatch):
    import gradio as gr

    modules = types.ModuleType("modules")
    modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=object())
    modules.script_callbacks = types.SimpleNamespace(
        on_before_ui=lambda fn: None, on_after_component=lambda fn: None, on_script_unloaded=lambda fn: None
    )
    modules.shared = types.SimpleNamespace(state=types.SimpleNamespace(interrupted=False, skipped=False))
    paths = types.ModuleType("modules.paths")
    paths.data_path = str(ROOT)
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.paths", paths)
    spec = importlib.util.spec_from_file_location(
        "test_jev_entry", ROOT / "extensions-builtin/jev-sparse-experiments/scripts/jev_sparse_experiments.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    with gr.Blocks():
        script = entry.Script()
        controls = script.ui(False)
        runtime = gr.Textbox(value="normal/ComfyUI", visible=False, interactive=False, elem_id="h3-runtime-path")
        integration.after_component(runtime)
    assert len(controls) == 7
    assert controls[0].value == "off"
    assert runtime.get_config()["visible"] is True
    assert runtime.get_config()["interactive"] is True
    p = types.SimpleNamespace(extra_generation_params={})
    script.process(p, "off", 75, 4096, 1, 4, 4, 3)
    script.process_before_every_sampling(p)
    assert not p.extra_generation_params


def test_report_preserves_timing_scope_without_speedup(tmp_path):
    spec = importlib.util.spec_from_file_location("jev_report_test", ROOT / "tools/report_jev_sparse.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    log = common.RunLog(tmp_path, "anima", {"mode": "fixed"}, "prompt")
    log.finish(
        "completed",
        timing_scope="sum_instrumented_model_evaluations",
        model_seconds=12.5,
        attention_calls={"sparse": 5},
        api_calls=0,
        api_wait_seconds=0,
    )
    summary = report.summarize(log.path)
    assert summary["measured_seconds"] == 12.5
    assert summary["timing_scope"] == "sum_instrumented_model_evaluations"
    assert summary["attention_calls"]["sparse"] == 5
    assert "speedup" not in summary
