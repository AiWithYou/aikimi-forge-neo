"""CPU-only contract tests. Fake CLI/model outputs are never quality validation."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
import types
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge.yue2_studio import core, service, worker


def request(**changes):
    return replace(core.Request(style="English, piano pop", lyrics="[Verse]\nAn original test lyric"), **changes)


@pytest.mark.parametrize("value", [True, None, [], {}, "1.1", "1e3", "NaN", float("nan"), float("inf"), 1.5])
def test_reject_nonintegers(value):
    with pytest.raises(core.YuE2Error):
        core.integer(value, "Seed", -1, 2**63 - 1)


@pytest.mark.parametrize("value", [-1, 0, 42, 2**53 + 1, 2**63 - 1])
def test_seed_preserved(value):
    assert core.integer(str(value), "Seed", -1, 2**63 - 1) == value
    resolved = request(seed=value).resolved()
    assert resolved.seed >= 0
    if value != -1:
        assert resolved.seed == value


@pytest.mark.parametrize("changes", [
    {"style": ""}, {"lyrics": ""}, {"style": "\x00"}, {"style": "a" * 2001},
    {"seed": 1.0}, {"seed": True}, {"seed": 2**63}, {"candidates": 9},
    {"steps": 0}, {"max_tokens": 100}, {"memory_gib": 1}, {"offload": "yes"},
    {"abc": "K:C\nCDEF", "cot": "off"}, {"engine": "url"}, {"gguf": "../../model"},
    {"engine": "cpp", "fp8": True},
])
def test_request_boundaries(changes):
    with pytest.raises(core.YuE2Error):
        request(**changes).validate()


def test_project_roundtrip_and_security(tmp_path):
    path = tmp_path / "project.json"
    value = request(seed=2**63 - 1)
    core.atomic_json(path, {"schema": 1, "request": asdict(value)})
    assert core.import_project(path) == value
    core.atomic_json(path, {"schema": 1, "request": {**asdict(value), "python": "evil.exe"}})
    with pytest.raises(core.YuE2Error):
        core.import_project(path)
    path.write_text('{"schema": 1, "request": {"seed": NaN}}')
    with pytest.raises(core.YuE2Error):
        core.import_project(path)
    path.write_bytes(b" " * (core.MAX_JSON_BYTES + 1))
    with pytest.raises(core.YuE2Error):
        core.read_json(path)


def test_atomic_failure_keeps_old_file(tmp_path):
    path = tmp_path / "project.json"
    core.atomic_json(path, {"old": 1})
    with pytest.raises(ValueError):
        core.atomic_json(path, {"bad": float("nan")})
    assert core.read_json(path) == {"old": 1}
    assert len(list(tmp_path.iterdir())) == 1


def test_output_boundary(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    with pytest.raises(core.YuE2Error):
        core.inside(root, root / "../private.txt")
    if os.name != "nt":
        (root / "link").symlink_to(tmp_path)
        with pytest.raises(core.YuE2Error):
            core.inside(root, root / "link" / "secret")


def test_environment_does_not_forward_secrets(monkeypatch):
    monkeypatch.setenv("USERNAME", "yue2-test-user")
    assert core.safe_environment()["USERNAME"] == "yue2-test-user"
    for key in ("HF_TOKEN", "OPENAI_API_KEY", "HTTP_PROXY", "PYTHONPATH"):
        monkeypatch.setenv(key, "must-not-leak")
    env = core.safe_environment()
    assert "must-not-leak" not in env.values()
    assert env["HF_HUB_OFFLINE"] == "1"


def test_setup_lock_excludes_second_owner(tmp_path):
    lock = core.runtime_lock(tmp_path)
    try:
        with pytest.raises(core.YuE2Error):
            core.runtime_lock(tmp_path)
    finally:
        lock.close()
    core.runtime_lock(tmp_path).close()


def cpp_files(tmp_path):
    models = tmp_path / "models"
    for path in core.required_cpp_files(models, "q8_0"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test-fixture-not-a-model")
    binary = tmp_path / "audiocpp_cli"
    binary.write_text("#!/usr/bin/env python3\n")
    binary.chmod(0o755)
    return binary, models


def test_cpp_arguments_are_literal_and_inputs_preserved(tmp_path):
    binary, models = cpp_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    value = request(engine="cpp", seed=2**63 - 1, style='piano; $(touch stolen) "x"', abc='K:C\nCDEF|')
    args = core.cpp_command(binary, models, value, out, 0)
    assert args[args.index("--lyrics") + 1] == value.lyrics
    assert "style=" + value.style in args
    assert str(2**63 - 1) in args
    assert (out / "input.abc").read_text() == value.abc
    assert value.song(1)["seed"] == 0
    (models / core.GGUF_MODELS["q8_0"]).unlink()
    with pytest.raises(core.YuE2Error):
        core.cpp_command(binary, models, value, out, 0)


class FakeLease:
    def __init__(self):
        self.acquired = False
        self.released = threading.Event()

    def acquire(self, blocking=False):
        self.acquired = True
        return True

    def release(self):
        self.acquired = False
        self.released.set()


@pytest.fixture
def cpp_studio(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX executable fixture; Windows needs native CLI smoke")
    binary, models = cpp_files(tmp_path)
    binary.write_text('''#!/usr/bin/env python3
import sys,time,wave,os
from pathlib import Path
args=sys.argv[1:]
lyrics=args[args.index('--lyrics')+1]
seed=int(args[args.index('--seed')+1])
if 'SLOW' in lyrics:
    Path(args[args.index('--out')+1]).with_suffix('.pid').write_text(str(os.getpid()))
    time.sleep(30)
if 'FAILSECOND' in lyrics and seed==43: raise SystemExit(2)
path=Path(args[args.index('--out')+1])
with wave.open(str(path),'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes(b'\\0'*1920)
''')
    runtime = tmp_path / "runtime"
    core.atomic_json(runtime / "runtime.json", {"schema": 1, "cpp": {"binary": str(binary), "models": str(models)}})
    lease = FakeLease()
    studio = service.Studio(runtime, tmp_path / "outputs", ownership_factory=lambda: lease, release_vram=lambda: None)
    yield studio, lease
    studio.shutdown()
    for job in studio._jobs.values():
        job.done.wait(8)


def wait_done(studio, identifier, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = studio.status(identifier, "owner")
        if state["done"]:
            return state
        time.sleep(0.05)
    pytest.fail("Worker did not finish")


def test_real_supervisor_worker_cli_pipeline(cpp_studio):
    studio, lease = cpp_studio
    identifier = studio.start(request(engine="cpp", seed=42, candidates=2), "owner")
    state = wait_done(studio, identifier)
    assert state["state"] == "complete", state
    assert lease.released.is_set()
    choices = studio.history()
    assert len(choices) == 2
    for i, (_, key) in enumerate(choices):
        directory = studio.artifact(key)
        assert (directory / "audio.wav").stat().st_size > 44
        metadata = core.read_json(directory / "studio-result.json")
        assert metadata["seed"] == 42 + i
        assert metadata["truncated"] is None
        assert core.import_project(directory / "project.json").candidates == 1


def test_failed_later_candidate_keeps_first(cpp_studio):
    studio, _ = cpp_studio
    identifier = studio.start(request(engine="cpp", seed=42, candidates=2, lyrics="FAILSECOND"), "owner")
    assert wait_done(studio, identifier)["state"] == "failed"
    assert len(studio.history()) == 1
    assert (studio.artifact(studio.history()[0][1]) / "audio.wav").exists()


def test_cancel_is_owned_and_lease_follows_process_exit(cpp_studio):
    studio, lease = cpp_studio
    identifier = studio.start(request(engine="cpp", lyrics="SLOW", seed=42), "owner")
    assert not studio.cancel(identifier, "other-session")
    assert not studio.cancel("stale-id", "owner")
    with pytest.raises(core.YuE2Error):
        studio.start(request(engine="cpp"), "owner")
    assert not lease.released.is_set()
    assert studio.cancel(identifier, "owner")
    state = wait_done(studio, identifier)
    assert state["state"] == "cancelled"
    assert lease.released.is_set()
    job = studio._jobs[identifier]
    assert job.process is None or job.process.poll() is not None
    assert not studio.cancel(identifier, "owner")


def test_install_lock_blocks_generation(cpp_studio):
    studio, lease = cpp_studio
    lock = core.runtime_lock(studio.runtime)
    try:
        with pytest.raises(core.YuE2Error):
            studio.start(request(engine="cpp"), "owner")
        assert not lease.acquired
    finally:
        lock.close()


def test_native_contract_with_fake_pipeline(tmp_path, monkeypatch):
    events = []
    class SongRequest:
        def __init__(self, **data): self.data = data
        def to_dict(self): return self.data
    class Plan:
        timing = {}
        truncated = False
        def save(self, directory): (directory / "score.abc").write_text("K:C\nCDEF|")
    class Pipeline:
        weights = {"fake": True}
        load_timing = {}
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs['local_files_only'] is True
            assert kwargs['backend'] == 'torch-eager'
            events.append('load'); return cls()
        def __enter__(self): return self
        def __exit__(self, *args): events.append('close')
        def plan(self, **kwargs): events.append('plan'); return Plan()
        def generate_semantic(self, plan, **kwargs): events.append('semantic'); return types.SimpleNamespace(timing={})
        def synthesize(self, semantic, **kwargs): events.append('synthesize'); return []
        def decode(self, latents): events.append('decode'); return []
        def effective_config(self, request): return {}
    class Result:
        truncated = {"abc": False, "semantic": True}
        def __init__(self, *args): assert len(args) == 8
        def save_artifacts(self, out): (out / "audio.flac").write_bytes(b'fake')
        def save(self, out): out.write_bytes(b'fake')
    cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
        get_device_capability=lambda: (8, 6), get_device_properties=lambda _: types.SimpleNamespace(total_memory=24*2**30))
    modules = {"torch": types.SimpleNamespace(cuda=cuda), "yue2": types.SimpleNamespace(YuE2Pipeline=Pipeline),
               "yue2.pipeline": types.SimpleNamespace(SongResult=Result),
               "yue2.protocol": types.SimpleNamespace(SongRequest=SongRequest, GenerationConfig=types.SimpleNamespace(from_dict=lambda d: d)),
               "yue2.storage": types.SimpleNamespace(identity=lambda d: "fake-identity")}
    for name, value in modules.items(): monkeypatch.setitem(sys.modules, name, value)
    worker.native(request(seed=42), {"model": "model", "vae": "vae"}, tmp_path, False, lambda _: None, lambda: False)
    assert events == ['load', 'plan', 'semantic', 'synthesize', 'decode', 'close']
    assert core.read_json(tmp_path / 'take-1' / 'studio-result.json')['truncated'] is True
    with pytest.raises(core.YuE2Error):
        worker.native(request(seed=42, fp8=True), {}, tmp_path, False, lambda _: None, lambda: False)


def load_ui(monkeypatch, tmp_path):
    modules = types.ModuleType("modules")
    modules.script_callbacks = types.SimpleNamespace(on_ui_tabs=lambda callback: None)
    paths = types.ModuleType("modules.paths")
    paths.data_path, paths.script_path = str(tmp_path), str(ROOT)
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.paths", paths)
    spec = importlib.util.spec_from_file_location("test_yue2_ui", ROOT / "extensions-builtin/yue2-studio/scripts/yue2_studio.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    return ui


def test_real_gradio_layout_and_callback_contract(monkeypatch, tmp_path):
    gr = pytest.importorskip("gradio")
    if int(gr.__version__.split('.')[0]) < 6:
        pytest.skip("Host uses Gradio 6")
    ui = load_ui(monkeypatch, tmp_path)
    result = ui.on_ui_tabs()
    assert result[0][1:] == ("YuE2 Music", "aikimi_yue2_studio")
    assert ui.make_request(ui.values_for(request(seed=2**63 - 1))).seed == 2**63 - 1
    assert ui.use_score("K:C\nCDEF", "off")[1] == "full"
    from gradio.helpers import special_args
    args, *_ = special_args(ui.run, [*ui.values_for(request()), False], request=gr.Request(session_hash="s"))
    assert len(args) == len(ui.FIELDS) + 2
    assert isinstance(args[-1], gr.Request)
    values = ui.values_for(request())
    values[ui.FIELDS.index("title")] = None
    values[ui.FIELDS.index("abc")] = None
    assert ui.make_request(values).abc == ""
    values[ui.FIELDS.index("style")] = None
    with pytest.raises(core.YuE2Error, match="曲調を入力"):
        ui.make_request(values)


def test_parent_pipe_loss_terminates_worker_tree(cpp_studio):
    if os.name == "nt":
        pytest.skip("Windows Job Object requires a Windows runner")
    studio, _ = cpp_studio
    directory = studio.outputs / "orphan-test"
    directory.mkdir(parents=True)
    core.atomic_json(directory / "project.json", {"schema": 1, "request": asdict(request(engine="cpp", lyrics="SLOW", seed=42))})
    process = subprocess.Popen([sys.executable, "-u", str(ROOT / "modules_forge/yue2_studio/worker.py"),
                                "--runtime", str(studio.runtime), "--job", str(directory)],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True, env=core.safe_environment())
    try:
        process.stdin.write(b"GO\n"); process.stdin.flush()
        pid_file = directory / ".take-1.partial" / "audio.pid"
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "Fake CLI did not start"
        child_pid = int(pid_file.read_text())
        process.stdin.close()
        assert process.wait(timeout=5) != 0
        stat = Path(f"/proc/{child_pid}/stat")
        deadline = time.monotonic() + 3
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()


def test_setup_keeps_venv_interpreter_path(tmp_path):
    from modules_forge.yue2_studio import setup
    directory = tmp_path / "isolated"
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    if os.name == "nt":
        python.write_bytes(b"test-fixture")
    else:
        python.symlink_to(sys.executable)
    assert setup.environment(directory) == python.absolute()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object")
@pytest.mark.parametrize("close_only", [False, True])
@pytest.mark.parametrize("attachment_delay", [0, 0.5])
def test_windows_job_object_stops_only_owned_tree(tmp_path, close_only, attachment_delay):
    import ctypes
    from ctypes import wintypes

    pytest.importorskip("numpy")
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from modules_forge.yue2_studio.worker import parent_guard; parent_guard(); "
        "import numpy; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    process = subprocess.Popen([sys.executable, "-c", code, str(pid_file)], stdin=subprocess.PIPE)
    tree = None
    child_handle = None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    try:
        time.sleep(attachment_delay)
        tree = service.ProcessTree(process)
        process.stdin.write(tree.handshake())
        process.stdin.flush()
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "Child did not start"
        child_handle = kernel.OpenProcess(0x00100001, False, int(pid_file.read_text()))
        assert child_handle
        if close_only:
            tree.close()
        else:
            tree.terminate()
        process.wait(timeout=10)
        assert kernel.WaitForSingleObject(child_handle, 10000) == 0
        assert unrelated.poll() is None
    finally:
        if tree:
            tree.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdin.close()
        if child_handle:
            kernel.TerminateProcess(child_handle, 1)
            kernel.CloseHandle(child_handle)
        unrelated.kill()
        unrelated.wait(timeout=10)
