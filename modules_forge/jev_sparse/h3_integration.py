"""Opt-in H3 Studio extension hooks; existing dense/sol/sla routes are delegated.

No base source rewriting. Only the H3 policy class and H3-local runtime namespace
are adapted; subprocess itself, global attention and other studios are untouched.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

from .common import cloud_environment as _cloud_environment
from .common import sdk_python
from .credentials import cloud_source

ROOT = Path(__file__).resolve().parents[2]
PACK = "Aikimi-H3-Jev"
NODE = "AikimiH3SparseExperiment"
COMFY_REVISION = "efa6c8f804bff78b46a0fd458ebd2e47bba07a30"
SPARSE_BLOB = "006d1eb352f946a7c85595edf75d3cda9ac79194"
SPARSE_BLOBS = {SPARSE_BLOB, "441b474c98249c2358f5b63f77c0d2e16b40e877"}
MODES = {"h3_dense": "dense", "h3_fixed5": "fixed5", "h3_fixed10": "fixed10", "h3_jev": "jev"}
CHOICES = [
    ("H3比較 Dense · 4 Steps / 計測", "h3_dense"),
    ("H3固定SLA 5% · 4 Steps / 通信なし", "h3_fixed5"),
    ("H3固定SLA 10% · 4 Steps / 通信なし", "h3_fixed10"),
    ("H3 Jev速度優先 · 4 Steps / 外部送信", "h3_jev"),
]
_START_MODE = contextvars.ContextVar("aikimi_h3_start_mode", default=None)
_RESTORES = []


def cloud_environment():
    return _cloud_environment(cloud_source())


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data, usedforsecurity=False).hexdigest()


def pack_files():
    source = Path(__file__).parent
    return {
        "__init__.py": b"from .jev_sparse.h3_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n",
        **{
            "jev_sparse/" + name: (source / name).read_bytes().replace(b"\r\n", b"\n")
            for name in ("__init__.py", "common.py", "sdk_worker.py", "h3_node.py")
        },
    }


def verify_runtime(root: Path):
    root = Path(root).resolve(strict=True)
    sparse = root / "comfy_extras/nodes_sparse_attention.py"
    if not sparse.is_file() or git_blob(sparse.read_bytes().replace(b"\r\n", b"\n")) not in SPARSE_BLOBS:
        raise ValueError(
            "H3実験の内部APIが固定版と一致しません。通常のH3環境は変更せず、setup_jev_sparse.py --create-h3-runtime で実験環境を作成してください。"
        )
    target = root / "custom_nodes" / PACK
    for name, expected in pack_files().items():
        path = target / name
        if not path.is_file() or path.is_symlink() or path.read_bytes().replace(b"\r\n", b"\n") != expected:
            raise ValueError(
                "H3実験ノードが未導入か版違いです。setup_jev_sparse.py --h3 --comfy-root <ComfyUI> を実行してください。"
            )


def patch_workflow(graph, mode, python=""):
    def unique(kind):
        hits = [key for key, node in graph.items() if node.get("class_type") == kind]
        if len(hits) != 1:
            raise ValueError(f"H3 experiment requires exactly one {kind}")
        return hits[0]

    if mode not in MODES.values():
        raise ValueError("Unknown experiment mode")
    scheduler, sampler, guider = unique("BasicScheduler"), unique("KSamplerSelect"), unique("BasicGuider")
    if graph[scheduler]["inputs"].get("steps") != 4 or graph[sampler]["inputs"].get("sampler_name") != "res_multistep":
        raise ValueError("H3実験は4 Steps / res_multistep限定です。Turbo + 4 Stepsを明示適用してください。")
    conditions = [
        node
        for node in graph.values()
        if node.get("class_type")
        in {
            "MiniMaxH3ImageToVideo",
            "MiniMaxH3ReferenceToVideo",
            "MiniMaxH3CLIPCachedFL2VA",
            "MiniMaxH3CLIPCachedRef2VA",
        }
    ]
    if len(conditions) != 1:
        raise ValueError("H3実験の初版は標準動画経路のみ対応します。H3 Image・Control・長尺は対象外です。")
    prompt = conditions[0]["inputs"].get("prompt", "")
    if mode == "jev" and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("H3 prompt input schema differs from the supported runtime")
    if any(
        "Control" in str(n.get("class_type", "")) or n.get("class_type") in {"BlockSparseAttention", NODE}
        for n in graph.values()
    ):
        raise ValueError("H3実験は他のSparse/Controlパッチと重ねられません。")
    ident = "aikimi_h3_sparse"
    if ident in graph:
        raise ValueError("Experiment node ID collision")
    model = graph[guider]["inputs"]["model"]
    # Validate first, mutate last; scheduler still uses the same model sampling.
    graph[ident] = {
        "class_type": NODE,
        "inputs": {
            "model": list(model),
            "mode": mode,
            "sdk_python": python,
            "prompt_context": prompt if isinstance(prompt, str) else "",
        },
    }
    graph[guider]["inputs"]["model"] = [ident, 0]


def strip_pack(arguments, parse_whitelist):
    """Remove only our known pack, then let every original CLI guard run."""
    values = parse_whitelist(arguments)
    if values is None or PACK not in values:
        return None
    result = []
    i = 0
    args = list(arguments)
    while i < len(args):
        arg = args[i]
        if arg == "--whitelist-custom-nodes" or arg.startswith("--whitelist-custom-nodes="):
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                i += 1
            rest = [name for name in values if name != PACK]
            if rest:
                result.extend(["--whitelist-custom-nodes", *rest])
        else:
            result.append(arg)
            i += 1
    return result


class _SubprocessProxy:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Popen(self, command, *args, **kwargs):
        mode = _START_MODE.get()
        if mode in MODES and isinstance(command, (list, tuple)) and "main.py" in command and PACK in command:
            env = dict(kwargs.get("env") or {})
            env["PYTHONUNBUFFERED"] = "1"
            env["AIKIMI_SPARSE_LOG_DIR"] = str(ROOT / "outputs" / "jev-sparse")
            if MODES[mode] == "jev":
                # Only the explicitly selected ComfyUI child receives this key.
                # git and pip still call the original subprocess.run, not this proxy.
                cloud = cloud_environment()
                env.update(TYPESAFE_API_KEY=cloud["TYPESAFE_API_KEY"], AIKIMI_JEV_ALLOW_CLOUD="1")
            else:
                env.pop("TYPESAFE_API_KEY", None)
                env.pop("AIKIMI_JEV_ALLOW_CLOUD", None)
            kwargs["env"] = env
        return self.original.Popen(command, *args, **kwargs)


def _set(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    _RESTORES.append((obj, name, old, value))


def install():
    if _RESTORES:
        return
    from modules_forge import minimax_h3_acceleration as policy
    from modules_forge import minimax_h3_acceleration_ui as ui
    from modules_forge import minimax_h3_bridge as bridge

    cls = policy.H3Acceleration
    required = ("validate", "runtime_packs", "extra_nodes", "apply_workflow", "validate_nodes")
    if any(not callable(getattr(cls, name, None)) for name in required):
        raise RuntimeError("Unsupported H3 policy interface")
    validate, packs, nodes, apply, validate_nodes = (getattr(cls, name) for name in required)

    def normal(option):
        return replace(option, attention="dense")

    def validate_experiment(self):
        if self.attention not in MODES:
            return validate(self)
        validate(replace(self, attention="sla"))
        if self.model_variant != "fused_turbo":
            raise ValueError("H3実験はFused Turbo + 4 Stepsに限定しています。")

    def runtime_packs(self):
        return packs(normal(self)) + (PACK,) if self.attention in MODES else packs(self)

    def extra_nodes(self):
        return nodes(normal(self)) | {NODE} if self.attention in MODES else nodes(self)

    def apply_workflow(self, workflow, base_files, mode):
        if self.attention not in MODES:
            return apply(self, workflow, base_files, mode)
        validate_experiment(self)
        if self.attention == "h3_jev":
            cloud_environment()
            python = str(sdk_python(ROOT))
        else:
            python = ""
        # Work on a copy so rejected experiments never leave a half-patched graph.
        import copy

        copy_graph = copy.deepcopy(workflow)
        apply(normal(self), copy_graph, base_files, mode)
        patch_workflow(copy_graph, MODES[self.attention], python)
        workflow.clear()
        workflow.update(copy_graph)

    def validate_node_schema(self, schemas):
        if self.attention not in MODES:
            return validate_nodes(self, schemas)
        validate_nodes(normal(self), schemas)
        schema = schemas.get(NODE, {})
        spec = schema.get("input", {})
        inputs = {**spec.get("required", {}), **spec.get("optional", {})}
        for name, kind in (("model", "MODEL"), ("sdk_python", "STRING"), ("prompt_context", "STRING")):
            if not inputs.get(name) or inputs[name][0] != kind:
                raise ValueError("H3実験ノードの入力仕様が一致しません。専用ノードを再導入してください。")
        values = inputs.get("mode", [()])[0]
        if (
            not isinstance(values, (list, tuple))
            or not set(MODES.values()) <= set(values)
            or tuple(schema.get("output", ())) != ("MODEL",)
        ):
            raise ValueError("H3 experiment node mode/output schema mismatch")

    for name, function in zip(
        required, (validate_experiment, runtime_packs, extra_nodes, apply_workflow, validate_node_schema), strict=True
    ):
        _set(cls, name, function)
    old_request = bridge.validate_request

    def validate_request(request):
        old_request(request)
        if request.acceleration.attention in MODES:
            if request.steps != 4 or isinstance(request.steps, bool):
                raise bridge.H3BridgeError("H3実験は4 Steps限定です。Turbo + 4 Stepsを適用してください。")
            if request.control.enabled:
                raise bridge.H3BridgeError("H3実験ではFun ControlNetをOFFにしてください。")

    _set(bridge, "validate_request", validate_request)
    old_guard = bridge._runtime_arguments_are_allowed

    def guard(arguments):
        stripped = strip_pack(arguments, bridge.custom_node_whitelist)
        return old_guard(arguments if stripped is None else stripped)

    _set(bridge, "_runtime_arguments_are_allowed", guard)
    old_start = bridge._start_runtime_locked
    signature = inspect.signature(old_start)

    @functools.wraps(old_start)
    def start(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        option = bound.arguments.get("acceleration")
        mode = getattr(option, "attention", None)
        if mode not in MODES:
            return old_start(*args, **kwargs)
        verify_runtime(bound.arguments["runtime_root"])
        if mode == "h3_jev":
            cloud_environment()
        token = _START_MODE.set(mode)
        try:
            return old_start(*args, **kwargs)
        finally:
            _START_MODE.reset(token)

    _set(bridge, "_start_runtime_locked", start)
    _set(bridge, "subprocess", _SubprocessProxy(bridge.subprocess))
    old_note, old_state = ui.acceleration_note, ui._control_state

    def note(*values):
        try:
            option = cls.from_values(values)
        except ValueError:
            return old_note(*values)
        if option.attention not in MODES:
            return old_note(*values)
        cloud_note = (
            "Jevは初段の集約統計を1回だけ外部APIへ送り、残りの保持率を選びます。プロンプト・生画像・音声・重みは送信しません。"
            if option.attention == "h3_jev"
            else "この比較モードはクラウド通信しません。"
        )
        return (
            '<div role="status">H3実験：4 Steps / res_multistep限定。'
            + cloud_note
            + " 全50層を実行し、音声・条件の保護はnative SLAに従います。既存のSparse開始位置は使用せず、開始から適用します。12288 tokens未満はdenseです。ログ：outputs/jev-sparse。選択変更後は実行環境を再起動してください。</div>"
        )

    def state(*values):
        result = old_state(*values)
        try:
            if cls.from_values(values).attention in MODES:
                import gradio as gr

                result = (
                    note(*values),
                    result[1],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
        except ValueError:
            pass
        return result

    _set(ui, "acceleration_note", note)
    _set(ui, "_control_state", state)


def after_component(component, **kwargs):
    if getattr(component, "elem_id", None) == "h3-runtime-path":
        component.visible = True
        component.interactive = True
        component.label = "H3実行環境（通常／実験用ComfyUIの絶対パス）"
        component.info = "標準はrepositories/minimax-h3/ComfyUI。Jev用ノードはaikimi-jev-setup.bat --h3で追加できます。環境切替前に実行環境を終了してください。"
    if getattr(component, "elem_id", None) == "h3-attention-mode":
        present = {value if isinstance(value, str) else value[1] for value in component.choices}
        component.choices = list(component.choices) + [choice for choice in CHOICES if choice[1] not in present]


def uninstall():
    while _RESTORES:
        obj, name, old, installed = _RESTORES.pop()
        if getattr(obj, name) is installed:
            setattr(obj, name, old)
