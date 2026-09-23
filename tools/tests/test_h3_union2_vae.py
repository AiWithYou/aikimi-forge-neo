"""Offline CPU contracts. Synthetic headers and stub runtime, no model inference."""

import copy
import io
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge import minimax_h3_union2_vae as feature


def header_fixture(precision="bf16"):
    """Shapes follow the reviewed native loader; no tensor data is allocated."""
    h = {
        "__metadata__": {
            "minimax_h3_fun_controlnet": "adaln_basis",
            "inpaint_masked_pixel_mode": "post_norm",
            "control_blocks_places": json.dumps(list(range(0, 50, 5))),
        }
    }
    cursor = 0

    def add(name, shape, dtype="BF16"):
        nonlocal cursor
        import math

        size = math.prod(shape) * feature._DTYPE_BITS[dtype] // 8
        h[name] = {"shape": shape, "dtype": dtype, "data_offsets": [cursor, cursor + size]}
        cursor += size

    add("control_proj_in.weight", [5376, 196], "F32")
    for i in range(10):
        prefix = f"control_blocks.{i}."
        add(prefix + "adaln_proj.linear.weight", [5376 * 12, 8], "F32")
        add(prefix + "after_proj.weight", [5376, 5376])
        add(prefix + "attn.qkv_proj.weight", [56 * 128 * 3, 5376], "I8" if precision == "int8" else "BF16")
        add(prefix + "attn.q_norm.weight", [128])
        add(prefix + "mlp.fc1.weight", [14336 * 2, 5376])
        add(prefix + "mlp.fc2.weight", [5376, 14336])
        if precision == "int8":
            add(prefix + "attn.qkv_proj.comfy_quant", [2], "U8")
    add("control_blocks.0.before_proj.weight", [5376, 5376])
    return h


def reindex(header):
    import math

    cursor = 0
    for name, item in header.items():
        if name == "__metadata__":
            continue
        size = math.prod(item["shape"]) * feature._DTYPE_BITS[item["dtype"]] // 8
        item["data_offsets"] = [cursor, cursor + size]
        cursor += size
    return header


@pytest.mark.parametrize("precision", ["bf16", "int8"])
def test_native_ten_block_shapes(precision):
    assert feature.validate_union2_header(header_fixture(precision)) == precision


@pytest.mark.parametrize(
    "kind",
    [
        "raw",
        "v1",
        "missing9",
        "postnorm",
        "full_adaln",
        "wrong_layers",
        "wrong_projection",
        "wrong_qkv",
        "quant_missing",
    ],
)
def test_incompatible_weights_fail_closed(kind):
    h = header_fixture("int8" if kind == "quant_missing" else "bf16")
    if kind == "raw":
        h["__metadata__"].pop("minimax_h3_fun_controlnet")
    elif kind == "v1":
        h = {k: v for k, v in h.items() if not any(k.startswith(f"control_blocks.{i}.") for i in range(5, 10))}
    elif kind == "missing9":
        h.pop("control_blocks.9.after_proj.weight")
    elif kind == "postnorm":
        h["__metadata__"]["inpaint_masked_pixel_mode"] = "pre_norm"
    elif kind == "full_adaln":
        h["control_blocks.0.adaln_proj.linear.weight"]["shape"][1] = 2688
    elif kind == "wrong_layers":
        h["__metadata__"]["control_blocks_places"] = json.dumps(list(range(0, 100, 10)))
    elif kind == "wrong_projection":
        h["control_proj_in.weight"]["shape"][1] = 48 * 4
    elif kind == "wrong_qkv":
        h["control_blocks.8.attn.qkv_proj.weight"]["shape"][0] = 1
    else:
        h = {k: v for k, v in h.items() if not k.endswith(".comfy_quant")}
    with pytest.raises(ValueError):
        feature.validate_union2_header(reindex(h))


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        struct.pack("<Q", 1) + b"1",
        struct.pack("<Q", feature.MAX_HEADER_BYTES + 1),
        struct.pack("<Q", 20) + b"{}",
        struct.pack("<Q", 13) + b'{"x":1,"x":2}',
        struct.pack("<Q", 2) + b"[]",
        struct.pack("<Q", 2) + b"\xff\xff",
    ],
)
def test_malformed_headers(raw):
    with pytest.raises(ValueError):
        feature.read_header(io.BytesIO(raw))


def test_small_header_table_and_bounds(tmp_path):
    h = {"value": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}}
    raw = json.dumps(h).encode()
    decoded, start = feature.read_header(io.BytesIO(struct.pack("<Q", len(raw)) + raw + b"abc"))
    assert decoded == h and start == len(raw) + 8
    feature.validate_tensor_table(h, 3)
    with pytest.raises(ValueError):
        feature.validate_tensor_table(h, 4)
    for field, value in [("data_offsets", [1, 4]), ("shape", [True]), ("dtype", "PICKLE")]:
        invalid = copy.deepcopy(h)
        invalid["value"][field] = value
        with pytest.raises(ValueError):
            feature.validate_tensor_table(invalid)


@pytest.mark.parametrize(
    "args",
    [
        ("--fast",),
        ("--fast=fp16_accumulation",),
        ("--fast", "fp8_matrix_mult"),
        ("--fast", "fp16_accumulation", "fp8_matrix_mult"),
        ("--fast", "fp16_accumulation", "--fast", "fp16_accumulation"),
    ],
)
def test_dangerous_fast_arguments_rejected(args):
    with pytest.raises(ValueError):
        feature.strip_selected_fast(args)


def test_explicit_flag_stripped_without_weakening_others():
    args = ("main.py", "--listen", "127.0.0.1", "--fast", "fp16_accumulation", "--disable-api-nodes")
    assert feature.strip_selected_fast(args) == ("main.py", "--listen", "127.0.0.1", "--disable-api-nodes")
    assert feature.has_fp16_accumulation(args)
    assert not feature.has_fp16_accumulation(("--cache-none",))


def ready(tmp_path, flags=()):
    return SimpleNamespace(
        runtime_root=tmp_path / "ComfyUI",
        runtime_args=flags,
        connected=True,
        package_versions={"comfy-kitchen": "0.2.35", "comfy-aimdo": "0.5.5"},
    )


def test_legacy_does_not_require_upgrade(tmp_path):
    ancestor = Mock(side_effect=AssertionError("must not query new core"))
    feature.check_runtime(ready(tmp_path), ancestor=ancestor)
    ancestor.assert_not_called()


def test_v2_requires_native_commit(tmp_path):
    check = Mock(return_value=True)
    feature.check_runtime(ready(tmp_path), union2=True, ancestor=check)
    check.assert_called_once_with(feature.UNION2_COMMIT)
    with pytest.raises(ValueError):
        feature.check_runtime(ready(tmp_path), union2=True, ancestor=lambda _: False)


def test_fp16_requires_fixed_vae_and_matching_explicit_flag(tmp_path):
    flag = ("--fast", "fp16_accumulation")
    check = Mock(return_value=True)
    feature.check_runtime(ready(tmp_path, flag), decode_mode="fp16_accumulation", ancestor=check)
    check.assert_called_once_with(feature.VAE_FIXED_COMMIT)
    with pytest.raises(ValueError):
        feature.check_runtime(ready(tmp_path), decode_mode="fp16_accumulation", ancestor=check)
    with pytest.raises(ValueError):
        feature.check_runtime(ready(tmp_path, flag), ancestor=check)


@pytest.mark.parametrize(
    "package,value", [("comfy-kitchen", "0.2.33"), ("comfy-aimdo", "0.5.3"), ("comfy-kitchen", "0.2.35.dev1")]
)
def test_dependency_floor_and_prerelease_rejection(tmp_path, package, value):
    r = ready(tmp_path, ("--fast", "fp16_accumulation"))
    r.package_versions[package] = value
    with pytest.raises(ValueError):
        feature.check_runtime(r, decode_mode="fp16_accumulation", ancestor=lambda _: True)


def activate(tmp_path):
    python = tmp_path / ".venv-union2-vae/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (tmp_path / "union2-vae-runtime.json").write_text(
        json.dumps({"schema_version": 1, "revision": feature.VAE_FIXED_COMMIT})
    )
    return python


def test_upgraded_standard_also_validates_core(tmp_path):
    python = activate(tmp_path)
    assert feature.preferred_python(tmp_path / "ComfyUI") == python
    check = Mock(return_value=True)
    feature.check_runtime(ready(tmp_path), ancestor=check)
    check.assert_called_once_with(feature.VAE_FIXED_COMMIT)


def test_pending_update_blocks_launch(tmp_path):
    activate(tmp_path)
    (tmp_path / "union2-vae-update-in-progress.json").write_text("{}")
    with pytest.raises(ValueError):
        feature.preferred_python(tmp_path / "ComfyUI")


@pytest.mark.parametrize("content", ["{}", "[]", "invalid", json.dumps({"schema_version": 1, "revision": "bad"})])
def test_invalid_activation_receipt(tmp_path, content):
    (tmp_path / "union2-vae-runtime.json").write_text(content)
    with pytest.raises(ValueError):
        feature.preferred_python(tmp_path / "ComfyUI")


def test_missing_environment_is_not_silently_old(tmp_path):
    (tmp_path / "union2-vae-runtime.json").write_text(
        json.dumps({"schema_version": 1, "revision": feature.VAE_FIXED_COMMIT})
    )
    with pytest.raises(ValueError):
        feature.preferred_python(tmp_path / "ComfyUI")


def test_failed_connection_not_masked_by_version_check(tmp_path):
    r = ready(tmp_path)
    r.connected = False
    feature.check_runtime(r, decode_mode="fp16_accumulation", ancestor=lambda _: False)


def test_no_duplicate_native_decode_choice():
    assert [value for _, value in feature.DECODE_CHOICES] == ["standard", "fp16_accumulation", "fast"]
