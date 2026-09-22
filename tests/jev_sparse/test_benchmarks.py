"""Offline matrix, provenance and completed-generation checks; no model or cloud."""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules_forge.jev_sparse import common
from modules_forge.jev_sparse.qwen21 import Options
from tools import benchmark_krea2_sparse as krea
from tools import benchmark_qwen21_sparse as qwen
from tools import sparse_benchmark as bench


def image_file(path, mode="RGB", size=(256, 256)):
    Image.new(mode, size).save(path)
    return path


def log_file(path, *, mode="fixed", replay=False, fallback=False, sparse=True):
    records = [{"event": "begin"}]
    if mode == "jev":
        records.append(
            {
                "event": "decision",
                "source": "dense_fallback" if fallback else "jev",
                "diagnostics": {"0": {"confidence": 0.9}},
            }
        )
    records.append(
        {
            "event": "end",
            "status": "completed",
            "attention_calls": {"sparse": int(sparse), "sparse_target": int(sparse)},
            "api_calls": int(mode == "jev" and not replay),
            "replay_calls": int(replay),
        }
    )
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return path


def test_matrix_distinct_keeps_seeds_order_and_skips(tmp_path):
    args = qwen.parser().parse_args(["--output", str(tmp_path), "--modes", "off", "fixed", "rules"])
    plan = qwen.make_plan(args)
    assert len(plan["groups"]) == 8  # four runnable subjects, two seeds
    assert plan["skipped_cases"][0]["case_id"] == "two-references"
    schedule = plan["groups"][0]["schedule"]
    assert schedule[0]["repetition"] == 0
    forward = [row["variant"] for row in schedule if row["repetition"] == 1]
    reverse = [row["variant"] for row in schedule if row["repetition"] == 2]
    assert forward == ["off", "fixed-25", "fixed-50", "fixed-75", "rules"]
    assert reverse == list(reversed(forward))
    assert len({group["identity"]["seed"] for group in plan["groups"]}) == 2


def test_manifest_resolves_multiple_inputs_and_rgba(tmp_path):
    first = image_file(tmp_path / "first.png")
    second = image_file(tmp_path / "second.png")
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "cases": [
                    {
                        "id": "edit",
                        "category": "multi_reference",
                        "prompt": "Combine them",
                        "input_images": [first.name, second.name],
                    },
                    {"id": "alpha", "category": "rgba", "prompt": "Glass", "transparent": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = bench.load_cases(manifest)
    assert cases[0].input_images == (str(first), str(second))
    assert bench.case_skip(cases[0], "qwen21", "native") is None
    assert "adapter" in bench.case_skip(cases[0], "krea2", "native")
    assert "RGBA" in bench.case_skip(cases[1], "krea2", "native")


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "../escape", "category": "text", "prompt": "x"},
        {"id": "alpha", "category": "rgba", "prompt": "x"},
        {"id": "edit", "category": "multi_reference", "prompt": "x", "input_images": "one.png"},
    ],
)
def test_invalid_manifest_is_rejected(tmp_path, entry):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"schema": 1, "cases": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError):
        bench.load_cases(path)


@pytest.mark.parametrize("module", [qwen, krea])
def test_dry_run_does_not_load_model_or_contact_api(tmp_path, monkeypatch, module):
    # Imports into these modules are lazy. A torch sentinel and forbidden API
    # method make accidental generation during planning a hard failure.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(krea.KreaRunner, "api_request", lambda *_: pytest.fail("dry-run contacted Forge"))
    module.main(["--output", str(tmp_path / "out"), "--dry-run"])
    report = json.loads((tmp_path / "out/benchmark.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry_run" and report["runs"] == []
    assert report["planned_live_generations"] > 0


def test_replay_requires_full_input_identity_and_completed_logs(tmp_path):
    path = log_file(tmp_path / "live.jsonl", mode="jev")
    identity = {"seed": 4, "prompt_sha256": "prompt", "input_sha256": ["image"], "settings": {"steps": 20}}
    row = {
        "status": "completed",
        "mode": "jev",
        "variant": "jev",
        "keep": 25,
        "identity": identity,
        "label": "live",
        "logs": {"attention": {"log_path": str(path)}},
    }
    index = bench.ReplayIndex()
    index.add(row)
    spec = {"replay_of": "jev", "keep": 25}
    assert index.lookup({"identity": identity}, spec)["logs"] == [str(path)]
    for field, value in (("seed", 5), ("input_sha256", ["changed"]), ("settings", {"steps": 21})):
        changed = {**identity, field: value}
        with pytest.raises(RuntimeError, match="matches"):
            index.lookup({"identity": changed}, spec)
    row["status"] = "failed"
    with pytest.raises(RuntimeError, match="matches"):
        index.lookup({"identity": identity}, spec)


def test_summary_excludes_warmup_and_failure():
    base = {"group_id": "one", "status": "completed", "repetition": 1}
    rows = [
        {**base, "variant": "off", "wall_seconds": 20},
        {**base, "variant": "fixed-25", "wall_seconds": 10},
        {**base, "variant": "fixed-25", "wall_seconds": 1000, "repetition": 0},
        {**base, "variant": "fixed-25", "wall_seconds": 0.01, "status": "failed"},
    ]
    summary = bench.summarize(rows)
    assert summary["measured_samples"] == 2
    assert summary["paired_speedups"]["fixed-25"]["median_speedup_vs_off"] == 2


def test_replay_pixel_identity_ignores_png_metadata(tmp_path):
    plan = bench.build_plan(
        model="qwen21",
        stage="native",
        cases=[bench.Case("pixels", "text", "same image")],
        seeds=[1],
        dimensions=[(2, 1)],
        variants=bench.build_variants(["jev", "replay"], [25], 25),
        repeats=1,
        settings={},
    )

    def generate(group, spec, directory, replay):
        output = directory / "output.png"
        metadata = PngInfo()
        metadata.add_text("generation_mode", spec["mode"])
        with Image.new("RGBA", (2, 1), (19, 43, 71, 0)) as image:
            image.putpixel((1, 0), (11, 29, 53, 255))
            image.save(output, pnginfo=metadata)
        log = log_file(directory / "run.jsonl", mode="jev")
        return {
            "wall_seconds": 1.0,
            "output_path": str(output),
            "artifact": bench.validate_output(output),
            "logs": {"attention": {"log_path": str(log)}},
        }

    runner = SimpleNamespace(prepare=lambda _: None, generate=generate)
    report = bench.execute_plan(plan, tmp_path, runner)
    live, replay = report["runs"][1:]
    assert live["artifact"]["sha256"] != replay["artifact"]["sha256"]
    assert live["artifact"]["pixels"] == replay["artifact"]["pixels"]
    assert replay["replay_pixel_identical"] is True
    assert replay["replay_file_identical"] is False
    pixels = replay["artifact"]["pixels"]
    assert (pixels["mode"], pixels["width"], pixels["height"]) == ("RGBA", 2, 1)

    # Fully transparent RGB values are still part of the exact RGBA contract.
    with Image.open(replay["output_path"]) as image:
        image.putpixel((0, 0), (20, 43, 71, 0))
        changed = tmp_path / "changed.png"
        image.save(changed)
    assert bench.validate_output(changed)["pixels"]["sha256"] != pixels["sha256"]


def test_failed_generation_is_written_without_success_statistics(tmp_path):
    args = qwen.parser().parse_args(
        ["--output", str(tmp_path), "--cases", "lettering", "--seed", "1", "--modes", "off"]
    )
    plan = qwen.make_plan(args)
    runner = SimpleNamespace(
        prepare=lambda _: None, generate=lambda *_: (_ for _ in ()).throw(RuntimeError("incomplete"))
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        bench.execute_plan(plan, tmp_path, runner)
    report = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert report["status"] == report["runs"][0]["status"] == "failed"
    assert report["summary"]["measured_samples"] == 0


@pytest.mark.parametrize(
    "replay,fallback,sparse", [(False, False, True), (True, False, True), (False, True, True), (False, False, False)]
)
def test_qwen_execution_requires_real_sparse_and_valid_controller(tmp_path, monkeypatch, replay, fallback, sparse):
    args = qwen.parser().parse_args(
        ["--output", str(tmp_path), "--width", "256", "--height", "256", "--cases", "lettering", "--modes", "off"]
    )
    group = qwen.make_plan(args)["groups"][0]
    directory = tmp_path / "run"
    directory.mkdir()
    calls = []

    def resident(payload):
        calls.append((copy.deepcopy(payload), os.environ.get(qwen.REPLAY_ENV)))
        output = image_file(directory / "output.png")
        log = log_file(directory / "run.jsonl", mode="jev", replay=replay, fallback=fallback, sparse=sparse)
        return {
            "output_path": str(output),
            "metadata": {
                "sparse_experiment": {"status": "completed", "log_path": str(log), "cuda_event_seconds": 1.0},
                "timings": {"sampling_seconds": 2},
                "memory": {},
                "reused_model": True,
            },
        }

    monkeypatch.setenv(qwen.REPLAY_ENV, "original_environment")
    runner = qwen.QwenRunner(
        args,
        SimpleNamespace(resident_run=resident),
        Options,
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    )
    spec = {"mode": "replay" if replay else "jev", "keep": 25}
    replay_source = {"logs": ["recorded.jsonl"]} if replay else None
    if fallback or not sparse:
        with pytest.raises(RuntimeError, match="fell back|No real sparse"):
            runner.generate(group, spec, directory, replay_source)
    else:
        row = runner.generate(group, spec, directory, replay_source)
        assert row["artifact"]["width"] == 256 and row["wall_seconds"] >= 0
        assert row["timing_breakdown"]["cuda_event_seconds"] == 1.0
    assert os.environ[qwen.REPLAY_ENV] == "original_environment"
    assert calls[0][0]["sparse_experiment"]["mode"] == "jev"
    assert calls[0][1] == (json.dumps(["recorded.jsonl"]) if replay else None)


@pytest.mark.parametrize("fallback", [False, True])
def test_krea_api_execution_validates_output_and_jev_fallback(tmp_path, fallback):
    args = krea.parser().parse_args(
        ["--output", str(tmp_path), "--sizes", "256", "--cases", "lettering", "--seed", "1", "--modes", "off"]
    )
    group = krea.make_plan(args)["groups"][0]
    directory = tmp_path / "run"
    directory.mkdir()
    log = log_file(tmp_path / "attention.jsonl", mode="jev", fallback=fallback)
    png = io.BytesIO()
    Image.new("RGB", (256, 256)).save(png, format="PNG")
    requests = []

    def request(endpoint, body):
        requests.append((endpoint, body))
        return {
            "images": [base64.b64encode(png.getvalue()).decode("ascii")],
            "info": json.dumps(
                {
                    "extra_generation_params": {"Krea2 Sparse status": "active", "Krea2 Sparse log": str(log)},
                }
            ),
        }

    runner = krea.KreaRunner(args, tmp_path, request)
    runner.bodies[group["id"]] = {"alwayson_scripts": {krea.SCRIPT: {"args": []}}}
    if fallback:
        with pytest.raises(RuntimeError, match="fell back"):
            runner.generate(group, {"mode": "jev", "keep": 10}, directory, None)
    else:
        result = runner.generate(group, {"mode": "jev", "keep": 10}, directory, None)
        assert Path(result["output_path"]).is_file()
    assert len(requests[0][1]["alwayson_scripts"][krea.SCRIPT]["args"]) == 11


def test_invalid_or_non_rgba_artifacts_do_not_pass(tmp_path):
    path = image_file(tmp_path / "rgb.png")
    with pytest.raises(RuntimeError, match="RGBA"):
        bench.validate_output(path, transparent=True)
    with pytest.raises(RuntimeError, match="dimensions"):
        bench.validate_output(path, expected_size=(512, 512))
    path.write_bytes(b"incomplete")
    with pytest.raises(OSError):
        bench.validate_output(path)


@pytest.mark.parametrize("module", [qwen, krea])
def test_legacy_keep_remains_a_single_fixed_variant(tmp_path, module):
    args = module.parser().parse_args(["--output", str(tmp_path), "--modes", "off", "fixed", "--keep", "42"])
    assert [v["key"] for v in module.make_plan(args)["variants"]] == ["off", "fixed-42"]


def test_full_qwen_schedule_uses_real_offline_replay_factory(tmp_path, monkeypatch):
    args = qwen.parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--width",
            "256",
            "--height",
            "256",
            "--steps",
            "2",
            "--cases",
            "lettering",
            "--seed",
            "1",
            "--modes",
            "off",
            "jev",
            "replay",
        ]
    )
    plan = qwen.make_plan(args)
    state, allowed = {"evaluation": 1, "blocks": {"0": {"relative_output_norm": 1.0}}}, {"0": (25.0, 50.0, 100.0)}
    answer = {"decisions": {"0": {"choice": 25.0, "confidence": 0.9}}}
    monkeypatch.delenv(qwen.REPLAY_ENV, raising=False)
    monkeypatch.delenv("AIKIMI_JEV_ALLOW_CLOUD", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    def resident(payload):
        directory = Path(payload["job_dir"])
        output = image_file(directory / "output.png")
        mode = payload["sparse_experiment"]["mode"]
        sparse = {"status": "off"}
        if mode == "jev":
            replay = bool(os.environ.get(qwen.REPLAY_ENV))
            log = common.RunLog(directory, "qwen21", {})
            if replay:
                client = common.create_client()
                assert client.decide(state, allowed, 100) == {"0": 25.0}
                client.close()
                replay_record = client.last_replay
            else:
                replay_record = {
                    "schema": 1,
                    "job_id": log.id,
                    "sequence": 1,
                    "request": common.request_contract(state, allowed),
                    "answer": answer,
                    "budget": {"max_calls": args.job_max_calls, "max_wait_seconds": args.job_max_wait_seconds},
                    "api_wait_seconds": 0.01,
                }
            log.write(
                "decision",
                source="replay" if replay else "jev",
                diagnostics={"0": {"confidence": 0.9}},
                replay=replay_record,
            )
            log.finish(
                "completed",
                attention_calls={"sparse_target": 1},
                api_calls=0 if replay else 1,
                replay_calls=int(replay),
            )
            sparse = {"status": "completed", "log_path": str(log.path)}
        return {
            "output_path": str(output),
            "metadata": {"sparse_experiment": sparse, "timings": {}, "memory": {}, "reused_model": True},
        }

    runner = qwen.QwenRunner(
        args,
        SimpleNamespace(resident_run=resident),
        Options,
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    )
    report = bench.execute_plan(plan, tmp_path, runner)
    assert report["status"] == "completed"
    assert report["summary"]["measured_samples"] == 6
    replays = [row for row in report["runs"] if row["mode"] == "replay"]
    assert len(replays) == 2 and all(row["replay_pixel_identical"] for row in replays)
    assert all(row["logs"]["attention"]["end"]["api_calls"] == 0 for row in replays)
    assert [row["mode"] for row in report["runs"]] == ["off", "off", "jev", "replay", "replay", "jev", "off"]
