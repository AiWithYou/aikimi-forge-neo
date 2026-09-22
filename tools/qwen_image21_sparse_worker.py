"""Experimental entrypoint for the existing, isolated Qwen 2.1 resident worker.

The normal loader/INT8 offload fix/annotations/RGBA writer remain upstream.
Only this dedicated process decorates the original worker's runtime lookup.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_base():
    name = "_aikimi_qwen21_base_worker"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/qwen_image21_worker.py")
    if spec is None or spec.loader is None:
        raise ImportError("The original Qwen 2.1 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_base()
from modules_forge.jev_sparse.common import ExperimentCancelled
from modules_forge.jev_sparse.qwen21 import OPTIONS_ENV, REVISION, Options, experiment


class PipelineProxy:
    def __init__(self, pipe, options, job, request):
        self.pipe, self.options, self.job, self.request = pipe, options, job, request
        self.run = None

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get("use_kv_cache", True) is not True or kwargs.get("true_cfg_scale", 1.0) != 1.0:
            raise ValueError("Qwen sparse benchmark requires prefix KV caching and true_cfg_scale=1")

        def cancelled():
            return (self.job / "cancel").exists()

        digests = []
        for value in self.request.get("input_images", []):
            with Path(value).open("rb") as stream:
                digests.append(hashlib.file_digest(stream, "sha256").hexdigest())
        identity = {key: self.request[key] for key in ("seed", "width", "height", "steps", "precision", "memory_mode")}
        identity["input_sha256"] = digests
        with experiment(
            self.pipe.transformer,
            self.options,
            self.job / "jev-sparse",
            prompt=self.request["prompt"],
            cancelled=cancelled,
            replay_identity=identity,
        ) as run:
            self.run = run
            if run is not None:
                run.log.write("generation_context", **identity)
            return self.pipe(*args, **kwargs)


def resident_run(payload):
    options = Options.parse(payload.get("sparse_experiment", os.environ.get(OPTIONS_ENV, "{}")))
    if base.DIFFUSERS_REVISION != REVISION:
        raise RuntimeError("Unreviewed Diffusers revision for Qwen 2.1 sparse adapter")
    job = Path(payload["job_dir"]).resolve()
    original_read, original_lookup = base._read_request, base._runtime_for_request
    proxies = []

    def read_request(value):
        j, model, request = original_read(value)
        if "sparse_experiment" in request and request["sparse_experiment"] != asdict(options):
            raise ValueError("Persisted Qwen sparse settings differ from this launch")
        request["sparse_experiment"] = asdict(options)
        base._atomic_json(j / "request.json", request)
        return j, model, request

    def lookup(model, request, directory):
        runtime, reused = original_lookup(model, request, directory)
        if options.mode == "off":
            return runtime, reused
        proxy = PipelineProxy(runtime["pipe"], options, directory, request)
        proxies.append(proxy)
        return {**runtime, "pipe": proxy}, reused

    base._read_request, base._runtime_for_request = read_request, lookup
    try:
        result = base.run_request(payload)
        report = (
            proxies[-1].run.summary if proxies and proxies[-1].run else {"status": "off", "settings": asdict(options)}
        )
        result["metadata"]["sparse_experiment"] = report
        base._atomic_json(job / "metadata.json", result["metadata"])
        base._atomic_json(job / "result.json", result)
        return result
    except BaseException as exc:
        base.clear_runtime()
        (job / "result.json").unlink(missing_ok=True)
        if isinstance(exc, ExperimentCancelled):
            raise base.GenerationCancelled("Qwen generation cancelled") from None
        raise
    finally:
        base._read_request, base._runtime_for_request = original_read, original_lookup
        # Restored processors no longer hold references to previous runs.
        proxies.clear()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen 2.1 experimental worker")
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    try:
        resident_run(json.loads(Path(args.request).read_text(encoding="utf-8")))
    finally:
        base.clear_runtime()


if __name__ == "__main__":
    main()
