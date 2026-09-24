# Dependency remediation — 2026-09-24

## Scope and changes

This follow-up addresses the three failing security jobs on `32f84536`:
Forge, SenseNova and Qwen. It also removes the existing diskcache/setuptools
exception baseline rather than extending its expiration.

- All three environments: CUDA 13.0 PyTorch 2.13.0 / torchvision 0.28.0,
  setuptools 83.0.0, locally patched Accelerate 1.15.0+aikimi.1.
- Accelerate: official source distribution, preserved Apache license and
  provenance, checkpoint shard validation at the actual load site. Public
  loading APIs and legitimate snapshot symlinks remain usable.
- Forge: replace diskcache with SQLite/JSON metadata storage. Old pickle
  caches are not deserialized. Models, settings and outputs are retained.
- SenseNova: Transformers 5.10.4 / Hub 1.5.0, with scoped configuration,
  rotary-buffer initialization and causal-mask compatibility. Explicit cache
  positions, padding, input IDs, embeddings and KV cache are preserved.
- Qwen: retain the exact required Diffusers Git commit. OSV strict mode can
  audit this development version and CUDA local versions without skipping
  packages. Dependency resolution remains enabled in all three audit jobs.
- Forge's audit manifest now includes the CUDA wheels selected by its launcher.

## Local results

Python 3.13.14, Windows 11, RTX 3090. No vulnerability IDs are ignored.

| Check | Result |
|---|---|
| Installed Forge environment, strict OSV | 164 packages, 0 findings, 0 skips |
| Installed SenseNova environment, strict OSV | 44 packages, 0 findings, 0 skips |
| Installed Qwen environment, strict OSV | 49 packages, 0 findings, 0 skips |
| `pip check`, each of the three environments | Passed |
| CPU offline suite | 1,688 tests; 52 skipped; 1 expected failure |
| H3 / YuE2 / Jev contracts | 378 passed, 5 skipped, 19 subtests passed |
| Vendor secret scan | No findings |
| Workflow syntax / changed-code lint | Passed |

The real Accelerate APIs are tested with tiny safetensors: parent traversal,
absolute paths, Windows drives/UNC/alternate streams and non-files are rejected
before model weights change. Nested relative paths and snapshot symlinks load
the expected weights. The cache tests cover persistence, tuple identities,
concurrent writers, unsupported Python objects and untouched legacy caches.

The actual pinned SenseNova backbone was compared against a saved 4.57.6 /
PyTorch 2.11 baseline with identical tiny-model weights. Input IDs, cached
prefix and image outputs matched exactly. A cached next-token output differed
by at most `3.5762786865234375e-07` after the PyTorch upgrade. These are CPU
float32 comparisons, not a claim that full GPU images are bit-identical.

## GPU and shutdown results

- SenseNova: the real final INT8 ConvRot checkpoint loaded all 588 layers and
  generated a nonuniform 512×512 PNG from two reference images, seed 42,
  one quality-profile step, SDPA, bfloat16, low-VRAM mode. No CUDA OOMs or
  allocation retries. Load: 13.470 s; sampling: 497.143 s. This is a functional
  smoke test, not a speed comparison or a full quality evaluation.
- Output SHA-256:
  `2780d7bd65bdeddd251b364d3c79fbf1aeaf8d3650c9a0697fbb788e90098d18`.
- Qwen: both native CUDA/Accelerate offload and actual tiny QwenImage21 /
  Qwen3VL component stream-load/forward tests passed, including repeated
  forwards without stale scales.
- The GPU smoke exposed a shutdown ordering bug: TemporaryDirectory's
  weakref finalizer could remove `startup.log` before the Windows worker
  released its handle. ResidentWorker now owns deletion explicitly, after
  confirmed process exit. A real-process regression invokes the weakref
  finalizer while the worker is alive, verifies the directory survives, and
  verifies close removes it. All 34 residency/service tests pass.

## GitHub results

The [security workflow on the remediation commit](https://github.com/AiWithYou/aikimi-forge-neo/actions/runs/35976917360)
passed all seven jobs: Forge, SenseNova, Qwen, full-history Gitleaks and the
three toolchain audits. There are no vulnerability exclusions. Lint and
persistence/streaming reliability also passed on that commit. Final-head
checks are verified separately after the shutdown follow-up is pushed.

## Reproduction

```powershell
venv/Scripts/python.exe tools/run_ci_tests.py --verbosity 1
venv/Scripts/python.exe -m unittest tools.tests.test_dependency_remediation -v
uv tool run --python 3.13 pip-audit==2.10.1 --path venv/Lib/site-packages --vulnerability-service osv --strict
```

Repeat the last command for each worker's `Lib/site-packages` directory.
The checked-in CI resolves each complete requirements manifest independently.
No model files, environment directories, audit caches or generated images are
included in the commit. The optional Japanese style-guard path referenced by
CONTRIBUTING.md was absent; the changed Japanese documentation was reviewed
manually instead of claiming that tool had run.

## Sources

- [Accelerate report](https://github.com/huggingface/accelerate/issues/4067)
- [Upstream patch proposal](https://github.com/huggingface/accelerate/pull/4214)
- [Local source provenance and license](../../vendor/accelerate/AIKIMI-PATCH.md)
- [pip-audit's supported services](https://github.com/pypa/pip-audit)
- [OSV API](https://google.github.io/osv.dev/api/)

Checked on 2026-09-24. A clean advisory lookup alone is not proof that the
unreleased Accelerate patch works; the behavioral regressions are required.
