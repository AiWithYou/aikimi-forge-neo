# CI coverage follow-up audit - 2026-09-24

Baseline: `e40a322d46662b21ee11048517e532f5e0c3978a` (`neo`, v1.5.1).
This is a targeted follow-up, not a repeat of the complete system or GPU audit.

## Scope and evidence

Reviewed the contribution policy, current release and prior audit record,
remote-image transport, H3 snapshot/receipt and pending-record implementations,
the CPU runner, unit/security workflows, and the separate Qwen/SenseNova
requirements. Changes are limited to test execution and audit coverage. No
inference algorithm, model setting, public API, installer, package version,
vulnerability exception, or exception expiry is changed.

The local environment could not clone GitHub. Selected files were read through
the GitHub connection at the baseline commit and reconstructed locally. Before
editing, Git blob hashes were verified against GitHub for the runner, both
workflows, and the unmodified H3 pending-record module used in validation.
This workspace was not a full checkout. Findings from the earlier audit are
not presented as tests performed again in this follow-up.

## Primary references

- [Python 3.13 unittest](https://docs.python.org/3.13/library/unittest.html):
  distinguish successful tests, failures, and an execution with no tests or
  skips. Class-level skips must not become failures, and fixture errors must
  remain failures even when no test method ran.
- [GitHub Actions variable scope](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-variables):
  an environment change inside one Python process does not configure a later
  workflow step. Shared policy must be applied in every test process.
- [pip-audit CLI](https://github.com/pypa/pip-audit#usage): `--no-deps` skips
  dependency resolution; `--strict` fails dependency collection errors.
- [Pinned audit-action inputs](https://github.com/pypa/gh-action-pip-audit/blob/1220774d901786e6f652ae159f7b6bc8fea6d266/action.yml):
  checked the supported extra-flags input rather than inventing a `strict`
  action input.
- [Earlier system audit](2026-09-24-system-audit.md): its dependency-coverage
  limitations are the starting point, not a claim of new scan results.

## Corrections

### Empty unittest discovery no longer reports success

The custom runner previously returned zero whenever `wasSuccessful()` was true,
including an empty test suite. The new regression test failed on the exact
baseline with `0 != 5`. The runner now returns 5 when no tests ran or were
skipped. It checks errors first, preserves class-level skips and expected
failures, and still fails unexpected successes.

### pytest uses the same CPU/offline and record-isolation policy

The H3, YuE2 and Jev workflow steps previously invoked `python -m pytest` after
the unittest runner had exited. Those steps did not inherit its Python-local
environment changes or temporary H3 record directory.

`tools/run_ci_tests.py --pytest <pytest arguments>` now applies the existing
CPU/offline/live-test policy and H3 isolation before importing pytest, plugins
or test modules. It preserves pytest exit codes and restores the record path
on normal return and exceptions. Existing `--module`, discovery and `--preload`
behavior remains available; mixed module/pytest mode and empty pytest arguments
are rejected. The three Python contract steps use this entry point. The
JavaScript conversion test remains in place.

The offline policy is the existing library flags and opt-in live-test gates;
it is not an operating-system network sandbox and does not block arbitrary
socket use by every third-party plugin.

### Qwen requirements have an independent strict audit job

The security workflow now includes `tools/requirements-qwen-image21.txt` in a
separate job with dependency resolution enabled. It does not inherit Forge's
vulnerability exceptions, depend on another audit job, or ignore audit failure.
The existing pinned action receives `--strict`, so an uncollectable dependency
must not yield a clean audit result.

This is a requirements-resolution audit, not certification of the actual
Windows/CUDA installation. Qwen includes a Git-pinned Diffusers dependency;
Git/dev or otherwise unmatched distributions can make this job fail. Such a
failure is an unresolved coverage finding, not a reason to silently skip that
package. Actual runtime inventories, separately installed Torch/CUDA wheels,
and Git-source provenance still require runtime-specific auditing. No online
Qwen vulnerability scan was completed locally in this follow-up.

## Verification performed

Environment: Linux, Python 3.13.5; no GPU or model downloads used.

- Baseline reproduction: the empty-suite regression failed (`0 != 5`).
- 18 distinct regression tests passed through the real unittest runner.
- The same 18 tests passed through the real pytest entry point, with six
  additional exit-code subtests. These are not 36 distinct tests.
- The actual H3 pending-record module was used to check restoration and file
  preservation, including exception cleanup. Owned temporary fixtures were
  used; no user job records were accessed.
- Separate CLI processes confirmed empty unittest and empty pytest exit 5;
  an empty `--pytest` option exits 2.
- Python AST parsing, YAML parsing of both changed workflows, and Git whitespace
  checks passed.

Reproduction from a full repository checkout with test requirements installed:

```powershell
python tools/run_ci_tests.py --module tools.tests.test_ci_audit_followup --verbosity 2
python tools/run_ci_tests.py --pytest tools/tests/test_ci_audit_followup.py -q
python tools/run_ci_tests.py --pytest tests/yue2 -q
python tools/run_ci_tests.py --pytest tests/jev_sparse -q
```

## Not verified here

The complete historical CPU suite, H3/YuE2/Jev model contracts beyond the new
runner regressions, live GPU generation, the running WebUI, and online
vulnerability resolution were not re-run locally. Ruff was unavailable, and
its installation attempt failed; no Ruff success is claimed. GitHub Actions
results must be inspected separately before merging. The earlier dependency
findings and unmatched-package limits remain open unless independently
resolved; this change does not mark them fixed.

## Full-checkout integration on Windows (2026-09-24)

The five-file archive patch was applied to the complete `e40a322d` checkout.
Its original Git blobs matched the archive's source hashes. In this checkout,
the existing `test_run_ci_tests.py` fixture returned a successful mock result
without `testsRun` or `skipped`. The new empty-suite check raised
`AttributeError`; the published draft PR's Linux CPU and Windows smoke jobs
failed for the same reason. The fixture now uses a real `unittest.TestResult`
with one completed test. The new regression module is also included in the explicit
Windows smoke selection and Ruff lint list. A CodeQL review comment about
mixing two import forms for `unittest` was addressed in that module.

Local verification used Windows and the existing Python 3.13.14 environment:

- Full CPU/offline unittest suite: 1,676 tests, success (52 skipped, one
  expected failure).
- H3 handoff and Union contracts: 89 passed, 19 subtests passed.
- YuE2 contracts: 44 passed, five skipped. Jev contracts: 245 passed.
- JavaScript H3 conversion contracts: 15 passed.
- Focused runner and legacy-runner tests: 25 passed. Separate CLI invocations
  returned 5 for empty unittest and pytest selections and 2 for `--pytest`
  without arguments.
- Ruff check and format passed for all three changed Python files. All four
  changed workflows parsed as YAML; actionlint 1.7.12, six existing CI-boundary
  tests and `git diff --check` passed.

The original [PR #9](https://github.com/AiWithYou/aikimi-forge-neo/pull/9) checks
at `b1484f9` also reported dependency review blocked by unavailable Dependency
Graph support. During integration, the repository's Dependency Graph was
enabled and its SBOM API returned 74 packages, resolving that prerequisite.

The dependency findings remain open: Qwen's Git/dev Diffusers version cannot
be matched by PyPI; Forge reports an Accelerate advisory; SenseNova reports 11
advisory rows across three packages, with CUDA-local Torch wheels unmatched.
PyPI was rechecked on 2026-09-24: the latest Accelerate is 1.15.0, diskcache
5.6.3, and Diffusers 0.40.0. The existing [compatibility assessment](../security-model.md)
documents why these versions do not provide a verified runtime-compatible
remediation. Accelerate's [proposed fix](https://github.com/huggingface/accelerate/pull/4138)
is closed without merging. SenseNova requires a separately validated
Transformers migration; Qwen requires its pinned development code. This
integration does not add a vulnerability exclusion or claim that the dependency
audits passed.
