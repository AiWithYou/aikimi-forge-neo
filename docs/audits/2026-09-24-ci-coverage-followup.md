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
