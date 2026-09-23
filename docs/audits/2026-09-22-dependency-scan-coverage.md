# Dependency scanning coverage audit - 2026-09-22

## Baseline and observed failure

Baseline: `a8ba1e2ab48e4cea34aa8639661a4c40a952881b` on `neo`.
This change is independent of the GPU queue correction in PR #7.

In [security run 35710355496](https://github.com/AiWithYou/aikimi-forge-neo/actions/runs/35710355496),
the main dependency audit failed on the existing Accelerate 1.14.0 requirement.
The subsequent asset, browser-test, and preprocessor audits were all skipped.
Their absence is not evidence that those environments are vulnerability-free.
SenseNova was already in a separate job and did execute, reporting dependency
findings. The full-history Gitleaks job completed successfully.

## Knowledge applied and correction

[GitHub's workflow syntax documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstrategyfail-fast)
distinguishes matrix cancellation (`fail-fast`) from suppressing the effect of a
failed check (`continue-on-error`). The former is disabled here; the latter is
not introduced.

Move the three existing toolchain audits into an independent matrix job with
`fail-fast: false` and no dependency on the main runtime job. Each environment
gets a clean Python 3.13 runner, the same pinned Actions and pip version, and the
same requirement file and dependency resolution as before. One failure no
longer prevents the other environments from reporting their own results.

The existing `pip-audit` job name, requirement pins, temporary baseline, expiry
of 2026-09-30, SenseNova job, Gitleaks controls, read-only permissions, and triggers
are preserved. No finding is newly ignored. Any audit failure continues to fail
the security workflow. The tradeoff is three separate runner setups instead of
three sequential scans sharing one job.

## Verification

The retrieved original security workflow was checked against Git blob
`40f21dce9e93c5daf4de2b4e551afd9a0d6ae786` before modification. Local YAML parsing
and structural assertions verified that the unchanged jobs and audit baseline
are identical, the same three requirement files are covered exactly once,
there is no `needs` link to a failing runtime job, matrix fail-fast is disabled,
and no `continue-on-error` or new advisory exclusions exist.

Live Actions execution and workflow lint results should be read from this PR's
checks. A successful audit-coverage correction does not mean all dependencies
have passed their security checks.

## Findings requiring separate remediation

- The main scan reported `PYSEC-2026-3804` in Accelerate 1.14.0.
  [The published Python advisory](https://osv.dev/vulnerability/PYSEC-2026-3804)
  describes checkpoint-index path validation concerns. A version number outside
  a scanner's affected range is not sufficient proof of a fix: inspect the
  upstream implementation and test the actual loading boundary before adoption.
- The isolated SenseNova scan reported Transformers 4.57.6, Accelerate 1.14.0,
  and setuptools 81.0.0 findings. Some rows repeat advisory IDs, so the scanner's
  row count must not be presented as a count of unique vulnerabilities.
  Runtime reachability and compatibility need separate validation; this CI
  correction does not change that worker's pinned runtime or suppress findings.
- The dependency-review Action failed because the repository's dependency graph
  is not enabled. An administrator must enable Dependency Graph in repository
  Settings / Advanced Security. See [GitHub's instructions](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/enable-dependency-graph).
  Do not bypass the failing review to make the PR appear green.
- The SenseNova scanner skipped CUDA-specific torch/torchvision versions because
  it could not audit those package identities on PyPI. Those skips do not prove
  that the CUDA wheels are free of vulnerabilities.

No GPU generation, dependency migration, or exploit against a real model was
performed as part of this workflow-only correction.
