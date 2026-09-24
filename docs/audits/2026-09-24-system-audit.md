# System audit and corrections — 2026-09-24

Baseline: `75b52d39` on `neo`, initially clean working tree. Environment:
Windows, Python 3.13.14, existing Forge virtual environment. This is a targeted
source, dependency, and regression audit, not certification of every model or
third-party extension.

## Scope and knowledge collected

Reviewed the repository architecture, contribution and release requirements,
generation ownership contract, previous dependency findings, CI boundaries,
GPU ownership/residency implementation, and remote-image transport. Prioritized
reproducible failures, runtime compatibility, and data/connection integrity.

Primary references checked on 2026-09-24:

- [Python 3.13 HTTP client documentation](https://docs.python.org/3.13/library/http.client.html)
  and [CPython implementation](https://github.com/python/cpython/blob/3.13/Lib/http/client.py):
  `getresponse()` clears the connection socket when ownership passes to a
  closing response; bounded `read(amount)` can reach EOF without raising
  `IncompleteRead`. Compared these behaviors with the installed Python source
  and exercised the real `HTTPConnection`/`HTTPResponse` parser using in-memory
  wire data and a mocked socket, without external requests.
- [AnyIO 4.14.2 release](https://github.com/agronholm/anyio/releases/tag/4.14.2)
  and [distribution metadata](https://pypi.org/pypi/anyio/4.14.2/json): the patch
  fixes TLS IDNA matching, process-worker stderr deadlock, and forwarding of
  POSIX supplementary groups, along with concurrency fixes.
- [pip-audit documentation](https://github.com/pypa/pip-audit): used installation
  path auditing to inspect the actual Forge environment rather than only its
  direct requirement pins. The scanner ran outside the application environment.

## Corrections

### Remote-image transport

1. Set the read timeout before `getresponse()`. Previously response headers used
   the connection timeout, and `Connection: close` responses kept that timeout
   for their bodies because `connection.sock` was already cleared. The configured
   15-second read timeout now applies before response parsing.
2. Compare received bytes against declared Content-Length. Premature EOF now
   raises a client-safe interruption error. Preserve chunked framing and
   close-delimited responses and the existing byte limit.
3. Reject explicit port zero before DNS instead of replacing it with 80/443.

Added four test methods covering these failures and valid transfer formats.
Before correction, the tests produced four assertion failures (including both
connection-mode subtests); after correction, all 23 image-fetch tests passed.
DNS/IP pinning, TLS certificate verification, proxy exclusion, redirect checks,
and image-size limits remain in place. This does not add a total wall-clock
download deadline: socket timeouts still limit individual blocking operations.

### Installed dependency and reproducible setup

Pinned AnyIO 4.14.2 in `requirements.txt` and updated the existing environment
from 4.14.1. This also puts AnyIO inside the current direct-pin CI audit.
No other installed dependency was changed; `pip check` passed.

The initial pip-audit 2.10.1 result had eight advisory rows across four packages.
After the update, AnyIO's three findings (`CVE-2026-63374`, `CVE-2026-64847`,
`CVE-2026-63349`) were absent. The final result contains five rows across three
packages, representing three distinct advisory IDs, not five unique issues.

### H3 legacy-format regression test

The separate pytest suite initially failed
`test_h3_cadence_roundtrip_and_workflow`. Its old-format fixtures removed a
fixed number of fields from the current tuple. Adding `compiler_mode` had
shifted those slices into invalid partial formats. Current-format roundtrip
already passed. Replaced the slices with explicit historical prefixes and
checked pre-cadence, cadence, and budget formats. Production H3 parsing and
generation behavior were not changed.

## Verification

- Initial CPU/offline suite: 1,650 tests, success, 52 skipped, one expected failure.
- Final CPU/offline suite: 1,654 tests, success, 52 skipped, one expected failure
  (156.904 seconds of test execution).
- Separate YuE2, Jev, H3 handoff and Union/VAE contracts: 378 passed,
  five skipped, 19 subtests passed after correction.
- JavaScript H3 workflow conversion: 15 passed.
- Changed Python files: Ruff lint and format checks; Git whitespace check.
- Existing environment: dependency consistency check passed. No additional
  runtime dependency is needed for the transport correction.

Reproduce the main checks from the repository root:

```powershell
.\venv\Scripts\python.exe tools/run_ci_tests.py --verbosity 1
.\venv\Scripts\python.exe -m unittest tools.tests.test_safe_image_fetch -q
.\venv\Scripts\python.exe -m pytest tests/jev_sparse tests/yue2 -q
.\venv\Scripts\python.exe -m pip check
```

Detailed local logs are under `tmp/system-audit-*`; they are intentionally not
tracked. Dependency JSON includes skipped packages as well as findings.

## Remaining findings and limits

- Accelerate 1.14.0: `PYSEC-2026-3804`; diskcache 5.6.3:
  `PYSEC-2026-2447`; setuptools 81.0.0: `PYSEC-2026-3447` remain detected.
  The scanner lists no fixed version for the first two; setuptools 83 conflicts
  with the installed Torch `<82` requirement. Preserve the existing documented
  compatibility constraints and exception expiry; no new suppression was added.
  See [the existing assessment](../security-model.md). Runtime reachability was
  not independently re-proven for every extension in this audit.
- The final scan enumerated 165 distributions. `depth-anything`,
  `depth-anything-v2`, and CUDA-local Torch/TorchVision could not be matched on
  PyPI. These four skips are not clean security results.
- The main CI dependency job uses `no-deps: true`; direct pins alone do not
  cover every installed transitive dependency. Qwen's separate requirement file
  is also absent from the security workflow. Future coverage work must audit
  each actual runtime, including Git-sourced and CUDA packages, without treating
  unmatched versions as safe. This local scan covers the Forge environment only.
- This initial audit did not perform live GPU generation; subsequent H3/Qwen
  work is recorded in [the live VRAM audit](2026-09-24-h3-qwen-vram.md).
  No model download, full running-WebUI browser audit,
  remote Actions execution, publication, or deployment was performed. Tests with
  skips and expected failures are reported as such, not as verified features.
