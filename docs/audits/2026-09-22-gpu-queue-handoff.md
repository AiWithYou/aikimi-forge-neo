# GPU queue handoff audit - 2026-09-22

## Baseline and scope

Repository: `AiWithYou/aikimi-forge-neo`, default branch `neo`.
Baseline commit: `a8ba1e2ab48e4cea34aa8639661a4c40a952881b`.

This is a focused source audit of shared GPU ownership and its adjacent lifecycle
and persistence boundaries, not certification of the entire application.
Reviewed code includes `modules/fifo_lock.py`, `modules_forge/gpu_ownership.py`,
`modules_forge/gpu_residency.py`, `modules_forge/resident_worker.py`,
`modules_forge/minimax_h3_pending.py`, `modules/atomic_file.py`, the process-tree
and startup portion of `modules_forge/yue2_studio/service.py`, and the URL
validation/connection portion of `modules/aikimi_security/url_fetch.py`.
The contribution instructions, current upstream-sync record, CPU test runner,
GPU-ownership tests, and persistence/CPU workflows were also inspected.

Only the queue defects below are classified as reproduced and fixed. Reading an
adjacent component is not evidence that all of its failure modes were tested.

## Primary-source knowledge and decisions

1. [Python 3.13 threading documentation](https://docs.python.org/3.13/library/threading.html#lock-objects)
   explicitly does not promise which waiting thread obtains a primitive lock.
   Therefore an Event followed by an ordinary competing `Lock.acquire()` does
   not establish FIFO order. Preserve a locked mutex during explicit handoff.
2. The same documentation permits primitive-lock release by a different thread.
   Keep that behavior: the application intentionally transfers GPU leases across
   HTTP, generator, and worker lifetimes. Do not replace this lock with an RLock.
3. [PyTorch loading documentation](https://docs.pytorch.org/docs/stable/generated/torch.load)
   warns against untrusted model input even with restricted loading.
   [The PyTorch advisory GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
   lists versions through 2.9.1 as affected and 2.10.0 or later as patched.
   The inspected main CPU workflow pins 2.11.0, so that advisory alone does not
   justify changing its runtime. This is not an audit of every optional worker
   environment or a claim that all dependency vulnerabilities are resolved.
4. `docs/upstream-sync.md` already records selective ControlLLLite/tiled-generation
   integration on this date. Do not replay that work or bulk-merge upstream and
   risk the existing quantization, security, and dedicated-Studio adaptations.

## Reproduced defects

### Q1: New arrivals bypass a signaled waiter

The original release wakes a queued Event and then unlocks the primitive mutex.
Before that waiter runs its second acquisition, a new nonblocking caller can
obtain the mutex. This defeats the FIFO contract and can postpone queued GPU
work under repeated arrivals. The test pauses the selected waiter explicitly,
then proves that the original implementation accepts the newcomer.

**Fix:** Keep the primitive mutex locked while transferring ownership to the
oldest waiter. A resumed waiter already owns the handoff and must not compete
for the mutex again. Unlock only when no queued waiter remains.

### Q2: Interrupted waits consume or strand a handoff

An exception from Event.wait leaves the original queue entry behind. If a waiter
was already selected when its wait is interrupted, the following waiter can
remain asleep until an unrelated new caller happens to release the lock.

**Fix:** Remove an interrupted waiter that is still queued. If its entry was
already selected, transfer the existing ownership to the next waiter or unlock
when the queue is empty. Re-raise the original exception; do not suppress it.
The low-level transfer does not call GPU residency cleanup a second time.

## Changes and compatibility

- `modules/fifo_lock.py`: explicit FIFO handoff and interrupted-wait recovery.
- `tools/tests/test_fifo_lock_handoff.py`: 22 dependency-free contracts covering
  both FIFOLock and the real GPUQueueLock class, including nonblocking calls,
  context-manager cleanup, cross-thread release, FIFO ordering, interruptions,
  mutual exclusion, and model-preparation/cleanup exceptions.
- `.github/workflows/persistence-reliability.yml`: include these contracts in
  the existing Windows/Linux Python 3.13 matrix, lint/format checks, and path
  triggers. No new runtime dependency or additional matrix is introduced.

The original reference attribution remains. Public acquisition/release
signatures and context-manager behavior remain unchanged. No model, prompt,
seed, sampler, retention policy, UI, launcher, or dependency pin is changed.

## Verification

Local execution used Python 3.13.5 on Linux, without GPU or model downloads.
The retrieved original FIFO, GPU ownership, GPU residency, and workflow files
were checked against their Git blob SHAs before local use.

- Verified original implementation: 22 tests, **6 failures** (three regression
  scenarios for each of the base and production GPU queue classes).
- Fixed implementation: **22 tests passed**.
- Repeated in 20 independent Python processes: **440 successful tests**.
- The repetitions included **48,000 lock-protected stress updates** in total.
- Changed Python files compile successfully.

Reproduce the focused contracts from the full checkout:

```console
python -m unittest -v tools.tests.test_fifo_lock_handoff
```

The local environment could not clone the entire repository over the network;
the focused tests ran against a source subset with verified original blob
identities. This is not a full-suite result. Windows execution, the full CPU
suite, and Ruff checks are delegated to the PR's existing CI and must be checked
there before merge. GPU generation quality, throughput, peak VRAM, shutdown with
real model workers, and optional dependency environments were not live-tested.
There is no measured generation-speed claim.
