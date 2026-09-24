# Accelerate 1.15.0+aikimi.1

Upstream: https://github.com/huggingface/accelerate/releases/tag/v1.15.0

Source: the official PyPI `accelerate-1.15.0.tar.gz`, SHA-256
`5654f8c5eaa0d4fa68b33e287a97765da6849bf6d51dcac874e73fbbddfb6134`.
The runtime package, original build metadata, README and Apache-2.0 LICENSE
are retained. Upstream tests/examples are not needed for installation.

Local changes: whitespace cleanup in README/setup.cfg; version suffix in setup.py and __init__.py; validate every
checkpoint index shard before loading any shards. Reject absolute paths,
Windows drives/UNC/alternate streams, directory traversal and non-files.
Nested relative paths and Hugging Face snapshot symlinks remain supported.
No public API is disabled or replaced at application startup.

This fixes the behavior reported in upstream issue 4067 / PYSEC-2026-3804.
Upstream 1.15.0 alone does **not** fix it. Local regression tests exercise both
load_checkpoint_in_model and load_checkpoint_and_dispatch with real tiny
safetensors. OSV scanning the version is additional evidence, not proof of
this fix. Remove the vendor copy only after an upstream release passes these
same tests. Never edit the package version without updating this record.

Reviewed 2026-09-24. Related upstream proposals:
https://github.com/huggingface/accelerate/pull/4138 and
https://github.com/huggingface/accelerate/pull/4214 (not merged at review time).
