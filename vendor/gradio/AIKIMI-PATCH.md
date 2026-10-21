# Gradio dependency compatibility build

This directory builds `gradio==6.17.3+aikimi.1` from the official Gradio 6.17.3 wheel identified by `upstream.json`.

The upstream wheel is downloaded from PyPI and its exact size and SHA-256 are checked before use. `UPSTREAM-METADATA`, `UPSTREAM-WHEEL`, and `LICENSE` preserve its package metadata and Apache-2.0 license.

The build changes only the distribution version and the Pillow requirement from `>=8.0,<13.0` to `>=8.0,<14.0`. Distribution metadata paths and `RECORD` are regenerated. All Python modules, frontend assets, entry points, and bundled licenses retain their original bytes. The runtime's `gradio.__version__` remains 6.17.3, as required by the existing frontend compatibility checks.

Pillow 13 compatibility is checked by image input/output contracts, Gradio component and browser tests, and the CPU suites. The build recipe does not suppress dependency audits or change vulnerability exclusions.
