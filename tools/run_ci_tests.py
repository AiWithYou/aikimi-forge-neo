"""Run the GPU-free Aikimi unit-test suite with network downloads disabled."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import unittest
from collections.abc import MutableMapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

OFFLINE_ENVIRONMENT: dict[str, str] = {
    "DIFFUSERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

DISABLED_LIVE_TESTS: dict[str, str] = {
    "ANIMA_29B_LIVE_API_TEST": "0",
    "ANIMA_38B_LIVE_API_TEST": "0",
    "HYPERWEAVE_RUN_GPU_TESTS": "0",
    "HYPERWEAVE_RUN_INTERRUPT_TESTS": "0",
    "KREA2_LIVE_API_TEST": "0",
    "MINIMAX_H3_LIVE_TEST": "0",
    "SENSENOVA_U15_LIVE_TEST": "0",
}


def configure_ci_environment(environment: MutableMapping[str, str]) -> None:
    """Apply the deterministic CPU/offline policy before importing application code."""

    environment.update(OFFLINE_ENVIRONMENT)
    environment.update(DISABLED_LIVE_TESTS)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"


def configure_import_path(search_path: list[str]) -> None:
    """Put the repository root first so direct script execution can import Forge."""

    repository = str(REPOSITORY_ROOT)
    while repository in search_path:
        search_path.remove(repository)
    search_path.insert(0, repository)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Aikimi tests in the Python 3.13 CPU/offline CI profile.")
    parser.add_argument(
        "--start-directory",
        default="tools/tests",
        help="repository-relative unittest discovery directory",
    )
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="unittest discovery filename pattern",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="unittest runner verbosity",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="dotted unittest module or test name; repeat to bypass filename discovery",
    )
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help="module imported after CPU policy setup but before tests are loaded",
    )
    return parser


def validate_python_version() -> None:
    if sys.version_info[:2] != (3, 13):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(f"Aikimi CI requires Python 3.13; found Python {version}")


def discover_tests(start_directory: str, pattern: str) -> unittest.TestSuite:
    start_path = (REPOSITORY_ROOT / start_directory).resolve()
    if not start_path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("test discovery directory must stay inside the repository")
    if not start_path.is_dir():
        raise FileNotFoundError(f"test discovery directory does not exist: {start_directory}")

    return unittest.defaultTestLoader.discover(str(start_path), pattern=pattern)


def load_tests(start_directory: str, pattern: str, modules: Sequence[str]) -> unittest.TestSuite:
    if modules:
        return unittest.defaultTestLoader.loadTestsFromNames(list(modules))
    return discover_tests(start_directory, pattern)


def preload_modules(modules: Sequence[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    validate_python_version()
    configure_ci_environment(os.environ)
    configure_import_path(sys.path)
    sys.dont_write_bytecode = True

    # backend.args reads sys.argv during module import. Keep runner-only arguments
    # away from the Forge parser and force its explicit CPU profile.
    sys.argv = [str(Path(__file__).resolve()), "--cpu"]
    os.chdir(REPOSITORY_ROOT)
    preload_modules(arguments.preload)

    print(
        "Aikimi CI test profile: Python 3.13, CPU only, external model downloads disabled",
        flush=True,
    )
    suite = load_tests(arguments.start_directory, arguments.pattern, arguments.module)
    result = unittest.TextTestRunner(verbosity=arguments.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
