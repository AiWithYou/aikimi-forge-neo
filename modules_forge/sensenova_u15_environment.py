"""SenseNova worker専用のPythonと依存バージョンを検査する。"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "models" / "SenseNova-U1" / "worker-env"
WORKER_PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "tools" / "requirements-sensenova.txt"


def expected_versions() -> dict[str, str]:
    result = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise ValueError("SenseNovaの依存定義には正確なバージョン指定が必要です。")
        result[name.lower().replace("_", "-")] = version
    return result


def environment_status() -> tuple[bool, str]:
    if not WORKER_PYTHON.is_file():
        return (
            False,
            "SenseNova専用Pythonが未準備です。download_sensenova_u15_int8.ps1 -RuntimeOnlyを実行してください。",
        )
    if os.name == "nt":
        paths = [ENVIRONMENT / "Lib" / "site-packages"]
    else:
        paths = list((ENVIRONMENT / "lib").glob("python*/site-packages"))
    versions = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(path) for path in paths])
        if distribution.metadata["Name"]
    }
    mismatched = [
        f"{name}=={version}（現在 {versions.get(name, '未導入')}）"
        for name, version in expected_versions().items()
        if versions.get(name) != version
    ]
    if mismatched:
        return False, "SenseNova専用環境の依存不一致: " + ", ".join(
            mismatched
        ) + "。-RuntimeOnlyで再セットアップしてください。"
    return True, "SenseNova専用Pythonと固定バージョンの依存関係を確認しました。"


def validate_running_versions() -> None:
    for name, expected in expected_versions().items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(
                f"SenseNova workerには{name}=={expected}が必要です（実行環境: {actual}）。専用Pythonで実行してください。"
            )
