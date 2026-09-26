"""Install the pinned Nanosaur2 ComfyUI runtime, node, and three model files.

No model or executable is downloaded on import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.aikimi_security.redaction import sanitized_subprocess_environment  # noqa: E402
from modules_forge.minimax_h3_runtime import setup_lock  # noqa: E402
from modules_forge.nanosaur2_studio import runtime_root  # noqa: E402
from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec, SetupError  # noqa: E402

MANIFEST_PATH = ROOT / "tools/nanosaur2_manifest.json"


def manifest() -> dict:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SetupError("Nanosaur2 manifestの版を確認できません。")
    return data


def profiles() -> dict[str, ProfileSpec]:
    data = manifest()
    prefix = f"https://huggingface.co/{data['repository']}/resolve/{data['revision']}"
    license_url = data["license_url"]
    artifacts = {}
    for group in ("source", "models"):
        specs = []
        for entry in data[group]:
            source_path = f"nanosaur2_support/{entry['path']}" if group == "source" else entry["path"].split("/")[-1]
            destination = (
                f"repositories/nanosaur2/ComfyUI/custom_nodes/nanosaur2_support/{entry['path']}"
                if group == "source"
                else f"repositories/nanosaur2/ComfyUI/models/{entry['path']}"
            )
            specs.append(
                ArtifactSpec(
                    artifact_id=f"nanosaur2-{group}-{Path(entry['path']).name}",
                    relative_path=destination,
                    url=f"{prefix}/{source_path}",
                    size=entry["size"],
                    sha256=entry["sha256"],
                    license_url=license_url,
                )
            )
        artifacts[group] = ProfileSpec(group, f"Nanosaur2 {group}", tuple(specs), (license_url,), 0)
    return artifacts


def runtime_ready(root: Path = ROOT) -> bool:
    runtime = runtime_root(root)
    if not (runtime / "main.py").is_file() or not (runtime.parent / ".venv/Scripts/python.exe").is_file():
        return False
    receipt = runtime.parent / "setup.json"
    if not receipt.is_file():
        return False
    try:
        if json.loads(receipt.read_text(encoding="utf-8")) != {
            "schema_version": 1,
            "fingerprint": runtime_fingerprint(root),
        }:
            return False
    except (OSError, ValueError, TypeError):
        return False
    git = shutil.which("git")
    if git is None:
        return False
    result = subprocess.run(  # noqa: S603 -- fixed git arguments and local managed path.
        [git, "-C", str(runtime), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0 or result.stdout.strip() != manifest()["comfy_revision"]:
        return False
    changed = subprocess.run(  # noqa: S603 -- fixed read-only git query.
        [git, "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return changed.returncode == 0 and not changed.stdout.strip()


def runtime_fingerprint(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for path in (
        root / "tools/nanosaur2_manifest.json",
        root / "tools/requirements-minimax-h3.lock",
        runtime_root(root) / "requirements.txt",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def install_runtime(root: Path = ROOT) -> None:
    if runtime_ready(root):
        return
    if sys.platform != "win32":
        raise SetupError("実行環境の自動導入はWindows専用です。ComfyUIを先に準備してください。")
    runtime = runtime_root(root)
    base = runtime.parent
    base.mkdir(parents=True, exist_ok=True)
    log = root / "logs/nanosaur2/setup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    data = manifest()
    environment = sanitized_subprocess_environment(os.environ)
    for name in list(environment):
        if name.startswith(("UV_", "PIP_", "PYTHON")) or name in {"VIRTUAL_ENV", "CONDA_PREFIX"}:
            environment.pop(name)
    environment.update(
        {
            "UV_NO_CONFIG": "1",
            "UV_PYTHON_INSTALL_DIR": str(base / "python"),
            "UV_CACHE_DIR": str(base / "cache"),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    def command(arguments: list[str | Path], *, cwd: Path = base) -> str:
        result = subprocess.run(  # noqa: S603 -- fixed tools and separated validated arguments.
            [str(value) for value in arguments],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        with log.open("a", encoding="utf-8") as stream:
            stream.write(result.stdout + result.stderr)
        if result.returncode:
            raise SetupError(f"{Path(arguments[0]).name}の処理に失敗しました。詳細: {log}")
        return result.stdout.strip()

    if shutil.which("git") is None:
        raise SetupError("Gitがありません。Git for Windowsを導入してください。")
    if not runtime.exists():
        sys.stdout.write("固定版ComfyUIを準備しています。\n")
        command(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--",
                "https://github.com/Comfy-Org/ComfyUI.git",
                runtime,
            ]
        )
    if not (runtime / ".git").is_dir():
        raise SetupError("Nanosaur2実行環境の場所に管理対象外のフォルダーがあります。")
    revision = command(["git", "rev-parse", "HEAD"], cwd=runtime)
    if revision != data["comfy_revision"]:
        if (runtime / "main.py").exists():
            raise SetupError("ComfyUIが固定版と異なります。既存の変更を確認してください。")
        command(["git", "fetch", "--no-tags", "origin", data["comfy_revision"]], cwd=runtime)
        command(["git", "checkout", "--detach", data["comfy_revision"]], cwd=runtime)
    if (runtime / "requirements.txt").is_file() is False:
        raise SetupError("ComfyUIのrequirements.txtがありません。")
    if command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=runtime):
        raise SetupError("ComfyUIの管理ファイルに変更があります。内容を確認してください。")

    from tools.setup_minimax_h3 import manifest as h3_manifest

    uv_data = h3_manifest()["uv"].copy()
    uv_data["relative_path"] = "bootstrap/uv.whl"
    uv_spec = ArtifactSpec(**uv_data)
    uv_profile = ProfileSpec("uv", "uv", (uv_spec,), (uv_spec.license_url,), uv_spec.size)
    Installer(base, {"uv": uv_profile}).install("uv", dry_run=False, keep_source=True)
    executable = base / "bootstrap/uv.exe"
    with zipfile.ZipFile(base / "bootstrap/uv.whl") as archive:
        members = [name for name in archive.namelist() if name.endswith("/uv.exe")]
        if len(members) != 1:
            raise SetupError("uv配布ファイルの構成が一致しません。")
        contents = archive.read(members[0])
        if not executable.is_file() or executable.read_bytes() != contents:
            executable.write_bytes(contents)

    sys.stdout.write("Nanosaur2専用PythonとComfyUI依存関係を準備しています。\n")
    command([executable, "python", "install", "3.12.13", "--no-bin", "--no-registry"])
    pythons = list((base / "python").glob("cpython-3.12.13-windows-x86_64-*/python.exe"))
    if len(pythons) != 1:
        raise SetupError("Nanosaur2専用Pythonを確認できません。")
    python = base / ".venv/Scripts/python.exe"
    if not python.is_file():
        command([executable, "venv", "--python", pythons[0], base / ".venv"])
    command(
        [
            executable,
            "pip",
            "sync",
            "--python",
            python,
            "--torch-backend",
            "cu130",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--index-url",
            "https://pypi.org/simple",
            root / "tools/requirements-minimax-h3.lock",
        ]
    )
    constraints = base / "torch-constraints.txt"
    lock_text = (root / "tools/requirements-minimax-h3.lock").read_text(encoding="utf-8")
    pins = re.findall(r"(?m)^(?:torch|torchvision)==[^\s\\]+", lock_text)
    if len(pins) != 2:
        raise SetupError("PyTorch固定版を確認できません。")
    constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
    command(
        [
            executable,
            "pip",
            "install",
            "--python",
            python,
            "--torch-backend",
            "cu130",
            "--only-binary",
            ":all:",
            "--index-url",
            "https://pypi.org/simple",
            "-r",
            runtime / "requirements.txt",
            "-c",
            constraints,
        ]
    )
    command([executable, "pip", "check", "--python", python])
    command(
        [
            python,
            "-I",
            "-c",
            "import torch,importlib.metadata as m; "
            "assert torch.__version__ == '2.11.0+cu130'; "
            "assert m.version('comfy-kitchen') == '0.2.35'; "
            "assert m.version('comfy-aimdo') == '0.5.5'; "
            "assert torch.cuda.is_available(); print('Nanosaur2 CUDA OK')",
        ]
    )
    receipt = base / "setup.json"
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "fingerprint": runtime_fingerprint(root)}) + "\n", encoding="utf-8"
    )
    temporary.replace(receipt)


def run(*, root: Path = ROOT, dry_run: bool = False, verify: bool = False, repair: bool = False) -> dict:
    root = root.resolve()
    installer = Installer(root, profiles())
    if dry_run:
        return {
            "runtime_ready": runtime_ready(root),
            "plans": [installer.install(name, dry_run=True, keep_source=True) for name in ("source", "models")],
        }
    if verify:
        result = installer.verify(("source", "models"))
        result["runtime_ready"] = runtime_ready(root)
        result["ok"] = bool(result["ok"] and result["runtime_ready"])
        return result
    runtime = runtime_root(root)
    with setup_lock(runtime):
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex(("127.0.0.1", 8189)) == 0:
                raise SetupError("ComfyUIが起動中です。生成キューが空になってから停止してください。")
        install_runtime(root)
        results = []
        for name in ("source", "models"):
            result = (
                installer.repair(name, dry_run=False, keep_source=True)
                if repair
                else installer.install(name, dry_run=False, keep_source=True)
            )
            results.append(result)
        verification = installer.verify(("source", "models"))
        ready = runtime_ready(root)
        return {"ok": bool(verification["ok"] and ready), "runtime_ready": ready, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="導入予定だけを表示")
    mode.add_argument("--verify", action="store_true", help="環境・ノード・モデルを検証")
    mode.add_argument("--repair", action="store_true", help="壊れたファイルを退避して再取得")
    args = parser.parse_args(argv)
    try:
        result = run(dry_run=args.dry_run, verify=args.verify, repair=args.repair)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0 if args.dry_run or result.get("ok", False) else 1
    except (SetupError, OSError, ValueError) as exc:
        sys.stderr.write(f"Nanosaur2の準備に失敗しました: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
