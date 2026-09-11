"""H3専用環境をNeo内に導入する。共有するモデルは読み取りだけで検証する。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.aikimi_security.redaction import (  # noqa: E402 -- CLIのimportルート設定後。
    sanitized_subprocess_environment,
)
from modules_forge.minimax_h3_runtime import (  # noqa: E402
    local_directory,
    managed_runtime_root,
    model_config,
    setup_fingerprint,
    setup_lock,
)
from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec, SetupError  # noqa: E402

MANIFEST_PATH = ROOT / "tools/minimax_h3_runtime_manifest.json"
LOCK_PATH = ROOT / "tools/requirements-minimax-h3.lock"
PATCH_PATH = ROOT / "patches/minimax-h3/comfyui-0.34.0-compiler.patch"


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def model_profile() -> ProfileSpec:
    data = manifest()
    artifacts = tuple(
        ArtifactSpec(
            artifact_id=entry["path"],
            relative_path=entry["path"],
            url=f"https://huggingface.co/{data['model_repository']}/resolve/{data['model_revision']}/{entry['path']}",
            size=entry["size"],
            sha256=entry["sha256"],
            license_url=data["model_license"],
        )
        for entry in data["models"]
    )
    return ProfileSpec("h3", "MiniMax H3", artifacts, (data["model_license"],), sum(x.size for x in artifacts))


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def plan(repository_root: Path = ROOT, shared_models: Path | None = None) -> dict:
    runtime = managed_runtime_root(repository_root)
    models = local_directory(shared_models) if shared_models else Path(repository_root).resolve() / "models/MiniMax-H3"
    if shared_models is not None and not models.is_dir():
        raise SetupError("共有するモデルフォルダーが見つかりません。")
    if shared_models is None and not models.resolve().is_relative_to(Path(repository_root).resolve()):
        raise SetupError("Neo内のモデル保存先が外部へリンクされています。既存モデルの共有先として指定してください。")
    profile = model_profile()
    missing = [
        item
        for item in profile.artifacts
        if not (models / item.relative_path).is_file() or (models / item.relative_path).stat().st_size != item.size
    ]
    return {
        "runtime": str(runtime),
        "models": str(models),
        "shared": shared_models is not None,
        "model_download_bytes": 0 if shared_models else sum(item.size for item in missing),
        "missing_models": [item.relative_path for item in missing],
    }


def verify_models(models: Path, *, shared: bool) -> dict:
    profile = model_profile()
    installer = Installer(models, {"h3": profile})
    if shared:
        result = installer.verify(["h3"])
        if not result["ok"]:
            failures = [item["path"] for item in result["profiles"][0]["artifacts"] if item["state"] != "ready"]
            raise SetupError("共有モデルが不足または破損しています: " + ", ".join(failures))
        return result
    return installer.install("h3", dry_run=False, keep_source=True)


class RuntimeInstaller:
    def __init__(self, repository_root: Path = ROOT):
        self.root = local_directory(repository_root)
        self.runtime = managed_runtime_root(self.root)
        self.base = self.runtime.parent
        if not self.base.resolve().is_relative_to(self.root):
            raise SetupError("H3実行環境の保存先がNeoの外にあります。")
        self.log = self.root / "logs/minimax_h3/setup.log"
        for target in (
            self.runtime,
            self.runtime / ".git",
            self.base / ".venv",
            self.base / "python",
            self.base / "bootstrap",
            self.base / "cache",
            self.log,
        ):
            if not target.resolve().is_relative_to(self.root):
                raise SetupError("H3の管理対象にNeo外へのリンクがあります。")
        self.data = manifest()
        self.environment = sanitized_subprocess_environment(os.environ)
        for name in list(self.environment):
            if name.startswith(("UV_", "PIP_", "PYTHON")) or name in {"VIRTUAL_ENV", "CONDA_PREFIX"}:
                self.environment.pop(name)
        self.environment.update(
            {
                "UV_PYTHON_INSTALL_DIR": str(self.base / "python"),
                "UV_CACHE_DIR": str(self.base / "cache"),
                "UV_NO_CONFIG": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            }
        )

    def run(
        self, arguments: list[str | Path], *, cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(  # noqa: S603 -- 固定コマンドと分離した引数のみ。shellは使わない。
            [str(value) for value in arguments],
            cwd=cwd or self.base,
            env=self.environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise SetupError(f"{Path(arguments[0]).name}の処理に失敗しました。詳細: {self.log}")
        return result

    def bootstrap(self) -> Path:
        spec = ArtifactSpec(**self.data["uv"])
        profile = ProfileSpec("uv", "uv", (spec,), (spec.license_url,), spec.size)
        Installer(self.base, {"uv": profile}).install("uv", dry_run=False, keep_source=True)
        with zipfile.ZipFile(self.base / spec.relative_path) as archive:
            members = [name for name in archive.namelist() if name.endswith("/uv.exe")]
            if len(members) != 1:
                raise SetupError("uv配布ファイルの構成が一致しません。")
            executable = self.base / "bootstrap/uv.exe"
            contents = archive.read(members[0])
            if not executable.is_file() or executable.read_bytes() != contents:
                executable.write_bytes(contents)
        return executable

    def install_core(self) -> None:
        if not self.runtime.exists():
            self.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--",
                    self.data["comfy_repository"],
                    self.runtime,
                ]
            )
        if not (self.runtime / ".git").is_dir():
            raise SetupError("H3実行環境の場所に管理対象外のフォルダーがあります。")
        revision = self.run(["git", "rev-parse", "HEAD"], cwd=self.runtime).stdout.strip()
        if revision != self.data["comfy_revision"] or not (self.runtime / "main.py").is_file():
            if (self.runtime / "main.py").exists():
                raise SetupError("H3のComfyUIが固定版と異なります。既存の変更を確認してください。")
            self.run(["git", "checkout", "--detach", self.data["comfy_revision"]], cwd=self.runtime)
        base_requirements = self.run(["git", "show", "HEAD:requirements.txt"], cwd=self.runtime).stdout
        updated = base_requirements.replace("comfy-aimdo==0.5.2", "comfy-aimdo==0.5.3")
        target = self.runtime / "requirements.txt"
        if target.read_text(encoding="utf-8") not in {base_requirements, updated}:
            raise SetupError("H3のrequirements.txtに管理対象外の変更があります。")
        if target.read_text(encoding="utf-8") != updated:
            target.write_text(updated, encoding="utf-8", newline="\n")
        # 過去のパッチに含まれる開発版aimdo指定は使わず、上の正式版pinを維持する。
        patch_arguments = ["git", "apply", "--exclude=requirements.txt"]
        applied = self.run([*patch_arguments, "--reverse", "--check", PATCH_PATH], cwd=self.runtime, check=False)
        if applied.returncode:
            self.run([*patch_arguments, "--check", PATCH_PATH], cwd=self.runtime)
            self.run([*patch_arguments, PATCH_PATH], cwd=self.runtime)

    def install_python(self, uv: Path) -> Path:
        self.run([uv, "python", "install", self.data["python_version"], "--no-bin", "--no-registry"])
        pythons = list(
            (self.base / "python").glob(f"cpython-{self.data['python_version']}-windows-x86_64-*/python.exe")
        )
        if len(pythons) != 1:
            raise SetupError("H3専用Pythonを確認できません。")
        python = self.base / ".venv/Scripts/python.exe"
        if not python.is_file():
            self.run([uv, "venv", "--python", pythons[0], self.base / ".venv"])
        self.run(
            [
                python,
                "-I",
                "-c",
                "import sys; from pathlib import Path; "
                "assert sys.version_info[:3] == (3,12,13); "
                "assert Path(sys.base_prefix).resolve().is_relative_to(Path(sys.argv[1]).resolve()); "
                "assert Path(sys.prefix).resolve() == Path(sys.argv[2]).resolve()",
                self.base / "python",
                self.base / ".venv",
            ]
        )
        self.run(
            [
                uv,
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
                LOCK_PATH,
            ]
        )
        self.run(
            [
                python,
                "-I",
                "-c",
                (
                    "import sys,torch,importlib.metadata as m; "
                    "from pathlib import Path; "
                    "assert Path(sys.base_prefix).resolve().is_relative_to(Path(sys.argv[1]).resolve()); "
                    "assert Path(sys.prefix).resolve() == Path(sys.argv[2]).resolve(); "
                    "assert sys.version_info[:3] == (3,12,13); "
                    "assert torch.__version__ == '2.11.0+cu130'; "
                    "assert m.version('comfy-aimdo') == '0.5.3'; "
                    "assert m.version('comfy-kitchen') == '0.2.33'; "
                    "assert torch.cuda.is_available(), 'CUDA GPUを利用できません。GPUドライバーを確認してください。'; "
                    "print('H3 Python / CUDA OK')"
                ),
                str(self.base / "python"),
                str(self.base / ".venv"),
            ]
        )
        return python

    def install_nodes(self) -> None:
        from modules_forge.minimax_h3_negpip import install_bundled_negpip
        from modules_forge.minimax_h3_negpip_cache import install as install_cache
        from tools.install_minimax_h3_clipcache import install as install_clipcache
        from tools.install_minimax_h3_hybrid import install as install_hybrid

        install_bundled_negpip(self.runtime)
        install_cache(self.runtime)
        install_clipcache(self.runtime)
        install_hybrid(self.runtime)

    def install(self, shared_models: Path | None = None) -> Path:
        if sys.platform != "win32":
            raise SetupError("H3の自動セットアップはWindowsで実行してください。")
        if shutil.which("git") is None:
            raise SetupError("Gitがありません。Git for Windowsを導入してください。")
        configuration = plan(self.root, shared_models)
        models = Path(configuration["models"])
        with setup_lock(self.runtime):
            with socket.socket() as connection:
                connection.settimeout(1)
                if connection.connect_ex(("127.0.0.1", 8189)) == 0:
                    raise SetupError("H3が起動中です。H3 Studioからセットアップを実行してください。")
            if shutil.disk_usage(self.base).free < 15 * 1024**3:
                raise SetupError("H3実行環境の導入に15 GiB以上の空き容量が必要です。")
            # 共有指定では、重い環境の導入前に不足を検出し、モデル側には書き込まない。
            if shared_models:
                emit("共有モデルを検証しています。モデルのコピーや再取得は行いません。")
                verify_models(models, shared=True)
            (self.base / "setup.json").unlink(missing_ok=True)
            emit("H3専用のComfyUIを準備しています。")
            self.install_core()
            emit("H3専用のPythonとライブラリを準備しています。")
            uv = self.bootstrap()
            self.install_python(uv)
            emit("H3の追加機能を準備しています。")
            self.install_nodes()
            if not shared_models:
                emit("標準H3モデルを準備しています。中断後も同じ操作で再開できます。")
                verify_models(models, shared=False)
            self.complete_setup(models, shared=bool(shared_models))
            emit("準備ができました。H3の画像・動画生成を利用できます。")
        return self.runtime

    def complete_setup(self, models: Path, *, shared: bool) -> None:
        """モデル・環境の検証完了後にだけ、利用する設定と完了記録を確定する。"""
        config = self.runtime / "extra_model_paths.yaml"
        config_temporary = config.with_suffix(".yaml.tmp")
        config_temporary.write_text(model_config(models), encoding="utf-8", newline="\n")
        config_temporary.replace(config)
        record = {"schema_version": 1, "fingerprint": setup_fingerprint(), "shared_models": shared}
        temporary = self.base / "setup.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(self.base / "setup.json")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--share-models", type=Path, help="既存モデルのフォルダー。読み取り専用で共有する")
    mode.add_argument("--models-only", action="store_true", help="標準モデルだけを取得し、実行環境は変更しない")
    parser.add_argument("--dry-run", action="store_true", help="保存・通信せず導入計画を表示する")
    args = parser.parse_args()
    try:
        if args.dry_run:
            emit(json.dumps(plan(shared_models=args.share_models), ensure_ascii=False, indent=2))
        elif args.models_only:
            configuration = plan()
            emit("標準H3モデルを準備しています。")
            verify_models(Path(configuration["models"]), shared=False)
            emit("標準H3モデルの準備ができました。")
        else:
            RuntimeInstaller().install(args.share_models)
    except (SetupError, ValueError, OSError) as error:
        emit(f"セットアップを完了できませんでした: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
