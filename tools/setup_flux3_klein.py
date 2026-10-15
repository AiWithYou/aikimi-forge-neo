"""Prepare FLUX 3's own Comfy checkout, Python environment, and verified weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.aikimi_security.redaction import redact_text, sanitized_subprocess_environment  # noqa: E402
from modules_forge.flux3_klein import core  # noqa: E402

LOCK = ROOT / "tools/requirements-flux3-klein.lock"


class Flux3SetupError(core.Flux3Error):
    """Setup failed without exposing network credentials or raw hub errors."""


def download_plan(*, model="distilled", precision="int8", with_pup=False) -> list[dict]:
    if type(with_pup) is not bool:
        raise Flux3SetupError("PUPの導入指定はtrueまたはfalseです。")
    return [
        dict(spec)
        for name, spec in core.model_specs({"model": model, "precision": precision}).items()
        if with_pup or name not in {"pup", "pup_prompts"}
    ]


def _target(root: Path, relative: str) -> Path:
    root = Path(root).absolute()
    if root.is_symlink() or root.resolve() != root:
        raise Flux3SetupError("管理フォルダーにリンクは使用できません。")
    relative_path = PurePosixPath(relative.replace("\\", "/"))
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"..", "."} or ":" in part for part in relative_path.parts)
    ):
        raise Flux3SetupError("管理フォルダー外への保存を拒否しました。")
    target = root.joinpath(*relative_path.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise Flux3SetupError("管理フォルダー外へのリンクを拒否しました。")
    return target


def _hub_download(**kwargs):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def download_asset(root: Path, spec: dict) -> Path:
    root = Path(root)
    target = _target(root, spec["path"])
    if core.asset_ready(root, spec):
        return target
    cache = _target(root, ".downloads")
    cache.mkdir(parents=True, exist_ok=True)
    try:
        source = Path(
            _hub_download(
                repo_id=spec["repository"],
                filename=spec["filename"],
                revision=spec["revision"],
                token=True,
                local_dir=cache,
            )
        )
    except Exception:
        raise Flux3SetupError(
            "Hugging Faceの固定版ファイルを取得できません。既存の認証と配布元でのアクセス承認、接続を確認してください。再実行で中断分を再開します。"
        ) from None
    try:
        if not source.is_file() or source.stat().st_size != spec["size"]:
            raise Flux3SetupError("取得ファイルのサイズが固定版と一致しません。")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise Flux3SetupError("取得ファイルのSHA-256が固定版と一致しません。")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hub local_dir resumes downloads in its own incomplete file. Publish
        # only after independent verification, without duplicating large weights.
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            os.replace(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
    except OSError:
        raise Flux3SetupError(
            "検証したモデルを管理フォルダーへ保存できません。空き容量とアクセス権を確認してください。"
        ) from None


@contextmanager
def setup_lock(root: Path):
    root = Path(root)
    lock = _target(root, ".setup.lock")
    root.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Flux3SetupError("FLUX 3の導入を別の処理で実行しています。") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_command(arguments, cwd: Path, environment: dict, log: Path) -> str:
    result = subprocess.run(  # noqa: S603 -- fixed tools and separate managed path arguments; no shell.
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
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(redact_text(result.stdout + result.stderr))
    if result.returncode:
        raise Flux3SetupError(
            f"{Path(arguments[0]).name}の実行に失敗しました。専用セットアップログを確認してください。"
        )
    return result.stdout.strip()


def runtime_fingerprint() -> str:
    return core.runtime_fingerprint()


def runtime_ready(root: Path = core.RUNTIME) -> bool:
    return core.runtime_ready(root)


def prepare_comfy_checkout(runtime: Path, command, git: str) -> None:
    if not runtime.exists():
        command(
            [git, "clone", "--filter=blob:none", "--no-checkout", "--", core.manifest()["comfy_repository"], runtime]
        )
    git_directory = runtime / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise Flux3SetupError("FLUX 3のComfyUI保存先に管理対象外のフォルダーがあります。")
    # A --no-checkout clone has no index and reports all tracked files as
    # deletions. Check out its fixed revision before testing the worktree.
    has_index = (git_directory / "index").is_file()
    if has_index and command([git, "status", "--porcelain", "--untracked-files=no"], cwd=runtime):
        raise Flux3SetupError("FLUX 3のComfyUIに変更があります。変更を保ったまま導入を中断しました。")
    revision = command([git, "rev-parse", "HEAD"], cwd=runtime)
    if not has_index or revision != core.COMFY_REVISION or not (runtime / "main.py").is_file():
        command([git, "fetch", "--no-tags", "origin", core.COMFY_REVISION], cwd=runtime)
        command([git, "checkout", "--detach", core.COMFY_REVISION], cwd=runtime)


def _worker_running(root: Path) -> bool:
    import psutil

    expected = os.path.normcase(str(core.python_path(root).resolve()))
    for process in psutil.process_iter(["exe"]):
        try:
            if process.info["exe"] and os.path.normcase(str(Path(process.info["exe"]).resolve())) == expected:
                return True
        except (OSError, psutil.Error):
            continue
    return False


def prepare_runtime(root: Path = core.RUNTIME, *, report=print, verify=True) -> None:
    root = Path(root).absolute()
    if runtime_ready(root):
        return
    if sys.version_info[:2] != (3, 13):
        raise Flux3SetupError("FLUX 3のセットアップはPython 3.13から実行してください。")
    if _worker_running(root):
        raise Flux3SetupError("FLUX 3のworkerが起動中です。生成を停止し、モデルを解放してから導入してください。")
    runtime = core.comfy_path(root)
    repository = root.parent.parent.resolve()
    for relative in ("worker-env", "bootstrap/uv.exe", "bootstrap/uv.whl", ".runtime.json"):
        _target(root, relative)
    _target(repository, "repositories/flux3-klein/ComfyUI")
    runtime.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    log = repository / "work/flux3-klein/setup-runtime.log"
    environment = sanitized_subprocess_environment(os.environ)
    for key in list(environment):
        if key.startswith(("UV_", "PIP_", "PYTHON")) or key in {"VIRTUAL_ENV", "CONDA_PREFIX"}:
            environment.pop(key)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "UV_NO_CONFIG": "1",
            "UV_CACHE_DIR": str(root / "package-cache"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    def command(arguments, *, cwd=runtime.parent):
        return runtime_command(arguments, cwd, environment, log)

    git = shutil.which("git")
    if not git:
        raise Flux3SetupError("Git for Windowsを導入してください。")
    report("FLUX 3専用の固定版ComfyUIを確認しています。")
    prepare_comfy_checkout(runtime, command, git)
    (root / ".runtime.json").unlink(missing_ok=True)
    from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec

    spec = ArtifactSpec(**core.manifest()["uv"])
    profile = ProfileSpec("uv", "FLUX 3 package installer", (spec,), (spec.license_url,), spec.size)
    Installer(root, {"uv": profile}).install("uv", dry_run=False, keep_source=True)
    executable = root / "bootstrap/uv.exe"
    with zipfile.ZipFile(root / "bootstrap/uv.whl") as archive:
        members = [name for name in archive.namelist() if name.endswith("/uv.exe")]
        if len(members) != 1:
            raise Flux3SetupError("固定版uvの構成が一致しません。")
        executable.write_bytes(archive.read(members[0]))
    python = core.python_path(root)
    if not python.is_file():
        command([sys.executable, "-I", "-m", "venv", root / "worker-env"])
    report("FLUX 3専用Pythonへ固定したGPU依存関係を導入しています。")
    command(
        [
            executable,
            "pip",
            "sync",
            "--python",
            python,
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
            "--index-strategy",
            "unsafe-best-match",
            LOCK,
        ]
    )
    command([executable, "pip", "check", "--python", python])
    if verify:
        verify_runtime(root)


def verify_runtime(root: Path = core.RUNTIME) -> None:
    root = Path(root)
    runtime = core.comfy_path(root)
    log = root.parent.parent / "work/flux3-klein/setup-runtime.log"
    environment = sanitized_subprocess_environment(os.environ)
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    python = core.python_path(root)

    def command(arguments, *, cwd=runtime):
        return runtime_command(arguments, cwd, environment, log)

    command(
        [
            python,
            "-I",
            "-c",
            "import sys,torch,importlib.metadata as m; "
            f"sys.path.insert(0,{str(runtime)!r}); "
            "assert torch.__version__=='2.13.0+cu130'; assert torch.cuda.is_available(); "
            "assert m.version('comfy-kitchen')=='0.2.37'; "
            "import comfy.cli_args; comfy.cli_args.args.cpu=True; "
            "import comfy.sd, comfy.ldm.flux3.model; print('FLUX 3 runtime OK')",
        ],
        cwd=runtime,
    )
    core.atomic_json(
        root / ".runtime.json",
        {"schema_version": 1, "fingerprint": runtime_fingerprint(), "comfy_revision": core.COMFY_REVISION},
    )


def run(
    *,
    model="distilled",
    precision="int8",
    with_pup=False,
    dry_run=False,
    root=core.RUNTIME,
    runtime_only=False,
    report=print,
) -> dict:
    root = Path(root)
    plan = download_plan(model=model, precision=precision, with_pup=with_pup)
    if dry_run:
        return {
            "dry_run": True,
            "model": model,
            "precision": precision,
            "assets": [] if runtime_only else plan,
            "comfy_revision": core.COMFY_REVISION,
            "model_bytes": 0 if runtime_only else sum(x["size"] for x in plan),
        }
    with setup_lock(root):
        if not runtime_only:
            for spec in plan:
                report(f"{spec['path']} を取得・検証しています。")
                download_asset(root, spec)
        prepare_runtime(root, report=report)
        ready = runtime_ready(root)
        models = core.model_ready(root, {"model": model, "precision": precision, "pup": with_pup})
        return {
            "ok": bool(ready and (runtime_only or models)),
            "runtime_ready": ready,
            "model_ready": models,
            "model": model,
            "precision": precision,
            "comfy_revision": core.COMFY_REVISION,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FLUX 3 Klein専用環境と固定版モデルを準備します。")
    parser.add_argument("--model", choices=("distilled", "base"), default="distilled")
    parser.add_argument("--precision", choices=("int8", "bf16", "nvfp4"), default="int8")
    pup = parser.add_mutually_exclusive_group()
    pup.add_argument("--with-pup", dest="with_pup", action="store_true", help="任意の追加機能PUPも導入します。")
    pup.add_argument("--without-pup", dest="with_pup", action="store_false", help="PUPを取得せず導入します。")
    parser.set_defaults(with_pup=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-only", action="store_true", help="重みを取得せず専用実行環境だけを準備します。")
    args = parser.parse_args(argv)
    try:
        result = run(**vars(args))
    except (Flux3SetupError, OSError, ValueError) as error:
        print(redact_text(str(error)), file=sys.stderr)  # noqa: T201 -- redacted CLI error.
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))  # noqa: T201 -- CLI result.
    return 0 if result.get("ok", result.get("dry_run", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
