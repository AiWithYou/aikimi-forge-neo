"""Install selected MOSS SFT formats and its isolated, pinned Comfy runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.aikimi_security.redaction import sanitized_subprocess_environment  # noqa: E402
from modules_forge.moss_image import core  # noqa: E402

MANIFEST, MODEL_ROOT, RECEIPT = core.MANIFEST, core.MODEL_ROOT, core.RECEIPT
atomic_json = core.atomic_json


class MossImageSetupError(ValueError):
    """Installation failed before verified assets became available."""


def manifest():
    return core.manifest()


def download_plan(*, precision="int8", text_precision="int8", with_dmd=False):
    selected = core.model_specs({"precision": precision, "text_precision": text_precision})
    return selected + ([dict(manifest()["assets"]["dmd_lora"])] if with_dmd else [])


def target_path(root, relative):
    relative = PurePosixPath(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in {".", ".."} or ":" in p or "\\" in p for p in relative.parts)
    ):
        raise MossImageSetupError("モデルファイルの保存先が不正です。")
    try:
        base = core.safe_model_root(root)
        target = core.safe_model_root(base.joinpath(*relative.parts))
        if not target.is_relative_to(base):
            raise core.MossImageError("outside")
        return target
    except core.MossImageError as error:
        raise MossImageSetupError("モデルの保存先にリンクは使用できません。") from error


def verified(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]


def runtime_command(arguments, *, cwd, environment, log):
    result = subprocess.run(  # noqa: S603 -- Pinned executables, separate arguments, no shell.
        [str(value) for value in arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    from modules.aikimi_security.redaction import redact_text

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(redact_text(result.stdout + result.stderr))
    if result.returncode:
        raise MossImageSetupError("実行環境を準備できませんでした。MOSS Imageのセットアップログを確認してください。")
    return result.stdout.strip()


def prepare_comfy_checkout(runtime, command, git):
    spec = manifest()["runtime"]
    runtime = core.safe_model_root(runtime)
    if not runtime.exists():
        command([git, "clone", "--filter=blob:none", "--no-checkout", "--", spec["comfy_repository"], runtime])
    if not (runtime / ".git").is_dir() or (runtime / ".git").is_symlink():
        raise MossImageSetupError("MOSS ImageのComfyUI保存先は管理対象のGitチェックアウトが必要です。")
    indexed = (runtime / ".git/index").is_file()
    if indexed and command([git, "status", "--porcelain", "--untracked-files=no"], cwd=runtime):
        raise MossImageSetupError("MOSS ImageのComfyUIに変更があります。変更を保ったまま導入を中断しました。")
    head = command([git, "rev-parse", "HEAD"], cwd=runtime)
    if not indexed or head != spec["comfy_revision"]:
        command([git, "fetch", "--no-tags", "origin", spec["comfy_revision"]], cwd=runtime)
        command([git, "checkout", "--detach", spec["comfy_revision"]], cwd=runtime)


def _worker_running():
    import psutil

    script = os.path.normcase(str(ROOT / "tools/moss_image_worker.py"))
    for process in psutil.process_iter(["cmdline"]):
        try:
            if any(os.path.normcase(str(Path(part).absolute())) == script for part in process.info["cmdline"] or []):
                return True
        except (OSError, ValueError, psutil.Error):
            continue
    return False


def prepare_runtime(root=MODEL_ROOT):
    if core.runtime_ready(root):
        return
    if sys.version_info[:2] != (3, 13):
        raise MossImageSetupError("MOSS ImageのセットアップはPython 3.13から実行してください。")
    if _worker_running():
        raise MossImageSetupError("MOSS Imageのモデルを解放してからセットアップを実行してください。")
    paths = core.runtime_paths(root)
    base = core.safe_model_root(paths["base"])
    for path in paths.values():
        core.safe_model_root(path)
    base.mkdir(parents=True, exist_ok=True)
    log = ROOT / "work/moss-image/setup-runtime.log"
    environment = sanitized_subprocess_environment(os.environ)
    for key in list(environment):
        if key.startswith(("UV_", "PIP_", "PYTHON")) or key in {"VIRTUAL_ENV", "CONDA_PREFIX"}:
            environment.pop(key)
    environment.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        UV_NO_CONFIG="1",
        UV_CACHE_DIR=str(ROOT / "cache/moss-image-packages"),
        HF_HUB_DISABLE_TELEMETRY="1",
    )

    def command(arguments, *, cwd=base):
        return runtime_command(arguments, cwd=cwd, environment=environment, log=log)

    git = shutil.which("git")
    if not git:
        raise MossImageSetupError("Gitを導入してください。")
    prepare_comfy_checkout(paths["comfy"], command, git)
    (base / ".runtime.json").unlink(missing_ok=True)
    from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec

    uv_spec = ArtifactSpec(**manifest()["uv"])
    profile = ProfileSpec("uv", "MOSS package installer", (uv_spec,), (uv_spec.license_url,), uv_spec.size)
    Installer(base, {"uv": profile}).install("uv", dry_run=False, keep_source=True)
    executable = target_path(base, "bootstrap/uv.exe")
    with zipfile.ZipFile(target_path(base, "bootstrap/uv.whl")) as archive:
        members = [name for name in archive.namelist() if name.endswith("/uv.exe")]
        if len(members) != 1:
            raise MossImageSetupError("固定版uvの構成が一致しません。")
        executable.write_bytes(archive.read(members[0]))
    if not paths["python"].is_file():
        command([sys.executable, "-I", "-m", "venv", base / "venv"])
    command(
        [
            executable,
            "pip",
            "sync",
            "--python",
            paths["python"],
            "--require-hashes",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
            "--index-strategy",
            "unsafe-best-match",
            core.LOCK,
        ]
    )
    command([executable, "pip", "check", "--python", paths["python"]])
    # Import validation deliberately uses CPU mode and never loads model weights.
    command(
        [
            paths["python"],
            "-I",
            "-c",
            "import sys,torch,importlib.metadata as m; "
            f"sys.path.insert(0,{str(paths['comfy'])!r}); sys.argv=['moss-runtime','--cpu']; "
            "import comfy.cli_args; comfy.cli_args.args.cpu=True; "
            "import comfy.sd,comfy.ldm.moss_image.model; "
            "assert torch.__version__=='2.13.0+cu130'; "
            "assert m.version('comfy-kitchen')=='0.2.39'; "
            "assert hasattr(comfy.sd.CLIPType,'MOSS_IMAGE')",
        ],
        cwd=paths["comfy"],
    )
    atomic_json(
        base / ".runtime.json",
        {
            "schema_version": 1,
            "fingerprint": core.runtime_fingerprint(),
            "comfy_revision": manifest()["runtime"]["comfy_revision"],
        },
    )


def run(
    *,
    root=MODEL_ROOT,
    dry_run=False,
    download=None,
    precision="int8",
    text_precision="int8",
    runtime_only=False,
    with_dmd=False,
):
    value = manifest()
    request = core.validate_request(
        {"prompt": "setup", "seed": 0, "precision": precision, "text_precision": text_precision}
    )
    root = core.safe_model_root(root)
    plan = download_plan(precision=precision, text_precision=text_precision, with_dmd=with_dmd)
    targets = [(spec, target_path(root, spec["path"])) for spec in plan]
    result = {
        "model_id": value["repo_id"],
        "model_revision": value["revision"],
        "precision": precision,
        "text_precision": text_precision,
        "with_dmd": with_dmd,
        "bytes": 0 if runtime_only else sum(spec["size"] for spec in plan),
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    prepare_runtime(root)
    if not core.runtime_ready(root):
        raise MossImageSetupError("MOSS Imageの実行環境を検証できませんでした。")
    if runtime_only:
        return {**result, "ok": True, "runtime_ready": True, "model_ready": core.model_ready(root, request)}
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    receipt = target_path(root, RECEIPT)
    try:
        previous = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    receipt.unlink(missing_ok=True)
    files = {}
    try:
        for spec, path in targets:
            if not verified(path, spec):
                # local_dir follows the Hub filename, which can differ from our managed path.
                filename_parts = PurePosixPath(spec["filename"]).parts
                local_dir = path.parents[len(filename_parts) - 1]
                source = core.safe_model_root(
                    download(
                        repo_id=spec["repo_id"],
                        revision=spec["revision"],
                        filename=spec["filename"],
                        local_dir=str(local_dir),
                        force_download=path.exists(),
                    )
                )
                if source != path or not verified(path, spec):
                    raise MossImageSetupError(f"モデルファイルの検証に失敗しました: {spec['path']}")
            stat = path.stat()
            files[spec["path"]] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": spec["sha256"]}
        # Keep receipts for other explicitly installed formats while rechecking their identity.
        for spec in [*value["files"], *value["assets"].values()]:
            if spec["path"] not in files and spec["path"] in previous.get("files", {}):
                path = target_path(root, spec["path"])
                if verified(path, spec):
                    stat = path.stat()
                    files[spec["path"]] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": spec["sha256"]}
        atomic_json(
            receipt,
            {
                "schema_version": 2,
                "model_id": value["repo_id"],
                "model_revision": value["revision"],
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "files": files,
            },
        )
    except (OSError, RuntimeError) as error:
        raise MossImageSetupError(
            "モデルを取得できませんでした。接続を確認してセットアップを再実行してください。"
        ) from error
    return {**result, "ok": True, "files": len(files), "runtime_ready": True, "model_ready": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--precision", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--text-precision", choices=("int8", "w4a8", "bf16"), default="int8")
    parser.add_argument("--with-dmd", action="store_true", help="also download the optional Euler-compatible DMD LoRA")
    args = parser.parse_args(argv)
    try:
        print(  # noqa: T201 -- Setup CLI status.
            json.dumps(
                run(
                    dry_run=args.dry_run,
                    runtime_only=args.runtime_only,
                    precision=args.precision,
                    text_precision=args.text_precision,
                    with_dmd=args.with_dmd,
                ),
                ensure_ascii=False,
            )
        )
    except (MossImageSetupError, core.MossImageError) as error:
        print(str(error), file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
