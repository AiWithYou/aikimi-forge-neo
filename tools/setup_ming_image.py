"""Install selected Ming SFT formats and its isolated, pinned Comfy runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.aikimi_security.redaction import sanitized_subprocess_environment  # noqa: E402
from modules_forge.ming_image import core  # noqa: E402

MANIFEST, MODEL_ROOT, RECEIPT = core.MANIFEST, core.MODEL_ROOT, core.RECEIPT
atomic_json = core.atomic_json


class MingImageSetupError(ValueError):
    """Installation failed before verified assets became available."""


def manifest():
    return core.manifest()


def download_plan(*, profile):
    return core.model_specs({"profile": profile})


def sync_hf_manifest(*, path=MANIFEST, api=None, download=None):
    """Resolve one immutable Hub commit, then cross-check all HQ files before writing."""
    if api is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download

        api = api or HfApi()
        download = download or hf_hub_download

    try:
        return _sync_manifest(path, api, download)
    except MingImageSetupError:
        raise
    except Exception as exc:
        raise MingImageSetupError("HQモデルの配布情報を取得・照合できませんでした。") from exc


def _sync_manifest(path, api, download):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    repo = "Aikimi/Ming-Image-0.2-Design-HQ-Quant"
    try:
        info = api.model_info(repo, files_metadata=True)
    except OSError as exc:
        raise MingImageSetupError(
            "HQモデルの公開を確認できません。公開後に --sync-hf-manifest を実行するか comfy_org_w4a8 を選択してください。"
        ) from exc
    revision = info.sha
    if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise MingImageSetupError("HQモデルのリビジョンを確認できません。")
    source = download(repo_id=repo, filename="manifest.json", revision=revision)
    published = json.loads(Path(source).read_text(encoding="utf-8"))
    expected = dict(
        source=data["repo_id"],
        revision=data["revision"],
        comfyui=data["runtime"]["comfy_revision"],
        comfy_kitchen="0.2.42",
    )
    if any(published.get(k) != v for k, v in expected.items()) or not isinstance(published.get("files"), dict):
        raise MingImageSetupError("HQモデルの変換元・実行環境・配布形式が一致しません。")
    declared = published["files"]
    profiles = ("vram24g_hq", "vram16g_hq")
    required = {item["path"] for key in profiles for item in data["profiles"][key]["files"]}
    # The publisher also lists its license, model card and comparison PNGs.
    weights = {
        name
        for name in declared
        if name not in ("LICENSE", "README.md") and not re.fullmatch(r"comparisons/eval_[^/]+\.png", name)
    }
    if weights != required:
        raise MingImageSetupError("HQモデルの配布ファイル名が一致しません。")
    siblings = {item.rfilename: item for item in info.siblings}
    for key in profiles:
        profile = data["profiles"][key]
        for item in profile["files"]:
            name = item["path"]
            record = declared.get(name, {})
            remote = siblings.get(name)
            if (
                not remote
                or not remote.lfs
                or not re.fullmatch(r"[0-9a-f]{64}", remote.lfs.sha256)
                or not isinstance(record, dict)
                or record.get("sha256") != remote.lfs.sha256
                or type(record.get("bytes")) is not int
                or type(remote.size) is not int
                or type(remote.lfs.size) is not int
                or record.get("bytes") != remote.size
                or remote.size != remote.lfs.size
                or remote.size <= 0
            ):
                raise MingImageSetupError(f"HQモデルの配布情報が一致しません: {name}")
            item.update(filename=name, size=remote.size, sha256=remote.lfs.sha256)
        profile.update(revision=revision, status="published")
    atomic_json(Path(path), data)
    return revision


def target_path(root, relative):
    relative = PurePosixPath(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in {".", ".."} or ":" in p or "\\" in p for p in relative.parts)
    ):
        raise MingImageSetupError("モデルファイルの保存先が不正です。")
    try:
        base = core.safe_model_root(root)
        target = core.safe_model_root(base.joinpath(*relative.parts))
        if not target.is_relative_to(base):
            raise core.MingImageError("outside")
        return target
    except core.MingImageError as error:
        raise MingImageSetupError("モデルの保存先にリンクは使用できません。") from error


def verified(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    from modules_forge import local_assets

    return local_assets.file_identity(path)["sha256"] == spec["sha256"]


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
        raise MingImageSetupError("実行環境を準備できませんでした。Ming Imageのセットアップログを確認してください。")
    return result.stdout.strip()


def prepare_comfy_checkout(runtime, command, git):
    spec = manifest()["runtime"]
    try:
        runtime = core.safe_model_root(runtime)
    except core.MingImageError as exc:
        raise MingImageSetupError(str(exc)) from exc
    fresh = not runtime.exists()
    if fresh:
        command([git, "clone", "--filter=blob:none", "--no-checkout", "--", spec["comfy_repository"], runtime])
    try:
        core.check_checkout_paths(runtime)
    except core.MingImageError as exc:
        raise MingImageSetupError(str(exc)) from exc
    indexed = (runtime / ".git/index").is_file()
    if not fresh and (not indexed or core.checkout_changed(command([git, *core.CHECKOUT_STATUS], cwd=runtime))):
        raise MingImageSetupError("Ming ImageのComfyUIに変更があります。変更を保ったまま導入を中断しました。")
    head = command([git, "rev-parse", "HEAD"], cwd=runtime)
    if not indexed or head != spec["comfy_revision"]:
        command([git, "fetch", "--no-tags", "origin", spec["comfy_revision"]], cwd=runtime)
        command([git, "checkout", "--detach", spec["comfy_revision"]], cwd=runtime)
        try:
            core.check_checkout_paths(runtime)
        except core.MingImageError as exc:
            raise MingImageSetupError(str(exc)) from exc
        if core.checkout_changed(command([git, *core.CHECKOUT_STATUS], cwd=runtime)):
            raise MingImageSetupError("Ming ImageのComfyUIに未追跡ファイルまたは変更があります。")


def _worker_running():
    import psutil

    script = os.path.normcase(str(ROOT / "tools/ming_image_worker.py"))
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
        raise MingImageSetupError("Ming ImageのセットアップはPython 3.13から実行してください。")
    if _worker_running():
        raise MingImageSetupError("Ming Imageのモデルを解放してからセットアップを実行してください。")
    paths = core.runtime_paths(root)
    base = core.safe_model_root(paths["base"])
    for path in paths.values():
        core.safe_model_root(path)
    base.mkdir(parents=True, exist_ok=True)
    log = ROOT / "work/ming-image/setup-runtime.log"
    environment = sanitized_subprocess_environment(os.environ)
    for key in list(environment):
        if key.startswith(("UV_", "PIP_", "PYTHON")) or key in {"VIRTUAL_ENV", "CONDA_PREFIX"}:
            environment.pop(key)
    environment.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        UV_NO_CONFIG="1",
        UV_CACHE_DIR=str(ROOT / "cache/ming-image-packages"),
        HF_HUB_DISABLE_TELEMETRY="1",
    )

    def command(arguments, *, cwd=base):
        return runtime_command(arguments, cwd=cwd, environment=environment, log=log)

    git = shutil.which("git")
    if not git:
        raise MingImageSetupError("Gitを導入してください。")
    prepare_comfy_checkout(paths["comfy"], command, git)
    (base / ".runtime.json").unlink(missing_ok=True)
    from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec
    from tools.uv_bootstrap import versioned_uv_spec

    uv_spec = versioned_uv_spec(ArtifactSpec(**manifest()["uv"]))
    profile = ProfileSpec("uv", "Ming package installer", (uv_spec,), (uv_spec.license_url,), uv_spec.size)
    Installer(base, {"uv": profile}).install("uv", dry_run=False, keep_source=True)
    executable = target_path(base, "bootstrap/uv.exe")
    with zipfile.ZipFile(target_path(base, uv_spec.relative_path)) as archive:
        members = [name for name in archive.namelist() if name.endswith("/uv.exe")]
        if len(members) != 1:
            raise MingImageSetupError("固定版uvの構成が一致しません。")
        executable.write_bytes(archive.read(members[0]))
    if not paths["python"].is_file():
        command([sys.executable, "-I", "-m", "venv", base / "venv313"])
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
            f"sys.path.insert(0,{str(paths['comfy'])!r}); sys.argv=['ming-runtime','--cpu']; "
            "import comfy.cli_args; comfy.cli_args.args.cpu=True; "
            "import comfy.sd,comfy_extras.nodes_ming; "
            "assert torch.__version__=='2.13.0+cu130'; "
            "assert m.version('comfy-kitchen')=='0.2.42'; "
            "assert sys.version_info[:2]==(3,13)",
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
    profile=core.PROFILES[0],
    runtime_only=False,
):
    value = manifest()
    request = {"profile": profile}
    root = core.safe_model_root(root)
    plan = [] if runtime_only else download_plan(profile=profile)
    targets = [(spec, target_path(root, spec["path"])) for spec in plan]
    result = {
        "model_id": value["repo_id"],
        "model_revision": value["revision"],
        "profile": profile,
        "bytes": 0 if runtime_only else sum(spec["size"] for spec in plan),
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    from tools.aikimi_setup import Installer

    with Installer(root, {})._mutation_lock():
        prepare_runtime(root)
        if not core.runtime_ready(root):
            raise MingImageSetupError("Ming Imageの実行環境を検証できませんでした。")
        if runtime_only:
            return {**result, "ok": True, "runtime_ready": True, "model_ready": core.model_ready(root, request)}
        files = install_models(root, targets, value, download)
    return {**result, "ok": True, "files": len(files), "runtime_ready": True, "model_ready": True}


def install_models(root, targets, value, download):
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    receipt = target_path(root, RECEIPT)
    try:
        previous = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
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
                    raise MingImageSetupError(f"モデルファイルの検証に失敗しました: {spec['path']}")
            stat = path.stat()
            files[spec["path"]] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": spec["sha256"]}
        # Each selected file is independently rechecked against its pinned hash before inference.
        files = {**previous.get("files", {}), **files}
        atomic_json(
            receipt,
            {
                "schema_version": 2,
                "model_id": value["repo_id"],
                "model_revision": value["revision"],
                "files": files,
            },
        )
    except (OSError, RuntimeError) as error:
        raise MingImageSetupError(
            "モデルを取得できませんでした。接続を確認してセットアップを再実行してください。"
        ) from error
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--profile", choices=core.PROFILES, default=core.PROFILES[0])
    parser.add_argument("--sync-hf-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (
            {"revision": sync_hf_manifest()}
            if args.sync_hf_manifest
            else run(profile=args.profile, dry_run=args.dry_run, runtime_only=args.runtime_only)
        )
        print(json.dumps(result, ensure_ascii=False))  # noqa: T201
    except (MingImageSetupError, core.MingImageError) as error:
        print(str(error), file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
