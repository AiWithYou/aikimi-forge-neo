"""Explicit, reversible update of Neo's managed H3 runtime, not the Forge env.

Run from a stopped Neo installation on Windows:
  python tools/upgrade_minimax_h3_union2_vae.py --check
  python tools/upgrade_minimax_h3_union2_vae.py --apply
  python tools/upgrade_minimax_h3_union2_vae.py --rollback

The original .venv is not modified. A separate .venv-union2-vae is built from
Neo's hashed lock first. Only then are the pinned new core requirements applied.
A receipt activates the new interpreter last. User edits cause an abort rather
than git reset --hard / git clean / stash. No model weights are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules_forge.minimax_h3_runtime import managed_runtime_root, setup_lock
from modules_forge.minimax_h3_union2_vae import VAE_FIXED_COMMIT

OLD_CORE = "efa6c8f804bff78b46a0fd458ebd2e47bba07a30"
CORE_URL = "https://github.com/Comfy-Org/ComfyUI.git"
REQ_BLOB = "d7fb11b74b707ac29ea01118bc55dcf2b31ea638"
PATCH_BLOB = "a035c784d732c8796b1e5eaf3ab64e5a132cc4fa"
NEW_PATCH_BLOB = "229782cc7414c9967ca77c941635e1561557fff5"
MODEL = "comfy/ldm/minimax/model.py"
RECEIPT = "union2-vae-runtime.json"


def blob(text):
    raw = text.replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False).hexdigest()


def apply_reviewed_file_patch(source: str, patch: str, filename: str) -> str:
    """Apply just one reviewed unified diff section, requiring unique context."""
    selected = False
    active = False
    before, after = [], []

    def flush(value):
        old, new = "".join(before), "".join(after)
        if old:
            if value.count(old) != 1:
                raise ValueError("既知のコンパイラーパッチを一意に確認できません。")
            value = value.replace(old, new, 1)
        before.clear()
        after.clear()
        return value

    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if selected:
                source = flush(source)
            selected = line.rstrip() == f"diff --git a/{filename} b/{filename}"
            active = False
        elif selected and line.startswith("@@"):
            source = flush(source)
            active = True
        elif selected and active:
            if line.startswith(" "):
                before.append(line[1:])
                after.append(line[1:])
            elif line == "\n":
                # The pinned patch contains bare blank context lines.
                before.append(line)
                after.append(line)
            elif line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
    return flush(source) if selected else source


def check_stopped(root: Path):
    for port in (8189, 7860):
        with socket.socket() as connection:
            connection.settimeout(0.3)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"ポート{port}が使用中です。Forge NeoとH3を終了してから更新してください。")
    import psutil

    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        if process.info["pid"] == os.getpid():
            continue
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            cwd = process.info.get("cwd") or ""
            relevant = str(root).lower() in command or (cwd and Path(cwd).resolve().is_relative_to(root))
            if relevant and any(name in command for name in ("launch.py", "webui.py", "main.py")):
                raise ValueError("このNeoまたはH3のPythonが実行中です。自動終了せず、更新を中止します。")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


class Upgrade:
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve(strict=True)
        self.runtime = managed_runtime_root(self.root)
        self.base = self.runtime.parent
        self.record = self.base / RECEIPT
        self.pending = self.base / "union2-vae-update-in-progress.json"
        self.new_venv = self.base / ".venv-union2-vae"
        self.new_python = self.new_venv / "Scripts/python.exe"
        self.patch_path = self.root / "patches/minimax-h3/comfyui-0.34.0-compiler.patch"
        self.new_patch_path = self.root / "patches/minimax-h3/comfyui-h3-compiler-912fca4.patch"
        for path in (self.runtime, self.record, self.pending, self.new_venv, self.patch_path, self.new_patch_path):
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ValueError("管理対象にリンクがあります。自動更新しません。")
        from modules.aikimi_security.redaction import sanitized_subprocess_environment

        self.environment = sanitized_subprocess_environment(os.environ)
        for name in list(self.environment):
            if name.startswith(("UV_", "PIP_", "PYTHON")) or name in {"VIRTUAL_ENV", "CONDA_PREFIX"}:
                self.environment.pop(name)
        self.environment.update(
            {
                "UV_NO_CONFIG": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "UV_CACHE_DIR": str(self.base / "cache"),
            }
        )

    def run(self, arguments, *, check=True):
        result = subprocess.run(
            [str(x) for x in arguments],
            cwd=self.runtime,
            env=self.environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log = self.base / "union2-vae-upgrade.log"
        with log.open("a", encoding="utf-8") as output:
            output.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise ValueError(f"{Path(str(arguments[0])).name}の処理に失敗しました。{log}を確認してください。")
        return result

    def git(self, *args, check=True):
        return self.run(["git", *args], check=check)

    def restore_reviewed_files(self, names):
        """Restore verified tracked files with Git's checkout newline convention."""
        if names:
            self.git("restore", "--source=HEAD", "--worktree", "--", *sorted(names))

    def preflight(self):
        if not (self.runtime / ".git").is_dir() or not (self.base / ".venv/Scripts/python.exe").is_file():
            raise ValueError("既存のH3専用環境がありません。先に通常のH3セットアップを完了してください。")
        if self.pending.exists():
            raise ValueError("未完了の更新記録があります。ログを確認し、先に旧環境を復元してください。")
        if self.record.exists():
            raise ValueError("更新記録が既にあります。--rollbackか、H3の状態確認を使用してください。")
        if self.new_venv.exists():
            raise ValueError(".venv-union2-vaeが既にあります。前回ログを確認し、手動で退避してから再試行してください。")
        head = self.git("rev-parse", "HEAD").stdout.strip()
        if head != OLD_CORE:
            raise ValueError("確認した旧固定版と異なるComfyUIです。ユーザーの版は自動上書きしません。")
        origin = self.git("remote", "get-url", "origin").stdout.strip()
        if origin.removesuffix(".git").lower() != CORE_URL.removesuffix(".git").lower():
            raise ValueError("ComfyUIの取得元が確認した公式リポジトリではありません。")
        if self.git("diff", "--cached", "--name-only").stdout.strip():
            raise ValueError("H3リポジトリにステージ済みの変更があります。")
        patch = self.patch_path.read_text("utf-8").replace("\r\n", "\n")
        if blob(patch) != PATCH_BLOB:
            raise ValueError("Neoの既知のコンパイラーパッチが変更されています。")
        if blob(self.new_patch_path.read_text("utf-8")) != NEW_PATCH_BLOB:
            raise ValueError("更新先Core用のコンパイラーパッチが変更されています。")
        changed = self.git("diff", "--name-only").stdout.splitlines()
        if set(changed) - {"requirements.txt", MODEL}:
            raise ValueError("H3の管理対象外の変更を検出しました。変更は保持します。")
        originals = {}
        for filename in changed:
            base = self.git("show", f"HEAD:{filename}").stdout
            current = (self.runtime / filename).read_text("utf-8").replace("\r\n", "\n")
            expected = (
                base.replace("comfy-aimdo==0.5.2", "comfy-aimdo==0.5.3")
                if filename == "requirements.txt"
                else apply_reviewed_file_patch(base, patch, filename)
            )
            if current not in {base, expected}:
                raise ValueError(f"{filename}に未知の編集があります。上書きしません。")
            originals[filename] = (self.runtime / filename).read_bytes()
        return head, originals

    def make_environment(self, requirements: Path):
        from tools.setup_minimax_h3 import RuntimeInstaller

        uv = RuntimeInstaller(self.root).bootstrap()
        old_python = self.base / ".venv/Scripts/python.exe"
        print("新しいH3専用環境を構築します。既存.venvとモデル重みは変更しません。", flush=True)
        self.run([uv, "venv", "--python", old_python, self.new_venv])
        lock = self.root / "tools/requirements-minimax-h3.lock"
        self.run(
            [
                uv,
                "pip",
                "sync",
                "--python",
                self.new_python,
                "--torch-backend",
                "cu130",
                "--require-hashes",
                "--only-binary",
                ":all:",
                "--index-url",
                "https://pypi.org/simple",
                lock,
            ]
        )
        constraints = requirements.with_name("torch-constraints.txt")
        pins = re.findall(r"(?m)^(?:torch|torchvision)==[^\s\\]+", lock.read_text("utf-8"))
        if len(pins) != 2:
            raise ValueError("既存のtorch/torchvision固定版を確認できません。")
        constraints.write_text("\n".join(pins) + "\n", "utf-8")
        self.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                self.new_python,
                "--torch-backend",
                "cu130",
                "--only-binary",
                ":all:",
                "--index-url",
                "https://pypi.org/simple",
                "-r",
                requirements,
                "-c",
                constraints,
            ]
        )
        self.run([uv, "pip", "check", "--python", self.new_python])
        self.run(
            [
                self.new_python,
                "-I",
                "-c",
                "import torch,importlib.metadata as m; "
                "assert torch.__version__ == '2.11.0+cu130'; "
                "assert m.version('comfy-kitchen') == '0.2.35'; "
                "assert m.version('comfy-aimdo') == '0.5.5'; "
                "assert m.version('comfyui-frontend-package') == '1.53.6'; "
                "assert torch.cuda.is_available(); print('Runtime dependencies / CUDA OK')",
            ]
        )
        return self.run([uv, "pip", "freeze", "--python", self.new_python]).stdout

    def restore_core(self, head, originals):
        # Only used immediately after our own edits, while Neo and H3 are stopped.
        changed = self.git("diff", "--name-only").stdout.splitlines()
        if set(changed) - {"requirements.txt", MODEL} or self.git("diff", "--cached", "--name-only").stdout.strip():
            raise ValueError(
                "更新中の外部編集を検出したため自動復元を中止しました。ログとバックアップを確認してください。"
            )
        self.restore_reviewed_files(changed)
        self.git("checkout", "--detach", head)
        for filename, raw in originals.items():
            (self.runtime / filename).write_bytes(raw)

    def apply(self):
        head, originals = self.preflight()
        self.git("fetch", "--no-tags", "origin", VAE_FIXED_COMMIT)
        requirements = self.git("show", f"{VAE_FIXED_COMMIT}:requirements.txt").stdout
        if blob(requirements) != REQ_BLOB:
            raise ValueError("固定版ComfyUIのrequirementsが確認した内容と異なります。")
        backup = self.base / "union2-vae-backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup.mkdir(parents=True, exist_ok=False)
        reqfile = backup / "requirements.txt"
        reqfile.write_text(requirements, "utf-8")
        for filename, raw in originals.items():
            target = backup / "originals" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        freeze = self.make_environment(reqfile)
        check_stopped(self.root)
        # Recheck all tracked changes after long network/dependency operations.
        if self.preflight_recheck(head, originals) is False:
            raise ValueError("導入中に既存コードが変更されました。切り替えません。")
        with self.pending.open("x", encoding="utf-8") as output:
            json.dump({"old_revision": head, "backup": backup.name}, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            self.restore_reviewed_files(originals)
            self.git("checkout", "--detach", VAE_FIXED_COMMIT)
            model_source = (self.runtime / MODEL).read_text("utf-8")
            if "def _forward_with_memory_graph(" not in model_source:
                self.git("apply", "--check", str(self.new_patch_path))
                self.git("apply", str(self.new_patch_path))
            # Compile touched core without loading model weights.
            self.run(
                [
                    self.new_python,
                    "-m",
                    "py_compile",
                    "comfy/ldm/minimax/vae.py",
                    "comfy/ldm/minimax/controlnet.py",
                    "comfy_extras/nodes_model_patch.py",
                    "comfy_extras/nodes_minimax_h3.py",
                ]
            )
            new_diff = self.git("diff", "--binary").stdout
            (backup / "packages.txt").write_text(freeze, "utf-8")
            record = {
                "schema_version": 1,
                "revision": VAE_FIXED_COMMIT,
                "previous_revision": head,
                "backup": backup.name,
                "original_files": list(originals),
                "activated_diff_sha256": hashlib.sha256(new_diff.encode()).hexdigest(),
            }
            with self.record.open("x", encoding="utf-8") as output:
                json.dump(record, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            self.record.unlink(missing_ok=True)
            self.restore_core(head, originals)
            self.pending.unlink(missing_ok=True)
            raise
        self.pending.unlink()
        print("更新完了。H3 Studioで『選択設定で再起動』を実行してください。推論/画質/速度の検証は別途必要です。")

    def preflight_recheck(self, head, originals):
        if (
            self.git("rev-parse", "HEAD").stdout.strip() != head
            or self.git("diff", "--cached", "--name-only").stdout.strip()
        ):
            return False
        if set(self.git("diff", "--name-only").stdout.splitlines()) != set(originals):
            return False
        return all((self.runtime / name).read_bytes() == raw for name, raw in originals.items())

    def rollback(self):
        if self.record.is_symlink() or not self.record.is_file() or self.record.stat().st_size > 65536:
            raise ValueError("有効な更新記録がありません。")
        data = json.loads(self.record.read_text("utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or data.get("revision") != VAE_FIXED_COMMIT
            or data.get("previous_revision") != OLD_CORE
        ):
            raise ValueError("更新記録のrevisionが不正です。")
        if not re.fullmatch(r"[0-9-]+", data.get("backup", "")):
            raise ValueError("バックアップの識別子が不正です。")
        backup = self.base / "union2-vae-backups" / data["backup"]
        if backup.is_symlink() or backup.resolve() != backup.absolute():
            raise ValueError("バックアップがリンクされています。自動復元しません。")
        diff = self.git("diff", "--binary").stdout
        if (
            self.git("rev-parse", "HEAD").stdout.strip() != VAE_FIXED_COMMIT
            or self.git("diff", "--cached", "--name-only").stdout.strip()
            or hashlib.sha256(diff.encode()).hexdigest() != data.get("activated_diff_sha256")
        ):
            raise ValueError("更新後にComfyUIが変更されています。ロールバックでユーザー編集を上書きしません。")
        names = data.get("original_files")
        if not isinstance(names, list) or set(names) - {"requirements.txt", MODEL}:
            raise ValueError("復元ファイル一覧が不正です。")
        originals = {name: (backup / "originals" / name).read_bytes() for name in names}
        self.restore_core(OLD_CORE, originals)
        self.record.unlink()
        print("旧ComfyUIと旧.venvに戻しました。入力素材・出力動画・モデルは削除していません。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    upgrade = Upgrade()
    if args.check:
        head, _ = upgrade.preflight()
        print(
            f"更新対象確認: {head} -> {VAE_FIXED_COMMIT}. --applyで専用依存関係を取得します。モデル取得はありません。"
        )
        return
    if sys.platform != "win32":
        raise ValueError("このH3専用環境更新はWindowsで実行してください。")
    check_stopped(ROOT)
    with setup_lock(upgrade.runtime):
        upgrade.rollback() if args.rollback else upgrade.apply()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"H3 runtime update: {error}", file=sys.stderr)
        raise SystemExit(1) from error
