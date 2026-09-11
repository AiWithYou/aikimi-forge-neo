"""H3の初回導入をGradioから実行する。ダウンロードは明示操作時だけ行う。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from modules.aikimi_security.redaction import sanitized_subprocess_environment
from modules_forge.minimax_h3_bridge import H3BridgeError, model_file_status, runtime_setup_session
from modules_forge.minimax_h3_runtime import configured_model_root, installed_runtime_root, managed_runtime_root


def initial_shared_models(repository_root: Path) -> str:
    models = configured_model_root(repository_root)
    default = Path(repository_root).resolve() / "models/MiniMax-H3"
    return "" if models == default else str(models)


def initial_setup_state(repository_root: Path) -> tuple[str, str]:
    try:
        shared = initial_shared_models(repository_root)
        installed = installed_runtime_root(repository_root)
        if installed and all(model_file_status(installed).values()):
            return shared, "準備済み"
        if shared:
            return shared, "共有モデルを使ってH3の実行環境を準備します。"
        return "", "初回は標準モデル約59.1 GiBを取得します。既存モデルを共有する場合はダウンロード不要です。"
    except (ValueError, H3BridgeError) as error:
        return "", str(error)


def setup_updates(shared_models: str, repository_root: Path):
    root = Path(repository_root).resolve()
    runtime = managed_runtime_root(root)
    try:
        with runtime_setup_session(runtime):
            yield (
                str(runtime),
                "H3のセットアップを開始しています…",
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
            yield from _setup_process(shared_models, root, runtime)
            if installed_runtime_root(root) is None:
                raise H3BridgeError("H3の導入完了を確認できません。")
    except (H3BridgeError, ValueError, OSError) as error:
        yield str(runtime), str(error), gr.update(interactive=True), gr.update(interactive=True)
        return
    yield (
        str(runtime),
        "準備ができました。画像・動画を生成できます。",
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def _setup_process(shared_models: str, root: Path, runtime: Path):
    command = [sys.executable, "-u", str(root / "tools/setup_minimax_h3.py")]
    if shared_models.strip():
        command.extend(["--share-models", shared_models.strip()])
    environment = sanitized_subprocess_environment(os.environ)
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    process = subprocess.Popen(  # noqa: S603 -- 固定されたNeo内の導入CLI。共有パスは独立した引数。
        command,
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        last_message = ""
        for line in process.stdout:
            if line.strip():
                last_message = line.strip()
                yield str(runtime), last_message, gr.update(interactive=False), gr.update(interactive=False)
        if process.wait() != 0:
            raise H3BridgeError(last_message or "セットアップを完了できませんでした。")
    finally:
        if process.poll() is None:
            if os.name == "nt":
                # ダウンロード・Python導入の子プロセスも、自分が開始したツリーだけを終了する。
                subprocess.run(  # noqa: S603 -- このcallbackが保持するプロセスIDだけを使用する。
                    [
                        str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"),
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                    timeout=15,
                )
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        if process.stdout is not None:
            process.stdout.close()
