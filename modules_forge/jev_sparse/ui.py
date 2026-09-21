"""Shared key entry UI. The saved key is never loaded into a browser component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .common import sdk_python
from .credentials import credential_path, save_key


def save_from_ui(value):
    try:
        save_key(value)
        return "", "保存しました。各モデルでJevを選ぶと使用します。"
    except ValueError as exc:
        return "", str(exc)
    except Exception:
        return "", "保存できませんでした。ユーザーフォルダーの書き込み権限を確認してください。"


def save_and_prepare(value):
    cleared, message = save_from_ui(value)
    value = ""
    if not message.startswith("保存しました"):
        yield cleared, message
        return
    try:
        sdk_python()
    except RuntimeError:
        yield "", "キーを保存しました。Jev接続用ライブラリを準備中です。"
        from tools.setup_jev_sparse import environment

        script = Path(__file__).resolve().parents[2] / "tools/setup_jev_sparse.py"
        try:
            result = subprocess.run(  # noqa: S603 -- fixed bundled installer, no key in arguments/environment
                [sys.executable, str(script), "--sdk"],
                env=environment(),
                capture_output=True,
                timeout=180,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode:
                raise RuntimeError
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            yield "", "キーは保存済みです。接続準備に失敗しました。aikimi-jev-setup.batを実行してください。"
            return
    yield "", message


def credential_controls(prefix, gradio_module=None):
    if gradio_module is None:
        import gradio as gr
    else:
        gr = gradio_module

    with gr.Accordion("Jev APIキー設定", open=False):
        key = gr.Textbox(
            value="", type="password", label="TypeSafe APIキー", placeholder="apikey_…", elem_id=f"{prefix}-jev-key"
        )
        save = gr.Button("保存してJevを準備", size="sm", elem_id=f"{prefix}-jev-save")
        status = gr.Textbox(
            value="保存済み" if credential_path().is_file() else "未設定",
            label="キーの保存状態",
            interactive=False,
            elem_id=f"{prefix}-jev-key-status",
        )
        gr.Markdown(
            "[TypeSafeでキーを取得](https://console.typesafe.ai/settings/keys)。"
            "キーはGit管理外のユーザー領域に保存します（Windowsは暗号化）。"
            "Jevを選んだ生成ではAPI利用料が発生する場合があります。"
        )
        save.click(save_and_prepare, inputs=key, outputs=[key, status], api_visibility="private", concurrency_limit=1)
    return key, save, status
