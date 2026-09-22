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


def decision_controls(prefix, mode, *, gradio_module=None, jev_value="jev", interactive=True):
    if gradio_module is None:
        import gradio as gr
    else:
        gr = gradio_module

    with gr.Group(visible=getattr(mode, "value", None) == jev_value) as controls:
        cadence = gr.Radio(
            choices=[("初回のみ", "once"), ("指定間隔", "interval"), ("毎step", "step")],
            value="once",
            label="Jevの再判定頻度",
            interactive=interactive,
            elem_id=prefix + "-jev-cadence",
            info="最初の判定後、選んだ間隔で更新します。各層をまとめて1回で問い合わせます。",
        )
        with gr.Group(visible=False) as interval_controls:
            interval = gr.Slider(
                1,
                100,
                value=2,
                step=1,
                label="再判定する間隔（step）",
                interactive=interactive,
                elem_id=prefix + "-jev-interval",
            )
    mode.change(
        lambda value: gr.update(visible=value == jev_value),
        inputs=mode,
        outputs=controls,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    cadence.change(
        lambda value: gr.update(visible=value == "interval"),
        inputs=cadence,
        outputs=interval_controls,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    return cadence, interval


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
