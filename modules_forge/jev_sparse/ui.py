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


def estimated_calls(steps, cadence="once", interval=2, warmup=1):
    """Estimate from model evaluations; a sampler may evaluate more than once per step."""
    if steps is None:
        return 1 if cadence == "once" else None
    remaining = max(0, int(steps) - max(1, int(warmup)))
    if cadence == "once":
        return min(1, remaining)
    spacing = max(1, int(interval)) if cadence in {"interval", "legacy"} else 1
    return (remaining + spacing - 1) // spacing


def budget_note(mode, maximum, wait, steps=None, cadence="once", interval=2, *, jev_value="jev"):
    if mode != jev_value:
        return "Jev API：0回"
    count = estimated_calls(steps, cadence, interval)
    if maximum:
        count = min(count, int(maximum)) if count is not None else None
    forecast = (
        f"1パスの呼び出し見込み：{count}回"
        if count is not None
        else f"呼び出し見込み：最初の評価後、{1 if cadence == 'step' else int(interval)}評価ごと"
    )
    limits = []
    if maximum:
        limits.append(f"最大{int(maximum)}回")
    if wait:
        limits.append(f"合計待ち時間{float(wait):g}秒")
    return forecast + "。生成全体の上限：" + (" / ".join(limits) if limits else "なし") + "。"


def budget_controls(
    prefix, mode, *, steps=None, cadence=None, interval=None, gradio_module=None, jev_value="jev", interactive=True
):
    if gradio_module is None:
        import gradio as gr
    else:
        gr = gradio_module
    with gr.Group(visible=getattr(mode, "value", None) == jev_value) as controls:
        with gr.Row():
            maximum = gr.Number(
                value=0,
                minimum=0,
                maximum=1000,
                precision=0,
                label="生成全体のAPI回数上限（0＝無制限）",
                interactive=interactive,
                elem_id=prefix + "-jev-job-max-calls",
            )
            wait = gr.Number(
                value=0,
                minimum=0,
                maximum=3600,
                precision=1,
                label="生成全体のAPI待ち時間上限・秒（0＝無制限）",
                interactive=interactive,
                elem_id=prefix + "-jev-job-max-wait",
            )
        note = gr.Markdown("Jev API：0回", elem_id=prefix + "-jev-budget-note")
    inputs = [mode, maximum, wait]
    specifications = []
    for value, default in ((steps, None), (cadence, "once"), (interval, 2)):
        if hasattr(value, "change"):
            specifications.append((len(inputs), None))
            inputs.append(value)
        else:
            specifications.append((None, default if value is None else value))

    def update(*values):
        parameters = [values[index] if index is not None else value for index, value in specifications]
        return budget_note(*values[:3], *parameters, jev_value=jev_value)

    for control in inputs:
        control.change(
            update, inputs=inputs, outputs=note, queue=False, show_progress="hidden", api_visibility="private"
        )
    mode.change(
        lambda value: gr.update(visible=value == jev_value),
        inputs=mode,
        outputs=controls,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    return maximum, wait


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
