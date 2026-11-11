"""P-H3 image controls, embedded in the H3 Image tab."""

from __future__ import annotations

import html
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import gradio as gr

from modules.aikimi_security.redaction import sanitized_subprocess_environment
from modules_forge import p_h3_image as ph3
from modules_forge.minimax_h3_images import integer
from modules_forge.minimax_h3_runtime import SERVER_URL, installed_runtime_root, managed_runtime_root

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "推奨 8 MP · 正方形": (2912, 2912),
    "推奨 8 MP · 横長 4:3": (3360, 2496),
    "推奨 8 MP · 縦長 3:4": (2496, 3360),
    "推奨 8 MP · 横長 3:2": (3552, 2368),
    "推奨 8 MP · 縦長 2:3": (2368, 3552),
    "推奨 8 MP · 横長 16:9": (3872, 2176),
    "推奨 8 MP · 縦長 9:16": (2176, 3872),
    "軽量 4 MP · 正方形": (2048, 2048),
    "軽量 4 MP · 横長 4:3": (2368, 1760),
    "軽量 4 MP · 縦長 3:4": (1760, 2368),
    "軽量 4 MP · 横長 3:2": (2496, 1664),
    "軽量 4 MP · 縦長 2:3": (1664, 2496),
    "軽量 4 MP · 横長 16:9": (2720, 1536),
    "軽量 4 MP · 縦長 9:16": (1536, 2720),
    "動作確認 · 512×512": (512, 512),
}


def make_request(mode, prompt, target, references, width, height, seed, use_upsampler, checkpoint,
                 lora="", lora_strength=1.0, *, official=False):
    selected = tuple(str(path) for path in (references or [])) if mode == "edit" else ()
    request = ph3.PH3Request(
        prompt=prompt or "", mode=mode,
        target_image=str(target) if target and mode == "edit" else "", reference_images=selected,
        width=integer(width, "幅", 32, 16384), height=integer(height, "高さ", 32, 16384),
        seed=integer(seed, "Seed", -1, 2**63 - 1), use_upsampler=use_upsampler,
        checkpoint=checkpoint or "",
        lora=lora or "", lora_strength=lora_strength, official=official,
    )
    request.validate()
    return request


def size_summary(width, height, mode, target, references):
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError, OverflowError):
        return "幅と高さを指定してください。"
    pixels = width * height
    count = (int(bool(target)) + len(references or [])) if mode == "edit" else 0
    result = f"{width} × {height} · 約 {pixels / 1_000_000:.2f} MP · 入力 {count} 枚"
    if pixels > 2912**2:
        result += "\n\n高解像度: VRAM・RAM使用量が増えます。このPCでの完走は未検証です。"
    if pixels > 4096**2:
        result += "\n\n16 MPを超える設定は公式掲載例の範囲外です。"
    return result


def _preset(value):
    return PRESETS.get(value, (gr.update(), gr.update()))


def preset_updates(value, mode, target, references):
    if value not in PRESETS:
        return gr.update(), gr.update(), gr.update()
    width, height = PRESETS[value]
    return width, height, size_summary(width, height, mode, target, references)


def initial_size_state(mode, preset, width, height, target, references):
    if PRESETS.get(preset) != (width, height):
        preset = "カスタム"
    state = {"active": mode, "text": ("推奨 8 MP · 正方形", 2912, 2912),
             "edit": ("軽量 4 MP · 正方形", 2048, 2048)}
    state[mode] = (preset, width, height)
    return gr.update(visible=mode == "edit"), preset, size_summary(width, height, mode, target, references), state


def mode_updates(mode, preset, width, height, target, references, state):
    state = dict(state)
    previous = state.get("active", "text")
    if mode != previous:
        state[previous] = (preset, width, height)
    selected, new_width, new_height = state[mode]
    state["active"] = mode
    return (gr.update(visible=mode == "edit"), selected, new_width, new_height,
            size_summary(new_width, new_height, mode, target, references), state)


def reuse_result(value):
    return value, [], "edit"


def lora_choices():
    from modules_forge.minimax_h3_runtime import model_root

    folder = model_root(managed_runtime_root(ROOT)) / "loras"
    names = sorted((str(path.relative_to(folder)) for path in folder.rglob("*.safetensors") if path.is_file()),
                   key=str.casefold)
    return [("使用しない", ""), *((name, name) for name in names)]


def refresh_loras(selected):
    choices = lora_choices()
    values = {value for _, value in choices}
    return gr.update(choices=choices, value=selected if selected in values else "")


def initial_status(checkpoint=None):
    from modules_forge import local_assets
    from modules_forge.minimax_h3_runtime import candidate_runtime_root, model_root

    runtime = installed_runtime_root(ROOT)
    if runtime is None or runtime.resolve() != candidate_runtime_root(ROOT).resolve():
        return "実行環境が未準備です。下の「準備を開く」から始めてください。"
    try:
        folder = model_root(runtime)
        selected = local_assets.selection("p-h3-image").get("checkpoint", "") if checkpoint is None else checkpoint
        model = Path(selected) if selected else folder / "diffusion_models" / ph3.MODEL
        paths = (model, folder / "text_encoders" / ph3.TEXT_ENCODER, folder / "vae" / ph3.VAE)
        if all(path.is_file() for path in paths):
            if not (folder / "text_encoders" / ph3.UPSAMPLER).is_file():
                return "必要ファイルの存在を確認しました。プロンプト補助はオフで使用できます。生成時に検証します。"
            return "必要ファイルの存在を確認しました。生成時に対応ノードとモデルを検証します。"
    except (OSError, ValueError, TypeError):
        pass
    return "P-H3の必要ファイルが不足または未確認です。「準備を開く」から確認してください。"


def _add_checkpoint(value):
    from modules_forge import local_assets, studio_checkpoint_ui

    try:
        path = local_assets.local_path(value or "", suffixes=(".safetensors",))
        if not ph3.compatible_checkpoint(path):
            raise ph3.PH3Error("P-H3互換構造の本体ではありません。")
        studio_checkpoint_ui.save("p-h3-image", str(path))
        choices = studio_checkpoint_ui.inventory("MiniMax-H3", (), ph3.compatible_checkpoint, current=str(path))
        return gr.update(choices=choices, value=str(path)), "本体を選択しました。"
    except (OSError, ValueError, ph3.PH3Error) as exc:
        return gr.update(), "確認が必要です: " + html.escape(str(exc))


def _setup(runtime, *, script="setup_p_h3_image.py"):
    bridge = ph3._bridge()
    runtime = str(managed_runtime_root(ROOT))
    process = None
    try:
        with bridge.runtime_setup_session(Path(runtime), SERVER_URL):
            yield runtime, "実行環境とモデルを準備しています。", gr.update(interactive=False)
            environment = sanitized_subprocess_environment(os.environ)
            environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            process = subprocess.Popen(  # noqa: S603 -- fixed repository script and argument list.
                [sys.executable, "-u", str(ROOT / "tools" / script)], cwd=ROOT,
                env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            last = ""
            for line in process.stdout:
                if line.strip():
                    last = line.strip()
                    yield runtime, html.escape(last), gr.update(interactive=False)
            if process.wait() != 0:
                raise ph3.PH3Error(last or "画像モデルの準備が完了しませんでした。")
            selected = installed_runtime_root(ROOT)
            if selected is None:
                raise ph3.PH3Error("共通実行環境の準備完了を確認できません。")
        yield str(selected), "準備ができました。生成操作で実行環境を確認します。", gr.update(interactive=True)
    except Exception as exc:
        yield runtime, "確認が必要です: " + html.escape(str(exc)), gr.update(interactive=True)
    finally:
        if process is not None:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(  # noqa: S603 -- only the process started by this callback.
                        [str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                else:
                    process.terminate()
                process.wait(timeout=15)
            if process.stdout is not None:
                process.stdout.close()


def _generate(runtime, profile, mode, prompt, target, references, width, height, seed, use_upsampler, checkpoint,
              lora="", lora_strength=1.0, *, data_directory=ROOT, official=False):
    yield "入力を確認しています。", gr.update(), gr.update(), gr.update(), gr.update(), {}, gr.update(interactive=False), gr.update(interactive=False)
    try:
        request = make_request(mode, prompt, target, references, width, height, seed, use_upsampler, checkpoint,
                               lora, lora_strength, official=official)
        output_name = "official_h3_images" if official else "p_h3_images"
        data = Path(data_directory)
        for event in ph3.run_generation(request, managed_runtime_root(ROOT), SERVER_URL, data / "logs/minimax_h3",
                                         data / "outputs/minimax_h3" / output_name, runtime_profile=profile):
            complete = event["stage"] == "complete"
            job = {"prompt_id": event["prompt_id"], "server_url": SERVER_URL} if event.get("prompt_id") and not complete else {}
            message = event["message"]
            if "elapsed" in event:
                message += f" 経過 {event['elapsed']:.0f} 秒。"
            metadata = event.get("metadata", {})
            yield (
                html.escape(message), event.get("path", gr.update()), event.get("files", gr.update()),
                metadata.get("upsampled_prompt") if complete else gr.update(),
                metadata if complete else gr.update(), job,
                gr.update(interactive=complete), gr.update(interactive=bool(job)),
            )
    except ph3.PH3Cancelled as exc:
        yield (html.escape(str(exc)), gr.update(), gr.update(), gr.update(), gr.update(), {},
               gr.update(interactive=True), gr.update(interactive=False))
    except Exception as exc:
        yield ("確認が必要です: " + html.escape(str(exc)), gr.update(), gr.update(), gr.update(), gr.update(), {},
               gr.update(interactive=True), gr.update(interactive=False))


def _cancel(job):
    if not isinstance(job, dict) or not job.get("prompt_id"):
        return "実行中のジョブはありません。", gr.update(interactive=False)
    try:
        ph3.cancel_generation(job["prompt_id"], job["server_url"])
        return "この画像ジョブに中断を要求しました。", gr.update(interactive=False)
    except (OSError, ValueError, RuntimeError) as exc:
        return "中断を要求できません: " + html.escape(str(exc)), gr.update(interactive=True)


def render(root=None, *, data_directory=ROOT):
    from modules_forge.studio_checkpoint_ui import create

    gr.Markdown("P-H3 Imageはテキストからの画像生成と、最大8枚の画像を使った編集に対応します。 **Powered by MiniMax H3**")
    checkpoint, refresh = create(
        "p-h3-image", "MiniMax-H3", (), ph3.compatible_checkpoint, gradio=gr,
        details="本体はP-H3互換構造を確認します。構造の一致は画質や出力の互換性を保証しません。",
    )
    refresh.min_width = 140
    with gr.Row():
        readiness = gr.Markdown(initial_status())
        open_setup = gr.Button("準備を開く", scale=0, min_width=120)
    with gr.Accordion("実行環境・モデルの準備", open=False) as setup_panel:
        gr.Markdown(
            "共通ComfyUIとP-H3用モデルを準備します。標準モデル約29.8 GBと実行環境の空き容量が必要です。 "
            "EU・英国・韓国・米国はライセンスの適用地域外です。 "
            "[モデルの利用条件](https://huggingface.co/PrunaAI/P-H3-Image-Comfy)と "
            "[ライセンス原文](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)を確認してください。"
        )
        runtime = gr.Textbox(value=str(managed_runtime_root(ROOT)), visible=False, interactive=False)
        runtime.do_not_save_to_config = True
        profile = gr.Radio([("高速", "fast"), ("省RAM", "low_ram")], value="fast", label="共通実行環境の起動設定")
        setup = gr.Button("環境とモデルを準備")
        setup_status = gr.Markdown("未確認。初回は準備を実行してください。")
        with gr.Row():
            local_path = gr.Textbox(label="このPCのP-H3本体パス", placeholder="別フォルダーの .safetensors を指定")
            add_local = gr.Button("本体を選択", scale=0)
        local_status = gr.Markdown()
    mode = gr.Radio([("画像生成", "text"), ("画像編集", "edit")], value="text", label="使い方")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            prompt = gr.Textbox(label="プロンプト", lines=5, placeholder="作りたい画像や、編集で変えたい部分を入力")
            with gr.Group(visible=False) as edit_group:
                with gr.Row():
                    target = gr.Image(label="編集する画像 · Picture 1", type="filepath", height=155, min_width=150)
                    references = gr.File(label="参照画像 · Picture 2〜8", file_count="multiple", file_types=[".png", ".jpg", ".jpeg", ".webp"], type="filepath", height=155, min_width=150, allow_reordering=True)
                gr.Markdown("対象は必須です。参照は表示順にPicture 2〜8となり、並べ替えできます。対象は入力時に約4 MPへ拡縮します。元のファイルは変わりません。")
            with gr.Row():
                lora = gr.Dropdown(lora_choices(), value="", label="LoRA", scale=5)
                refresh_lora = gr.Button("一覧を更新", scale=0, min_width=105)
            lora_strength = gr.Slider(-2, 2, value=1.0, step=0.05, label="LoRAの強度")
            preset = gr.Dropdown([*PRESETS, "カスタム"], value="推奨 8 MP · 正方形", label="画像サイズ")
            with gr.Accordion("サイズ・Seed・プロンプト補助", open=False):
                with gr.Row():
                    width = gr.Number(value=2912, minimum=32, maximum=16384, step=32, precision=0, label="幅 px")
                    height = gr.Number(value=2912, minimum=32, maximum=16384, step=32, precision=0, label="高さ px")
                seed = gr.Textbox(value="-1", label="Seed（-1でランダム）")
                upsampler = gr.Checkbox(value=True, label="プロンプト補助")
            summary = gr.Markdown(size_summary(2912, 2912, "text", None, []))
            with gr.Row():
                generate = gr.Button("生成", variant="primary")
                cancel = gr.Button("中断", interactive=False)
            status = gr.Markdown("未実行。")
        with gr.Column(scale=1, min_width=300):
            result = gr.Image(label="生成画像", type="filepath", format="png", interactive=False)
            reuse = gr.Button("この画像を編集", interactive=False)
            files = gr.File(label="PNG・生成条件", file_count="multiple", interactive=False)
            with gr.Accordion("生成条件", open=False):
                actual_prompt = gr.Textbox(label="補助後のプロンプト", interactive=False, lines=4)
                metadata = gr.JSON(label="生成条件")
    job = gr.State({})
    size_state = gr.State({"active": "text", "text": ("推奨 8 MP · 正方形", 2912, 2912),
                           "edit": ("軽量 4 MP · 正方形", 2048, 2048)})
    private = {"api_visibility": "private", "show_progress": "hidden"}
    if root is not None:
        root.load(initial_size_state, [mode, preset, width, height, target, references],
                  [edit_group, preset, summary, size_state], queue=False, **private)
    open_setup.click(lambda: gr.update(open=True), outputs=setup_panel, queue=False, **private)
    setup.click(_setup, [runtime], [runtime, setup_status, setup], concurrency_limit=1,
                concurrency_id="h3-runtime-control", trigger_mode="once", **private).then(
        initial_status, outputs=readiness, queue=False, **private
    )
    add_local.click(_add_checkpoint, [local_path], [checkpoint, local_status], queue=False, **private).then(
        initial_status, inputs=checkpoint, outputs=readiness, queue=False, **private
    )
    checkpoint.input(initial_status, inputs=checkpoint, outputs=readiness, queue=False, **private)
    mode.change(mode_updates, [mode, preset, width, height, target, references, size_state],
                [edit_group, preset, width, height, summary, size_state], queue=False, **private)
    preset.change(preset_updates, [preset, mode, target, references], [width, height, summary], queue=False, **private)
    for control in (width, height):
        control.input(lambda w, h, m, t, r: ("カスタム", size_summary(w, h, m, t, r)),
                      [width, height, mode, target, references], [preset, summary], queue=False, **private)
    for control in (target, references):
        control.change(size_summary, [width, height, mode, target, references], [summary], queue=False, **private)
    refresh_lora.click(refresh_loras, [lora], [lora], queue=False, **private)
    generate.click(partial(_generate, data_directory=data_directory), [runtime, profile, mode, prompt, target, references, width, height, seed, upsampler, checkpoint, lora, lora_strength],
                   [status, result, files, actual_prompt, metadata, job, generate, cancel],
                   concurrency_limit=1, concurrency_id="minimax-h3-generation", trigger_mode="once", **private)
    cancel.click(_cancel, [job], [status, cancel], queue=False, **private)
    result.change(lambda value: gr.update(interactive=bool(value)), result, reuse, queue=False, **private)
    reuse.click(reuse_result, [result], [target, references, mode], queue=False, **private)
