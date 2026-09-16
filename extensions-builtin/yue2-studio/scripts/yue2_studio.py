"""YuE2 tab: additive to Forge, with no model imports at UI startup."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gradio as gr
from modules import script_callbacks
from modules.paths import data_path, script_path
from modules_forge.yue2_studio.core import Request, YuE2Error, import_project, inside, integer, read_json, runtime_manifest
from modules_forge.yue2_studio.service import Studio

RUNTIME = Path(script_path) / "extensions-builtin" / "yue2-studio" / "runtime"
STUDIO = Studio(RUNTIME, Path(data_path) / "outputs" / "yue2")
FIELDS = list(Request.__dataclass_fields__)
PRIVATE = {"api_visibility": "private", "show_progress": "hidden"}


def owner(request: gr.Request) -> str:
    if not request.session_hash:
        raise YuE2Error("ブラウザーのYuE2タブから操作してください。")
    return f"{request.username or ''}:{request.session_hash}"


def make_request(values):
    data = dict(zip(FIELDS, values, strict=True))
    for field, lo, hi in (("seed", -1, 2**63 - 1), ("candidates", 1, 8),
                          ("steps", 1, 128), ("max_tokens", 200, 9000), ("memory_gib", 0, 192)):
        data[field] = integer(data[field], field, lo, hi)
    return Request.from_dict(data)


def run(title, style, lyrics, abc, cot, engine, seed, candidates, steps, max_tokens,
        memory_gib, offload, fp8, gguf, plan_only, request: gr.Request):
    values = [title, style, lyrics, abc, cot, engine, seed, candidates, steps, max_tokens,
              memory_gib, offload, fp8, gguf]
    return start(values, plan_only, request)


def start(values, plan_only, request):
    try:
        identifier = STUDIO.start(make_request(values), owner(request), plan_only)
        return identifier, "開始しました。", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(active=True)
    except Exception as exc:
        return gr.update(), str(exc), gr.update(), gr.update(), gr.update(), gr.update()


def poll(identifier, request: gr.Request):
    try:
        if not identifier:
            return [gr.update()] * 7
        state = STUDIO.status(identifier, owner(request))
        text = f"{state['message']}  経過 {state['elapsed']:.0f} 秒"
        done = state["done"]
        choices = STUDIO.history() if done else None
        latest = choices[0][1] if choices else None
        return (text, gr.update(interactive=done), gr.update(interactive=done), gr.update(interactive=not done),
                gr.update(active=not done),
                gr.update(choices=choices, value=latest) if done else gr.update(),
                gr.update(choices=choices, value=None) if done else gr.update())
    except Exception as exc:
        return str(exc), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def cancel(identifier, request: gr.Request):
    return "停止を要求しました。終了確認中です。" if STUDIO.cancel(identifier, owner(request)) else "この画面で停止できる実行中ジョブはありません。"


def load_result(key):
    if not key:
        return None, "", [], ""
    try:
        directory = STUDIO.artifact(key)
        metadata = read_json(directory / "studio-result.json")
        files = []
        for name in ("audio.flac", "audio.wav", "score.abc", "project.json", "result.json", "config.json", "studio-result.json"):
            path = inside(STUDIO.outputs, directory / name)
            if path.is_file():
                files.append(str(path))
        audio = next((p for p in files if p.endswith((".flac", ".wav"))), None)
        score = directory / "score.abc"
        abc = read_score(inside(STUDIO.outputs, score)) if score.is_file() else ""
        warning = "楽譜のみを保存しました。" if metadata.get("plan_only") else ""
        if metadata.get("truncated"):
            warning += " トークン上限に到達しています。曲や歌詞が途中で終わっていないか確認してください。"
        elif metadata.get("truncated") is None:
            warning += " audio.cppは上限到達の自動判定ができません。曲の終わりを試聴して確認してください。"
        return audio, abc, files, warning
    except Exception as exc:
        return None, "", [], str(exc)


def read_score(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(300001)
    if len(data) > 300000:
        raise YuE2Error("楽譜は300 KB以内にしてください。")
    text = data.decode("utf-8-sig")
    if len(text) > 100000 or "\x00" in text:
        raise YuE2Error("楽譜の形式が不正です。")
    return text


def values_for(request: Request):
    data = asdict(request)
    data["seed"] = str(data["seed"])
    return [data[name] for name in FIELDS]


def restore(key):
    return values_for(import_project(STUDIO.artifact(key) / "project.json"))


def restore_file(path):
    return values_for(import_project(Path(path))) if path else [gr.update()] * len(FIELDS)


def use_score(score, cot):
    if not score.strip():
        raise gr.Error("この候補には楽譜がありません。")
    return score, "full" if cot == "off" else cot


def check_runtime():
    messages = []
    for engine, label in (("official", "公式Python"), ("cpp", "audio.cpp")):
        try:
            entry = runtime_manifest(RUNTIME, engine)
            fields = ("python", "model", "vae") if engine == "official" else ("binary", "models")
            missing = [name for name in fields if not Path(entry.get(name, "__missing__")).exists()]
            messages.append(f"{label}: " + ("不足: " + ", ".join(missing) if missing else "登録済み（GPU生成は未確認）"))
        except (OSError, ValueError) as exc:
            messages.append(f"{label}: {exc}")
    return "\n".join(messages)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, elem_id="aikimi-yue2") as tab:
        gr.Markdown("## YuE2 Music\n歌詞から作曲。楽譜を編集して、別のアレンジへ。")
        with gr.Accordion("実行環境・利用条件", open=False):
            gr.Markdown(
                "初回のみリポジトリ直下の **`aikimi-yue2-setup.bat`** を実行します。"
                "専用Python 3.12環境を作り、モデルを別途ダウンロードします。Forgeの依存関係は変更しません。\n\n"
                "公式の出発点はLinux・BF16対応NVIDIA GPU・24GB VRAMです。Windowsと16GB環境での実生成は検証が必要です。"
                "日本語の歌唱品質も未検証です。\n\n"
                "モデルの表示ライセンスはCC BY-NC 4.0です。"
                "[公式組織の回答](https://huggingface.co/m-a-p/YuE2-3B/discussions/5)では、"
                "個人クリエイター・音楽家・研究者は出力の収益化も可能、企業には商用ライセンスが必要と説明されています。"
                "第三者の楽曲や追加モデルの権利は別途確認してください。"
            )
            check = gr.Button("導入状態を確認", size="sm")
            environment = gr.Textbox(label="導入状態", interactive=False, lines=3)
        with gr.Row():
            with gr.Column(scale=5, min_width=340):
                title = gr.Textbox(label="曲名（保存・履歴用）", max_lines=1)
                style = gr.Textbox(label="曲調・楽器・歌声", lines=3,
                                   placeholder="例: English, piano pop, warm lead vocal, violin, gentle drums, 90 BPM")
                lyrics = gr.Textbox(label="歌詞", lines=10,
                                    placeholder="[Verse]\n歌詞を入力\n\n[Chorus]\nサビの歌詞\n\nインストは [instrumental]")
                with gr.Accordion("楽譜から作る・アレンジする", open=False):
                    abc = gr.Textbox(label="ABC楽譜（空欄なら新しく作曲）", lines=8)
                    score_file = gr.File(label="ABCを読み込む", file_types=[".abc", ".txt"], type="filepath")
                    gr.Markdown("曲調を変え、メロディを指定して再生成できます。新しい録音を作る機能で、元音声の部分修正ではありません。")
                with gr.Accordion("生成設定", open=False):
                    engine = gr.Radio([("公式Python", "official"), ("audio.cpp · GGUF", "cpp")], value="official", label="エンジン")
                    cot = gr.Radio([("メロディ＋コード", "full"), ("メロディ（伴奏は自由）", "melody"), ("楽譜なし", "off")], value="full", label="作曲方式")
                    with gr.Row():
                        seed = gr.Textbox(value="-1", label="Seed（-1: ランダム）")
                        candidates = gr.Slider(1, 8, value=1, step=1, label="候補数（順番に生成）")
                    with gr.Row():
                        steps = gr.Slider(1, 128, value=32, step=1, label="音響合成Steps")
                        max_tokens = gr.Slider(200, 9000, value=9000, step=100, label="音声トークン上限（秒数ではありません）")
                    with gr.Group() as native_controls:
                        memory_gib = gr.Number(value=0, precision=0, label="VRAM予算 GiB（0: 自動）")
                        offload = gr.Checkbox(value=True, label="ARモデルをCPUへ退避（VRAM節約・転送時間が増加）")
                        fp8 = gr.Checkbox(value=False, label="実験: FP8 AR（Compute Capability 8.9以上のみ）")
                    gguf = gr.Dropdown([("Q8_0", "q8_0"), ("Q4_0", "q4_0"), ("BF16", "bf16")], value="q8_0", label="導入済みGGUF形式", visible=False)
                    gr.Markdown("FP8・量子化の音質と速度は未検証です。audio.cpp v0.8.0では生成したABCの書き出しとLoRAはこの統合の対象外です。")
                with gr.Row():
                    generate = gr.Button("曲を生成", variant="primary")
                    plan = gr.Button("楽譜だけ作る")
                    stop = gr.Button("停止", interactive=False)
                status = gr.Textbox(value="未実行", label="進行状況", interactive=False, lines=2)
            with gr.Column(scale=4, min_width=340):
                history = gr.Dropdown(choices=[], label="生成履歴・候補", interactive=True)
                refresh = gr.Button("履歴を更新", size="sm")
                audio = gr.Audio(label="試聴 A", interactive=False, type="filepath")
                warning = gr.Textbox(label="結果の確認", interactive=False, lines=2)
                with gr.Accordion("別の候補と比較", open=False):
                    comparison = gr.Dropdown(choices=[], label="比較する候補", interactive=True)
                    audio_b = gr.Audio(label="試聴 B", interactive=False, type="filepath")
                with gr.Accordion("この候補の楽譜", open=False):
                    score_result = gr.Textbox(label="生成されたABC", lines=8, interactive=False)
                    edit = gr.Button("この楽譜を編集に使う", size="sm")
                restore_button = gr.Button("この候補の入力・Seedを復元", size="sm")
                files = gr.File(label="音声・楽譜・生成条件", file_count="multiple", interactive=False)
                with gr.Accordion("プロジェクトを読み込む", open=False):
                    project_file = gr.File(label="project.json", file_types=[".json"], type="filepath")
                gr.Markdown("保存先: `outputs/yue2/`。再生成は別の履歴になり、元データを上書きしません。")
        controls = dict(title=title, style=style, lyrics=lyrics, abc=abc, cot=cot, engine=engine,
                        seed=seed, candidates=candidates, steps=steps, max_tokens=max_tokens,
                        memory_gib=memory_gib, offload=offload, fp8=fp8, gguf=gguf)
        ordered = [controls[name] for name in FIELDS]
        job = gr.State("")
        timer = gr.Timer(1, active=False)
        normal_mode, plan_mode = gr.State(False), gr.State(True)
        output_controls = [job, status, generate, plan, stop, timer]
        for button, mode in ((generate, normal_mode), (plan, plan_mode)):
            button.click(run, inputs=[*ordered, mode], outputs=output_controls,
                         concurrency_limit=1, concurrency_id="aikimi-yue2-submit", trigger_mode="once", **PRIVATE)
        timer.tick(poll, inputs=job, outputs=[status, generate, plan, stop, timer, history, comparison], **PRIVATE)
        stop.click(cancel, inputs=job, outputs=status, queue=False, **PRIVATE)
        history.change(load_result, inputs=history, outputs=[audio, score_result, files, warning], **PRIVATE)
        comparison.change(lambda key: load_result(key)[0], inputs=comparison, outputs=audio_b, **PRIVATE)
        refresh.click(lambda: (gr.update(choices=STUDIO.history(), value=None), gr.update(choices=STUDIO.history(), value=None)),
                      outputs=[history, comparison], **PRIVATE)
        restore_button.click(restore, inputs=history, outputs=ordered, **PRIVATE)
        project_file.change(restore_file, inputs=project_file, outputs=ordered, **PRIVATE)
        score_file.change(lambda path: read_score(Path(path)) if path else gr.update(), inputs=score_file, outputs=abc, **PRIVATE)
        edit.click(use_score, inputs=[score_result, cot], outputs=[abc, cot], **PRIVATE)
        engine.change(lambda mode: (gr.update(visible=mode == "official"), gr.update(visible=mode == "cpp"), False if mode == "cpp" else gr.update()),
                      inputs=engine, outputs=[native_controls, gguf, fp8], **PRIVATE)
        check.click(check_runtime, outputs=environment, **PRIVATE)
    return [(tab, "YuE2 Music", "aikimi_yue2_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs)
