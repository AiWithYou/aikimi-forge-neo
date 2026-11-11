"""P-H3 interface behavior independent of model or GPU availability."""

import importlib.util
import io
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr
import pytest

from modules_forge import p_h3_image_ui as ui


def test_edit_target_and_references_are_not_promoted_or_silently_reordered():
    request = ui.make_request("edit", "Change the hat", "target.png", ["a.png", "b.png"], 2048, 2048, "42", False, "")
    assert request.images == ("target.png", "a.png", "b.png")
    text = ui.make_request("text", "A photo", "stale.png", ["stale-ref.png"], 2912, 2912, "-1", False, "")
    assert text.images == ()


def test_lora_selector_lists_the_current_shared_model_directory(tmp_path):
    runtime = tmp_path / "runtime"
    folder = runtime / "models/loras/styles"
    folder.mkdir(parents=True)
    (folder / "pastel.safetensors").touch()
    (folder / "notes.txt").touch()
    with patch.object(ui, "managed_runtime_root", return_value=runtime):
        choices = ui.lora_choices()
        selection = str(Path("styles") / "pastel.safetensors")
        assert choices == [("使用しない", ""), (selection, selection)]
        assert ui.refresh_loras(selection)["value"] == selection
        assert ui.refresh_loras("missing.safetensors")["value"] == ""


def test_lora_controls_forward_selection_and_strength():
    request = ui.make_request("edit", "Change the hat", "target.png", [], 512, 512, "42", False, "",
                              "pastel.safetensors", 0.7)
    assert (request.lora, request.lora_strength) == ("pastel.safetensors", 0.7)
    with gr.Blocks() as app:
        ui.render(app)
    labels = {component.get("props", {}).get("label") for component in app.get_config_file()["components"]}
    assert {"LoRA", "LoRAの強度"} <= labels


def test_size_summary_distinguishes_high_load_and_official_examples():
    assert "8.48 MP" in ui.size_summary(2912, 2912, "text", None, [])
    assert "高解像度" not in ui.size_summary(2912, 2912, "text", None, [])
    assert "高解像度" in ui.size_summary(4096, 4096, "edit", "target.png", ["ref.png"])
    assert "公式掲載例の範囲外" in ui.size_summary(8192, 4096, "text", None, [])


def test_preset_updates_both_dimensions_and_the_summary_together():
    width, height, summary = ui.preset_updates("軽量 4 MP · 正方形", "text", None, [])
    assert (width, height) == (2048, 2048)
    assert "2048 × 2048" in summary
    assert "4.19 MP" in summary


def test_mode_switch_preserves_each_modes_last_size():
    state = {"active": "text", "text": ("推奨 8 MP · 正方形", 2912, 2912), "edit": ("軽量 4 MP · 正方形", 2048, 2048)}
    _, preset, width, height, _, state = ui.mode_updates("edit", "軽量 4 MP · 正方形", 2048, 2048, None, [], state)
    assert (preset, width, height) == ("軽量 4 MP · 正方形", 2048, 2048)
    _, preset, width, height, _, state = ui.mode_updates("text", "推奨 8 MP · 縦長 3:4", 2496, 3360, "target", [], state)
    assert (preset, width, height) == ("軽量 4 MP · 正方形", 2048, 2048)
    _, preset, width, height, _, _ = ui.mode_updates("edit", preset, width, height, None, [], state)
    assert (preset, width, height) == ("推奨 8 MP · 縦長 3:4", 2496, 3360)


@pytest.mark.parametrize("mode,preset,width,height", [
    ("text", "軽量 4 MP · 正方形", 2048, 2048),
    ("edit", "推奨 8 MP · 正方形", 2496, 2048),
])
def test_saved_defaults_initialize_matching_summary_and_per_mode_state(mode, preset, width, height):
    visibility, shown_preset, summary, state = ui.initial_size_state(mode, preset, width, height, None, [])
    expected_preset = preset if ui.PRESETS.get(preset) == (width, height) else "カスタム"
    assert visibility["visible"] == (mode == "edit")
    assert shown_preset == expected_preset
    assert f"{width} × {height}" in summary
    assert state["active"] == mode
    assert state[mode] == (expected_preset, width, height)


def test_reusing_a_result_clears_old_references():
    assert ui.reuse_result("result.png") == ("result.png", [], "edit")


def test_generation_uses_current_runtime_if_setup_changed_after_page_load(tmp_path):
    current = tmp_path / "new-runtime"
    event = {"stage": "complete", "message": "done", "path": "image.png", "files": [], "metadata": {}}
    with patch.object(ui, "managed_runtime_root", return_value=current), patch.object(ui.ph3, "run_generation", return_value=iter([event])) as run:
        list(ui._generate(str(tmp_path / "old-runtime"), "fast", "text", "photo", None, [], 2912, 2912, "42", False, ""))
    assert run.call_args.args[1] == current


def test_generation_uses_the_active_forge_data_directory(tmp_path):
    data = tmp_path / "separate-data"
    event = {"stage": "complete", "message": "done", "path": "image.png", "files": [], "metadata": {}}
    with patch.object(ui, "managed_runtime_root", return_value=tmp_path / "runtime"), patch.object(ui.ph3, "run_generation", return_value=iter([event])) as run:
        list(ui._generate("runtime", "fast", "text", "photo", None, [], 512, 512, "42", False, "", data_directory=data))
    assert run.call_args.args[3] == data / "logs" / "minimax_h3"
    assert run.call_args.args[4] == data / "outputs" / "minimax_h3" / "p_h3_images"


def test_forge_data_directory_is_passed_to_the_generation_callback(tmp_path):
    with gr.Blocks() as app:
        ui.render(app, data_directory=tmp_path)
    callbacks = [function.fn for function in app.fns.values() if getattr(function.fn, "func", None) is ui._generate]
    assert len(callbacks) == 1
    assert callbacks[0].keywords["data_directory"] == tmp_path


def test_hash_verification_exposes_cancel_and_reports_a_normal_stop(tmp_path):
    def run(*_args, **_kwargs):
        yield {"stage": "verifying", "message": "本体を確認中", "prompt_id": "job-1"}
        raise ui.ph3.PH3Cancelled("画像生成を停止しました。")

    with patch.object(ui, "managed_runtime_root", return_value=tmp_path), patch.object(ui.ph3, "run_generation", side_effect=run):
        events = list(ui._generate("runtime", "fast", "text", "photo", None, [], 512, 512, "42", False, ""))
    assert events[1][5]["prompt_id"] == "job-1"
    assert events[1][7]["interactive"] is True
    assert "停止しました" in events[-1][0]
    assert "確認が必要" not in events[-1][0]


def test_setup_uses_current_runtime_if_setup_changed_after_page_load(tmp_path):
    current = tmp_path / "new-runtime"
    bridge = types.SimpleNamespace(runtime_setup_session=Mock(return_value=nullcontext()))
    process = Mock(stdout=io.StringIO("ready\n"))
    process.wait.return_value = 0
    process.poll.return_value = 0
    with (
        patch.object(ui, "managed_runtime_root", return_value=current),
        patch.object(ui, "installed_runtime_root", return_value=current),
        patch.object(ui.ph3, "_bridge", return_value=bridge),
        patch.object(ui.subprocess, "Popen", return_value=process),
    ):
        events = list(ui._setup(str(tmp_path / "old-runtime")))
    bridge.runtime_setup_session.assert_called_once_with(current, ui.SERVER_URL)
    assert events[-1][0] == str(current)


def test_preparation_status_accepts_selected_model_without_optional_upsampler(tmp_path):
    runtime = tmp_path / "runtime"
    for relative in (f"text_encoders/{ui.ph3.TEXT_ENCODER}", f"vae/{ui.ph3.VAE}"):
        file = runtime / "models" / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    selected = tmp_path / "custom.safetensors"
    selected.touch()
    with (patch.object(ui, "installed_runtime_root", return_value=runtime),
          patch("modules_forge.minimax_h3_runtime.candidate_runtime_root", return_value=runtime)):
        status = ui.initial_status(str(selected))
    assert "必要ファイル" in status
    assert "オフ" in status
    assert "未準備" not in status


def test_p_h3_controls_build_inside_the_existing_blocks_context():
    with gr.Blocks() as app:
        with gr.Tab("P-H3"):
            ui.render(app)
    config = app.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}
    assert "編集する画像 · Picture 1" in labels
    assert "参照画像 · Picture 2〜8" in labels
    assert "プロンプト補助" in labels
    assert "ネガティブプロンプト" not in labels
    reference = next(component for component in config["components"] if component.get("props", {}).get("label") == "参照画像 · Picture 2〜8")
    assert reference["props"]["allow_reordering"] is True
    assert any(function.fn is ui.initial_size_state for function in app.fns.values())


def test_h3_image_tab_exposes_p_h3_and_legacy_methods():
    import modules.aikimi_status  # noqa: F401 -- initialize before replacing the modules package.

    root = Path(__file__).resolve().parents[2]
    module = types.ModuleType("modules")
    module.script_callbacks = types.SimpleNamespace(on_ui_tabs=Mock())
    paths = types.ModuleType("modules.paths")
    paths.data_path = paths.script_path = str(root)
    spec = importlib.util.spec_from_file_location("h3_image_ui_both", root / "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py")
    script = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modules": module, "modules.paths": paths, "modules.script_callbacks": module.script_callbacks}):
        spec.loader.exec_module(script)
        with gr.Blocks() as app:
            tab = script.on_ui_tabs()[0][0]
    assert tab is not None
    labels = {component.get("props", {}).get("label") for component in app.get_config_file()["components"]}
    assert "P-H3 Image" in labels
    assert "標準H3（実験）" in labels


@pytest.mark.parametrize("existing_settings", [False, True])
def test_h3_methods_have_separate_saved_input_keys(tmp_path, existing_settings):
    import modules.aikimi_status  # noqa: F401 -- initialize before replacing the modules package.

    root = Path(__file__).resolve().parents[2]
    components = types.ModuleType("modules.ui_components")
    components.InputAccordionImpl = type("InputAccordionImpl", (), {})
    components.ToolButton = gr.Button
    loadsave_spec = importlib.util.spec_from_file_location("test_p_h3_loadsave", root / "modules/ui_loadsave.py")
    loadsave_module = importlib.util.module_from_spec(loadsave_spec)
    with patch.dict(sys.modules, {"modules.ui_components": components}):
        loadsave_spec.loader.exec_module(loadsave_module)
    module = types.ModuleType("modules")
    module.script_callbacks = types.SimpleNamespace(on_ui_tabs=Mock())
    paths = types.ModuleType("modules.paths")
    paths.data_path = paths.script_path = str(root)
    spec = importlib.util.spec_from_file_location("h3_image_ui_config", root / "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py")
    script = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modules": module, "modules.paths": paths, "modules.script_callbacks": module.script_callbacks}):
        spec.loader.exec_module(script)
        with gr.Blocks():
            tab = script.on_ui_tabs()[0][0]
    config = tmp_path / "ui-config.json"
    if existing_settings:
        config.write_text(json.dumps({"minimax_h3_image_studio/幅 px/value": 768,
                                      "minimax_h3_image_studio/高さ px/value": 768,
                                      "minimax_h3_image_studio/プロンプト/value": "legacy"}), encoding="utf-8")
    loadsave = loadsave_module.UiLoadsave(str(config))
    loadsave.add_block(tab, "minimax_h3_image_studio")
    default = "minimax_h3_image_studio"
    separate = "customscript/p_h3_image/" + default
    assert loadsave.ui_settings[f"{separate}/幅 px/value"] == 2912
    official = "customscript/minimax_h3_image_official/" + default
    assert loadsave.ui_settings[f"{official}/幅 px/value"] == 2048
    assert loadsave.ui_settings[f"{default}/幅 px/value"] == 768
    assert loadsave.ui_settings[f"{separate}/高さ px/value"] == 2912
    assert loadsave.ui_settings[f"{default}/高さ px/value"] == 768
    assert f"{separate}/プロンプト/value" in loadsave.ui_settings
    assert f"{default}/プロンプト/value" in loadsave.ui_settings
    # Checkpoints use the independent local-asset selection store, not ui-config.
    assert f"{separate}/チェックポイント/value" not in loadsave.ui_settings
    assert f"{default}/チェックポイント/value" not in loadsave.ui_settings
    assert loadsave.component_mapping[f"{separate}/幅 px/value"].value == 2912
    assert loadsave.component_mapping[f"{default}/幅 px/value"].value == 768
    if existing_settings:
        assert loadsave.component_mapping[f"{default}/プロンプト/value"].value == "legacy"
        assert loadsave.component_mapping[f"{separate}/プロンプト/value"].value != "legacy"
