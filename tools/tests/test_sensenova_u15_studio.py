import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from PIL import Image

from modules_forge.sensenova_u15_bridge import RuntimeStatus

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "extensions-builtin"
    / "sensenova-u15-studio"
    / "scripts"
    / "sensenova_u15_studio.py"
)
STYLE = ROOT / "extensions-builtin" / "sensenova-u15-studio" / "style.css"
JAVASCRIPT = (
    ROOT
    / "extensions-builtin"
    / "sensenova-u15-studio"
    / "javascript"
    / "sensenova_u15_studio.js"
)
BRIDGE = ROOT / "modules_forge" / "sensenova_u15_bridge.py"


def load_studio_module():
    modules_package = ModuleType("modules")
    callbacks_module = ModuleType("modules.script_callbacks")
    callbacks_module.on_ui_tabs = mock.Mock()
    paths_module = ModuleType("modules.paths")
    paths_module.data_path = str(ROOT)
    modules_package.script_callbacks = callbacks_module

    spec = importlib.util.spec_from_file_location("_test_sensenova_u15_studio", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "modules": modules_package,
        "modules.script_callbacks": callbacks_module,
        "modules.paths": paths_module,
    }
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class SenseNovaStudioSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.style = STYLE.read_text(encoding="utf-8")
        cls.javascript = JAVASCRIPT.read_text(encoding="utf-8")
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.studio = load_studio_module()

    def test_dedicated_tab_is_registered(self):
        self.assertIn('"SenseNova U1.5"', self.script)
        self.assertIn(
            'script_callbacks.on_ui_tabs(_build_ui, name="sensenova_u15_studio")',
            self.script,
        )

    def test_multi_image_order_controls_are_present(self):
        self.assertIn("gr.Gallery", self.script)
        self.assertIn("末尾へ追加", self.script)
        self.assertIn("選択ファイルを一括追加", self.script)
        self.assertIn('file_count="multiple"', self.script)
        self.assertIn('elem_id="sn-bulk-add"', self.script)
        self.assertIn(".sn-bulk-row", self.style)
        self.assertIn("選択画像を差し替え", self.script)
        self.assertIn("_move_reference", self.script)
        self.assertIn(
            'label=f"参照画像（最大{MAX_REFERENCE_IMAGES}枚）"', self.script
        )

    def test_final_int8_convrot_is_the_fixed_model(self):
        self.assertIn("gr.State(QUANT_INT8_CONVROT)", self.script)
        self.assertIn("正式版 · INT8 ConvRot", self.script)
        self.assertNotIn("QUANT_BF16", self.script)
        self.assertNotIn("Q8_0 GGUF", self.script)
        self.assertIn(
            "community-maintained",
            (ROOT / "download_sensenova_u15_int8.ps1").read_text(encoding="utf-8"),
        )
        self.assertIn("interactive=False", self.script)

    def test_generation_and_cancel_share_job_state(self):
        with mock.patch.object(self.studio, "cancel_generation", return_value="停止しました") as cancel:
            _, empty_button = self.studio._cancel("")
            cancel.assert_not_called()
            self.assertFalse(empty_button["interactive"])
            _, own_button = self.studio._cancel("this-session-job")
            cancel.assert_called_once_with("this-session-job")
            self.assertFalse(own_button["interactive"])

    def test_stop_is_enabled_only_after_this_generation_has_a_job_id(self):
        with (
            mock.patch.object(self.studio, "_request_from_ui"),
            mock.patch.object(self.studio, "run_generation", return_value=iter([
                {"stage": "prepare", "message": "準備中", "job_id": "own-job"},
                {"stage": "complete", "message": "完了", "job_id": "own-job"},
            ])),
        ):
            updates = list(self.studio._generate(*([None] * 18)))
        self.assertEqual([value[6]["interactive"] for value in updates], [False, True, False])
        self.assertEqual([value[5] for value in updates], ["", "own-job", ""])

    def test_measured_24gb_safe_defaults_are_visible(self):
        self.assertIn('value="2048x2048"', self.script)
        self.assertIn("24GB Safe · 2K出力優先", self.script)
        self.assertIn("各約0.26MP · 比率保護", self.script)
        self.assertIn("元の入力1枚目を基準", self.bridge)
        self.assertIn("Uncapped streaming", self.script)
        updates = self.studio._mode_updates(
            self.studio.MODE_EDIT, "2048x2048", True
        )
        self.assertEqual(updates[1]["value"], "auto")
        self.assertEqual(updates[3]["value"], str(512 * 512))
        self.assertEqual(updates[5]["value"], self.studio.PROFILE_QUALITY)
        self.assertEqual(updates[6]["value"], 50)
        self.assertEqual(updates[7]["value"], 4.0)

    def test_missing_official_lora_defaults_explicitly_to_quality(self):
        updates = self.studio._mode_updates(
            self.studio.MODE_TEXT, "2048x2048", False
        )
        self.assertEqual(updates[5]["value"], self.studio.PROFILE_QUALITY)
        self.assertEqual(updates[6]["value"], 50)
        self.assertEqual(updates[7]["value"], 4.0)
        self.assertTrue(updates[6]["interactive"])
        self.assertIn("LoRAが未準備", updates[4]["value"])

        status = RuntimeStatus(
            ready=True,
            source_ready=True,
            dependencies_ready=True,
            checkpoint_ready=True,
            source_path=ROOT / "runtime-final",
            checkpoint_path=ROOT / "model.safetensors",
            messages=("quality ready",),
            lora_ready=False,
        )
        with mock.patch.object(self.studio, "inspect_runtime", return_value=status):
            refreshed = self.studio._refresh_runtime(
                "runtime", "checkpoint", self.studio.MODE_TEXT, self.studio.PROFILE_OFFICIAL_8STEP
            )
        self.assertFalse(refreshed[1])
        self.assertEqual(refreshed[2]["value"], self.studio.PROFILE_QUALITY)
        self.assertEqual(refreshed[3]["value"], 50)
        self.assertIn("Quality生成できます", refreshed[0])

    def test_official_8step_and_quality_profiles_update_as_one_preset(self):
        self.assertIn("公式8-Step · 高速T2I", self.script)
        fast = self.studio._profile_updates(self.studio.PROFILE_OFFICIAL_8STEP)
        self.assertEqual([update["value"] for update in fast], [8, 1.0, 3.0])
        self.assertTrue(all(update["interactive"] is False for update in fast))

        quality = self.studio._profile_updates(self.studio.PROFILE_QUALITY)
        self.assertEqual(
            [update["value"] for update in quality], [50, 4.0, 3.0]
        )
        self.assertTrue(all(update["interactive"] is True for update in quality))

    def test_responsive_and_accessible_ui_contracts(self):
        self.assertIn("@media (max-width: 640px)", self.style)
        self.assertIn("prefers-reduced-motion", self.style)
        self.assertIn("prefers-reduced-transparency", self.style)
        self.assertIn("prefers-contrast: more", self.style)
        self.assertIn("forced-colors: active", self.style)
        self.assertIn(":focus-visible", self.style)
        self.assertIn('role="status"', self.bridge)
        self.assertIn("Ctrl / ⌘ + Enter", self.script)
        self.assertIn("onUiLoaded(setupStudio)", self.javascript)
        self.assertIn("window.localStorage", self.javascript)
        self.assertNotIn("function restoreDraft", self.javascript)
        self.assertIn("sensenova-studio-active", self.javascript)
        self.assertIn('aria-keyshortcuts", "Control+Enter Meta+Enter"', self.javascript)
        self.assertIn("min-height: 44px", self.style)

    def test_real_gradio_ui_builds_with_unique_controls(self):
        status = RuntimeStatus(
            ready=False,
            source_ready=True,
            dependencies_ready=True,
            checkpoint_ready=False,
            source_path=ROOT / "runtime-final",
            checkpoint_path=ROOT / "model.safetensors",
            messages=("test",),
        )
        with mock.patch.object(self.studio, "inspect_runtime", return_value=status):
            interface = self.studio._build_ui()[0][0]
        config = interface.get_config_file()
        element_ids = [
            component["props"].get("elem_id")
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        ]
        self.assertEqual(len(element_ids), len(set(element_ids)))
        self.assertIn("sn-reference-gallery", element_ids)
        self.assertIn("sn-reference-bulk-upload", element_ids)
        self.assertIn("sn-generate", element_ids)
        self.assertIn("sn-result-png", element_ids)
        self.assertIn("sn-result-json", element_ids)
        self.assertIn("sn-advanced", element_ids)
        self.assertIn("sn-metadata-panel", element_ids)
        self.assertIn("sn-runtime-setup", element_ids)

        profile = next(
            component["props"]
            for component in config["components"]
            if component.get("props", {}).get("label") == "生成プロファイル"
        )
        self.assertEqual(profile["value"], self.studio.PROFILE_QUALITY)
        refresh_dependencies = [
            dependency
            for dependency in config["dependencies"]
            if isinstance(dependency.get("api_name"), str)
            and dependency["api_name"].startswith("_refresh_runtime")
        ]
        self.assertEqual(len(refresh_dependencies), 2)
        self.assertTrue(
            all(len(dependency["outputs"]) == 6 for dependency in refresh_dependencies)
        )

        prompt_id = next(
            component["id"]
            for component in config["components"]
            if component.get("props", {}).get("elem_id") == "sn-prompt"
        )
        draft_loads = [
            dependency
            for dependency in config["dependencies"]
            if dependency.get("js")
            and dependency.get("outputs") == [prompt_id]
            and "localStorage.getItem" in dependency["js"]
        ]
        self.assertEqual(len(draft_loads), 1)
        self.assertFalse(draft_loads[0]["backend_fn"])
        self.assertLess(
            self.script.index('elem_id="sn-validation"'),
            self.script.index('elem_classes=["sn-actions"]'),
        )
        self.assertLess(
            self.script.index('elem_id="sn-generate"'),
            self.script.index('elem_id="sn-result-image"'),
        )
        self.assertLess(
            self.script.index('elem_id="sn-result-image"'),
            self.script.index('elem_id="sn-runtime-setup"'),
        )
        self.assertNotIn('class="sn-eyebrow"', self.script)

    def test_bulk_reference_files_keep_selection_order(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.png"
            second = Path(temp) / "second.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(first)
            Image.new("RGB", (8, 8), (0, 0, 255)).save(second)
            gallery, uploads, order, selected = self.studio._append_reference_files(
                [], [str(first), str(second)]
            )
        values = gallery["value"]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(values[1][0].getpixel((0, 0)), (0, 0, 255))
        self.assertIsNone(uploads["value"])
        self.assertIn("Image-1", order)
        self.assertEqual(selected, 1)

    def test_gallery_selection_event_keeps_zero_based_index(self):
        self.assertEqual(self.studio._select_reference(mock.Mock(index=3)), 3)
        self.assertEqual(self.studio._select_reference(None), -1)

    def test_deselected_gallery_item_is_not_a_destructive_action_target(self):
        self.assertEqual(self.studio._select_reference(mock.Mock(index=1, selected=False)), -1)

    def test_bulk_add_only_decodes_files_that_fit(self):
        existing = [(Image.new("RGB", (8, 8)), None)] * 63
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first.png"
            Image.new("RGB", (8, 8), "red").save(path)
            update, _, message, index = self.studio._append_reference_files(existing, [str(path), "not-read.png"])
        self.assertEqual(len(update["value"]), 64)
        self.assertEqual(index, 63)
        self.assertIn("1枚", message)

    def test_refresh_preserves_edit_profile_and_custom_sampling(self):
        status = mock.Mock(lora_ready=True)
        with (
            mock.patch.object(self.studio, "inspect_runtime", return_value=status),
            mock.patch.object(self.studio, "runtime_status_html", return_value="ready"),
        ):
            updates = self.studio._refresh_runtime(
                "runtime", "checkpoint", self.studio.MODE_EDIT, self.studio.PROFILE_QUALITY
            )
        self.assertTrue(updates[1])
        self.assertTrue(all("value" not in update for update in updates[2:]))

    def test_reordering_relabels_references_and_tracks_the_moved_image(self):
        red = Image.new("RGB", (24, 16), "red")
        blue = Image.new("RGB", (16, 24), "blue")
        update, _, index = self.studio._move_reference(1, [(red, None), (blue, None)], -1)
        self.assertEqual(index, 0)
        self.assertEqual(update["selected_index"], 0)
        self.assertIs(update["value"][0][0], blue)
        self.assertEqual(update["value"][0][1], "Image-1 · 16×24")
        controls = self.studio._selection_controls(update["value"], index)
        self.assertFalse(controls[1]["interactive"])
        self.assertTrue(controls[2]["interactive"])
        self.assertIn('aria-current="true"', controls[-1])
        deleted = self.studio._after_gallery_delete(update["value"], mock.Mock(index=1))
        self.assertEqual(len(deleted[0]["value"]), 1)
        self.assertIs(deleted[0]["value"][0][0], blue)
        self.assertEqual(deleted[2], -1)
        self.assertIsNone(deleted[0]["selected_index"])

    def test_reference_instruction_preserves_prompt_and_does_not_duplicate(self):
        gallery = [(Image.new("RGB", (8, 8)), None)] * 2
        result = self.studio._append_reference_instruction("Keep the blue scarf.", "outfit", gallery)
        self.assertTrue(result.startswith("Keep the blue scarf.\n\n"))
        self.assertIn("Image-1", result)
        self.assertIn("Image-2", result)
        self.assertNotIn("value", self.studio._append_reference_instruction(result, "outfit", gallery))
        with mock.patch.object(self.studio.gr, "Warning"):
            self.assertNotIn("value", self.studio._append_reference_instruction("Keep it.", "outfit", gallery[:1]))

    def test_reuse_result_only_replaces_first_reference(self):
        second = Image.new("RGB", (8, 8), "blue")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.png"
            Image.new("RGB", (16, 24), "red").save(path)
            update, _, selected, mode = self.studio._reuse_result(str(path), [(second, None), (second, None)])
        self.assertEqual(len(update["value"]), 2)
        self.assertEqual(update["value"][0][0].getpixel((0, 0)), (255, 0, 0))
        self.assertIs(update["value"][1][0], second)
        self.assertEqual(selected, 0)
        self.assertEqual(mode, self.studio.MODE_EDIT)

    def test_one_megapixel_auto_keeps_reference_aspect_without_forcing_four_megapixels(self):
        image = Image.new("RGB", (768, 1024))
        request = self.studio._request_from_ui(
            mode=self.studio.MODE_EDIT,
            prompt="Recolor the coat.",
            gallery=[(image, None)],
            model_path=self.studio.DEFAULT_MODEL_ID,
            quantization=self.studio.QUANT_INT8_CONVROT,
            checkpoint_path=str(self.studio.DEFAULT_CHECKPOINT_PATH),
            source_path=str(self.studio.DEFAULT_SOURCE_PATH),
            resolution="auto_1mp",
            input_max_pixels=str(512 * 512),
            generation_profile=self.studio.PROFILE_QUALITY,
            steps=50,
            cfg_scale=4,
            img_cfg_scale=1,
            timestep_shift=3,
            seed=42,
            vram_mode="low",
            attn_backend="sdpa",
            dtype="bfloat16",
        )
        self.assertIsNone(request.width)
        self.assertIsNone(request.height)
        self.assertEqual(request.target_pixels, 1024 * 1024)
        self.assertEqual(request.input_images[0].size, image.size)


if __name__ == "__main__":
    unittest.main()
