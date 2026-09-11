import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABS_JS = ROOT / "extensions-builtin" / "aikimi-ui" / "javascript" / "aikimi_tabs.js"
AIKIMI_CSS = ROOT / "extensions-builtin" / "aikimi-ui" / "style.css"
ROOT_JS = ROOT / "script.js"
UI_JS = ROOT / "javascript" / "ui.js"
GENERATION_PARAMS_JS = ROOT / "javascript" / "generationParams.js"
EXTRA_NETWORKS_JS = ROOT / "javascript" / "extraNetworks.js"
CONTROLNET_UNITS_JS = ROOT / "extensions-builtin" / "sd_forge_controlnet" / "javascript" / "active_units.js"
ANIMA_SCRIPT = ROOT / "extensions-builtin" / "anima-3-8b" / "scripts" / "anima_3_8b.py"
H3_SCRIPT = ROOT / "extensions-builtin" / "minimax-h3-studio" / "scripts" / "minimax_h3_studio.py"
SENSENOVA_SCRIPT = ROOT / "extensions-builtin" / "sensenova-u15-studio" / "scripts" / "sensenova_u15_studio.py"
INPUT_ACCORDION_JS = ROOT / "javascript" / "inputAccordion.js"


class AikimiTabsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.javascript = TABS_JS.read_text(encoding="utf-8")

    def test_public_contract_and_change_event_are_stable(self):
        self.assertIn("window.AikimiTabs = Object.freeze({", self.javascript)
        self.assertIn("getActiveFeature: function ()", self.javascript)
        self.assertIn("getActiveContainer: function ()", self.javascript)
        self.assertIn('"aikimi:feature-tab-change"', self.javascript)
        self.assertIn("ready: !warning", self.javascript)
        self.assertIn("warning: warning || null", self.javascript)

    def test_feature_buttons_use_stable_ids_and_order(self):
        expected = (
            'buttonId: "aikimi-tab-krea2"',
            'buttonId: "aikimi-tab-anima38"',
            'buttonId: "aikimi-tab-sensenova"',
            'buttonId: "aikimi-tab-minimax-h3"',
        )
        positions = [self.javascript.index(value) for value in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('"krea2",\n        "anima38",\n        "sensenova",\n        "minimax_h3"', self.javascript)

    def test_aliases_reuse_forge_generation_tabs(self):
        self.assertIn('containerId: "tab_txt2img"', self.javascript)
        self.assertEqual(
            self.javascript.count('containerIds: Object.freeze(["tab_txt2img", "tab_img2img"])'),
            2,
        )
        self.assertIn('preset: "krea"', self.javascript)
        self.assertIn('preset: "anima"', self.javascript)
        self.assertIn('querySelector("#forge_ui_preset")', self.javascript)
        self.assertIn("function animaAccordionId(containerId)", self.javascript)
        self.assertIn("if (!containerId || !clickNativeTab(containerId))", self.javascript)
        self.assertIn('const FEATURE_NAV_ID = "aikimi-feature-nav"', self.javascript)
        self.assertNotIn("Krea2 2-Stage Upscale", self.javascript)

    def test_aliases_keep_the_selected_forge_mode_visually_selected(self):
        stylesheet = (ROOT / "extensions-builtin" / "aikimi-ui" / "style.css").read_text(encoding="utf-8")
        self.assertNotIn('[data-active-feature="krea2"] + #tabs', stylesheet)
        self.assertNotIn('[data-active-feature="anima38"] + #tabs', stylesheet)
        self.assertNotIn("gr.Blocks", self.javascript)

    def test_aliases_do_not_require_lazy_controls_before_base_tab_click(self):
        create_navigation = self.javascript[
            self.javascript.index("function createFeatureNavigation") : self.javascript.index(
                "function ensureFeatureNavigation"
            )
        ]
        self.assertNotIn("forge_ui_preset", create_navigation)
        self.assertNotIn("aikimi-txt2img-anima38", create_navigation)
        self.assertIn('querySelector("#forge_ui_preset")', self.javascript)
        self.assertIn("const accordionId = animaAccordionId(containerId)", self.javascript)
        self.assertIn("async function selectFeaturePreset(feature, sequence)", self.javascript)

    def test_preset_option_uses_semantic_exact_match_and_gradio6_mousedown(self):
        self.assertIn('option.getAttribute("aria-label")', self.javascript)
        self.assertIn('trimmed.startsWith("✓")', self.javascript)
        self.assertIn("normalizeGradioOptionLabel(candidate) === label", self.javascript)
        self.assertIn('input.getAttribute("aria-controls")', self.javascript)
        self.assertIn("const optionRoot = controlledListbox || dropdown", self.javascript)
        self.assertNotIn("appRoot().querySelectorAll(\"[role='option']\")", self.javascript)
        self.assertNotIn('startsWith("✔")', self.javascript)
        self.assertNotIn("candidate.includes", self.javascript)
        self.assertIn('new MouseEvent("mousedown"', self.javascript)
        self.assertIn("bubbles: true", self.javascript)
        self.assertIn("cancelable: true", self.javascript)
        self.assertIn("option.closest(\"ul[role='listbox']\")", self.javascript)
        self.assertNotIn("setNativeInputValue", self.javascript)
        self.assertNotIn("aikimi_select_krea2_script", self.javascript)
        self.assertIn('new KeyboardEvent("keydown"', self.javascript)
        self.assertIn('key: "ArrowDown"', self.javascript)
        self.assertIn('code: "ArrowDown"', self.javascript)
        select_preset = self.javascript[
            self.javascript.index("async function selectFeaturePreset") : self.javascript.index(
                "async function expandAnimaAccordion"
            )
        ]
        self.assertNotIn("input.click()", select_preset)

    def test_native_button_mapping_prefers_aria_controls_before_mounted_panels(self):
        self.assertIn('button.getAttribute("aria-controls") === containerId', self.javascript)
        self.assertIn("const panelIndex = topTabItems().findIndex", self.javascript)
        self.assertNotIn("dataset.aikimiControlsId", self.javascript)

    def test_anima_alias_verifies_the_real_gradio_boolean(self):
        self.assertIn("inputAccordionChecked(accordionId, true)", self.javascript)
        self.assertIn("controls.visibleCheckbox.checked", self.javascript)
        self.assertIn("controls.hiddenCheckbox.checked", self.javascript)
        self.assertNotIn("label.click();", self.javascript)

    def test_input_accordion_setup_recovers_after_lazy_mount(self):
        source = INPUT_ACCORDION_JS.read_text(encoding="utf-8")

        self.assertIn("if (!accordion.visibleCheckbox && !setupAccordion(accordion))", source)
        self.assertIn('accordion.dataset.inputAccordionReady = "true"', source)
        self.assertIn("onUiLoaded(setupInputAccordions)", source)
        self.assertIn("onUiUpdate(function ()", source)
        self.assertIn("setupInputAccordions();", source)

    def test_controlnet_active_units_waits_for_lazy_panels_once(self):
        source = CONTROLNET_UNITS_JS.read_text(encoding="utf-8")

        self.assertIn("MAX_INITIALIZATION_ATTEMPTS = 100", source)
        self.assertIn("const initializedByTab = new Map()", source)
        self.assertIn("function initializeControlNet(tabName)", source)
        self.assertIn("if (!ext) return false", source)
        self.assertIn("if (units.some((elements) => elements === null))", source)
        self.assertIn("onUiUpdate((records) =>", source)
        self.assertIn("mutationTouchesControlNet(records)", source)
        self.assertIn("onUiTabChange(() =>", source)
        self.assertNotIn("AllControlnet", source)
        self.assertNotIn(
            "document.getElementById(`${tab}_controlnet`).querySelector",
            source,
        )

    def test_extra_network_prompt_registration_recovers_after_lazy_mount(self):
        source = EXTRA_NETWORKS_JS.read_text(encoding="utf-8")

        self.assertIn("function registerExtraNetworkPrompt(tabname, id)", source)
        self.assertIn("if (!textarea) return false", source)
        self.assertIn('textarea.dataset.extraNetworksPromptRegistered === "true"', source)
        self.assertIn('search.dataset.extraNetworksInitialized === "true"', source)
        self.assertIn('tabnav.querySelector(":scope > .extra-networks-controls-div")', source)
        self.assertIn("onUiUpdate(function ()", source)
        self.assertIn("setupExtraNetworks();", source)
        self.assertIn("if (applyFunction) applyFunction(true)", source)

    def test_aikimi_group_is_an_external_sibling_of_forge_tabs(self):
        self.assertIn("tabs.parentElement.insertBefore(row, tabs)", self.javascript)
        self.assertIn('row.className = "aikimi-feature-nav"', self.javascript)
        self.assertNotIn("function placeFeatureButtons", self.javascript)
        self.assertNotIn("function decorateNativeFeatureButton", self.javascript)

    def test_programmatic_host_click_keeps_alias_activation(self):
        self.assertIn("activatingFeature && selectedButtonMatches", self.javascript)
        self.assertIn("if (sequence !== activationSequence) return;", self.javascript)
        self.assertIn("let activationQueue = Promise.resolve()", self.javascript)
        self.assertIn("function queueFeatureActivation(feature)", self.javascript)
        self.assertIn('FEATURES[activeFeature]?.kind === "alias"', self.javascript)

    def test_alias_aria_does_not_claim_the_native_selected_tab(self):
        self.assertIn('button.setAttribute("aria-current", "page")', self.javascript)
        self.assertIn('row.setAttribute("aria-label", "Aikimi機能")', self.javascript)
        self.assertIn("handleFeatureNavigationKeydown", self.javascript)
        self.assertNotIn('button.setAttribute("aria-selected"', self.javascript)

    def test_ui_lifecycle_hooks_are_idempotent(self):
        self.assertIn("window.AikimiTabs?.apiVersion === API_VERSION", self.javascript)
        self.assertIn("onUiLoaded(function ()", self.javascript)
        self.assertIn("onUiUpdate(handleUiUpdate)", self.javascript)
        self.assertIn("scheduleAliasReconciliation()", self.javascript)
        self.assertIn("repairCount >= REPAIR_LIMIT", self.javascript)
        self.assertIn("record.target === row || row?.contains(record.target)", self.javascript)
        self.assertIn("onUiTabChange(syncFromSelectedNativeTab)", self.javascript)

    def test_alias_state_is_reconciled_with_real_forge_controls(self):
        self.assertIn("function aliasStateMatches(feature)", self.javascript)
        self.assertIn("function presetMatches(feature)", self.javascript)
        self.assertIn('querySelector("#forge_ui_preset input")?.value === preset', self.javascript)
        self.assertIn("selectedButtonMatches(feature, selectedNativeButton())", self.javascript)
        self.assertIn("visibleCheckbox?.checked", self.javascript)
        self.assertIn("setActiveFeature(null, null)", self.javascript)

    def test_gradio_owned_tablist_is_never_mutated(self):
        native_lookup = self.javascript[
            self.javascript.index("function nativeButtonFor") : self.javascript.index("function featureContainer")
        ]
        for mutation in ("append(", "appendChild", "insertBefore", "remove(", "classList", ".dataset", ".style"):
            with self.subTest(mutation=mutation):
                self.assertNotIn(mutation, native_lookup)
        self.assertNotIn("topTabNav().append", self.javascript)
        self.assertNotIn("topTabNav().insertBefore", self.javascript)

    def test_anima_accordions_have_tab_specific_ids(self):
        source = ANIMA_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('tab_name = "img2img" if is_img2img else "txt2img"', source)
        self.assertIn('elem_id=f"aikimi-{tab_name}-anima38"', source)

    def test_native_studio_keeps_config_labels_and_external_row_uses_short_names(self):
        h3 = H3_SCRIPT.read_text(encoding="utf-8")
        sensenova = SENSENOVA_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('return [(interface, "H3 Studio", "minimax_h3_studio")]', h3)
        self.assertIn(
            'return [(interface, "SenseNova U1.5", "sensenova_u15_studio")]',
            sensenova,
        )
        self.assertIn("button.textContent = config.label", self.javascript)
        self.assertNotIn("aikimiOriginalLabel", self.javascript)

    def test_gradio6_and_legacy_tablist_shapes_are_supported(self):
        root_javascript = ROOT_JS.read_text(encoding="utf-8")
        aikimi_css = AIKIMI_CSS.read_text(encoding="utf-8")
        self.assertIn(
            ':scope > .tab-wrapper > .tab-container[role="tablist"]',
            root_javascript,
        )
        self.assertIn(":scope > .tab-nav", root_javascript)
        self.assertIn("function get_uiTabList(tabs)", root_javascript)
        self.assertIn("function get_uiTabButtons(tabs)", root_javascript)
        self.assertIn("function get_uiTopTabButton(panelId)", root_javascript)
        self.assertIn(
            ':scope > .tab-wrapper > .tab-container[role="tablist"]',
            self.javascript,
        )
        self.assertIn("function tabListFor(tabs)", self.javascript)
        self.assertIn(
            '#tabs > .tab-wrapper > .tab-container[role="tablist"]',
            aikimi_css,
        )
        self.assertIn("#tabs > .tab-nav", aikimi_css)
        self.assertIn("#aikimi-feature-nav + #tabs", aikimi_css)

    def test_runtime_consumers_do_not_depend_on_legacy_top_level_only_selectors(self):
        sources = {
            "script.js": ROOT_JS.read_text(encoding="utf-8"),
            "aikimi_tabs.js": self.javascript,
            "ui.js": UI_JS.read_text(encoding="utf-8"),
            "generationParams.js": GENERATION_PARAMS_JS.read_text(encoding="utf-8"),
            "extraNetworks.js": EXTRA_NETWORKS_JS.read_text(encoding="utf-8"),
            "active_units.js": CONTROLNET_UNITS_JS.read_text(encoding="utf-8"),
        }
        forbidden = (
            "#tabs > .tab-nav > button.selected",
            'querySelector("#tabs").querySelectorAll("button")',
            'querySelector("#tabs div button.selected")',
            "\"[id$='_extra_tabs'] > .tab-nav > button\"",
            'querySelector(".tab-nav").querySelectorAll("button")',
        )
        for name, source in sources.items():
            with self.subTest(name=name):
                for selector in forbidden:
                    self.assertNotIn(selector, source)

        self.assertIn('get_uiTopTabButton("tab_txt2img")', sources["ui.js"])
        self.assertIn('get_uiTopTabButton("tab_img2img")', sources["ui.js"])
        self.assertIn('get_uiTopTabButton("tab_extras")', sources["ui.js"])
        self.assertIn("get_uiCurrentTab()?.innerText", sources["generationParams.js"])
        self.assertIn("get_uiTabList(this_tab)", sources["extraNetworks.js"])
        self.assertIn("get_uiTabButtons(tab.parentNode)[index]", sources["active_units.js"])


if __name__ == "__main__":
    unittest.main()
