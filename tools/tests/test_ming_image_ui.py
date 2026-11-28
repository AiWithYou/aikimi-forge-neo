import unittest
from pathlib import Path


class ContractTests(unittest.TestCase):
    def test_tab_registered(self):
        assert "ming_image_studio" in Path("extensions-builtin/aikimi-ui/javascript/aikimi_tabs.js").read_text(
            encoding="utf-8"
        )

    def load_module(self):
        import importlib.util
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        callbacks = SimpleNamespace(on_ui_tabs=lambda fn: None, on_app_started=lambda fn: None)
        path = Path("extensions-builtin/ming-image-studio/scripts/ming_image_studio.py")
        spec = importlib.util.spec_from_file_location("test_ming_controls", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
            spec.loader.exec_module(module)
        return module

    def build_screen(self):
        return self.load_module().on_ui_tabs()[0]

    def test_shared_job_status(self):
        from unittest.mock import patch

        ui = self.load_module()
        for state in ("running", "complete", "cancelled", "stopping", "error"):
            record = dict(status=state, stage="sampling", current=1, total=4, image_path="output.png")
            with self.subTest(state=state), patch.object(ui.SERVICE, "poll", return_value=record):
                _, html, _ = ui.poll("job")
                self.assertIn('data-aikimi-source="ming_image"', html)
                self.assertIn(f'data-stage="{state if state != "stopping" else "running"}"', html)
                if state == "running":
                    self.assertIn('data-progress="0.25"', html)
        script = Path("extensions-builtin/aikimi-ui/javascript/aikimiStatus.js").read_text(encoding="utf-8")
        self.assertIn('ming_image: "Ming Image"', script)

    def test_build_controls(self):
        blocks, label, tab_id = self.build_screen()
        self.addCleanup(blocks.close)
        cfg, = (item for item in blocks.blocks.values() if getattr(item, "label", None) == "CFG")
        self.assertEqual((cfg.minimum, cfg.maximum), (1, 20))
        self.assertEqual((label, tab_id), ("Ming Image", "ming_image_studio"))
        self.assertTrue(any(getattr(item, "elem_id", None) == "ming-image-modes" for item in blocks.blocks.values()))

    def test_mobile_template_and_edit_mode(self):
        from tools.tests.chromium_helpers import cdp_page, find_chromium, reserve_local_port

        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chrome or Chromium is not installed")
        blocks, _, _ = self.build_screen()
        self.addCleanup(blocks.close)
        css = "\n".join(
            Path(p).read_text(encoding="utf-8")
            for p in (
                "style.css",
                "extensions-builtin/aikimi-ui/style.css",
                "extensions-builtin/ming-image-studio/style.css",
            )
        )
        _, url, _ = blocks.launch(
            server_name="127.0.0.1", server_port=reserve_local_port(), prevent_thread_lock=True, quiet=True, css=css
        )
        with cdp_page(chromium, url) as page:
            page.send(
                "Emulation.setDeviceMetricsOverride", dict(width=390, height=900, deviceScaleFactor=1, mobile=False)
            )
            self.assertTrue(page.wait_for("#ming-image-prompt textarea", "element.getBoundingClientRect().width > 0"))
            page.evaluate(
                "[...document.querySelectorAll('#ming-image-settings button')].find(b=>b.textContent.trim()==='Figma 風 JSON').click()"
            )
            self.assertTrue(page.wait_for("#ming-image-prompt textarea", "element.value.includes('FRAME')"))
            page.evaluate("document.querySelector('#ming-image-img2img-button').click()")
            self.assertTrue(
                page.wait_for("#ming-image-references-gallery", "element.getBoundingClientRect().height > 0")
            )
            self.assertTrue(
                page.evaluate("document.querySelector('#ming-image-actions').getBoundingClientRect().right <= 390")
            )
