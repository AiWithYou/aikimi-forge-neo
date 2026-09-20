"""Exercise Qwen's result downloads through the actual Gradio frontend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.tests.chromium_helpers import find_chromium, reserve_local_port
from tools.tests.test_gradio_frontend_compat_chromium import _wait_expression, cdp_page
from tools.tests.test_qwen_image21_service import load_ui


class QwenImage21DownloadChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chromium is required for Qwen result download checks")
        cls.temporary = tempfile.TemporaryDirectory(prefix="qwen-ui-download-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.output = cls.directory / "output.png"
        Image.new("RGBA", (64, 64), (40, 100, 180, 100)).save(cls.output)
        (cls.directory / "result.json").write_text(json.dumps({"seed": 123}), encoding="utf-8")
        cls.ui = load_ui()
        cls.poll_counts = {}

        def delayed_status(identifier, owner):
            cls.poll_counts[identifier] = cls.poll_counts.get(identifier, 0) + 1
            done = cls.poll_counts[identifier] >= 5
            return {
                "done": done,
                "state": "complete" if done else "running",
                "message": f"完了 · {identifier}" if done else "生成中 · fixture",
                "elapsed": float(cls.poll_counts[identifier]),
            }

        cls.enterClassContext(patch.object(cls.ui.STUDIO, "start", side_effect=["fixture-job-1", "fixture-job-2"]))
        cls.enterClassContext(
            patch.object(
                cls.ui.STUDIO,
                "status",
                side_effect=delayed_status,
            )
        )
        cls.enterClassContext(patch.object(cls.ui.STUDIO, "artifact", return_value=cls.output))
        cls.demo = cls.ui.on_ui_tabs()[0][0]
        # Give only the isolated test instance a selector; production visibility
        # and callback behavior remain unchanged.
        component = next(
            component
            for component in cls.demo.blocks.values()
            if getattr(component, "label", None) == "PNG・生成条件を保存"
        )
        component.elem_id = "qwen21-downloads"
        cls.demo.config = cls.demo.get_config_file()
        cls.port = reserve_local_port()
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            prevent_thread_lock=True,
            quiet=True,
            inbrowser=False,
            ssr_mode=False,
            allowed_paths=[str(cls.directory)],
        )
        cls.addClassCleanup(cls.demo.close)
        cls.url = f"http://127.0.0.1:{cls.port}/"

    def test_completed_result_has_png_and_json_downloads_after_each_generation(self):
        with cdp_page(self.chromium, self.url) as page:
            self.assertTrue(page.evaluate(_wait_expression("#qwen21-generate", "true")))
            self.assertFalse(page.evaluate("document.body.innerText.includes('PNG・生成条件を保存')"))
            page.evaluate("""(() => {
                const input = document.querySelector('#qwen21-prompt textarea');
                input.value = 'Fixture image';
                input.dispatchEvent(new Event('input', {bubbles: true}));
            })()""")
            for generation in range(2):
                with self.subTest(generation=generation):
                    page.evaluate("document.querySelector('#qwen21-generate').click()")
                    self.assertTrue(
                        page.evaluate(
                            _wait_expression(
                                "body",
                                "!document.querySelector('#qwen21-downloads') || "
                                "document.querySelector('#qwen21-downloads').getClientRects().length === 0",
                            )
                        )
                    )
                    self.assertTrue(
                        page.evaluate(
                            _wait_expression(
                                "#qwen21-status textarea",
                                f"element.value.includes('完了 · fixture-job-{generation + 1}')",
                            )
                        )
                    )
                    self.assertTrue(
                        page.evaluate(
                            _wait_expression(
                                "#qwen21-downloads",
                                "element.innerText.includes('PNG・生成条件を保存') && "
                                "element.getBoundingClientRect().height > 0",
                                5000,
                            )
                        ),
                        page.evaluate("document.body.innerText"),
                    )
                    links = page.evaluate(
                        "Array.from(document.querySelectorAll('#qwen21-downloads a[href]'))"
                        ".map(a=>({text:a.innerText,href:a.href}))"
                    )
                    self.assertTrue(any("output.png" in item["href"] for item in links), links)
                    self.assertTrue(any("result.json" in item["href"] for item in links), links)
                    downloaded = page.evaluate("""Promise.all(
                        Array.from(document.querySelectorAll('#qwen21-downloads a[href]')).map(async a => {
                            const response = await fetch(a.href);
                            const bytes = new Uint8Array(await response.arrayBuffer());
                            return {url:a.href, status:response.status, size:bytes.length,
                                header:Array.from(bytes.slice(0,8)),
                                json:a.href.includes('result.json') ? JSON.parse(new TextDecoder().decode(bytes)) : null};
                        })
                    )""")
                    png = next(item for item in downloaded if "output.png" in item["url"])
                    metadata = next(item for item in downloaded if "result.json" in item["url"])
                    self.assertEqual(png["status"], 200)
                    self.assertEqual(png["header"], [137, 80, 78, 71, 13, 10, 26, 10])
                    self.assertEqual(metadata["status"], 200)
                    self.assertEqual(metadata["json"], {"seed": 123})


if __name__ == "__main__":
    unittest.main()
