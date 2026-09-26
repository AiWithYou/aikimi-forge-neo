"""Exercise Qwen's result downloads through the actual Gradio frontend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np
import psutil
from PIL import Image

from modules import ui_tempdir
from modules.gradio_frontend_compat import build_patched_tabs_asset, create_gradio_compatibility_app
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
        # Use Forge's real cache redirect, isolated from the user's temp files.
        cls.enterClassContext(
            patch.object(
                ui_tempdir,
                "_shared_module",
                return_value=SimpleNamespace(
                    opts=SimpleNamespace(temp_dir=str(cls.directory)),
                    demo=None,
                ),
            )
        )
        cls.enterClassContext(patch("gradio.processing_utils.save_pil_to_cache", ui_tempdir.save_pil_to_file))
        cls.output = cls.directory / "output.png"
        pixels = np.random.default_rng(7).integers(0, 256, (1280, 1024, 3), dtype=np.uint8)
        fixture = Image.fromarray(pixels)
        fixture.putalpha(100)
        fixture.save(cls.output)
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
                "effective_prompt": "A detailed glass bird on a wooden table.",
            }

        cls.submitted = []
        cls.rewrite_settings = []

        def submit(request, owner):
            cls.rewrite_settings.append(request.rewrite_prompt)
            bounds = None
            if request.annotation_layers:
                with Image.open(request.annotation_layers[0]) as layer:
                    bounds = layer.getchannel("A").getbbox()
            cls.submitted.append((request.annotation_reference, bounds))
            return f"fixture-job-{len(cls.submitted)}"

        cls.enterClassContext(patch.object(cls.ui.STUDIO, "start", side_effect=submit))
        cls.enterClassContext(
            patch.object(
                cls.ui.STUDIO,
                "status",
                side_effect=delayed_status,
            )
        )
        cls.enterClassContext(patch.object(cls.ui.STUDIO, "artifact", return_value=cls.output))
        studio = cls.ui.on_ui_tabs()[0][0]
        # Match Forge: build the extension separately, then render it inside
        # a tab that is not selected at page load.
        with gr.Blocks() as cls.demo:
            with gr.Tabs():
                with gr.Tab("Home"):
                    gr.Markdown("Other workspace")
                    # Forge's initial tab has form controls. Gradio 6.17 loads
                    # their frontend module from that initial visible tab;
                    # a Markdown-only stub omits it from nested accordions.
                    gr.Textbox(label="Main workspace prompt", elem_id="main-workspace-prompt")
                with gr.Tab("Qwen editing"):
                    studio.render()
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
            allowed_paths=[str(cls.directory), str(Path(__file__).resolve().parents[2] / "modules_forge/forge_canvas")],
            _app=create_gradio_compatibility_app(build_patched_tabs_asset()),
        )
        cls.addClassCleanup(cls.demo.close)
        cls.url = f"http://127.0.0.1:{cls.port}/"

    def test_fun_controlnet_inputs_render_in_qwen_tab(self):
        with cdp_page(self.chromium, self.url) as page:
            self.assertTrue(page.evaluate(_wait_expression('[role="tab"]', "true", 25000), timeout=30))
            self.assertTrue(page.evaluate(_wait_expression("#main-workspace-prompt textarea", "true")))
            page.evaluate(
                "Array.from(document.querySelectorAll('[role=tab]')).find(e=>e.innerText==='Qwen editing').click()"
            )
            self.assertTrue(page.evaluate(_wait_expression("#qwen21-generate", "true")))
            # Scroll into view before opening, matching the user's interaction.
            page.evaluate("document.querySelector('#qwen21-control-section').scrollIntoView({block: 'center'})")
            page.evaluate("document.querySelector('#qwen21-control-section .label-wrap').click()")
            self.assertTrue(
                page.evaluate(_wait_expression("#qwen21-control-kind", "true", 8000), timeout=10), page.exceptions
            )
            page.evaluate("document.querySelector('#qwen21-control-kind input[value=canny]').click()")
            self.assertTrue(
                page.evaluate(
                    _wait_expression("#qwen21-control-image", "element.getBoundingClientRect().height > 0", 8000),
                    timeout=10,
                ),
                page.exceptions,
            )
            visible = page.evaluate("""(() => {
                const ids=['qwen21-control-kind','qwen21-control-image'];
                return ids.every(id => {
                    const e=document.getElementById(id);
                    return e && e.getBoundingClientRect().height > 0;
                });
            })()""")
            self.assertTrue(visible, page.exceptions)
            self.assertIn("前処理済みの制御画像", page.evaluate("document.body.innerText"))
            self.assertFalse(page.exceptions)

    def test_fun_acc_switch_sets_int8_and_four_steps(self):
        with cdp_page(self.chromium, self.url) as page:
            self.assertTrue(page.evaluate(_wait_expression('[role="tab"]', "true", 45000), timeout=50))
            self.assertTrue(page.evaluate(_wait_expression("#main-workspace-prompt textarea", "true")))
            page.evaluate(
                "Array.from(document.querySelectorAll('[role=tab]')).find(e=>e.innerText==='Qwen editing').click()"
            )
            self.assertTrue(page.evaluate(_wait_expression("#qwen21-fun-acc input", "true", 15000)))
            page.evaluate("document.querySelector('#qwen21-fun-acc input').click()")
            self.assertTrue(
                page.evaluate(
                    _wait_expression(
                        "#qwen21-precision",
                        "element.querySelector('input[role=listbox]')?.value === '通常 · INT8 · 拡張対応' && element.querySelector('input[role=listbox]')?.disabled",
                        10000,
                    )
                ),
                page.exceptions,
            )
            self.assertTrue(
                page.evaluate(
                    _wait_expression(
                        "#qwen21-steps",
                        "Array.from(element.querySelectorAll('input')).some(input => input.value === '4')",
                        10000,
                    )
                ),
                str(page.exceptions) + str(page.evaluate("document.querySelector('#qwen21-steps')?.outerHTML")),
            )
            self.assertFalse(page.exceptions)

    def test_completed_result_has_png_and_json_downloads_after_each_generation(self):
        with cdp_page(self.chromium, self.url) as page:
            peak_rss = 0
            peak_private = 0
            # Leave transport headroom after the DOM deadline, including cold
            # browser startup while the full offline suite is running.
            self.assertTrue(page.evaluate(_wait_expression('[role="tab"]', "true", 25000), timeout=30))
            self.assertTrue(page.evaluate(_wait_expression("#main-workspace-prompt textarea", "true")))
            page.evaluate(
                "Array.from(document.querySelectorAll('[role=tab]')).find(e=>e.innerText==='Qwen editing').click()"
            )
            self.assertTrue(page.evaluate(_wait_expression("#qwen21-generate", "true")))
            self.assertTrue(
                page.evaluate(_wait_expression("body", "typeof ForgeCanvas === 'function'")), page.exceptions
            )
            self.assertFalse(page.evaluate("document.body.innerText.includes('PNG・生成条件を保存')"))
            page.evaluate("""(() => {
                const input = document.querySelector('#qwen21-prompt textarea');
                input.value = 'Fixture image';
                input.dispatchEvent(new Event('input', {bubbles: true}));
            })()""")
            for generation in range(2):
                with self.subTest(generation=generation):
                    if generation == 1:
                        page.evaluate("document.querySelector('#qwen21-rewrite-prompt input').click()")
                    previous_src = page.evaluate("document.querySelector('#qwen21-output img')?.src || null")
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
                    if generation:
                        self.assertTrue(
                            page.evaluate(
                                _wait_expression("#qwen21-output", "element.innerText.includes('前の結果')", 3000)
                            )
                        )
                        self.assertTrue(previous_src)
                        self.assertEqual(
                            page.evaluate("document.querySelector('#qwen21-output img')?.src"), previous_src
                        )
                    self.assertTrue(
                        page.evaluate(
                            _wait_expression(
                                "#qwen21-status textarea",
                                f"element.value.includes('完了 · fixture-job-{generation + 1}')",
                            )
                        )
                    )
                    self.assertEqual(self.rewrite_settings[-1], generation == 1)
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
                    page.evaluate("""(() => {
                        const button = Array.from(document.querySelectorAll('.label-wrap'))
                            .find(b => b.innerText.includes('使用したプロンプト'));
                        button.scrollIntoView({block: 'center'});
                        if (!button.classList.contains('open')) button.click();
                    })()""")
                    self.assertTrue(
                        page.evaluate(
                            _wait_expression(
                                "#qwen21-effective-prompt textarea",
                                "element.value === 'A detailed glass bird on a wooden table.'",
                                5000,
                            )
                        ),
                        str(page.exceptions) + str(page.evaluate("document.body.innerText")),
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
                    page.evaluate("document.querySelector('#qwen21-edit-result').click()")
                    self.assertTrue(
                        page.evaluate(
                            _wait_expression(
                                "#qwen21-annotation-panel",
                                "element.getBoundingClientRect().height > 0 && "
                                "element.innerText.includes('Image 1') && "
                                "document.querySelector('#qwen21-references img') !== null",
                            )
                        ),
                        page.evaluate("document.body.innerText"),
                    )
                    # Wait for the Gallery change event after it moves the
                    # artifact into Gradio's cache. The panel must stay open.
                    self.assertTrue(
                        page.evaluate("""new Promise(resolve => setTimeout(() => resolve(
                            document.querySelector('#qwen21-annotation-panel').getBoundingClientRect().height > 0
                        ), 1200))""")
                    )
                    canvas_ready = page.evaluate(
                        _wait_expression(
                            "#qwen21-annotation-editor .forge-image",
                            "element.complete && element.naturalWidth === 1024 && element.getBoundingClientRect().width > 0",
                        )
                    )
                    self.assertTrue(
                        canvas_ready,
                        {
                            "errors": page.exceptions,
                            "canvas": page.evaluate(
                                "({config:typeof gradio_config,truth:typeof True,background:Array.from(document.querySelectorAll('.logical_image_background textarea')).map(e=>e.value.length),image:document.querySelector('#qwen21-annotation-editor .forge-image')?.getAttribute('src')?.slice(0,40)})"
                            ),
                        },
                    )
                    self.assertGreater(
                        page.evaluate("document.querySelectorAll('#qwen21-references input[type=file]').length"),
                        0,
                        "Continuing an edit hid the reference upload controls",
                    )
                    if generation == 0:
                        page.evaluate(
                            "document.querySelector('#qwen21-references button[aria-label^=Thumbnail]').click()"
                        )
                        self.assertTrue(
                            page.evaluate("""new Promise(resolve => setTimeout(() => resolve(
                            !!document.querySelector('#qwen21-references input[type=file]')
                        ), 700))"""),
                            "Selecting a reference hid its upload controls",
                        )
                        page.evaluate(
                            "Array.from(document.querySelectorAll('#qwen21-reference-controls button')).find(b=>b.innerText==='後へ').click()"
                        )
                        self.assertTrue(
                            page.evaluate(
                                _wait_expression("#qwen21-references", "!!element.querySelector('input[type=file]')")
                            ),
                            "Reordering a selected reference did not restore upload controls",
                        )
                    owned = psutil.Process(page.process.pid)
                    rss = private = 0
                    for process in [owned, *owned.children(recursive=True)]:
                        try:
                            memory = process.memory_full_info()
                        except psutil.NoSuchProcess:
                            continue
                        rss += memory.rss
                        private += memory.uss
                    peak_rss = max(peak_rss, rss)
                    peak_private = max(peak_private, private)
                    # Summed RSS double-counts shared Chromium pages differently
                    # on Linux and Windows. Gate on unique resident pages (USS)
                    # and retain RSS as the locally comparable benchmark value.
                    self.assertLess(private, 1536 * 2**20, "Drawing UI consumed excessive private resident memory")
                    if generation:
                        # Opening a new result must never resurrect the guide
                        # drawn on the previous source, even through Undo.
                        stale_marks = page.evaluate("""(() => {
                            const root=document.querySelector('#qwen21-annotation-editor');
                            const canvas=root.querySelector('.forge-drawing-canvas');
                            const undo=root.querySelector('[id^=undoButton_]');
                            const states=[];
                            for(let n=0;n<20 && !undo.disabled;n++) {
                                undo.click();
                                const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
                                states.push(pixels.some((v,i)=>i%4===3 && v>0));
                            }
                            return states;
                        })()""")
                        self.assertFalse(any(stale_marks), "Undo restored markings from the previous editing source")
                    if generation == 0:
                        page.evaluate(
                            "document.querySelector('#qwen21-annotation-editor .forge-drawing-canvas').scrollIntoView({block: 'center'})"
                        )
                        rectangle = page.evaluate(
                            "document.querySelector('#qwen21-annotation-editor .forge-drawing-canvas').getBoundingClientRect().toJSON()"
                        )
                        x, y = rectangle["x"] + rectangle["width"] * 0.45, rectangle["y"] + rectangle["height"] * 0.25
                        page.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
                        page.send(
                            "Input.dispatchMouseEvent",
                            {"type": "mousePressed", "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1},
                        )
                        page.send(
                            "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x + 45, "y": y + 35, "buttons": 1}
                        )
                        page.send(
                            "Input.dispatchMouseEvent",
                            {
                                "type": "mouseReleased",
                                "x": x + 45,
                                "y": y + 35,
                                "button": "left",
                                "buttons": 0,
                                "clickCount": 1,
                            },
                        )
            self.assertEqual(self.submitted[1][0], 0)
            self.assertIsNotNone(self.submitted[1][1])
            print(f"Isolated drawing browser peak RSS: {peak_rss / 2**20:.1f} MiB")  # noqa: T201 - regression measurement
            print(f"Isolated drawing browser peak USS: {peak_private / 2**20:.1f} MiB")  # noqa: T201


if __name__ == "__main__":
    unittest.main()
