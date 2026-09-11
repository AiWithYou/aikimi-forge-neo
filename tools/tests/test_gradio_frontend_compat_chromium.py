from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from urllib.request import Request, urlopen

import gradio as gr
from websockets.sync.client import connect

from modules import gradio_frontend_compat
from tools.tests.chromium_helpers import close_chromium, find_chromium, reserve_local_port


def _wait_expression(selector: str, condition: str, timeout_ms: int = 15_000) -> str:
    return f"""
        new Promise((resolve) => {{
            const started = performance.now();
            const check = () => {{
                const element = document.querySelector({json.dumps(selector)});
                if (element && ({condition})) {{ resolve(true); return; }}
                if (performance.now() - started > {timeout_ms}) {{ resolve(false); return; }}
                setTimeout(check, 25);
            }};
            check();
        }})
    """


class CdpPage:
    def __init__(self, websocket):
        self.websocket = websocket
        self.command_id = 0

    def send(self, method: str, params: dict | None = None, *, timeout: float = 10):
        self.command_id += 1
        command_id = self.command_id
        self.websocket.send(json.dumps({"id": command_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CDP command timed out: {method}")
            message = json.loads(self.websocket.recv(timeout=remaining))
            if message.get("id") != command_id:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})

    def evaluate(self, expression: str, *, timeout: float = 15):
        result = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
            timeout=timeout,
        )
        return result["result"].get("value")


@contextmanager
def cdp_page(chromium: str, url: str):
    debugging_port = reserve_local_port()
    with tempfile.TemporaryDirectory(prefix="gradio-tabs-compat-chrome-") as profile:
        command = [
            chromium,
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=900,800",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={debugging_port}",
            "--remote-allow-origins=*",
            "about:blank",
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        websocket = None
        page = None
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    request = Request(  # noqa: S310 - fixed loopback debugging endpoint
                        f"http://127.0.0.1:{debugging_port}/json/new?about:blank",
                        method="PUT",
                    )
                    with urlopen(request, timeout=1) as response:  # noqa: S310
                        target = json.load(response)
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Chrome DevTools did not start") from error
                    time.sleep(0.05)

            websocket = connect(
                target["webSocketDebuggerUrl"],
                origin=f"http://127.0.0.1:{debugging_port}",
                open_timeout=5,
                close_timeout=2,
                max_size=None,
            )
            page = CdpPage(websocket)
            page.send("Page.enable")
            page.send("Runtime.enable")
            # Windows CIでは新しいChromeプロセスの初回読込が10秒を超える場合がある。
            page.send("Page.navigate", {"url": url}, timeout=30)
            yield page
        finally:
            close_chromium(
                process,
                websocket=websocket,
                close_browser=(lambda: page.send("Browser.close", timeout=2)) if page is not None else None,
                owned_port=debugging_port,
            )


class GradioFrontendCompatibilityChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")

        with gr.Blocks() as cls.demo:
            with gr.Tabs(elem_id="compat-many-tabs"):
                for index in range(49):
                    with gr.Tab(
                        f"Tab {index:02d}",
                        id=f"many-{index}",
                        elem_id=f"compat-many-panel-{index:02d}",
                    ):
                        gr.Markdown(f"Panel {index:02d}")

            with gr.Tabs(elem_id="compat-visible-tabs"):
                with gr.Tab("Always", id="always", elem_id="compat-visible-always"):
                    gr.Markdown("Always visible")
                with gr.Tab("Toggle", id="toggle", elem_id="compat-visible-toggle") as toggle_tab:
                    gr.Markdown("Dynamically visible")

            with gr.Tabs(elem_id="compat-keyboard-tabs"):
                for index in range(3):
                    with gr.Tab(
                        f"Key {index}",
                        id=f"key-{index}",
                        elem_id=f"compat-keyboard-panel-{index}",
                    ):
                        gr.Markdown(f"Keyboard panel {index}")

            hide_toggle = gr.Button("Hide Toggle", elem_id="compat-hide-toggle")
            show_toggle = gr.Button("Show Toggle", elem_id="compat-show-toggle")
            hide_toggle.click(lambda: gr.update(visible=False), outputs=toggle_tab, queue=False)
            show_toggle.click(lambda: gr.update(visible=True), outputs=toggle_tab, queue=False)

            render_count = gr.State(2)
            add_render_tab = gr.Button("Add Render Tab", elem_id="compat-add-render-tab")

            @gr.render(inputs=render_count)
            def render_tabs(count):
                with gr.Tabs(elem_id="compat-render-tabs"):
                    for index in range(count):
                        with gr.Tab(
                            f"Render {index}",
                            id=f"render-{index}",
                            elem_id=f"compat-render-panel-{index}",
                        ):
                            gr.Markdown(f"Rendered panel {index}")

            add_render_tab.click(lambda count: min(count + 1, 4), inputs=render_count, outputs=render_count)

        cls.port = reserve_local_port()
        app, _, _ = cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            prevent_thread_lock=True,
            quiet=True,
            css=(
                '.tab-wrapper > .tab-container[role="tablist"] {'
                "overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; }"
                '.tab-wrapper > .tab-container[role="tablist"] > button { flex: 0 0 auto; }'
            ),
        )
        cls.addClassCleanup(cls.demo.close)
        gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)
        cls.url = f"http://127.0.0.1:{cls.port}/"

        with gr.Blocks() as cls.unpatched_demo:
            with gr.Tabs(elem_id="unpatched-visible-tabs"):
                with gr.Tab("Always", id="always", elem_id="unpatched-visible-always"):
                    gr.Markdown("Always visible")
                with gr.Tab("Toggle", id="toggle", elem_id="unpatched-visible-toggle") as unpatched_toggle:
                    gr.Markdown("Dynamically visible")
            unpatched_hide = gr.Button("Hide Toggle", elem_id="unpatched-hide-toggle")
            unpatched_hide.click(
                lambda: gr.update(visible=False),
                outputs=unpatched_toggle,
                queue=False,
            )

        cls.unpatched_port = reserve_local_port()
        cls.unpatched_demo.launch(
            server_name="127.0.0.1",
            server_port=cls.unpatched_port,
            prevent_thread_lock=True,
            quiet=True,
        )
        cls.addClassCleanup(cls.unpatched_demo.close)
        cls.unpatched_url = f"http://127.0.0.1:{cls.unpatched_port}/"

    def test_49_tabs_mount_switch_keyboard_and_reload_stay_responsive(self):
        with cdp_page(self.chromium, self.url) as page:
            ready = page.evaluate(
                _wait_expression(
                    '#compat-many-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                    "element.querySelectorAll(':scope > button').length === 49",
                ),
                timeout=20,
            )
            self.assertTrue(ready)

            initial = page.evaluate(
                """
                (() => {
                    const nav = document.querySelector(
                        '#compat-many-tabs > .tab-wrapper > .tab-container[role="tablist"]'
                    );
                    const buttons = Array.from(nav.children).filter((node) => node.tagName === 'BUTTON');
                    buttons[48].click();
                    return {
                        count: buttons.length,
                        first: buttons[0].textContent.trim(),
                        last: buttons[48].textContent.trim(),
                        horizontalFallback: getComputedStyle(nav).overflowX === 'auto',
                        scrollable: nav.scrollWidth >= nav.clientWidth,
                    };
                })()
                """
            )
            self.assertEqual(initial["count"], 49)
            self.assertEqual(initial["first"], "Tab 00")
            self.assertEqual(initial["last"], "Tab 48")
            self.assertTrue(initial["horizontalFallback"])
            self.assertTrue(initial["scrollable"])
            self.assertTrue(
                page.evaluate(_wait_expression("#compat-many-panel-48", "element.innerText.includes('Panel 48')"))
            )

            page.evaluate(
                """
                (() => {
                    const nav = document.querySelector(
                        '#compat-keyboard-tabs > .tab-wrapper > .tab-container[role="tablist"]'
                    );
                    nav.querySelectorAll(':scope > button')[0].focus();
                })()
                """
            )
            page.send(
                "Input.dispatchKeyEvent",
                {
                    "type": "keyDown",
                    "key": "Tab",
                    "code": "Tab",
                    "windowsVirtualKeyCode": 9,
                    "nativeVirtualKeyCode": 9,
                },
            )
            page.send(
                "Input.dispatchKeyEvent",
                {
                    "type": "keyUp",
                    "key": "Tab",
                    "code": "Tab",
                    "windowsVirtualKeyCode": 9,
                    "nativeVirtualKeyCode": 9,
                },
            )
            focused = page.evaluate(
                "({ role: document.activeElement?.getAttribute('role'), text: document.activeElement?.textContent.trim() })"
            )
            self.assertEqual(focused, {"role": "tab", "text": "Key 1"})
            page.evaluate("document.activeElement.click()")
            self.assertTrue(page.evaluate(_wait_expression("#compat-keyboard-panel-1", "true")))

            time.sleep(2)
            self.assertEqual(page.evaluate("document.readyState"), "complete")

            page.send("Page.reload", {"ignoreCache": True})
            reloaded = page.evaluate(
                _wait_expression(
                    '#compat-many-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                    "element.querySelectorAll(':scope > button').length === 49",
                ),
                timeout=20,
            )
            self.assertTrue(reloaded)

    @unittest.expectedFailure
    def test_backend_tab_visibility_update_is_an_upstream_gradio617_limit(self):
        with cdp_page(self.chromium, self.url) as page:
            self.assertTrue(
                page.evaluate(
                    _wait_expression(
                        '#compat-visible-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                        "element.querySelectorAll(':scope > button').length === 2",
                    ),
                    timeout=20,
                )
            )
            page.evaluate("document.querySelector('#compat-hide-toggle').click()")
            hidden = page.evaluate(
                _wait_expression(
                    '#compat-visible-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                    "Array.from(element.children).filter((node) => node.tagName === 'BUTTON').length === 1",
                    timeout_ms=5_000,
                ),
                timeout=10,
            )

        self.assertTrue(hidden, "backend Tab.visible updates must remain observable")

    def test_unpatched_gradio617_has_the_same_backend_visibility_limit(self):
        with cdp_page(self.chromium, self.unpatched_url) as page:
            self.assertTrue(
                page.evaluate(
                    _wait_expression(
                        '#unpatched-visible-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                        "element.querySelectorAll(':scope > button').length === 2",
                    ),
                    timeout=20,
                )
            )
            page.evaluate("document.querySelector('#unpatched-hide-toggle').click()")
            time.sleep(2)
            button_count = page.evaluate(
                """
                document.querySelectorAll(
                    '#unpatched-visible-tabs > .tab-wrapper > .tab-container[role="tablist"] > button'
                ).length
                """
            )

        self.assertEqual(button_count, 2)

    def test_gr_render_tab_addition_remains_observable(self):
        with cdp_page(self.chromium, self.url) as page:
            self.assertTrue(
                page.evaluate(
                    _wait_expression(
                        '#compat-render-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                        "element.querySelectorAll(':scope > button').length === 2",
                    ),
                    timeout=20,
                )
            )
            page.evaluate("document.querySelector('#compat-add-render-tab').click()")
            rendered = page.evaluate(
                _wait_expression(
                    '#compat-render-tabs > .tab-wrapper > .tab-container[role="tablist"]',
                    "element.querySelectorAll(':scope > button').length === 3",
                    timeout_ms=8_000,
                ),
                timeout=12,
            )

        self.assertTrue(rendered, "gr.render tab additions must remain observable")


if __name__ == "__main__":
    unittest.main()
