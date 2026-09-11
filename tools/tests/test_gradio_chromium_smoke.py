import html
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import gradio as gr
from websockets.sync.client import connect

from modules import gradio_compat
from modules.ui_components import InputAccordion
from tools.tests.chromium_helpers import close_chromium, find_chromium, reserve_local_port

ROOT = Path(__file__).resolve().parents[2]


def inline_javascript(path):
    source = path.read_text(encoding="utf-8").replace("</script", r"<\/script")
    return f"<script>\n{source}\n</script>"


def probe_navigation_with_cdp(chromium, url, *, blocked_urls=(), timeout=60):
    """Wait for the real top navigation without relying on virtual-time timers."""

    debugging_port = reserve_local_port()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="forge-full-ui-cdp-") as profile:
        command = [
            chromium,
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
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
        close_browser = None
        try:
            deadline = started + timeout
            while True:
                try:
                    request = Request(  # noqa: S310
                        f"http://127.0.0.1:{debugging_port}/json/new?about:blank",
                        method="PUT",
                    )
                    with urlopen(request, timeout=1) as response:  # noqa: S310
                        target = json.load(response)
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Chrome DevTools did not start in time") from error
                    time.sleep(0.05)

            websocket = connect(
                target["webSocketDebuggerUrl"],
                origin=f"http://127.0.0.1:{debugging_port}",
                open_timeout=5,
                close_timeout=2,
                max_size=None,
            )
            command_id = 0

            def send(method, params=None, *, response_timeout=5):
                nonlocal command_id
                command_id += 1
                current_id = command_id
                websocket.send(
                    json.dumps(
                        {
                            "id": current_id,
                            "method": method,
                            "params": params or {},
                        }
                    )
                )
                response_deadline = time.monotonic() + response_timeout
                while True:
                    remaining = response_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"CDP command timed out: {method}")
                    message = json.loads(websocket.recv(timeout=remaining))
                    if message.get("id") == current_id:
                        if "error" in message:
                            raise RuntimeError(message["error"])
                        return message.get("result", {})

            def request_browser_close():
                return send("Browser.close", response_timeout=2)

            close_browser = request_browser_close

            send("Network.enable")
            if blocked_urls:
                send("Network.setBlockedURLs", {"urls": list(blocked_urls)})
            send(
                "Page.navigate",
                {"url": url},
                response_timeout=max(5, deadline - time.monotonic()),
            )

            expression = """
                new Promise((resolve) => {
                    const started = performance.now();
                    const check = () => {
                        const tabs = document.querySelector("#tabs");
                        const tabList = tabs?.querySelector(
                            ":scope > .tab-wrapper > .tab-container[role='tablist'], " +
                            ":scope > .tab-nav[role='tablist'], :scope > .tab-nav"
                        );
                        const buttons = Array.from(tabList?.children || [])
                            .filter((node) => node.tagName === "BUTTON");
                        if (buttons.length) {
                            resolve({
                                ready: true,
                                navCount: buttons.length,
                                labels: buttons.map((button) => button.textContent.trim()),
                                title: document.title,
                            });
                            return;
                        }
                        if (performance.now() - started >= 55000) {
                            resolve({
                                ready: false,
                                navCount: 0,
                                labels: [],
                                title: document.title,
                            });
                            return;
                        }
                        setTimeout(check, 50);
                    };
                    check();
                })
            """
            while True:
                wait_seconds = max(1, deadline - time.monotonic())
                try:
                    evaluation = send(
                        "Runtime.evaluate",
                        {
                            "expression": expression,
                            "awaitPromise": True,
                            "returnByValue": True,
                        },
                        response_timeout=wait_seconds,
                    )
                    break
                except RuntimeError as error:
                    navigation_race = any(
                        message in str(error)
                        for message in (
                            "Inspected target navigated or closed",
                            "Execution context was destroyed",
                        )
                    )
                    if not navigation_race or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            state = evaluation["result"]["value"]
            state["elapsed_seconds"] = time.monotonic() - started
            return state
        finally:
            close_chromium(
                process,
                websocket=websocket,
                close_browser=close_browser,
                owned_port=debugging_port,
            )


VISIBILITY_PROBE = """
<script>
(() => {
  let attempts = 0;
  let phase = 0;
  const probe = () => {
    const show = document.querySelector(
      "#visibility-show button, button#visibility-show"
    );
    const hide = document.querySelector(
      "#visibility-hide button, button#visibility-hide"
    );
    const sliders = [
      document.querySelector("#visibility-slider-a"),
      document.querySelector("#visibility-slider-b"),
    ];
    const visible = sliders.every(
      (node) => node?.getBoundingClientRect().width > 0
    );
    if (phase === 0 && show && hide) {
      phase = 1;
      show.click();
    }
    if (phase === 1 && visible) {
      phase = 2;
      hide.click();
    } else if (phase === 2 && !visible) {
      phase = 3;
      show.click();
    } else if (phase === 3 && visible) {
      document.documentElement.dataset.gradioVisibilitySmoke = "pass";
      return;
    }
    attempts += 1;
    if (attempts >= 160) {
      document.documentElement.dataset.gradioVisibilitySmoke = "fail";
      return;
    }
    setTimeout(probe, 50);
  };
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", probe, { once: true });
  } else {
    probe();
  }
})();
</script>
"""

TAB_SELECTOR_PROBE = """
<script>
(() => {
  let attempts = 0;
  const probe = () => {
    const tabs = document.querySelector("#tabs");
    const tabList = window.get_uiTabList?.(tabs);
    const txt2img = window.get_uiTopTabButton?.("tab_txt2img");
    const img2img = window.get_uiTopTabButton?.("tab_img2img");
    const extras = window.get_uiTopTabButton?.("tab_extras");
    if (tabList && txt2img && img2img && extras) {
      window.switch_to_extras();
      setTimeout(() => {
        const extrasSelected = extras.getAttribute("aria-selected") === "true";
        window.switch_to_img2img();
        setTimeout(() => {
          const mirrorButton = tabs.querySelector(
            ":scope > .tab-wrapper > .tab-container.visually-hidden button"
          );
          const result = document.querySelector("#tab-selector-result");
          result.dataset.ready = "true";
          result.textContent = JSON.stringify({
            tabListClass: tabList.className,
            tabListRole: tabList.getAttribute("role"),
            mirrorExists: Boolean(mirrorButton),
            mirrorExcluded: !window.get_uiTabButtons(tabs).includes(mirrorButton),
            txt2imgPanel: txt2img.getAttribute("aria-controls"),
            img2imgPanel: img2img.getAttribute("aria-controls"),
            extrasPanel: extras.getAttribute("aria-controls"),
            extrasSelected,
            img2imgSelected: img2img.getAttribute("aria-selected") === "true",
            nestedSelectedIndex: window.get_tab_index("mode_img2img"),
          });
        }, 100);
      }, 100);
      return;
    }
    attempts += 1;
    if (attempts >= 200) {
      document.querySelector("#tab-selector-result").dataset.ready = "fail";
      return;
    }
    setTimeout(probe, 50);
  };
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", probe, { once: true });
  } else {
    probe();
  }
})();
</script>
"""

TAB_SELECTOR_RUNTIME = (
    inline_javascript(ROOT / "script.js") + inline_javascript(ROOT / "javascript" / "ui.js") + TAB_SELECTOR_PROBE
)

INPUT_ACCORDION_PROBE = """
<script>
(() => {
  let attempts = 0;
  const probe = () => {
    const accordion = document.querySelector("#chromium-input-accordion");
    const hidden = document.querySelector("#chromium-input-accordion-checkbox input");
    const result = document.querySelector("#input-accordion-result");
    if (accordion && hidden && typeof setupAccordion === "function") {
      const first = setupAccordion(accordion);
      const second = setupAccordion(accordion);
      const toggled = inputAccordionChecked("chromium-input-accordion", true);
      setTimeout(() => {
        result.dataset.ready = "true";
        result.textContent = JSON.stringify({
          first,
          second,
          toggled,
          visibleCheckboxes: accordion.querySelectorAll(
            "#chromium-input-accordion-visible-checkbox"
          ).length,
          hiddenChecked: hidden.checked,
          visibleChecked: accordion.visibleCheckbox?.checked,
          open: accordion.querySelector(".label-wrap")?.classList.contains("open"),
        });
      }, 100);
      return;
    }
    attempts += 1;
    if (attempts >= 200) {
      result.dataset.ready = "fail";
      return;
    }
    setTimeout(probe, 50);
  };
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", probe, { once: true });
  } else {
    probe();
  }
})();
</script>
"""

INPUT_ACCORDION_RUNTIME = inline_javascript(ROOT / "javascript" / "inputAccordion.js") + INPUT_ACCORDION_PROBE

CONTROLNET_ACTIVE_UNITS_PROBE = """
<script>
(() => {
  window.controlnetProbeErrors = [];
  window.controlnetListenerCounts = {};
  window.localGet = () => null;
  window.addEventListener("error", (event) => {
    window.controlnetProbeErrors.push(event.message);
  });
  const nativeAddEventListener = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, listener, options) {
    if (this.id?.startsWith("fixture-controlnet-")) {
      const key = `${this.id}:${type}`;
      window.controlnetListenerCounts[key] =
        (window.controlnetListenerCounts[key] || 0) + 1;
    }
    return nativeAddEventListener.call(this, type, listener, options);
  };

  const unitMarkup = (tabName) => `
    <button class="label-wrap"><span>ControlNet</span></button>
    <div class="tabs">
      <div class="tab-wrapper">
        <div class="tab-container" role="tablist">
          <button type="button" role="tab">Unit 0</button>
        </div>
      </div>
      <div class="tabitem">
        <label class="cnet-unit-enabled">
          <input id="fixture-controlnet-${tabName}-enabled" type="checkbox">
        </label>
        <div class="cnet-input-image-group">
          <div class="cnet-image">
            <input id="fixture-controlnet-${tabName}-upload" type="file">
          </div>
        </div>
        <label class="controlnet_control_type_filter_group">
          <input id="fixture-controlnet-${tabName}-radio" type="radio" value="All" checked>
        </label>
      </div>
    </div>`;

  const waitFor = async (predicate, attempts = 160) => {
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const value = predicate();
      if (value) return value;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    return null;
  };

  const run = async () => {
    const host = await waitFor(() => document.querySelector("#controlnet-fixture-host"));
    const result = document.querySelector("#controlnet-active-units-result");
    if (!host || !result) return;

    executeCallbacks(uiLoadedCallbacks);
    await new Promise((resolve) => setTimeout(resolve, 50));

    host.insertAdjacentHTML(
      "beforeend",
      '<div id="txt2img_controlnet"><div id="controlnet">' +
        '<button class="label-wrap"><span>ControlNet</span></button>' +
      '</div></div>'
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    const partialBadgeCount = document.querySelectorAll(
      "#txt2img_controlnet .cnet-badge"
    ).length;

    document.querySelector("#txt2img_controlnet #controlnet")
      .insertAdjacentHTML("beforeend", unitMarkup("txt2img").replace(
        '<button class="label-wrap"><span>ControlNet</span></button>', ""
      ));
    const txtBadge = await waitFor(() =>
      document.querySelector("#txt2img_controlnet .cnet-badge")
    );

    host.insertAdjacentHTML(
      "beforeend",
      '<div id="img2img_controlnet"><div id="controlnet">' +
        unitMarkup("img2img") +
      '</div></div>'
    );
    const imgBadge = await waitFor(() =>
      document.querySelector("#img2img_controlnet .cnet-badge")
    );

    executeCallbacks(uiLoadedCallbacks);
    executeCallbacks(uiTabChangeCallbacks);
    for (const panel of document.querySelectorAll(
      "#txt2img_controlnet #controlnet, #img2img_controlnet #controlnet"
    )) {
      panel.appendChild(document.createElement("span"));
    }
    await new Promise((resolve) => setTimeout(resolve, 100));

    const txtEnabled = document.querySelector(
      "#txt2img_controlnet .cnet-unit-enabled input"
    );
    const imgEnabled = document.querySelector(
      "#img2img_controlnet .cnet-unit-enabled input"
    );
    for (const checkbox of [txtEnabled, imgEnabled]) {
      checkbox.checked = true;
      checkbox.dispatchEvent(new Event("change", { bubbles: true }));
    }

    result.dataset.ready = "true";
    result.textContent = JSON.stringify({
      partialBadgeCount,
      txtBadgeCount: document.querySelectorAll("#txt2img_controlnet .cnet-badge").length,
      imgBadgeCount: document.querySelectorAll("#img2img_controlnet .cnet-badge").length,
      txtBadgeText: document.querySelector(
        "#txt2img_controlnet .cnet-badge"
      )?.textContent || "",
      imgBadgeText: document.querySelector(
        "#img2img_controlnet .cnet-badge"
      )?.textContent || "",
      txtActive: document.querySelector(
        "#txt2img_controlnet [role='tablist'] > button"
      )?.classList.contains("cnet-unit-active"),
      imgActive: document.querySelector(
        "#img2img_controlnet [role='tablist'] > button"
      )?.classList.contains("cnet-unit-active"),
      listenerCounts: window.controlnetListenerCounts,
      errors: window.controlnetProbeErrors,
    });
  };

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => void run(), { once: true });
  } else {
    void run();
  }
})();
</script>
"""

CONTROLNET_ACTIVE_UNITS_RUNTIME = (
    inline_javascript(ROOT / "extensions-builtin" / "sd_forge_controlnet" / "javascript" / "active_units.js")
    + CONTROLNET_ACTIVE_UNITS_PROBE
)


class GradioChromiumVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")

        hidden = gradio_compat.keep_hidden_component_mounted(False)
        with gr.Blocks() as cls.demo:
            show = gr.Button("Show controls", elem_id="visibility-show")
            hide = gr.Button("Hide controls", elem_id="visibility-hide")
            first = gr.Slider(visible=hidden, elem_id="visibility-slider-a")
            second = gr.Slider(visible=hidden, elem_id="visibility-slider-b")
            with InputAccordion(False, label="Input accordion", elem_id="chromium-input-accordion"):
                gr.Markdown("Input accordion content")
            with gr.Tabs(elem_id="tabs"):
                with gr.Tab("txt2img", elem_id="tab_txt2img"):
                    gr.Markdown("txt2img content")
                with gr.Tab("img2img", elem_id="tab_img2img"):
                    with gr.Tabs(elem_id="mode_img2img"):
                        with gr.Tab("img2img", elem_id="mode_img2img_base"):
                            gr.Markdown("img2img content")
                        with gr.Tab("Sketch", elem_id="mode_img2img_sketch"):
                            gr.Markdown("sketch content")
                with gr.Tab("Extras", elem_id="tab_extras"):
                    gr.Markdown("extras content")
            gr.HTML('<pre id="tab-selector-result">pending</pre>')
            gr.HTML('<pre id="input-accordion-result">pending</pre>')
            gr.HTML('<div id="controlnet-fixture-host"></div><pre id="controlnet-active-units-result">pending</pre>')
            show.click(
                lambda: (gr.update(visible=True), gr.update(visible=True)),
                outputs=[first, second],
                queue=False,
            )
            hide.click(
                lambda: (
                    gr.update(visible=hidden),
                    gr.update(visible=hidden),
                ),
                outputs=[first, second],
                queue=False,
            )

        cls.port = reserve_local_port()
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            prevent_thread_lock=True,
            quiet=True,
            head=(VISIBILITY_PROBE + TAB_SELECTOR_RUNTIME + INPUT_ACCORDION_RUNTIME + CONTROLNET_ACTIVE_UNITS_RUNTIME),
        )

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_multiple_hidden_sliders_become_visible_without_freezing_chromium(self):
        with tempfile.TemporaryDirectory(prefix="gradio617-chrome-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1280,900",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=12000",
                "--dump-dom",
                f"http://127.0.0.1:{self.port}/",
            ]
            # The executable is an existing local Chrome binary and arguments
            # are passed as a fixed list without a shell.
            completed = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertIn('data-gradio-visibility-smoke="pass"', completed.stdout)
        self.assertNotIn('data-gradio-visibility-smoke="fail"', completed.stdout)

    def test_shared_tab_helpers_target_real_gradio6_tablist(self):
        with tempfile.TemporaryDirectory(prefix="gradio617-tabs-chrome-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1280,900",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=12000",
                "--dump-dom",
                f"http://127.0.0.1:{self.port}/",
            ]
            completed = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(
            r'<pre id="tab-selector-result"[^>]*>(.*?)</pre>',
            completed.stdout,
            re.DOTALL,
        )
        self.assertIsNotNone(match, completed.stdout[-3000:])
        self.assertIn('data-ready="true"', match.group(0))
        result = json.loads(html.unescape(match.group(1)))
        self.assertIn("tab-container", result["tabListClass"])
        self.assertEqual(result["tabListRole"], "tablist")
        self.assertTrue(result["mirrorExists"])
        self.assertTrue(result["mirrorExcluded"])
        self.assertEqual(result["txt2imgPanel"], "tab_txt2img")
        self.assertEqual(result["img2imgPanel"], "tab_img2img")
        self.assertEqual(result["extrasPanel"], "tab_extras")
        self.assertTrue(result["extrasSelected"])
        self.assertTrue(result["img2imgSelected"])
        self.assertEqual(result["nestedSelectedIndex"], 0)

    def test_input_accordion_setup_is_late_mount_safe_and_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="gradio617-input-accordion-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1280,900",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=12000",
                "--dump-dom",
                f"http://127.0.0.1:{self.port}/",
            ]
            completed = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(
            r'<pre id="input-accordion-result"[^>]*>(.*?)</pre>',
            completed.stdout,
            re.DOTALL,
        )
        self.assertIsNotNone(match, completed.stdout[-3000:])
        self.assertIn('data-ready="true"', match.group(0))
        result = json.loads(html.unescape(match.group(1)))
        self.assertTrue(result["first"])
        self.assertTrue(result["second"])
        self.assertTrue(result["toggled"])
        self.assertEqual(result["visibleCheckboxes"], 1)
        self.assertTrue(result["hiddenChecked"])
        self.assertTrue(result["visibleChecked"])
        self.assertTrue(result["open"])

    def test_controlnet_active_units_waits_for_lazy_panels_once(self):
        with tempfile.TemporaryDirectory(prefix="gradio617-controlnet-lazy-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1280,900",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=12000",
                "--dump-dom",
                f"http://127.0.0.1:{self.port}/",
            ]
            completed = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                # Windows CIのChrome終了待ちに余裕を持たせる。DOM内の判定は変更しない。
                timeout=60,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(
            r'<pre id="controlnet-active-units-result"[^>]*>(.*?)</pre>',
            completed.stdout,
            re.DOTALL,
        )
        self.assertIsNotNone(match, completed.stdout[-3000:])
        self.assertIn('data-ready="true"', match.group(0))
        result = json.loads(html.unescape(match.group(1)))
        self.assertEqual(result["partialBadgeCount"], 0)
        self.assertEqual(result["txtBadgeCount"], 1)
        self.assertEqual(result["imgBadgeCount"], 1)
        self.assertEqual(result["txtBadgeText"], "1x Unit", result)
        self.assertEqual(result["imgBadgeText"], "1x Unit", result)
        self.assertTrue(result["txtActive"], result)
        self.assertTrue(result["imgActive"], result)
        expected_listeners = {
            f"fixture-controlnet-{tab_name}-{control}:change"
            for tab_name in ("txt2img", "img2img")
            for control in ("enabled", "upload", "radio")
        }
        self.assertEqual(set(result["listenerCounts"]), expected_listeners)
        self.assertTrue(
            all(count == 1 for count in result["listenerCounts"].values()),
            result,
        )
        self.assertEqual(result["errors"], [])


class LazyLifecycleFixtureHandler(BaseHTTPRequestHandler):
    script_paths = {
        "/ui.js": ROOT / "javascript" / "ui.js",
        "/settings.js": ROOT / "javascript" / "settings.js",
        "/token-counters.js": ROOT / "javascript" / "token-counters.js",
        "/imageDragAndDrop.js": ROOT / "javascript" / "imageDragAndDrop.js",
    }

    def log_message(self, *_):
        return

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path in self.script_paths:
            self.send_bytes(
                self.script_paths[self.path].read_bytes(),
                "text/javascript; charset=utf-8",
            )
            return
        if self.path == "/":
            self.send_bytes(self.fixture_html().encode(), "text/html; charset=utf-8")
            return
        self.send_bytes(b"not found", "text/plain", 404)

    @staticmethod
    def fixture_html():
        return """<!doctype html>
<html><head><meta charset="utf-8"></head><body>
<div id="tabs">
  <div class="tab-wrapper">
    <div class="tab-container visually-hidden" aria-hidden="true"><button>mirror</button></div>
    <div class="tab-container" role="tablist">
      <button role="tab" class="selected" aria-selected="true" aria-controls="tab_txt2img">txt2img</button>
      <button role="tab" aria-selected="false" aria-controls="tab_img2img">img2img</button>
      <button role="tab" aria-selected="false" aria-controls="tab_extras">Extras</button>
      <button role="tab" aria-selected="false" aria-controls="tab_settings">Settings</button>
    </div>
  </div>
  <div id="tab_txt2img" class="tabitem"></div>
</div>
<div id="settings_json"><textarea>{"disable_token_counters":false,"remove_image_on_hover":true,"_categories":[]}</textarea></div>
<div id="lazy-mounts"></div>
<pre id="lazy-result">pending</pre>
<script>
window.fixtureErrors = [];
window.fixtureListenerCounts = {};
window.fixtureMetrics = { tokenClicks: 0, removeClicks: 0, inputEvents: 0 };
window.fixtureStarted = performance.now();
window.addEventListener("error", (event) => fixtureErrors.push(event.message));
window.addEventListener("unhandledrejection", (event) => fixtureErrors.push(String(event.reason)));
const nativeConsoleError = console.error.bind(console);
console.error = (...args) => {
  fixtureErrors.push(args.map(String).join(" "));
  nativeConsoleError(...args);
};
const nativeAddEventListener = EventTarget.prototype.addEventListener;
EventTarget.prototype.addEventListener = function(type, listener, options) {
  if (this.id) {
    const key = `${this.id}:${type}`;
    fixtureListenerCounts[key] = (fixtureListenerCounts[key] || 0) + 1;
  }
  return nativeAddEventListener.call(this, type, listener, options);
};

const uiLoadedCallbacks = [];
const uiUpdateCallbacks = [];
const uiTabChangeCallbacks = [];
const optionsChangedCallbacks = [];
const optionsAvailableCallbacks = [];
let opts = {};
const localization = {};
window.gradioApp = () => document;
window.onUiLoaded = (callback) => uiLoadedCallbacks.push(callback);
window.onUiUpdate = (callback) => uiUpdateCallbacks.push(callback);
window.onUiTabChange = (callback) => uiTabChangeCallbacks.push(callback);
window.onOptionsChanged = (callback) => optionsChangedCallbacks.push(callback);
window.onOptionsAvailable = (callback) => {
  if (Object.keys(opts).length) callback();
  else optionsAvailableCallbacks.push(callback);
};
window.executeCallbacks = (callbacks, value) => callbacks.forEach((callback) => callback(value));
window.get_uiCurrentTab = () => document.querySelector(
  "#tabs > .tab-wrapper > .tab-container[role='tablist'] > button[aria-selected='true']"
);
window.localGet = () => null;
window.updateInput = (input) => {
  fixtureMetrics.inputEvents += 1;
  input.dispatchEvent(new Event("input", { bubbles: true }));
};

function promptMarkup(tabname) {
  return `
    <button id="${tabname}_restore_progress"></button>
    <div class="prompt-row">
      <div id="${tabname}_token_counter"></div>
      <div id="${tabname}_prompt"><label><textarea id="${tabname}_prompt_input"></textarea></label></div>
      <button id="${tabname}_token_button"></button>
    </div>
    <div class="prompt-row">
      <div id="${tabname}_negative_token_counter"></div>
      <div id="${tabname}_neg_prompt"><label><textarea id="${tabname}_neg_prompt_input"></textarea></label></div>
      <button id="${tabname}_negative_token_button"></button>
    </div>
    <div id="${tabname}_width"><input id="${tabname}_width_input" type="number" value="64"></div>
    <div id="${tabname}_height"><input id="${tabname}_height_input" type="number" value="64"></div>
    <div id="${tabname}_styles"><div class="token">Style<div class="token-remove">remove</div></div></div>`;
}

function wireFixtureMetrics(root) {
  root.querySelectorAll("button[id$='_token_button']").forEach((button) => {
    button.addEventListener("click", () => fixtureMetrics.tokenClicks += 1);
  });
  root.querySelectorAll(".token-remove").forEach((remove) => {
    remove.addEventListener("click", () => fixtureMetrics.removeClicks += 1);
  });
}

const txtPanel = document.querySelector("#tab_txt2img");
txtPanel.innerHTML = promptMarkup("txt2img");
wireFixtureMetrics(txtPanel);

function selectTab(panelId) {
  document.querySelectorAll("#tabs [role='tablist'] > button[role='tab']").forEach((button) => {
    const selected = button.getAttribute("aria-controls") === panelId;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  uiTabChangeCallbacks.forEach((callback) => callback());
}

function mountImg2img() {
  const panel = document.createElement("div");
  panel.id = "tab_img2img";
  panel.className = "tabitem";
  panel.innerHTML = promptMarkup("img2img");
  document.querySelector("#tabs").appendChild(panel);
  wireFixtureMetrics(panel);
  selectTab("tab_img2img");
}

function mountSettings() {
  const host = document.querySelector("#lazy-mounts");
  host.insertAdjacentHTML("beforeend", `
    <div id="settings">
      <div class="tab-wrapper">
        <div class="tab-container visually-hidden" aria-hidden="true"><button>mirror</button></div>
        <div class="tab-container" role="tablist"><button>General</button><button>Advanced</button></div>
      </div>
      <div id="settings_pages" class="tabitem"><div id="column_settings_general">
      <div id="setting_alpha">Alpha option</div><div id="setting_beta">Beta option</div>
    </div><button id="settings_show_one_page"></button></div></div>
    <div id="settings_search"><label><input id="settings_search_input"></label></div>
    <button id="settings_show_all_pages"></button>`);
  selectTab("tab_settings");
}

function mountImageTargets() {
  const host = document.querySelector("#lazy-mounts");
  host.insertAdjacentHTML("beforeend", `
    <div id="extras_image"><button aria-label="Remove Image"></button></div>
    <div id="pnginfo_image"><button aria-label="Remove Image"></button></div>`);
  host.querySelectorAll('[aria-label="Remove Image"]').forEach((button) => {
    button.addEventListener("click", () => fixtureMetrics.removeClicks += 1);
  });
  selectTab("tab_extras");
}

new MutationObserver((records) => {
  uiUpdateCallbacks.forEach((callback) => callback(records));
}).observe(document.documentElement, { childList: true, subtree: true });
</script>
<script src="/ui.js"></script>
<script src="/settings.js"></script>
<script src="/token-counters.js"></script>
<script src="/imageDragAndDrop.js"></script>
<script>
setTimeout(async () => {
  uiLoadedCallbacks.forEach((callback) => callback());
  await new Promise((resolve) => setTimeout(resolve, 100));
  mountImg2img();
  mountSettings();
  mountImageTargets();
  uiTabChangeCallbacks.forEach((callback) => callback());
  await new Promise((resolve) => setTimeout(resolve, 250));

  const width = document.querySelector("#img2img_width input");
  const height = document.querySelector("#img2img_height input");
  const paste = new Event("paste", { bubbles: true, cancelable: true });
  Object.defineProperty(paste, "clipboardData", {
    value: { getData: () => "512 x 768" }
  });
  width.addEventListener("input", () => fixtureMetrics.inputEvents += 1);
  height.addEventListener("input", () => fixtureMetrics.inputEvents += 1);
  width.dispatchEvent(paste);

  const search = document.querySelector("#settings_search_input");
  search.value = "alpha";
  search.dispatchEvent(new Event("input", { bubbles: true }));

  document.querySelector("#img2img_prompt_input")
    .dispatchEvent(new Event("input", { bubbles: true }));

  const drag = new Event("dragover", { bubbles: true, cancelable: true });
  Object.defineProperty(drag, "dataTransfer", {
    value: { types: ["text/uri-list"] }
  });
  document.querySelector("#extras_image").dispatchEvent(drag);

  await new Promise((resolve) => setTimeout(resolve, 1200));
  const result = document.querySelector("#lazy-result");
  result.dataset.ready = "true";
  result.textContent = JSON.stringify({
    errors: fixtureErrors,
    listenerCounts: fixtureListenerCounts,
    width: width.value,
    height: height.value,
    alphaDisplay: document.querySelector("#setting_alpha").style.display,
    betaDisplay: document.querySelector("#setting_beta").style.display,
    tokenClicks: fixtureMetrics.tokenClicks,
    removeClicks: fixtureMetrics.removeClicks,
    inputEvents: fixtureMetrics.inputEvents,
    elapsedMs: performance.now() - fixtureStarted,
    settingsTabLabels: Array.from(
      document.querySelectorAll("#settings > .tab-wrapper > .tab-container[role='tablist'] > button")
    ).map((button) => button.textContent),
    settingsWrapperChildren: document.querySelector("#settings > .tab-wrapper").children.length,
    settingsSearchParent: document.querySelector("#settings_search").parentElement.id,
    settingsShowAllParent: document.querySelector("#settings_show_all_pages").parentElement.id,
    bindings: {
      settings: search.dataset.forgeSettingsSearchBound,
      imgWidth: width.dataset.forgeResolutionPasteBound,
      imgStyle: document.querySelector("#img2img_styles").dataset.forgeStyleDeselectionBound,
      imgToken: document.querySelector("#img2img_prompt_input").dataset.forgeTokenCounterBound,
      extras: document.querySelector("#extras_image").dataset.forgeImageDropBound,
      pnginfo: document.querySelector("#pnginfo_image").dataset.forgeImageDropBound,
    },
  });
}, 0);
</script></body></html>"""


class GradioLazyLifecycleChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LazyLifecycleFixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_lazily_mounted_targets_bind_once_without_console_errors(self):
        with tempfile.TemporaryDirectory(prefix="gradio-lazy-js-chrome-") as profile:
            completed = subprocess.run(  # noqa: S603
                [
                    self.chromium,
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--window-size=1280,900",
                    f"--user-data-dir={profile}",
                    "--virtual-time-budget=6000",
                    "--dump-dom",
                    self.url,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(
            r'<pre id="lazy-result"[^>]*>(.*?)</pre>',
            completed.stdout,
            re.DOTALL,
        )
        self.assertIsNotNone(match, completed.stdout[-4000:])
        self.assertIn('data-ready="true"', match.group(0))
        result = json.loads(html.unescape(match.group(1)))
        self.assertEqual(result["errors"], [], result)
        self.assertEqual(result["width"], "512")
        self.assertEqual(result["height"], "768")
        self.assertEqual(result["alphaDisplay"], "")
        self.assertEqual(result["betaDisplay"], "none")
        self.assertEqual(result["tokenClicks"], 1)
        self.assertEqual(result["removeClicks"], 1)
        self.assertEqual(result["inputEvents"], 2)
        self.assertLess(result["elapsedMs"], 5000)
        self.assertEqual(result["settingsTabLabels"], ["General", "Advanced"])
        self.assertEqual(result["settingsWrapperChildren"], 2)
        self.assertEqual(result["settingsSearchParent"], "lazy-mounts")
        self.assertEqual(result["settingsShowAllParent"], "lazy-mounts")
        self.assertTrue(all(value == "true" for value in result["bindings"].values()))
        expected_once = {
            "settings_search_input:input",
            "settings_show_all_pages:click",
            "img2img_width_input:paste",
            "img2img_height_input:paste",
            "img2img_styles:click",
            "img2img_prompt_input:input",
            "img2img_neg_prompt_input:input",
            "extras_image:dragover",
            "pnginfo_image:dragover",
        }
        for key in expected_once:
            with self.subTest(listener=key):
                self.assertEqual(result["listenerCounts"].get(key), 1, result)


class GradioFullUiChromiumTests(unittest.TestCase):
    """Opt-in bounded probe against a separately started full Forge UI."""

    @classmethod
    def setUpClass(cls):
        cls.url = os.environ.get("AIKIMI_FULL_UI_URL", "").rstrip("/") + "/"
        if cls.url == "/":
            raise unittest.SkipTest("set AIKIMI_FULL_UI_URL to probe the full UI")

        parsed = urlparse(cls.url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise RuntimeError("AIKIMI_FULL_UI_URL must use loopback HTTP")

        cls.chromium = find_chromium()
        if not cls.chromium:
            raise RuntimeError("Chrome or Chromium is required when AIKIMI_FULL_UI_URL is set")

    def test_full_config_keeps_lazy_components_unmounted(self):
        with urlopen(urljoin(self.url, "config"), timeout=10) as response:  # noqa: S310
            config = json.load(response)

        components = config["components"]
        lazy_count = sum(component.get("props", {}).get("visible") is False for component in components)
        mounted_hidden_count = sum(component.get("props", {}).get("visible") == "hidden" for component in components)
        element_ids = {component.get("props", {}).get("elem_id") for component in components}
        self.assertGreater(lazy_count, 0)
        self.assertGreater(mounted_hidden_count, 0)
        self.assertLess(mounted_hidden_count, lazy_count)
        self.assertTrue(
            {
                "aikimi-txt2img-anima38",
                "h3-generate",
                "script_krea2_2stage_upscale_quick_4k",
                "sn-generate",
            }.issubset(element_ids)
        )

    def test_full_page_reaches_navigation_within_sixty_seconds(self):
        blocked_urls = tuple(
            filter(
                None,
                os.environ.get("AIKIMI_FULL_UI_BLOCKED_URLS", "").split(";"),
            )
        )
        try:
            state = probe_navigation_with_cdp(
                self.chromium,
                self.url,
                blocked_urls=blocked_urls,
                timeout=60,
            )
        except TimeoutError as error:
            self.fail(str(error))

        self.assertTrue(state["ready"], state)
        self.assertGreater(state["navCount"], 0)
        self.assertIn("txt2img", state["labels"])
        self.assertLess(state["elapsed_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
