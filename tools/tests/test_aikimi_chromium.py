import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

ROOT = Path(__file__).resolve().parents[2]


def find_chromium():
    configured = os.environ.get("CHROME_BIN")
    candidates = [
        configured,
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


class AikimiFixtureHandler(BaseHTTPRequestHandler):
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
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_fixture(parsed.query)
            return
        if parsed.path == "/extensions-builtin/aikimi-ui/javascript/aikimiStatus.js":
            self.send_bytes(
                (ROOT / "extensions-builtin" / "aikimi-ui" / "javascript" / "aikimiStatus.js").read_bytes(),
                "text/javascript; charset=utf-8",
            )
            return
        if parsed.path == "/extensions-builtin/aikimi-ui/style.css":
            self.send_bytes(
                (ROOT / "extensions-builtin" / "aikimi-ui" / "style.css").read_bytes(),
                "text/css; charset=utf-8",
            )
            return
        if parsed.path == "/aikimi-assets/manifest.json":
            self.send_bytes(
                (ROOT / "assets" / "aikimi" / "manifest.json").read_bytes(),
                "application/json",
            )
            return
        if parsed.path.startswith("/aikimi-assets/"):
            filename = Path(parsed.path).name
            asset = ROOT / "assets" / "aikimi" / filename
            if not asset.is_file():
                self.send_bytes(b"not found", "text/plain", 404)
                return
            content_type = "image/webp" if asset.suffix == ".webp" else "image/png"
            self.send_bytes(asset.read_bytes(), content_type)
            return
        if parsed.path == "/internal/aikimi-status":
            payload = {
                "generation": {"active": False, "progress": 0, "queue_size": 0},
                "model": {"loading": False, "loaded": False, "reload_pending": False},
                "memory": {"available": False},
                "backend": {"ready": True, "uptime_seconds": 1},
            }
            self.send_bytes(json.dumps(payload).encode(), "application/json")
            return
        self.send_bytes(b"not found", "text/plain", 404)

    def send_fixture(self, query):
        values = parse_qs(query)
        size = values.get("size", ["medium"])[0]
        feature = values.get("feature", ["krea2"])[0]
        warning = values.get("warning", [None])[0]
        clear_warning = values.get("clear_warning", ["0"])[0] == "1"
        dialogue = values.get("dialogue", ["1"])[0] != "0"
        animation = values.get("animation", ["1"])[0] != "0"
        options = {
            "aikimi_assistant_enabled": True,
            "aikimi_assistant_size": size,
            "aikimi_assistant_position": "bottom-left",
            "aikimi_assistant_dialogue_enabled": dialogue,
            "aikimi_assistant_animation_enabled": animation,
        }
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><link rel="stylesheet" href="/extensions-builtin/aikimi-ui/style.css"></head>
<body><div id="tabs"><div id="aikimi-feature"></div><div id="forge-feature"></div></div><pre id="result">pending</pre>
<script>
window.opts = {json.dumps(options)};
window.gradioApp = () => document;
window.fixtureFeature = {json.dumps(feature)};
window.fixtureWarning = {json.dumps(warning)};
window.fixtureClearWarning = {json.dumps(clear_warning)};
window.fixtureWarningObserved = false;
window.fixtureRequests = [];
window.fixtureFetch = window.fetch;
window.fetch = (...args) => {{
    window.fixtureRequests.push(String(args[0]));
    return window.fixtureFetch(...args);
}};
window.AikimiTabs = {{
    getActiveFeature: () => window.fixtureFeature === "none" ? null : window.fixtureFeature,
    getActiveContainer: () => window.fixtureFeature === "none" ? null : document.querySelector("#aikimi-feature")
}};
window.onUiLoaded = (callback) => setTimeout(callback, 0);
window.onOptionsAvailable = (callback) => setTimeout(callback, 5);
window.onOptionsChanged = () => {{}};
window.onAfterUiUpdate = () => {{}};
</script>
<script src="/extensions-builtin/aikimi-ui/javascript/aikimiStatus.js"></script>
<script>
if (window.fixtureWarning) {{
    new MutationObserver((mutations) => {{
        if (mutations.some((mutation) => mutation.target?.dataset?.state === "warning")) {{
            window.fixtureWarningObserved = true;
        }}
    }}).observe(document.documentElement, {{
        attributes: true,
        attributeFilter: ["data-state"],
        subtree: true
    }});
    setTimeout(() => document.dispatchEvent(new CustomEvent("aikimi:feature-tab-change", {{
        detail: {{
            feature: window.fixtureFeature,
            container: document.querySelector("#aikimi-feature"),
            ready: false,
            warning: window.fixtureWarning
        }}
    }})), 15);
    if (window.fixtureClearWarning) {{
        setTimeout(() => document.dispatchEvent(new CustomEvent("aikimi:feature-tab-change", {{
            detail: {{
                feature: window.fixtureFeature,
                container: document.querySelector("#aikimi-feature"),
                ready: true,
                warning: null
            }}
        }})), 110);
    }}
}}
(function report(attempt) {{
    const panel = document.querySelector("#aikimi-status");
    const portrait = panel?.querySelector(".aikimi-status__portrait");
    const expectsPanel = window.fixtureFeature !== "none";
    if (expectsPanel && (!portrait?.currentSrc || !portrait.naturalWidth) && attempt < 120) {{
        setTimeout(() => report(attempt + 1), 50);
        return;
    }}
    if (!expectsPanel && attempt < 10) {{
        setTimeout(() => report(attempt + 1), 50);
        return;
    }}
    if (window.fixtureWarning && attempt < 4) {{
        setTimeout(() => report(attempt + 1), 50);
        return;
    }}
    const message = panel?.querySelector(".aikimi-status__message");
    const portraitWrap = panel?.querySelector(".aikimi-status__portrait-wrap");
    const result = document.querySelector("#result");
    result.dataset.ready = "true";
    result.textContent = JSON.stringify({{
        reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
        panelPresent: Boolean(panel),
        state: panel?.dataset.state || "",
        message: message?.textContent || "",
        technical: panel?.querySelector('[data-field="error"]')?.textContent || "",
        warningObserved: window.fixtureWarningObserved,
        src: portrait?.currentSrc || "",
        size: panel?.dataset.size,
        dialogue: panel?.dataset.dialogue,
        motion: panel?.dataset.motion,
        messageHidden: Boolean(message?.hidden),
        messageDisplay: message ? getComputedStyle(message).display : "",
        messageWidth: Math.round(message?.getBoundingClientRect().width || 0),
        characterWidth: Math.round(portraitWrap?.getBoundingClientRect().width || 0),
        summaryHeight: Math.round(panel?.querySelector("summary")?.getBoundingClientRect().height || 0),
        parentId: panel?.parentElement?.id || "",
        statusRequests: window.fixtureRequests.filter((url) => url.includes("/internal/aikimi-status")).length
    }});
}})(0);
</script></body></html>"""
        self.send_bytes(document.encode(), "text/html; charset=utf-8")


class AikimiChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AikimiFixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def render_fixture(self, query="", reduced_motion=False, window_size="1280,900"):
        url = self.base_url + (f"?{query}" if query else "")
        with tempfile.TemporaryDirectory(prefix="aikimi-chrome-") as profile:
            command = [
                self.chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                f"--window-size={window_size}",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=7000",
                "--dump-dom",
            ]
            if reduced_motion:
                command.append("--force-prefers-reduced-motion")
            command.append(url)
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        match = re.search(r'<pre id="result"[^>]*>(.*?)</pre>', completed.stdout, re.DOTALL)
        self.assertIsNotNone(match, completed.stdout[-2000:])
        self.assertIn('data-ready="true"', match.group(0))
        return json.loads(html.unescape(match.group(1)))

    def test_os_reduced_motion_uses_still_asset_in_chromium(self):
        result = self.render_fixture(reduced_motion=True)

        self.assertTrue(result["reduced"])
        self.assertEqual(urlparse(result["src"]).path, "/aikimi-assets/idle-still.webp")
        self.assertEqual(result["motion"], "reduced")

    def test_default_preferences_use_animated_asset_in_chromium(self):
        result = self.render_fixture()

        self.assertFalse(result["reduced"])
        self.assertEqual(urlparse(result["src"]).path, "/aikimi-assets/idle.png")
        self.assertEqual(result["size"], "medium")
        self.assertEqual(result["dialogue"], "on")
        self.assertEqual(result["motion"], "animated")
        self.assertFalse(result["messageHidden"])
        self.assertEqual(result["characterWidth"], 52)
        self.assertLessEqual(result["summaryHeight"], 64)
        self.assertEqual(result["parentId"], "aikimi-feature")

    def test_user_preferences_apply_in_chromium(self):
        result = self.render_fixture("size=large&dialogue=0&animation=0")

        self.assertFalse(result["reduced"])
        self.assertEqual(urlparse(result["src"]).path, "/aikimi-assets/idle-still.webp")
        self.assertEqual(result["size"], "large")
        self.assertEqual(result["dialogue"], "off")
        self.assertEqual(result["motion"], "disabled")
        self.assertTrue(result["messageHidden"])
        self.assertEqual(result["characterWidth"], 64)

    def test_normal_forge_tab_does_not_mount_or_poll_status(self):
        result = self.render_fixture("feature=none")

        self.assertFalse(result["panelPresent"])
        self.assertEqual(result["statusRequests"], 0)

    def test_narrow_layout_keeps_the_strip_compact(self):
        result = self.render_fixture(window_size="480,800")

        self.assertTrue(result["panelPresent"])
        self.assertLessEqual(result["summaryHeight"], 64)
        self.assertEqual(result["messageDisplay"], "block")
        self.assertEqual(result["messageWidth"], 1)

    def test_navigation_warning_is_visible_and_redacted(self):
        result = self.render_fixture(urlencode({"warning": r"C:\private\model token=abc123456789"}))

        self.assertTrue(result["warningObserved"], result)
        self.assertEqual(result["state"], "warning")
        self.assertIn("<local-path>", result["message"])
        self.assertIn("token=<redacted>", result["technical"])
        self.assertNotIn("abc123456789", result["message"])
        self.assertNotIn("abc123456789", result["technical"])

    def test_successful_feature_event_clears_navigation_warning(self):
        result = self.render_fixture(urlencode({"warning": "Runtime unavailable", "clear_warning": "1"}))

        self.assertTrue(result["warningObserved"], result)
        self.assertEqual(result["state"], "idle")
        self.assertNotEqual(result["message"], "Runtime unavailable")
        self.assertEqual(result["technical"], "None")


if __name__ == "__main__":
    unittest.main()
