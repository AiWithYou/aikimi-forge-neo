"""遅延描画されたタブでも生成ボタンの移動先と操作を保つ。"""

import html
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.tests.chromium_helpers import find_chromium

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "extensions-builtin/mobile/javascript/mobile.js"
FIXTURE = """<!doctype html><meta charset="utf-8">
<style>
section{position:relative} .results{position:relative;margin-left:360px}
.narrow .results{margin-left:0} [hidden]{display:none!important}
</style><main></main><pre id="result"></pre>
<script>
const loaded=[], changed=[], updated=[], errors=[];
window.gradioApp=()=>document;
window.onUiLoaded=fn=>loaded.push(fn);
window.onUiTabChange=fn=>changed.push(fn);
window.onAfterUiUpdate=fn=>updated.push(fn);
window.addEventListener('error',e=>errors.push(e.message));
const root=document.querySelector('main');
const get=id=>document.getElementById(id);
function mount(tab, hidden=false) {
    const section=document.createElement('section'); section.id=tab; section.hidden=hidden;
    section.innerHTML=`<div id="${tab}_actions_column"><div id="${tab}_generate_box"><button>生成</button></div></div><div class="results" id="${tab}_results"></div>`;
    root.append(section);
}
function run(callbacks) { callbacks.forEach(fn=>fn()); }
function resize(narrow) { root.classList.toggle('narrow',narrow); window.dispatchEvent(new Event('resize')); }
function state(tab) {return {parent:get(tab+'_generate_box')?.parentElement.id, mobile:get(tab+'_results')?.classList.contains('mobile')};}
</script>
"""


class MobileLayoutChromiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chrome or Chromium is not installed")

    def run_scenario(self, scenario):
        source = SCRIPT.read_text(encoding="utf-8")
        document = FIXTURE + "<script>" + source + "</script><script>"
        document += "const checks={}; try {" + scenario + "} catch(e) {errors.push(e.message);}"
        document += "get('result').textContent=JSON.stringify({checks,errors});</script>"
        with tempfile.TemporaryDirectory(prefix="aikimi-mobile-test-") as directory:
            page = Path(directory) / "fixture.html"
            page.write_text(document, encoding="utf-8")
            result = subprocess.run(  # noqa: S603
                [
                    self.chromium,
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-data-dir={Path(directory) / 'profile'}",
                    "--dump-dom",
                    page.as_uri(),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-1000:])
        match = re.search(r'<pre id="result">(.*?)</pre>', result.stdout, re.DOTALL)
        self.assertIsNotNone(match, result.stdout[-2000:])
        payload = json.loads(html.unescape(match.group(1)))
        self.assertEqual(payload["errors"], [])
        return payload["checks"]

    def test_resize_with_only_one_tab_mounted(self):
        checks = self.run_scenario("""
            mount('txt2img'); run(loaded); resize(true); checks.small=state('txt2img');
            resize(false); checks.wide=state('txt2img');
        """)
        self.assertEqual(checks["small"], {"parent": "txt2img_results", "mobile": True})
        self.assertEqual(checks["wide"], {"parent": "txt2img_actions_column", "mobile": False})

    def test_late_mount_and_hidden_tab_are_reconciled_when_shown(self):
        checks = self.run_scenario("""
            resize(true); run(loaded); mount('txt2img'); run(updated);
            checks.first=state('txt2img'); mount('img2img',true); run(updated);
            checks.hidden=state('img2img'); get('txt2img').hidden=true; get('img2img').hidden=false;
            run(changed); checks.second=state('img2img');
        """)
        self.assertEqual(checks["first"]["parent"], "txt2img_results")
        self.assertEqual(checks["hidden"]["parent"], "img2img_actions_column")
        self.assertEqual(checks["second"], {"parent": "img2img_results", "mobile": True})

    def test_missing_destination_and_replaced_controls_recover(self):
        checks = self.run_scenario("""
            mount('txt2img'); resize(true); const actions=get('txt2img_actions_column'); actions.remove();
            resize(false); checks.waiting=state('txt2img'); get('txt2img').prepend(actions); run(updated);
            checks.restored=state('txt2img'); resize(true);
            get('txt2img_generate_box').remove(); actions.innerHTML='<div id="txt2img_generate_box"><button>生成</button></div>';
            run(updated); checks.replaced=state('txt2img');
        """)
        self.assertEqual(checks["waiting"]["parent"], "txt2img_results")
        self.assertEqual(checks["restored"], {"parent": "txt2img_actions_column", "mobile": False})
        self.assertEqual(checks["replaced"], {"parent": "txt2img_results", "mobile": True})

    def test_repeated_updates_preserve_button_and_compact_layout(self):
        checks = self.run_scenario("""
            mount('txt2img'); const box=get('txt2img_generate_box'); let clicks=0;
            box.querySelector('button').addEventListener('click',()=>clicks++);
            resize(true); let moves=0; const results=get('txt2img_results'); const insert=results.insertBefore;
            results.insertBefore=function(...args){moves++;return insert.apply(this,args);};
            for(let i=0;i<10;i++){run(updated);run(changed);resize(true);}
            box.querySelector('button').click(); checks.moves=moves; checks.clicks=clicks;
            checks.same=get('txt2img_generate_box')===box;
            resize(false); const compact=document.createElement('div'); compact.className='toprow-compact-tools'; root.append(compact);
            resize(true); run(updated); checks.compact=state('txt2img');
        """)
        self.assertEqual(checks["moves"], 0)
        self.assertEqual(checks["clicks"], 1)
        self.assertTrue(checks["same"])
        self.assertEqual(checks["compact"], {"parent": "txt2img_actions_column", "mobile": False})


if __name__ == "__main__":
    unittest.main()
