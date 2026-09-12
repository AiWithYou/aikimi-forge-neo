"""実HTTPでH3の正常完了・応答消失・アプリ再起動を検証する。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil

from modules.fifo_lock import FIFOLock
from modules_forge import gpu_ownership
from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_pending as pending

ROOT = Path(__file__).resolve().parents[2]


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.media = self.root / "runtime/input/forge_h3/123456789abc_reference.png"
        self.media.parent.mkdir(parents=True)
        self.media.write_bytes(b"reference")
        self.record_dir = self.root / "pending"
        self.enterContext(patch.object(pending, "DIRECTORY", self.record_dir))
        self.enterContext(patch.object(gpu_ownership, "queue_lock", FIFOLock()))
        self.enterContext(patch.object(bridge, "_GPU_OWNERSHIPS", {}))
        self.enterContext(patch.object(bridge, "_ACTIVE_GENERATION_IDS", set()))
        self.enterContext(patch.object(bridge, "_CANCELLED_JOB_IDS", set()))
        self.enterContext(patch.object(bridge, "_CANCEL_ACK_IDS", set()))
        self.enterContext(patch.object(bridge, "release_forge_vram"))
        self.enterContext(patch.object(bridge, "ensure_ready", return_value=SimpleNamespace()))
        self.enterContext(patch.object(bridge, "_validate_request_runtime_constraints"))
        self.enterContext(
            patch.object(bridge, "prepare_media", return_value={"images": ["forge_h3/" + self.media.name]})
        )
        self.enterContext(patch.object(bridge, "build_workflow", return_value={"fixture": {}}))
        self.enterContext(patch.object(bridge, "extract_history_video", return_value=self.root / "source.mp4"))
        self.enterContext(patch.object(bridge, "mirror_result", return_value=self.root / "result.mp4"))
        self.enterContext(patch.object(bridge, "_loopback_server_process", return_value=psutil.Process()))
        self.submissions = []
        self.status = "in_progress"
        self.drop_response = False
        self.cancel_ack = False
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, value, status=200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/prompt":
                    # サーバー受付時点でID・素材・送信先の記録が既に存在する。
                    records = pending.read_all()
                    if len(records) != 1 or records[0]["prompt_id"] != body["prompt_id"] or not case.media.exists():
                        self.reply({"error": "missing write-ahead record"}, 500)
                        return
                    case.submissions.append(body["prompt_id"])
                    if case.drop_response:
                        self.close_connection = True
                        return
                    self.reply({"prompt_id": body["prompt_id"]})
                elif self.path.endswith("/cancel"):
                    self.reply({"cancelled": case.cancel_ack})
                else:
                    self.reply({})

            def do_GET(self):
                if self.path.startswith("/history/"):
                    self.reply({})
                elif case.status == "missing":
                    self.reply({"error": "Job not found"}, 404)
                else:
                    self.reply({"status": case.status})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def generation(self):
        return bridge.run_generation(
            bridge.H3Request(mode=bridge.MODE_TEXT, prompt="test"),
            self.root / "runtime",
            self.url,
            self.root / "logs",
            self.root / "outputs",
            poll_seconds=0,
        )

    def test_normal_completion_reclaims_inputs_and_gpu(self):
        self.status = "completed"
        events = list(self.generation())
        self.assertEqual(events[-1]["stage"], "complete")
        self.assertEqual(len(self.submissions), 1)
        self.assertFalse(self.media.exists())
        self.assertEqual(pending.read_all(), [])
        owner = gpu_ownership.GPUOwnership()
        self.assertTrue(owner.acquire())
        owner.release()

    def test_accepted_request_with_lost_response_uses_same_id_until_complete(self):
        self.drop_response = True
        events = self.generation()
        self.addCleanup(events.close)
        next(events)
        next(events)
        queued = next(events)
        self.assertEqual(self.submissions, [queued["prompt_id"]])
        self.assertTrue(self.media.exists())
        self.assertEqual(pending.read_all()[0]["state"], "submitting")
        other = gpu_ownership.GPUOwnership()
        self.assertFalse(other.acquire())
        self.status = "completed"
        self.assertEqual(list(events)[-1]["stage"], "complete")
        self.assertEqual(len(self.submissions), 1)
        self.assertFalse(self.media.exists())
        self.assertEqual(pending.read_all(), [])
        self.assertTrue(other.acquire())
        other.release()

    def test_restart_reconciles_record_before_allowing_forge_api_lock(self):
        self.drop_response = True
        child = r"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import psutil
from modules_forge import minimax_h3_bridge as b, minimax_h3_pending as p
root, url, server_pid = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
p.DIRECTORY = root / 'pending'
b.release_forge_vram = lambda: None
b.ensure_ready = lambda *a, **k: SimpleNamespace()
b._validate_request_runtime_constraints = lambda *a: None
b._loopback_server_process = lambda *a: psutil.Process(server_pid)
b.prepare_media = lambda *a: {'images': ['forge_h3/123456789abc_reference.png']}
b.build_workflow = lambda *a, **k: {'fixture': {}}
g = b.run_generation(b.H3Request(mode=b.MODE_TEXT, prompt='test'), root / 'runtime', url, root / 'logs', root / 'outputs')
next(g); next(g); next(g)
os._exit(0)
"""
        subprocess.run(  # noqa: S603 -- 固定テストコードを実行する。
            [sys.executable, "-c", child, str(self.root), self.url, str(os.getpid())], cwd=ROOT, check=True, timeout=20
        )
        self.assertEqual(len(self.submissions), 1)
        self.assertTrue(self.media.exists())
        self.assertEqual(len(pending.read_all()), 1)
        restarted = r"""
import sys, time
from pathlib import Path
from modules_forge import gpu_ownership as gpu, minimax_h3_pending as pending
root = Path(sys.argv[1])
pending.DIRECTORY = root / 'pending'
if gpu.queue_lock.acquire(False):
    raise RuntimeError('Forge/API acquired GPU before reconciliation')
(root / 'blocked').touch()
with gpu.queue_lock:
    if list(pending.DIRECTORY.glob('*.json')):
        raise RuntimeError('GPU released before terminal cleanup')
(root / 'recovered').touch()
"""
        process = subprocess.Popen([sys.executable, "-c", restarted, str(self.root)], cwd=ROOT)  # noqa: S603 -- 固定テストコードを実行する。
        try:
            deadline = time.monotonic() + 15
            while not (self.root / "blocked").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue((self.root / "blocked").exists())
            self.assertTrue(self.media.exists())
            # 古い入力でも未確定記録に属する間は削除しない。
            os.utime(self.media, (0, 0))
            bridge.cleanup_stale_prepared_media(self.root / "runtime")
            self.assertTrue(self.media.exists())
            self.status = "completed"
            self.assertEqual(process.wait(timeout=15), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        self.assertTrue((self.root / "recovered").exists())
        self.assertFalse(self.media.exists())
        self.assertEqual(pending.read_all(), [])
        self.assertEqual(len(self.submissions), 1)

    def test_cancel_intent_and_404_without_ack_keep_inputs_and_gpu(self):
        events = self.generation()
        next(events)
        next(events)
        prompt_id = next(events)["prompt_id"]
        self.status = "missing"
        with patch.object(bridge, "_schedule_deferred_cleanup"):
            events.close()
        self.assertTrue(self.media.exists())
        self.assertFalse(gpu_ownership.GPUOwnership().acquire())
        client = bridge.ComfyH3Client(self.url)
        bridge._cleanup_after_terminal(
            client, prompt_id, {"images": ["forge_h3/" + self.media.name]}, self.root / "runtime", wait_seconds=0.05
        )
        self.assertTrue(self.media.exists())
        self.assertEqual(len(pending.read_all()), 1)
        self.status = "failed"
        client = bridge.ComfyH3Client(self.url)
        bridge._cleanup_after_terminal(
            client, prompt_id, {"images": ["forge_h3/" + self.media.name]}, self.root / "runtime", wait_seconds=1
        )
        self.assertFalse(self.media.exists())


if __name__ == "__main__":
    unittest.main()
