"""Real local HTTP regressions for P-H3 submission ownership."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import psutil

from modules.fifo_lock import FIFOLock
from modules_forge import gpu_ownership, gpu_residency, local_assets
from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_pending as pending
from modules_forge import p_h3_image as studio


class ImageSubmissionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory(prefix="p-h3-job-lifecycle-")))
        self.runtime = self.root / "runtime"
        self.lock = FIFOLock()
        self.enterContext(patch.object(gpu_ownership, "queue_lock", self.lock))
        self.enterContext(patch.object(gpu_residency, "register"))
        self.enterContext(patch.object(local_assets, "LIBRARY", self.root / "assets/library.json"))
        self.enterContext(patch.object(local_assets, "HASH_CACHE", self.root / "assets/hashes.json"))
        self.enterContext(patch.object(pending, "DIRECTORY", self.root / "pending"))
        for name, value in (
            ("_GPU_OWNERSHIPS", {}), ("_ACTIVE_GENERATION_IDS", set()),
            ("_CANCELLED_JOB_IDS", set()), ("_CANCEL_ACK_IDS", set()),
        ):
            self.enterContext(patch.object(bridge, name, value))
        self.enterContext(patch.object(bridge, "release_forge_vram"))
        self.enterContext(patch.object(bridge, "cleanup_stale_prepared_media"))
        self.enterContext(patch.object(bridge, "_loopback_server_process", return_value=psutil.Process()))
        self.deferred = self.enterContext(patch.object(bridge, "_schedule_deferred_cleanup"))
        self.client_class = bridge.ComfyH3Client
        self.clients = []

        def client(*args, **kwargs):
            created = self.client_class(*args, **kwargs)
            self.clients.append(created)
            return created

        self.enterContext(patch.object(bridge, "ComfyH3Client", side_effect=client))
        self.submissions = []
        self.reject = False
        self.status = "in_progress"
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
                    if case.reject:
                        self.reply({"error": "fixture validation rejection"}, 400)
                        return
                    records = pending.read_all()
                    if len(records) != 1 or records[0]["prompt_id"] != body["prompt_id"]:
                        self.reply({"error": "missing write-ahead record"}, 500)
                        return
                    case.submissions.append(body["prompt_id"])
                    self.reply({"prompt_id": body["prompt_id"]})
                elif self.path.endswith("/cancel"):
                    self.reply({"cancelled": False})
                else:
                    self.reply({}, 404)

            def do_GET(self):
                self.reply({"status": case.status})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.enterContext(patch.object(studio, "ensure_runtime", return_value=SimpleNamespace()))
        self.enterContext(patch.object(studio, "prepare_images", return_value={"images": []}))
        self.enterContext(patch.object(studio, "check_nodes"))
        self.enterContext(patch("modules_forge.studio_checkpoints.snapshot_checkpoint", return_value=None))
        self.enterContext(patch("modules_forge.studio_checkpoints.verify_snapshot"))
        self.addCleanup(self.close_owned_resources)

    def close_owned_resources(self):
        for prompt_id in list(bridge._GPU_OWNERSHIPS):
            bridge._finish_gpu_generation(prompt_id)
        for client in self.clients:
            client.close()

    def generation(self):
        request = studio.PH3Request("test", seed=42, use_upsampler=False)
        updates = studio.run_generation(request, self.runtime, self.url, self.root / "logs", self.root / "outputs", poll_seconds=0)
        self.addCleanup(updates.close)
        self.assertEqual(next(updates)["stage"], "verifying")
        self.assertEqual(next(updates)["stage"], "runtime")
        return updates

    def assert_deferred_then_recovered(self):
        self.assertEqual(len(self.submissions), 1)
        prompt_id = self.submissions[0]
        self.deferred.assert_called_once_with(self.clients[0], prompt_id, {"images": []}, self.runtime)
        self.assertFalse(self.clients[0]._client.is_closed)
        self.assertFalse(self.lock.acquire(False), "running remote job must retain GPU ownership")
        self.assertEqual(pending.read_all()[0]["state"], "submitting")
        self.status = "completed"
        bridge._cleanup_after_terminal(*self.deferred.call_args.args, wait_seconds=1)
        self.assertTrue(self.clients[0]._client.is_closed)
        self.assertEqual(pending.read_all(), [])
        self.assertEqual(bridge._GPU_OWNERSHIPS, {})
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()
        self.assertEqual(len(self.submissions), 1, "recovery must never resubmit")

    def test_record_update_failure_after_acceptance_transfers_gpu_to_supervisor(self):
        updates = self.generation()
        original_update = pending.update

        def update(prompt_id, **changes):
            if changes.get("state") == "submitted":
                raise OSError("injected submitted-record failure")
            return original_update(prompt_id, **changes)

        with patch.object(pending, "update", side_effect=update):
            with self.assertRaisesRegex(OSError, "submitted-record failure"):
                next(updates)
        self.assert_deferred_then_recovered()

    def test_unexpected_failure_after_acceptance_transfers_gpu_to_supervisor(self):
        updates = self.generation()
        original_submit = self.client_class.submit

        def submit(client, workflow, prompt_id):
            original_submit(client, workflow, prompt_id)
            raise OSError("injected post-acceptance failure")

        with patch.object(self.client_class, "submit", submit):
            with self.assertRaisesRegex(OSError, "post-acceptance failure"):
                next(updates)
        self.assert_deferred_then_recovered()

    def test_persistent_record_update_failure_still_reconciles_completion(self):
        updates = self.generation()
        with patch.object(pending, "update", side_effect=OSError("injected persistent-record failure")):
            with self.assertRaisesRegex(OSError, "persistent-record failure"):
                next(updates)
            self.assert_deferred_then_recovered()

    def test_record_failure_before_send_does_not_submit_and_returns_gpu(self):
        updates = self.generation()
        with patch.object(pending, "write", side_effect=OSError("injected write-ahead failure")):
            with self.assertRaisesRegex(OSError, "write-ahead failure"):
                next(updates)
        self.assertEqual(self.submissions, [])
        self.deferred.assert_not_called()
        self.assertEqual(pending.read_all(), [])
        self.assertEqual(bridge._GPU_OWNERSHIPS, {})
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()

    def test_explicit_rejection_reclaims_record_and_gpu_without_supervisor(self):
        self.reject = True
        updates = self.generation()
        with self.assertRaises(bridge.H3SubmissionRejected):
            next(updates)
        self.assertEqual(self.submissions, [])
        self.deferred.assert_not_called()
        self.assertEqual(pending.read_all(), [])
        self.assertEqual(bridge._GPU_OWNERSHIPS, {})
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()


if __name__ == "__main__":
    unittest.main()
