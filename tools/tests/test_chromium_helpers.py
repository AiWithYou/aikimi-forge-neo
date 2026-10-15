from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import psutil
from websockets.exceptions import ConnectionClosedOK

from tools.tests import chromium_helpers


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("chromium", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -1

    def terminate(self):
        self.terminated = True
        self.returncode = -1


class FakeWebSocket:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    def close(self):
        self.closed = True
        if self.error is not None:
            raise self.error


class ChromiumPageLifecycleTests(unittest.TestCase):
    def start_page(self, *, page=None):
        process = FakeProcess()
        websocket = FakeWebSocket()
        page = page or Mock()
        page.send.return_value = {}
        self.enterContext(patch.object(chromium_helpers, "reserve_local_port", return_value=9222))
        self.spawn = self.enterContext(patch.object(chromium_helpers, "spawn_chromium", return_value=process))
        self.endpoint = self.enterContext(
            patch.object(
                chromium_helpers,
                "urlopen",
                return_value=BytesIO(
                    b'[{"type":"page","webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/page/fixture"}]'
                ),
            )
        )
        self.connect = self.enterContext(patch.object(chromium_helpers, "_connect_websocket", return_value=websocket))
        self.enterContext(patch.object(chromium_helpers, "CdpPage", return_value=page))
        self.cleanup = self.enterContext(patch.object(chromium_helpers, "close_chromium"))
        return process, websocket, page

    def test_uses_initial_foreground_page_without_creating_background_target(self):
        _, _, page = self.start_page()
        self.endpoint.return_value = BytesIO(
            b'[{"type":"page","url":"about:blank","webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/page/fixture"}]'
        )
        with chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture"):
            pass
        request = self.endpoint.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9222/json/list")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(
            page.send.call_args_list[:4],
            [
                call("Page.enable"),
                call("Runtime.enable"),
                call("Page.bringToFront"),
                call("Page.navigate", {"url": "http://127.0.0.1/fixture"}, timeout=30),
            ],
        )
        arguments = self.spawn.call_args.args[0]
        for flag in (
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ):
            self.assertIn(flag, arguments)

    def test_expired_devtools_readiness_cleans_up_exact_owned_process(self):
        process, _, _ = self.start_page()
        self.endpoint.side_effect = OSError("not listening")
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=[0, 0, 11]),
            self.assertRaisesRegex(TimeoutError, "DevTools did not start"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture"),
        ):
            self.fail("unready Chrome must not yield a page")
        self.cleanup.assert_called_once_with(process, websocket=None, close_browser=None, owned_port=9222)

    def test_websocket_failure_still_cleans_up_owned_process(self):
        process, _, _ = self.start_page()
        self.connect.side_effect = OSError("connect failed")
        with (
            self.assertRaisesRegex(OSError, "connect failed"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture"),
        ):
            self.fail("failed connection must not yield a page")
        self.cleanup.assert_called_once_with(process, websocket=None, close_browser=None, owned_port=9222)

    def test_devtools_startup_can_use_remaining_fixture_deadline_after_ten_seconds(self):
        _, websocket, page = self.start_page()
        clock = [0.0]

        def endpoint(*_args, **kwargs):
            self.assertLessEqual(kwargs["timeout"], 1)
            if clock[0] == 0:
                clock[0] = 11
                raise OSError("browser still starting")
            return BytesIO(b'[{"type":"page","webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/page/fixture"}]')

        self.endpoint.side_effect = endpoint
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(chromium_helpers.time, "sleep"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture", deadline=30) as opened,
        ):
            self.assertIs(opened, page)
        self.assertEqual(self.endpoint.call_count, 2)
        self.assertIs(self.cleanup.call_args.kwargs["websocket"], websocket)

    def test_devtools_response_after_fixture_deadline_is_rejected_and_cleaned_up(self):
        process, _, _ = self.start_page()
        clock = [0.0]

        def late_endpoint(*_args, **_kwargs):
            clock[0] = 31
            return BytesIO(b'[{"type":"page","webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/page/fixture"}]')

        self.endpoint.side_effect = late_endpoint
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "DevTools did not start"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture", deadline=30),
        ):
            self.fail("late readiness must not yield a page")
        self.connect.assert_not_called()
        self.cleanup.assert_called_once_with(process, websocket=None, close_browser=None, owned_port=9222)

    def test_devtools_request_uses_only_remaining_startup_budget(self):
        process, _, _ = self.start_page()
        clock = [29.75]
        timeouts = []

        def unavailable(*_args, **kwargs):
            timeouts.append(kwargs["timeout"])
            clock[0] = 30
            raise OSError("not ready")

        self.endpoint.side_effect = unavailable
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "DevTools did not start"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture", deadline=30),
        ):
            self.fail("unready Chrome must not yield a page")
        self.assertEqual(timeouts, [0.25])
        self.cleanup.assert_called_once_with(process, websocket=None, close_browser=None, owned_port=9222)

    def test_navigation_error_closes_socket_and_owned_tree(self):
        process, websocket, page = self.start_page()
        page.send.side_effect = [{}, {}, {}, {"errorText": "net::ERR_CONNECTION_REFUSED"}]
        with (
            self.assertRaisesRegex(RuntimeError, "ERR_CONNECTION_REFUSED"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture"),
        ):
            self.fail("failed navigation must not yield a page")
        self.assertEqual(self.cleanup.call_args.args, (process,))
        self.assertIs(self.cleanup.call_args.kwargs["websocket"], websocket)
        self.assertEqual(self.cleanup.call_args.kwargs["owned_port"], 9222)
        self.assertTrue(callable(self.cleanup.call_args.kwargs["close_browser"]))

    def test_owned_job_is_closed_even_after_the_browser_root_exits(self):
        process = FakeProcess()
        process.returncode = 0
        process.close_tree = Mock()
        with (
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=True) as port,
            patch.object(chromium_helpers.subprocess, "run") as taskkill,
        ):
            chromium_helpers._stop_process_tree(process, owned_port=9222)
        process.close_tree.assert_called_once_with(timeout=5)
        port.assert_called_once_with(9222)
        taskkill.assert_not_called()

    def test_owned_job_snapshots_live_children_without_sending_browser_close(self):
        process = FakeProcess()
        process.close_tree = Mock()
        websocket = FakeWebSocket()
        close_browser = Mock()
        with patch.object(chromium_helpers, "_wait_for_port_release", return_value=True):
            chromium_helpers.close_chromium(
                process,
                websocket=websocket,
                close_browser=close_browser,
                owned_port=9222,
                profile_flush_delay=0,
            )
        close_browser.assert_not_called()
        process.close_tree.assert_called_once_with(timeout=5)
        self.assertTrue(websocket.closed)

    def test_fixture_exception_keeps_failure_and_cleans_up_before_profile_removal(self):
        process, websocket, _ = self.start_page()
        with (
            self.assertRaisesRegex(ValueError, "fixture failed"),
            chromium_helpers.cdp_page("chrome", "http://127.0.0.1/fixture", reduced_motion=True),
        ):
            raise ValueError("fixture failed")
        arguments = self.spawn.call_args.args[0]
        self.assertIn("--force-prefers-reduced-motion", arguments)
        self.assertNotIn("--dump-dom", arguments)
        self.assertNotIn("--virtual-time-budget=7000", arguments)
        self.assertEqual(sum(arg.startswith("--user-data-dir=") for arg in arguments), 1)
        self.assertEqual(self.cleanup.call_args.args, (process,))
        self.assertIs(self.cleanup.call_args.kwargs["websocket"], websocket)


class ChromiumDomCompletionTests(unittest.TestCase):
    def test_dom_wait_polls_without_page_timer_or_transport_deadline_collision(self):
        clock = [0.0]
        page = chromium_helpers.CdpPage(Mock())

        def pending(expression, *, timeout):
            self.assertNotIn("new Promise", expression)
            self.assertNotIn("setTimeout", expression)
            self.assertIn("document.querySelector", expression)
            self.assertLessEqual(timeout, 0.05 - clock[0])
            return False

        def advance(duration):
            clock[0] += duration

        with (
            patch.object(page, "evaluate", side_effect=pending) as evaluate,
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(chromium_helpers.time, "sleep", side_effect=advance),
        ):
            self.assertFalse(page.wait_for("#generate", "true", 50))
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(clock[0], 0.05)

    def test_dom_wait_ready_after_deadline_is_not_accepted(self):
        clock = [0.0]
        page = chromium_helpers.CdpPage(Mock())

        def late(*_args, **_kwargs):
            clock[0] = 0.051
            return True

        with (
            patch.object(page, "evaluate", side_effect=late),
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
        ):
            self.assertFalse(page.wait_for("#generate", "true", 50))

    def test_dom_wait_allows_response_within_remaining_dom_budget(self):
        clock = [0.0]
        page = chromium_helpers.CdpPage(Mock())

        def slow_response(*_args, timeout):
            if timeout < 6:
                raise TimeoutError("transport expired before the DOM deadline")
            clock[0] += 6
            return True

        with (
            patch.object(page, "evaluate", side_effect=slow_response),
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
        ):
            self.assertTrue(page.wait_for("#generate", "true", 15_000))

    def fixture_page(self, page):
        @contextmanager
        def opened(*_args, **_kwargs):
            try:
                yield page
            finally:
                self.cleaned = True

        self.cleaned = False
        return self.enterContext(patch.object(chromium_helpers, "cdp_page", side_effect=opened))

    def test_dom_snapshot_uses_remaining_fixture_budget(self):
        clock = [100.0]
        page = Mock(exceptions=[], console_errors=[])

        def slow_snapshot(*_args, timeout):
            if timeout < 6:
                raise TimeoutError("transport expired before the fixture deadline")
            clock[0] += 6
            return {"ready": True, "dom": "complete"}

        page.evaluate.side_effect = slow_snapshot
        self.fixture_page(page)
        with patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertEqual(chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture"), "complete")

    def test_ready_dom_returns_without_waiting_for_browser_exit(self):
        page = Mock(exceptions=[], console_errors=[])
        page.evaluate.side_effect = [
            {"ready": False, "dom": '<pre id="result">pending</pre>'},
            {"ready": True, "dom": '<pre id="result" data-ready="true">{"ok":true}</pre>'},
        ]
        opened = self.fixture_page(page)
        with patch.object(chromium_helpers.time, "sleep"):
            document = chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture")
        self.assertIn('data-ready="true"', document)
        self.assertEqual(page.evaluate.call_count, 2)
        page.process.wait.assert_not_called()
        self.assertEqual(opened.call_args.kwargs["window_size"], "1280,900")
        self.assertTrue(self.cleaned)

    def test_json_without_completion_marker_times_out_with_last_dom_and_cleans_up(self):
        clock = [100.0]
        page = Mock(exceptions=["fixture error"], console_errors=["bad asset"])

        def pending(*_args, **_kwargs):
            clock[0] += 30
            return {"ready": False, "dom": '<pre id="result">{"ok":true}</pre>'}

        page.evaluate.side_effect = pending
        self.fixture_page(page)
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "DOM completion") as failure,
        ):
            chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture")
        notes = "\n".join(failure.exception.__notes__)
        self.assertIn('id=\\"result\\"', notes)
        self.assertIn("fixture error", notes)
        self.assertTrue(self.cleaned)

    def test_evaluation_failure_is_not_accepted_as_dom_completion(self):
        page = Mock(exceptions=[], console_errors=[])
        page.evaluate.side_effect = RuntimeError("JavaScript failed")
        self.fixture_page(page)
        with self.assertRaisesRegex(RuntimeError, "JavaScript failed"):
            chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture")
        self.assertTrue(self.cleaned)

    def test_startup_and_dom_completion_share_one_absolute_deadline(self):
        clock = [100.0]
        page = Mock(exceptions=[], console_errors=[])

        @contextmanager
        def slow_start(*_args, **kwargs):
            self.assertEqual(kwargs["deadline"], 130)
            clock[0] += 29
            yield page

        def pending(*_args, **kwargs):
            self.assertLessEqual(kwargs["timeout"], 1)
            clock[0] += 2
            return {"ready": False, "dom": "pending"}

        page.evaluate.side_effect = pending
        with (
            patch.object(chromium_helpers, "cdp_page", side_effect=slow_start),
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "DOM completion"),
        ):
            chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture")
        self.assertEqual(page.evaluate.call_count, 1)

    def test_ready_response_after_deadline_is_rejected(self):
        clock = [100.0]
        page = Mock(exceptions=[], console_errors=[])

        def late_completion(*_args, **_kwargs):
            clock[0] += 31
            return {"ready": True, "dom": '<pre id="result" data-ready="true">{"ok":true}</pre>'}

        page.evaluate.side_effect = late_completion
        self.fixture_page(page)
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "DOM completion"),
        ):
            chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture")
        self.assertTrue(self.cleaned)

    def test_failure_notes_redact_secret_and_personal_path_and_bound_output(self):
        clock = [100.0]
        page = Mock(exceptions=[], console_errors=[])

        def pending(*_args, **_kwargs):
            clock[0] += 30
            return {
                "ready": False,
                "dom": 'token=HARNESS_CANARY C:\\Users\\private\\input.png {"token":"QUOTED_CANARY"} ' + "x" * 6000,
            }

        page.evaluate.side_effect = pending
        self.fixture_page(page)
        with (
            patch.object(chromium_helpers.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaises(TimeoutError) as failure,
        ):
            chromium_helpers.render_chromium_dom("chrome", "http://127.0.0.1/fixture?token=URL_CANARY")
        notes = "\n".join(failure.exception.__notes__)
        self.assertNotIn("HARNESS_CANARY", notes)
        self.assertNotIn("QUOTED_CANARY", notes)
        self.assertNotIn("URL_CANARY", notes)
        self.assertNotIn("private", notes)
        self.assertIn("redacted", notes)
        self.assertLessEqual(len(notes), 4000)


class ChromiumHelperImportTests(unittest.TestCase):
    def test_helper_import_needs_no_browser_or_application_packages(self):
        root = str(Path(__file__).resolve().parents[2])
        script = (
            f"import sys; sys.path.insert(0, {root!r}); "
            "import tools.tests.chromium_helpers; "
            "assert not {'websockets', 'gradio', 'torch'}.intersection(sys.modules)"
        )
        completed = subprocess.run(  # noqa: S603 - current Python and fixed local import probe
            [sys.executable, "-I", "-S", "-c", script], capture_output=True, text=True, check=False, timeout=10
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class ChromiumMemoryTests(unittest.TestCase):
    def snapshot(self, root, children):
        root.children.return_value = children
        with patch.object(psutil, "Process", return_value=root):
            return chromium_helpers.chromium_memory_usage(root.pid)

    def test_complete_tree_keeps_rss_and_uss_separate(self):
        root, child = Mock(pid=1), Mock(pid=2)
        root.memory_full_info.return_value = SimpleNamespace(rss=10, uss=4)
        child.memory_full_info.return_value = SimpleNamespace(rss=20, uss=7)
        self.assertEqual(self.snapshot(root, [child]), (30, 11))

    def test_denied_uss_can_keep_rss_but_never_reports_partial_uss(self):
        root, child = Mock(pid=1), Mock(pid=2)
        root.memory_full_info.return_value = SimpleNamespace(rss=10, uss=4)
        child.memory_full_info.side_effect = psutil.AccessDenied(pid=2)
        child.memory_info.return_value = SimpleNamespace(rss=20)
        self.assertEqual(self.snapshot(root, [child]), (30, None))

    def test_denied_rss_never_reports_partial_tree_memory(self):
        root, child = Mock(pid=1), Mock(pid=2)
        root.memory_full_info.return_value = SimpleNamespace(rss=10, uss=4)
        child.memory_full_info.side_effect = psutil.AccessDenied(pid=2)
        child.memory_info.side_effect = psutil.AccessDenied(pid=2)
        self.assertEqual(self.snapshot(root, [child]), (None, None))

    def test_a_child_that_exited_is_no_longer_counted(self):
        root, child = Mock(pid=1), Mock(pid=2)
        root.memory_full_info.return_value = SimpleNamespace(rss=10, uss=4)
        child.memory_full_info.side_effect = psutil.NoSuchProcess(pid=2)
        self.assertEqual(self.snapshot(root, [child]), (10, 4))

    def test_denied_tree_enumeration_is_unavailable(self):
        root = Mock(pid=1)
        root.children.side_effect = psutil.AccessDenied(pid=1)
        with patch.object(psutil, "Process", return_value=root):
            self.assertEqual(chromium_helpers.chromium_memory_usage(1), (None, None))

    def test_unexpected_measurement_errors_are_not_hidden(self):
        root = Mock(pid=1)
        root.memory_full_info.side_effect = RuntimeError("unexpected")
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            self.snapshot(root, [])


class ChromiumCleanupTests(unittest.TestCase):
    def windows_taskkill(self, process: FakeProcess):
        def complete(*_args, **_kwargs):
            process.returncode = 0
            return subprocess.CompletedProcess([], 0)

        return patch.object(chromium_helpers.subprocess, "run", side_effect=complete)

    def test_port_release_probe_distinguishes_a_listener_from_a_closed_port(self):
        with chromium_helpers.socket.socket(
            chromium_helpers.socket.AF_INET,
            chromium_helpers.socket.SOCK_STREAM,
        ) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertFalse(chromium_helpers._wait_for_port_release(port, timeout=0))

        self.assertTrue(chromium_helpers._wait_for_port_release(port, timeout=0))

    def test_connection_closed_during_browser_close_still_closes_socket_and_process_tree(self):
        process = FakeProcess()
        websocket = FakeWebSocket()

        def close_browser():
            raise ConnectionClosedOK(None, None)

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            self.windows_taskkill(process) as taskkill,
        ):
            chromium_helpers.close_chromium(
                process,
                websocket=websocket,
                close_browser=close_browser,
                profile_flush_delay=0,
            )

        self.assertTrue(websocket.closed)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(taskkill.call_args.args[0][1:], ["/PID", "4242", "/T", "/F"])

    def test_unexpected_socket_close_error_does_not_skip_process_tree_cleanup(self):
        process = FakeProcess()
        websocket = FakeWebSocket(ValueError("unexpected close failure"))

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            self.windows_taskkill(process) as taskkill,
            self.assertRaisesRegex(ValueError, "unexpected close failure"),
        ):
            chromium_helpers.close_chromium(
                process,
                websocket=websocket,
                profile_flush_delay=0,
            )

        self.assertEqual(process.returncode, 0)
        taskkill.assert_called_once()

    def test_already_exited_process_needs_no_tree_termination(self):
        process = FakeProcess()
        process.returncode = 0

        with patch.object(chromium_helpers.subprocess, "run") as taskkill:
            chromium_helpers.close_chromium(process, profile_flush_delay=0)

        taskkill.assert_not_called()

    def test_failed_taskkill_is_reported_after_stopping_the_root_process(self):
        process = FakeProcess()

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            patch.object(
                chromium_helpers.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1),
            ),
            self.assertRaisesRegex(RuntimeError, "taskkill failed with exit code 1"),
        ):
            chromium_helpers.close_chromium(process, profile_flush_delay=0)

        self.assertTrue(process.killed)
        self.assertEqual(process.returncode, -1)

    def test_missing_taskkill_is_reported_after_stopping_the_root_process(self):
        process = FakeProcess()

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=None),
            self.assertRaisesRegex(RuntimeError, "taskkill is unavailable"),
        ):
            chromium_helpers.close_chromium(process, profile_flush_delay=0)

        self.assertTrue(process.terminated)
        self.assertEqual(process.returncode, -1)

    def test_nonzero_taskkill_needs_root_exit_and_owned_port_release(self):
        process = FakeProcess()

        def root_exits_during_taskkill(*_args, **_kwargs):
            process.returncode = 0
            return subprocess.CompletedProcess([], 128)

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            patch.object(chromium_helpers.subprocess, "run", side_effect=root_exits_during_taskkill),
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=False) as port_release,
            self.assertRaisesRegex(RuntimeError, "taskkill failed with exit code 128"),
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

        port_release.assert_called_once_with(9222)
        self.assertFalse(process.killed)

    def test_nonzero_taskkill_is_normalized_only_after_owned_port_release(self):
        process = FakeProcess()

        def root_exits_during_taskkill(*_args, **_kwargs):
            process.returncode = 0
            return subprocess.CompletedProcess([], 128)

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_find_windows_taskkill", return_value=r"C:\Windows\System32\taskkill.exe"),
            patch.object(chromium_helpers.subprocess, "run", side_effect=root_exits_during_taskkill),
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=True) as port_release,
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

        port_release.assert_called_once_with(9222)
        self.assertFalse(process.killed)

    def test_already_exited_root_still_requires_owned_port_release(self):
        process = FakeProcess()
        process.returncode = 0

        with (
            patch.object(chromium_helpers.os, "name", "nt"),
            patch.object(chromium_helpers, "_wait_for_port_release", return_value=False),
            self.assertRaisesRegex(RuntimeError, "CDP port is still in use"),
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

    def test_posix_exited_root_terminates_the_lingering_process_group(self):
        process = FakeProcess()
        process.returncode = 0

        with (
            patch.object(chromium_helpers.os, "name", "posix"),
            patch.object(chromium_helpers.signal, "SIGKILL", 9, create=True),
            patch.object(
                chromium_helpers,
                "_wait_for_port_release",
                side_effect=[False, True],
            ) as port_release,
            patch.object(chromium_helpers.os, "killpg", create=True) as killpg,
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

        self.assertEqual(port_release.call_args_list, [call(9222), call(9222)])
        killpg.assert_called_once_with(process.pid, chromium_helpers.signal.SIGTERM)

    def test_posix_root_exit_still_kills_a_child_that_keeps_the_port(self):
        process = FakeProcess()
        sigkill = 9

        def signal_group(_process_id, group_signal):
            if group_signal == chromium_helpers.signal.SIGTERM:
                process.returncode = 0

        with (
            patch.object(chromium_helpers.os, "name", "posix"),
            patch.object(chromium_helpers.signal, "SIGKILL", sigkill, create=True),
            patch.object(
                chromium_helpers,
                "_wait_for_port_release",
                side_effect=[False, True],
            ) as port_release,
            patch.object(
                chromium_helpers.os,
                "killpg",
                side_effect=signal_group,
                create=True,
            ) as killpg,
        ):
            chromium_helpers.close_chromium(
                process,
                owned_port=9222,
                profile_flush_delay=0,
            )

        self.assertEqual(port_release.call_args_list, [call(9222), call(9222)])
        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, chromium_helpers.signal.SIGTERM),
                call(process.pid, sigkill),
            ],
        )


if __name__ == "__main__":
    unittest.main()
