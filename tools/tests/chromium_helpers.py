"""Dependency-light process helpers shared by Chromium smoke tests."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def find_chromium() -> str | None:
    candidates = [
        os.environ.get("CHROME_BIN"),
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


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def chromium_memory_usage(pid: int) -> tuple[int | None, int | None]:
    """Return total RSS/USS, using None for metrics with unreadable processes."""
    import psutil

    try:
        owned = psutil.Process(pid)
        processes = [owned, *owned.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None, None

    rss = private = 0
    for process in processes:
        try:
            memory = process.memory_full_info()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            private = None
            try:
                memory = process.memory_info()
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                rss = None
                continue
        if rss is not None:
            rss += memory.rss
        if private is not None:
            private += memory.uss
    return rss, private


def _wait_for_exit(process: subprocess.Popen[Any], timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _wait_for_port_release(port: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _find_windows_taskkill() -> str | None:
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidate = Path(system_root) / "System32" / "taskkill.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("taskkill.exe")


def _stop_process_tree(process: subprocess.Popen[Any], *, owned_port: int | None = None) -> None:
    if process.poll() is not None or _wait_for_exit(process, 0.5):
        if owned_port is None or _wait_for_port_release(owned_port):
            return
        if os.name == "nt":
            raise RuntimeError("Chromium root exited but its CDP port is still in use")

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if _wait_for_port_release(owned_port):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not _wait_for_port_release(owned_port):
            raise RuntimeError("Chromium root exited but its CDP process group still owns the port")
        return

    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        taskkill = _find_windows_taskkill()
        if taskkill is None:
            process.terminate()
            if _wait_for_exit(process, 5):
                raise RuntimeError("taskkill is unavailable; Chromium tree cleanup was not guaranteed")
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError("taskkill is unavailable; Chromium tree cleanup was not guaranteed")
        try:
            taskkill_result = subprocess.run(  # noqa: S603
                [taskkill, "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=creationflags,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            process.terminate()
            if not _wait_for_exit(process, 5):
                process.kill()
                process.wait(timeout=5)
            raise RuntimeError("taskkill could not run; Chromium tree cleanup was not guaranteed") from error
        else:
            if taskkill_result.returncode != 0:
                if process.poll() is None and not _wait_for_exit(process, 1):
                    process.kill()
                    process.wait(timeout=5)
                if owned_port is not None and _wait_for_port_release(owned_port):
                    return
                raise RuntimeError(f"taskkill failed with exit code {taskkill_result.returncode}")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if not _wait_for_exit(process, 5):
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)

    if owned_port is None or _wait_for_port_release(owned_port):
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if _wait_for_port_release(owned_port):
            return
    raise RuntimeError("Chromium exited but its CDP port is still in use")


def close_chromium(
    process: subprocess.Popen[Any],
    *,
    websocket: Any | None = None,
    close_browser: Callable[[], Any] | None = None,
    owned_port: int | None = None,
    profile_flush_delay: float = 0.25,
) -> None:
    """Close CDP, its socket, and the exact Chromium process tree independently."""

    from websockets.exceptions import ConnectionClosed

    try:
        if close_browser is not None:
            try:
                close_browser()
            except (ConnectionClosed, OSError, RuntimeError, TimeoutError):
                pass
    finally:
        try:
            if websocket is not None:
                try:
                    websocket.close()
                except (ConnectionClosed, OSError, RuntimeError, TimeoutError):
                    pass
        finally:
            _stop_process_tree(process, owned_port=owned_port)
            if profile_flush_delay > 0:
                time.sleep(profile_flush_delay)


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


def _remaining_timeout(deadline: float | None, maximum: float) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Chromium deadline elapsed")
    return min(maximum, remaining)


def _connect_websocket(*args, **kwargs):
    from websockets.sync.client import connect

    return connect(*args, **kwargs)


def _diagnostic_note(value: object, limit: int = 4000) -> str:
    from modules.aikimi_security.redaction import redact_mapping, redact_text

    text = redact_text(value) if isinstance(value, str) else json.dumps(redact_mapping(value), ensure_ascii=False)
    return text[:limit]


class CdpPage:
    def __init__(self, websocket, *, deadline: float | None = None):
        self.websocket = websocket
        self.deadline = deadline
        self.command_id = 0
        self.exceptions = []
        self.console_errors = []

    def send(self, method: str, params: dict | None = None, *, timeout: float = 10):
        timeout = _remaining_timeout(self.deadline, timeout)
        self.command_id += 1
        command_id = self.command_id
        self.websocket.send(json.dumps({"id": command_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CDP command timed out: {method}")
            message = json.loads(self.websocket.recv(timeout=remaining))
            if message.get("method") == "Runtime.exceptionThrown":
                detail = message.get("params", {}).get("exceptionDetails", {})
                self.exceptions.append(detail.get("exception", {}).get("description", detail.get("text")))
                self.exceptions = self.exceptions[-20:]
            if message.get("method") == "Runtime.consoleAPICalled" and message.get("params", {}).get("type") == "error":
                self.console_errors.append(message["params"].get("args", []))
                self.console_errors = self.console_errors[-20:]
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
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            raise RuntimeError(detail.get("exception", {}).get("description", detail.get("text")))
        return result["result"].get("value")


@contextmanager
def cdp_page(
    chromium: str,
    url: str,
    *,
    window_size: str = "900,800",
    reduced_motion: bool = False,
    deadline: float | None = None,
):
    debugging_port = reserve_local_port()
    with (
        tempfile.TemporaryDirectory(prefix="forge-test-chrome-") as profile,
        tempfile.TemporaryFile() as stderr,
    ):
        command = [
            chromium,
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            f"--window-size={window_size}",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={debugging_port}",
            "--remote-allow-origins=*",
            "about:blank",
        ]
        if reduced_motion:
            command.insert(-1, "--force-prefers-reduced-motion")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        websocket = None
        page = None
        try:
            startup_deadline = deadline if deadline is not None else time.monotonic() + 10
            while True:
                try:
                    request = Request(  # noqa: S310 - fixed loopback debugging endpoint
                        f"http://127.0.0.1:{debugging_port}/json/new?about:blank",
                        method="PUT",
                    )
                    with urlopen(request, timeout=_remaining_timeout(startup_deadline, 1)) as response:  # noqa: S310
                        target = json.load(response)
                    _remaining_timeout(startup_deadline, 1)
                    break
                except OSError as error:
                    if process.poll() is not None:
                        raise RuntimeError(f"Chrome exited before DevTools started: {process.returncode}") from error
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError("Chrome DevTools did not start") from error
                    time.sleep(_remaining_timeout(startup_deadline, 0.05))

            websocket = _connect_websocket(
                target["webSocketDebuggerUrl"],
                origin=f"http://127.0.0.1:{debugging_port}",
                open_timeout=_remaining_timeout(deadline, 5),
                close_timeout=2,
                max_size=None,
            )
            page = CdpPage(websocket, deadline=deadline)
            page.process = process
            page.send("Page.enable")
            page.send("Runtime.enable")
            # Windows CIでは新しいChromeプロセスの初回読込が10秒を超える場合がある。
            navigation = page.send("Page.navigate", {"url": url}, timeout=30)
            if navigation.get("errorText"):
                raise RuntimeError(f"Chrome navigation failed: {navigation['errorText']}")
            yield page
        except BaseException as error:
            stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, stderr.tell() - 2000))
            error.add_note(
                _diagnostic_note("Chrome stderr (tail): " + stderr.read().decode("utf-8", errors="replace"), 2000)
            )
            raise
        finally:
            # Browser.close must remain available after the fixture deadline.
            if page is not None:
                page.deadline = None
            close_chromium(
                process,
                websocket=websocket,
                close_browser=(lambda: page.send("Browser.close", timeout=2)) if page is not None else None,
                owned_port=debugging_port,
            )


def render_chromium_dom(
    chromium: str,
    url: str,
    *,
    completion: str = "document.querySelector('#result')?.dataset.ready === 'true'",
    window_size: str = "1280,900",
    reduced_motion: bool = False,
    timeout: float = 30,
) -> str:
    """Return a completed fixture DOM while its owned browser is still running."""
    deadline = time.monotonic() + timeout
    snapshot = None
    with cdp_page(
        chromium,
        url,
        window_size=window_size,
        reduced_motion=reduced_motion,
        deadline=deadline,
    ) as page:
        try:
            while True:
                snapshot = page.evaluate(
                    "(() => {const ready = Boolean(" + completion + ");"
                    "const dom = document.documentElement?.outerHTML || '';"
                    "return {ready, url: location.href, dom: ready ? dom : dom.slice(-2000)};})()",
                    timeout=_remaining_timeout(deadline, 5),
                )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Chromium DOM completion predicate timed out")
                if snapshot["ready"]:
                    return snapshot["dom"]
                time.sleep(_remaining_timeout(deadline, 0.05))
        except BaseException as error:
            error.add_note(
                _diagnostic_note(
                    {
                        "stage": "Chromium DOM completion",
                        "url": url,
                        "last_dom": snapshot,
                        "exceptions": page.exceptions,
                        "console": page.console_errors,
                    }
                )
            )
            raise
