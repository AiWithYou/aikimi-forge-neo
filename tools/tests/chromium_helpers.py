"""Dependency-light process helpers shared by Chromium smoke tests."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.exceptions import ConnectionClosed


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
