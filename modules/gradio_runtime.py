"""Lifecycle helpers for Gradio resources that Blocks.close() does not own."""

from __future__ import annotations

import subprocess
import warnings
from collections.abc import Iterable
from typing import Any

from gradio.tunneling import CURRENT_TUNNELS


def tunnel_snapshot() -> frozenset[int]:
    return frozenset(id(tunnel) for tunnel in CURRENT_TUNNELS)


def close_gradio_runtime(demo: Any, existing_tunnel_ids: Iterable[int] = ()) -> None:
    """Close one Blocks server and only the share tunnels created for its launch."""

    cleanup_errors: list[Exception] = []
    try:
        demo.close()
    except Exception as error:  # pragma: no cover - defensive cleanup boundary
        cleanup_errors.append(error)

    baseline = set(existing_tunnel_ids)
    server_port = getattr(demo, "server_port", None)
    owned_tunnels = [
        tunnel
        for tunnel in list(CURRENT_TUNNELS)
        if id(tunnel) not in baseline and (server_port is None or getattr(tunnel, "local_port", None) == server_port)
    ]
    for tunnel in owned_tunnels:
        process = getattr(tunnel, "proc", None)
        try:
            tunnel.kill()
        except Exception as error:  # pragma: no cover - third-party cleanup boundary
            cleanup_errors.append(error)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as error:  # pragma: no cover
                    cleanup_errors.append(error)
            except OSError as error:  # pragma: no cover
                cleanup_errors.append(error)

    if owned_tunnels:
        owned_ids = {id(tunnel) for tunnel in owned_tunnels}
        CURRENT_TUNNELS[:] = [tunnel for tunnel in CURRENT_TUNNELS if id(tunnel) not in owned_ids]
    if cleanup_errors:
        names = ", ".join(type(error).__name__ for error in cleanup_errors)
        warnings.warn(f"Gradio cleanup completed with errors: {names}", RuntimeWarning, stacklevel=2)
