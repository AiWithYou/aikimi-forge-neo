"""Keep Gradio's startup probe subject to certificate and hostname validation."""

import ssl
import time

import httpx


def gradio_url_ok_fix(url: str) -> bool:
    from modules import shared

    verify = True
    if shared.cmd_opts.tls_certfile:
        verify = ssl.create_default_context()
        verify.load_verify_locations(cafile=shared.cmd_opts.tls_certfile)
    try:
        for _ in range(5):
            response = httpx.head(url, timeout=999, verify=verify)
            if response.status_code in (200, 401, 302):
                return True
            time.sleep(0.500)
    except (ConnectionError, httpx.ConnectError, httpx.TimeoutException):
        return False
    return False
