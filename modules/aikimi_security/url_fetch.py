"""SSRF-resistant remote raster-image downloader.

The transport deliberately bypasses proxy environment variables and pins each
connection to an address that was included in the validated DNS result. Every
redirect is resolved and checked again before a new connection is opened.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit


class SafeFetchError(ValueError):
    """A client-safe remote image validation failure."""


@dataclass(frozen=True)
class FetchPolicy:
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 15.0
    max_redirects: int = 4
    max_response_bytes: int = 20 * 1024 * 1024
    max_decoded_pixels: int = 64 * 1024 * 1024
    chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("Remote image timeouts must be greater than zero.")
        if self.max_redirects < 0:
            raise ValueError("The remote image redirect limit cannot be negative.")
        if self.max_response_bytes <= 0 or self.max_decoded_pixels <= 0 or self.chunk_bytes <= 0:
            raise ValueError("Remote image size limits must be greater than zero.")


DEFAULT_POLICY = FetchPolicy()
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_ALLOWED_IMAGE_TYPES = {
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/heic",
    "image/heif",
    "image/jpeg",
    "image/jpg",
    "image/jxl",
    "image/png",
    "image/tiff",
    "image/webp",
}


class ResponseLike(Protocol):
    status: int

    def getheader(self, name: str, default: str | None = None) -> str | None: ...

    def read(self, amount: int | None = None) -> bytes: ...

    def close(self) -> None: ...


def _is_public_address(value: str) -> bool:
    if "%" in value:
        return False
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


def _normalize_host(value: str) -> str:
    """Normalize a URL hostname and reject local-name and scoped-IP forms."""

    candidate = value.strip()
    # A zone identifier makes an IPv6 literal interface-relative. Percent-encoded
    # zones are still visible as ``%25`` in ``SplitResult.hostname``.
    if not candidate or "%" in candidate:
        raise SafeFetchError("The remote image URL is blocked by the network policy.")
    try:
        candidate = candidate.encode("idna").decode("ascii").rstrip(".").casefold()
    except UnicodeError as exc:
        raise SafeFetchError("The remote image URL is invalid.") from exc
    if not candidate or len(candidate) > 253:
        raise SafeFetchError("The remote image URL is invalid.")
    if candidate == "localhost" or candidate.endswith(".localhost"):
        raise SafeFetchError("The remote image URL is blocked by the network policy.")
    return candidate


def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve a host and fail when any answer is not globally routable."""

    try:
        answers = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except (OSError, UnicodeError) as exc:
        raise SafeFetchError("The remote image host could not be resolved.") from exc

    addresses: list[str] = []
    for answer in answers:
        family = answer[0]
        if family not in {socket.AF_INET, socket.AF_INET6}:
            raise SafeFetchError("The remote image URL is blocked by the network policy.")
        if family == socket.AF_INET6 and len(answer[4]) >= 4 and answer[4][3]:
            raise SafeFetchError("The remote image URL is blocked by the network policy.")
        address = answer[4][0]
        if address not in addresses:
            addresses.append(address)
    if not addresses or any(not _is_public_address(value) for value in addresses):
        raise SafeFetchError("The remote image URL is blocked by the network policy.")
    return tuple(addresses)


@dataclass(frozen=True)
class ValidatedTarget:
    url: str
    scheme: str
    host: str
    port: int
    request_target: str
    connect_ip: str


def validate_remote_url(url: str) -> ValidatedTarget:
    """Validate syntax, userinfo, DNS, and all resolved IP classifications."""

    if not isinstance(url, str) or len(url) > 4096:
        raise SafeFetchError("The remote image URL is invalid.")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise SafeFetchError("The remote image URL is invalid.") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise SafeFetchError("Only http and https remote images are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise SafeFetchError("Credentials are not allowed in a remote image URL.")
    if not parsed.hostname:
        raise SafeFetchError("The remote image URL is invalid.")
    try:
        host = _normalize_host(parsed.hostname)
        port = parsed.port or (443 if scheme == "https" else 80)
    except SafeFetchError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise SafeFetchError("The remote image URL is invalid.") from exc
    if not 1 <= port <= 65535:
        raise SafeFetchError("The remote image URL uses an invalid port.")
    addresses = resolve_public_addresses(host, port)
    path = parsed.path or "/"
    request_target = f"{path}?{parsed.query}" if parsed.query else path
    display_host = f"[{host}]" if ":" in host else host
    explicit_port = f":{port}" if parsed.port is not None else ""
    normalized_url = urlunsplit((scheme, f"{display_host}{explicit_port}", path, parsed.query, ""))
    return ValidatedTarget(
        url=normalized_url,
        scheme=scheme,
        host=host,
        port=port,
        request_target=request_target,
        connect_ip=addresses[0],
    )


def _connection_for(target: ValidatedTarget, timeout: float) -> http.client.HTTPConnection:
    connection_type: type[http.client.HTTPConnection]
    kwargs: dict[str, object] = {"timeout": timeout}
    if target.scheme == "https":
        connection_type = http.client.HTTPSConnection
        kwargs["context"] = ssl.create_default_context()
    else:
        connection_type = http.client.HTTPConnection
    connection = connection_type(target.host, target.port, **kwargs)

    def create_connection(_address, timeout_value=None, source_address=None):
        return socket.create_connection(
            (target.connect_ip, target.port),
            timeout=timeout_value,
            source_address=source_address,
        )

    # HTTPConnection and HTTPSConnection both call this hook. Keeping ``host``
    # unchanged preserves the HTTP Host header and HTTPS SNI/certificate check.
    connection._create_connection = create_connection  # type: ignore[method-assign]
    return connection


def _request_once(
    target: ValidatedTarget, headers: dict[str, str], policy: FetchPolicy
) -> tuple[http.client.HTTPConnection, ResponseLike]:
    connection = _connection_for(target, policy.connect_timeout_seconds)
    try:
        connection.request("GET", target.request_target, headers=headers)
        response = connection.getresponse()
        if connection.sock is not None:
            connection.sock.settimeout(policy.read_timeout_seconds)
        return connection, response
    except (
        OSError,
        ValueError,
        UnicodeError,
        http.client.HTTPException,
        ssl.SSLError,
    ) as exc:
        connection.close()
        raise SafeFetchError("The remote image could not be downloaded safely.") from exc


def _read_bounded(response: ResponseLike, policy: FetchPolicy) -> bytes:
    content_length = response.getheader("Content-Length")
    if content_length:
        try:
            expected = int(content_length)
        except ValueError as exc:
            raise SafeFetchError("The remote server returned an invalid response.") from exc
        if expected < 0 or expected > policy.max_response_bytes:
            raise SafeFetchError("The remote image exceeds the download size limit.")

    payload = bytearray()
    while True:
        try:
            chunk = response.read(min(policy.chunk_bytes, policy.max_response_bytes + 1 - len(payload)))
        except (OSError, http.client.HTTPException) as exc:
            raise SafeFetchError("The remote image download was interrupted.") from exc
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > policy.max_response_bytes:
            raise SafeFetchError("The remote image exceeds the download size limit.")
    return bytes(payload)


def validate_image_payload(payload: bytes, policy: FetchPolicy = DEFAULT_POLICY) -> None:
    """Check raster dimensions and decode integrity before application decoding."""

    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            frames = int(getattr(image, "n_frames", 1) or 1)
            if (
                width <= 0
                or height <= 0
                or width * height > policy.max_decoded_pixels
                or width * height * frames > policy.max_decoded_pixels
            ):
                raise SafeFetchError("The remote image exceeds the decoded size limit.")
            image.verify()
    except SafeFetchError:
        raise
    except Exception as exc:
        raise SafeFetchError("The remote resource is not a valid raster image.") from exc


def validate_decoded_image(image, policy: FetchPolicy = DEFAULT_POLICY) -> None:
    """Re-check dimensions after the application's final decoder has run."""

    try:
        width, height = image.size
        frames = int(getattr(image, "n_frames", 1) or 1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SafeFetchError("The decoded remote image is invalid.") from exc
    if (
        width <= 0
        or height <= 0
        or width * height > policy.max_decoded_pixels
        or width * height * frames > policy.max_decoded_pixels
    ):
        raise SafeFetchError("The remote image exceeds the decoded size limit.")


def fetch_remote_image(
    url: str,
    *,
    user_agent: str = "",
    policy: FetchPolicy = DEFAULT_POLICY,
) -> bytes:
    """Fetch a validated raster image without proxies or credential forwarding."""

    normalized_user_agent = str(user_agent or "").strip()
    try:
        user_agent_is_safe = (
            bool(normalized_user_agent)
            and len(normalized_user_agent) <= 256
            and "\r" not in normalized_user_agent
            and "\n" not in normalized_user_agent
            and normalized_user_agent.encode("latin-1") is not None
        )
    except UnicodeEncodeError:
        user_agent_is_safe = False

    headers = {
        "Accept": "image/avif,image/webp,image/png,image/jpeg,image/gif,image/*;q=0.8",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": (normalized_user_agent if user_agent_is_safe else "Aikimi-Neo-safe-image-fetch/1"),
    }
    current = url
    visited: set[tuple[str, str, int, str]] = set()
    for redirect_count in range(policy.max_redirects + 1):
        target = validate_remote_url(current)
        normalized = (
            target.scheme,
            target.host,
            target.port,
            target.request_target,
        )
        if normalized in visited:
            raise SafeFetchError("The remote image redirect loop was blocked.")
        visited.add(normalized)

        connection, response = _request_once(target, headers, policy)
        try:
            if response.status in _REDIRECT_STATUSES:
                location = response.getheader("Location")
                if not location or redirect_count >= policy.max_redirects:
                    raise SafeFetchError("The remote image redirected too many times.")
                current = urljoin(target.url, location)
                continue
            if not 200 <= response.status < 300:
                raise SafeFetchError("The remote image server rejected the request.")
            content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().casefold()
            if content_type not in _ALLOWED_IMAGE_TYPES:
                raise SafeFetchError("The remote resource is not an allowed image type.")
            payload = _read_bounded(response, policy)
            validate_image_payload(payload, policy)
            return payload
        finally:
            response.close()
            connection.close()
    raise SafeFetchError("The remote image redirected too many times.")
