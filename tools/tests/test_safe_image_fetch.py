import os
import socket
import unittest
from io import BytesIO
from unittest import mock

from PIL import Image

from modules.aikimi_security import url_fetch

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:4700:4700::1111"


def _answer(address: str, port: int = 80):
    if ":" in address:
        return (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, port, 0, 0),
        )
    return (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        (address, port),
    )


def _png_bytes(size=(2, 2)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, "white").save(stream, format="PNG")
    return stream.getvalue()


class _FakeResponse:
    def __init__(self, status=200, headers=None, payload=b""):
        self.status = status
        self.headers = {str(key).casefold(): value for key, value in (headers or {}).items()}
        self.payload = payload
        self.offset = 0
        self.closed = False

    def getheader(self, name, default=None):
        return self.headers.get(name.casefold(), default)

    def read(self, amount=None):
        if amount is None:
            amount = len(self.payload) - self.offset
        result = self.payload[self.offset : self.offset + amount]
        self.offset += len(result)
        return result

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class SafeImageFetchTests(unittest.TestCase):
    def test_policy_rejects_non_positive_limits(self):
        with self.assertRaises(ValueError):
            url_fetch.FetchPolicy(max_response_bytes=0)
        with self.assertRaises(ValueError):
            url_fetch.FetchPolicy(max_redirects=-1)
        with self.assertRaises(ValueError):
            url_fetch.FetchPolicy(read_timeout_seconds=0)

    def test_only_http_and_https_are_accepted(self):
        for value in ("file:///etc/passwd", "ftp://example.com/a.png", "data:image/png,x"):
            with self.subTest(value=value), self.assertRaises(url_fetch.SafeFetchError):
                url_fetch.validate_remote_url(value)

    def test_userinfo_is_rejected_before_dns(self):
        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo") as resolver,
            self.assertRaises(url_fetch.SafeFetchError),
        ):
            url_fetch.validate_remote_url("https://user:password@example.test/a.png")
        resolver.assert_not_called()

    def test_localhost_names_and_trailing_dot_are_rejected_before_dns(self):
        values = (
            "http://localhost/a.png",
            "http://localhost./a.png",
            "http://images.localhost/a.png",
            "http://images.localhost./a.png",
        )
        for value in values:
            with self.subTest(value=value):
                with (
                    mock.patch.object(url_fetch.socket, "getaddrinfo") as resolver,
                    self.assertRaises(url_fetch.SafeFetchError),
                ):
                    url_fetch.validate_remote_url(value)
                resolver.assert_not_called()

    def test_ipv6_zone_identifier_is_rejected_before_dns(self):
        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo") as resolver,
            self.assertRaises(url_fetch.SafeFetchError),
        ):
            url_fetch.validate_remote_url("http://[fe80::1%25Ethernet]/a.png")
        resolver.assert_not_called()

    def test_non_public_address_classes_are_rejected(self):
        blocked = (
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.1",
            "169.254.1.1",
            "224.0.0.1",
            "0.0.0.0",  # noqa: S104 - policy rejection fixture
            "::1",
            "fe80::1",
            "ff02::1",
            "::",
            "::ffff:127.0.0.1",
        )
        for address in blocked:
            with self.subTest(address=address):
                answer = _answer(address)
                with (
                    mock.patch.object(url_fetch.socket, "getaddrinfo", return_value=[answer]),
                    self.assertRaises(url_fetch.SafeFetchError),
                ):
                    url_fetch.validate_remote_url("http://blocked.example/a.png")

    def test_every_dns_answer_must_be_public(self):
        answers = [_answer(PUBLIC_V4), _answer("192.168.1.10")]
        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo", return_value=answers),
            self.assertRaises(url_fetch.SafeFetchError),
        ):
            url_fetch.validate_remote_url("https://mixed.example/a.png")

    def test_dns_failure_is_client_safe(self):
        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo", side_effect=socket.gaierror("private detail")),
            self.assertRaisesRegex(url_fetch.SafeFetchError, "could not be resolved"),
        ):
            url_fetch.validate_remote_url("https://missing.example/a.png")

    def test_public_ipv4_ipv6_and_explicit_port_are_allowed(self):
        cases = (
            ("https://public.example:8443/a.png", PUBLIC_V4, 8443),
            ("https://[2606:4700:4700::1111]/a.png", PUBLIC_V6, 443),
        )
        for value, address, expected_port in cases:
            with self.subTest(value=value):
                with mock.patch.object(
                    url_fetch.socket,
                    "getaddrinfo",
                    return_value=[_answer(address, expected_port)],
                ):
                    target = url_fetch.validate_remote_url(value)
                self.assertEqual(target.connect_ip, address)
                self.assertEqual(target.port, expected_port)

    def test_connection_is_pinned_to_validated_ip_and_ignores_proxy_environment(self):
        target = url_fetch.ValidatedTarget(
            url="http://origin.example:8080/a.png",
            scheme="http",
            host="origin.example",
            port=8080,
            request_target="/a.png",
            connect_ip=PUBLIC_V4,
        )
        fake_socket = mock.Mock()
        with (
            mock.patch.dict(
                os.environ,
                {"HTTP_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9"},
            ),
            mock.patch.object(url_fetch.socket, "create_connection", return_value=fake_socket) as create_connection,
        ):
            connection = url_fetch._connection_for(target, 2.5)
            connection.connect()

        self.assertEqual(connection.host, "origin.example")
        create_connection.assert_called_once_with((PUBLIC_V4, 8080), timeout=2.5, source_address=None)

    def test_https_connection_pins_ip_but_keeps_original_sni_hostname(self):
        target = url_fetch.ValidatedTarget(
            url="https://origin.example/a.png",
            scheme="https",
            host="origin.example",
            port=443,
            request_target="/a.png",
            connect_ip=PUBLIC_V4,
        )
        raw_socket = mock.Mock()
        tls_socket = mock.Mock()
        context = mock.Mock()
        context.wrap_socket.return_value = tls_socket
        with (
            mock.patch.object(url_fetch.ssl, "create_default_context", return_value=context),
            mock.patch.object(url_fetch.socket, "create_connection", return_value=raw_socket) as create_connection,
        ):
            connection = url_fetch._connection_for(target, 3.0)
            connection.connect()

        create_connection.assert_called_once_with((PUBLIC_V4, 443), timeout=3.0, source_address=None)
        context.wrap_socket.assert_called_once_with(
            raw_socket,
            server_hostname="origin.example",
        )
        self.assertIs(connection.sock, tls_socket)

    def test_public_to_private_redirect_is_rejected_before_second_request(self):
        response = _FakeResponse(status=302, headers={"Location": "http://private.example/secret.png"})
        request_count = 0

        def resolver(host, port, **_kwargs):
            address = PUBLIC_V4 if host == "public.example" else "127.0.0.1"
            return [_answer(address, port)]

        def request_once(*_args):
            nonlocal request_count
            request_count += 1
            return _FakeConnection(), response

        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo", side_effect=resolver),
            mock.patch.object(url_fetch, "_request_once", side_effect=request_once),
            self.assertRaises(url_fetch.SafeFetchError),
        ):
            url_fetch.fetch_remote_image("https://public.example/start.png")

        self.assertEqual(request_count, 1)
        self.assertTrue(response.closed)

    def test_relative_redirect_is_revalidated_and_image_is_returned(self):
        payload = _png_bytes()
        responses = [
            _FakeResponse(status=302, headers={"Location": "/final.png"}),
            _FakeResponse(
                headers={
                    "Content-Type": "image/png; charset=binary",
                    "Content-Length": str(len(payload)),
                },
                payload=payload,
            ),
        ]
        targets = []

        def request_once(target, _headers, _policy):
            targets.append(target.request_target)
            return _FakeConnection(), responses.pop(0)

        with (
            mock.patch.object(
                url_fetch.socket,
                "getaddrinfo",
                return_value=[_answer(PUBLIC_V4, 443)],
            ),
            mock.patch.object(url_fetch, "_request_once", side_effect=request_once),
        ):
            result = url_fetch.fetch_remote_image("https://public.example/start.png")

        self.assertEqual(result, payload)
        self.assertEqual(targets, ["/start.png", "/final.png"])

    def test_redirect_limit_is_enforced(self):
        responses = [_FakeResponse(status=302, headers={"Location": f"/next-{index}.png"}) for index in range(3)]

        def request_once(_target, _headers, _policy):
            return _FakeConnection(), responses.pop(0)

        with (
            mock.patch.object(
                url_fetch.socket,
                "getaddrinfo",
                return_value=[_answer(PUBLIC_V4)],
            ),
            mock.patch.object(url_fetch, "_request_once", side_effect=request_once),
            self.assertRaisesRegex(url_fetch.SafeFetchError, "too many times"),
        ):
            url_fetch.fetch_remote_image(
                "http://public.example/start.png",
                policy=url_fetch.FetchPolicy(max_redirects=2),
            )

    def test_transport_timeout_is_reported_without_internal_detail(self):
        target = url_fetch.ValidatedTarget(
            url="http://origin.example/a.png",
            scheme="http",
            host="origin.example",
            port=80,
            request_target="/a.png",
            connect_ip=PUBLIC_V4,
        )
        connection = mock.Mock()
        connection.request.side_effect = TimeoutError("internal socket detail")
        with (
            mock.patch.object(url_fetch, "_connection_for", return_value=connection),
            self.assertRaises(url_fetch.SafeFetchError) as raised,
        ):
            url_fetch._request_once(target, {}, url_fetch.DEFAULT_POLICY)
        self.assertNotIn("internal socket detail", str(raised.exception))
        connection.close.assert_called_once()

    def test_declared_and_streamed_oversized_responses_are_rejected(self):
        cases = (
            _FakeResponse(
                headers={"Content-Type": "image/png", "Content-Length": "11"},
                payload=b"x" * 11,
            ),
            _FakeResponse(headers={"Content-Type": "image/png"}, payload=b"x" * 11),
        )
        policy = url_fetch.FetchPolicy(max_response_bytes=10, chunk_bytes=4)
        for response in cases:
            with self.subTest(headers=response.headers):
                with (
                    mock.patch.object(
                        url_fetch.socket,
                        "getaddrinfo",
                        return_value=[_answer(PUBLIC_V4)],
                    ),
                    mock.patch.object(
                        url_fetch,
                        "_request_once",
                        return_value=(_FakeConnection(), response),
                    ),
                    self.assertRaisesRegex(url_fetch.SafeFetchError, "size limit"),
                ):
                    url_fetch.fetch_remote_image("http://public.example/a.png", policy=policy)

    def test_non_image_content_type_is_rejected_before_body_read(self):
        response = _FakeResponse(headers={"Content-Type": "text/html"}, payload=b"not an image")
        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo", return_value=[_answer(PUBLIC_V4)]),
            mock.patch.object(
                url_fetch,
                "_request_once",
                return_value=(_FakeConnection(), response),
            ),
            self.assertRaisesRegex(url_fetch.SafeFetchError, "allowed image type"),
        ):
            url_fetch.fetch_remote_image("http://public.example/a.png")
        self.assertEqual(response.offset, 0)

    def test_invalid_image_and_decoded_pixel_limit_are_rejected(self):
        with self.assertRaisesRegex(url_fetch.SafeFetchError, "valid raster image"):
            url_fetch.validate_image_payload(b"not an image")

        with self.assertRaisesRegex(url_fetch.SafeFetchError, "decoded size limit"):
            url_fetch.validate_image_payload(
                _png_bytes((2, 2)),
                url_fetch.FetchPolicy(max_decoded_pixels=3),
            )

    def test_unsafe_user_agent_falls_back_to_fixed_value(self):
        payload = _png_bytes()
        response = _FakeResponse(headers={"Content-Type": "image/png"}, payload=payload)
        captured_headers = {}

        def request_once(_target, headers, _policy):
            captured_headers.update(headers)
            return _FakeConnection(), response

        with (
            mock.patch.object(url_fetch.socket, "getaddrinfo", return_value=[_answer(PUBLIC_V4)]),
            mock.patch.object(url_fetch, "_request_once", side_effect=request_once),
        ):
            url_fetch.fetch_remote_image(
                "http://public.example/a.png",
                user_agent="unsafe\r\nAuthorization: secret",
            )

        self.assertEqual(captured_headers["User-Agent"], "Aikimi-Neo-safe-image-fetch/1")


if __name__ == "__main__":
    unittest.main()
