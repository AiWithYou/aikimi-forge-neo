from __future__ import annotations

import hashlib
import io
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.aikimi_setup import (
    ArtifactSpec,
    DiskSpaceError,
    DownloadError,
    Installer,
    IntegrityError,
    ManifestError,
    ProfileSpec,
    SetupError,
)

FIXTURE_BYTES = (b"Aikimi Neo installer fixture\n" * 257) + b"complete"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_BYTES).hexdigest()


class FixtureHandler(BaseHTTPRequestHandler):
    payload = FIXTURE_BYTES
    ignore_range = False
    wrong_content_range = False
    truncate_once = False
    ranges: list[str | None] = []

    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        type(self).ranges.append(range_header)
        if range_header and not type(self).ignore_range:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            body = type(self).payload[offset:]
            self.send_response(206)
            start = offset + 1 if type(self).wrong_content_range else offset
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(type(self).payload) - 1}/{len(type(self).payload)}",
            )
        else:
            body = type(self).payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).truncate_once and len(type(self).ranges) == 1:
            self.wfile.write(body[: len(body) // 2])
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def fixture_server(
    *,
    ignore_range: bool = False,
    wrong_content_range: bool = False,
    truncate_once: bool = False,
):
    handler = type("ConfiguredFixtureHandler", (FixtureHandler,), {})
    handler.ignore_range = ignore_range
    handler.wrong_content_range = wrong_content_range
    handler.truncate_once = truncate_once
    handler.ranges = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fixture_profile(url: str, *, sha256: str = FIXTURE_SHA256) -> ProfileSpec:
    artifact = ArtifactSpec(
        artifact_id="fixture",
        relative_path="models/fixture/model.bin",
        url=url,
        size=len(FIXTURE_BYTES),
        sha256=sha256,
        license_url=url.replace("artifact.bin", "LICENSE"),
    )
    return ProfileSpec(
        name="fixture",
        description="tiny local fixture",
        artifacts=(artifact,),
        licenses=(artifact.license_url,),
        legacy_peak_bytes=len(FIXTURE_BYTES),
    )


def ample_disk(_path: Path) -> SimpleNamespace:
    return SimpleNamespace(free=1024**3)


class AikimiSetupTests(unittest.TestCase):
    def make_installer(
        self,
        root: Path,
        profile: ProfileSpec,
        *,
        disk_usage=ample_disk,
    ) -> Installer:
        return Installer(
            root,
            {profile.name: profile},
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            allow_test_http=True,
            disk_usage=disk_usage,
        )

    def test_manifest_rejects_parent_traversal(self) -> None:
        with fixture_server() as (server, _handler), self.subTest("traversal"):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            unsafe = ArtifactSpec(
                "unsafe",
                "models/../outside.bin",
                url,
                len(FIXTURE_BYTES),
                FIXTURE_SHA256,
                f"http://127.0.0.1:{server.server_port}/LICENSE",
            )
            profile = ProfileSpec("unsafe", "unsafe", (unsafe,), (unsafe.license_url,), len(FIXTURE_BYTES))
            with self.assertRaises(ManifestError):
                Installer(Path.cwd(), {"unsafe": profile}, allow_test_http=True)

    def test_manifest_rejects_query_tokens_without_echoing_them(self) -> None:
        sentinel = "super-secret-token"
        artifact = ArtifactSpec(
            "unsafe-url",
            "models/model.bin",
            f"http://127.0.0.1/artifact.bin?token={sentinel}",
            len(FIXTURE_BYTES),
            FIXTURE_SHA256,
            "http://127.0.0.1/LICENSE",
        )
        profile = ProfileSpec(
            "unsafe-url",
            "unsafe-url",
            (artifact,),
            (artifact.license_url,),
            len(FIXTURE_BYTES),
        )
        with self.assertRaises(ManifestError) as raised:
            Installer(Path.cwd(), {profile.name: profile}, allow_test_http=True)
        self.assertNotIn(sentinel, str(raised.exception))

    def test_install_downloads_to_partial_then_atomically_finalizes(self) -> None:
        with fixture_server() as (server, handler), self.subTest("fresh"):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                installer = self.make_installer(root, fixture_profile(url))
                result = installer.install("fixture", dry_run=False, keep_source=False)
                target = root / "models/fixture/model.bin"
                self.assertTrue(result["ok"])
                self.assertEqual(FIXTURE_BYTES, target.read_bytes())
                self.assertFalse(Path(f"{target}.part").exists())
                self.assertEqual([None], handler.ranges)

    def test_resume_requires_and_uses_the_expected_content_range(self) -> None:
        with fixture_server() as (server, handler), self.subTest("resume"):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                partial = root / "models/fixture/model.bin.part"
                partial.parent.mkdir(parents=True)
                partial.write_bytes(FIXTURE_BYTES[:31])
                installer = self.make_installer(root, fixture_profile(url))
                installer.install("fixture", dry_run=False, keep_source=False)
                self.assertEqual(["bytes=31-"], handler.ranges)
                self.assertEqual(FIXTURE_BYTES, (root / "models/fixture/model.bin").read_bytes())

    def test_server_ignoring_range_restarts_instead_of_appending(self) -> None:
        with fixture_server(ignore_range=True) as (server, handler), self.subTest("ignored-range"):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                partial = root / "models/fixture/model.bin.part"
                partial.parent.mkdir(parents=True)
                partial.write_bytes(FIXTURE_BYTES[:17])
                installer = self.make_installer(root, fixture_profile(url))
                installer.install("fixture", dry_run=False, keep_source=False)
                self.assertEqual(["bytes=17-"], handler.ranges)
                self.assertEqual(FIXTURE_BYTES, (root / "models/fixture/model.bin").read_bytes())

    def test_interrupted_response_is_resumed_on_the_next_attempt(self) -> None:
        with fixture_server(truncate_once=True) as (server, handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                installer = self.make_installer(root, fixture_profile(url))
                installer.install("fixture", dry_run=False, keep_source=False)
                self.assertEqual(2, len(handler.ranges))
                self.assertIsNone(handler.ranges[0])
                self.assertRegex(handler.ranges[1] or "", r"^bytes=\d+-$")
                self.assertEqual(FIXTURE_BYTES, (root / "models/fixture/model.bin").read_bytes())

    def test_wrong_content_range_is_rejected(self) -> None:
        with fixture_server(wrong_content_range=True) as (server, _handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                partial = root / "models/fixture/model.bin.part"
                partial.parent.mkdir(parents=True)
                partial.write_bytes(FIXTURE_BYTES[:9])
                installer = self.make_installer(root, fixture_profile(url))
                with self.assertRaises(DownloadError):
                    installer.install("fixture", dry_run=False, keep_source=False)
                self.assertFalse((root / "models/fixture/model.bin").exists())
                self.assertEqual(FIXTURE_BYTES[:9], partial.read_bytes())

    def test_hash_mismatch_never_creates_the_final_file(self) -> None:
        with fixture_server() as (server, _handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile = fixture_profile(url, sha256="0" * 64)
                installer = self.make_installer(root, profile)
                with self.assertRaises(IntegrityError):
                    installer.install("fixture", dry_run=False, keep_source=False)
                target = root / "models/fixture/model.bin"
                self.assertFalse(target.exists())
                self.assertEqual(FIXTURE_BYTES, Path(f"{target}.part").read_bytes())

    def test_disk_preflight_stops_before_network_or_writes(self) -> None:
        with fixture_server() as (server, handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                installer = self.make_installer(
                    root,
                    fixture_profile(url),
                    disk_usage=lambda _path: SimpleNamespace(free=1),
                )
                with self.assertRaises(DiskSpaceError):
                    installer.install("fixture", dry_run=False, keep_source=False)
                self.assertEqual([], handler.ranges)
                self.assertFalse((root / "models").exists())

    def test_concurrent_install_is_rejected_by_the_repository_lock(self) -> None:
        with fixture_server() as (server, handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first = self.make_installer(root, fixture_profile(url))
                second = self.make_installer(root, fixture_profile(url))
                with (
                    first._mutation_lock(),
                    self.assertRaisesRegex(  # noqa: SLF001
                        SetupError, "already running"
                    ),
                ):
                    second.install("fixture", dry_run=False, keep_source=False)
                self.assertEqual([], handler.ranges)

    def test_dry_run_does_not_write_or_contact_the_server(self) -> None:
        with fixture_server() as (server, handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                installer = self.make_installer(root, fixture_profile(url))
                report = installer.install("fixture", dry_run=True, keep_source=False)
                self.assertTrue(report["dry_run"])
                self.assertEqual([], handler.ranges)
                self.assertFalse((root / "models").exists())

    def test_dry_run_does_not_hash_an_existing_large_file(self) -> None:
        with fixture_server() as (server, handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "models/fixture/model.bin"
                target.parent.mkdir(parents=True)
                target.write_bytes(FIXTURE_BYTES)
                installer = self.make_installer(root, fixture_profile(url))
                with mock.patch(
                    "tools.aikimi_setup._sha256",
                    side_effect=AssertionError("dry-run must not hash"),
                ):
                    report = installer.install("fixture", dry_run=True, keep_source=False)
                self.assertEqual([], report["planned_paths"])
                self.assertEqual([], handler.ranges)

    def test_repair_quarantines_an_invalid_final_and_reinstalls(self) -> None:
        with fixture_server() as (server, _handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "models/fixture/model.bin"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"corrupt")
                installer = self.make_installer(root, fixture_profile(url))
                result = installer.repair("fixture", dry_run=False, keep_source=False)
                self.assertTrue(result["ok"])
                self.assertEqual(FIXTURE_BYTES, target.read_bytes())
                quarantined = list((root / "tmp/aikimi-setup/quarantine/fixture").glob("*-model.bin"))
                self.assertEqual(1, len(quarantined))
                self.assertEqual(b"corrupt", quarantined[0].read_bytes())

    def test_repair_quarantines_an_incomplete_partial_before_retry(self) -> None:
        with fixture_server() as (server, handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                partial = root / "models/fixture/model.bin.part"
                partial.parent.mkdir(parents=True)
                partial.write_bytes(FIXTURE_BYTES[:19])
                installer = self.make_installer(root, fixture_profile(url))
                installer.repair("fixture", dry_run=False, keep_source=False)
                self.assertEqual([None], handler.ranges)
                self.assertEqual(FIXTURE_BYTES, (root / "models/fixture/model.bin").read_bytes())
                quarantined = list((root / "tmp/aikimi-setup/quarantine/fixture").glob("*-model.bin.part"))
                self.assertEqual(1, len(quarantined))
                self.assertEqual(FIXTURE_BYTES[:19], quarantined[0].read_bytes())

    def test_verify_reports_ready_after_install(self) -> None:
        with fixture_server() as (server, _handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                installer = self.make_installer(root, fixture_profile(url))
                self.assertFalse(installer.verify(["fixture"])["ok"])
                installer.install("fixture", dry_run=False, keep_source=False)
                report = installer.verify(["fixture"])
                self.assertTrue(report["ok"])
                self.assertEqual("ready", report["profiles"][0]["artifacts"][0]["state"])

    def test_existing_symlink_escape_is_rejected(self) -> None:
        with fixture_server() as (server, _handler):
            url = f"http://127.0.0.1:{server.server_port}/artifact.bin"
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "root"
                outside = Path(directory) / "outside"
                root.mkdir()
                outside.mkdir()
                try:
                    (root / "models").symlink_to(outside, target_is_directory=True)
                except OSError as error:
                    self.skipTest(f"directory symlink is unavailable: {error}")
                installer = self.make_installer(root, fixture_profile(url))
                with self.assertRaisesRegex(SetupError, "escapes the repository root"):
                    installer.install("fixture", dry_run=False, keep_source=False)


if __name__ == "__main__":
    unittest.main()
