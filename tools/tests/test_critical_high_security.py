"""Exercise untrusted inputs at the affected production boundaries without GPU work."""

from __future__ import annotations

import importlib.util
import io
import ssl
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

import httpx

from modules.aikimi_security.startup_probe import gradio_url_ok_fix

ROOT = Path(__file__).resolve().parents[2]
DATA_TOOL = ROOT / "extensions-builtin/forge_legacy_preprocessors/annotator/leres/pix2pix/util/get_data.py"


class StartupTLSValidationTests(unittest.TestCase):
    def setUp(self):
        shared = types.ModuleType("modules.shared")
        shared.cmd_opts = types.SimpleNamespace(tls_certfile=None)
        modules = types.ModuleType("modules")
        modules.shared = shared
        self.enterContext(mock.patch.dict(sys.modules, {"modules": modules, "modules.shared": shared}))
        self.shared = shared
        self.probe = gradio_url_ok_fix

    def test_https_probe_never_disables_certificate_validation(self):
        with mock.patch.object(httpx, "head", return_value=types.SimpleNamespace(status_code=200)) as head:
            self.assertTrue(self.probe("https://localhost:7861"))
        self.assertIs(head.call_args.kwargs.get("verify", True), True)

    def test_configured_local_certificate_is_trusted_with_hostname_checks(self):
        self.shared.cmd_opts.tls_certfile = "test-local-certificate.pem"
        context = ssl.create_default_context()
        with (
            mock.patch.object(ssl, "create_default_context", return_value=context),
            mock.patch.object(ssl.SSLContext, "load_verify_locations") as load,
            mock.patch.object(httpx, "head", return_value=types.SimpleNamespace(status_code=200)) as head,
        ):
            self.assertTrue(self.probe("https://localhost:7861"))
        load.assert_called_once_with(cafile=self.shared.cmd_opts.tls_certfile)
        self.assertIs(head.call_args.kwargs["verify"], context)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_untrusted_certificate_failure_is_not_retried_without_verification(self):
        with mock.patch.object(httpx, "head", side_effect=httpx.ConnectError("certificate verify failed")) as head:
            self.assertFalse(self.probe("https://localhost:7861"))
        self.assertEqual(head.call_count, 1)
        self.assertIs(head.call_args.kwargs.get("verify", True), True)

    def test_http_and_authenticated_startup_responses_still_work(self):
        for code in (200, 401, 302):
            with (
                self.subTest(code=code),
                mock.patch.object(httpx, "head", return_value=types.SimpleNamespace(status_code=code)),
            ):
                self.assertTrue(self.probe("http://127.0.0.1:7861"))


class LegacyArchiveBoundaryTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_legacy_dataset_security", DATA_TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        # The independent preprocessor environment owns the HTML parser. These
        # tests execute the real archive downloader without parsing a web page.
        parser = types.ModuleType("bs4")
        parser.BeautifulSoup = mock.Mock(side_effect=AssertionError("HTML parsing is outside this test"))
        with mock.patch.dict(sys.modules, {"bs4": parser}):
            spec.loader.exec_module(self.tool)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.destination = self.root / "datasets"
        self.outside = self.root / "outside.txt"
        self.outside.write_bytes(b"private")

    def archive(self, name, *, kind=tarfile.REGTYPE, linkname=""):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            member = tarfile.TarInfo(name)
            member.type, member.linkname = kind, linkname
            if member.isfile():
                member.size = 7
                archive.addfile(member, io.BytesIO(b"dataset"))
            else:
                archive.addfile(member)
        return output.getvalue()

    def download(self, payload, name="dataset.tar.gz"):
        with mock.patch.object(self.tool.requests, "get", return_value=types.SimpleNamespace(content=payload)):
            self.tool.GetData(verbose=False)._download_data("https://dataset.example/" + name, str(self.destination))

    def test_tar_parent_traversal_cannot_overwrite_a_file_outside_destination(self):
        with self.assertRaises(tarfile.FilterError):
            self.download(self.archive("../outside.txt"))
        self.assertEqual(self.outside.read_bytes(), b"private")

    def test_tar_link_outside_destination_is_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(kind=kind), self.assertRaises(tarfile.FilterError):
                self.download(self.archive("escape", kind=kind, linkname="../outside.txt"))
        self.assertEqual(self.outside.read_bytes(), b"private")

    def test_tar_special_files_are_rejected(self):
        with self.assertRaises(tarfile.FilterError):
            self.download(self.archive("device", kind=tarfile.CHRTYPE))

    def test_normal_tar_and_zip_datasets_are_extracted(self):
        self.download(self.archive("normal/image.txt"))
        self.assertEqual((self.destination / "normal/image.txt").read_bytes(), b"dataset")
        output = io.BytesIO()
        with ZipFile(output, "w") as archive:
            archive.writestr("normal/zip.txt", b"zip dataset")
        self.download(output.getvalue(), "dataset.zip")
        self.assertEqual((self.destination / "normal/zip.txt").read_bytes(), b"zip dataset")


class H3CommandBoundaryTests(unittest.TestCase):
    def test_shared_model_path_is_a_literal_argument_and_never_a_shell_command(self):
        from modules_forge import minimax_h3_setup_ui

        value = 'C:\\models\\shared & echo unintended | command "quoted"'
        process = mock.Mock(stdout=io.StringIO(""))
        process.wait.return_value = process.poll.return_value = 0
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(minimax_h3_setup_ui.subprocess, "Popen", return_value=process) as popen:
            list(minimax_h3_setup_ui._setup_process(value, root, root / "runtime"))
        arguments = popen.call_args.args[0]
        self.assertEqual(arguments[-2:], ["--share-models", value])
        self.assertEqual(arguments[0], sys.executable)
        self.assertEqual(arguments[2], str(root / "tools/setup_minimax_h3.py"))
        self.assertFalse(popen.call_args.kwargs.get("shell", False))


if __name__ == "__main__":
    unittest.main()
