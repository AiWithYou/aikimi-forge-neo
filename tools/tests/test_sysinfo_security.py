# ruff: noqa: E402, I001
# shared_cmd_options reads this environment switch while modules.sysinfo imports.
import json
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")

from modules import sysinfo  # noqa: E402


SENTINEL = "sysinfo-secret-3d7b2f"
PRIVATE_PATH = r"Q:\private\workspace\credential.txt"


class SysinfoSecurityTests(unittest.TestCase):
    def test_argv_and_environment_redact_credentials_urls_and_paths(self):
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "launch.py",
                    "--api-auth",
                    f"user:{SENTINEL}",
                    f"--future-secret-token={SENTINEL}",
                    "--tls-keyfile",
                    PRIVATE_PATH,
                ],
            ),
            mock.patch.dict(
                os.environ,
                {
                    "INDEX_URL": f"https://user:{SENTINEL}@packages.example/simple",
                    "TORCH_INDEX_URL": f"https://packages.example/simple?token={SENTINEL}",
                    "COMMANDLINE_ARGS": f"--gradio-auth user:{SENTINEL}",
                },
                clear=False,
            ),
        ):
            serialized = json.dumps(
                {
                    "argv": sysinfo.get_argv(),
                    "environment": sysinfo.get_environment(),
                },
                ensure_ascii=False,
            )

        self.assertNotIn(SENTINEL, serialized)
        self.assertNotIn(PRIVATE_PATH, serialized)
        self.assertIn("<redacted>", serialized)

    def test_complete_support_payload_is_redacted_before_checksum(self):
        fake_config = {
            "provider_token": SENTINEL,
            "custom_path": PRIVATE_PATH,
            "safe_option": True,
        }
        fake_exceptions = [
            {
                "exception": f"Authorization: Bearer {SENTINEL}",
                "traceback": [[PRIVATE_PATH, f"password={SENTINEL}"]],
            }
        ]
        fake_extensions = [
            {
                "name": "fixture",
                "path": PRIVATE_PATH,
                "remote": f"https://user:{SENTINEL}@git.example/repo.git",
            }
        ]
        with (
            mock.patch.object(sysinfo, "get_config", return_value=fake_config),
            mock.patch.object(sysinfo, "git_status", return_value=f"token={SENTINEL}"),
            mock.patch.object(sysinfo, "get_argv", return_value=[f"--api-auth={SENTINEL}"]),
            mock.patch.object(sysinfo, "get_torch_sysinfo", return_value={"path": PRIVATE_PATH}),
            mock.patch.object(sysinfo.errors, "get_exceptions", return_value=fake_exceptions),
            mock.patch.object(sysinfo, "get_cpu_info", return_value={"model": "CPU"}),
            mock.patch.object(sysinfo, "get_ram_info", return_value={"total": "64GB"}),
            mock.patch.object(sysinfo, "get_extensions", return_value=fake_extensions),
            mock.patch.object(
                sysinfo,
                "get_environment",
                return_value={"INDEX_URL": f"https://user:{SENTINEL}@packages.example"},
            ),
            mock.patch.object(sysinfo, "get_packages", return_value=[f"pkg @ https://user:{SENTINEL}@example/pkg.whl"]),
            mock.patch.object(sysinfo.launch_utils, "git_tag", return_value="test"),
            mock.patch.object(sysinfo.paths_internal, "script_path", PRIVATE_PATH),
            mock.patch.object(sysinfo.paths_internal, "data_path", PRIVATE_PATH),
            mock.patch.object(sysinfo.paths_internal, "extensions_dir", PRIVATE_PATH),
        ):
            payload = sysinfo.get()

        self.assertNotIn(SENTINEL, payload)
        self.assertNotIn(PRIVATE_PATH, payload)
        self.assertTrue(sysinfo.check(payload))


if __name__ == "__main__":
    unittest.main()
