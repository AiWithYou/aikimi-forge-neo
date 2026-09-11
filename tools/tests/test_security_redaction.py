import logging
import unittest
from pathlib import Path

from modules import logging_config
from modules.aikimi_security.redaction import (
    REDACTED,
    is_sensitive_key,
    redact_argv,
    redact_mapping,
    redact_text,
    redact_url,
    safe_error_message,
    sanitized_subprocess_environment,
)

SECRET = "s3cr3t-ABC-987"  # gitleaks:allow  # noqa: S105 - synthetic redaction sentinel


class SecurityRedactionTests(unittest.TestCase):
    def assert_secret_absent(self, value):
        self.assertNotIn(SECRET, str(value))

    def test_sensitive_mapping_keys_are_redacted_recursively(self):
        value = {
            "api_auth": SECRET,
            "authtoken": SECRET,
            "ngrok": SECRET,
            "ngrok_options": {"basic_auth": SECRET},
            "tls_keyfile": Path(f"C:/private/{SECRET}.key"),
            "nested": [{"client_secret": SECRET}, {"access_token": SECRET.encode()}],
            "ngrok_region": "jp",
            "authentication_enabled_status": True,
            "safe": "visible",
        }

        result = redact_mapping(value)

        self.assert_secret_absent(result)
        self.assertEqual(result["api_auth"], REDACTED)
        self.assertEqual(result["authtoken"], REDACTED)
        self.assertEqual(result["ngrok"], REDACTED)
        self.assertEqual(result["ngrok_options"], REDACTED)
        self.assertEqual(result["tls_keyfile"], REDACTED)
        self.assertEqual(result["ngrok_region"], "jp")
        self.assertIs(result["authentication_enabled_status"], True)
        self.assertEqual(result["safe"], "visible")

    def test_sensitive_key_detection_covers_future_and_provider_names(self):
        for key in (
            "api-key",
            "client_secret",
            "HF_TOKEN",
            "Authorization",
            "x-amz-signature",
            "private.key",
            "authtoken",
        ):
            with self.subTest(key=key):
                self.assertTrue(is_sensitive_key(key))
        self.assertFalse(is_sensitive_key("ngrok_region"))
        self.assertTrue(is_sensitive_key("authentication_enabled_status"))

    def test_argv_redacts_separate_inline_ngrok_and_future_secret_options(self):
        result = redact_argv(
            [
                "launch.py",
                "--api-auth",
                SECRET,
                f"--gradio-auth={SECRET}",
                "--ngrok",
                SECRET,
                "--ngrok-options",
                '{"authtoken":"' + SECRET + '","basic_auth":"user:pass"}',
                f"--provider-client-secret={SECRET}",
                "--provider-credential-file",
                f"C:/private/{SECRET}.json",
                "--ngrok-region",
                "jp",
            ]
        )

        self.assert_secret_absent(result)
        self.assertIn("--api-auth", result)
        self.assertIn(f"--gradio-auth={REDACTED}", result)
        self.assertEqual(result[-2:], ["--ngrok-region", "jp"])

    def test_commandline_text_redacts_ngrok_and_auth_values(self):
        value = f"COMMANDLINE_ARGS=--ngrok {SECRET} --api-auth={SECRET} --tls-keyfile C:/private/{SECRET}.key"
        result = redact_text(value)
        self.assert_secret_absent(result)
        self.assertIn(REDACTED, result)

    def test_json_and_assignment_secret_forms_are_redacted(self):
        value = f'worker options={{"authtoken":"{SECRET}","safe":"visible"}} ngrok={SECRET} X_AMZ_SIGNATURE={SECRET}'

        result = redact_text(value)

        self.assert_secret_absent(result)
        self.assertIn('"safe":"visible"', result)

    def test_subprocess_environment_drops_proxy_index_and_secret_variables(self):
        result = sanitized_subprocess_environment(
            {
                "PATH": "C:/runtime/bin",
                "HF_TOKEN": SECRET,
                "HTTP_PROXY": f"http://user:{SECRET}@proxy.test",
                "INDEX_URL": f"https://user:{SECRET}@packages.test/simple",
                "SAFE_MODE": "1",
            }
        )

        self.assertEqual(result, {"PATH": "C:/runtime/bin", "SAFE_MODE": "1"})

    def test_url_redacts_userinfo_and_sensitive_query_variants(self):
        value = (
            f"https://alice:{SECRET}@example.test/image.png?token={SECRET}"
            f"&key={SECRET}&X-Amz-Signature={SECRET}&safe=visible#fragment"
        )

        result = redact_url(value)

        self.assert_secret_absent(result)
        self.assertNotIn("alice", result)
        self.assertNotIn("fragment", result)
        self.assertIn("safe=visible", result)

    def test_headers_and_local_paths_are_redacted(self):
        value = (
            f"Authorization: Bearer {SECRET}\n"
            f"Cookie: sid={SECRET}\n"
            f"X-Api-Key: {SECRET}\n"
            f"windows=C:/Users/person/private/{SECRET}.txt\n"
            f"linux=/home/person/private/{SECRET}.txt"
        )

        result = redact_text(value)

        self.assert_secret_absent(result)
        self.assertGreaterEqual(result.count(REDACTED), 3)
        self.assertGreaterEqual(result.count("<local-path>"), 2)

    def test_exception_message_is_bounded_and_redacted(self):
        error = RuntimeError(f"download failed: https://user:{SECRET}@example.test/a?token={SECRET} " + "x" * 1000)

        result = safe_error_message(error, limit=120)

        self.assert_secret_absent(result)
        self.assertLessEqual(len(result), 120)
        self.assertNotIn("user", result)

    def test_console_logging_filter_redacts_messages_and_exceptions(self):
        record = logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            f"Authorization: Bearer {SECRET} at C:/private/{SECRET}.txt",
            (),
            (RuntimeError, RuntimeError(f"token={SECRET}"), None),
        )

        self.assertTrue(logging_config.RedactingFilter().filter(record))
        self.assert_secret_absent(record.getMessage())
        self.assertIsNone(record.exc_info)

    def test_ngrok_failure_source_never_prints_token_value(self):
        source = (Path(__file__).resolve().parents[2] / "modules" / "ngrok.py").read_text(encoding="utf-8")
        self.assertNotIn("Your token", source)
        self.assertNotIn("{token}", source)


if __name__ == "__main__":
    unittest.main()
