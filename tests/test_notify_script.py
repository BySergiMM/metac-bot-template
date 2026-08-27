"""scripts/notify.py: the CallMeBot WhatsApp client.

No test here makes a real HTTP request. `urllib.request.urlopen` is always
mocked; the only "real" thing exercised is URL construction, env var
handling, and error mapping.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
import unittest.mock
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import notify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeResponse:
    def read(self):
        return b"Message queued."

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SendWhatsapp(unittest.TestCase):
    def test_builds_the_documented_endpoint_and_params(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return _FakeResponse()

        with unittest.mock.patch("scripts.notify.urllib.request.urlopen", side_effect=fake_urlopen):
            notify.send_whatsapp("hello", api_key="KEY123", phone_number="+34600000000")

        self.assertTrue(captured["url"].startswith("https://api.callmebot.com/whatsapp.php?"))
        self.assertIn("apikey=KEY123", captured["url"])
        self.assertIn("phone=%2B34600000000", captured["url"])
        self.assertIn("text=hello", captured["url"])

    def test_refuses_to_send_an_empty_message(self):
        with self.assertRaises(notify.NotifyError):
            notify.send_whatsapp("   ", api_key="k", phone_number="+1")

    def test_http_error_is_a_delivery_error(self):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

        with unittest.mock.patch("scripts.notify.urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(notify.DeliveryError):
                notify.send_whatsapp("hi", api_key="k", phone_number="+1")

    def test_connection_error_is_a_delivery_error(self):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("unreachable")

        with unittest.mock.patch("scripts.notify.urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(notify.DeliveryError):
                notify.send_whatsapp("hi", api_key="k", phone_number="+1")

    def test_a_successful_call_raises_nothing(self):
        with unittest.mock.patch(
            "scripts.notify.urllib.request.urlopen", return_value=_FakeResponse()
        ):
            notify.send_whatsapp("hi", api_key="k", phone_number="+1")  # must not raise

    def test_the_api_key_never_appears_in_a_raised_exception(self):
        secret = "sk-super-secret-000111"

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 500, "error", {}, None)

        with unittest.mock.patch("scripts.notify.urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(notify.DeliveryError) as caught:
                notify.send_whatsapp("hi", api_key=secret, phone_number="+1")
        self.assertNotIn(secret, str(caught.exception))

    def test_the_phone_number_never_appears_in_a_raised_exception(self):
        phone = "+34600999888"

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 500, "error", {}, None)

        with unittest.mock.patch("scripts.notify.urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(notify.DeliveryError) as caught:
                notify.send_whatsapp("hi", api_key="k", phone_number=phone)
        self.assertNotIn(phone, str(caught.exception))


class RequireEnv(unittest.TestCase):
    def test_missing_env_var_names_the_variable_not_a_value(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOME_TOTALLY_UNSET_VAR", None)
            with self.assertRaises(notify.MissingConfigError) as caught:
                notify._require_env("SOME_TOTALLY_UNSET_VAR")
        self.assertIn("SOME_TOTALLY_UNSET_VAR", str(caught.exception))


class CliWhatsappCommand(unittest.TestCase):
    def setUp(self):
        self.env_patch = unittest.mock.patch.dict(
            os.environ,
            {"CALLMEBOT_APIKEY": "sk-test-000", "CALLMEBOT_PHONE_NUMBER": "+34600000000"},
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_missing_api_key_exits_1_without_a_traceback(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CALLMEBOT_APIKEY", None)
            stderr = io.StringIO()
            with unittest.mock.patch("sys.stderr", stderr):
                rc = notify.main(["whatsapp", "--message", "hi"])
        self.assertEqual(rc, 1)
        self.assertIn("CALLMEBOT_APIKEY", stderr.getvalue())

    def test_missing_phone_number_exits_1(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CALLMEBOT_PHONE_NUMBER", None)
            rc = notify.main(["whatsapp", "--message", "hi"])
        self.assertEqual(rc, 1)

    def test_a_successful_send_exits_0(self):
        with unittest.mock.patch(
            "scripts.notify.urllib.request.urlopen", return_value=_FakeResponse()
        ):
            rc = notify.main(["whatsapp", "--message", "hi"])
        self.assertEqual(rc, 0)

    def test_a_delivery_failure_exits_1_not_a_traceback(self):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("down")

        stderr = io.StringIO()
        with unittest.mock.patch("scripts.notify.urllib.request.urlopen", side_effect=fake_urlopen):
            with unittest.mock.patch("sys.stderr", stderr):
                rc = notify.main(["whatsapp", "--message", "hi"])
        self.assertEqual(rc, 1)
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_the_api_key_is_never_printed_anywhere_on_success_or_failure(self):
        secret = "sk-test-000"  # matches the env patch above
        stdout, stderr = io.StringIO(), io.StringIO()
        with unittest.mock.patch(
            "scripts.notify.urllib.request.urlopen", return_value=_FakeResponse()
        ):
            with unittest.mock.patch("sys.stdout", stdout), unittest.mock.patch("sys.stderr", stderr):
                notify.main(["whatsapp", "--message", "hi"])
        self.assertNotIn(secret, stdout.getvalue())
        self.assertNotIn(secret, stderr.getvalue())

    def test_the_phone_number_is_never_printed_anywhere_on_success_or_failure(self):
        phone = "+34600000000"  # matches the env patch in setUp
        stdout, stderr = io.StringIO(), io.StringIO()
        with unittest.mock.patch(
            "scripts.notify.urllib.request.urlopen", return_value=_FakeResponse()
        ):
            with unittest.mock.patch("sys.stdout", stdout), unittest.mock.patch("sys.stderr", stderr):
                notify.main(["whatsapp", "--message", "hi"])
        self.assertNotIn(phone, stdout.getvalue())
        self.assertNotIn(phone, stderr.getvalue())

    def test_real_subprocess_invocation_never_touches_the_network(self):
        """No mocking possible across a process boundary -- so this asserts
        the CLI fails safely (missing real credentials in this test
        environment) rather than ever attempting a live CallMeBot call."""
        env = dict(os.environ)
        env.pop("CALLMEBOT_APIKEY", None)
        env.pop("CALLMEBOT_PHONE_NUMBER", None)
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "notify.py"), "whatsapp",
             "--message", "test"],
            cwd=ROOT, capture_output=True, text=True, env=env, timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("CALLMEBOT_APIKEY", result.stderr)


if __name__ == "__main__":
    unittest.main()
