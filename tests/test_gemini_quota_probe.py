"""Offline tests for the Gemini quota-bucket probe.

Every test here runs without network and without credentials. The probe's own
network phases are exercised only in CI, where the secrets exist.

The 429 fixture below is transcribed verbatim from a real Google refusal
captured in workflow run 32295883751 -- not invented, so a change in Google's
error shape breaks these tests instead of silently returning {}.
"""

from __future__ import annotations

import json
import os
import unittest

from research import probe_gemini_quota_buckets as probe

# Real payload shape, from google.rpc.QuotaFailure in run 32295883751.
REAL_429 = json.dumps({
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{
                 "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                 "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                 "quotaDimensions": {"model": "gemini-3.5-flash-lite", "location": "global"},
                 "quotaValue": "15",
             }]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "11s"},
        ],
    }
})


class QuotaExtractionTests(unittest.TestCase):
    def test_it_finds_every_quota_field_in_a_real_refusal(self):
        facts = probe.quota_facts(REAL_429)
        self.assertEqual(facts["quotaValue"], "15")
        self.assertEqual(
            facts["quotaId"],
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
        )
        self.assertEqual(
            facts["quotaDimensions"],
            {"model": "gemini-3.5-flash-lite", "location": "global"},
        )
        self.assertIn("free_tier", facts["quotaMetric"])

    def test_the_free_tier_marker_is_visible(self):
        """After a paid upgrade this substring must disappear. That is how the
        probe proves the upgrade actually took effect, rather than assuming it."""
        self.assertTrue(probe.quota_facts(REAL_429)["quotaId"].endswith("-FreeTier"))

    def test_a_non_quota_error_yields_no_facts_instead_of_guesses(self):
        body = json.dumps({"error": {"code": 401, "message": "API key not valid"}})
        self.assertEqual(probe.quota_facts(body), {})

    def test_garbage_does_not_raise(self):
        self.assertEqual(probe.quota_facts("not json at all"), {})
        self.assertEqual(probe.quota_facts(""), {})


class CredentialHygieneTests(unittest.TestCase):
    def test_safe_id_is_stable_for_the_same_secret(self):
        self.assertEqual(probe.safe_id("abc123"), probe.safe_id("abc123"))

    def test_safe_id_separates_different_secrets(self):
        self.assertNotEqual(probe.safe_id("abc123"), probe.safe_id("abc124"))

    def test_safe_id_reveals_no_key_material(self):
        # Deliberately self-describing rather than key-shaped-and-opaque.
        # The assertions below hold for ANY string, so nothing is weakened,
        # and a public repository should not carry text that a secret
        # scanner -- or a reader -- has to measure to rule out.
        key = "AIza-SYNTHETIC-FIXTURE-NOT-A-REAL-CREDENTIAL"
        ident = probe.safe_id(key)
        self.assertEqual(len(ident), 8)
        self.assertNotIn(ident, key)
        for size in range(4, len(key)):
            self.assertNotIn(key[:size], ident)

    def test_redact_removes_a_live_key_from_provider_text(self):
        keys = [{"_key": "SUPERSECRET"}, {"_key": ""}]
        out = probe.redact("rejected SUPERSECRET here", keys)
        self.assertNotIn("SUPERSECRET", out)
        self.assertIn("<REDACTED>", out)

    def test_redact_tolerates_absent_keys(self):
        self.assertEqual(probe.redact("nothing", [{"_key": ""}]), "nothing")


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self._saved = {n: os.environ.pop(n, None) for n in probe.KEY_VARS}

    def tearDown(self):
        for name, value in self._saved.items():
            if value is not None:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)

    def test_all_four_slots_are_reported_even_when_absent(self):
        keys = probe.load_keys()
        self.assertEqual([k["env"] for k in keys], probe.KEY_VARS)
        self.assertTrue(all(k["present"] is False for k in keys))
        self.assertTrue(all(k["id"] is None for k in keys))

    def test_a_present_key_gets_an_identity(self):
        os.environ["GEMINI2_API_KEY"] = "value-two"
        keys = probe.load_keys()
        second = [k for k in keys if k["env"] == "GEMINI2_API_KEY"][0]
        self.assertTrue(second["present"])
        self.assertEqual(second["id"], probe.safe_id("value-two"))

    def test_two_secrets_holding_the_same_value_are_visibly_identical(self):
        """The failure mode that would silently invalidate the whole experiment:
        four secrets, one underlying key."""
        os.environ["GEMINI_API_KEY"] = "same"
        os.environ["GEMINI3_API_KEY"] = "same"
        keys = {k["env"]: k["id"] for k in probe.load_keys()}
        self.assertEqual(keys["GEMINI_API_KEY"], keys["GEMINI3_API_KEY"])

    def test_no_keys_present_exits_without_calling_out(self):
        # argparse reads sys.argv, which under a test runner holds the runner's
        # own flags; isolate it so this asserts the probe's behaviour, not
        # unittest's command line.
        import sys

        saved = sys.argv
        sys.argv = ["probe_gemini_quota_buckets.py"]
        try:
            self.assertEqual(probe.main(), 2)
        finally:
            sys.argv = saved


class SafetyTests(unittest.TestCase):
    def test_saturation_is_bounded(self):
        """An unbounded burst is exactly what this repo's rate limiter exists to
        prevent; the probe must not reintroduce one."""
        self.assertLessEqual(probe.SATURATION_ATTEMPTS, 25)

    def test_the_configurable_cap_still_has_a_ceiling(self):
        """A cap that any invocation can raise without limit is not a cap."""
        self.assertLessEqual(probe.MAX_ATTEMPTS_CEILING, 100)
        self.assertGreater(probe.MAX_ATTEMPTS_CEILING, probe.SATURATION_ATTEMPTS)

    def test_max_attempts_above_the_ceiling_is_refused(self):
        import sys

        saved = sys.argv
        sys.argv = ["p", "--max-attempts", str(probe.MAX_ATTEMPTS_CEILING + 1)]
        try:
            with self.assertRaises(SystemExit):
                probe.main()
        finally:
            sys.argv = saved

    def test_an_unknown_key_name_is_refused(self):
        import sys

        saved = sys.argv
        sys.argv = ["p", "--keys", "NOT_A_REAL_KEY"]
        try:
            with self.assertRaises(SystemExit):
                probe.main()
        finally:
            sys.argv = saved

    def test_the_probe_never_imports_the_forecaster(self):
        source = open(probe.__file__, encoding="utf-8").read()
        self.assertNotIn("forecasting_tools", source)
        self.assertNotIn("import main", source)

    def test_calls_request_a_single_token(self):
        source = open(probe.__file__, encoding="utf-8").read()
        self.assertIn('"max_tokens": 1', source)


class RoundAccountingTests(unittest.TestCase):
    """A round must report what it actually cost, so a 'no refusal' result can
    be read as a lower bound instead of being mistaken for a measured quota."""

    def test_a_no_refusal_round_is_marked_and_never_yields_a_quota(self):
        calls = {"n": 0}

        def always_ok(key, timeout=30):
            calls["n"] += 1
            return {"ok": True, "http": 200, "latency_s": 0.0}

        original, probe.call = probe.call, always_ok
        try:
            facts = probe.saturate({"_key": "x"}, [], max_attempts=7)
        finally:
            probe.call = original

        self.assertTrue(facts["_no_refusal"])
        self.assertEqual(facts["_attempts_made"], 7)
        self.assertEqual(calls["n"], 7, "must respect the cap exactly")
        self.assertNotIn("quotaValue", facts)

    def test_the_exact_attempt_of_the_first_429_is_recorded(self):
        state = {"n": 0}

        def fail_after_three(key, timeout=30):
            state["n"] += 1
            if state["n"] <= 3:
                return {"ok": True, "http": 200, "latency_s": 0.0}
            return {"ok": False, "http": 429, "latency_s": 0.0, "body": REAL_429}

        original, probe.call = probe.call, fail_after_three
        try:
            facts = probe.saturate({"_key": "x"}, [], max_attempts=60)
        finally:
            probe.call = original

        self.assertEqual(facts["_attempt"], 4)
        self.assertEqual(facts["quotaValue"], "15")
        self.assertIn("_elapsed_s", facts)


if __name__ == "__main__":
    unittest.main()
