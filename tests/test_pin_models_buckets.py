"""Bucket wiring in pin_models: what gets generated, and what must not change.

pin_models reads os.environ at import time (deliberately: the generated main.py
is meant to be an honest record of what ran), so these tests reload it under a
controlled environment rather than mutating module state in place.
"""

from __future__ import annotations

import importlib
import os
import unittest


def load(**env):
    """Reload pin_models with exactly the given credential environment."""
    saved = {}
    managed = [
        "GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY", "GEMINI4_API_KEY",
        "GROQ_API_KEY", "OPENROUTER_API_KEY",
    ]
    for name in managed:
        saved[name] = os.environ.pop(name, None)
    os.environ.update(env)
    try:
        import backtest.pin_models as pm

        return importlib.reload(pm)
    finally:
        for name in managed:
            os.environ.pop(name, None)
            if saved[name] is not None:
                os.environ[name] = saved[name]


class BucketDetectionTests(unittest.TestCase):
    def test_one_key_means_no_balancing(self):
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertFalse(pm.BALANCED)
        self.assertEqual(len(pm.ACTIVE_GEMINI_BUCKETS), 1)

    def test_absent_secondary_keys_degrade_to_the_buckets_available(self):
        """Safety requirement 2."""
        pm = load(GEMINI_API_KEY="a", GEMINI3_API_KEY="c", GROQ_API_KEY="g")
        self.assertTrue(pm.BALANCED)
        self.assertEqual([e for e, _k in pm.ACTIVE_GEMINI_BUCKETS],
                         ["GEMINI_API_KEY", "GEMINI3_API_KEY"])

    def test_four_keys_give_four_buckets_in_a_fixed_order(self):
        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GEMINI3_API_KEY="c",
                  GEMINI4_API_KEY="d", GROQ_API_KEY="g")
        self.assertEqual([k for _e, k in pm.ACTIVE_GEMINI_BUCKETS],
                         list(pm.GEMINI_BUCKET_KEYS))

    def test_no_gemini_key_at_all_means_no_buckets(self):
        pm = load(GROQ_API_KEY="g")
        self.assertFalse(pm.BALANCED)
        self.assertEqual(pm.ACTIVE_GEMINI_BUCKETS, [])


class GeneratedBlockTests(unittest.TestCase):
    def test_single_key_output_contains_no_bucket_machinery(self):
        """REQUIREMENT 6: the one-key install must look exactly as before."""
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        self.assertNotIn("BalancedLlm", block)
        self.assertNotIn("bucket_backend", block)
        self.assertNotIn("limiter_key", block)
        self.assertIn("FallbackLlm", block)

    def test_balanced_output_wraps_one_chain_per_credential(self):
        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GEMINI3_API_KEY="c",
                  GEMINI4_API_KEY="d", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        self.assertIn("BalancedLlm", block)
        for env_var in pm.GEMINI_BUCKET_ENV_VARS:
            self.assertIn('"{0}"'.format(env_var), block)
        for bucket_key in pm.GEMINI_BUCKET_KEYS:
            self.assertIn('"{0}"'.format(bucket_key), block)

    def test_the_parser_keeps_gemini_and_never_gains_groq(self):
        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        parser = block[block.index('"parser"'):]
        self.assertIn("bucket_backend", parser)
        self.assertNotIn("gpt-oss-120b", parser,
                         "the parser must not gain a backend that answers in prose")

    def test_no_credential_value_is_ever_written_into_the_block(self):
        """Only variable NAMES may appear; a secret must not reach main.py."""
        pm = load(GEMINI_API_KEY="SECRET-A", GEMINI2_API_KEY="SECRET-B",
                  GROQ_API_KEY="SECRET-G")
        block = pm.build_block(pm.DEFAULTS)
        for secret in ("SECRET-A", "SECRET-B", "SECRET-G"):
            self.assertNotIn(secret, block)

    def test_generated_code_is_valid_python(self):
        import ast

        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GEMINI3_API_KEY="c",
                  GEMINI4_API_KEY="d", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS).strip().rstrip(",")
        # The block is `llms={...}`; parse the dict literal it assigns.
        self.assertTrue(block.startswith("llms="))
        tree = ast.parse("x = " + block[len("llms="):])
        assigned = tree.body[0].value
        self.assertIsInstance(assigned, ast.Dict,
                              "llms= must be a dict literal, not a set")
        keys = [k.value for k in assigned.keys]
        self.assertEqual(sorted(keys),
                         ["default", "parser", "researcher", "summarizer"])


class ImportWiringTests(unittest.TestCase):
    def _main_src(self):
        import io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return io.open(os.path.join(root, "main.py"), encoding="utf-8").read()

    def test_balanced_import_is_added_exactly_once_and_is_idempotent(self):
        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GROQ_API_KEY="g")
        once = pm.patch(self._main_src(), pm.DEFAULTS)
        self.assertIn("from backtest.balanced_llm import", once)
        self.assertEqual(once.count("from backtest.balanced_llm import"), 1)
        self.assertEqual(pm.patch(once, pm.DEFAULTS), once, "patch must be idempotent")

    def test_single_key_patch_adds_no_balanced_import(self):
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        once = pm.patch(self._main_src(), pm.DEFAULTS)
        self.assertNotIn("from backtest.balanced_llm import", once)
        self.assertIn("from backtest.fallback_llm import FallbackLlm", once)

    def test_dropping_a_key_removes_the_balanced_import_again(self):
        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GROQ_API_KEY="g")
        balanced = pm.patch(self._main_src(), pm.DEFAULTS)
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        reverted = pm.patch(balanced, pm.DEFAULTS)
        self.assertNotIn("from backtest.balanced_llm import", reverted)


class SelftestAcrossBucketCountsTests(unittest.TestCase):
    """pin_models runs its own selftest on every invocation, before touching
    main.py. Balancing multiplies how many times a model string appears in the
    generated block -- once per chain per role -- and an assertion written for
    the single-chain shape fails only when a second credential appears. That is
    exactly how it escaped local checks and broke in CI (run 32385415823), so
    every bucket count is exercised here.

    selftest() operates on synthetic sources; it never writes main.py.
    """

    def _run_selftest(self, **env):
        pm = load(**env)
        pm.selftest()  # raises AssertionError on failure

    def test_no_credentials(self):
        self._run_selftest()

    def test_one_credential(self):
        self._run_selftest(GEMINI_API_KEY="a", GROQ_API_KEY="g")

    def test_two_credentials(self):
        self._run_selftest(GEMINI_API_KEY="a", GEMINI2_API_KEY="b",
                           GROQ_API_KEY="g")

    def test_three_credentials(self):
        self._run_selftest(GEMINI_API_KEY="a", GEMINI2_API_KEY="b",
                           GEMINI3_API_KEY="c", GROQ_API_KEY="g")

    def test_four_credentials(self):
        self._run_selftest(GEMINI_API_KEY="a", GEMINI2_API_KEY="b",
                           GEMINI3_API_KEY="c", GEMINI4_API_KEY="d",
                           GROQ_API_KEY="g")

    def test_groq_only(self):
        self._run_selftest(GROQ_API_KEY="g")


class DriftTests(unittest.TestCase):
    """pin_models duplicates the bucket keys because it runs as a bare script
    and cannot import backtest.*. That duplication must never drift."""

    def test_bucket_keys_match_the_rate_limiter(self):
        from backtest import rate_limiter as rl

        pm = load(GEMINI_API_KEY="a")
        self.assertEqual(tuple(pm.GEMINI_BUCKET_KEYS), tuple(rl.GEMINI_BUCKET_KEYS))

    def test_bucket_model_matches_the_rate_limiter(self):
        from backtest import rate_limiter as rl

        pm = load(GEMINI_API_KEY="a")
        self.assertEqual(pm.GEMINI_BUCKET_MODEL, rl.GEMINI_MODEL)

    def test_pin_models_imports_nothing_from_backtest(self):
        """It runs as `python backtest/pin_models.py`, with the repo root off
        sys.path. A backtest.* import raises before the bot ever starts."""
        import io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = io.open(os.path.join(root, "backtest", "pin_models.py"),
                      encoding="utf-8").read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import backtest", "from backtest")):
                self.fail("pin_models must not import backtest.*: " + stripped)


if __name__ == "__main__":
    unittest.main()
