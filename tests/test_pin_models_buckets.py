"""Bucket wiring in pin_models: what gets generated, and what must not change.

pin_models reads os.environ at import time (deliberately: the generated main.py
is meant to be an honest record of what ran), so these tests reload it under a
controlled environment rather than mutating module state in place.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import unittest


@contextlib.contextmanager
def pinned_env(**env):
    """Reload pin_models AND hold the credential environment for the whole
    block, so call-time os.environ reads see the simulated set too."""
    managed = [
        "GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY", "GEMINI4_API_KEY",
        "GROQ_API_KEY", "OPENROUTER_API_KEY",
    ]
    saved = {name: os.environ.pop(name, None) for name in managed}
    os.environ.update(env)
    try:
        import backtest.pin_models as pm

        yield importlib.reload(pm)
    finally:
        for name in managed:
            os.environ.pop(name, None)
            if saved[name] is not None:
                os.environ[name] = saved[name]
        import backtest.pin_models as pm

        importlib.reload(pm)


def load(**env):
    """Reload pin_models with exactly the given credential environment.

    HAZARD, and it has bitten once. The environment is applied only for the
    duration of the IMPORT and restored before this function returns. Anything
    the module resolves at import time -- ACTIVE_FALLBACKS, ACTIVE_PARSER_EXTRA
    -- is captured correctly. Anything that reads os.environ at CALL time sees
    the real environment instead, so a test using `load` to simulate a
    credential set silently exercises a different one.

    That is how the crash of 2026-09-10 reached production: _parser_primary
    read os.environ when called, `test_one_credential` restored the env before
    calling selftest(), the selftest passed in CI, and the identical
    combination failed on the first scheduled run. Use `pinned_env` for
    anything that must hold the environment across a CALL.
    """
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
        """REQUIREMENT 6: one Gemini credential must produce no PER-CREDENTIAL
        machinery.

        Narrowed deliberately when the forecaster ensemble landed. BalancedLlm
        is no longer evidence of bucketing: the ensemble reuses that class to
        spread the five forecast calls over the three distinct primary models,
        which needs exactly one credential each. What must still be absent with
        one Gemini key is the bucket wiring -- bucket_backend, limiter_key, and
        any GEMINI2/3/4 env var name -- because that, not BalancedLlm, is what
        R11 declined to run in production.
        """
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        self.assertNotIn("bucket_backend", block)
        self.assertNotIn("limiter_key", block)
        for env_var in pm.GEMINI_BUCKET_ENV_VARS[1:]:
            self.assertNotIn(env_var, block)
        self.assertIn("FallbackLlm", block)

    def test_single_key_output_still_ensembles_the_forecaster(self):
        """The other half of the narrowing above: with one Gemini credential
        the default role must STILL ensemble across primary models, or the
        change silently reverts to five samples of one model."""
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        default_entry = block[block.index('"default"'):block.index('"summarizer"')]
        self.assertIn("BalancedLlm", default_entry)
        for primary in pm._ensemble_primaries(pm.DEFAULTS["default"]):
            self.assertIn('"{0}"'.format(primary), default_entry)
        # Roles called once per question gain nothing from a lottery over
        # models, so they stay single chains.
        rest = block[block.index('"summarizer"'):]
        self.assertNotIn("BalancedLlm", rest)

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

    def test_the_balanced_import_tracks_whether_the_block_uses_it(self):
        """The import must be present exactly when the generated block names
        BalancedLlm, and absent otherwise -- disagreeing either way is the one
        thing this generator can do that makes main.py raise at startup.

        Both directions are checked against the block itself rather than
        against a re-derived condition, so this stays true whichever feature
        (buckets or the forecaster ensemble) is the reason it appears.
        """
        for env in (
            {"GEMINI_API_KEY": "a", "GROQ_API_KEY": "g"},
            {"GEMINI_API_KEY": "a", "GEMINI2_API_KEY": "b", "GROQ_API_KEY": "g"},
            {},  # no fallback keys at all: single GeneralLlm per role
        ):
            with self.subTest(env=sorted(env)):
                pm = load(**env)
                once = pm.patch(self._main_src(), pm.DEFAULTS)
                uses = "BalancedLlm(" in once[once.index("llms={"):]
                imported = "from backtest.balanced_llm import" in once
                self.assertEqual(
                    uses, imported,
                    "block uses BalancedLlm={0} but import present={1}".format(
                        uses, imported),
                )

    def test_dropping_every_fallback_key_removes_the_balanced_import_again(self):
        """Converge-either-way: a source patched WITH the import must lose it
        when the keys that justified it are gone."""
        pm = load(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GROQ_API_KEY="g")
        balanced = pm.patch(self._main_src(), pm.DEFAULTS)
        self.assertIn("from backtest.balanced_llm import", balanced)
        pm = load()  # no fallback keys: nothing to balance or fall back to
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
        # pinned_env, not load: selftest() is a CALL, and the environment has
        # to still be in place while it runs. With `load` the env was restored
        # first, which is exactly how a real crash passed CI here.
        with pinned_env(**env) as pm:
            pm.selftest()  # raises AssertionError on failure

    def test_no_credentials(self):
        self._run_selftest()

    def test_the_production_pin_step_combination(self):
        """The exact credential set the scoring workflows give the pin step.

        Named explicitly because it is the one that broke: on 2026-09-10 three
        consecutive scheduled runs died here (34417432835, 34425442320,
        34445843668) in a combination the matrix already covered on paper --
        but `load` restored the environment before selftest() ran, so the
        simulated set was never the one under test. With pinned_env it is."""
        self._run_selftest(GEMINI_API_KEY="a", GROQ_API_KEY="g")

    def test_the_production_run_step_combination(self):
        """Every provider key present, as the Run bot step sees them."""
        self._run_selftest(GEMINI_API_KEY="a", GROQ_API_KEY="g",
                           OPENROUTER_API_KEY="o")

    def test_openrouter_only(self):
        self._run_selftest(OPENROUTER_API_KEY="o")

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

    def test_rate_limited_models_matches_the_real_registry(self):
        """pin_models duplicates the set of rate-limiter-registered model
        strings because it cannot import backtest.* (see the test below). This
        pins the copy to the original.

        It is a safety property, not tidiness: DEFAULT_LIMITS.get(model,
        ProviderLimits()) hands back an UNTHROTTLED limiter for an unknown key,
        so an ensemble bucket key that drifted out of the registry would look
        controlled and enforce nothing -- silently removing Groq's measured
        8000 TPM cap.
        """
        from backtest import rate_limiter

        pm = load()
        self.assertEqual(
            set(pm.RATE_LIMITED_MODELS),
            set(rate_limiter.DEFAULT_LIMITS),
            "pin_models.RATE_LIMITED_MODELS has drifted from "
            "rate_limiter.DEFAULT_LIMITS",
        )

    def test_every_ensemble_bucket_key_is_rate_limited(self):
        """The keys the generator actually emits must all be registered, for
        the same reason. Checked against the real registry, not the copy."""
        from backtest import rate_limiter

        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        for primary in pm._ensemble_primaries(pm.DEFAULTS["default"]):
            self.assertIn(primary, rate_limiter.DEFAULT_LIMITS, primary)

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
