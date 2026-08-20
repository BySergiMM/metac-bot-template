"""Four independent Gemini quota buckets, balanced at admission.

Offline throughout: no network, no credentials, virtual time.

The properties under test are the ones that make this safe rather than merely
faster. In order of how badly they fail if broken:

  1. a bucket with no explicit limits is UNTHROTTLED, because
     limits_for() falls back to ProviderLimits(). A bucket added without a
     DEFAULT_LIMITS entry would look controlled and rate-limit nothing.
  2. a backend with no limiter_key must behave exactly as before buckets
     existed, or the single-key install changes underneath us.
  3. a dead secondary credential must not cost a prediction.
"""

from __future__ import annotations

import asyncio
import bisect
import heapq
import sys
import types
import unittest

if "forecasting_tools" not in sys.modules:
    _pkg = types.ModuleType("forecasting_tools")
    _ai = types.ModuleType("forecasting_tools.ai_models")
    _gl = types.ModuleType("forecasting_tools.ai_models.general_llm")

    class _StubGeneralLlm:
        def __init__(self, model, temperature=None, timeout=None,
                     allowed_tries=1, **kwargs):
            self.model = model
            self.allowed_tries = allowed_tries
            self.litellm_kwargs = {"temperature": temperature, "timeout": timeout}
            self.litellm_kwargs.update(kwargs)

        async def invoke(self, prompt, system_prompt=None):  # pragma: no cover
            raise NotImplementedError

    _gl.GeneralLlm = _StubGeneralLlm
    _ai.general_llm = _gl
    _pkg.ai_models = _ai
    sys.modules["forecasting_tools"] = _pkg
    sys.modules["forecasting_tools.ai_models"] = _ai
    sys.modules["forecasting_tools.ai_models.general_llm"] = _gl

from backtest import rate_limiter as rl  # noqa: E402
from backtest.balanced_llm import BalancedLlm, bucket_backend  # noqa: E402
from backtest.fallback_llm import FallbackLlm  # noqa: E402
from forecasting_tools.ai_models.general_llm import GeneralLlm  # noqa: E402

GEMINI = rl.GEMINI_MODEL
GROQ = "groq/openai/gpt-oss-120b"
OPENROUTER = "openrouter/nvidia/nemotron-3.5-lightning:free"


class Clock:
    """Same shape as tests/test_fallback_e2e_simulation.py's clock."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds
        await asyncio.sleep(0)


class ScheduledClock:
    """Virtual time that advances only when nothing can run.

    The simpler clock above ADDS every sleeper's duration to a shared counter,
    so two tasks sleeping 4s concurrently move it 8s. That is harmless for
    pacing assertions but inflates elapsed time, which makes the limiter's
    300-second wait ceiling fire spuriously under a large burst. This clock
    overlaps concurrent sleeps the way a real loop does, and reproduces the
    live E2E run's duration to within 0.5%.
    """

    def __init__(self):
        self.now = 0.0
        self._waiters = []
        self._seq = 0

    def time(self):
        return self.now

    async def sleep(self, seconds):
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        self._seq += 1
        future = asyncio.get_event_loop().create_future()
        heapq.heappush(self._waiters, (self.now + seconds, self._seq, future))
        await future

    async def drive(self, main):
        while not main.done():
            for _ in range(64):
                await asyncio.sleep(0)
                if main.done():
                    return
            if not self._waiters:
                continue
            when = self._waiters[0][0]
            if when > self.now:
                self.now = when
            while self._waiters and self._waiters[0][0] <= self.now:
                _when, _seq, future = heapq.heappop(self._waiters)
                if not future.done():
                    future.set_result(None)


def run_scheduled(clock, factory):
    """Run `factory()` under `clock`. The coroutine is built inside the loop."""
    loop = asyncio.new_event_loop()
    try:
        async def outer():
            main = asyncio.ensure_future(factory())
            driver = asyncio.ensure_future(clock.drive(main))
            try:
                return await main
            finally:
                driver.cancel()

        return loop.run_until_complete(outer())
    finally:
        loop.close()


# Gemini's measured median latency is 1.05s (153 calls, E2E run 32297317091).
# Latency matters here: a provider that answers instantly lets every task enter
# the limiter at t=0, and the tail then waits past the 300s ceiling. Real
# latency staggers arrivals, which is why production sees no such timeouts.
#
# A WHOLE second is used rather than 1.05 so virtual timestamps stay exactly
# representable. The limiter's own wait is computed as `oldest + 60 - now`, so
# under virtual time admissions land exactly 60.000s apart, and `_prune`'s
# `ts <= now - 60.0` then turns on float representation error: 549.45 - 60.0
# is not bit-identical to the stored 489.45, so an entry that should expire can
# survive. Real runs use time.monotonic() at nanosecond resolution and never
# align on that boundary, so this is a property of the simulated clock, not of
# production. Keeping the arithmetic exact tests the limiter rather than IEEE
# 754.
MEASURED_LATENCY_S = 1.0


class Recording(GeneralLlm):
    def __init__(self, model, clock, fail=False, limiter_key=None, latency=0.0):
        super().__init__(model=model, allowed_tries=1)
        self.clock = clock
        self.fail = fail
        self.latency = latency
        self.call_times = []
        if limiter_key is not None:
            self.limiter_key = limiter_key

    async def invoke(self, prompt, system_prompt=None):
        self.call_times.append(self.clock.time())
        if self.fail:
            raise RuntimeError("simulated provider failure")
        if self.latency:
            await self.clock.sleep(self.latency)
        return "TEXT:" + getattr(self, "limiter_key", self.model)


def install(clock, keys):
    rl.reset_registry()
    for key in keys:
        rl._REGISTRY[key] = rl.ProviderRateLimiter(
            key, limits=rl.limits_for(key), clock=clock.time, sleep=clock.sleep
        )


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def worst_window(times):
    """Occupancy of the window the limiter enforces: (x-60, x]."""
    ordered = sorted(times)
    worst = 0
    for i, x in enumerate(ordered):
        lo = bisect.bisect_right(ordered, x - 60.0)
        worst = max(worst, i - lo + 1)
    return worst


# ---------------------------------------------------------------- identity

class LimiterIdentityTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_a_backend_without_limiter_key_uses_the_model_as_before(self):
        """REQUIREMENT 1: today's behaviour must be untouched."""
        clock = Clock()
        plain = Recording(GEMINI, clock)
        self.assertFalse(hasattr(plain, "limiter_key"))
        chain = FallbackLlm([plain])
        run(chain.invoke("x"))
        self.assertEqual(rl.registry_size(), 1)
        self.assertIn(GEMINI, rl._REGISTRY)

    def test_distinct_limiter_keys_get_distinct_limiters(self):
        """REQUIREMENT 2."""
        clock = Clock()
        a = Recording(GEMINI, clock, limiter_key=rl.GEMINI_BUCKET_KEYS[0])
        b = Recording(GEMINI, clock, limiter_key=rl.GEMINI_BUCKET_KEYS[1])
        run(FallbackLlm([a]).invoke("x"))
        run(FallbackLlm([b]).invoke("x"))
        self.assertEqual(rl.registry_size(), 2)
        self.assertIsNot(
            rl.get_limiter(rl.GEMINI_BUCKET_KEYS[0]),
            rl.get_limiter(rl.GEMINI_BUCKET_KEYS[1]),
        )

    def test_same_model_different_credentials_do_not_share_a_limiter(self):
        """The trap this whole design exists to avoid: two keys, one bucket."""
        first = rl.get_limiter(rl.GEMINI_BUCKET_KEYS[0])
        second = rl.get_limiter(rl.GEMINI_BUCKET_KEYS[1])
        self.assertIsNot(first, second)
        self.assertNotEqual(first.model, second.model)


class ExplicitLimitsTests(unittest.TestCase):
    """REQUIREMENT 3 / safety requirement 4. The highest-severity property."""

    def test_every_registered_bucket_has_an_explicit_rpm(self):
        for key in rl.GEMINI_BUCKET_KEYS:
            limits = rl.limits_for(key)
            self.assertIsNotNone(
                limits.requests_per_minute,
                "bucket {0!r} has no explicit limit, so its limiter would be "
                "created UNTHROTTLED".format(key),
            )
            self.assertEqual(limits.requests_per_minute, 15.0)

    def test_every_bucket_is_present_in_default_limits(self):
        for key in rl.GEMINI_BUCKET_KEYS:
            self.assertIn(key, rl.DEFAULT_LIMITS)

    def test_an_unregistered_bucket_key_would_be_unthrottled(self):
        """Proves the hazard is real, so the guard above is not decorative."""
        rogue = GEMINI + "#not-registered"
        self.assertNotIn(rogue, rl.DEFAULT_LIMITS)
        self.assertIsNone(rl.limits_for(rogue).requests_per_minute)

    def test_bucket_zero_is_the_bare_model(self):
        """Keeps the single-key registry identical to the pre-bucket one."""
        self.assertEqual(rl.GEMINI_BUCKET_KEYS[0], GEMINI)


# --------------------------------------------------------------- balancing

def build(clock, n_buckets, fail_indexes=(), with_groq=True, latency=0.0):
    keys = list(rl.GEMINI_BUCKET_KEYS[:n_buckets])
    install(clock, keys + [OPENROUTER, GROQ])
    openrouter = Recording(OPENROUTER, clock, fail=True)
    groq = Recording(GROQ, clock, latency=latency)
    gems = [
        Recording(GEMINI, clock, fail=(i in fail_indexes), limiter_key=k,
                  latency=latency)
        for i, k in enumerate(keys)
    ]
    chains = [
        FallbackLlm([openrouter, g] + ([groq] if with_groq else []))
        for g in gems
    ]
    return BalancedLlm(chains, keys), gems, groq, openrouter


class BalancingTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_calls_are_spread_across_every_bucket(self):
        """REQUIREMENT 4."""
        clock = Clock()
        balanced, gems, _groq, _orr = build(clock, 4)

        async def scenario():
            for _ in range(40):
                await balanced.invoke("x")

        run(scenario())
        counts = [len(g.call_times) for g in gems]
        self.assertEqual(sum(counts), 40)
        for i, count in enumerate(counts):
            self.assertGreater(count, 0, "bucket {0} received nothing".format(i))
        self.assertLessEqual(
            max(counts) - min(counts), 1,
            "least-loaded selection should stay within one call: {0}".format(counts),
        )

    def test_no_bucket_exceeds_its_measured_quota(self):
        """REQUIREMENT 5."""
        clock = Clock()
        balanced, gems, _groq, _orr = build(clock, 4)

        async def scenario():
            await asyncio.gather(*[balanced.invoke("x") for _ in range(120)])

        run(scenario())
        quota = int(rl.limits_for(rl.GEMINI_BUCKET_KEYS[0]).requests_per_minute)
        for i, gem in enumerate(gems):
            self.assertLessEqual(
                worst_window(gem.call_times), quota,
                "bucket {0} admitted {1} calls in one window, quota {2}".format(
                    i, worst_window(gem.call_times), quota),
            )

    def test_one_bucket_behaves_exactly_like_no_balancing(self):
        """REQUIREMENT 6, at the object level."""
        clock = Clock()
        balanced, gems, _groq, _orr = build(clock, 1)

        async def scenario():
            for _ in range(10):
                await balanced.invoke("x")

        run(scenario())
        self.assertEqual(len(gems[0].call_times), 10)
        self.assertEqual(rl.registry_size(), 3)  # bucket0 + openrouter + groq

    def test_a_dead_secondary_credential_still_yields_text(self):
        """REQUIREMENT 7. Bucket 1 is dead and its chain has no Groq link -
        the parser's exact shape - so recovery must come from another bucket."""
        clock = Clock()
        balanced, gems, _groq, _orr = build(clock, 4, fail_indexes={1},
                                            with_groq=False)

        async def scenario():
            return [await balanced.invoke("x") for _ in range(20)]

        results = run(scenario())
        self.assertEqual(len(results), 20)
        self.assertTrue(all(isinstance(r, str) and r for r in results))
        self.assertGreater(gems[1].call_times, [], "the dead bucket was tried")
        healthy = sum(len(gems[i].call_times) for i in (0, 2, 3))
        self.assertEqual(healthy + 0, sum(len(g.call_times) for g in gems)
                         - len(gems[1].call_times))

    def test_every_chain_keeps_the_production_order(self):
        """Balancing chooses between chains; it must never reorder one."""
        clock = Clock()
        balanced, _gems, _groq, _orr = build(clock, 4)
        for chain in balanced._chains:
            models = [b.model for b in chain._backends]
            self.assertEqual(models, [OPENROUTER, GEMINI, GROQ])

    def test_a_healthy_bucket_never_reaches_groq(self):
        clock = Clock()
        balanced, _gems, groq, _orr = build(clock, 4)

        async def scenario():
            for _ in range(12):
                await balanced.invoke("x")

        run(scenario())
        self.assertEqual(len(groq.call_times), 0)

    def test_all_chains_failing_raises_rather_than_inventing_text(self):
        clock = Clock()
        balanced, _gems, _groq, _orr = build(clock, 4, fail_indexes={0, 1, 2, 3},
                                             with_groq=False)
        with self.assertRaises(RuntimeError):
            run(balanced.invoke("x"))

    def test_construction_rejects_a_chain_without_a_bucket_key(self):
        clock = Clock()
        balanced, _gems, _groq, _orr = build(clock, 2)
        with self.assertRaises(ValueError):
            BalancedLlm(balanced._chains, [rl.GEMINI_BUCKET_KEYS[0]])
        with self.assertRaises(ValueError):
            BalancedLlm([], [])


class ObservabilityTests(unittest.TestCase):
    """Four buckets share one model string, so `provider=` cannot tell them
    apart. Without `bucket=` in the log, per-bucket distribution and RPM are
    not measurable from a real run at all."""

    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_the_bucket_is_named_in_the_success_log(self):
        import logging

        clock = Clock()
        backend = Recording(GEMINI, clock, limiter_key=rl.GEMINI_BUCKET_KEYS[2])
        chain = FallbackLlm([backend])
        with self.assertLogs("backtest.fallback_llm", level=logging.INFO) as caught:
            run(chain.invoke("x"))
        joined = "\n".join(caught.output)
        self.assertIn("bucket=" + rl.GEMINI_BUCKET_KEYS[2], joined)

    def test_a_backend_without_a_bucket_reports_its_model(self):
        import logging

        clock = Clock()
        chain = FallbackLlm([Recording(GEMINI, clock)])
        with self.assertLogs("backtest.fallback_llm", level=logging.INFO) as caught:
            run(chain.invoke("x"))
        self.assertIn("bucket=" + GEMINI, "\n".join(caught.output))

    def test_the_log_never_carries_credential_material(self):
        import logging
        import os

        os.environ["TMP_LOG_KEY"] = "super-secret-value"
        try:
            clock = Clock()
            backend = bucket_backend(GEMINI, "TMP_LOG_KEY",
                                     rl.GEMINI_BUCKET_KEYS[1], 180, 1)
            backend.invoke = Recording(
                GEMINI, clock, limiter_key=rl.GEMINI_BUCKET_KEYS[1]).invoke
            chain = FallbackLlm([backend])
            with self.assertLogs("backtest.fallback_llm", level=logging.INFO) as caught:
                run(chain.invoke("x"))
            self.assertNotIn("super-secret-value", "\n".join(caught.output))
        finally:
            os.environ.pop("TMP_LOG_KEY", None)


class BucketBackendTests(unittest.TestCase):
    def test_limiter_key_is_an_attribute_not_a_litellm_kwarg(self):
        """A limiter_key= kwarg would be forwarded to acompletion as a request
        parameter, which is why it is attached after construction."""
        backend = bucket_backend(GEMINI, "NO_SUCH_ENV", rl.GEMINI_BUCKET_KEYS[1],
                                 180, 1)
        self.assertEqual(backend.limiter_key, rl.GEMINI_BUCKET_KEYS[1])
        self.assertNotIn("limiter_key", backend.litellm_kwargs)

    def test_an_absent_credential_does_not_inject_an_empty_api_key(self):
        """Safety requirement 2: falls through to litellm's own env lookup."""
        backend = bucket_backend(GEMINI, "NO_SUCH_ENV", rl.GEMINI_BUCKET_KEYS[1],
                                 180, 1)
        self.assertNotIn("api_key", backend.litellm_kwargs)

    def test_a_present_credential_is_passed_but_never_logged(self):
        import os

        os.environ["TMP_BUCKET_KEY"] = "secret-value"
        try:
            backend = bucket_backend(GEMINI, "TMP_BUCKET_KEY",
                                     rl.GEMINI_BUCKET_KEYS[2], 180, 1)
            self.assertEqual(backend.litellm_kwargs["api_key"], "secret-value")
            self.assertNotIn("secret-value", repr(backend.limiter_key))
        finally:
            os.environ.pop("TMP_BUCKET_KEY", None)


class MiniBenchBurstTests(unittest.TestCase):
    """REQUIREMENT 8: the whole MiniBench shape, balanced over four buckets.

    Reproduces the real call structure: 60 questions, research serialised by
    main.py's Semaphore(1), then 5 concurrent predictions each costing one
    reasoning call plus two parser samples -- 17 calls per question, the figure
    validated against the live E2E run (9 questions, 154 attempts predicted 153).
    """

    QUESTIONS = 60
    PREDICTIONS = 5
    PARSER_SAMPLES = 2

    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_every_prediction_survives_a_sixty_question_burst(self):
        clock = ScheduledClock()
        balanced, gems, _groq, _orr = build(clock, 4, latency=MEASURED_LATENCY_S)
        results = {}
        timeouts = {"n": 0}
        quota = int(rl.limits_for(rl.GEMINI_BUCKET_KEYS[0]).requests_per_minute)
        original = rl.ProviderRateLimiter.acquire
        original_prune = rl.ProviderRateLimiter._prune
        occupancy = {}

        async def counting(limiter, tokens=None):
            try:
                return await original(limiter, tokens)
            except rl.RateLimitTimeout:
                timeouts["n"] += 1
                raise

        def watched_prune(limiter, now):
            # Runs INSIDE the limiter's lock, immediately before the admit
            # decision, so this is the true occupancy of the window the limiter
            # enforces. Sampling the backend's own call times instead would be
            # racy: virtual time can advance between admission and invocation.
            original_prune(limiter, now)
            if limiter.limits.requests_per_minute is not None:
                occupancy[limiter.model] = max(
                    occupancy.get(limiter.model, 0), len(limiter._requests)
                )

        rl.ProviderRateLimiter.acquire = counting
        rl.ProviderRateLimiter._prune = watched_prune
        try:
            async def prediction(question, index):
                reasoning = await balanced.invoke("reason")
                for _ in range(self.PARSER_SAMPLES):
                    await balanced.invoke("parse")
                results[(question, index)] = reasoning

            async def one_question(question, semaphore):
                async with semaphore:
                    await balanced.invoke("research")
                await balanced.invoke("summarize")
                await asyncio.gather(*[
                    prediction(question, i) for i in range(self.PREDICTIONS)
                ])

            async def burst():
                # Both the gather and the Semaphore must be created INSIDE the
                # loop: asyncio primitives bind to the running loop, and one
                # built beforehand belongs to another. This is the same hazard
                # rate_limiter._get_lock() documents for its own lock.
                semaphore = asyncio.Semaphore(1)  # main.py:129, unchanged
                await asyncio.gather(*[
                    one_question(q, semaphore) for q in range(self.QUESTIONS)
                ])

            run_scheduled(clock, burst)
        finally:
            rl.ProviderRateLimiter.acquire = original
            rl.ProviderRateLimiter._prune = original_prune

        expected_predictions = self.QUESTIONS * self.PREDICTIONS
        self.assertEqual(len(results), expected_predictions,
                         "predictions must not be lost to pacing")
        for question in range(self.QUESTIONS):
            got = [k for k in results if k[0] == question]
            self.assertEqual(len(got), self.PREDICTIONS,
                             "question {0} lost a prediction".format(question))

        self.assertEqual(timeouts["n"], 0, "no call should hit the wait ceiling")

        expected_calls = self.QUESTIONS * (
            2 + self.PREDICTIONS * (1 + self.PARSER_SAMPLES)
        )
        served = sum(len(g.call_times) for g in gems)
        self.assertEqual(served, expected_calls,
                         "waiting must not duplicate or drop a call")

        for i, gem in enumerate(gems):
            self.assertGreater(len(gem.call_times), 0,
                               "bucket {0} carried none of the burst".format(i))
        self.assertLessEqual(
            max(occupancy.values()), quota,
            "a bucket held {0} requests inside its 60s window, quota {1}".format(
                max(occupancy.values()), quota),
        )


if __name__ == "__main__":
    unittest.main()
