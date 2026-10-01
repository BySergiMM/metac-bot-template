"""BalancedLlm: calls spread over several fallback chains, chosen at admission.

Offline throughout: no network, no credentials, virtual time.

Production uses the class for the forecaster ensemble, one chain per primary
model (backtest/pin_models.py). The chains below stand in for that: each one
leads with its own model, and every model has its own rate-limiter window, so
"load" means what it means in production.

The properties under test are the ones that make this safe rather than merely
faster. In order of how badly they fail if broken:

  1. a key with no explicit limits is UNTHROTTLED, because limits_for() falls
     back to ProviderLimits(). A model missing from DEFAULT_LIMITS would look
     controlled and rate-limit nothing.
  2. no limiter window may be driven past its quota, and waiting must neither
     drop nor duplicate a call.
  3. a dead chain must not cost a prediction.
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
from backtest.balanced_llm import BalancedLlm  # noqa: E402
from backtest.fallback_llm import FallbackLlm  # noqa: E402
from forecasting_tools.ai_models.general_llm import GeneralLlm  # noqa: E402

GEMINI = "gemini/gemini-3.5-flash-lite"
GROQ = "groq/openai/gpt-oss-120b"
OPENROUTER = "openrouter/nvidia/nemotron-3.5-lightning:free"

# Four chains, each leading with a model of its own. The names are deliberately
# not real providers, so nothing here can be mistaken for production config, and
# they are NOT in rate_limiter.DEFAULT_LIMITS: each test installs its own
# limiter for them, with an explicit quota.
MODEL_KEYS = ["test/model-{0}".format(i) for i in range(4)]
QUOTA_RPM = 15.0


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
    def __init__(self, model, clock, fail=False, latency=0.0):
        super().__init__(model=model, allowed_tries=1)
        self.clock = clock
        self.fail = fail
        self.latency = latency
        self.call_times = []

    async def invoke(self, prompt, system_prompt=None):
        self.call_times.append(self.clock.time())
        if self.fail:
            raise RuntimeError("simulated provider failure")
        if self.latency:
            await self.clock.sleep(self.latency)
        return "TEXT:" + self.model


def install(clock, limits):
    """Register one limiter per key, on the virtual clock.

    ``limits`` maps a limiter key to the ProviderLimits it should enforce."""
    rl.reset_registry()
    for key, key_limits in limits.items():
        rl._REGISTRY[key] = rl.ProviderRateLimiter(
            key, limits=key_limits, clock=clock.time, sleep=clock.sleep
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

    def test_a_backend_is_rate_limited_under_its_model_string(self):
        clock = Clock()
        plain = Recording(GEMINI, clock)
        chain = FallbackLlm([plain])
        run(chain.invoke("x"))
        self.assertEqual(rl.registry_size(), 1)
        self.assertIn(GEMINI, rl._REGISTRY)


class ExplicitLimitsTests(unittest.TestCase):
    """The highest-severity property: a key nobody registered is not limited."""

    def test_an_unregistered_key_would_be_unthrottled(self):
        """Proves the hazard is real. pin_models checks every key it emits
        against this registry for exactly this reason."""
        rogue = "test/not-registered"
        self.assertNotIn(rogue, rl.DEFAULT_LIMITS)
        self.assertIsNone(rl.limits_for(rogue).requests_per_minute)
        self.assertIsNone(rl.limits_for(rogue).tokens_per_minute)


# --------------------------------------------------------------- balancing

def build(clock, n_chains, fail_indexes=(), with_groq=True, latency=0.0):
    """``n_chains`` chains, each [OpenRouter, its own model, Groq?].

    OpenRouter always fails, so the chain's own model is what serves the call;
    that model's limiter (``QUOTA_RPM``) is the one BalancedLlm measures."""
    keys = MODEL_KEYS[:n_chains]
    limits = {key: rl.ProviderLimits(requests_per_minute=QUOTA_RPM) for key in keys}
    limits[OPENROUTER] = rl.limits_for(OPENROUTER)
    limits[GROQ] = rl.limits_for(GROQ)
    install(clock, limits)
    openrouter = Recording(OPENROUTER, clock, fail=True)
    groq = Recording(GROQ, clock, latency=latency)
    models = [
        Recording(key, clock, fail=(i in fail_indexes), latency=latency)
        for i, key in enumerate(keys)
    ]
    chains = [
        FallbackLlm([openrouter, m] + ([groq] if with_groq else []))
        for m in models
    ]
    return BalancedLlm(chains, keys), models, groq, openrouter


class BalancingTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_calls_are_spread_across_every_chain(self):
        clock = Clock()
        balanced, models, _groq, _orr = build(clock, 4)

        async def scenario():
            for _ in range(40):
                await balanced.invoke("x")

        run(scenario())
        counts = [len(m.call_times) for m in models]
        self.assertEqual(sum(counts), 40)
        for i, count in enumerate(counts):
            self.assertGreater(count, 0, "chain {0} received nothing".format(i))
        self.assertLessEqual(
            max(counts) - min(counts), 1,
            "least-loaded selection should stay within one call: {0}".format(counts),
        )

    def test_no_chain_exceeds_its_limiters_quota(self):
        clock = Clock()
        balanced, models, _groq, _orr = build(clock, 4)

        async def scenario():
            await asyncio.gather(*[balanced.invoke("x") for _ in range(120)])

        run(scenario())
        quota = int(QUOTA_RPM)
        for i, model in enumerate(models):
            self.assertLessEqual(
                worst_window(model.call_times), quota,
                "chain {0} admitted {1} calls in one window, quota {2}".format(
                    i, worst_window(model.call_times), quota),
            )

    def test_one_chain_behaves_exactly_like_no_balancing(self):
        clock = Clock()
        balanced, models, _groq, _orr = build(clock, 1)

        async def scenario():
            for _ in range(10):
                await balanced.invoke("x")

        run(scenario())
        self.assertEqual(len(models[0].call_times), 10)
        self.assertEqual(rl.registry_size(), 3)  # the model + openrouter + groq

    def test_a_dead_chain_still_yields_text(self):
        """Chain 1 is dead and has no Groq link behind it, so recovery must
        come from another chain: a prediction must not be lost because one
        route is down."""
        clock = Clock()
        balanced, models, _groq, _orr = build(clock, 4, fail_indexes={1},
                                              with_groq=False)

        async def scenario():
            return [await balanced.invoke("x") for _ in range(20)]

        results = run(scenario())
        self.assertEqual(len(results), 20)
        self.assertTrue(all(isinstance(r, str) and r for r in results))
        self.assertGreater(len(models[1].call_times), 0, "the dead chain was tried")
        served_by_healthy = {r for r in results}
        self.assertNotIn("TEXT:" + MODEL_KEYS[1], served_by_healthy)

    def test_every_chain_keeps_its_own_order(self):
        """Balancing chooses between chains; it must never reorder one."""
        clock = Clock()
        balanced, _models, _groq, _orr = build(clock, 4)
        for chain, key in zip(balanced._chains, MODEL_KEYS):
            served = [b.model for b in chain._backends]
            self.assertEqual(served, [OPENROUTER, key, GROQ])

    def test_a_healthy_chain_never_reaches_groq(self):
        clock = Clock()
        balanced, _models, groq, _orr = build(clock, 4)

        async def scenario():
            for _ in range(12):
                await balanced.invoke("x")

        run(scenario())
        self.assertEqual(len(groq.call_times), 0)

    def test_all_chains_failing_raises_rather_than_inventing_text(self):
        clock = Clock()
        balanced, _models, _groq, _orr = build(clock, 4, fail_indexes={0, 1, 2, 3},
                                               with_groq=False)
        with self.assertRaises(RuntimeError):
            run(balanced.invoke("x"))

    def test_construction_rejects_a_chain_without_a_bucket_key(self):
        clock = Clock()
        balanced, _models, _groq, _orr = build(clock, 2)
        with self.assertRaises(ValueError):
            BalancedLlm(balanced._chains, [MODEL_KEYS[0]])
        with self.assertRaises(ValueError):
            BalancedLlm([], [])


class ObservabilityTests(unittest.TestCase):
    """FallbackLlm's log lines are the only per-call record of which backend
    served what, so their shape is pinned."""

    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_the_log_names_the_bucket_and_it_is_the_model(self):
        import logging

        clock = Clock()
        chain = FallbackLlm([Recording(GEMINI, clock)])
        with self.assertLogs("backtest.fallback_llm", level=logging.INFO) as caught:
            run(chain.invoke("x"))
        joined = "\n".join(caught.output)
        self.assertIn("provider=" + GEMINI, joined)
        self.assertIn("bucket=" + GEMINI, joined)

    def test_the_log_never_carries_credential_material(self):
        import logging

        clock = Clock()
        backend = Recording(GEMINI, clock)
        backend.litellm_kwargs["api_key"] = "super-secret-value"
        chain = FallbackLlm([backend])
        with self.assertLogs("backtest.fallback_llm", level=logging.INFO) as caught:
            run(chain.invoke("x"))
        self.assertNotIn("super-secret-value", "\n".join(caught.output))


class MiniBenchBurstTests(unittest.TestCase):
    """The whole MiniBench shape, balanced over four chains.

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
        balanced, models, _groq, _orr = build(clock, 4, latency=MEASURED_LATENCY_S)
        results = {}
        timeouts = {"n": 0}
        quota = int(QUOTA_RPM)
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
        served = sum(len(m.call_times) for m in models)
        self.assertEqual(served, expected_calls,
                         "waiting must not duplicate or drop a call")

        for i, model in enumerate(models):
            self.assertGreater(len(model.call_times), 0,
                               "chain {0} carried none of the burst".format(i))
        self.assertLessEqual(
            max(occupancy.values()), quota,
            "a limiter held {0} requests inside its 60s window, quota {1}".format(
                max(occupancy.values()), quota),
        )


if __name__ == "__main__":
    unittest.main()
