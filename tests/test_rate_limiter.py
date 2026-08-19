"""Rate limiting must fix pacing WITHOUT touching forecasting semantics.

The E2E run (workflow 32295883751) is the reference failure: ~185 calls in 18
seconds, peak 30 in one second, against Gemini's measured 15 requests/minute
and Groq's measured 8000 tokens/minute. Zero forecasts came out.

These tests pin down the two things that must both be true:
  1. calls are paced to the providers' real meters;
  2. no prediction, forecast, threshold or metric changes as a result.

Time is mocked throughout. A limiter that genuinely sleeps for minutes cannot
be tested, and a test that sleeps is a test nobody runs.
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest

# ------------------------------------------------------------- stub setup
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

        async def invoke(self, prompt, system_prompt=None):  # pragma: no cover
            raise NotImplementedError

    _gl.GeneralLlm = _StubGeneralLlm
    _ai.general_llm = _gl
    _pkg.ai_models = _ai
    sys.modules["forecasting_tools"] = _pkg
    sys.modules["forecasting_tools.ai_models"] = _ai
    sys.modules["forecasting_tools.ai_models.general_llm"] = _gl

from backtest import rate_limiter as rl  # noqa: E402
from backtest.fallback_llm import FallbackLlm  # noqa: E402
from forecasting_tools.ai_models.general_llm import GeneralLlm  # noqa: E402

GEMINI = "gemini/gemini-3.5-flash-lite"
GROQ = "groq/openai/gpt-oss-120b"
OPENROUTER = "openrouter/nvidia/nemotron-3.5-lightning:free"


class FakeClock:
    """Virtual time. `sleep` advances the clock instead of blocking, so a
    60-second window is exercised in microseconds and deterministically."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def run(coro):
    """Run one coroutine on a fresh loop and close it.

    Closing matters: leaked loops emit ResourceWarnings that bury real output.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def limiter(model, clock, **overrides):
    limits = rl.limits_for(model)
    for key, value in overrides.items():
        setattr(limits, key, value)
    return rl.ProviderRateLimiter(model, limits=limits, clock=clock.time, sleep=clock.sleep)


class RequestPacingTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def test_measured_gemini_quota_is_the_default(self):
        """15 req/min is not a guess: Gemini returned quotaValue "15" for
        GenerateRequestsPerMinutePerProjectPerModel-FreeTier."""
        self.assertEqual(rl.limits_for(GEMINI).requests_per_minute, 15.0)

    def test_first_fifteen_calls_pass_without_waiting(self):
        clock = FakeClock()
        lim = limiter(GEMINI, clock)
        waits = [run(lim.acquire()) for _ in range(15)]
        self.assertEqual(waits, [0.0] * 15)
        self.assertEqual(clock.sleeps, [])

    def test_the_sixteenth_call_waits_for_the_window(self):
        clock = FakeClock()
        lim = limiter(GEMINI, clock)
        for _ in range(15):
            run(lim.acquire())
        waited = run(lim.acquire())
        self.assertAlmostEqual(waited, 60.0, places=6)
        self.assertEqual(len(clock.sleeps), 1)

    def test_a_slot_frees_as_the_window_slides(self):
        clock = FakeClock()
        lim = limiter(GEMINI, clock)
        for _ in range(15):
            run(lim.acquire())
        clock.now += 61.0          # the whole first batch ages out
        self.assertEqual(run(lim.acquire()), 0.0)

    def test_the_burst_that_broke_the_e2e_run_is_paced(self):
        """30 calls in one second was the observed peak. Under the limiter the
        same 30 calls still all happen - they just take two windows."""
        clock = FakeClock()
        lim = limiter(GEMINI, clock)
        for _ in range(30):
            run(lim.acquire())
        self.assertEqual(len(lim._requests) + 0, 15, "window holds at most the quota")
        self.assertGreaterEqual(clock.now - 1000.0, 60.0, "the excess was deferred, not dropped")

    def test_openrouter_is_not_slowed_down(self):
        """Its cap is per DAY (50), never per minute - 63 run logs, not one
        per-minute error. Pacing it would change working behaviour."""
        clock = FakeClock()
        lim = limiter(OPENROUTER, clock)
        waits = [run(lim.acquire()) for _ in range(100)]
        self.assertEqual(set(waits), {0.0})
        self.assertEqual(clock.sleeps, [])


class TokenPacingTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def test_measured_groq_quota_is_tokens_not_requests(self):
        """Groq's 429 said "tokens per minute (TPM): Limit 8000"."""
        limits = rl.limits_for(GROQ)
        self.assertEqual(limits.tokens_per_minute, 8000.0)
        self.assertIsNone(limits.requests_per_minute)

    def test_token_budget_paces_calls(self):
        clock = FakeClock()
        lim = limiter(GROQ, clock)          # 8000 TPM / 2500 per call -> 3 fit
        self.assertEqual([run(lim.acquire()) for _ in range(3)], [0.0, 0.0, 0.0])
        self.assertGreater(run(lim.acquire()), 0.0, "the fourth must wait")

    def test_estimate_is_evidence_based(self):
        """Groq reported "Requested 2396" for a real forecasting prompt."""
        self.assertGreaterEqual(rl.limits_for(GROQ).estimated_tokens_per_call, 2396)

    def test_an_oversized_request_is_admitted_rather_than_stalled_forever(self):
        clock = FakeClock()
        lim = limiter(GROQ, clock, tokens_per_minute=1000.0)
        # 2500 > the entire budget: waiting could never help, so let the
        # provider decide. A rejection there is a provider failure the chain
        # handles; an infinite wait would lose the call outright.
        self.assertEqual(run(lim.acquire()), 0.0)


class SharingTests(unittest.TestCase):
    """The failure mode this design exists to prevent."""

    def setUp(self):
        rl.reset_registry()

    def test_the_same_model_always_yields_the_same_limiter(self):
        self.assertIs(rl.get_limiter(GEMINI), rl.get_limiter(GEMINI))

    def test_different_models_get_different_limiters(self):
        self.assertIsNot(rl.get_limiter(GEMINI), rl.get_limiter(GROQ))

    def test_two_fallback_chains_on_the_same_model_share_one_limiter(self):
        """get_llm() returns one instance per ROLE - four of them, each with
        its own GeneralLlm for Gemini. Per-instance limiters would allow 4x the
        quota while looking like a control."""
        chain_a = FallbackLlm([GeneralLlm(model=OPENROUTER), GeneralLlm(model=GEMINI)])
        chain_b = FallbackLlm([GeneralLlm(model=OPENROUTER), GeneralLlm(model=GEMINI)])
        self.assertIsNot(chain_a, chain_b)
        self.assertIs(
            rl.get_limiter(chain_a._backends[1].model),
            rl.get_limiter(chain_b._backends[1].model),
        )

    def test_a_limiter_is_not_created_per_call(self):
        for _ in range(50):
            rl.get_limiter(GEMINI)
        self.assertEqual(rl.registry_size(), 1)

    def test_four_roles_produce_one_limiter_per_model_not_four(self):
        for _role in ("default", "summarizer", "researcher", "parser"):
            FallbackLlm([GeneralLlm(model=OPENROUTER), GeneralLlm(model=GEMINI)])
            rl.get_limiter(OPENROUTER)
            rl.get_limiter(GEMINI)
        self.assertEqual(rl.registry_size(), 2)


class FairnessTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def test_waiting_calls_are_admitted_in_arrival_order(self):
        """A prediction must not starve behind newer ones."""
        clock = FakeClock()
        lim = limiter(GEMINI, clock, requests_per_minute=2.0)
        order: list[int] = []

        async def scenario():
            async def caller(index):
                await lim.acquire()
                order.append(index)
            await asyncio.gather(*[caller(i) for i in range(6)])

        run(scenario())
        self.assertEqual(order, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(order), 6, "every caller was admitted; none dropped")

    def test_a_saturated_provider_gives_up_instead_of_blocking_forever(self):
        clock = FakeClock()
        lim = limiter(GEMINI, clock, requests_per_minute=1.0, max_wait_seconds=10.0)
        run(lim.acquire())
        with self.assertRaises(rl.RateLimitTimeout):
            run(lim.acquire())


class CountersAreInfrastructureOnlyTests(unittest.TestCase):
    def setUp(self):
        rl.reset_registry()

    def test_counters_expose_only_infrastructure_fields(self):
        self.assertEqual(
            sorted(rl.COUNTERS.snapshot()),
            ["llm_attempts_total", "llm_fallback_total", "llm_success_total",
             "rate_limit_wait_seconds_total", "rate_limit_waits_total"],
        )

    def test_no_counter_mentions_forecasts_or_predictions(self):
        for name in rl.COUNTERS.snapshot():
            for forbidden in ("forecast", "prediction", "question"):
                self.assertNotIn(forbidden, name)

    def test_the_official_metric_module_never_imports_the_counters(self):
        """n_questions_with_own_forecast must stay derived from Metaculus'
        records, structurally unable to see infrastructure counters."""
        import pathlib
        src = (pathlib.Path(__file__).parent.parent
               / "research" / "fetch_own_track_record.py").read_text()
        for forbidden in ("rate_limiter", "COUNTERS", "llm_attempts_total"):
            self.assertNotIn(forbidden, src)


class MultipleEventLoopTests(unittest.TestCase):
    """main.py calls asyncio.run() twice - tournament then MiniBench
    (main.py:708 and :713). A limiter whose lock was bound to the first loop
    raises "got Future attached to a different loop" for every call in the
    second, which would have silently wiped out the entire MiniBench phase.
    Found by the 49-question simulation before deployment."""

    def setUp(self):
        rl.reset_registry()

    def test_a_limiter_survives_a_second_event_loop(self):
        lim = rl.get_limiter(GEMINI)

        async def use():
            return await lim.acquire()

        # The assertion is "does not raise". A real clock makes an unimpeded
        # acquire return a microsecond rather than exactly zero.
        self.assertLess(run(use()), 0.1)    # loop #1, as in the tournament pass
        self.assertLess(run(use()), 0.1)    # loop #2, as in the MiniBench pass
        self.assertEqual(rl.COUNTERS.rate_limit_waits_total, 0,
                         "neither call actually waited, so neither may be counted")

    def test_window_state_persists_across_loops(self):
        """The provider's quota does not reset because our process started a
        new loop, so neither may the window."""
        clock = FakeClock()
        lim = limiter(GEMINI, clock)

        async def use():
            return await lim.acquire()

        for _ in range(15):
            run(use())
        waited = run(use())                 # 16th call, now in a fresh loop
        self.assertAlmostEqual(waited, 60.0, places=6)



if __name__ == "__main__":
    unittest.main()
