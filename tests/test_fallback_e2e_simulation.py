"""Simulated MiniBench burst: 49 questions x 5 predictions, no network.

This reproduces the shape of the run that failed for real (workflow
32295883751) and asserts the two properties that must survive the fix:

  1. PACING     no burst exceeds the providers' measured meters
  2. SEMANTICS  every one of the 245 predictions still exists, each served by
                exactly one provider, and a rate-limit wait never turns into a
                lost prediction or an extra forecast

The bot's real concurrency shape is reproduced faithfully: questions run
concurrently (forecast_bot.py:241) and, inside each, the 5 predictions run
concurrently too (forecast_bot.py:480). Time is virtual, so 49 questions'
worth of pacing is verified in milliseconds.
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from collections import Counter

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

QUESTIONS = 49                      # the largest observed MiniBench day-one drop
PREDICTIONS_PER_QUESTION = 5        # main.py:675 - unchanged
PARSER_SAMPLES = 2                  # main.py:130 - unchanged


class VirtualClock:
    """Shared virtual time for every limiter in one scenario."""

    def __init__(self):
        self.now = 0.0

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)      # yield so other tasks can progress


class RecordingBackend(GeneralLlm):
    """A provider that records call times and can be told to fail."""

    def __init__(self, model, clock, fail=False):
        super().__init__(model=model, allowed_tries=1)
        self.clock = clock
        self.fail = fail
        self.call_times: list[float] = []

    async def invoke(self, prompt, system_prompt=None):
        self.call_times.append(self.clock.time())
        if self.fail:
            raise RuntimeError("simulated provider failure")
        return "REASONING"


def install_clock(clock, models):
    """Rebuild the registry so every limiter shares the virtual clock."""
    rl.reset_registry()
    for model in models:
        rl._REGISTRY[model] = rl.ProviderRateLimiter(
            model, limits=rl.limits_for(model), clock=clock.time, sleep=clock.sleep
        )


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class BurstSimulationTests(unittest.TestCase):
    """OpenRouter dead, exactly as in the real E2E run."""

    def setUp(self):
        self.clock = VirtualClock()
        install_clock(self.clock, [OPENROUTER, GEMINI, GROQ])
        self.openrouter = RecordingBackend(OPENROUTER, self.clock, fail=True)
        self.gemini = RecordingBackend(GEMINI, self.clock)
        self.groq = RecordingBackend(GROQ, self.clock)
        # One chain per role, as pin_models generates. All share the registry.
        self.chain = FallbackLlm([self.openrouter, self.gemini, self.groq])
        self.results: dict[tuple[int, int], str] = {}

    async def _one_prediction(self, question_id: int, index: int):
        """1 reasoning call + PARSER_SAMPLES parser calls, matching
        structure_output(num_validation_samples=2)."""
        reasoning = await self.chain.invoke("reason about {0}".format(question_id))
        for _ in range(PARSER_SAMPLES):
            await self.chain.invoke("parse")
        self.results[(question_id, index)] = reasoning

    async def _one_question(self, question_id: int):
        await self.chain.invoke("research")          # researcher
        await self.chain.invoke("summarize")         # summarizer
        await asyncio.gather(*[                      # 5 predictions, concurrent
            self._one_prediction(question_id, i)
            for i in range(PREDICTIONS_PER_QUESTION)
        ])

    async def _whole_burst(self, n_questions: int):
        await asyncio.gather(*[self._one_question(q) for q in range(n_questions)])

    def test_every_prediction_survives_the_burst(self):
        run(self._whole_burst(QUESTIONS))
        expected = QUESTIONS * PREDICTIONS_PER_QUESTION
        self.assertEqual(len(self.results), expected,
                         "245 predictions requested, 245 must exist")
        for question_id in range(QUESTIONS):
            got = [k for k in self.results if k[0] == question_id]
            self.assertEqual(len(got), PREDICTIONS_PER_QUESTION,
                             "question {0} lost a prediction".format(question_id))

    def test_no_window_ever_exceeds_the_measured_gemini_quota(self):
        run(self._whole_burst(QUESTIONS))
        times = sorted(self.gemini.call_times)
        rpm = int(rl.limits_for(GEMINI).requests_per_minute)
        worst = 0
        for i, start in enumerate(times):
            in_window = sum(1 for t in times[i:] if t < start + 60.0)
            worst = max(worst, in_window)
        self.assertLessEqual(worst, rpm,
                             "saw {0} Gemini calls in one minute, quota is {1}".format(worst, rpm))

    def test_the_original_failure_shape_is_gone(self):
        """The real run peaked at 30 calls in one second."""
        run(self._whole_burst(QUESTIONS))
        per_second = Counter(int(t) for t in self.gemini.call_times)
        self.assertLessEqual(max(per_second.values()), 15,
                             "still bursting: {0}".format(per_second.most_common(3)))

    def test_a_delayed_call_is_still_exactly_one_call(self):
        run(self._whole_burst(QUESTIONS))
        calls_per_prediction = 1 + PARSER_SAMPLES
        expected_calls = QUESTIONS * (2 + PREDICTIONS_PER_QUESTION * calls_per_prediction)
        served = len(self.gemini.call_times) + len(self.groq.call_times)
        self.assertEqual(served, expected_calls,
                         "waiting must not duplicate or drop calls")

    def test_openrouter_stays_first_in_line_even_when_dead(self):
        run(self._whole_burst(5))
        self.assertGreater(len(self.openrouter.call_times), 0,
                           "the primary must still be attempted first")


class SemanticsUnderPacingTests(unittest.TestCase):
    def setUp(self):
        self.clock = VirtualClock()
        install_clock(self.clock, [OPENROUTER, GEMINI, GROQ])

    def test_a_mixed_provider_question_is_still_five_predictions(self):
        """3 OpenRouter, 1 Gemini, 1 Groq -> five predictions, not eight."""
        openrouter = RecordingBackend(OPENROUTER, self.clock)
        gemini = RecordingBackend(GEMINI, self.clock)
        groq = RecordingBackend(GROQ, self.clock)

        calls = {"n": 0}
        original = openrouter.invoke

        async def flaky(prompt, system_prompt=None):
            calls["n"] += 1
            if calls["n"] in (4, 5):          # two of the five fail over
                raise RuntimeError("429")
            return await original(prompt, system_prompt)

        openrouter.invoke = flaky
        gemini_fails_once = {"n": 0}
        gemini_original = gemini.invoke

        async def gemini_flaky(prompt, system_prompt=None):
            gemini_fails_once["n"] += 1
            if gemini_fails_once["n"] == 2:   # so one lands on Groq
                raise RuntimeError("429")
            return await gemini_original(prompt, system_prompt)

        gemini.invoke = gemini_flaky
        chain = FallbackLlm([openrouter, gemini, groq])

        async def scenario():
            return await asyncio.gather(*[chain.invoke("predict") for _ in range(5)])

        predictions = run(scenario())
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, str) for p in predictions))
        self.assertEqual(len(groq.call_times), 1, "exactly one reached the third link")

    def test_rate_limit_waits_are_counted_as_infrastructure_only(self):
        gemini = RecordingBackend(GEMINI, self.clock)
        chain = FallbackLlm([gemini])

        async def scenario():
            for _ in range(20):               # over the 15/min quota
                await chain.invoke("x")

        run(scenario())
        snapshot = rl.COUNTERS.snapshot()
        self.assertEqual(snapshot["llm_success_total"], 20)
        self.assertGreater(snapshot["rate_limit_waits_total"], 0)
        # None of these is, or feeds, a forecast count.
        self.assertNotIn("forecast", " ".join(snapshot))

    def test_a_fully_exhausted_chain_yields_no_text_at_all(self):
        dead = [RecordingBackend(m, self.clock, fail=True)
                for m in (OPENROUTER, GEMINI, GROQ)]
        chain = FallbackLlm(dead)
        with self.assertRaises(RuntimeError):
            run(chain.invoke("x"))
        # No string returned => no prediction => nothing for the bot to
        # aggregate, publish or count.


if __name__ == "__main__":
    unittest.main()
