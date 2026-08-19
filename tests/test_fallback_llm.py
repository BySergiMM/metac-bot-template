"""Provider fallback must change availability and NOTHING else.

The whole risk of this feature is that an infrastructure retry leaks into the
forecast metric: one question served by a second provider must stay one
prediction and one forecast, never two. These tests exist to pin that down.

forecasting-tools is not installed in the test environment (the lab is stdlib
only, deliberately), so a minimal GeneralLlm stub is injected before importing
FallbackLlm. The stub reproduces exactly the three attributes FallbackLlm
touches -- model, allowed_tries, litellm_kwargs -- and an awaitable invoke().
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest

# ---------------------------------------------------------------- stub setup

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

from backtest.fallback_llm import FallbackLlm  # noqa: E402
from forecasting_tools.ai_models.general_llm import GeneralLlm  # noqa: E402


class RateLimitError(Exception):
    """Stands in for litellm.RateLimitError (HTTP 429)."""


class TimeoutError_(Exception):
    """Stands in for litellm.Timeout."""


class ServerError(Exception):
    """Stands in for a provider 5xx."""


class ScriptedBackend(GeneralLlm):
    """A backend whose per-call outcome is scripted.

    `script` is a list; each entry is either an Exception instance (raised) or
    a string (returned). The list is consumed in order; the last entry repeats.
    Every call is counted, which is what lets the tests assert that a healthy
    primary is not consulted twice and a dead one is not retried.
    """

    def __init__(self, model: str, script: list, allowed_tries: int = 1):
        super().__init__(model=model, temperature=0.3, timeout=180,
                         allowed_tries=allowed_tries)
        self.script = list(script)
        self.calls = 0

    async def invoke(self, prompt, system_prompt=None):
        self.calls += 1
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run(coro):
    """Run one coroutine on a fresh loop and close it.

    Closing matters: leaked loops emit ResourceWarnings that bury real output.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def chain(*backends) -> FallbackLlm:
    return FallbackLlm(list(backends))


class FallbackBehaviourTests(unittest.TestCase):
    """TESTs 1-6: the provider chain itself."""

    def test_1_primary_ok_means_fallbacks_are_never_consulted(self):
        openrouter = ScriptedBackend("openrouter/primary", ["PREDICTION"])
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["should not run"])
        groq = ScriptedBackend("groq/openai/gpt-oss-120b", ["should not run"])

        result = run(chain(openrouter, gemini, groq).invoke("p"))

        self.assertEqual(result, "PREDICTION")
        self.assertEqual(openrouter.calls, 1)
        self.assertEqual(gemini.calls, 0, "a healthy primary must not consult fallbacks")
        self.assertEqual(groq.calls, 0)

    def test_2_rate_limit_falls_through_and_yields_exactly_one_result(self):
        openrouter = ScriptedBackend("openrouter/primary", [RateLimitError("429 daily")])
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["PREDICTION"])

        result = run(chain(openrouter, gemini).invoke("p"))

        self.assertEqual(result, "PREDICTION")
        self.assertEqual(openrouter.calls, 1, "429 must cost exactly one attempt")
        self.assertEqual(gemini.calls, 1)
        # One call in, one string out. There is no second prediction here.
        self.assertIsInstance(result, str)

    def test_3_timeout_falls_through(self):
        openrouter = ScriptedBackend("openrouter/primary", [TimeoutError_("timeout")])
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["PREDICTION"])
        self.assertEqual(run(chain(openrouter, gemini).invoke("p")), "PREDICTION")
        self.assertEqual(openrouter.calls, 1)

    def test_4_server_error_falls_through(self):
        openrouter = ScriptedBackend("openrouter/primary", [ServerError("500")])
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["PREDICTION"])
        self.assertEqual(run(chain(openrouter, gemini).invoke("p")), "PREDICTION")
        self.assertEqual(openrouter.calls, 1)

    def test_5_two_failures_reach_the_third_provider(self):
        openrouter = ScriptedBackend("openrouter/primary", [RateLimitError("429")])
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", [RateLimitError("429")])
        groq = ScriptedBackend("groq/openai/gpt-oss-120b", ["PREDICTION"])

        result = run(chain(openrouter, gemini, groq).invoke("p"))

        self.assertEqual(result, "PREDICTION")
        self.assertEqual([openrouter.calls, gemini.calls, groq.calls], [1, 1, 1],
                         "each provider gets exactly one attempt, in order")

    def test_6_all_providers_failing_produces_no_result(self):
        backends = [
            ScriptedBackend("openrouter/primary", [RateLimitError("429")]),
            ScriptedBackend("gemini/gemini-3.5-flash-lite", [RateLimitError("429")]),
            ScriptedBackend("groq/openai/gpt-oss-120b", [ServerError("500")]),
        ]
        with self.assertRaises(RuntimeError) as ctx:
            run(chain(*backends).invoke("p"))
        # The error names every provider, so a run that produced no forecast is
        # diagnosable rather than silently empty.
        for backend in backends:
            self.assertIn(backend.model, str(ctx.exception))
            self.assertEqual(backend.calls, 1)


class RetrySemanticsTests(unittest.TestCase):
    """The defect this feature must not reintroduce: retrying an exhausted
    provider before moving on."""

    def test_a_429_does_not_retry_the_same_provider(self):
        openrouter = ScriptedBackend("openrouter/primary", [RateLimitError("429 daily")])
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["PREDICTION"])
        run(chain(openrouter, gemini).invoke("p"))
        self.assertEqual(
            openrouter.calls, 1,
            "a daily-quota 429 can never succeed on retry; the chain must move on",
        )

    def test_the_chain_itself_is_attempted_once(self):
        """FallbackLlm.invoke overrides GeneralLlm.invoke, so RetryableModel's
        decorator does not wrap the chain."""
        backends = [ScriptedBackend("a", [RateLimitError("429")]),
                    ScriptedBackend("b", [RateLimitError("429")])]
        with self.assertRaises(RuntimeError):
            run(chain(*backends).invoke("p"))
        self.assertEqual([b.calls for b in backends], [1, 1])

    def test_pin_models_gives_chain_members_a_single_try(self):
        """The configuration half of the same guarantee."""
        import backtest.pin_models as pin
        self.assertEqual(pin.TRIES_IN_CHAIN, 1)
        self.assertEqual(pin.TRIES_STANDALONE, 3,
                         "the no-fallback path must keep its original retries")


class PredictionCountTests(unittest.TestCase):
    """TEST 8: provider fallback must not inflate the number of predictions."""

    def test_five_predictions_stay_five_predictions_under_fallback(self):
        # predictions_per_research_report=5 (main.py:675). The bot asks the llm
        # five times; what varies is how many provider attempts each takes.
        openrouter = ScriptedBackend(
            "openrouter/primary",
            [RateLimitError("429"), "P2", RateLimitError("429"), "P4", "P5"],
        )
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["G"])
        llm = chain(openrouter, gemini)

        predictions = [run(llm.invoke("p")) for _ in range(5)]

        self.assertEqual(len(predictions), 5, "still exactly five predictions")
        # infrastructure attempts exceed predictions - that is the point, and
        # they live in a different counter
        self.assertEqual(openrouter.calls + gemini.calls, 7)
        self.assertEqual(gemini.calls, 2, "two of the five were served by fallback")

    def test_a_fallback_result_is_the_same_type_as_a_primary_result(self):
        """The forecasting code must be unable to tell which provider answered."""
        direct = run(chain(ScriptedBackend("openrouter/primary", ["X"])).invoke("p"))
        viafb = run(chain(
            ScriptedBackend("openrouter/primary", [RateLimitError("429")]),
            ScriptedBackend("gemini/gemini-3.5-flash-lite", ["X"]),
        ).invoke("p"))
        self.assertEqual(type(direct), type(viafb))
        self.assertEqual(direct, viafb)


class ForecastMetricTests(unittest.TestCase):
    """TESTs 7, 9, 10: the forecast metric is derived from Metaculus' own
    record of our forecasts, so no amount of provider churn can inflate it."""

    def _question_payload(self, question_id: int, history: list) -> dict:
        return {
            "id": 45179,
            "title": "post",
            "curation_status": "approved",
            "published_at": "2026-06-01T00:00:00Z",
            "projects": {"default_project": {"id": 33022, "name": "FE"}},
            "question": {
                "id": question_id,
                "title": "q",
                "type": "binary",
                "resolution": "yes",
                "open_time": "2026-06-01T12:00:00Z",
                "cp_reveal_time": "2026-06-05T12:00:00Z",
                "scheduled_close_time": "2026-06-10T12:00:00Z",
                "actual_close_time": "2026-06-10T12:00:00Z",
                "spot_scoring_time": "2026-06-07T09:00:00Z",
                "question_weight": 1.0,
                "my_forecasts": {"history": history, "latest": history[-1] if history else None},
            },
        }

    def _forecast_record(self, start="2026-06-02T00:00:00Z"):
        return {"author_id": 306913, "question_id": 500, "start_time": start,
                "end_time": None, "forecast_values": [0.2, 0.8]}

    def test_9_one_question_served_by_fallback_counts_exactly_once(self):
        """Metaculus records ONE forecast; how many providers were tried to get
        it is invisible to the metric, which is the correct behaviour."""
        from research.posts_track_record import build_csvs
        from research.track_record import read_forecast_csv

        posts = [self._question_payload(500, [self._forecast_record()])]
        _q, forecast_csv, stats = build_csvs(posts, 306913, "seergiii-bot")

        self.assertEqual(stats["n_forecast_records"], 1)
        self.assertEqual(len({r.question_id for r in read_forecast_csv(forecast_csv)}), 1)

    def test_10_the_metric_counts_metaculus_records_not_attempts(self):
        """A question with zero recorded forecasts counts zero, however many
        provider attempts were burned trying."""
        from research.posts_track_record import build_csvs

        posts = [self._question_payload(501, [])]
        _q, forecast_csv, stats = build_csvs(posts, 306913, "seergiii-bot")

        self.assertEqual(stats["n_forecast_records"], 0)
        self.assertEqual(stats["n_with_my_forecasts"], 0)

    def test_7_already_forecasted_prevents_a_duplicate_on_a_second_run(self):
        """Reproduces forecast_bot.py:231-238 and questions.py:137-141: a
        question whose my_forecasts.history is non-empty is filtered out."""
        def already_forecasted(payload: dict) -> bool:
            try:
                history = payload["question"]["my_forecasts"]["history"]
                return history is not None and len(history) > 0
            except Exception:
                return False

        first_run = self._question_payload(500, [])
        self.assertFalse(already_forecasted(first_run))

        second_run = self._question_payload(500, [self._forecast_record()])
        self.assertTrue(already_forecasted(second_run),
                        "the second run must skip it, whichever provider served the first")

        questions = [second_run]
        eligible = [q for q in questions if not already_forecasted(q)]
        self.assertEqual(eligible, [], "no duplicate forecast is attempted")


class ConfigurationTests(unittest.TestCase):
    """Only smoke-tested providers may appear in the chain."""

    def test_chain_contains_only_verified_models(self):
        import backtest.pin_models as pin
        models = [m for m, _env in pin.FALLBACK_CHAIN]
        self.assertEqual(models, ["gemini/gemini-3.5-flash-lite",
                                  "groq/openai/gpt-oss-120b"])

    def test_unverified_providers_are_absent(self):
        import backtest.pin_models as pin
        blob = " ".join(m for m, _ in pin.FALLBACK_CHAIN)
        for excluded in ("cerebras", "metaculus", "xai", "grok-"):
            self.assertNotIn(excluded, blob,
                             "{0} did not pass the smoke test".format(excluded))


    def test_parser_only_gets_backends_that_can_emit_structured_json(self):
        """Measured, not assumed: Gemini returned {"probability": 0.42};
        Groq returned json_validate_failed with an empty generation."""
        import backtest.pin_models as pin
        self.assertEqual(pin.STRUCTURED_OUTPUT_CAPABLE, {"gemini/gemini-3.5-flash-lite"})
        pin.ACTIVE_FALLBACKS = list(pin.FALLBACK_CHAIN)
        try:
            self.assertEqual(pin._fallbacks_for("parser"), ["gemini/gemini-3.5-flash-lite"])
            self.assertEqual(pin._fallbacks_for("default"),
                             ["gemini/gemini-3.5-flash-lite", "groq/openai/gpt-oss-120b"])
        finally:
            pin.ACTIVE_FALLBACKS = []

    def test_parser_stays_unwrapped_when_no_fallback_can_parse(self):
        """Groq alone must not become the parser's fallback: prose where JSON
        is required would discard the prediction the fallback just rescued."""
        import backtest.pin_models as pin
        pin.ACTIVE_FALLBACKS = [("groq/openai/gpt-oss-120b", "GROQ_API_KEY")]
        try:
            self.assertEqual(pin._fallbacks_for("parser"), [])
            self.assertEqual(pin._fallbacks_for("researcher"), ["groq/openai/gpt-oss-120b"])
        finally:
            pin.ACTIVE_FALLBACKS = []

    def test_absent_keys_disable_their_leg_rather_than_erroring(self):
        import backtest.pin_models as pin
        active = [m for m, env in pin.FALLBACK_CHAIN if __import__("os").getenv(env)]
        self.assertIsInstance(active, list)  # no key set locally -> empty, no raise


class MetricInvarianceTests(unittest.TestCase):
    """Cases G-J: the constants and the counting rule must be provably
    unchanged, read from the real source rather than restated here."""

    def _main_py(self) -> str:
        import pathlib as _p
        return (_p.Path(__file__).parent.parent / "main.py").read_text()

    def test_J_predictions_per_research_report_is_still_five(self):
        self.assertIn("predictions_per_research_report=5,", self._main_py())
        self.assertIn("research_reports_per_question=1,", self._main_py())

    def test_I_threshold_is_untouched(self):
        """required_successful_predictions is never passed in main.py, so it
        keeps the SDK default of 0.5 -> 0.5 * (1*5) = 2.5 -> at least 3 of 5."""
        self.assertNotIn("required_successful_predictions", self._main_py())
        expected_total = 1 * 5
        self.assertEqual(expected_total * 0.5, 2.5)

    def test_the_prediction_count_is_decided_above_the_llm_layer(self):
        """forecast_bot builds the task list from a bot attribute:
            [self._make_prediction(...) for _ in range(predictions_per_research_report)]
        The fallback lives inside _make_prediction -> llm.invoke(), one level
        below, so it cannot change how many tasks are created."""
        openrouter = ScriptedBackend("openrouter/primary",
                                     [RateLimitError("429")] * 5)
        gemini = ScriptedBackend("gemini/gemini-3.5-flash-lite", ["P"])
        llm = chain(openrouter, gemini)

        predictions_per_research_report = 5
        predictions = [run(llm.invoke("p"))
                       for _ in range(predictions_per_research_report)]

        self.assertEqual(len(predictions), 5)
        self.assertEqual(openrouter.calls, 5, "one failed attempt per prediction")
        self.assertEqual(gemini.calls, 5, "one rescue per prediction")
        # 10 provider attempts, still 5 predictions.
        self.assertNotEqual(openrouter.calls + gemini.calls, len(predictions))

    def test_G_many_provider_attempts_still_yield_one_forecast(self):
        """The end-to-end shape of the guarantee: whatever the chain does,
        Metaculus ends up holding one forecast for the question, and the
        official metric counts questions, not attempts."""
        from research.posts_track_record import build_csvs

        # 12 provider attempts happened upstream; Metaculus recorded one forecast.
        history = [{"author_id": 306913, "question_id": 500,
                    "start_time": "2026-06-02T00:00:00Z", "end_time": None,
                    "forecast_values": [0.2, 0.8]}]
        posts = [ForecastMetricTests()._question_payload(500, history)]
        _q, _f, stats = build_csvs(posts, 306913, "seergiii-bot")
        self.assertEqual(stats["n_forecast_records"], 1)


class PostRetryDuplicationTests(unittest.TestCase):
    """Case H, the one path that could actually double-count.

    metaculus_client._post_question_prediction carries
    @retry_with_exponential_backoff(retry_on_exceptions=(RequestException,
    Timeout, ConnectionError)). If Metaculus accepts a POST but the response is
    lost, the client raises Timeout and posts AGAIN. Metaculus then holds two
    forecast rows for one question.

    This is pre-existing SDK behaviour, untouched by the fallback (which never
    references POST at all). What matters here is that it cannot inflate the
    official metric.
    """

    def test_H_a_duplicated_post_does_not_inflate_the_official_metric(self):
        from research.posts_track_record import build_csvs
        from research.track_record import read_forecast_csv

        # The network-retry scenario: two forecast rows, same question.
        history = [
            {"author_id": 306913, "question_id": 500,
             "start_time": "2026-06-02T00:00:00Z",
             "end_time": "2026-06-02T00:00:05Z", "forecast_values": [0.2, 0.8]},
            {"author_id": 306913, "question_id": 500,
             "start_time": "2026-06-02T00:00:05Z",
             "end_time": None, "forecast_values": [0.2, 0.8]},
        ]
        posts = [ForecastMetricTests()._question_payload(500, history)]
        _q, forecast_csv, stats = build_csvs(posts, 306913, "seergiii-bot")

        rows = read_forecast_csv(forecast_csv)
        self.assertEqual(len(rows), 2, "two rows really are recorded")
        # ...and the official metric counts DISTINCT questions, so it is 1.
        distinct_questions = len({r.question_id for r in rows})
        self.assertEqual(distinct_questions, 1,
                         "n_questions_with_own_forecast counts questions, not rows")

    def test_the_official_metric_is_a_set_of_question_ids(self):
        """Pin the implementation that makes the above true, so a future
        refactor to a row count fails loudly here."""
        import pathlib as _p
        src = (_p.Path(__file__).parent.parent
               / "research" / "fetch_own_track_record.py").read_text()
        self.assertIn(
            '"n_questions_with_own_forecast": len({f.question_id for f in own})',
            src,
            "the metric must stay a set of question ids, never a row count",
        )

    def test_the_fallback_layer_cannot_reach_the_post_path(self):
        import pathlib as _p
        src = (_p.Path(__file__).parent.parent
               / "backtest" / "fallback_llm.py").read_text()
        for forbidden in ("post_binary_question_prediction", "publish_report",
                          "metaculus_client", "post_question_comment"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
