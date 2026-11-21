"""The forecaster boost: Fable 5.2 on FutureEval and the Market Pulse final,
dated and capped in dollars, off unless both variables say otherwise.

spend.forecaster_boost decides; main.apply_forecaster_boost hands the bot a
FallbackLlm of Fable, then the pinned forecaster's Opus, then Sonnet. Off is
today's chain, so every doubt must turn it off -- and with the repository
variables unset (production's state, where the workflow passes them as empty
strings) nothing may change at all.

The fallback cases run the REAL main.py bot, ClaudeLlm and FallbackLlm with
only anthropic.AsyncAnthropic replaced, and make Fable actually fail: a guard
that is never made to fail proves nothing (the 2026-09 env-restore hole).
Every question, answer and dollar here is synthetic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as Q
from unittest import mock

import yaml

import spend
from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    import anthropic
    from forecasting_tools.data_models.questions import BinaryQuestion

    from backtest import claude_llm, fallback_llm, pin_models
    from backtest.rate_limiter import reset_registry
    from publication import PublishingClient

try:
    import httpx2 as httpx
except ImportError:  # an anthropic SDK before 1.x
    import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
FABLE = "anthropic/claude-fable-5-2"
VARS = ("FORECASTER_BOOST_MODEL", "FORECASTER_BOOST_UNTIL",
        "FORECASTER_BOOST_SAMPLES", "FORECASTER_BOOST_MAX_USD")
#: Would be a forecast if it reached the log; it must not.
LEAK = "Probability: 91%"


def load_main():
    """main.py without its __main__ block (see test_pipeline_hostile_model)."""
    with real_forecasting_tools():
        spec = importlib.util.spec_from_file_location(
            "forecaster_boost_main", os.path.join(ROOT, "main.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


BOTMOD = load_main()


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def call(t, usd, model="claude-opus-5-5", tour="futureeval", phase="forecast"):
    return {"t": t.isoformat(timespec="seconds"), "run": "r1", "tour": tour, "q": 1,
            "phase": phase, "provider": "anthropic", "model": model, "in": 1, "out": 1,
            "searches": 0, "usd": usd}


class LedgerCase(unittest.TestCase):
    """A temporary ledger, the owner's cycle (28th) and budget, and the four
    variables unset whatever the developer's shell holds."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "ledger.jsonl")
        env = mock.patch.dict(os.environ, {
            "SPEND_LEDGER_PATH": self.path, "GITHUB_RUN_ID": "r9", "CLAUDE_BUDGET_USD": "100",
            "CLAUDE_BUDGET_CYCLE_DAY": "28", "SPEND_THROTTLE_AT": "0.9",
            "SPEND_ALERT_THRESHOLDS": "0.8,1.0"})
        env.start()
        self.addCleanup(env.stop)
        for name in VARS + ("GITHUB_STEP_SUMMARY",):
            os.environ.pop(name, None)
        spend.set_tournament(None)
        self.addCleanup(spend.set_tournament, None)
        spend._boost_noted.clear()
        self.addCleanup(spend._boost_noted.clear)
        warned = spend._warned
        self.addCleanup(setattr, spend, "_warned", warned)
        spend._warned = False

    def write(self, *lines):
        with open(self.path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")

    def history(self, usd_per_day=2.0, days=10):
        """A restored production ledger: its start marker and a cycle of Opus."""
        start = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
        self.write({"t": "2026-10-20T00:00:00+00:00", "kind": "start"},
                   *[call(start + timedelta(days=d), usd_per_day) for d in range(days)])

    def on(self, **extra):
        os.environ.update({"FORECASTER_BOOST_MODEL": FABLE, "FORECASTER_BOOST_UNTIL": "2026-11-30",
                           **extra})


class SwitchTests(LedgerCase):
    def setUp(self):
        super().setUp()
        self.history()

    def test_on_for_futureeval_the_market_pulse_final_and_test_bot(self):
        self.on()
        for tour in ("futureeval", "market_pulse", "test"):
            boost = spend.forecaster_boost(NOW, tour)
            self.assertEqual((boost.model, boost.samples, boost.cap), (FABLE, 5, 20.0), tour)
            self.assertEqual(boost.spent, 0.0)

    def test_the_tournament_defaults_to_the_one_set(self):
        self.on()
        spend.set_tournament("futureeval")
        self.assertIsNotNone(spend.forecaster_boost(NOW))
        spend.set_tournament("minibench")
        self.assertIsNone(spend.forecaster_boost(NOW))

    def test_off_everywhere_else(self):
        self.on()
        for tour in ("minibench", "cup", "market_pulse_early", None, "other"):
            self.assertIsNone(spend.forecaster_boost(NOW, tour), tour)

    def test_off_when_unset_and_silent(self):
        with self.assertNoLogs(spend.logger, level="INFO"):
            for tour in ("futureeval", "market_pulse", "test"):
                self.assertIsNone(spend.forecaster_boost(NOW, tour))

    def test_off_without_a_date(self):
        os.environ["FORECASTER_BOOST_MODEL"] = FABLE
        self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))

    def test_the_date_is_inclusive_through_its_utc_day_and_then_expires(self):
        self.on()
        last = datetime(2026, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
        self.assertIsNotNone(spend.forecaster_boost(last, "futureeval"))
        self.assertIsNone(spend.forecaster_boost(last + timedelta(seconds=1), "futureeval"))

    def test_off_on_a_date_it_cannot_read(self):
        for raw in ("30/11/2026", "2026-11-31", "soon", "2026-11-30T12:00"):
            self.on(FORECASTER_BOOST_UNTIL=raw)
            self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"), raw)

    def test_off_with_one_warning_for_a_model_without_a_price_row(self):
        self.on(FORECASTER_BOOST_MODEL="anthropic/claude-fable-9")
        with self.assertLogs(spend.logger, level="WARNING") as logs:
            self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))
            self.assertIsNone(spend.forecaster_boost(NOW, "market_pulse"))
        self.assertEqual(logs.output, ["WARNING:spend:forecaster_boost_off reason=unpriced"])
        for model in ("claude-fable-5-2", "openrouter/anthropic/claude-fable-5-2"):
            self.on(FORECASTER_BOOST_MODEL=model)
            self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"), model)

    def test_off_once_over_budget(self):
        self.write(*[call(NOW - timedelta(hours=h), 10.0) for h in range(1, 10)])  # +$90
        self.on()
        self.assertTrue(spend.over_budget(NOW))
        self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))

    def test_off_on_the_tight_pace_of_prize_forecasts(self):
        self.write(*[call(datetime(2026, 10, 28, 13, tzinfo=timezone.utc) + timedelta(days=d), 2.0)
                     for d in range(10)])  # $4 a day: ~$122 by the cycle's end
        self.on()
        self.assertEqual(spend.prize_forecasts(NOW, "futureeval"), spend.WIDE_FORECASTS_TIGHT)
        self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))

    def test_off_once_the_cap_is_reached(self):
        # Small sums: Fable's dollars count towards the pace too.
        self.write(call(NOW - timedelta(hours=2), 3.0, model="claude-fable-5-2"),
                   call(NOW - timedelta(hours=1), 1.99, model="claude-fable-5-2"))
        self.on(FORECASTER_BOOST_MAX_USD="5")
        self.assertAlmostEqual(spend.forecaster_boost(NOW, "futureeval").spent, 4.99)
        self.write(call(NOW - timedelta(minutes=30), 0.01, model="claude-fable-5-2"))
        self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))
        self.on(FORECASTER_BOOST_MAX_USD="6")
        self.assertEqual(spend.forecaster_boost(NOW, "futureeval").cap, 6.0)

    def test_a_cap_it_cannot_read_is_the_default(self):
        for raw in ("nan", "inf", "lots"):
            self.on(FORECASTER_BOOST_MAX_USD=raw)
            self.assertEqual(spend.forecaster_boost(NOW, "futureeval").cap, 20.0, raw)
        self.on(FORECASTER_BOOST_MAX_USD="0")
        self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))

    def test_the_cap_counts_only_the_boost_models_rows_in_any_phase(self):
        self.write(call(NOW - timedelta(hours=3), 5.0, model="claude-sonnet-5-5"),
                   call(NOW - timedelta(hours=2), 2.0, model="claude-fable-5-2", phase="research"),
                   call(NOW - timedelta(hours=1), 1.5, model="claude-fable-5-2", tour="test"))
        self.on()
        self.assertAlmostEqual(spend.forecaster_boost(NOW, "futureeval").spent, 3.5)

    def test_samples_default_to_five_and_clamp_to_one_through_five(self):
        for raw, samples in (("", 5), ("3", 3), ("1", 1), ("0", 1), ("-2", 1), ("9", 5), ("x", 5)):
            self.on(FORECASTER_BOOST_SAMPLES=raw)
            self.assertEqual(spend.forecaster_boost(NOW, "futureeval").samples, samples, raw)

    def test_fails_closed_on_any_error(self):
        self.on()
        with mock.patch.object(spend, "load", side_effect=OSError("disk")):
            self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))
            self.assertFalse(spend.boost_room(FABLE, 20.0))

    def test_the_room_check_before_each_call(self):
        self.write(call(NOW - timedelta(hours=1), 14.5, model="claude-fable-5-2"))
        self.assertTrue(spend.boost_room(FABLE, 15.0))
        self.write(call(NOW, 0.5, model="claude-fable-5-2"))
        self.assertFalse(spend.boost_room(FABLE, 15.0))


class LedgerHistoryTests(LedgerCase):
    def test_production_without_a_history_is_off_and_test_bot_is_on(self):
        """A failed cache restore (2026-10-30) starts an empty ledger, where the
        cap would read $0; Test Bot's ledger starts empty on purpose."""
        self.on()
        self.assertIsNone(spend.forecaster_boost(NOW, "futureeval"))
        self.assertIsNone(spend.forecaster_boost(NOW, "market_pulse"))
        self.assertIsNotNone(spend.forecaster_boost(NOW, "test"))

    def test_the_cap_survives_the_cycle_reset_and_the_end_of_run_pruning(self):
        """The cycle resets on the 28th, the credit lapses on the 30th: what
        Fable spent on the 26th and 27th still counts on the 29th."""
        self.history(usd_per_day=1.0, days=30)  # Opus, 28 Oct to 26 Nov
        self.write(call(datetime(2026, 11, 26, 20, tzinfo=timezone.utc), 9.0, model="claude-fable-5-2"),
                   call(datetime(2026, 11, 27, 20, tzinfo=timezone.utc), 6.0, model="claude-fable-5-2"),
                   call(datetime(2026, 11, 28, 13, tzinfo=timezone.utc), 1.0))
        now = datetime(2026, 11, 29, 12, tzinfo=timezone.utc)
        self.on(FORECASTER_BOOST_MAX_USD="16")
        self.assertAlmostEqual(spend.month_spend(spend.load(), now), 1.0)  # the new cycle
        self.assertAlmostEqual(spend.forecaster_boost(now, "futureeval").spent, 15.0)
        with mock.patch("builtins.print"):
            spend.after_run(now, send=lambda _text: True)
        self.assertAlmostEqual(spend.forecaster_boost(now, "futureeval").spent, 15.0)
        self.on(FORECASTER_BOOST_MAX_USD="15")
        self.assertIsNone(spend.forecaster_boost(now, "futureeval"))


class DegradedAlertTests(unittest.TestCase):
    def test_a_run_fable_forecast_alone_is_not_the_2026_10_25_failure(self):
        lines = [call(NOW, 0.4, model="claude-fable-5-2")]
        lines[0]["run"] = "r9"
        self.assertEqual(spend.degraded_forecasts(lines, "r9"), [])
        lines[0]["model"] = "claude-haiku-5-5"
        self.assertEqual(spend.degraded_forecasts(lines, "r9"), ["claude-haiku-5-5"])


############################## PRODUCTION INVARIANT ##############################


class UnsetIsTodayTests(LedgerCase):
    """With the variables unset -- or empty, as `vars.X || ''` passes them --
    the model chain is the one production runs today, byte for byte."""

    def setUp(self):
        super().setUp()
        self.history()
        os.environ.update({name: "" for name in VARS})

    def test_no_tournament_gets_a_boost(self):
        for tour in list(spend.TOURNAMENT_NAMES) + [None]:
            self.assertIsNone(spend.forecaster_boost(NOW, tour), tour)

    def test_apply_leaves_the_bot_exactly_as_it_was_and_says_nothing(self):
        pinned, sentinel = object(), object()
        for tour in ("futureeval", "market_pulse", "test"):
            bot = Q(forecaster_override=pinned, predictions_per_research_report=sentinel,
                    cheap_forecast=sentinel)
            with self.assertNoLogs(BOTMOD.logger, level="INFO"):
                self.assertFalse(BOTMOD.apply_forecaster_boost(bot, tour))
            self.assertEqual(vars(bot), {"forecaster_override": pinned,
                                         "predictions_per_research_report": sentinel,
                                         "cheap_forecast": sentinel})

    def test_main_applies_it_only_where_it_may_and_resets_it_after(self):
        src = read("main.py")
        tail = src[src.index('if run_mode == "tournament":'):]
        self.assertEqual(tail.count("apply_forecaster_boost("), 3)
        for tour in ("futureeval", "market_pulse", "test"):
            self.assertIn(f'apply_forecaster_boost(template_bot, "{tour}")', tail)
        # After FutureEval, MiniBench resets the chain; after the Market Pulse
        # final, the early safety net does; the Cup never sees it.
        fe = tail.index('apply_forecaster_boost(template_bot, "futureeval")')
        minibench = tail.index('spend.set_tournament("minibench")')
        self.assertLess(fe, minibench)
        self.assertIn("forecast_with_opus()", tail[minibench:tail.index("minibench_reports =")])
        mp = tail.index('apply_forecaster_boost(template_bot, "market_pulse")')
        early = tail.index('spend.set_tournament("market_pulse_early")')
        self.assertLess(tail.index("forecast_questions(final"), early)
        self.assertLess(mp, tail.index("forecast_questions(final"))
        self.assertIn("forecast_with_haiku()", tail[early:tail.index("forecast_questions(early")])
        cup = tail[tail.index('elif run_mode == "metaculus_cup":'):tail.index('elif run_mode == "market_pulse":')]
        self.assertNotIn("apply_forecaster_boost", cup)
        # Both resets really put the override back.
        body = src[src.index("def forecast_with_opus("):src.index("forecast_with_opus()\n")]
        self.assertIn("template_bot.forecaster_override = budget_forecaster() if opus_forecasts == 0 else None",
                      body)
        self.assertIn("template_bot.forecaster_override = None", body)


class WorkflowTests(unittest.TestCase):
    def steps(self, name):
        flow = yaml.safe_load(read(".github", "workflows", name))
        return flow, {step["name"]: step for step in flow["jobs"]["forecast_job"]["steps"] if "name" in step}

    def test_production_passes_the_variables_with_an_empty_default_where_it_may(self):
        flow, steps = self.steps("run_bot_on_tournament.yaml")
        for name in ("Run bot", "Forecast Market Pulse questions (prize)"):
            env = steps[name]["env"]
            for var in VARS:
                self.assertEqual(env[var], "${{ vars.%s || '' }}" % var, (name, var))
        for name in ("Pin LLM models", "Forecast new Metaculus Cup questions (practice)"):
            self.assertFalse(set(VARS) & set(steps[name].get("env") or {}), name)
        self.assertFalse(set(VARS) & set(flow["jobs"]["forecast_job"]["env"]))

    def test_test_bot_has_a_forecaster_input_that_defaults_to_opus(self):
        flow, steps = self.steps("test_bot.yaml")
        choice = (flow.get("on") or flow.get(True))["workflow_dispatch"]["inputs"]["forecaster"]
        self.assertEqual((choice["default"], choice["options"]), ("opus", ["opus", "fable"]))
        env = steps["Run bot"]["env"]
        self.assertEqual(env["FORECASTER_BOOST_MODEL"],
                         "${{ inputs.forecaster == 'fable' && 'anthropic/claude-fable-5-2' || '' }}")
        self.assertEqual(env["FORECASTER_BOOST_SAMPLES"], "${{ inputs.forecaster == 'fable' && '5' || '' }}")
        self.assertEqual(env["FORECASTER_BOOST_MAX_USD"], "${{ inputs.forecaster == 'fable' && '15' || '' }}")
        self.assertIn("inputs.forecaster == 'fable'", env["FORECASTER_BOOST_UNTIL"])
        self.assertFalse(set(VARS) & set(steps["Pin and verify LLM models"].get("env") or {}))


############################## THE CHAIN, FOR REAL ##############################


def reply(text, stop="end_turn"):
    return Q(content=[Q(type="thinking", thinking=""), Q(type="text", text=text)], stop_reason=stop,
             usage=Q(input_tokens=900, cache_creation_input_tokens=0,
                     cache_read_input_tokens=0, output_tokens=300, server_tool_use=None))


class ByModel:
    """Stands in for anthropic.AsyncAnthropic: answers per model and records
    each client's settings and each request."""

    def __init__(self, **answers):
        self.answers = answers
        self.asked: list[str] = []
        self.clients: list[dict] = []
        self.requests: dict[str, dict] = {}

    def __call__(self, **kwargs):
        self.clients.append(kwargs)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def beta(self):
        return Q(messages=Q(create=self.create))

    async def create(self, **kwargs):
        model = kwargs["model"]
        self.asked.append(model)
        self.requests[model] = kwargs
        answer = self.answers[model]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def timeout():
    return anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def binary():
    return BinaryQuestion(
        question_text="Will the synthetic event happen?",
        id_of_post=7401,
        id_of_question=7402,
        page_url="https://www.metaculus.com/questions/7401",
        background_info="background",
        resolution_criteria="criteria",
        fine_print="fine print",
    )


BOOST = spend.ForecasterBoost(FABLE, 5, 0.0, 15.0)


class ChainTests(LedgerCase):
    def setUp(self):
        super().setUp()
        os.environ.pop("SPEND_LEDGER_PATH")  # these calls are fake: no ledger
        reset_registry()
        self.addCleanup(reset_registry)

    def bot(self, override):
        with real_forecasting_tools():
            bot = BOTMOD.SummerTemplateBot2026(
                metaculus_client=PublishingClient(sleep=lambda _s: None),
                research_reports_per_question=1,
                predictions_per_research_report=1,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
            )
        bot.forecaster_override = override
        return bot

    def predict(self, bot, fake):
        question = binary()

        async def go():
            bot._note_pads.append(await bot._initialize_notepad(question))
            return await bot._make_prediction(question, "synthetic research")

        with mock.patch("anthropic.AsyncAnthropic", fake), real_forecasting_tools():
            return asyncio.run(go())

    def test_it_is_fable_then_the_pinned_opus_then_sonnet(self):
        llm = BOTMOD.boost_forecaster(BOOST)
        self.assertIsInstance(llm, fallback_llm.FallbackLlm)
        fable, opus, sonnet = llm._backends
        self.assertEqual([b.model for b in llm._backends],
                         [FABLE, pin_models.OPUS_FORECASTER, spend.BUDGET_FORECASTER])
        self.assertEqual((fable.effort, fable.max_tokens, fable.max_retries), ("high", 32000, 0))
        self.assertEqual(opus.effort, pin_models.FORECASTER_REASONING[pin_models.OPUS_FORECASTER])
        self.assertEqual((opus.max_tokens, opus.max_retries, opus.gate), (claude_llm.MAX_TOKENS, None, None))
        self.assertEqual(sonnet.effort, "high")
        self.assertEqual(llm.model, FABLE)

    def test_fable_gets_adaptive_thinking_high_effort_and_room_to_think(self):
        fake = ByModel(**{"claude-fable-5-2": reply("Probability: 37%")})
        prediction = self.predict(self.bot(BOTMOD.boost_forecaster(BOOST)), fake)
        self.assertAlmostEqual(prediction.prediction_value, 0.37)
        sent = fake.requests["claude-fable-5-2"]
        self.assertEqual(sent["thinking"], {"type": "adaptive"})
        self.assertEqual(sent["output_config"], {"effort": "high"})
        self.assertEqual(sent["max_tokens"], 32000)
        self.assertEqual(fake.clients, [{"timeout": 420, "max_retries": 0}])

    def test_a_fable_refusal_is_served_by_opus(self):
        fake = ByModel(**{"claude-fable-5-2": reply(LEAK, stop="refusal"),
                          "claude-opus-5-5": reply("Probability: 37%")})
        prediction = self.predict(self.bot(BOTMOD.boost_forecaster(BOOST)), fake)
        self.assertEqual(fake.asked, ["claude-fable-5-2", "claude-opus-5-5"])
        self.assertAlmostEqual(prediction.prediction_value, 0.37)
        # Opus keeps the SDK's retries; only Fable's are off.
        self.assertEqual(fake.clients[1], {"timeout": 420})

    def test_a_fable_timeout_is_served_by_opus(self):
        fake = ByModel(**{"claude-fable-5-2": timeout(),
                          "claude-opus-5-5": reply("Probability: 37%")})
        prediction = self.predict(self.bot(BOTMOD.boost_forecaster(BOOST)), fake)
        self.assertEqual(fake.asked, ["claude-fable-5-2", "claude-opus-5-5"])
        self.assertAlmostEqual(prediction.prediction_value, 0.37)

    def test_once_the_cap_is_reached_fable_is_not_even_asked(self):
        os.environ["SPEND_LEDGER_PATH"] = self.path
        self.write(call(NOW, 15.0, model="claude-fable-5-2"))
        fake = ByModel(**{"claude-fable-5-2": reply("Probability: 80%"),
                          "claude-opus-5-5": reply("Probability: 37%")})
        prediction = self.predict(self.bot(BOTMOD.boost_forecaster(BOOST)), fake)
        self.assertEqual(fake.asked, ["claude-opus-5-5"])
        self.assertAlmostEqual(prediction.prediction_value, 0.37)

    def test_control_a_bare_fable_loses_the_sample_on_a_refusal(self):
        fake = ByModel(**{"claude-fable-5-2": reply(LEAK, stop="refusal"),
                          "claude-opus-5-5": reply("Probability: 37%")})
        bare = claude_llm.ClaudeLlm(model=FABLE, effort="high", timeout=420)
        with self.assertRaises(RuntimeError):
            self.predict(self.bot(bare), fake)
        self.assertEqual(fake.asked, ["claude-fable-5-2"])

    def test_apply_hands_the_bot_the_chain_and_the_samples_and_logs_counts(self):
        os.environ["SPEND_LEDGER_PATH"] = self.path
        self.history()
        # The real clock (apply passes no time), so a date that never expires.
        self.on(FORECASTER_BOOST_SAMPLES="4", FORECASTER_BOOST_UNTIL="2099-12-31")
        bot = self.bot(None)
        with self.assertLogs(BOTMOD.logger, level="INFO") as logs:
            self.assertTrue(BOTMOD.apply_forecaster_boost(bot, "futureeval"))
        self.assertEqual(bot.predictions_per_research_report, 4)
        self.assertIs(bot.get_llm("default", "llm"), bot.forecaster_override)
        self.assertEqual(bot.get_llm("default", "string_name"), FABLE)
        lines = [line for line in logs.output if "forecaster_boost" in line]
        self.assertEqual(lines, ["INFO:%s:forecaster_boost model=claude-fable-5-2 samples=4 "
                                 "spent=0.00 cap=20.00" % BOTMOD.logger.name])


if __name__ == "__main__":
    unittest.main()
