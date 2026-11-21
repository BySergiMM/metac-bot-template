"""The over-budget forecaster keeps a fallback, and a failed sample says why.

SummerTemplateBot2026.get_llm hands forecaster_override out as it is, so until
2026-11-22 the over-budget forecaster -- a bare ClaudeLlm on Sonnet 5.5 -- lost
the sample on one refusal or timeout, where the pinned forecaster's chain would
have asked the next model. main.budget_forecaster now returns a FallbackLlm,
Sonnet then Haiku.

These run the REAL main.py bot, ClaudeLlm and FallbackLlm with only the
Anthropic client replaced, and make the first model actually fail: a guard
that is never made to fail proves nothing (the 2026-09 env-restore hole, where
CI passed and production broke). The control case shows that the same failure
does lose the sample without the chain, so the test can tell the two apart.

Every question and answer here is synthetic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import logging
import os
import unittest
from types import SimpleNamespace as Q
from unittest import mock

from bot_helpers import RedactForecastContent
from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    from forecasting_tools.data_models.questions import BinaryQuestion

    from backtest import claude_llm, fallback_llm
    from backtest.rate_limiter import reset_registry
    from publication import PublishingClient

import spend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Would be a forecast if it reached the log; it must not.
LEAK = "Probability: 91%"


def load_main():
    """main.py without its __main__ block (see test_pipeline_hostile_model)."""
    with real_forecasting_tools():
        spec = importlib.util.spec_from_file_location(
            "override_fallback_main", os.path.join(ROOT, "main.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


BOTMOD = load_main()


def reply(text, stop="end_turn"):
    return Q(content=[Q(type="text", text=text)], stop_reason=stop,
             usage=Q(input_tokens=900, cache_creation_input_tokens=0,
                     cache_read_input_tokens=0, output_tokens=300, server_tool_use=None))


class ByModel:
    """Stands in for anthropic.AsyncAnthropic: answers per model and records
    the order in which the models were asked."""

    def __init__(self, **answers):
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, timeout=None):  # noqa: ARG002
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
        answer = self.answers[model]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def binary():
    return BinaryQuestion(
        question_text="Will the synthetic event happen?",
        id_of_post=7301,
        id_of_question=7302,
        page_url="https://www.metaculus.com/questions/7301",
        background_info="background",
        resolution_criteria="criteria",
        fine_print="fine print",
    )


class Case(unittest.TestCase):
    def setUp(self):
        # No ledger, whatever the developer's shell has: these calls are fake.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SPEND_LEDGER_PATH", None)
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


class BudgetForecasterTests(Case):
    def test_it_is_a_chain_sonnet_then_haiku(self):
        llm = BOTMOD.budget_forecaster()
        self.assertIsInstance(llm, fallback_llm.FallbackLlm)
        self.assertEqual([b.model for b in llm._backends],
                         [spend.BUDGET_FORECASTER, spend.BUDGET_FORECASTER_FALLBACK])
        self.assertEqual(llm._backends[0].effort, "high")
        self.assertEqual(llm.model, spend.BUDGET_FORECASTER)

    def test_a_sonnet_refusal_is_served_by_haiku(self):
        fake = ByModel(**{"claude-sonnet-5-5": reply(LEAK, stop="refusal"),
                          "claude-haiku-5-5": reply("Reasoning.\nProbability: 37%")})
        prediction = self.predict(self.bot(BOTMOD.budget_forecaster()), fake)
        self.assertEqual(fake.asked, ["claude-sonnet-5-5", "claude-haiku-5-5"])
        self.assertAlmostEqual(prediction.prediction_value, 0.37)

    def test_a_sonnet_timeout_is_served_by_haiku(self):
        fake = ByModel(**{"claude-sonnet-5-5": TimeoutError("read timed out"),
                          "claude-haiku-5-5": reply("Probability: 37%")})
        prediction = self.predict(self.bot(BOTMOD.budget_forecaster()), fake)
        self.assertEqual(fake.asked, ["claude-sonnet-5-5", "claude-haiku-5-5"])
        self.assertAlmostEqual(prediction.prediction_value, 0.37)

    def test_control_the_bare_sonnet_of_before_loses_the_sample(self):
        fake = ByModel(**{"claude-sonnet-5-5": reply(LEAK, stop="refusal"),
                          "claude-haiku-5-5": reply("Probability: 37%")})
        bare = claude_llm.ClaudeLlm(model=spend.BUDGET_FORECASTER, effort="high", timeout=420)
        with self.assertRaises(RuntimeError):
            self.predict(self.bot(bare), fake)
        self.assertEqual(fake.asked, ["claude-sonnet-5-5"])

    def test_the_override_is_what_get_llm_hands_out(self):
        llm = BOTMOD.budget_forecaster()
        bot = self.bot(llm)
        self.assertIs(bot.get_llm("default", "llm"), llm)
        self.assertEqual(bot.get_llm("default", "string_name"), spend.BUDGET_FORECASTER)


class SampleFailedTests(Case):
    def failing(self):
        """The whole chain fails, each link with the would-be forecast in its
        exception message."""
        fake = ByModel(**{"claude-sonnet-5-5": RuntimeError(LEAK),
                          "claude-haiku-5-5": RuntimeError(LEAK)})
        return self.bot(BOTMOD.budget_forecaster()), fake

    def test_a_failed_sample_logs_its_class_and_question_type_and_still_raises(self):
        bot, fake = self.failing()
        with self.assertLogs(BOTMOD.logger, level="WARNING") as logs:
            with self.assertRaises(RuntimeError):
                self.predict(bot, fake)
        lines = [line for line in logs.output if "sample_failed" in line]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith("sample_failed type=RuntimeError qtype=binary"), lines[0])

    def test_the_line_never_carries_the_exception_message(self):
        bot, fake = self.failing()
        with self.assertLogs(BOTMOD.logger, level="WARNING") as logs:
            with self.assertRaises(RuntimeError):
                self.predict(bot, fake)
        line = next(line for line in logs.output if "sample_failed" in line)
        self.assertNotIn("91", line)
        self.assertNotIn("Probability", line)

    def test_a_successful_sample_logs_nothing_of_the_kind(self):
        fake = ByModel(**{"claude-sonnet-5-5": reply("Probability: 37%"),
                          "claude-haiku-5-5": reply("Probability: 37%")})
        with self.assertLogs(BOTMOD.logger, level="INFO") as logs:
            self.predict(self.bot(BOTMOD.budget_forecaster()), fake)
        self.assertFalse(any("sample_failed" in line for line in logs.output))

    def test_the_question_type_is_one_safe_word(self):
        self.assertEqual(BOTMOD._qtype(binary()), "binary")
        self.assertEqual(BOTMOD._qtype(Q(question_type="multiple_choice")), "multiple_choice")
        self.assertEqual(BOTMOD._qtype(Q(question_type="Probability: 91%")), "other")

    def test_the_line_survives_the_public_log_filter_unchanged(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(RedactForecastContent())
        log = logging.getLogger("sample_failed_redaction_" + self.id())
        log.handlers, log.propagate = [handler], False
        log.warning("sample_failed type=%s qtype=%s", "RuntimeError", "numeric")
        self.assertEqual(stream.getvalue().strip(), "sample_failed type=RuntimeError qtype=numeric")


if __name__ == "__main__":
    unittest.main()
