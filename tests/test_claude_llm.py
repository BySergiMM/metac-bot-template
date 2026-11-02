"""ClaudeLlm, the budget plan and the alert when Opus stops forecasting.

No network: the Anthropic client is a fake. Since 2026-11-03 every role runs on
the owner's Anthropic key (backtest/pin_models.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as Q
from unittest import mock

import spend
from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    from backtest.claude_llm import ClaudeLlm

NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)


def usage(uncached=1000, write=0, read=0, out=500):
    return Q(input_tokens=uncached, cache_creation_input_tokens=write, cache_read_input_tokens=read,
             output_tokens=out, server_tool_use=None)


def reply(text="Probability: 42%", stop="end_turn", **tokens):
    return Q(content=[Q(type="thinking", thinking="hmm"), Q(type="text", text=text)],
             stop_reason=stop, usage=usage(**tokens))


class FakeAnthropic:
    """Stands in for anthropic.AsyncAnthropic; records every request."""

    def __init__(self, replies):
        self.replies, self.requests, self.active, self.peak = list(replies), [], 0, 0

    def __call__(self, timeout=None):
        self.timeout = timeout
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def beta(self):
        return Q(messages=Q(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ledger.jsonl")
        self.env = mock.patch.dict(os.environ, {
            "SPEND_LEDGER_PATH": self.path, "GITHUB_RUN_ID": "r9", "CLAUDE_BUDGET_USD": "100",
            "CLAUDE_BUDGET_CYCLE_DAY": "28", "SPEND_THROTTLE_AT": "0.9",
            "SPEND_ALERT_THRESHOLDS": "0.8,1.0"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.dir.cleanup()

    def lines(self):
        return spend.load(self.path)


class InvokeTests(LedgerCase):
    def run_llm(self, fake, coroutine_factory):
        with mock.patch("anthropic.AsyncAnthropic", fake):
            return asyncio.run(coroutine_factory())

    def test_opus_asks_for_adaptive_thinking_at_its_effort_and_returns_the_text(self):
        fake = FakeAnthropic([reply()])
        llm = ClaudeLlm(model="anthropic/claude-opus-5-5", effort="xhigh", timeout=420)
        self.assertEqual(self.run_llm(fake, lambda: llm.invoke("prompt")), "Probability: 42%")
        request = fake.requests[0]
        self.assertEqual(request["model"], "claude-opus-5-5")
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["output_config"], {"effort": "xhigh"})
        self.assertEqual(request["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(request["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("temperature", request)
        self.assertEqual(fake.timeout, 420)

    def test_haiku_without_effort_does_not_think(self):
        fake = FakeAnthropic([reply()])
        llm = ClaudeLlm(model="anthropic/claude-haiku-5-5")
        self.run_llm(fake, lambda: llm.invoke("prompt", "system"))
        self.assertNotIn("thinking", fake.requests[0])
        self.assertNotIn("output_config", fake.requests[0])
        self.assertEqual(fake.requests[0]["system"], "system")

    def test_the_call_goes_to_the_ledger_as_the_budgeted_provider_at_the_billed_price(self):
        fake = FakeAnthropic([reply(uncached=10, write=1000, read=2000, out=500)])
        llm = ClaudeLlm(model="anthropic/claude-opus-5-5", effort="xhigh")
        self.run_llm(fake, lambda: llm.invoke("prompt"))
        line = self.lines()[-1]
        self.assertEqual((line["provider"], line["model"]), (spend.BUDGET_PROVIDER, "claude-opus-5-5"))
        expected = ((10 + 1.25 * 1000 + 0.05 * 2000) * 4.0 + 500 * 20.0) / 1e6
        self.assertAlmostEqual(line["usd"], expected, places=6)
        self.assertEqual((line["in"], line["out"]), (3010, 500))

    def test_a_refusal_or_no_text_raises_for_the_fallback_but_is_still_billed(self):
        for bad in (reply(stop="refusal"), reply(text="  ")):
            fake = FakeAnthropic([bad])
            llm = ClaudeLlm(model="anthropic/claude-opus-5-5")
            with self.assertRaises(RuntimeError):
                self.run_llm(fake, lambda: llm.invoke("prompt"))
        self.assertEqual(len(self.lines()), 2)

    def test_the_forecasts_of_one_question_wait_for_the_first_so_they_read_its_cache(self):
        fake = FakeAnthropic([reply(), reply(), reply()])
        llm = ClaudeLlm(model="anthropic/claude-opus-5-5")
        seen = []

        async def three():
            async def watched():
                out = await llm.invoke("same prompt")
                seen.append(fake.peak)
                return out
            return await asyncio.gather(watched(), watched(), watched())

        self.assertEqual(len(self.run_llm(fake, three)), 3)
        # The first went alone: the peak was still 1 when it came back.
        self.assertEqual(seen[0], 1)
        self.assertEqual(len(fake.requests), 3)

    def test_different_prompts_do_not_wait_for_each_other(self):
        fake = FakeAnthropic([reply(), reply()])
        llm = ClaudeLlm(model="anthropic/claude-opus-5-5")

        async def two():
            return await asyncio.gather(llm.invoke("one"), llm.invoke("two"))

        self.run_llm(fake, two)
        self.assertEqual(fake.peak, 2)

    def test_a_failed_first_call_releases_the_others(self):
        fake = FakeAnthropic([RuntimeError("overloaded"), reply()])
        llm = ClaudeLlm(model="anthropic/claude-opus-5-5")

        async def two():
            return await asyncio.gather(llm.invoke("p"), llm.invoke("p"), return_exceptions=True)

        results = self.run_llm(fake, two)
        self.assertIsInstance(results[0], RuntimeError)
        self.assertEqual(results[1], "Probability: 42%")


def call(t, usd, tour="futureeval", model="claude-opus-5-5", run="r1", phase="forecast"):
    return {"t": t.isoformat(timespec="seconds"), "run": run, "tour": tour, "q": 1, "phase": phase,
            "provider": "anthropic", "model": model, "in": 1, "out": 1, "searches": 0, "usd": usd}


class PlanTests(LedgerCase):
    def write(self, *lines):
        with open(self.path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")

    def days(self, usd_per_day, days=10):
        start = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
        self.write({"t": "2026-10-20T00:00:00+00:00", "kind": "start"},
                   *[call(start + timedelta(days=d), usd_per_day) for d in range(days)])

    def test_three_opus_forecasts_while_the_pace_fits(self):
        self.days(2.0)   # about $62 by the cycle's end
        self.assertEqual(spend.prize_forecasts(NOW), 3)

    def test_two_when_the_pace_would_end_over_budget(self):
        self.days(4.0)   # $40 so far, about $122 by the cycle's end
        self.assertEqual(spend.prize_forecasts(NOW), 2)

    def test_none_once_the_budget_is_spent(self):
        self.days(9.5)   # $95, past the 0.9 throttle
        self.assertEqual(spend.prize_forecasts(NOW), 0)

    def test_no_ledger_keeps_opus_because_over_budget_cannot_tell(self):
        self.assertEqual(spend.prize_forecasts(NOW), 3)


class DegradedAlertTests(PlanTests):
    def after(self, ok=True):
        sent = []

        def send(text):
            sent.append(text)
            return ok

        with mock.patch("sys.stdout"):
            spend.after_run(NOW, send=send)
        return sent

    def test_a_prize_question_forecast_without_opus_alerts_once_a_day(self):
        self.write({"t": "2026-10-20T00:00:00+00:00", "kind": "start"},
                   call(NOW, 0.01, model="claude-haiku-5-5", run="r9"))
        sent = self.after()
        self.assertEqual(len(sent), 1)
        self.assertIn("sin Opus", sent[0])
        self.assertEqual(self.after(), [])

    def test_opus_answering_any_of_them_is_not_degraded(self):
        self.write({"t": "2026-10-20T00:00:00+00:00", "kind": "start"},
                   call(NOW, 0.01, model="claude-haiku-5-5", run="r9"), call(NOW, 0.1, run="r9"))
        self.assertEqual(self.after(), [])

    def test_haiku_is_expected_in_minibench_and_the_cup_and_in_other_phases(self):
        self.write({"t": "2026-10-20T00:00:00+00:00", "kind": "start"},
                   call(NOW, 0.01, tour="minibench", model="claude-haiku-5-5", run="r9"),
                   call(NOW, 0.01, tour="cup", model="claude-haiku-5-5", run="r9"),
                   call(NOW, 0.01, model="claude-haiku-5-5", run="r9", phase="research"))
        self.assertEqual(self.after(), [])

    def test_no_alert_when_the_budget_is_why(self):
        self.days(9.5)
        self.write(call(NOW, 0.01, model="claude-haiku-5-5", run="r9"))
        sent = [text for text in self.after() if "sin Opus" in text]
        self.assertEqual(sent, [])

    def test_a_failed_send_is_retried_next_run(self):
        self.write({"t": "2026-10-20T00:00:00+00:00", "kind": "start"},
                   call(NOW, 0.01, model="claude-haiku-5-5", run="r9"))
        self.assertEqual(len(self.after(ok=False)), 1)
        self.assertEqual(len(self.after()), 1)


class MainWiringTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "main.py"), encoding="utf-8") as handle:
            self.src = handle.read()
        self.tail = self.src[self.src.index("opus_forecasts = spend.prize_forecasts()"):]

    def test_opus_for_the_prize_tournaments_and_haiku_for_the_rest(self):
        fe = self.tail.index('spend.set_tournament("futureeval")')
        mb = self.tail.index('spend.set_tournament("minibench")')
        cup = self.tail.index('spend.set_tournament("cup")')
        mp = self.tail.index('spend.set_tournament("market_pulse")')
        self.assertIn("forecast_with_opus()", self.tail[fe:mb])
        self.assertIn("forecast_with_haiku()", self.tail[mb:mb + 200])
        self.assertIn("forecast_with_haiku()", self.tail[cup - 120:cup])
        self.assertIn("forecast_with_opus()", self.tail[mp:mp + 200])

    def test_the_cheap_forecaster_is_the_summarizers_model(self):
        self.assertIn('if purpose == "default" and self.cheap_forecast:\n            purpose = "summarizer"',
                      self.src)


if __name__ == "__main__":
    unittest.main()
