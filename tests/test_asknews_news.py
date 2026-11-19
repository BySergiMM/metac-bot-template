"""AskNews as one more research section (asknews_news.py). No network: the
SDK is a fake, and the ledger a temporary file."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

import asknews_news

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)


class FakeAsk:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.news = NS(search_news=self.search)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("quota")
        art = NS(pub_date=datetime(2026, 11, 15), eng_title="T", summary="S " * 10,
                 source_id="src", article_url="https://news.example/a")
        return NS(as_dicts=[art])


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ledger.jsonl")
        self.env = mock.patch.dict(os.environ, {"SPEND_LEDGER_PATH": self.path,
                                                "ASKNEWS_API_KEY": "k"}, clear=False)
        self.env.start()
        for var in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"):
            os.environ.pop(var, None)
        self.nosleep = mock.patch.object(asknews_news, "SPACING_SECONDS", 0)
        self.nosleep.start()
        self.unpaused = mock.patch.object(asknews_news, "PAUSED", False)
        self.unpaused.start()

    def tearDown(self):
        self.unpaused.stop()
        self.nosleep.stop()
        self.env.stop()
        self.dir.cleanup()

    def ledger(self, *lines):
        with open(self.path, "a", encoding="utf-8") as h:
            for line in lines:
                h.write(json.dumps(line) + "\n")

    def run_news(self, mode, ask):
        return asyncio.run(asknews_news.news("Will X?", mode, sdk_factory=lambda: ask))


class PlanTests(Base):
    def test_full_reads_latest_and_archive_latest_reads_latest(self):
        self.assertEqual(asknews_news.plan("full", 0), ["latest news", "news knowledge"])
        self.assertEqual(asknews_news.plan("latest", 0), ["latest news"])
        self.assertEqual(asknews_news.plan(None, 0), [])

    def test_the_archive_stops_first_then_everything(self):
        self.assertEqual(asknews_news.plan("full", asknews_news.STOP_ARCHIVE_AT), ["latest news"])
        self.assertEqual(asknews_news.plan("full", asknews_news.STOP_ALL_AT), [])
        self.assertLess(asknews_news.STOP_ALL_AT, asknews_news.MONTHLY_CALLS)

    def test_calls_count_this_calendar_month_only(self):
        lines = [
            {"t": "2026-10-31T23:00:00+00:00", "provider": "asknews", "searches": 5, "usd": 0},
            {"t": "2026-11-02T10:00:00+00:00", "provider": "asknews", "searches": 6, "usd": 0},
            {"t": "2026-11-02T10:00:00+00:00", "provider": "anthropic", "searches": 9, "usd": 0.1},
        ]
        self.assertEqual(asknews_news.calls_this_month(lines, NOW), 6)

    def test_credentials(self):
        self.assertEqual(asknews_news.credentials({"ASKNEWS_API_KEY": "a"}), {"api_key": "a"})
        self.assertEqual(asknews_news.credentials({"ASKNEWS_CLIENT_ID": "i", "ASKNEWS_SECRET": "s"}),
                         {"client_id": "i", "client_secret": "s"})
        self.assertEqual(asknews_news.credentials({"ASKNEWS_CLIENT_ID": "i"}), {})
        self.assertEqual(asknews_news.credentials({}), {})


class NewsTests(Base):
    def test_full_mode_makes_two_searches_and_records_six_calls(self):
        ask = FakeAsk()
        out = self.run_news("full", ask)
        self.assertTrue(out.startswith("## Recent news (AskNews"))
        self.assertIn("[2026-11-15]", out)
        self.assertEqual([c["strategy"] for c in ask.calls], ["latest news", "news knowledge"])
        with open(self.path, encoding="utf-8") as h:
            lines = [json.loads(l) for l in h]
        self.assertEqual(sum(l["searches"] for l in lines if l["provider"] == "asknews"), 6)

    def test_no_credential_or_no_mode_does_nothing(self):
        ask = FakeAsk()
        self.assertEqual(self.run_news(None, ask), "")
        with mock.patch.dict(os.environ, {"ASKNEWS_API_KEY": ""}):
            self.assertEqual(self.run_news("full", ask), "")
        self.assertEqual(ask.calls, [])

    def test_paused_nothing_is_called_or_counted(self):
        ask = FakeAsk()
        with mock.patch.object(asknews_news, "PAUSED", True):
            self.assertEqual(self.run_news("full", ask), "")
        self.assertEqual(ask.calls, [])
        self.assertFalse(os.path.exists(self.path))

    def test_paused_in_production_until_the_tier_is_granted(self):
        # 2026-11-19: the free tier reopens on 1 January; flip it then.
        self.unpaused.stop()
        try:
            self.assertTrue(asknews_news.PAUSED)
        finally:
            self.unpaused.start()

    def test_over_the_allowance_nothing_is_called(self):
        self.ledger({"t": datetime.now(timezone.utc).isoformat(), "provider": "asknews",
                     "searches": asknews_news.STOP_ALL_AT, "usd": 0, "q": 1, "phase": "research"})
        ask = FakeAsk()
        self.assertEqual(self.run_news("full", ask), "")
        self.assertEqual(ask.calls, [])

    def test_a_failure_logs_asknews_code_and_detail(self):
        class Err(Exception):
            code = 403011
            detail = "Usage limit reached for your plan"

        class Failing(FakeAsk):
            async def search(self, **kwargs):
                raise Err()

        with self.assertLogs("asknews_news", level="WARNING") as logs:
            self.assertEqual(self.run_news("latest", Failing()), "")
        self.assertIn("code=403011", logs.output[0])
        self.assertIn("Usage limit reached", logs.output[0])

    def test_a_failure_costs_only_the_news_and_is_still_counted(self):
        ask = FakeAsk(fail=True)
        self.assertEqual(self.run_news("latest", ask), "")
        with open(self.path, encoding="utf-8") as h:
            self.assertEqual(sum(json.loads(l)["searches"] for l in h), 1)


class WiringTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            self.src = h.read()
        self.tail = self.src[self.src.index("opus_forecasts = spend.prize_forecasts()"):]

    def test_modes_per_tournament(self):
        for tour, mode in (("futureeval", '"full"'), ("minibench", '"latest"')):
            at = self.tail.index(f'spend.set_tournament("{tour}")')
            self.assertIn(f"template_bot.asknews_mode = {mode}", self.tail[at:at + 400], tour)
        mp = self.tail.index('spend.set_tournament("market_pulse")\n')
        self.assertIn('template_bot.asknews_mode = "latest"', self.tail[mp:mp + 400])
        early = self.tail.index('spend.set_tournament("market_pulse_early")')
        self.assertIn("template_bot.asknews_mode = None", self.tail[early:early + 400])
        cup = self.tail.index('spend.set_tournament("cup")')
        self.assertNotIn("asknews_mode", self.tail[cup - 600:cup + 600])

    def test_off_by_default_and_added_not_replacing(self):
        self.assertIn("    asknews_mode = None\n", self.src)
        body = self.src[self.src.index("    async def _run_research(self, question"):]
        body = body[:body.index("    async def _run_research_body")]
        self.assertIn('body = f"{body}\\n\\n{news}" if body else news', body)


if __name__ == "__main__":
    unittest.main()
