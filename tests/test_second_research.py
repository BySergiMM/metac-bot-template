"""The second, independent research pass on FutureEval and Market Pulse.

It searches alongside the main rundown without seeing it, both reach the
forecasters, it never costs the main rundown, and it runs only while the
cycle's projected Claude spend leaves room. No network: claude_research is
mocked.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import spend
from tests._real_forecasting_tools import real_forecasting_tools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_main():
    """main.py without its __main__ block (see test_prompt_wiring)."""
    with real_forecasting_tools():
        spec = importlib.util.spec_from_file_location(
            "second_research_main", os.path.join(ROOT, "main.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


main = load_main()
Bot = main.SummerTemplateBot2026


def fake_bot(on=True, depth="full"):
    return SimpleNamespace(second_research=on, research_depth=depth, SECOND_ANGLE=Bot.SECOND_ANGLE)


def run(bot, research):
    with mock.patch.object(main.claude_research, "research", research):
        return asyncio.run(Bot._claude_research(bot, "PROMPT"))


class SecondPassTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def research(self, fail_second=False, fail_main=False, empty_second=False):
        async def fake(prompt, depth="full"):
            self.calls.append((prompt, depth))
            second = prompt.startswith(Bot.SECOND_ANGLE)
            if second and fail_second:
                raise RuntimeError("boom")
            if not second and fail_main:
                raise RuntimeError("main down")
            if second:
                return "" if empty_second else "DECIDING FACTS"
            return "MAIN RUNDOWN"
        return fake

    def test_both_passes_reach_the_forecasters(self):
        out = run(fake_bot(), self.research())
        self.assertTrue(out.startswith("MAIN RUNDOWN"))
        self.assertIn("## Second, independent research pass", out)
        self.assertIn("DECIDING FACTS", out)
        # The second pass is light and never sees the first one's text.
        second = next(c for c in self.calls if c[0].startswith(Bot.SECOND_ANGLE))
        self.assertEqual(second, (f"{Bot.SECOND_ANGLE}\n\nPROMPT", "light"))
        self.assertIn(("PROMPT", "full"), self.calls)

    def test_off_or_not_full_means_one_pass(self):
        self.assertEqual(run(fake_bot(on=False), self.research()), "MAIN RUNDOWN")
        self.assertEqual(run(fake_bot(depth="light"), self.research()), "MAIN RUNDOWN")
        self.assertEqual([c[1] for c in self.calls], ["full", "light"])
        self.assertFalse(any(c[0].startswith(Bot.SECOND_ANGLE) for c in self.calls))

    def test_a_failed_or_empty_second_pass_costs_only_itself(self):
        self.assertEqual(run(fake_bot(), self.research(fail_second=True)), "MAIN RUNDOWN")
        self.assertEqual(run(fake_bot(), self.research(empty_second=True)), "MAIN RUNDOWN")

    def test_a_failed_main_pass_still_raises_for_the_fallback(self):
        with self.assertRaises(RuntimeError):
            run(fake_bot(), self.research(fail_main=True))

    def test_main_sets_it_for_the_prize_tournaments_only(self):
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            src = h.read()
        tail = src[src.index("second_research = (prize_depth"):]
        fe = tail.index('spend.set_tournament("futureeval")')
        mb = tail.index('spend.set_tournament("minibench")')
        mp = tail.index('spend.set_tournament("market_pulse")')
        self.assertIn("template_bot.second_research = second_research", tail[fe:mb])
        self.assertIn("template_bot.second_research = False", tail[mb:mb + 200])
        self.assertIn("template_bot.second_research = second_research", tail[mp:mp + 200])
        self.assertIn('os.getenv("SECOND_RESEARCH", "on") != "off"', tail)
        self.assertIn("spend.room_for_extras()", tail)


class RoomForExtrasTests(unittest.TestCase):
    NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)  # day 10 of the 28th cycle

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ledger.jsonl")
        self.env = mock.patch.dict(os.environ, {
            "SPEND_LEDGER_PATH": self.path, "CLAUDE_BUDGET_USD": "100",
            "CLAUDE_BUDGET_CYCLE_DAY": "28"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.dir.cleanup()

    def ledger(self, usd_per_day, days=10):
        with open(self.path, "w", encoding="utf-8") as h:
            h.write(json.dumps({"t": "2026-10-20T00:00:00+00:00", "kind": "start"}) + "\n")
            for d in range(days):
                t = datetime(2026, 10, 28, 12, tzinfo=timezone.utc) + timedelta(days=d)
                h.write(json.dumps({"t": t.isoformat(), "provider": "anthropic", "usd": usd_per_day}) + "\n")

    def test_a_slow_cycle_leaves_room(self):
        self.ledger(1.5)   # about $46 by the cycle's end
        self.assertTrue(spend.room_for_extras(self.NOW))

    def test_a_pace_past_80_percent_stops_the_extras(self):
        self.ledger(3.0)   # about $93 by the cycle's end
        self.assertFalse(spend.room_for_extras(self.NOW))

    def test_no_ledger_or_a_broken_one_means_no_extras(self):
        self.assertFalse(spend.room_for_extras(self.NOW))
        with open(self.path, "w", encoding="utf-8") as h:
            h.write("not json\n")
        self.assertFalse(spend.room_for_extras(self.NOW))


if __name__ == "__main__":
    unittest.main()
