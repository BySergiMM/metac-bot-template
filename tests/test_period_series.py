"""Market Pulse's biweekly series from their own sources (period_series).
A fake fetcher stands in for FRED and Yahoo; synthetic numbers."""

from __future__ import annotations

import json
import math
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as Q

import live_context
import period_series as ps

NOW = datetime(2026, 10, 30, 21, tzinfo=timezone.utc)  # a Friday


def question(text, option, resolves, criteria=""):
    return Q(question_text=text, resolution_criteria=criteria, background_info="", fine_print="",
             group_question_option=option, scheduled_resolution_time=resolves)


def weekdays(until=date(2026, 10, 30), n=70):
    days, d = [], until
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def yahoo(closes, days, highs=None):
    stamps = [int(datetime(d.year, d.month, d.day, 13, 30, tzinfo=timezone.utc).timestamp()) for d in days]
    return json.dumps({"chart": {"result": [{"meta": {"gmtoffset": -14400}, "timestamp": stamps,
                                             "indicators": {"quote": [{"close": closes,
                                                                       "high": highs or closes}]}}]}}).encode()


class Web:
    def __init__(self, routes):
        self.routes = routes

    def __call__(self, url, headers):
        for prefix, body in self.routes.items():
            if url.startswith(prefix):
                return body
        raise OSError(url)


class PeriodTests(unittest.TestCase):
    def test_the_period_and_its_year(self):
        q = question("x", "Oct 26 - Nov 6", datetime(2026, 11, 9, tzinfo=timezone.utc))
        self.assertEqual(ps.period(q), (date(2026, 10, 26), date(2026, 11, 6)))
        q = question("x", "Dec 21 - Jan 1", datetime(2027, 1, 4, tzinfo=timezone.utc))
        self.assertEqual(ps.period(q), (date(2026, 12, 21), date(2027, 1, 1)))
        self.assertIsNone(ps.period(question("x", "Tesla", datetime(2026, 11, 9, tzinfo=timezone.utc))))


class FredTests(unittest.TestCase):
    def test_level_and_the_move_to_the_period_end(self):
        days = weekdays()
        rows = "observation_date,DGS10\n" + "\n".join(
            f"{d.isoformat()},{4.0 + (0.05 if i % 2 else -0.05):.2f}" for i, d in enumerate(days))
        rows += "\n2026-10-31,."
        q = question("What will be the ending value of the UST 10Y Yield?", "Nov 9 - Nov 20",
                     datetime(2026, 11, 23, tzinfo=timezone.utc),
                     "U.S. Treasury Securities at 10-Year Constant Maturity published on the FRED website")
        out = ps.fred_level(q, Web({"https://fred.stlouisfed.org/": rows.encode()}), NOW)
        self.assertIn("FRED DGS10: latest 4.05 on 2026-10-30", out)
        self.assertIn("15 sessions to 2026-11-20", out)
        self.assertIn("daily change s.d. 0.101", out)  # sample s.d. of +-0.1
        self.assertIn("1 s.d. change ±0.39", out)   # 0.1 x sqrt(15)

    def test_a_question_on_something_else_fetches_nothing(self):
        q = question("Will Tesla deliver 500k cars?", "Nov 9 - Nov 20", NOW)
        self.assertEqual(ps.build(q, Web({}), NOW), "")


class ReturnDifferenceTests(unittest.TestCase):
    def test_the_period_so_far_matches_the_resolution_formula(self):
        days = weekdays()
        a = [100.0 * (1.01 if i % 2 else 1.0) for i in range(len(days))]
        b = [50.0] * len(days)
        # Period Oct 26 - Nov 6: P0 is the Oct 23 close.
        i0, i1 = days.index(date(2026, 10, 23)), days.index(date(2026, 10, 30))
        a[i0], a[i1] = 100.0, 110.0
        b[i0], b[i1] = 50.0, 51.0
        web = Web({"https://query1.finance.yahoo.com/v8/finance/chart/CL%3DF": yahoo(a, days),
                   "https://query1.finance.yahoo.com/v8/finance/chart/ES%3DF": yahoo(b, days)})
        q = question("How much will Crude Oil Futures total price returns exceed S&P 500 Futures?",
                     "Oct 26 - Nov 6", datetime(2026, 11, 9, tzinfo=timezone.utc))
        out = ps.return_difference(q, web, NOW)
        self.assertIn("CL=F minus ES=F", out)
        self.assertIn("this period so far (closes 2026-10-23 to 2026-10-30): +8.00 pp", out)  # 10% - 2%
        self.assertIn("5 sessions left", out)

    def test_the_order_follows_the_question(self):
        q = question("How much will Nvidia's stock price returns exceed Apple's?", "Oct 26 - Nov 6", NOW)
        self.assertEqual(ps._pair(q), ("NVDA", "AAPL"))
        self.assertIsNone(ps._pair(question("What will Nvidia's revenue be?", "Oct 26 - Nov 6", NOW)))


class VixTests(unittest.TestCase):
    def test_the_maximum_so_far_is_a_floor(self):
        days = weekdays(n=250)
        highs = [15.0 + (i % 7) for i in range(len(days))]
        highs[days.index(date(2026, 10, 28))] = 30.0
        web = Web({"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX": yahoo(highs, days, highs)})
        q = question("What will be the maximum intraday value of the VIX over these biweekly periods?",
                     "Oct 26 - Nov 6", datetime(2026, 11, 9, tzinfo=timezone.utc))
        out = ps.vix_maximum(q, web, NOW)
        self.assertIn("highest so far this period: 30.00 (the outcome cannot be lower)", out)
        self.assertIn("over 10 consecutive sessions", out)


class WiringTests(unittest.TestCase):
    def test_live_data_carries_it(self):
        titles = [t for t, _ in live_context.SECTIONS]
        self.assertTrue(any("resolution source's own history" in t for t in titles))

    def test_a_failing_source_never_raises(self):
        q = question("What will be the ending value of the UST 10Y Yield?", "Nov 9 - Nov 20",
                     datetime(2026, 11, 23, tzinfo=timezone.utc), "10-Year Constant Maturity")
        self.assertEqual(ps.build(q, Web({}), NOW), "")


if __name__ == "__main__":
    unittest.main()
