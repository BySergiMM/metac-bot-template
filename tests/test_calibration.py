"""Calibration of the current system: which forecast counts, where the outcome
sits on the question's axis, and the aggregates the weekly report sends.

Synthetic questions only (ids in the 900000s): the repository is public.
"""

from __future__ import annotations

import math
import unittest
from datetime import date

from research import calibration as cal
from research import weekly_report as wr

NEW = "2026-10-20T12:00:00Z"
OLD = "2026-10-01T12:00:00Z"
CLOSE = "2026-10-25T12:00:00Z"


def uniform_cdf(n=201, below=0.02, above=0.02):
    """Uniform inside the range, ``below``/``above`` past the bounds."""
    return [below + (1 - below - above) * i / (n - 1) for i in range(n)]


def numeric(resolution, cdf=None, start=NEW, zero=None, qtype="numeric",
            lo=0.0, hi=100.0, open_lower=True, open_upper=True):
    return {"id": 900001, "type": qtype, "resolution": resolution,
            "actual_close_time": CLOSE, "open_lower_bound": open_lower,
            "open_upper_bound": open_upper,
            "scaling": {"range_min": lo, "range_max": hi, "zero_point": zero},
            "my_forecasts": {"history": [{"start_time": start,
                                          "forecast_values": cdf or uniform_cdf()}]}}


def binary(resolution, p, start=NEW):
    return {"id": 900002, "type": "binary", "resolution": resolution,
            "actual_close_time": CLOSE,
            "my_forecasts": {"history": [{"start_time": start, "forecast_values": [1 - p, p]}]}}


def multiple_choice(resolution, values, start=NEW):
    return {"id": 900003, "type": "multiple_choice", "resolution": resolution,
            "options": ["A", "B", "C"], "actual_close_time": CLOSE,
            "my_forecasts": {"history": [{"start_time": start, "forecast_values": values}]}}


class ScoredForecastTests(unittest.TestCase):
    def test_the_last_forecast_before_the_close_counts(self):
        q = binary("yes", 0.7)
        q["my_forecasts"]["history"] = [
            {"start_time": "2026-10-21T00:00:00Z", "forecast_values": [0.4, 0.6]},
            {"start_time": "2026-10-23T00:00:00Z", "forecast_values": [0.2, 0.8]},
            {"start_time": "2026-10-26T00:00:00Z", "forecast_values": [0.9, 0.1]},
        ]
        self.assertEqual(cal.scored_forecast(q)["forecast_values"], [0.2, 0.8])

    def test_a_forecast_of_the_old_system_is_left_out(self):
        self.assertIsNone(cal.scored_forecast(binary("yes", 0.7, start=OLD)))

    def test_epoch_seconds_are_read_too(self):
        q = binary("yes", 0.7, start=1761000000)  # 2025: before the cutoff
        self.assertIsNone(cal.scored_forecast(q))


class TimingTests(unittest.TestCase):
    def test_a_question_left_out_says_when_its_forecasts_were_made(self):
        q = binary("yes", 0.7, start=OLD)
        q["scheduled_close_time"] = CLOSE
        s = cal.summarise([q])
        self.assertEqual(s["not_current"], [{
            "id": 900002, "type": "binary", "records": 1, "first": "2026-10-01T12:00Z",
            "last": "2026-10-01T12:00Z", "actual_close": "2026-10-25T12:00Z",
            "scheduled_close": "2026-10-25T12:00Z", "resolved": None}])
        self.assertNotIn("0.7", str(s["not_current"]))


class LocationTests(unittest.TestCase):
    def test_linear_log_and_out_of_range(self):
        self.assertAlmostEqual(cal.location(numeric("25")), 0.25)
        q = numeric("10", lo=1.0, hi=100.0, zero=0.0)
        self.assertAlmostEqual(cal.location(q), 0.5)
        self.assertEqual(cal.location(numeric("below_lower_bound")), -math.inf)
        self.assertEqual(cal.location(numeric("above_upper_bound")), math.inf)
        self.assertGreater(cal.location(numeric("150")), 1)

    def test_a_date_resolution(self):
        q = numeric("2026-11-11T00:00:00Z", qtype="date",
                    lo=1793836800.0, hi=1794700800.0)  # 2026-11-05 .. 2026-11-15
        self.assertAlmostEqual(cal.location(q), 0.6)

    def test_an_unreadable_resolution_is_none(self):
        self.assertIsNone(cal.location(numeric("n/a")))
        self.assertIsNone(cal.location(numeric("5", lo=10.0, hi=10.0)))

    def test_cdf_at_interpolates(self):
        self.assertAlmostEqual(cal.cdf_at([0.0, 0.5, 1.0], 0.25), 0.25)
        self.assertEqual(cal.cdf_at([0.1, 0.5, 0.9], -1), 0.1)
        self.assertEqual(cal.cdf_at([0.1, 0.5, 0.9], 2), 0.9)


class SummaryTests(unittest.TestCase):
    def test_the_aggregates(self):
        qs = [numeric("50"), numeric("15"), numeric("95"), numeric("above_upper_bound"),
              numeric("50", start=OLD), binary("yes", 0.8), binary("no", 0.3),
              binary("yes", 0.9, start=OLD), {"type": "binary", "resolution": "annulled"}]
        s = cal.summarise(qs)
        self.assertEqual(s["binary"], 2)
        self.assertAlmostEqual(s["brier"], (0.04 + 0.09) / 2)
        self.assertEqual(s["continuous"], 4)
        self.assertEqual(s["in_iqr"], 1)   # 50 only
        self.assertEqual(s["in_80"], 2)    # 50 and 15
        self.assertEqual(s["outside"], 1)
        self.assertAlmostEqual(s["past_open_bound"], 0.02)

    def test_left_out_questions_are_counted_by_reason(self):
        no_values = binary("yes", 0.7)
        no_values["my_forecasts"]["history"][0] = {"start_time": NEW, "probability_yes": 0.7}
        s = cal.summarise([binary("yes", 0.7, start=OLD), no_values, numeric("n/a"),
                           {"type": "binary", "resolution": "yes", "my_forecasts": None}])
        self.assertEqual(s["skipped"], {"binary:not_current_system": 1,
                                        "binary:no_values:probability_yes,start_time": 1,
                                        "numeric:unreadable_resolution": 1,
                                        "binary:no_history": 1})

    def test_multiple_choice(self):
        s = cal.summarise([multiple_choice("B", [0.2, 0.5, 0.3]),
                           multiple_choice("C", [0.6, 0.3, 0.1]),
                           multiple_choice("D", [0.6, 0.3, 0.1])])
        self.assertEqual(s["multiple_choice"], 2)
        self.assertAlmostEqual(s["mc_mean_p"], 0.3)
        self.assertEqual(s["mc_top"], 1)
        self.assertEqual(s["skipped"], {"multiple_choice:unreadable_resolution": 1})
        self.assertIn("   opción múltiple: 2 · prob. media en la que salió 30% · era la favorita 1/2",
                      cal.render(s))

    def test_closed_bounds_add_no_past_bound_mass(self):
        s = cal.summarise([numeric("50", open_lower=False, open_upper=False,
                                   cdf=uniform_cdf(below=0.0, above=0.0))])
        self.assertIsNone(s["past_open_bound"])

    def test_render_and_the_weekly_message(self):
        s = cal.summarise([numeric("50"), binary("no", 0.3)])
        lines = cal.render(s)
        self.assertIn("   binarias: 1 · Brier 0.090", lines)
        self.assertIn("dentro 25–75 100% (ideal 50%)", lines[-1])
        self.assertIn("masa fuera de límites abiertos 2.0%", lines[-1])
        msg = wr.render(date(2026, 10, 26), date(2026, 11, 1), [], {"standings": {}}, s)
        self.assertIn("Calibración del sistema actual", msg)
        self.assertEqual(cal.render(cal.summarise([]))[-1], "   aún ninguna")


if __name__ == "__main__":
    unittest.main()
