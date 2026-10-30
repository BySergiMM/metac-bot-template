"""The sibling-question survey: which groups are ladders, and when the bot's
forecasts along one contradict each other. Synthetic posts only."""

from __future__ import annotations

import unittest

from research import siblings_survey as ss


def binary(label, p=None):
    q = {"type": "binary", "label": label}
    if p is not None:
        q["my_forecasts"] = {"latest": {"forecast_values": [1 - p, p]}}
    return q


class OptionTests(unittest.TestCase):
    def test_numbers_money_and_months_order(self):
        self.assertEqual(ss.option_key("$1.5B"), 1.5e9)
        self.assertEqual(ss.option_key("≥ 40%"), 40.0)
        self.assertLess(ss.option_key("March 2027"), ss.option_key("June 2027"))
        self.assertIsNone(ss.option_key("Tesla"))

    def test_a_ladder_needs_three_distinct_ordered_options(self):
        self.assertTrue(ss.is_ladder([binary("10"), binary("20"), binary("30")]))
        self.assertFalse(ss.is_ladder([binary("10"), binary("20")]))
        self.assertFalse(ss.is_ladder([binary("Tesla"), binary("Meta"), binary("SpaceX")]))


class CoherenceTests(unittest.TestCase):
    def test_binaries_must_be_monotone_either_way(self):
        down = [binary("10", 0.9), binary("20", 0.6), binary("30", 0.2)]
        self.assertEqual(ss.ladder_coherence(down), "coherent")
        bad = [binary("10", 0.9), binary("20", 0.3), binary("30", 0.5)]
        self.assertEqual(ss.ladder_coherence(bad), "incoherent")
        self.assertEqual(ss.ladder_coherence([binary("10", 0.9), binary("20"), binary("30")]),
                         "unforecast")

    def test_numeric_medians(self):
        def numeric(label, med):
            cdf = [0.0 if i / 200 < med else 1.0 for i in range(201)]
            return {"type": "numeric", "label": label, "scaling": {"range_min": 0, "range_max": 100},
                    "my_forecasts": {"latest": {"forecast_values": cdf}}}
        self.assertEqual(ss.ladder_coherence([numeric("2026", .2), numeric("2027", .4),
                                              numeric("2028", .6)]), "coherent")
        self.assertEqual(ss.ladder_coherence([numeric("2026", .2), numeric("2027", .6),
                                              numeric("2028", .4)]), "incoherent")

    def test_entities(self):
        self.assertEqual(ss.entity("What will Apple (AAPL) report?"), "AAPL")
        self.assertEqual(ss.entity("Will Tesla deliver 500k cars?"), "Tesla")


if __name__ == "__main__":
    unittest.main()
