"""Offline scoring.

The spot peer assertions check against values derived by hand from the formula
in ``scoring/score_math.py``::

    score = 100 * (N/(N-1)) * ln(p / gmp)      # halved for continuous

with p=0.8, gmp=0.5, N=11:

    100 * (11/10) * ln(1.6) = 110 * 0.4700036292457356 = 51.70039921703093

Asserting against an independently computed constant rather than against
whatever the implementation returns is the whole point: a test that recomputes
the formula it is testing proves only that the code is self-consistent.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import unittest

from research.scorer import (
    EXACT,
    UNAVAILABLE,
    brier_score,
    is_directional_hit,
    implied_geometric_mean,
    log_score,
    peer_factor,
    score_question,
    score_track_record,
    select_forecast_at_spot,
    select_geometric_mean_at_spot,
)
from research.track_record import load_track_record, parse_dt, read_forecast_csv, read_question_csv
from tests import fixtures

EXPECTED_BINARY_SPOT_PEER = 51.70039921703093
EXPECTED_CONTINUOUS_SPOT_PEER = 25.850199608515464


def _question(**kwargs):
    return read_question_csv(fixtures.questions_csv([fixtures.question_row(1, **kwargs)]))[0]


def _forecasts(*rows):
    return read_forecast_csv(fixtures.forecasts_csv(list(rows)))


class PrimitiveTests(unittest.TestCase):
    def test_peer_factor(self):
        self.assertEqual(peer_factor(2), 2.0)
        self.assertAlmostEqual(peer_factor(11), 1.1)
        self.assertAlmostEqual(peer_factor(101), 1.01)

    def test_peer_factor_degenerate_cases_are_zero_not_undefined(self):
        """``get_geometric_means`` stores 0 when there was at most one
        forecaster; a peer score against nobody is 0."""
        self.assertEqual(peer_factor(0), 0.0)
        self.assertEqual(peer_factor(1), 0.0)
        self.assertEqual(peer_factor(None), 0.0)

    def test_log_score(self):
        self.assertAlmostEqual(log_score(0.8), math.log(0.8))
        self.assertIsNone(log_score(None))

    def test_log_score_floors_zero_instead_of_returning_negative_infinity(self):
        value = log_score(0.0)
        self.assertTrue(math.isfinite(value))
        self.assertLess(value, -20)

    def test_brier(self):
        self.assertAlmostEqual(brier_score(0.8, True), 0.04)
        self.assertAlmostEqual(brier_score(0.8, False), 0.64)
        self.assertIsNone(brier_score(None, True))
        self.assertIsNone(brier_score(0.8, None))


class SelectionTests(unittest.TestCase):
    def test_selects_the_forecast_covering_the_spot_instant(self):
        rows = _forecasts(
            fixtures.forecast_row(1, 0.6, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID),
            fixtures.forecast_row(1, 0.9, start_time=fixtures.T_MID, end_time=None),
        )
        spot = parse_dt(fixtures.T_REVEAL).timestamp()
        chosen = select_forecast_at_spot(rows, spot)
        self.assertEqual(chosen.probability_of_resolution, 0.9)

    def test_returns_none_when_nothing_is_active(self):
        rows = _forecasts(
            fixtures.forecast_row(1, 0.6, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID)
        )
        self.assertIsNone(select_forecast_at_spot(rows, parse_dt(fixtures.T_REVEAL).timestamp()))
        self.assertIsNone(select_forecast_at_spot(rows, None))

    def test_geometric_mean_selection_is_strictly_before_the_spot(self):
        """``if gm.timestamp < spot_forecast_timestamp`` -- an aggregate
        recomputed exactly at the spot instant is not the one used."""
        rows = _forecasts(
            fixtures.geometric_mean_row(1, 0.5, 11, start_time=fixtures.T_OPEN),
            fixtures.geometric_mean_row(1, 0.7, 12, start_time=fixtures.T_REVEAL),
        )
        spot = parse_dt(fixtures.T_REVEAL).timestamp()
        chosen = select_geometric_mean_at_spot(rows, spot)
        self.assertEqual(chosen.probability_of_resolution, 0.5)

    def test_geometric_mean_selection_takes_the_latest_eligible_entry(self):
        rows = _forecasts(
            fixtures.geometric_mean_row(1, 0.4, 9, start_time=fixtures.T_OPEN),
            fixtures.geometric_mean_row(1, 0.5, 11, start_time=fixtures.T_MID),
        )
        chosen = select_geometric_mean_at_spot(rows, parse_dt(fixtures.T_REVEAL).timestamp())
        self.assertEqual(chosen.probability_of_resolution, 0.5)


class SpotPeerTests(unittest.TestCase):
    def test_exact_binary_spot_peer(self):
        result = score_question(
            _question(),
            _forecasts(fixtures.forecast_row(1, 0.8, probability_yes=0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertEqual(result.spot_peer_tier, EXACT)
        self.assertAlmostEqual(result.spot_peer, EXPECTED_BINARY_SPOT_PEER, places=9)
        self.assertEqual(result.reproduced_coverage, 1.0)
        self.assertAlmostEqual(result.spot_log_score, math.log(0.8))
        self.assertAlmostEqual(result.spot_brier, 0.04)

    def test_continuous_questions_are_halved(self):
        result = score_question(
            _question(question_type="numeric"),
            _forecasts(fixtures.forecast_row(1, 0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertAlmostEqual(result.spot_peer, EXPECTED_CONTINUOUS_SPOT_PEER, places=9)

    def test_discrete_counts_as_continuous(self):
        result = score_question(
            _question(question_type="discrete"),
            _forecasts(fixtures.forecast_row(1, 0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertAlmostEqual(result.spot_peer, EXPECTED_CONTINUOUS_SPOT_PEER, places=9)

    def test_brier_is_binary_only(self):
        result = score_question(
            _question(question_type="numeric"),
            _forecasts(fixtures.forecast_row(1, 0.8, probability_yes=0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertIsNone(result.spot_brier)

    def test_being_worse_than_the_crowd_scores_negative(self):
        result = score_question(
            _question(),
            _forecasts(fixtures.forecast_row(1, 0.2)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.4, 3)),
        )
        self.assertAlmostEqual(result.spot_peer, -103.97207708399179, places=9)

    def test_single_forecaster_scores_zero(self):
        result = score_question(
            _question(),
            _forecasts(fixtures.forecast_row(1, 0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 1)),
        )
        self.assertEqual(result.spot_peer, 0.0)
        self.assertEqual(result.spot_peer_tier, EXACT)


class DirectionalAccuracyTests(unittest.TestCase):
    """The "% of calls we got right" reading of accuracy.

    Added because a target was set on this number and it was not being
    measured at all. It lives NEXT TO the proper scores, never instead of
    them: see test_it_is_gameable_in_a_way_brier_catches for why.
    """

    def test_more_than_half_the_mass_on_the_realised_outcome_is_a_hit(self):
        self.assertIs(is_directional_hit(0.51), True)
        self.assertIs(is_directional_hit(0.99), True)

    def test_less_than_half_is_a_miss(self):
        self.assertIs(is_directional_hit(0.49), False)
        self.assertIs(is_directional_hit(0.01), False)

    def test_exactly_half_is_not_a_hit(self):
        """A coin flip expresses no direction. Counting it as correct would
        let a bot that answers 50% to everything report 100% accuracy."""
        self.assertIs(is_directional_hit(0.5), False)

    def test_no_forecast_is_neither_hit_nor_miss(self):
        """None, not False: folding a coverage failure into the accuracy
        number would mix the two things this lab keeps apart."""
        self.assertIsNone(is_directional_hit(None))

    def test_a_binary_question_gets_a_verdict(self):
        result = score_question(
            _question(),
            _forecasts(fixtures.forecast_row(1, 0.8, probability_yes=0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertIs(result.directional_hit, True)

    def test_a_non_binary_question_gets_none(self):
        """probability_of_resolution > 0.5 only means "the side we leaned on"
        when there are two outcomes. With k > 2 the modal option can hold well
        under half the mass, so the same test would score a correct call as a
        miss."""
        result = score_question(
            _question(question_type="numeric"),
            _forecasts(fixtures.forecast_row(1, 0.8, probability_yes=0.8)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertIsNone(result.directional_hit)

    def test_it_is_gameable_in_a_way_brier_catches(self):
        """The reason this metric is never reported alone.

        A forecaster that always says 99% on the side it leans scores a
        PERFECT hit rate on the questions it calls right, and its Brier is
        excellent there too -- but one wrong call at 99% costs ~0.96 of Brier,
        far more than the 0.25 a 50% answer would have cost. Accuracy cannot
        see that trade; Brier can, which is why both are printed together.
        """
        confident_and_right = brier_score(0.99, True)
        confident_and_wrong = brier_score(0.99, False)
        humble = brier_score(0.5, False)
        self.assertIs(is_directional_hit(0.99), True)
        self.assertLess(confident_and_right, humble)
        self.assertGreater(confident_and_wrong, humble)
        # One overconfident miss outweighs three overconfident hits.
        self.assertGreater(confident_and_wrong, 3 * (humble - confident_and_right))


class SkillMarginTests(unittest.TestCase):
    """A hit rate is unreadable without the trivial benchmark beside it.

    82% is excellent against a 55% base rate and worthless against an 85% one,
    and on this platform the cheap lean -- "will X happen by date Y" mostly
    resolves NO -- is exactly the one a bot can fall into without trying.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._n = 0

    def _summary(self, rows):
        """rows: (probability_yes, resolved_yes) per synthetic binary question."""
        questions, forecasts = [], []
        for index, (probability_yes, resolved_yes) in enumerate(rows, start=1):
            questions.append(fixtures.question_row(
                index, resolution="yes" if resolved_yes else "no"))
            # probability_of_resolution is the mass on the outcome that
            # happened, which is what the server fills in on resolution.
            on_outcome = probability_yes if resolved_yes else 1.0 - probability_yes
            forecasts.append(fixtures.forecast_row(
                index, on_outcome, probability_yes=probability_yes))
        self._n += 1
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "ds{0}".format(self._n)),
            questions, forecasts, [],
        )
        _results, summary = score_track_record(
            load_track_record(directory), user_id=fixtures.OUR_USER_ID)
        return summary

    def test_the_majority_benchmark_counts_the_question_mix_not_the_bot(self):
        # 8 of 10 resolve NO; a bot that always said NO would score 80%.
        rows = [(0.2, False)] * 8 + [(0.2, True)] * 2
        summary = self._summary(rows)
        self.assertEqual(summary.n_resolved_no, 8)
        self.assertEqual(summary.n_resolved_yes, 2)
        self.assertAlmostEqual(summary.majority_class_hit_rate, 0.8)

    def test_always_saying_no_scores_the_base_rate_and_zero_skill(self):
        """The lean this metric has to be able to expose."""
        rows = [(0.2, False)] * 8 + [(0.2, True)] * 2
        summary = self._summary(rows)
        self.assertAlmostEqual(summary.directional_hit_rate, 0.8)
        self.assertAlmostEqual(summary.directional_skill_margin, 0.0)

    def test_real_directional_skill_shows_a_positive_margin(self):
        rows = [(0.2, False)] * 8 + [(0.8, True)] * 2  # every call correct
        summary = self._summary(rows)
        self.assertAlmostEqual(summary.directional_hit_rate, 1.0)
        self.assertAlmostEqual(summary.majority_class_hit_rate, 0.8)
        self.assertAlmostEqual(summary.directional_skill_margin, 0.2)

    def test_a_marginal_miss_and_a_confident_miss_are_told_apart(self):
        """0.48 is a coin flip that landed badly; 0.05 is a wrong conviction.
        They cost very differently, so the aggregate separates them."""
        marginal = self._summary([(0.52, False), (0.8, True), (0.8, True)])
        confident = self._summary([(0.95, False), (0.8, True), (0.8, True)])
        self.assertGreater(
            marginal.mean_confidence_when_wrong,
            confident.mean_confidence_when_wrong,
        )


class MissingDataTests(unittest.TestCase):
    def test_no_geometric_mean_is_unavailable_not_approximated(self):
        result = score_question(_question(), _forecasts(fixtures.forecast_row(1, 0.8)))
        self.assertEqual(result.spot_peer_tier, UNAVAILABLE)
        self.assertIsNone(result.spot_peer)
        self.assertIn("geometric_mean_of_other_forecasters", result.spot_peer_missing)
        # the parts we can compute are still computed
        self.assertAlmostEqual(result.spot_log_score, math.log(0.8))
        self.assertEqual(result.reproduced_coverage, 1.0)

    def test_no_active_forecast_is_coverage_zero_and_an_exact_zero_score(self):
        """A forfeited question is exactly 0 under the spot rule; that is a
        known value, not a missing one."""
        result = score_question(
            _question(),
            _forecasts(
                fixtures.forecast_row(1, 0.8, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID)
            ),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertFalse(result.active_forecast)
        self.assertEqual(result.reproduced_coverage, 0.0)
        self.assertEqual(result.spot_peer, 0.0)
        self.assertEqual(result.spot_peer_tier, EXACT)

    def test_unresolved_question_is_not_scored(self):
        result = score_question(_question(resolution=None), _forecasts(fixtures.forecast_row(1, 0.8)))
        self.assertIn("resolution", result.spot_peer_missing)
        self.assertIsNone(result.spot_log_score)

    def test_annulled_question_is_not_scored(self):
        result = score_question(_question(resolution="annulled"), _forecasts(fixtures.forecast_row(1, 0.8)))
        self.assertIn("resolution", result.spot_peer_missing)

    def test_missing_probability_of_resolution_is_named(self):
        result = score_question(
            _question(),
            _forecasts(fixtures.forecast_row(1, None)),
            geometric_means=_forecasts(fixtures.geometric_mean_row(1, 0.5, 11)),
        )
        self.assertIn("probability_of_resolution", result.spot_peer_missing)
        self.assertIsNone(result.spot_log_score)

    def test_question_without_timestamps_reports_missing_spot_time(self):
        result = score_question(
            _question(cp_reveal_time=None, actual_close_time=None, scheduled_close_time=None),
            _forecasts(fixtures.forecast_row(1, 0.8)),
        )
        self.assertIn("spot_scoring_time", result.spot_peer_missing)


class InversionTests(unittest.TestCase):
    def test_inversion_recovers_the_geometric_mean_we_started_from(self):
        implied = implied_geometric_mean(
            EXPECTED_BINARY_SPOT_PEER, 0.8, num_forecasters=11, is_continuous=False
        )
        self.assertAlmostEqual(implied, 0.5, places=9)

    def test_inversion_handles_the_continuous_halving(self):
        implied = implied_geometric_mean(
            EXPECTED_CONTINUOUS_SPOT_PEER, 0.8, num_forecasters=11, is_continuous=True
        )
        self.assertAlmostEqual(implied, 0.5, places=9)

    def test_inversion_of_an_impossible_score_yields_an_invalid_probability(self):
        """The falsifiability property: a score that cannot come from our p
        implies a 'probability' above 1."""
        implied = implied_geometric_mean(-500.0, 0.9, num_forecasters=None, is_continuous=False)
        self.assertGreater(implied, 1.0)

    def test_inversion_returns_none_when_the_factor_collapses(self):
        self.assertIsNone(
            implied_geometric_mean(10.0, 0.5, num_forecasters=1, is_continuous=False)
        )


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_summary_aggregates_and_downgrades_the_tier(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "ds"),
            [
                fixtures.question_row(1),
                fixtures.question_row(2, question_type="numeric", resolution="42"),
                fixtures.question_row(3, resolution=None),
            ],
            [
                fixtures.forecast_row(1, 0.8, probability_yes=0.8),
                fixtures.geometric_mean_row(1, 0.5, 11),
                fixtures.forecast_row(2, 0.3),  # no geometric mean for q2
                fixtures.forecast_row(3, 0.5),
            ],
            [],
        )
        record = load_track_record(directory)
        results, summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)

        self.assertEqual(summary.n_questions_in_dataset, 3)
        self.assertEqual(summary.n_resolved, 2)
        self.assertEqual(summary.n_scored, 2)
        self.assertEqual(summary.n_covered_at_spot, 2)
        self.assertEqual(summary.coverage_rate, 1.0)
        # one question lacks the peer denominator, so the aggregate total is
        # not claimed as exact
        self.assertEqual(summary.spot_peer_tier, UNAVAILABLE)
        self.assertIsNone(summary.total_spot_peer)
        self.assertAlmostEqual(
            summary.mean_spot_log_score, (math.log(0.8) + math.log(0.3)) / 2
        )
        self.assertEqual(summary.by_question_type["binary"]["n"], 1)
        self.assertEqual(summary.by_question_type["numeric"]["n"], 1)
        self.assertEqual(summary.missing_inputs["geometric_mean_of_other_forecasters"], 1)

    def test_summary_is_exact_when_every_question_has_its_denominator(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "exact"),
            [fixtures.question_row(1), fixtures.question_row(2, weight=0.5)],
            [
                fixtures.forecast_row(1, 0.8),
                fixtures.geometric_mean_row(1, 0.5, 11),
                fixtures.forecast_row(2, 0.8),
                fixtures.geometric_mean_row(2, 0.5, 11),
            ],
            [],
        )
        record = load_track_record(directory)
        _results, summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(summary.spot_peer_tier, EXACT)
        self.assertAlmostEqual(summary.total_spot_peer, 2 * EXPECTED_BINARY_SPOT_PEER, places=6)
        self.assertAlmostEqual(
            summary.weighted_total_spot_peer, 1.5 * EXPECTED_BINARY_SPOT_PEER, places=6
        )
        self.assertAlmostEqual(summary.mean_spot_peer, EXPECTED_BINARY_SPOT_PEER, places=6)

    def test_missed_questions_are_counted(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "missed"),
            [fixtures.question_row(1), fixtures.question_row(2)],
            [fixtures.forecast_row(1, 0.8)],  # nothing at all for q2
            [],
        )
        record = load_track_record(directory)
        _results, summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(summary.n_scored, 2)
        self.assertEqual(summary.n_covered_at_spot, 1)
        self.assertEqual(summary.n_missed_at_spot, 1)
        self.assertEqual(summary.coverage_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
