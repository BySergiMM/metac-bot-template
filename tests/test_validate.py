"""Validation of the offline reconstruction against Metaculus' own scores."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from research.scorer import EXACT, score_track_record
from research.track_record import load_track_record
from research.validate import pearson, validate
from tests import fixtures

EXPECTED_BINARY_SPOT_PEER = 51.70039921703093


class PearsonTests(unittest.TestCase):
    def test_perfect_positive(self):
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)

    def test_perfect_negative(self):
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]), -1.0)

    def test_undefined_cases_return_none_rather_than_zero(self):
        """Correlation with a constant series is meaningless; returning 0
        would read as 'no relationship', which is a different claim."""
        self.assertIsNone(pearson([1.0], [2.0]))
        self.assertIsNone(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(pearson([1.0, 2.0], [1.0, 2.0, 3.0]))


class ValidationBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def build(self, name, questions, forecasts, scores):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, name), questions, forecasts, scores
        )
        record = load_track_record(directory)
        results, _summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        return record, results


class ExactReproductionTests(ValidationBase):
    def test_matching_scores_produce_an_exact_verdict(self):
        record, results = self.build(
            "exact",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8), fixtures.geometric_mean_row(1, 0.5, 11)],
            [fixtures.score_row(1, EXPECTED_BINARY_SPOT_PEER, coverage=1.0)],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.spot_peer.tier, EXACT)
        self.assertEqual(report.spot_peer.n_compared, 1)
        self.assertLess(report.spot_peer.max_absolute_error, 1e-9)
        self.assertEqual(report.spot_peer.within_tolerance, 1)
        self.assertEqual(report.coverage.n_matching, 1)
        self.assertIn("EXACT REPRODUCTION", report.verdict)

    def test_a_wrong_reconstruction_is_reported_as_partial(self):
        record, results = self.build(
            "wrong",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8), fixtures.geometric_mean_row(1, 0.5, 11)],
            [fixtures.score_row(1, 999.0, coverage=1.0)],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.spot_peer.tier, EXACT)
        self.assertGreater(report.spot_peer.max_absolute_error, 900)
        self.assertEqual(report.spot_peer.within_tolerance, 0)
        self.assertIn("PARTIAL REPRODUCTION", report.verdict)
        self.assertEqual(report.spot_peer.worst[0]["question_id"], 1)


class CoverageCheckTests(ValidationBase):
    def test_coverage_mismatch_is_surfaced(self):
        # Metaculus says we were covered; our forecast interval closed before
        # the spot instant, so we say we were not.
        record, results = self.build(
            "mismatch",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID)],
            [fixtures.score_row(1, 12.0, coverage=1.0)],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.coverage.n_compared, 1)
        self.assertEqual(report.coverage.n_matching, 0)
        self.assertEqual(len(report.coverage.mismatches), 1)
        self.assertIn("RECONSTRUCTION DISAGREES", report.verdict)

    def test_coverage_zero_on_both_sides_counts_as_a_match(self):
        record, results = self.build(
            "zero",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID)],
            [fixtures.score_row(1, 0.0, coverage=0.0)],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.coverage.n_matching, 1)

    def test_other_users_scores_are_ignored(self):
        record, results = self.build(
            "others",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [
                fixtures.score_row(1, 10.0, user_id=fixtures.OTHER_USER_ID, user_username="someone"),
                fixtures.score_row(1, 20.0, user_id=None, user_username="recency_weighted"),
            ],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.n_metaculus_spot_peer_rows, 0)
        self.assertIn("NO GROUND TRUTH", report.verdict)


class UnavailablePathTests(ValidationBase):
    def test_missing_denominator_falls_back_to_the_inversion_diagnostic(self):
        record, results = self.build(
            "nogm",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],  # no geometric_mean rows
            [fixtures.score_row(1, EXPECTED_BINARY_SPOT_PEER, coverage=1.0)],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.spot_peer.tier, "UNAVAILABLE")
        self.assertIsNotNone(report.spot_peer.blocked_reason)
        self.assertIn("geometric_mean_of_other_forecasters", report.spot_peer.missing_terms)
        self.assertEqual(report.inversion.n_evaluated, 1)
        self.assertEqual(report.inversion.n_valid_probability, 1)
        self.assertIn("COVERAGE REPRODUCED, PEER SCORE UNAVAILABLE", report.verdict)

    def test_inversion_flags_an_impossible_implied_probability(self):
        """A hugely negative score against a p of 0.9 implies a peer aggregate
        above 1, which cannot be a probability -- so something upstream is
        wrong and the diagnostic must say so."""
        record, results = self.build(
            "impossible",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.9)],
            [fixtures.score_row(1, -500.0, coverage=1.0)],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.inversion.n_invalid_probability, 1)
        self.assertFalse(report.inversion.examples[0]["valid_probability"])

    def test_score_type_census_is_recorded(self):
        record, results = self.build(
            "types",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [
                fixtures.score_row(1, 1.0, score_type="spot_peer"),
                fixtures.score_row(1, 2.0, score_type="peer"),
                fixtures.score_row(1, 3.0, score_type="baseline"),
            ],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.metaculus_score_types_present["spot_peer"], 1)
        self.assertEqual(report.metaculus_score_types_present["peer"], 1)
        self.assertEqual(report.metaculus_score_types_present["baseline"], 1)

    def test_no_scores_at_all_reports_no_ground_truth(self):
        record, results = self.build(
            "empty",
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [],
        )
        report = validate(record, results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.coverage.n_compared, 0)
        self.assertIn("NO GROUND TRUTH", report.verdict)


if __name__ == "__main__":
    unittest.main()
