"""Coverage analysis: production (what we played) and benchmark (what we can score)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from research.coverage import UniverseQuestion, benchmark_coverage, production_coverage
from research.scorer import score_track_record
from research.track_record import load_track_record
from tests import fixtures


class UniverseParsingTests(unittest.TestCase):
    def test_single_question_post(self):
        questions = UniverseQuestion.from_post_json(fixtures.post_json(10, 100), tournament="t")
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].question_id, 100)
        self.assertEqual(questions[0].question_type, "binary")
        self.assertEqual(questions[0].tournament, "t")

    def test_group_post_expands_to_every_subquestion(self):
        """Group subquestions are scored individually, so counting the post
        once would understate the tournament."""
        questions = UniverseQuestion.from_post_json(fixtures.group_post_json(11, [201, 202, 203]))
        self.assertEqual(len(questions), 3)
        self.assertEqual({q.question_id for q in questions}, {201, 202, 203})
        self.assertTrue(all(q.question_type == "numeric" for q in questions))

    def test_conditional_post_expands_to_both_branches(self):
        post = {
            "id": 12,
            "title": "cond",
            "status": "resolved",
            "conditional": {
                "question_yes": {"id": 301, "type": "binary"},
                "question_no": {"id": 302, "type": "binary"},
            },
        }
        questions = UniverseQuestion.from_post_json(post)
        self.assertEqual({q.question_id for q in questions}, {301, 302})

    def test_undecomposable_post_still_counts(self):
        """Dropping a post we cannot parse would flatter coverage, which is
        the opposite of what this report is for."""
        questions = UniverseQuestion.from_post_json({"id": 13, "title": "notebook"})
        self.assertEqual(len(questions), 1)
        self.assertIsNone(questions[0].question_id)
        self.assertEqual(questions[0].question_type, "unknown")


class ProductionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _record(self, questions, forecasts, scores):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "ds"), questions, forecasts, scores
        )
        record = load_track_record(directory)
        results, _summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        return record, results

    def test_counts_forecasted_and_uncovered(self):
        record, results = self._record(
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [fixtures.score_row(1, 20.0, coverage=1.0)],
        )
        universe = (
            UniverseQuestion.from_post_json(fixtures.post_json(1, 1, "binary"))
            + UniverseQuestion.from_post_json(fixtures.post_json(2, 2, "numeric"))
            + UniverseQuestion.from_post_json(fixtures.post_json(3, 3, "multiple_choice"))
        )
        report = production_coverage(
            universe, record, results=results, user_id=fixtures.OUR_USER_ID, tournament="t"
        )
        self.assertEqual(report.n_universe_questions, 3)
        self.assertEqual(report.n_forecasted, 1)
        self.assertEqual(report.n_not_forecasted, 2)
        self.assertAlmostEqual(report.coverage_rate, 1 / 3)
        self.assertEqual(report.universe_by_type, {"binary": 1, "numeric": 1, "multiple_choice": 1})
        self.assertEqual(report.uncovered_by_type, {"numeric": 1, "multiple_choice": 1})

    def test_duplicate_universe_entries_are_counted_once(self):
        record, results = self._record(
            [fixtures.question_row(1)], [fixtures.forecast_row(1, 0.8)], []
        )
        universe = UniverseQuestion.from_post_json(fixtures.post_json(1, 1)) * 3
        report = production_coverage(universe, record, results=results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.n_universe_questions, 1)

    def test_metaculus_coverage_flags_are_summarised(self):
        record, results = self._record(
            [fixtures.question_row(1), fixtures.question_row(2)],
            [fixtures.forecast_row(1, 0.8), fixtures.forecast_row(2, 0.4)],
            [
                fixtures.score_row(1, 20.0, coverage=1.0),
                fixtures.score_row(2, 0.0, coverage=0.0),
            ],
        )
        report = production_coverage([], record, results=results, user_id=fixtures.OUR_USER_ID)
        self.assertEqual(report.n_scored_by_metaculus, 2)
        self.assertEqual(report.n_scored_with_coverage_1, 1)
        self.assertEqual(report.n_scored_with_coverage_0, 1)

    def test_forfeit_estimate_is_labelled_a_proxy(self):
        record, results = self._record(
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [fixtures.score_row(1, 20.0, coverage=1.0)],
        )
        universe = (
            UniverseQuestion.from_post_json(fixtures.post_json(1, 1))
            + UniverseQuestion.from_post_json(fixtures.post_json(2, 2))
            + UniverseQuestion.from_post_json(fixtures.post_json(3, 3))
        )
        report = production_coverage(universe, record, results=results, user_id=fixtures.OUR_USER_ID)
        forfeit = report.forfeited_points_estimate
        self.assertTrue(forfeit["available"])
        self.assertEqual(forfeit["tier"], "PROXY")
        self.assertEqual(forfeit["n_missed"], 2)
        self.assertAlmostEqual(forfeit["mean_spot_peer_on_covered_questions"], 20.0)
        self.assertAlmostEqual(forfeit["estimated_forfeited_points"], 40.0)
        self.assertTrue(report.caveats)

    def test_forfeit_estimate_unavailable_without_realised_scores(self):
        record, results = self._record(
            [fixtures.question_row(1)], [fixtures.forecast_row(1, 0.8)], []
        )
        report = production_coverage([], record, results=results, user_id=fixtures.OUR_USER_ID)
        self.assertFalse(report.forfeited_points_estimate["available"])
        self.assertIn("reason", report.forfeited_points_estimate)


class BenchmarkCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_reports_evaluable_and_discarded_with_reasons(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "bench"),
            [
                fixtures.question_row(1),
                fixtures.question_row(2, question_type="numeric", resolution="42"),
                fixtures.question_row(3, resolution=None),
                fixtures.question_row(4, resolution="annulled"),
            ],
            [
                fixtures.forecast_row(1, 0.8),
                fixtures.forecast_row(2, 0.3),
                fixtures.forecast_row(3, 0.5),
                fixtures.forecast_row(4, 0.5),
            ],
            [],
        )
        record = load_track_record(directory)
        results, _summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        report = benchmark_coverage(record, results)

        self.assertEqual(report.n_questions, 4)
        self.assertEqual(report.n_resolved, 2)
        self.assertEqual(report.n_evaluable, 2)
        self.assertEqual(report.n_discarded, 2)
        self.assertEqual(report.discard_reasons["unresolved_or_annulled"], 2)
        self.assertAlmostEqual(report.evaluable_rate, 0.5)
        self.assertEqual(report.by_type["binary"]["evaluable"], 1)
        self.assertEqual(report.by_type["numeric"]["evaluable"], 1)

    def test_every_competition_type_appears_even_when_absent(self):
        """A type that never shows up is the blind spot this report exists to
        surface, so it must be listed with a zero rather than omitted."""
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "onlybinary"),
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [],
        )
        record = load_track_record(directory)
        results, _summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        report = benchmark_coverage(record, results)
        for qtype in ("binary", "multiple_choice", "numeric", "discrete", "date"):
            self.assertIn(qtype, report.by_type)
        self.assertEqual(report.by_type["numeric"]["total"], 0)

    def test_question_without_probability_of_resolution_is_discarded_with_a_reason(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "nop"),
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, None)],
            [],
        )
        record = load_track_record(directory)
        results, _summary = score_track_record(record, user_id=fixtures.OUR_USER_ID)
        report = benchmark_coverage(record, results)
        self.assertEqual(report.n_evaluable, 0)
        self.assertEqual(report.discard_reasons["no_probability_of_resolution"], 1)


if __name__ == "__main__":
    unittest.main()
