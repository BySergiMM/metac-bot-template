"""End-to-end: a synthetic dataset, through the whole lab, twice.

Covers the two properties that make the lab trustworthy rather than merely
functional:

- **Reproducibility.** The same dataset analysed twice must produce byte-identical
  results. An analysis that drifts between runs cannot support a claim about
  whether a change helped.
- **Honest labelling.** The report must say UNAVAILABLE when the peer
  denominator is absent, and EXACT when it is present -- never quietly
  substitute one for the other.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from research.analyze_track_record import analyze, load_universe, render
from research.fetch_own_track_record import build_limitations, read_token, summarise
from research.provenance import (
    FileRecord,
    Manifest,
    content_digest,
    make_dataset_id,
    utc_now_iso,
    write_manifest,
)
from research.scorer import EXACT
from tests import fixtures

EXPECTED_BINARY_SPOT_PEER = 51.70039921703093


def build_dataset(directory: str, with_geometric_mean: bool, account_user_id: int = fixtures.OUR_USER_ID) -> str:
    """A dataset complete with a valid manifest, as the fetch script writes it."""
    question_rows = [
        fixtures.question_row(1),
        fixtures.question_row(2, question_type="numeric", resolution="42"),
        fixtures.question_row(3, question_type="multiple_choice", resolution="Option A"),
        fixtures.question_row(4, resolution=None),
    ]
    forecast_rows = [
        fixtures.forecast_row(1, 0.8, probability_yes=0.8),
        fixtures.forecast_row(2, 0.30),
        # question 3: our forecast closes before the spot instant -> forfeited
        fixtures.forecast_row(3, 0.60, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID),
        fixtures.forecast_row(4, 0.50),
    ]
    if with_geometric_mean:
        forecast_rows += [
            fixtures.geometric_mean_row(1, 0.5, 11),
            fixtures.geometric_mean_row(2, 0.5, 11),
            fixtures.geometric_mean_row(3, 0.5, 11),
        ]
    score_rows = [
        fixtures.score_row(1, EXPECTED_BINARY_SPOT_PEER, coverage=1.0),
        fixtures.score_row(3, 0.0, coverage=0.0),
        fixtures.score_row(1, 12.0, score_type="baseline"),
    ]
    universe = {
        "33022": [
            fixtures.post_json(1, 1, "binary"),
            fixtures.post_json(2, 2, "numeric"),
            fixtures.post_json(3, 3, "multiple_choice"),
            fixtures.post_json(5, 5, "discrete"),
        ]
    }
    fixtures.write_dataset(
        directory,
        question_rows,
        forecast_rows,
        score_rows,
        universe=universe,
        account={"user_id": account_user_id, "username": fixtures.OUR_USERNAME},
    )

    records = [
        FileRecord.from_path(os.path.join(directory, name))
        for name in sorted(os.listdir(directory))
    ]
    created = utc_now_iso()
    dataset_id = make_dataset_id("track-record", created, content_digest(records))
    write_manifest(
        directory,
        Manifest(
            dataset_id=dataset_id,
            kind="track-record",
            created_at=created,
            files=records,
            account={"user_id": account_user_id, "username": fixtures.OUR_USERNAME},
            git={"commit": "0" * 40, "branch": "test", "dirty": False, "available": True},
            limitations=["synthetic dataset used by the test suite"],
        ),
    )
    return directory


class UnavailablePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.directory = build_dataset(os.path.join(self.tmp, "ds"), with_geometric_mean=False)
        self.report = analyze(self.directory)

    def test_dataset_integrity_is_checked_as_part_of_analysis(self):
        self.assertEqual(self.report["integrity"], "OK")

    def test_scoring_counts(self):
        scoring = self.report["scoring"]
        self.assertEqual(scoring["n_questions_in_dataset"], 4)
        self.assertEqual(scoring["n_resolved"], 3)
        self.assertEqual(scoring["n_scored"], 3)
        self.assertEqual(scoring["n_covered_at_spot"], 2)
        self.assertEqual(scoring["n_missed_at_spot"], 1)

    def test_peer_score_is_reported_unavailable_not_approximated(self):
        self.assertEqual(self.report["scoring"]["spot_peer_tier"], "UNAVAILABLE")
        self.assertIsNone(self.report["scoring"]["total_spot_peer"])
        self.assertEqual(self.report["validation"]["spot_peer"]["tier"], "UNAVAILABLE")
        self.assertIn(
            "geometric_mean_of_other_forecasters",
            self.report["validation"]["spot_peer"]["missing_terms"],
        )

    def test_coverage_flags_reproduce_metaculus(self):
        coverage = self.report["validation"]["coverage"]
        self.assertEqual(coverage["n_compared"], 2)
        self.assertEqual(coverage["n_matching"], 2)
        self.assertIn("COVERAGE REPRODUCED, PEER SCORE UNAVAILABLE", self.report["validation"]["verdict"])

    def test_inversion_diagnostic_runs_and_passes(self):
        inversion = self.report["validation"]["inversion"]
        self.assertEqual(inversion["n_evaluated"], 1)
        self.assertEqual(inversion["n_invalid_probability"], 0)

    def test_production_coverage_sees_the_uncovered_question(self):
        block = self.report["production_coverage"]["33022"]
        self.assertEqual(block["n_universe_questions"], 4)
        self.assertEqual(block["n_forecasted"], 3)
        self.assertEqual(block["uncovered_by_type"], {"discrete": 1})
        self.assertEqual(block["n_scored_with_coverage_0"], 1)

    def test_benchmark_coverage_lists_every_type(self):
        by_type = self.report["benchmark_coverage"]["by_type"]
        for qtype in ("binary", "multiple_choice", "numeric", "discrete", "date"):
            self.assertIn(qtype, by_type)
        self.assertEqual(self.report["benchmark_coverage"]["n_evaluable"], 3)

    def test_render_produces_a_report_without_raising(self):
        text = render(self.report)
        self.assertIn("TRACK RECORD REPORT", text)
        self.assertIn("VERDICT:", text)
        self.assertIn("UNAVAILABLE", text)

    def test_analysis_is_reproducible(self):
        again = analyze(self.directory)
        self.assertEqual(
            json.dumps(self.report, sort_keys=True, default=str),
            json.dumps(again, sort_keys=True, default=str),
        )


class ExactPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.directory = build_dataset(os.path.join(self.tmp, "ds"), with_geometric_mean=True)
        self.report = analyze(self.directory)

    def test_exact_reproduction_when_the_denominator_is_present(self):
        self.assertEqual(self.report["scoring"]["spot_peer_tier"], EXACT)
        self.assertEqual(self.report["validation"]["spot_peer"]["tier"], EXACT)
        self.assertLess(self.report["validation"]["spot_peer"]["max_absolute_error"], 1e-9)
        self.assertIn("EXACT REPRODUCTION", self.report["validation"]["verdict"])

    def test_forfeited_question_scores_zero_and_matches_metaculus(self):
        by_id = {row["question_id"]: row for row in self.report["per_question"]}
        self.assertEqual(by_id[3]["spot_peer"], 0.0)
        self.assertEqual(by_id[3]["reproduced_coverage"], 0.0)
        self.assertEqual(self.report["validation"]["coverage"]["n_matching"], 2)

    def test_render_shows_the_totals(self):
        text = render(self.report)
        self.assertIn("total spot peer", text)
        self.assertIn("EXACT", text)


class UniverseLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_missing_universe_file_is_not_fatal(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "nouni"),
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            [],
        )
        self.assertEqual(load_universe(directory), {})


class FetchHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_summarise_counts_only_our_own_rows(self):
        questions = fixtures.questions_csv(
            [fixtures.question_row(1), fixtures.question_row(2, resolution=None)]
        )
        forecasts = fixtures.forecasts_csv(
            [
                fixtures.forecast_row(1, 0.8),
                fixtures.forecast_row(2, 0.4, forecaster_id=fixtures.OTHER_USER_ID),
                fixtures.geometric_mean_row(1, 0.5, 11),
            ]
        )
        scores = fixtures.scores_csv(
            [
                fixtures.score_row(1, 5.0),
                fixtures.score_row(1, 9.0, user_id=fixtures.OTHER_USER_ID, user_username="other"),
            ]
        )
        summary = summarise(questions, forecasts, scores, fixtures.OUR_USER_ID)
        self.assertEqual(summary["n_questions"], 2)
        self.assertEqual(summary["n_resolved_questions"], 1)
        self.assertEqual(summary["n_own_forecasts"], 1)
        self.assertEqual(summary["n_scored_forecasts"], 1)
        self.assertEqual(summary["n_aggregate_rows"], 1)
        self.assertEqual(summary["aggregation_methods_present"], ["geometric_mean"])
        self.assertEqual(summary["question_types"], {"binary": 2})
        self.assertIsNotNone(summary["date_range"]["first_forecast"])

    def test_summarise_handles_empty_inputs(self):
        summary = summarise("", "", "", None)
        self.assertEqual(summary["n_questions"], 0)
        self.assertIsNone(summary["date_range"]["first_forecast"])

    def test_limitations_name_the_missing_denominator(self):
        limitations = build_limitations(
            {"aggregation_methods_present": [], "n_scored_forecasts": 0}, []
        )
        joined = " ".join(limitations)
        self.assertIn("geometric_mean", joined)
        self.assertIn("No per-question scores", joined)
        self.assertIn("Terms of Use", joined)
        self.assertIn("spot_scoring_time", joined)

    def test_limitations_flag_a_partial_download(self):
        limitations = build_limitations(
            {"aggregation_methods_present": ["geometric_mean"], "n_scored_forecasts": 3},
            [{"status": "error_after_fallback"}],
        )
        self.assertTrue(any("incomplete" in note for note in limitations))

    def test_read_token_prefers_an_explicit_file(self):
        path = os.path.join(self.tmp, "token.txt")
        with open(path, "w") as handle:
            handle.write("  secret-token\n")

        class Args:
            token_file = path

        self.assertEqual(read_token(Args()), "secret-token")

    def test_read_token_rejects_an_empty_file(self):
        path = os.path.join(self.tmp, "empty.txt")
        open(path, "w").close()

        class Args:
            token_file = path

        with self.assertRaises(SystemExit):
            read_token(Args())


if __name__ == "__main__":
    unittest.main()
