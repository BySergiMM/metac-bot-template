"""Building a track record from the posts API payload.

The download endpoint is closed to this account, so this is the route that
actually produces data. The payload shapes below are the ones observed in the
live `--diagnose` run, not invented ones:

    question keys include: resolution, spot_scoring_time, question_weight,
                           type, scaling, options, cp_reveal_time, ...
    my_forecasts.history[0] keys: author_id, centers, distribution_input,
                           end_time, forecast_values, interval_lower_bounds,
                           interval_upper_bounds, question_id, start_time
"""

from __future__ import annotations

import unittest

from research.posts_track_record import (
    build_csvs,
    probability_of_resolution,
    resolution_bucket_index,
)
from research.track_record import read_forecast_csv, read_question_csv

USER_ID = 306913
USERNAME = "seergiii-bot"


def question(**overrides):
    base = {
        "id": 500,
        "title": "A question",
        "type": "binary",
        "resolution": "yes",
        "open_time": "2026-06-01T12:00:00Z",
        "cp_reveal_time": "2026-06-05T12:00:00Z",
        "scheduled_close_time": "2026-06-10T12:00:00Z",
        "actual_close_time": "2026-06-10T12:00:00Z",
        "spot_scoring_time": "2026-06-07T09:00:00Z",
        "question_weight": 1.0,
        "include_bots_in_aggregates": True,
        "scaling": {"range_min": None, "range_max": None, "zero_point": None},
    }
    base.update(overrides)
    return base


def forecast_record(values, start="2026-06-02T00:00:00Z", end=None):
    return {
        "author_id": USER_ID,
        "question_id": 500,
        "start_time": start,
        "end_time": end,
        "forecast_values": values,
        "centers": None,
        "distribution_input": None,
        "interval_lower_bounds": None,
        "interval_upper_bounds": None,
    }


def post(question_payload, history=None):
    if history is not None:
        question_payload = dict(question_payload)
        question_payload["my_forecasts"] = {"history": history, "latest": history[-1] if history else None, "score_data": {}}
    return {
        "id": 45179,
        "title": "A post",
        "curation_status": "approved",
        "published_at": "2026-06-01T00:00:00Z",
        "projects": {"default_project": {"id": 33022, "name": "FutureEval"}},
        "question": question_payload,
    }


class BucketIndexTests(unittest.TestCase):
    def test_binary_yes_and_no(self):
        """``forecast_values`` for a binary question is ``[P(no), P(yes)]``."""
        self.assertEqual(resolution_bucket_index(question(resolution="yes")), 1)
        self.assertEqual(resolution_bucket_index(question(resolution="no")), 0)

    def test_binary_is_case_insensitive(self):
        self.assertEqual(resolution_bucket_index(question(resolution="YES")), 1)

    def test_multiple_choice_uses_the_option_position(self):
        payload = question(
            type="multiple_choice", options=["Alpha", "Beta", "Gamma"], resolution="Beta"
        )
        self.assertEqual(resolution_bucket_index(payload), 1)

    def test_multiple_choice_with_an_unknown_option_is_none(self):
        payload = question(type="multiple_choice", options=["Alpha"], resolution="Omega")
        self.assertIsNone(resolution_bucket_index(payload))

    def test_continuous_types_are_declined_rather_than_guessed(self):
        """The bucket index depends on the question's scaling; a wrong guess
        would look like a real probability."""
        for qtype in ("numeric", "discrete", "date"):
            self.assertIsNone(resolution_bucket_index(question(type=qtype, resolution="42")))

    def test_unresolved_and_annulled_are_none(self):
        for value in (None, "", "annulled", "ambiguous"):
            self.assertIsNone(resolution_bucket_index(question(resolution=value)))


class ProbabilityTests(unittest.TestCase):
    def test_binary_probability_on_the_resolved_outcome(self):
        self.assertAlmostEqual(probability_of_resolution(question(), [0.3, 0.7]), 0.7)
        self.assertAlmostEqual(
            probability_of_resolution(question(resolution="no"), [0.3, 0.7]), 0.3
        )

    def test_multiple_choice_probability(self):
        payload = question(type="multiple_choice", options=["A", "B", "C"], resolution="C")
        self.assertAlmostEqual(probability_of_resolution(payload, [0.2, 0.3, 0.5]), 0.5)

    def test_missing_or_malformed_values_are_none(self):
        self.assertIsNone(probability_of_resolution(question(), None))
        self.assertIsNone(probability_of_resolution(question(), "not a list"))
        self.assertIsNone(probability_of_resolution(question(), [0.5]))  # index 1 absent
        self.assertIsNone(probability_of_resolution(question(), [None, None]))


class BuildCsvTests(unittest.TestCase):
    def test_round_trips_through_the_lab_readers(self):
        """The point of emitting the CSV schema is that every downstream module
        keeps working unchanged."""
        posts = [post(question(), history=[forecast_record([0.2, 0.8])])]
        question_text, forecast_text, stats = build_csvs(posts, USER_ID, USERNAME)

        questions = read_question_csv(question_text)
        forecasts = read_forecast_csv(forecast_text)

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].question_id, 500)
        self.assertEqual(questions[0].question_type, "binary")
        self.assertEqual(questions[0].resolution, "yes")
        self.assertEqual(questions[0].question_weight, 1.0)

        self.assertEqual(len(forecasts), 1)
        self.assertEqual(forecasts[0].forecaster_id, USER_ID)
        self.assertAlmostEqual(forecasts[0].probability_of_resolution, 0.8)
        self.assertAlmostEqual(forecasts[0].probability_yes, 0.8)
        self.assertEqual(stats["n_with_probability_of_resolution"], 1)

    def test_explicit_spot_scoring_time_is_carried_and_wins(self):
        """The posts API exposes the override the CSV export omits, so the
        derived instant is exact instead of inferred from cp_reveal_time."""
        posts = [post(question(), history=[forecast_record([0.2, 0.8])])]
        question_text, _forecast_text, stats = build_csvs(posts, USER_ID, USERNAME)
        row = read_question_csv(question_text)[0]
        self.assertIsNotNone(row.explicit_spot_scoring_time)
        self.assertEqual(row.spot_scoring_time(), row.explicit_spot_scoring_time)
        self.assertNotEqual(row.spot_scoring_time(), row.cp_reveal_time)
        self.assertEqual(stats["n_with_spot_scoring_time"], 1)

    def test_falls_back_when_no_override_is_present(self):
        posts = [post(question(spot_scoring_time=None), history=[forecast_record([0.2, 0.8])])]
        question_text, _f, _s = build_csvs(posts, USER_ID, USERNAME)
        row = read_question_csv(question_text)[0]
        self.assertIsNone(row.explicit_spot_scoring_time)
        self.assertEqual(row.spot_scoring_time(), row.cp_reveal_time)

    def test_multiple_forecast_records_become_multiple_intervals(self):
        posts = [
            post(
                question(),
                history=[
                    forecast_record([0.5, 0.5], start="2026-06-02T00:00:00Z", end="2026-06-04T00:00:00Z"),
                    forecast_record([0.2, 0.8], start="2026-06-04T00:00:00Z"),
                ],
            )
        ]
        _q, forecast_text, stats = build_csvs(posts, USER_ID, USERNAME)
        forecasts = read_forecast_csv(forecast_text)
        self.assertEqual(len(forecasts), 2)
        self.assertIsNotNone(forecasts[0].end_time)
        self.assertIsNone(forecasts[1].end_time)
        self.assertEqual(stats["n_forecast_records"], 2)

    def test_continuous_question_records_the_gap_instead_of_inventing_a_number(self):
        posts = [
            post(
                question(type="numeric", resolution="42"),
                history=[forecast_record([0.0, 0.1, 0.5, 0.9, 1.0])],
            )
        ]
        _q, forecast_text, stats = build_csvs(posts, USER_ID, USERNAME)
        forecasts = read_forecast_csv(forecast_text)
        self.assertIsNone(forecasts[0].probability_of_resolution)
        self.assertIsNotNone(forecasts[0].continuous_cdf)
        self.assertEqual(stats["unsupported_probability_types"], {"numeric": 1})
        self.assertEqual(stats["n_with_probability_of_resolution"], 0)

    def test_post_without_my_forecasts_yields_a_question_but_no_forecast(self):
        posts = [post(question())]
        question_text, forecast_text, stats = build_csvs(posts, USER_ID, USERNAME)
        self.assertEqual(len(read_question_csv(question_text)), 1)
        self.assertEqual(read_forecast_csv(forecast_text), [])
        self.assertEqual(stats["n_with_my_forecasts"], 0)

    def test_group_post_without_a_single_question_is_skipped_not_crashed(self):
        posts = [{"id": 1, "title": "group", "group_of_questions": {"questions": []}}]
        question_text, _f, stats = build_csvs(posts, USER_ID, USERNAME)
        self.assertEqual(read_question_csv(question_text), [])
        self.assertEqual(stats["n_questions"], 0)
        self.assertEqual(stats["n_posts"], 1)

    def test_stats_count_resolved_questions(self):
        posts = [
            post(question(id=1), history=[forecast_record([0.2, 0.8])]),
            post(question(id=2, resolution=None), history=[forecast_record([0.2, 0.8])]),
            post(question(id=3, resolution="annulled"), history=[forecast_record([0.2, 0.8])]),
        ]
        _q, _f, stats = build_csvs(posts, USER_ID, USERNAME)
        self.assertEqual(stats["n_questions"], 3)
        self.assertEqual(stats["n_resolved"], 1)

    def test_empty_input_produces_headers_only(self):
        question_text, forecast_text, stats = build_csvs([], USER_ID, USERNAME)
        self.assertEqual(read_question_csv(question_text), [])
        self.assertEqual(read_forecast_csv(forecast_text), [])
        self.assertEqual(stats["n_posts"], 0)


if __name__ == "__main__":
    unittest.main()
