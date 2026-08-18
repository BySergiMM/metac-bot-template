"""Parsing of the Metaculus CSV export: value coercion, schema guards, timing."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone

from research.track_record import (
    SchemaError,
    extract_zip,
    load_track_record,
    merge_csv_texts,
    normalise_resolution,
    parse_bool,
    parse_dt,
    parse_float,
    parse_int,
    parse_list,
    read_forecast_csv,
    read_question_csv,
    read_score_csv,
)
from tests import fixtures


class ValueParsingTests(unittest.TestCase):
    def test_parses_django_str_datetime(self):
        parsed = parse_dt("2026-06-05 12:00:00+00:00")
        self.assertEqual(parsed, datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc))

    def test_parses_microseconds(self):
        parsed = parse_dt("2026-08-18 11:04:49.472000+00:00")
        self.assertEqual(parsed.microsecond, 472000)

    def test_parses_iso_zulu(self):
        self.assertEqual(
            parse_dt("2026-06-05T12:00:00Z"),
            datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
        )

    def test_naive_datetimes_are_assumed_utc(self):
        """Guessing local time here would silently move every spot instant by
        the runner's timezone offset."""
        parsed = parse_dt("2026-06-05 12:00:00")
        self.assertEqual(parsed.tzinfo, timezone.utc)
        self.assertEqual(parsed.hour, 12)

    def test_blank_and_garbage_are_none(self):
        for value in (None, "", "  ", "None", "null", "nan", "not a date"):
            self.assertIsNone(parse_dt(value), value)

    def test_numeric_parsing(self):
        self.assertEqual(parse_float("0.25"), 0.25)
        self.assertIsNone(parse_float(""))
        self.assertIsNone(parse_float("None"))
        self.assertIsNone(parse_float("abc"))
        self.assertEqual(parse_int("7"), 7)
        self.assertEqual(parse_int("7.0"), 7)
        self.assertIsNone(parse_int(""))

    def test_bool_parsing(self):
        self.assertTrue(parse_bool("True"))
        self.assertFalse(parse_bool("False"))
        self.assertTrue(parse_bool("1"))
        self.assertIsNone(parse_bool(""))
        self.assertIsNone(parse_bool("maybe"))

    def test_list_parsing(self):
        self.assertEqual(parse_list("[0.1, 0.9]"), [0.1, 0.9])
        self.assertEqual(parse_list("['a', 'b']"), ["a", "b"])
        self.assertIsNone(parse_list(""))
        self.assertIsNone(parse_list("None"))
        self.assertIsNone(parse_list("[1, "))

    def test_list_parsing_cannot_execute_code(self):
        # literal_eval only accepts literals; a call expression must not run.
        self.assertIsNone(parse_list("__import__('os').system('true')"))

    def test_annulled_and_ambiguous_are_not_resolutions(self):
        for value in ("annulled", "AMBIGUOUS", "", None, "none"):
            self.assertIsNone(normalise_resolution(value), value)
        self.assertEqual(normalise_resolution("yes"), "yes")
        self.assertEqual(normalise_resolution(" 42.5 "), "42.5")


class SchemaGuardTests(unittest.TestCase):
    def test_missing_column_raises_with_the_name(self):
        with self.assertRaises(SchemaError) as ctx:
            read_question_csv("Question ID,Question Type\n1,binary\n")
        self.assertIn("Resolution", str(ctx.exception))

    def test_anonymised_forecasts_are_rejected(self):
        text = (
            "Question ID,Forecaster (Anonymized),Is Bot,Start Time,End Time,"
            "Forecaster Count,Probability Yes,Probability Yes Per Category,"
            "Continuous CDF,Probability Below Lower Bound,Probability Above Upper Bound,"
            "5th Percentile,25th Percentile,Median,75th Percentile,95th Percentile,"
            "Probability of Resolution,PDF at Resolution\n"
            "1,deadbeef,True,2026-06-01 12:00:00+00:00,,,0.5,,,,,,,,,,0.5,\n"
        )
        with self.assertRaises(SchemaError) as ctx:
            read_forecast_csv(text)
        self.assertIn("anonymised", str(ctx.exception))

    def test_rows_without_question_id_are_skipped(self):
        rows = read_score_csv(
            "Question ID,User ID,User Username,Score Type,Score,Coverage\n"
            ",1,bot,spot_peer,1.0,1.0\n"
            "5,1,bot,spot_peer,2.0,1.0\n"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].question_id, 5)


class QuestionTimingTests(unittest.TestCase):
    def _question(self, **kwargs):
        return read_question_csv(fixtures.questions_csv([fixtures.question_row(1, **kwargs)]))[0]

    def test_spot_time_prefers_cp_reveal_when_after_open(self):
        question = self._question()
        self.assertEqual(question.spot_scoring_time(), parse_dt(fixtures.T_REVEAL))

    def test_spot_time_falls_back_to_actual_close_when_reveal_precedes_open(self):
        question = self._question(cp_reveal_time=fixtures.T_OPEN)
        self.assertEqual(question.spot_scoring_time(), parse_dt(fixtures.T_CLOSE))

    def test_spot_time_falls_back_to_scheduled_close(self):
        question = self._question(cp_reveal_time=None, actual_close_time=None)
        self.assertEqual(question.spot_scoring_time(), parse_dt(fixtures.T_CLOSE))

    def test_spot_time_is_none_without_any_timestamps(self):
        question = self._question(cp_reveal_time=None, actual_close_time=None, scheduled_close_time=None)
        self.assertIsNone(question.spot_scoring_time())
        self.assertIsNone(question.spot_timestamp())

    def test_spot_timestamp_is_clamped_to_actual_close(self):
        """``min(spot_scoring_time, actual_close_time)`` -- a question that
        resolved early is scored at its real close, not at the later reveal."""
        early_close = "2026-06-04 00:00:00+00:00"
        question = self._question(actual_close_time=early_close)
        self.assertEqual(question.spot_timestamp(), parse_dt(early_close).timestamp())

    def test_continuous_flag(self):
        for qtype, expected in (
            ("binary", False),
            ("multiple_choice", False),
            ("numeric", True),
            ("discrete", True),
            ("date", True),
        ):
            self.assertEqual(self._question(question_type=qtype).is_continuous, expected, qtype)

    def test_annulled_question_is_not_resolved(self):
        self.assertFalse(self._question(resolution="annulled").is_resolved)


class ForecastRowTests(unittest.TestCase):
    def _rows(self, *rows):
        return read_forecast_csv(fixtures.forecasts_csv(list(rows)))

    def test_interval_is_start_inclusive_end_exclusive(self):
        row = self._rows(
            fixtures.forecast_row(1, 0.6, start_time=fixtures.T_OPEN, end_time=fixtures.T_REVEAL)
        )[0]
        start = parse_dt(fixtures.T_OPEN).timestamp()
        end = parse_dt(fixtures.T_REVEAL).timestamp()
        self.assertTrue(row.is_active_at(start))
        self.assertTrue(row.is_active_at(end - 1))
        self.assertFalse(row.is_active_at(end))
        self.assertFalse(row.is_active_at(start - 1))

    def test_open_ended_forecast_stays_active(self):
        row = self._rows(fixtures.forecast_row(1, 0.6, end_time=None))[0]
        self.assertTrue(row.is_active_at(parse_dt(fixtures.T_CLOSE).timestamp()))

    def test_row_without_start_time_is_never_active(self):
        row = self._rows(fixtures.forecast_row(1, 0.6, start_time=None))[0]
        self.assertFalse(row.is_active_at(0.0))

    def test_aggregate_rows_are_identified_by_null_forecaster_id(self):
        rows = self._rows(
            fixtures.forecast_row(1, 0.6),
            fixtures.geometric_mean_row(1, 0.5, 11),
        )
        ours, aggregate = rows
        self.assertFalse(ours.is_aggregate)
        self.assertIsNone(ours.aggregation_method)
        self.assertTrue(aggregate.is_aggregate)
        self.assertEqual(aggregate.aggregation_method, "geometric_mean")


class TrackRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_load_and_select(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "ds"),
            [fixtures.question_row(1), fixtures.question_row(2, resolution=None)],
            [
                fixtures.forecast_row(1, 0.8),
                fixtures.geometric_mean_row(1, 0.5, 11),
                fixtures.forecast_row(2, None, forecaster_id=fixtures.OTHER_USER_ID),
            ],
            [fixtures.score_row(1, 51.7)],
        )
        record = load_track_record(directory)
        self.assertEqual(len(record.questions), 2)
        self.assertEqual(len(record.own_forecasts()), 2)
        self.assertEqual(len(record.own_forecasts(user_id=fixtures.OUR_USER_ID)), 1)
        self.assertEqual(len(record.aggregate_forecasts("geometric_mean")), 1)
        self.assertEqual(len(record.own_scores(score_type="spot_peer")), 1)
        self.assertEqual(record.infer_user_id(), fixtures.OUR_USER_ID)

    def test_score_csv_is_optional(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "noscores"),
            [fixtures.question_row(1)],
            [fixtures.forecast_row(1, 0.8)],
            score_rows=None,
        )
        record = load_track_record(directory)
        self.assertEqual(record.scores, [])

    def test_missing_question_csv_raises(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(FileNotFoundError):
            load_track_record(empty)

    def test_forecasts_by_question_are_time_sorted(self):
        directory = fixtures.write_dataset(
            os.path.join(self.tmp, "sorted"),
            [fixtures.question_row(1)],
            [
                fixtures.forecast_row(1, 0.7, start_time=fixtures.T_MID),
                fixtures.forecast_row(1, 0.6, start_time=fixtures.T_OPEN, end_time=fixtures.T_MID),
            ],
            [],
        )
        record = load_track_record(directory)
        rows = record.forecasts_by_question()[1]
        self.assertEqual([r.probability_of_resolution for r in rows], [0.6, 0.7])


class MergeAndZipTests(unittest.TestCase):
    def test_merge_keeps_one_header(self):
        merged = merge_csv_texts(["a,b\n1,2\n", "a,b\n3,4\n"])
        self.assertEqual(merged.splitlines(), ["a,b", "1,2", "3,4"])

    def test_merge_ignores_empty_chunks(self):
        self.assertEqual(merge_csv_texts(["", "a,b\n1,2\n", "   "]).splitlines(), ["a,b", "1,2"])

    def test_merge_of_nothing_is_empty(self):
        self.assertEqual(merge_csv_texts([]), "")

    def test_merge_refuses_mismatched_headers(self):
        with self.assertRaises(SchemaError):
            merge_csv_texts(["a,b\n1,2\n", "a,c\n3,4\n"])

    def test_extract_zip_writes_members(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("question_data.csv", "Question ID\n1\n")
        with tempfile.TemporaryDirectory() as tmp:
            written = extract_zip(buffer.getvalue(), tmp)
            self.assertEqual(len(written), 1)
            self.assertTrue(os.path.exists(os.path.join(tmp, "question_data.csv")))

    def test_extract_zip_refuses_path_traversal(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../escape.csv", "nope")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_zip(buffer.getvalue(), tmp)


if __name__ == "__main__":
    unittest.main()
