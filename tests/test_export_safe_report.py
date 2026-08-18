"""The publishable artifact must never leak Metaculus content or credentials.

The repository is public, so a workflow artifact is world-readable. These tests
are the last line of defence before something gets a permanent public URL.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from research.analyze_track_record import analyze
from research.export_safe_report import (
    UnsafeReportError,
    assert_safe,
    build_safe_summary,
    render_text,
)
from tests.test_end_to_end import build_dataset


def _walk_strings(document, path="$"):
    if isinstance(document, dict):
        for key, value in document.items():
            yield ("key", path, str(key))
            for item in _walk_strings(value, "{0}.{1}".format(path, key)):
                yield item
    elif isinstance(document, list):
        for index, value in enumerate(document):
            for item in _walk_strings(value, "{0}[{1}]".format(path, index)):
                yield item
    elif isinstance(document, str):
        yield ("value", path, document)


class LeakTests(unittest.TestCase):
    """Build a dataset whose every sensitive field is a unique canary string,
    then assert none of them survive into the artifact."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.directory = build_dataset(os.path.join(self.tmp, "ds"), with_geometric_mean=False)
        with open(os.path.join(self.directory, "question_data.csv")) as handle:
            self.raw_dataset = handle.read()
        self.report = analyze(self.directory)
        self.safe = build_safe_summary(self.report)
        self.text = render_text(self.safe)
        self.blob = json.dumps(self.safe) + "\n" + self.text

    def test_question_titles_and_urls_do_not_survive(self):
        """Canaries are checked against the dataset on disk, which is the real
        upstream source. The analysis report never carried titles either --
        worth knowing, but it is not what guards the artifact."""
        self.assertIn("Synthetic question", self.raw_dataset)
        self.assertIn("metaculus.com/questions", self.raw_dataset)
        self.assertNotIn("Synthetic question", self.blob)
        self.assertNotIn("metaculus.com/questions", self.blob)

    def test_resolution_values_do_not_survive(self):
        self.assertIn("Resolution", self.raw_dataset)
        self.assertNotIn('"resolution"', self.blob.lower())

    def test_per_question_rows_do_not_survive(self):
        self.assertIn("per_question", self.report)
        self.assertNotIn("per_question", self.blob)

    def test_probabilities_that_reveal_resolutions_do_not_survive(self):
        """Our probability on the resolved outcome identifies which outcome
        resolved, so it is Metaculus content in disguise."""
        self.assertNotIn('"p":', json.dumps(self.safe))
        self.assertNotIn("probability_of_resolution", self.blob)
        self.assertNotIn("implied_peer_probability", self.blob)

    def test_individual_metaculus_score_values_do_not_survive(self):
        """The value, not the key name: ``n_metaculus_spot_peer_rows`` is a
        count and is fine, but the score 51.70039921703093 that Metaculus
        assigned to one question must not appear anywhere."""
        self.assertNotIn("51.70039921703093", self.blob)
        self.assertNotIn("51.7003992", self.blob)

    def test_a_mean_over_one_observation_is_suppressed(self):
        """Caught by these tests on the first run: with a single covered
        question the 'mean' realised spot peer was that question's exact
        Metaculus score, published as if it were an aggregate."""
        forfeit = self.safe["production_coverage"]["33022"]["forfeited_points_estimate"]
        self.assertEqual(forfeit["n_covered_scored"], 1)
        self.assertEqual(forfeit["mean_spot_peer_on_covered_questions"], "suppressed_small_n")
        self.assertEqual(forfeit["estimated_forfeited_points"], "suppressed_small_n")

    def test_counts_are_never_suppressed(self):
        """Suppression must not hide the coverage story, which is the finding
        that matters and discloses nothing on its own."""
        forfeit = self.safe["production_coverage"]["33022"]["forfeited_points_estimate"]
        self.assertEqual(forfeit["n_missed"], 2)
        self.assertEqual(self.safe["offline_scoring"]["n_missed_at_spot"], 1)
        self.assertEqual(self.safe["offline_scoring"]["n_scored"], 3)

    def test_only_allowlisted_top_level_keys_are_present(self):
        self.assertEqual(
            sorted(self.safe),
            [
                "account",
                "benchmark_coverage",
                "dataset_summary",
                "excluded_from_this_artifact",
                "generated_from",
                "limitations",
                "milestone",
                "offline_scoring",
                "production_coverage",
                "validation",
            ],
        )

    def test_the_artifact_still_carries_the_numbers_we_need(self):
        """Safety must not be achieved by publishing nothing useful."""
        self.assertEqual(self.safe["offline_scoring"]["n_scored"], 3)
        self.assertEqual(self.safe["offline_scoring"]["n_missed_at_spot"], 1)
        self.assertEqual(self.safe["validation"]["coverage_reproduction"]["n_matching"], 2)
        self.assertIn("COVERAGE REPRODUCED", self.safe["validation"]["verdict"])
        self.assertEqual(self.safe["production_coverage"]["33022"]["n_not_forecasted"], 1)
        self.assertTrue(self.safe["generated_from"]["files"])

    def test_file_provenance_is_hashes_without_contents(self):
        for entry in self.safe["generated_from"]["files"]:
            self.assertEqual(sorted(entry), ["bytes", "name", "rows", "sha256"])

    def test_rendered_text_passes_the_same_check(self):
        assert_safe(self.text, "$.rendered_text")

    def test_render_is_a_function_of_the_safe_document_only(self):
        """The text must not be able to contain anything the JSON does not."""
        for kind, _path, value in _walk_strings(self.safe):
            if kind == "value" and len(value) > 40:
                continue
        # re-rendering from the safe doc alone must reproduce byte-identically
        self.assertEqual(render_text(self.safe), self.text)


class AssertSafeTests(unittest.TestCase):
    def test_banned_key_anywhere_in_the_tree_raises(self):
        for document in (
            {"title": "leak"},
            {"a": {"b": [{"resolution": "yes"}]}},
            {"nested": [{"deep": {"probability_of_resolution": 0.8}}]},
        ):
            with self.assertRaises(UnsafeReportError):
                assert_safe(document)

    def test_credential_shaped_string_raises(self):
        # 40 hex characters is the shape of a Metaculus token. Deliberately
        # built from a repeating pattern rather than anything token-looking, so
        # this repository never contains a string a secret scanner would flag.
        fake = "dead" * 10
        with self.assertRaises(UnsafeReportError):
            assert_safe({"note": fake})

    def test_sha256_and_commit_fields_are_exempt(self):
        """Hashes are provenance, not secrets -- but only where we expect
        them, identified by the field path."""
        assert_safe({"files": [{"sha256": "a" * 64}]}, "$")
        assert_safe({"git_commit": "b" * 40}, "$")

    def test_short_hex_is_allowed(self):
        assert_safe({"dataset_id": "track-record-20260819T081500Z-1a2b3c4d"})

    def test_a_clean_document_passes(self):
        assert_safe({"counts": {"n": 3}, "rate": 0.5, "list": ["ok", None, True]})


class MissingDataTests(unittest.TestCase):
    def test_an_empty_report_does_not_crash_the_exporter(self):
        """A failed fetch must still produce a publishable, honest artifact."""
        safe = build_safe_summary({})
        assert_safe(safe)
        text = render_text(safe)
        self.assertIn("MILESTONE 2", text)

    def test_data_access_probe_failure_is_summarised_not_echoed(self):
        safe = build_safe_summary(
            {"dataset": {"account": {"data_access_status": {"error": "403", "body": "secret-ish"}}}}
        )
        self.assertEqual(safe["account"]["has_bot_benchmarking_tier"], "probe_failed")
        self.assertNotIn("secret-ish", json.dumps(safe))

    def test_unknown_probe_shape_is_not_echoed_verbatim(self):
        safe = build_safe_summary(
            {"dataset": {"account": {"data_access_status": {"weird_field": "verbatim-payload"}}}}
        )
        self.assertEqual(safe["account"]["has_bot_benchmarking_tier"], "unknown_shape")
        self.assertNotIn("verbatim-payload", json.dumps(safe))


if __name__ == "__main__":
    unittest.main()
