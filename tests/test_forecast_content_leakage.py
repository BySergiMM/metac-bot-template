"""No forecast content may reach a public log, by any route.

This repository is a PUBLIC fork, so GitHub Actions logs are world readable.
Metaculus' AI Benchmark rules say: "Bots may not have a human in the loop when
forecasting. This includes running a bot on questions that are open or
upcoming in the tournament, seeing the bot's output, and responding to its
output by modifying the bot to improve its output." A public log of the bot's
rationale on an open tournament question puts the first two of those three
in front of the maker on every run.

The original filter (commit 2b70cdf) covered the research body, the prediction
value line and the report summary. It did NOT cover
``Reasoning for URL <url>: <rationale ending in "Probability: ZZ%">``, which
main.py emits once per prediction -- five times per question, for every
question type. Production runs 32239144510, 32268972325 and 32366841649
emitted it in the clear.

These tests are written against the INVARIANT, not against the patterns:
"push a realistic record through the real machinery and assert the content is
not in the output". A future leak through a route nobody anticipated fails
these tests without anyone having to predict the route.
"""

from __future__ import annotations

import contextlib
import io
import logging
import unittest

import bot_helpers
from bot_helpers import (
    RedactForecastContent,
    forecast_content_is_withheld,
    install_forecast_redaction,
    log_forecast_content,
    print_run_summary_banner,
    scrub,
)

URL = "https://www.metaculus.com/questions/45182"

RATIONALE = (
    "(a) The time left until the outcome is known is 22 days.\n"
    "(b) The status quo outcome is No.\n"
    "(c) A scenario resulting in No: negotiations stall again.\n"
    "(d) A scenario resulting in Yes: a framework is signed in Cairo.\n"
    "Weighing the status quo heavily, I settle on a low probability.\n"
    "Probability: 15%"
)

RESEARCH = (
    "Israeli officials have repeatedly stated conditions for a second phase.\n"
    "- Any agreement text published in the Federal Register.\n"
    "See https://example.com/news/12345 for the primary source."
)


class RedactionHarness(unittest.TestCase):
    """Drives the real filter through a real logging stack."""

    def setUp(self):
        self.stream = io.StringIO()
        self.logger = logging.getLogger("leakage_test_" + self.id())
        self.logger.handlers = []
        self.logger.propagate = False
        handler = logging.StreamHandler(self.stream)
        handler.addFilter(RedactForecastContent())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.addCleanup(setattr, self.logger, "handlers", [])

    @property
    def output(self) -> str:
        return self.stream.getvalue()

    def assertNotLeaked(self, *needles: str):
        for needle in needles:
            self.assertNotIn(needle, self.output,
                             f"forecast content leaked into the log: {needle!r}")


class ReasoningRecordTests(RedactionHarness):
    """The specific hole the previous audit found."""

    def test_reasoning_for_url_is_redacted(self):
        self.logger.info(f"Reasoning for URL {URL}: {RATIONALE}")
        self.assertNotLeaked("Probability: 15%", "status quo outcome is No",
                             "framework is signed in Cairo")
        self.assertIn(URL, self.output, "the question URL must survive")

    def test_multiline_reasoning_is_redacted_entirely(self):
        self.logger.info(f"Reasoning for URL {URL}: line one\nline two\nline three")
        self.assertNotLeaked("line one", "line two", "line three")

    def test_every_probability_value_is_redacted(self):
        for value in ("0%", "1%", "15%", "50%", "87%", "99%", "100%"):
            with self.subTest(value=value):
                self.setUp()
                self.logger.info(f"Reasoning for URL {URL}: text\nProbability: {value}")
                self.assertNotLeaked(f"Probability: {value}")

    def test_reasoning_containing_urls_is_still_redacted(self):
        self.logger.info(
            f"Reasoning for URL {URL}: I read https://example.com/leak and "
            "concluded.\nProbability: 33%"
        )
        self.assertNotLeaked("https://example.com/leak", "Probability: 33%")

    def test_a_bare_probability_anywhere_is_redacted(self):
        """No known headline, no known logger: the value guard alone."""
        self.logger.info("some future upstream record ... Probability: 71% ... tail")
        self.assertNotLeaked("Probability: 71%")


class ResearchAndSummaryTests(RedactionHarness):
    def test_the_research_body_is_redacted(self):
        self.logger.info(f"Found Research for URL {URL}:\n{RESEARCH}")
        self.assertNotLeaked("Israeli officials", "Federal Register",
                             "https://example.com/news/12345")
        self.assertIn(URL, self.output)

    def test_the_report_summary_block_is_redacted(self):
        self.logger.info(
            f"URL: {URL}\n"
            "<<<<<<<<<<<<<<<<<<<< Summary >>>>>>>>>>>>>>>>>>>>>\n"
            "The research summarises that no framework exists.\n"
            "<<<<<<<<<<<<<<<<<<<< First Rationale >>>>>>>>>>>>>>>>>>>>>\n"
            "Probability: 15%"
        )
        self.assertNotLeaked("no framework exists", "Probability: 15%")


class PredictionValueTests(RedactionHarness):
    def test_the_binary_prediction_line_is_redacted(self):
        self.logger.info(f"Forecasted URL {URL} with prediction: 0.15.")
        self.assertNotLeaked("0.15")

    def test_multiple_choice_repr_is_redacted(self):
        self.logger.info(
            f"Forecasted URL {URL} with prediction: predicted_options="
            "[PredictedOption(option_name='Guilty', probability=0.12)]"
        )
        self.assertNotLeaked("0.12", "PredictedOption")

    def test_numeric_repr_is_redacted(self):
        self.logger.info(
            f"Forecasted URL {URL} with prediction: declared_percentiles="
            "[Percentile(value=3.5, percentile=0.1)]"
        )
        self.assertNotLeaked("3.5", "Percentile(")

    def test_a_prediction_value_buried_in_surrounding_text_is_redacted(self):
        """Not at the start of the record, so no headline pattern matches."""
        self.logger.warning(
            "retrying question after parser disagreement; last parse was "
            "prediction_in_decimal=0.83 from the second sample"
        )
        self.assertNotLeaked("0.83", "prediction_in_decimal")

    def test_the_comment_bodys_own_forecast_display_is_redacted(self):
        self.logger.error(
            "Metaculus rejected the comment. Body was: # SUMMARY "
            "*Final Prediction*: 42% ..."
        )
        self.assertNotLeaked("42%", "*Final Prediction*")


class UpstreamPayloadTests(RedactionHarness):
    """forecasting_tools records that carry provider or model text."""

    def test_research_errors_keep_the_headline_and_drop_the_payload(self):
        self.logger.warning(
            "Encountered errors while researching: "
            "['litellm.APIError: {\"choices\":[{\"text\":\"Probability: 12%\"}]}']"
        )
        self.assertNotLeaked("Probability: 12%", "choices")
        self.assertIn("Encountered errors while researching:", self.output)

    def test_prediction_errors_keep_the_headline_and_drop_the_payload(self):
        self.logger.warning(
            "Encountered errors while predicting: ['ValidationError: got 0.91']"
        )
        self.assertNotLeaked("0.91")
        self.assertIn("Encountered errors while predicting:", self.output)

    def test_a_traceback_record_is_collapsed(self):
        self.logger.error(
            "Exception occurred during forecasting:\n"
            "Traceback (most recent call last):\n"
            '  File "main.py", line 285, in _binary_prompt_to_forecast\n'
            "ValueError: parser returned Probability: 64%"
        )
        self.assertNotLeaked("Probability: 64%", "Traceback", "_binary_prompt_to_forecast")

    def test_a_live_traceback_attached_to_a_record_is_dropped(self):
        """logger.exception() attaches exc_info, which the HANDLER renders
        after the filter runs. Replacing msg alone would not remove it."""
        try:
            raise ValueError("model said Probability: 55%")
        except ValueError:
            self.logger.exception(f"Reasoning for URL {URL}: boom")
        self.assertNotLeaked("Probability: 55%", "Traceback")

    def test_summarize_failure_keeps_its_headline(self):
        self.logger.warning("Could not summarize research. litellm returned junk")
        self.assertIn("Could not summarize research.", self.output)
        self.assertNotIn("litellm returned junk", self.output)


class OperationalMessagesSurviveTests(RedactionHarness):
    """Redaction must not blind the audits. Every line here is one an
    operator or an audit reads out of a production log."""

    SURVIVORS = [
        "Retrieved 3 questions from tournament 33022",
        "Retrieving questions from tournament minibench",
        "discovery_complete tournament=minibench pages=3 questions=250 "
        "duplicates_dropped=0 truncated=False",
        "discovery_page_limit_reached tournament=t pages=200 page_size=100 "
        "state=discovery_may_be_incomplete",
        "llm_attempt provider_index=1 provider=gemini/gemini-3.5-flash-lite "
        "bucket=gemini/gemini-3.5-flash-lite#b2 wait_ms=0",
        "llm_success provider_index=1 provider=gemini/gemini-3.5-flash-lite "
        "bucket=gemini/gemini-3.5-flash-lite#b1 latency_s=1.05 wait_ms=0 "
        "fallback_used=False",
        "llm_failure provider_index=0 provider=openrouter/x bucket=openrouter/x "
        "latency_s=0.03 reason='RateLimitError'",
        "Posting prediction on question 45375",
        "Posted prediction on question 45375",
        "Posted comment on post 45182",
        "publication_prediction_written question_id=45375 post_id=45182",
        "publication_comment_written post_id=45182 attempts=1",
        "PUBLICATION_ORPHAN post_id=45182 question_ids=45375 comment_attempts=3 "
        "state=prediction_published_without_comment",
        "comment_suppressed_duplicate post_id=45182 "
        "reason=already_commented_this_process",
        "Skipping 2 previously forecasted questions",
        "rate_limiter_created model=gemini/gemini-3.5-flash-lite rpm=15.0 tpm=None",
        "Total cost estimated: $0.00000",
    ]

    def test_operational_lines_pass_through_unchanged(self):
        for message in self.SURVIVORS:
            with self.subTest(message=message[:50]):
                self.setUp()
                self.logger.info(message)
                self.assertIn(message, self.output,
                              "redaction must not touch: " + message)


class SourceLevelSuppressionTests(unittest.TestCase):
    """Layer 1: on a scored run the content never reaches logging at all."""

    def setUp(self):
        self.saved = bot_helpers._withhold_forecast_content
        self.addCleanup(
            setattr, bot_helpers, "_withhold_forecast_content", self.saved
        )
        self.stream = io.StringIO()
        self.logger = logging.getLogger("layer1_" + self.id())
        self.logger.handlers = [logging.StreamHandler(self.stream)]
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.addCleanup(setattr, self.logger, "handlers", [])

    def test_content_is_withheld_on_a_scored_run(self):
        install_forecast_redaction("tournament")
        log_forecast_content(self.logger, f"Reasoning for URL {URL}", RATIONALE)
        out = self.stream.getvalue()
        self.assertNotIn("Probability: 15%", out)
        self.assertIn(URL, out)

    def test_content_is_emitted_on_the_unscored_practice_area(self):
        """bot-testing-area is unscored; full output is what makes it useful."""
        install_forecast_redaction("test_questions")
        log_forecast_content(self.logger, f"Reasoning for URL {URL}", RATIONALE)
        self.assertIn("Probability: 15%", self.stream.getvalue())

    def test_the_default_before_any_mode_is_declared_is_to_withhold(self):
        """Fail closed: forgetting to call install_forecast_redaction must not
        be the thing that publishes a rationale."""
        bot_helpers._withhold_forecast_content = True
        self.assertTrue(forecast_content_is_withheld())
        log_forecast_content(self.logger, "Reasoning for URL x", RATIONALE)
        self.assertNotIn("Probability: 15%", self.stream.getvalue())

    def test_the_module_default_is_withheld(self):
        import importlib

        fresh = importlib.reload(bot_helpers)
        try:
            self.assertTrue(fresh._withhold_forecast_content)
        finally:
            importlib.reload(bot_helpers)

    def test_the_cup_is_treated_as_scored(self):
        install_forecast_redaction("metaculus_cup")
        self.assertTrue(forecast_content_is_withheld())


class StdoutAndStderrTests(unittest.TestCase):
    """print() bypasses logging entirely, so it needs its own proof."""

    def test_the_summary_banner_scrubs_exception_text(self):
        class Boom(Exception):
            pass

        exc = Boom(
            "Expected at least 2.5 successful predictions, but only got 1. "
            "Errors encountered: ['parser said Probability: 77%']"
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_run_summary_banner([exc], will_publish=True)
        out = buffer.getvalue()
        self.assertNotIn("Probability: 77%", out)
        self.assertIn("Boom", out, "the exception TYPE stays useful")

    def test_the_banner_still_truncates_long_but_safe_errors(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_run_summary_banner([RuntimeError("x" * 500)], will_publish=True)
        self.assertIn("...", buffer.getvalue())
        self.assertLess(len(buffer.getvalue()), 600)

    def test_no_forecast_content_reaches_stderr(self):
        """stderr carries interpreter and dependency warnings that are not
        ours, so the assertion is about CONTENT, not emptiness."""
        exc = RuntimeError("parser said Probability: 77% and probability=0.3")
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            with contextlib.redirect_stdout(io.StringIO()):
                print_run_summary_banner([exc], will_publish=True)
        err = buffer.getvalue()
        self.assertNotIn("Probability: 77%", err)
        self.assertNotIn("probability=0.3", err)


class ScrubTests(unittest.TestCase):
    def test_scrub_drops_anything_carrying_a_forecast_value(self):
        for text in ("Probability: 42%", "probability=0.3", "*Final Prediction*: 9%",
                     "declared_percentiles=[1,2]", "PredictedOption(x)",
                     "prediction_in_decimal"):
            with self.subTest(text=text):
                self.assertIn("redacted", scrub(text))

    def test_scrub_keeps_ordinary_diagnostics(self):
        self.assertEqual(scrub("Timeout after 180s"), "Timeout after 180s")
        self.assertEqual(scrub("ConnectionError: reset by peer"),
                         "ConnectionError: reset by peer")

    def test_scrub_collapses_newlines(self):
        self.assertNotIn("\n", scrub("a\nb\rc"))

    def test_scrub_respects_the_limit(self):
        self.assertLessEqual(len(scrub("y" * 1000, limit=50)), 53)

    def test_scrub_does_not_trip_on_operational_numbers(self):
        for safe in ("rpm=15.0 tpm=None", "latency_s=1.05", "wait_ms=0",
                     "Posted prediction on question 45375", "pages=3 questions=250"):
            with self.subTest(safe=safe):
                self.assertEqual(scrub(safe), safe)


class MainPyHasNoRawContentLoggingTests(unittest.TestCase):
    """The source-level guarantee, asserted against the source.

    A regex over main.py is a blunt instrument, but it is the only thing that
    catches a future edit reintroducing a raw logger.info(f"...{reasoning}")
    without anyone running a forecast.
    """

    def _main_src(self) -> str:
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with io.open(os.path.join(root, "main.py"), encoding="utf-8") as handle:
            return handle.read()

    def test_no_logger_call_interpolates_reasoning_or_research(self):
        import re

        src = self._main_src()
        offenders = re.findall(
            r"logger\.\w+\([^)]*\{(?:reasoning|research|decimal_pred|"
            r"predicted_option_list|prediction)\b[^)]*\)",
            src,
        )
        self.assertEqual(offenders, [], f"raw content logging returned: {offenders}")

    def test_every_content_site_goes_through_the_helper(self):
        src = self._main_src()
        self.assertEqual(src.count("log_forecast_content("), 9,
                         "9 call sites: research, 4 rationales, 4 prediction "
                         "values. The import has no paren and is not counted.")

    def test_redaction_is_installed_before_the_first_forecast(self):
        src = self._main_src()
        init_at = src.index("template_bot = SummerTemplateBot2026(")
        self.assertLess(src.index("install_forecast_redaction(run_mode)"),
                        src.index("forecast_on_tournament(", init_at))


if __name__ == "__main__":
    unittest.main()
