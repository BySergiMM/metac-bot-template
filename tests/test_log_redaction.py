"""Forecast content must not reach a world-readable log.

This repository is a PUBLIC fork, so GitHub Actions logs are readable by
anyone. Production run 32268972325 emitted, in the clear, for a question that
was still OPEN in tournament 33022:

  * the full research body            ("Found Research for URL ...:\\n<body>")
  * the report summary and rationale  (log_report_summary)
  * all five individual predictions   ("... with prediction: 0.15")

Two rules bear on that, and the filter under test addresses both:

  private comments  Metaculus requires bots to comment privately, with
                    publication on Metaculus' own schedule. Publishing the same
                    reasoning immediately elsewhere defeats the purpose.
  no preview        FutureEval forbids previewing the bot's forecasts on open
                    questions and then updating the bot. Not printing the
                    predictions removes the preview entirely.

What must SURVIVE redaction matters as much as what must not: every audit of
this bot reads discovery counts, provider/bucket lines and publication
confirmations out of these logs.
"""

from __future__ import annotations

import logging
import unittest

from bot_helpers import RedactForecastContent, install_forecast_redaction


def apply(message: str) -> str:
    """Push one message through the filter and return what would be emitted."""
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )
    RedactForecastContent().filter(record)
    return record.getMessage()


class RedactsForecastContentTests(unittest.TestCase):
    def test_the_research_body_is_removed_but_the_url_survives(self):
        out = apply(
            "Found Research for URL https://www.metaculus.com/questions/45182:\n"
            "Israeli officials have repeatedly stated conditions for a second phase..."
        )
        self.assertIn("https://www.metaculus.com/questions/45182", out)
        self.assertNotIn("Israeli officials", out)
        self.assertIn("redacted", out)

    def test_the_prediction_value_is_removed(self):
        out = apply(
            "Forecasted URL https://www.metaculus.com/questions/45182 "
            "with prediction: 0.15."
        )
        self.assertNotIn("0.15", out)
        self.assertIn("https://www.metaculus.com/questions/45182", out)

    def test_multiple_choice_and_numeric_predictions_are_also_removed(self):
        for value in ("[0.1, 0.2, 0.7]", "declared_percentiles=[...]"):
            out = apply(
                "Forecasted URL https://www.metaculus.com/questions/1 "
                "with prediction: " + value
            )
            self.assertNotIn(value, out)

    def test_the_summary_and_rationale_block_is_removed_entirely(self):
        out = apply(
            "URL: https://www.metaculus.com/questions/45182\n"
            "<<<<<<<<<<<<<<<<<<<< Summary >>>>>>>>>>>>>>>>>>>>>\n"
            "The research summarizes that no publicly recognized framework...\n"
            "<<<<<<<<<<<<<<<<<<<< First Rationale >>>>>>>>>>>>>>>>>>>>>\n"
            "Probability: 15%"
        )
        self.assertNotIn("Probability: 15%", out)
        self.assertNotIn("no publicly recognized", out)
        self.assertIn("redacted", out)


class KeepsOperationalContentTests(unittest.TestCase):
    """Redaction must not blind the audits."""

    SURVIVORS = [
        "Retrieved 3 questions from tournament 33022",
        "Retrieving questions from tournament minibench",
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
        "Skipping 2 previously forecasted questions",
        "rate_limiter_created model=gemini/gemini-3.5-flash-lite rpm=15.0 tpm=None",
    ]

    def test_operational_lines_pass_through_unchanged(self):
        for message in self.SURVIVORS:
            self.assertEqual(apply(message), message,
                             "redaction must not touch: " + message)

    def test_the_bucket_stays_visible(self):
        """Without bucket= the four-bucket distribution is unmeasurable."""
        message = ("llm_attempt provider_index=1 provider=gemini/gemini-3.5-flash-lite "
                   "bucket=gemini/gemini-3.5-flash-lite#b3 wait_ms=0")
        self.assertIn("#b3", apply(message))


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self._root = logging.getLogger()
        self._saved = list(self._root.filters)

    def tearDown(self):
        self._root.filters = self._saved

    def test_it_is_installed_for_the_tournament(self):
        self.assertTrue(install_forecast_redaction("tournament"))

    def test_it_is_installed_for_the_metaculus_cup(self):
        self.assertTrue(install_forecast_redaction("metaculus_cup"))

    def test_it_is_NOT_installed_for_the_unscored_practice_area(self):
        """bot-testing-area is unscored; full output is what makes it useful."""
        self.assertFalse(install_forecast_redaction("test_questions"))

    def test_a_malformed_record_does_not_kill_the_run(self):
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="%d %d", args=(1,), exc_info=None,  # too few args -> getMessage raises
        )
        self.assertTrue(RedactForecastContent().filter(record))


class WiredIntoMainTests(unittest.TestCase):
    """A filter nobody calls protects nothing."""

    def _main_src(self):
        import io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return io.open(os.path.join(root, "main.py"), encoding="utf-8").read()

    def test_main_installs_the_filter(self):
        src = self._main_src()
        self.assertIn("install_forecast_redaction(run_mode)", src)

    def test_it_is_installed_before_any_forecasting_starts(self):
        src = self._main_src()
        # main.py's module docstring mentions forecast_on_tournament() before
        # the real call, so the search is anchored to the bot instantiation --
        # the same docstring trap pin_models._llms_block_span documents.
        init_at = src.index("template_bot = SummerTemplateBot2026(")
        self.assertLess(
            src.index("install_forecast_redaction(run_mode)"),
            src.index("forecast_on_tournament(", init_at),
            "redaction must be active before the first forecast is produced",
        )

    def test_it_is_imported_from_bot_helpers(self):
        self.assertIn("install_forecast_redaction", self._main_src()
                      [:self._main_src().index("silence_noisy_dependencies()")])


if __name__ == "__main__":
    unittest.main()
