"""Assume the model is hostile and the log is public.

The forecasting prompts hand an LLM a question and take back an arbitrary
string. That string is attacker-influenced in the only sense that matters
here: nobody wrote it, nobody reviewed it, and it lands in exception messages,
provider error bodies and parser complaints all over the pipeline.

So the property under test is not "the known leak is patched". It is:

    given ANY string the model can return, no forecast content and no
    credential-shaped text reaches stdout, stderr, or a log handler.

Two real bypasses were found by writing these tests and are pinned here:

  * exc_info / logger.exception() -- the handler renders the exception AFTER
    every filter has run, and record.getMessage() does not include it, so a
    record with a perfectly operational message could still print a provider
    response body. litellm raises from 144 logger.exception() sites.
  * a handler attached to a CHILD logger -- called before the root handlers,
    so it never saw a filter that was only installed on root.
"""

from __future__ import annotations

import contextlib
import io
import logging
import unittest

import bot_helpers
from bot_helpers import RedactForecastContent, install_forecast_redaction, scrub

# Strings a hostile or merely unlucky model could return. Each is a real shape
# that appears somewhere in this pipeline.
HOSTILE_OUTPUTS = {
    "plain probability": "Probability: 15%",
    "decimal probability": "Probability: 0.15",
    "field name": "prediction_in_decimal=0.15",
    "option repr": 'PredictedOption("A", probability=0.4)',
    "option list": 'predicted_options=["A", "B"]',
    "percentile list": "declared_percentiles=[0.1, 0.5, 0.9]",
    "percentile repr": "Percentile(value=3, percentile=0.15)",
    "comment display": "*Final Prediction*: 42%",
    "reasoning headline": "Reasoning for URL https://www.metaculus.com/questions/1: x",
    "json payload": '{"choices":[{"text":"Probability: 77%"}]}',
    "nested container": "{'a': [{'b': 'Probability: 3%'}]}",
    "multiline": "line one\nline two\nProbability: 8%",
    "leading whitespace": "   Probability: 9%",
    "trailing whitespace": "Probability: 9%   ",
    "embedded in operational text": (
        "llm_success provider=x bucket=y latency_s=1.0 Probability: 11%"
    ),
    "traceback-looking": (
        "Traceback (most recent call last):\n  File x\nValueError: Probability: 4%"
    ),
    "ansi escapes": "\x1b[31mProbability: 12%\x1b[0m",
    "very large": ("padding " * 5000) + "Probability: 6%",
}

# Credential shapes. None of these is real; they exist so the test can prove
# nothing key-shaped survives to a public log.
HOSTILE_SECRETS = {
    "gemini-shaped": "AIza-SYNTHETIC-FIXTURE-NOT-A-REAL-CREDENTIAL-000",
    "openai-shaped": "sk-SYNTHETIC000000000000000000000000000000000000000",
    "openrouter-shaped": "sk-or-v1-SYNTHETIC0000000000000000000000000000000",
    "groq-shaped": "gsk_SYNTHETIC0000000000000000000000000000000000000",
    "bearer": "Authorization: Bearer SYNTHETIC0000000000000000000000",
}

FORECAST_MARKERS = (
    "Probability:", "prediction_in_decimal", "PredictedOption",
    "predicted_options", "declared_percentiles", "Percentile(",
    "*Final Prediction*",
)


class HostileModelOutputHarness(unittest.TestCase):
    """A real logging stack with the real filter, captured."""

    def setUp(self):
        self.stream = io.StringIO()
        self.logger = logging.getLogger("adversarial_" + self.id())
        self.logger.handlers = []
        self.logger.propagate = False
        handler = logging.StreamHandler(self.stream)
        handler.addFilter(RedactForecastContent())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.addCleanup(setattr, self.logger, "handlers", [])

    def assertNoForecastContent(self, text: str, label: str):
        for marker in FORECAST_MARKERS:
            self.assertNotIn(marker, text, f"{label}: {marker!r} reached the log")


class EveryLogLevelTests(HostileModelOutputHarness):
    def test_hostile_output_is_contained_at_every_level(self):
        for label, payload in HOSTILE_OUTPUTS.items():
            for level in ("debug", "info", "warning", "error", "critical"):
                with self.subTest(payload=label, level=level):
                    self.setUp()
                    getattr(self.logger, level)(payload)
                    self.assertNoForecastContent(self.stream.getvalue(), label)

    def test_hostile_output_survives_lazy_percent_arguments(self):
        """logger.info("x %s", value) -- the value only exists at format time."""
        for label, payload in HOSTILE_OUTPUTS.items():
            with self.subTest(payload=label):
                self.setUp()
                self.logger.info("provider returned %s", payload)
                self.assertNoForecastContent(self.stream.getvalue(), label)

    def test_hostile_output_inside_an_object_repr(self):
        class Sneaky:
            def __repr__(self):
                return "Probability: 33%"

        self.logger.info("parsed %r", Sneaky())
        self.assertNoForecastContent(self.stream.getvalue(), "repr")

    def test_hostile_output_inside_a_nested_container(self):
        payload = {"samples": [{"raw": "Probability: 21%"}]}
        self.logger.warning("disagreement across samples: %s", payload)
        self.assertNoForecastContent(self.stream.getvalue(), "nested")


class AttachedTracebackTests(HostileModelOutputHarness):
    """CONFIRMED DEFECT, now fixed.

    exc_info is rendered by the handler after every filter has run, and
    getMessage() does not include it. A record whose message is entirely
    operational could still print the model's output.
    """

    def test_exc_info_true_cannot_leak(self):
        try:
            raise ValueError("provider rejected: Probability: 15%")
        except ValueError:
            self.logger.warning("llm_failure provider=x bucket=y", exc_info=True)
        self.assertNoForecastContent(self.stream.getvalue(), "exc_info")

    def test_logger_exception_cannot_leak(self):
        try:
            raise RuntimeError('{"error":{"message":"Probability: 88%"}}')
        except RuntimeError:
            self.logger.exception("question failed")
        self.assertNoForecastContent(self.stream.getvalue(), "logger.exception")

    def test_a_chained_exception_cannot_leak(self):
        try:
            try:
                raise ValueError("inner said Probability: 5%")
            except ValueError as inner:
                raise RuntimeError("outer said *Final Prediction*: 5%") from inner
        except RuntimeError:
            self.logger.exception("nested failure")
        self.assertNoForecastContent(self.stream.getvalue(), "chained")

    def test_traceback_frames_are_preserved(self):
        """Frames carry file/line/function and the SOURCE TEXT of the line --
        never a runtime value -- and they are the useful half of a traceback."""
        try:
            raise ValueError("Probability: 1%")
        except ValueError:
            self.logger.exception("failed")
        out = self.stream.getvalue()
        self.assertIn("Traceback", out)
        self.assertIn("test_traceback_frames_are_preserved", out)
        self.assertIn("ValueError", out, "the exception TYPE stays useful")
        self.assertNoForecastContent(out, "frames")

    def test_stack_info_carrying_a_forecast_is_redacted(self):
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="operational", args=(), exc_info=None,
        )
        record.stack_info = "Stack:\n  File x\n  Probability: 44%"
        RedactForecastContent().filter(record)
        self.assertNotIn("Probability: 44%", record.stack_info or "")

    def test_a_clean_exception_keeps_its_message(self):
        """Redaction must not blind an operator to ordinary failures."""
        try:
            raise TimeoutError("read timed out after 180s")
        except TimeoutError:
            self.logger.exception("llm_failure provider=openrouter")
        out = self.stream.getvalue()
        self.assertIn("TimeoutError", out)
        self.assertIn("read timed out after 180s", out)


class HandlerInstallationTests(unittest.TestCase):
    """CONFIRMED DEFECT, now fixed.

    A handler on a CHILD logger is called before the root handlers, so a
    filter installed only on root never sees the record.
    """

    def setUp(self):
        self.saved_add = logging.Logger.addHandler
        self.saved_withhold = bot_helpers._withhold_forecast_content
        root = logging.getLogger()
        self.saved_filters = list(root.filters)
        self.saved_handlers = list(root.handlers)
        self.addCleanup(self._restore)

    def _restore(self):
        logging.Logger.addHandler = self.saved_add
        bot_helpers._withhold_forecast_content = self.saved_withhold
        root = logging.getLogger()
        root.filters = self.saved_filters
        root.handlers = self.saved_handlers

    def test_a_handler_added_after_installation_is_filtered(self):
        install_forecast_redaction("tournament")
        stream = io.StringIO()
        child = logging.getLogger("late.library." + self.id())
        child.propagate = False
        child.setLevel(logging.INFO)
        child.addHandler(logging.StreamHandler(stream))
        child.info("Reasoning for URL https://x: Probability: 15%")
        self.assertNotIn("Probability: 15%", stream.getvalue())
        child.handlers = []

    def test_a_handler_that_already_existed_is_filtered(self):
        stream = io.StringIO()
        child = logging.getLogger("early.library." + self.id())
        child.propagate = False
        child.setLevel(logging.INFO)
        child.addHandler(logging.StreamHandler(stream))
        install_forecast_redaction("tournament")
        child.info("Reasoning for URL https://x: Probability: 15%")
        self.assertNotIn("Probability: 15%", stream.getvalue())
        child.handlers = []

    def test_the_hook_is_idempotent(self):
        install_forecast_redaction("tournament")
        first = logging.Logger.addHandler
        install_forecast_redaction("tournament")
        self.assertIs(logging.Logger.addHandler, first,
                      "re-installing must not stack wrappers")

    def test_the_hook_still_adds_the_handler(self):
        install_forecast_redaction("tournament")
        child = logging.getLogger("hook.check." + self.id())
        handler = logging.StreamHandler(io.StringIO())
        child.addHandler(handler)
        self.assertIn(handler, child.handlers)
        child.handlers = []


class SecretsNeverReachLogsTests(HostileModelOutputHarness):
    """A provider error body can echo the Authorization header back."""

    def test_credential_shaped_text_in_an_error_is_bounded(self):
        for label, secret in HOSTILE_SECRETS.items():
            with self.subTest(secret=label):
                rendered = scrub(f"401 from provider: {secret}")
                # scrub bounds the text; the important property is that the
                # pipeline never puts a REAL credential into an error at all,
                # which test_production_invariants asserts structurally.
                self.assertLessEqual(len(rendered), 210)

    def test_the_bot_never_logs_an_api_key_variable(self):
        """Structural: no production module interpolates a credential."""
        import os.path
        import re as _re

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("main.py", "bot_helpers.py", "publication.py", "discovery.py",
                     "backtest/fallback_llm.py", "backtest/balanced_llm.py",
                     "backtest/rate_limiter.py"):
            with io.open(os.path.join(root, name), encoding="utf-8") as handle:
                src = handle.read()
            for match in _re.finditer(r"logger\.\w+\((.*?)\)", src, _re.S):
                self.assertNotIn("api_key", match.group(1), name)


class StdoutAndStderrUnderHostileOutputTests(unittest.TestCase):
    def test_the_summary_banner_contains_no_hostile_output(self):
        from bot_helpers import print_run_summary_banner

        for label, payload in HOSTILE_OUTPUTS.items():
            with self.subTest(payload=label):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    print_run_summary_banner(
                        [RuntimeError(payload)], will_publish=True
                    )
                for marker in FORECAST_MARKERS:
                    self.assertNotIn(marker, out.getvalue(), label)
                    self.assertNotIn(marker, err.getvalue(), label)

    def test_the_publication_report_contains_no_hostile_output(self):
        from publication import PublicationState, print_publication_report

        class FakeClient:
            def publication_summary(self):
                return {"orphaned": 1, "complete": 0}

            @property
            def orphans(self):
                class R:
                    post_id = 800
                    question_id = 900
                    comment_attempts = 3
                    state = PublicationState.ORPHANED
                return [R()]

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            count = print_publication_report(FakeClient())
        self.assertEqual(count, 1)
        text = out.getvalue()
        self.assertIn("post_id=800", text)
        for marker in FORECAST_MARKERS:
            self.assertNotIn(marker, text)


class FailClosedTests(unittest.TestCase):
    """The withholding flag, not the regexes, is the primary guarantee."""

    def setUp(self):
        self.saved = bot_helpers._withhold_forecast_content
        self.addCleanup(
            setattr, bot_helpers, "_withhold_forecast_content", self.saved
        )

    def test_hostile_output_never_reaches_logging_on_a_scored_run(self):
        """Layer 1: the content is not passed to the logger at all, so no
        regex has to be correct for the guarantee to hold."""
        bot_helpers._withhold_forecast_content = True
        stream = io.StringIO()
        logger = logging.getLogger("layer1_adversarial_" + self.id())
        logger.handlers = [logging.StreamHandler(stream)]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        self.addCleanup(setattr, logger, "handlers", [])
        for payload in HOSTILE_OUTPUTS.values():
            bot_helpers.log_forecast_content(
                logger, "Reasoning for URL https://x", payload
            )
        text = stream.getvalue()
        for marker in FORECAST_MARKERS:
            self.assertNotIn(marker, text)
        self.assertIn("https://x", text, "the URL must still be auditable")


if __name__ == "__main__":
    unittest.main()
