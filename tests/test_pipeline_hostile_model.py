"""The whole pipeline, offline, against a model that is trying to leak.

Every other test exercises one component. This one runs the REAL
``SummerTemplateBot2026`` from main.py -- research, five forecasts,
aggregation, publication -- with a model whose every response carries
credentials, ANSI escapes, JSON, a traceback and a probability, and captures
logging, stdout and stderr together.

Only two things are replaced, and both are network calls:
``MetaculusClient.post_question_comment`` and
``MetaculusClient._post_question_prediction``. The bot, the report, the
publication state machine, the redaction layers and the banners are all the
code that ships.

This is the test that would have caught the original R2 leak without anyone
having to guess which log line carried it.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import logging
import os
import unittest

from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    from forecasting_tools.ai_models.general_llm import GeneralLlm
    from forecasting_tools.data_models.questions import BinaryQuestion
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    import bot_helpers
    from bot_helpers import install_forecast_redaction, print_run_summary_banner
    from publication import PublishingClient, print_publication_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One response, used for research, all five rationales and the summary. Every
# line is a shape that has actually appeared in this pipeline's logs or in a
# provider error body.
HOSTILE_RESPONSE = (
    "(a) 22 days.\n"
    "(b) The status quo outcome is No.\n"
    "My key is AIza-SYNTHETIC-FIXTURE-NOT-A-REAL-CREDENTIAL-000 and "
    "Authorization: Bearer SYNTHETIC0000000000000000000000\n"
    "\x1b[31mTraceback (most recent call last):\x1b[0m\n"
    '{"choices":[{"text":"SECRET-RESEARCH-BODY"}]}\n'
    "Probability: 15%"
)

MUST_NOT_APPEAR = {
    "the stated probability": "Probability: 15%",
    "the prediction value": "0.15",
    "the research body": "SECRET-RESEARCH-BODY",
    "a credential-shaped string": "AIza-SYNTHETIC",
    "a bearer token": "SYNTHETIC00000000",
    "the model's rationale": "status quo outcome is No",
    "a raw provider payload": '"choices"',
}

MUST_APPEAR = {
    "the question URL": "questions/800",
    "the prediction confirmation": "publication_prediction_written",
    "the comment confirmation": "publication_comment_written",
}


def load_production_bot_class():
    """Import main.py without executing its __main__ block.

    Inside the real-package window: main.py does
    `from forecasting_tools import (...)` at module scope, which the suite's
    stubs would shadow.
    """
    with real_forecasting_tools():
        spec = importlib.util.spec_from_file_location(
            "production_main", os.path.join(ROOT, "main.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class HostileLlm(GeneralLlm):
    """A real GeneralLlm subclass -- run_research does isinstance() checks."""

    def __init__(self):
        super().__init__(
            model="openrouter/fake/hostile", temperature=0.3,
            timeout=180, allowed_tries=1,
        )

    async def invoke(self, prompt, system_prompt=None):  # noqa: ARG002
        return HOSTILE_RESPONSE


class HostileModelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.posted = []
        self._saved_comment = MetaculusClient.post_question_comment
        self._saved_prediction = MetaculusClient._post_question_prediction
        MetaculusClient.post_question_comment = (
            lambda self_, post_id, text, is_private=True, included_forecast=True:
            self.posted.append(("comment", post_id))
        )
        MetaculusClient._post_question_prediction = (
            lambda self_, question_id, payload:
            self.posted.append(("prediction", question_id))
        )
        self.addCleanup(setattr, MetaculusClient, "post_question_comment",
                        self._saved_comment)
        self.addCleanup(setattr, MetaculusClient, "_post_question_prediction",
                        self._saved_prediction)

        self.saved_withhold = bot_helpers._withhold_forecast_content
        self.addCleanup(setattr, bot_helpers, "_withhold_forecast_content",
                        self.saved_withhold)
        root = logging.getLogger()
        self.saved_filters, self.saved_handlers = list(root.filters), list(root.handlers)
        self.saved_add_handler = logging.Logger.addHandler
        self.addCleanup(self._restore_logging)

    def _restore_logging(self):
        logging.Logger.addHandler = self.saved_add_handler
        root = logging.getLogger()
        root.filters, root.handlers = self.saved_filters, self.saved_handlers

    def run_pipeline(self):
        """Everything main.py does for one question, with the streams captured."""
        botmod = load_production_bot_class()

        async def fake_structure_output(text, output_type, model=None, **kwargs):  # noqa: ARG001
            from forecasting_tools import BinaryPrediction

            return BinaryPrediction(prediction_in_decimal=0.15)

        botmod.structure_output = fake_structure_output

        class HostileBot(botmod.SummerTemplateBot2026):
            def get_llm(self, purpose="default", guarantee_type=None):  # noqa: ARG002
                return HostileLlm()

        log_stream = io.StringIO()
        root = logging.getLogger()
        root.handlers = [logging.StreamHandler(log_stream)]
        root.filters = []
        root.setLevel(logging.DEBUG)
        install_forecast_redaction("tournament")

        client = PublishingClient(sleep=lambda _s: None)
        client.note_question(900, 800)
        bot = HostileBot(
            metaculus_client=client,
            research_reports_per_question=1,
            predictions_per_research_report=5,
            publish_reports_to_metaculus=True,
            skip_previously_forecasted_questions=True,
            folder_to_save_reports_to=None,
        )
        question = BinaryQuestion(
            question_text="Will X happen?",
            id_of_post=800,
            id_of_question=900,
            page_url="https://www.metaculus.com/questions/800",
            background_info="background",
            resolution_criteria="criteria",
            fine_print="fine print",
        )

        out, err = io.StringIO(), io.StringIO()
        with real_forecasting_tools():
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                reports = asyncio.run(
                    bot.forecast_questions([question], return_exceptions=True)
                )
                print_publication_report(client)
                print_run_summary_banner(reports, will_publish=True)
        return {
            "reports": reports,
            "client": client,
            "log": log_stream.getvalue(),
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
        }

    # ------------------------------------------------------------------

    def test_no_hostile_content_reaches_any_stream(self):
        result = self.run_pipeline()
        combined = result["log"] + result["stdout"] + result["stderr"]
        for label, needle in MUST_NOT_APPEAR.items():
            self.assertNotIn(needle, combined, f"{label} reached a public stream")

    def test_each_stream_individually_is_clean(self):
        result = self.run_pipeline()
        for stream in ("log", "stdout", "stderr"):
            for label, needle in MUST_NOT_APPEAR.items():
                self.assertNotIn(needle, result[stream],
                                 f"{label} reached {stream}")

    def test_the_run_stays_operationally_observable(self):
        """Redaction must not blind the operator to what happened."""
        result = self.run_pipeline()
        combined = result["log"] + result["stdout"]
        for label, needle in MUST_APPEAR.items():
            self.assertIn(needle, combined, f"{label} was lost to redaction")

    def test_one_prediction_and_one_comment_are_published(self):
        result = self.run_pipeline()
        self.assertEqual(
            result["client"].publication_summary()["complete"], 1
        )
        self.assertEqual(
            [kind for kind, _id in self.posted], ["prediction", "comment"],
            "exactly one of each, prediction first",
        )

    def test_five_predictions_collapse_to_one_published_forecast(self):
        """The hostile model answers five times; that must remain one forecast."""
        result = self.run_pipeline()
        self.assertEqual(len(result["reports"]), 1)
        self.assertEqual(
            len([k for k, _ in self.posted if k == "prediction"]), 1
        )

    def test_no_orphan_on_the_happy_path(self):
        result = self.run_pipeline()
        self.assertEqual(result["client"].orphans, [])


if __name__ == "__main__":
    unittest.main()
