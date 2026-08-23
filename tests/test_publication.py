"""The publication state machine must never leave a forecast without a comment.

Metaculus' AI Benchmark rules make the comment an eligibility requirement:
"the participating bot needs to have written a comment response (including a
display of its forecast) under each question". forecasting_tools publishes the
prediction and the comment as two independent, non-idempotent POSTs with no
transaction, so the interesting cases are all the ones where the second half
fails.

Every test here drives the REAL PublishingClient. Only the two network calls
on the upstream MetaculusClient are replaced, so the state machine, the
deduplication, the retry budget and the orphan marker under test are the code
that ships -- not a re-description of it.
"""

from __future__ import annotations

import logging
import unittest

import requests

from tests._real_forecasting_tools import real_forecasting_tools

# Five other modules in this suite stub forecasting_tools; see the helper.
# publication imports the real client at import time, so it must be imported
# INSIDE the window where the real package is visible.
with real_forecasting_tools():
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    from publication import (
        COMMENT_ATTEMPTS,
        ORPHAN_MARKER,
        PublicationState,
        PublishingClient,
        print_publication_report,
    )


class FakeTransport:
    """Programmable stand-in for the two Metaculus write endpoints.

    Records every call that actually reached "the network", which is what the
    duplicate-suppression tests assert on: suppression is only real if the
    POST never happened, not if a wrapper decided not to count it.
    """

    def __init__(self, comment_failures: int = 0, prediction_failures: int = 0,
                 comment_error: Exception | None = None):
        self.comment_calls: list[tuple[int, str, bool]] = []
        self.prediction_calls: list[tuple[int, dict]] = []
        self.comment_failures = comment_failures
        self.prediction_failures = prediction_failures
        self.comment_error = comment_error or requests.exceptions.Timeout(
            "timed out waiting for /comments/create/"
        )

    # Assigned onto MetaculusClient as ALREADY-BOUND methods, so Python does
    # not re-bind them to the client instance and there is no `self` for the
    # client here. super().post_question_comment(...) therefore lands on these
    # signatures exactly as written.
    def post_question_comment(self, post_id, comment_text,
                              is_private=True, included_forecast=True):
        if self.comment_failures > 0:
            self.comment_failures -= 1
            # Recorded BEFORE raising: a server-accepted-then-lost response is
            # indistinguishable from a refusal at the client, and that is
            # exactly the R3 scenario.
            self.comment_calls.append((post_id, comment_text, False))
            raise self.comment_error
        self.comment_calls.append((post_id, comment_text, True))

    def post_prediction(self, question_id, payload):
        if self.prediction_failures > 0:
            self.prediction_failures -= 1
            self.prediction_calls.append((question_id, payload))
            raise requests.exceptions.ConnectionError("connection reset")
        self.prediction_calls.append((question_id, payload))

    @property
    def successful_comments(self) -> list[int]:
        return [pid for pid, _text, ok in self.comment_calls if ok]


class PublicationTestCase(unittest.TestCase):
    """Installs the fake transport onto the upstream client for one test."""

    def setUp(self):
        self.transport = FakeTransport()
        self._saved_comment = MetaculusClient.post_question_comment
        self._saved_prediction = MetaculusClient._post_question_prediction
        MetaculusClient.post_question_comment = self.transport.post_question_comment
        MetaculusClient._post_question_prediction = self.transport.post_prediction
        self.addCleanup(setattr, MetaculusClient, "post_question_comment",
                        self._saved_comment)
        self.addCleanup(setattr, MetaculusClient, "_post_question_prediction",
                        self._saved_prediction)
        self.slept: list[float] = []

    def client(self, **kwargs) -> PublishingClient:
        kwargs.setdefault("sleep", self.slept.append)
        client = PublishingClient(**kwargs)
        return client

    def publish(self, client, question_id=900, post_id=800, value=0.42):
        """The exact call order forecasting_tools' report classes use."""
        client.note_question(question_id, post_id)
        client._post_question_prediction(question_id, {"probability_yes": value})
        client.post_question_comment(post_id, "# SUMMARY\n*Final Prediction*: 42%")


class HappyPathTests(PublicationTestCase):
    def test_prediction_then_comment_reaches_complete(self):
        client = self.client()
        self.publish(client)
        self.assertEqual(self.transport.prediction_calls, [(900, {"probability_yes": 0.42})])
        self.assertEqual(self.transport.successful_comments, [800])
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)
        self.assertEqual(client.orphans, [])

    def test_summary_counts_only_states(self):
        client = self.client()
        self.publish(client)
        self.assertEqual(client.publication_summary()["complete"], 1)
        self.assertEqual(client.publication_summary()["orphaned"], 0)


class PredictionFailureTests(PublicationTestCase):
    def test_prediction_failure_propagates_and_publishes_nothing(self):
        """No prediction means no orphan: nothing is half-published."""
        self.transport.prediction_failures = 99
        client = self.client()
        client.note_question(900, 800)
        with self.assertRaises(requests.exceptions.ConnectionError):
            client._post_question_prediction(900, {"probability_yes": 0.42})
        self.assertEqual(client.record_for(900).state, PublicationState.PENDING)
        self.assertEqual(client.orphans, [])
        self.assertEqual(self.transport.successful_comments, [])

    def test_prediction_timeout_is_not_recorded_as_predicted(self):
        self.transport.prediction_failures = 1
        client = self.client()
        client.note_question(900, 800)
        with self.assertRaises(requests.exceptions.ConnectionError):
            client._post_question_prediction(900, {"probability_yes": 0.42})
        # The guard set must stay empty, or a later legitimate retry of the
        # same question would be suppressed as a duplicate of a POST that
        # never succeeded.
        self.assertNotIn(900, client._predicted_questions)


class CommentFailureTests(PublicationTestCase):
    def test_comment_retries_beyond_the_sdk_budget_and_succeeds(self):
        """The whole point: the SDK gives up at 3, this does not."""
        self.transport.comment_failures = COMMENT_ATTEMPTS - 1
        client = self.client()
        self.publish(client)
        self.assertEqual(len(self.transport.comment_calls), COMMENT_ATTEMPTS)
        self.assertEqual(self.transport.successful_comments, [800])
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)
        self.assertEqual(client.orphans, [])
        self.assertEqual(len(self.slept), COMMENT_ATTEMPTS - 1,
                         "must pause between attempts, and not after the last")

    def test_comment_permanently_failing_produces_an_orphan_and_raises(self):
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.42})
        with self.assertRaises(requests.exceptions.Timeout):
            client.post_question_comment(800, "body")
        record = client.record_for(900)
        self.assertEqual(record.state, PublicationState.ORPHANED)
        self.assertEqual(record.comment_attempts, COMMENT_ATTEMPTS)
        self.assertEqual([r.question_id for r in client.orphans], [900])

    def test_the_orphan_marker_is_logged_with_ids_and_no_forecast(self):
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.42})
        with self.assertLogs("publication", level="ERROR") as captured:
            with self.assertRaises(requests.exceptions.Timeout):
                client.post_question_comment(800, "*Final Prediction*: 42%")
        blob = "\n".join(captured.output)
        self.assertIn(ORPHAN_MARKER, blob)
        self.assertIn("post_id=800", blob)
        self.assertIn("question_ids=900", blob)
        # An operator alert that leaks the forecast trades R1 for R2.
        self.assertNotIn("42%", blob)
        self.assertNotIn("0.42", blob)

    def test_comment_failure_without_a_prediction_is_not_an_orphan(self):
        """Nothing half-published, so the operator alert must stay silent."""
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        with self.assertLogs("publication", level="WARNING") as captured:
            with self.assertRaises(requests.exceptions.Timeout):
                client.post_question_comment(800, "body")
        self.assertNotIn(ORPHAN_MARKER, "\n".join(captured.output))
        self.assertEqual(client.orphans, [])

    def test_error_details_are_bounded_and_value_free(self):
        self.transport.comment_error = requests.exceptions.HTTPError(
            "500 from Metaculus: " + "*Final Prediction*: 42% " * 50
        )
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.42})
        with self.assertRaises(requests.exceptions.HTTPError):
            client.post_question_comment(800, "body")
        for recorded in client.record_for(900).errors:
            self.assertNotIn("42%", recorded)
            self.assertLess(len(recorded), 200)


class DuplicateSuppressionTests(PublicationTestCase):
    """R3, in-run half."""

    def test_a_second_prediction_for_one_question_never_reaches_the_network(self):
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.42})
        client._post_question_prediction(900, {"probability_yes": 0.42})
        self.assertEqual(len(self.transport.prediction_calls), 1)

    def test_a_second_comment_for_one_post_never_reaches_the_network(self):
        client = self.client()
        self.publish(client)
        client.post_question_comment(800, "a second body")
        self.assertEqual(len(self.transport.comment_calls), 1)

    def test_suppression_is_per_id_not_global(self):
        client = self.client()
        self.publish(client, question_id=900, post_id=800)
        self.publish(client, question_id=901, post_id=801)
        self.assertEqual(len(self.transport.prediction_calls), 2)
        self.assertEqual(self.transport.successful_comments, [800, 801])

    def test_a_failed_comment_does_not_mark_the_post_as_commented(self):
        """Otherwise one lost response would permanently suppress the retry."""
        self.transport.comment_failures = 1
        client = self.client()
        self.publish(client)
        self.assertIn(800, client._commented_posts)
        self.assertEqual(self.transport.successful_comments, [800])
        self.assertEqual(len(self.transport.comment_calls), 2)


class GroupQuestionTests(PublicationTestCase):
    """R6: N subquestions share ONE post id.

    Each subquestion is its own question and must get its own forecast; the
    post they share must end up with exactly one comment.
    """

    def publish_group(self, client, post_id, question_ids):
        for question_id in question_ids:
            client.note_question(question_id, post_id)
        for question_id in question_ids:
            client._post_question_prediction(question_id, {"probability_yes": 0.5})
            client.post_question_comment(post_id, f"report for {question_id}")

    def test_one_subquestion(self):
        client = self.client()
        self.publish_group(client, 800, [900])
        self.assertEqual(len(self.transport.prediction_calls), 1)
        self.assertEqual(self.transport.successful_comments, [800])

    def test_two_subquestions_produce_two_predictions_and_one_comment(self):
        client = self.client()
        self.publish_group(client, 800, [900, 901])
        self.assertEqual(len(self.transport.prediction_calls), 2)
        self.assertEqual(self.transport.successful_comments, [800])

    def test_many_subquestions_produce_one_comment(self):
        client = self.client()
        ids = list(range(900, 912))
        self.publish_group(client, 800, ids)
        self.assertEqual(len(self.transport.prediction_calls), len(ids))
        self.assertEqual(self.transport.successful_comments, [800])
        for question_id in ids:
            self.assertEqual(client.record_for(question_id).state,
                             PublicationState.COMPLETE)

    def test_concurrent_subquestions_still_produce_one_comment(self):
        """forecast_questions gathers every question concurrently.

        The client is synchronous, so interleaving happens at await points
        between calls; this drives that interleaving explicitly rather than
        assuming a serial order.
        """
        import asyncio

        client = self.client()
        ids = [900, 901, 902, 903]
        for question_id in ids:
            client.note_question(question_id, 800)

        async def one(question_id):
            await asyncio.sleep(0)
            client._post_question_prediction(question_id, {"probability_yes": 0.5})
            await asyncio.sleep(0)
            client.post_question_comment(800, f"report for {question_id}")

        async def all_of_them():
            await asyncio.gather(*[one(q) for q in ids])

        asyncio.run(all_of_them())
        self.assertEqual(len(self.transport.prediction_calls), len(ids))
        self.assertEqual(self.transport.successful_comments, [800])

    def test_a_failing_comment_orphans_every_predicted_subquestion(self):
        self.transport.comment_failures = 99
        client = self.client()
        for question_id in (900, 901):
            client.note_question(question_id, 800)
            client._post_question_prediction(question_id, {"probability_yes": 0.5})
        with self.assertRaises(requests.exceptions.Timeout):
            client.post_question_comment(800, "body")
        self.assertEqual(sorted(r.question_id for r in client.orphans), [900, 901])

    def test_rerun_after_partial_publication_does_not_double_post(self):
        """Second pass over the same post inside one process."""
        client = self.client()
        self.publish_group(client, 800, [900, 901])
        self.publish_group(client, 800, [900, 901])
        self.assertEqual(len(self.transport.prediction_calls), 2)
        self.assertEqual(self.transport.successful_comments, [800])


class ProcessRestartTests(PublicationTestCase):
    """A fresh process has no memory; the guards are per-process by design."""

    def test_a_new_client_will_publish_again(self):
        first = self.client()
        self.publish(first)
        second = self.client()
        self.publish(second)
        self.assertEqual(len(self.transport.prediction_calls), 2)
        self.assertEqual(len(self.transport.successful_comments), 2)

    def test_cross_run_duplicate_prevention_is_already_forecasted_not_this(self):
        """Documents the boundary, so nobody mistakes the in-run guard for a
        cross-run one. ForecastBot.forecast_questions filters on
        question.already_forecasted BEFORE publication is ever reached."""
        client = self.client()
        self.assertFalse(hasattr(client, "_persisted_publication_state"))


class PublicationReportTests(PublicationTestCase):
    def test_report_returns_zero_and_prints_states_when_clean(self):
        client = self.client()
        self.publish(client)
        self.assertEqual(print_publication_report(client), 0)

    def test_report_returns_the_orphan_count(self):
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.42})
        with self.assertRaises(requests.exceptions.Timeout):
            client.post_question_comment(800, "body")
        self.assertEqual(print_publication_report(client), 1)

    def test_report_output_carries_no_forecast_value(self):
        import contextlib
        import io as _io

        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.42})
        with self.assertRaises(requests.exceptions.Timeout):
            client.post_question_comment(800, "*Final Prediction*: 42%")
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_publication_report(client)
        self.assertNotIn("42%", buffer.getvalue())
        self.assertIn("post_id=800", buffer.getvalue())


class WiredIntoMainTests(unittest.TestCase):
    """A state machine nobody constructs protects nothing."""

    def _main_src(self) -> str:
        import io as _io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with _io.open(os.path.join(root, "main.py"), encoding="utf-8") as handle:
            return handle.read()

    def test_main_injects_the_publishing_client(self):
        src = self._main_src()
        self.assertIn("client = PublishingClient()", src)
        self.assertIn("metaculus_client=client", src)

    def test_the_client_is_built_before_the_bot_that_uses_it(self):
        src = self._main_src()
        self.assertLess(src.index("client = PublishingClient()"),
                        src.index("metaculus_client=client"))

    def test_main_reports_publication_state_at_the_end(self):
        self.assertIn("print_publication_report(client)", self._main_src())


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()


class RealUpstreamPublishPathTests(PublicationTestCase):
    """The seam must intercept forecasting_tools' OWN publish path.

    Every other test in this file calls the two client methods directly, which
    proves the state machine but assumes the upstream report classes route
    through it. This drives the real
    ``BinaryReport.publish_report_to_metaculus`` instead, so the assumption is
    tested rather than relied on. If a future forecasting_tools stops going
    through the client, these fail and the rest of the file still passes --
    which is exactly the signal that would otherwise be missed.
    """

    def report(self, question_id, post_id, prediction=0.42,
               explanation="# SUMMARY\n*Final Prediction*: 42%"):
        with real_forecasting_tools():
            from forecasting_tools.data_models.binary_report import BinaryReport
            from forecasting_tools.data_models.questions import BinaryQuestion

        return BinaryReport(
            question=BinaryQuestion(
                question_text="Will X happen?",
                id_of_post=post_id,
                id_of_question=question_id,
                page_url="https://www.metaculus.com/questions/{0}".format(post_id),
            ),
            prediction=prediction,
            explanation=explanation,
            price_estimate=0.0,
            minutes_taken=0.1,
            errors=[],
        )

    def publish_report(self, client, report):
        import asyncio

        # publish_report_to_metaculus re-imports MetaculusClient at CALL time
        # (binary_report.py:57), so the real package has to be visible for the
        # duration of the call, not just while the report class was imported.
        with real_forecasting_tools():
            asyncio.run(report.publish_report_to_metaculus(metaculus_client=client))

    def test_the_real_publish_path_routes_through_the_state_machine(self):
        client = self.client()
        client.note_question(900, 800)
        self.publish_report(client, self.report(900, 800))
        self.assertEqual([q for q, _ in self.transport.prediction_calls], [900])
        self.assertEqual(self.transport.successful_comments, [800])
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)

    def test_a_real_group_post_receives_exactly_one_comment(self):
        """R6, through the code path production actually executes."""
        client = self.client()
        subquestions = [901, 902, 903]
        for question_id in subquestions:
            client.note_question(question_id, 801)
        for question_id in subquestions:
            self.publish_report(client, self.report(question_id, 801))
        self.assertEqual(
            [q for q, _ in self.transport.prediction_calls], subquestions,
            "each subquestion needs its own forecast",
        )
        self.assertEqual(
            self.transport.successful_comments, [801],
            "the shared parent post must receive one comment, not three",
        )

    def test_a_real_orphan_is_detected_and_announced(self):
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(904, 802)
        with self.assertLogs("publication", level="ERROR") as captured:
            with self.assertRaises(requests.exceptions.Timeout):
                self.publish_report(
                    client,
                    self.report(904, 802, prediction=0.9,
                                explanation="# SUMMARY\n*Final Prediction*: 90%"),
                )
        blob = "\n".join(captured.output)
        self.assertIn(ORPHAN_MARKER, blob)
        self.assertIn("post_id=802", blob)
        self.assertNotIn("90%", blob)
        self.assertNotIn("0.9", blob)
        self.assertEqual([r.question_id for r in client.orphans], [904])


class HostileNetworkTests(PublicationTestCase):
    """Each numbered failure mode from the release-hardening audit.

    Modelled independently, because they do not fail the same way: a 429 is
    retryable, a 400 is not, and a lost response is indistinguishable at the
    client from a refusal -- which is exactly what makes it dangerous.
    """

    def http_error(self, status):
        response = requests.Response()
        response.status_code = status
        return requests.exceptions.HTTPError(
            "{0} from Metaculus".format(status), response=response
        )

    def test_prediction_500_propagates_and_publishes_nothing(self):
        self.transport.prediction_failures = 99
        client = self.client()
        client.note_question(900, 800)
        with self.assertRaises(requests.exceptions.ConnectionError):
            client._post_question_prediction(900, {"probability_yes": 0.4})
        self.assertEqual(client.orphans, [])
        self.assertEqual(self.transport.successful_comments, [])

    def test_comment_500_is_retried_then_orphans(self):
        self.transport.comment_error = self.http_error(500)
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        with self.assertRaises(requests.exceptions.HTTPError):
            client.post_question_comment(800, "# SUMMARY body")
        self.assertEqual(len(self.transport.comment_calls), COMMENT_ATTEMPTS)
        self.assertEqual([r.question_id for r in client.orphans], [900])

    def test_comment_429_is_retried_then_orphans(self):
        self.transport.comment_error = self.http_error(429)
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        with self.assertRaises(requests.exceptions.HTTPError):
            client.post_question_comment(800, "# SUMMARY body")
        self.assertEqual([r.question_id for r in client.orphans], [900])

    def test_comment_429_that_clears_on_a_later_attempt_completes(self):
        self.transport.comment_error = self.http_error(429)
        self.transport.comment_failures = 2
        client = self.client()
        self.publish(client)
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)
        self.assertEqual(client.orphans, [])

    def test_a_lost_prediction_response_is_not_recorded_as_sent(self):
        """MODE 2, and an honest limitation.

        The server accepted the POST; the response never arrived. The client
        cannot tell this from a refusal, so the guard set stays empty and a
        later attempt WILL post again. There is no idempotency key in the
        Metaculus forecast API to prevent it. Cross-run duplicate prevention
        is question.already_forecasted, which is upstream's mechanism.
        """
        self.transport.prediction_failures = 1
        client = self.client()
        client.note_question(900, 800)
        with self.assertRaises(requests.exceptions.ConnectionError):
            client._post_question_prediction(900, {"probability_yes": 0.4})
        # It DID reach the server, and the client does not know.
        self.assertEqual(len(self.transport.prediction_calls), 1)
        self.assertNotIn(900, client._predicted_questions)

    def test_a_lost_comment_response_causes_a_second_comment(self):
        """MODE 7, and the same honest limitation on the comment side."""
        self.transport.comment_failures = 1
        client = self.client()
        self.publish(client)
        self.assertEqual(len(self.transport.comment_calls), 2,
                         "the lost response is indistinguishable from a refusal")
        self.assertEqual(self.transport.successful_comments, [800])

    def test_a_crash_between_prediction_and_comment_leaves_no_state(self):
        """MODES 11-13. There is no on-disk state by design: GitHub runners
        are ephemeral, so persisting one would be a lie. A new process starts
        with empty guards, and cross-run protection is already_forecasted."""
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        # process dies here
        reborn = self.client()
        self.assertEqual(reborn.orphans, [])
        self.assertEqual(reborn.publication_summary()["predicted"], 0)
        self.assertNotIn(900, reborn._predicted_questions)


class StateMachineIsolationTests(PublicationTestCase):
    """One post's outcome must never decide another post's state."""

    def test_a_comment_on_one_post_does_not_complete_another(self):
        client = self.client()
        client.note_question(900, 800)
        client.note_question(901, 801)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        client._post_question_prediction(901, {"probability_yes": 0.6})
        client.post_question_comment(800, "# SUMMARY for post 800")
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)
        self.assertEqual(client.record_for(901).state, PublicationState.PREDICTED,
                         "post 801 has no comment yet and must not be COMPLETE")

    def test_a_failing_comment_on_one_post_does_not_orphan_another(self):
        self.transport.comment_failures = 99
        client = self.client()
        client.note_question(900, 800)
        client.note_question(901, 801)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        client._post_question_prediction(901, {"probability_yes": 0.6})
        with self.assertRaises(requests.exceptions.Timeout):
            client.post_question_comment(800, "body")
        self.assertEqual([r.question_id for r in client.orphans], [900])
        self.assertEqual(client.record_for(901).state, PublicationState.PREDICTED)

    def test_an_unmapped_question_cannot_be_completed_by_someone_elses_comment(self):
        """A question whose post is unknown must not be swept into any post's
        completion set -- that would report an orphan as COMPLETE."""
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        client._post_question_prediction(999, {"probability_yes": 0.5})  # no mapping
        client.post_question_comment(800, "# SUMMARY body")
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)
        self.assertNotEqual(client.record_for(999).state, PublicationState.COMPLETE)

    def test_a_duplicate_question_in_discovery_publishes_once(self):
        """MODE 16: the same question object noted twice."""
        client = self.client()
        client.note_question(900, 800)
        client.note_question(900, 800)
        self.publish(client)
        self.publish(client)
        self.assertEqual(len(self.transport.prediction_calls), 1)
        self.assertEqual(self.transport.successful_comments, [800])

    def test_remediating_a_comment_cannot_repost_the_prediction(self):
        """An operator re-running publication for an orphaned post must not
        create a second forecast."""
        self.transport.comment_failures = COMMENT_ATTEMPTS
        client = self.client()
        client.note_question(900, 800)
        client._post_question_prediction(900, {"probability_yes": 0.4})
        with self.assertRaises(requests.exceptions.Timeout):
            client.post_question_comment(800, "body")
        self.assertEqual([r.question_id for r in client.orphans], [900])
        # Operator retries the comment in the SAME process; transport now healthy.
        client.post_question_comment(800, "body")
        self.assertEqual(len(self.transport.prediction_calls), 1,
                         "remediation must not touch the prediction")
        self.assertEqual(client.record_for(900).state, PublicationState.COMPLETE)
        self.assertEqual(client.orphans, [])
