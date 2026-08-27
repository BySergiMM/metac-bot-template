"""The WhatsApp mini-dashboard system: events, state, dashboard, integration.

No test in this file makes a real network call. `notifications.integration
.handle_run`'s `send` parameter is always a fake in these tests; the one file
that could reach a real socket (`scripts/notify.py`) has its own test module
(`tests/test_notify_script.py`) where `urllib.request.urlopen` is always
mocked.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from notifications import dashboard
from notifications.events import Event, ForecastOutcome, compute_run_state, detect_events, select_headline
from notifications.integration import handle_run
from notifications.state import NotificationState, load_state, save_state

FIXED_NOW = __import__("datetime").datetime(2026, 8, 26, 17, 42)


# =====================================================  EVENTS: pure detection logic


def _outcome(qid=101, title="A question", published=True, had_minor_errors=False):
    return ForecastOutcome(question_id=qid, title=title, published=published,
                            had_minor_errors=had_minor_errors)


class NoNoveltyMeansNoEvents(unittest.TestCase):
    def test_empty_outcomes_produce_no_events(self):
        events = detect_events(outcomes=[], orphan_question_ids=[], run_id="r1",
                                state=NotificationState())
        self.assertEqual(events, [])

    def test_an_already_notified_question_produces_no_event_again(self):
        state = NotificationState(notified_event_ids=["new_question:101", "forecast_published:101"])
        events = detect_events(outcomes=[_outcome(101)], orphan_question_ids=[], run_id="r2",
                                state=state)
        self.assertEqual(events, [])


class NewQuestionEvents(unittest.TestCase):
    def test_one_new_question_produces_a_new_question_event(self):
        events = detect_events(outcomes=[_outcome(123, "Will X?")], orphan_question_ids=[],
                                run_id="r1", state=NotificationState())
        kinds = [e.kind for e in events]
        self.assertIn("new_question", kinds)

    def test_several_new_questions_all_get_events(self):
        outcomes = [_outcome(1, "Q1"), _outcome(2, "Q2"), _outcome(3, "Q3")]
        events = detect_events(outcomes=outcomes, orphan_question_ids=[], run_id="r1",
                                state=NotificationState())
        new_q = [e for e in events if e.kind == "new_question"]
        self.assertEqual(len(new_q), 3)

    def test_a_question_with_no_id_is_skipped_not_crashed_on(self):
        events = detect_events(outcomes=[_outcome(None, "mystery")], orphan_question_ids=[],
                                run_id="r1", state=NotificationState())
        self.assertEqual([e for e in events if e.kind == "new_question"], [])


class ForecastPublishedEvents(unittest.TestCase):
    def test_a_successful_report_produces_a_forecast_published_event(self):
        events = detect_events(outcomes=[_outcome(123, published=True)], orphan_question_ids=[],
                                run_id="r1", state=NotificationState())
        self.assertIn("forecast_published", [e.kind for e in events])

    def test_a_failed_attempt_produces_no_forecast_published_event(self):
        events = detect_events(outcomes=[_outcome(None, published=False)], orphan_question_ids=[],
                                run_id="r1", state=NotificationState())
        self.assertNotIn("forecast_published", [e.kind for e in events])

    def test_the_event_detail_never_carries_a_probability(self):
        events = detect_events(outcomes=[_outcome(123, published=True)], orphan_question_ids=[],
                                run_id="r1", state=NotificationState())
        published = [e for e in events if e.kind == "forecast_published"][0]
        self.assertNotIn("probability", published.detail)
        self.assertNotIn("probability", published.headline.lower())


class RunErrorAndPartialRun(unittest.TestCase):
    def test_all_failures_is_a_run_error(self):
        outcomes = [_outcome(None, published=False), _outcome(None, published=False)]
        events = detect_events(outcomes=outcomes, orphan_question_ids=[], run_id="r1",
                                state=NotificationState())
        self.assertIn("run_error", [e.kind for e in events])
        self.assertNotIn("partial_run", [e.kind for e in events])

    def test_a_mix_of_success_and_failure_is_a_partial_run(self):
        outcomes = [_outcome(1, published=True), _outcome(None, published=False)]
        events = detect_events(outcomes=outcomes, orphan_question_ids=[], run_id="r1",
                                state=NotificationState())
        self.assertIn("partial_run", [e.kind for e in events])
        self.assertNotIn("run_error", [e.kind for e in events])

    def test_all_success_produces_neither(self):
        outcomes = [_outcome(1, published=True), _outcome(2, published=True)]
        events = detect_events(outcomes=outcomes, orphan_question_ids=[], run_id="r1",
                                state=NotificationState())
        kinds = [e.kind for e in events]
        self.assertNotIn("run_error", kinds)
        self.assertNotIn("partial_run", kinds)

    def test_compute_run_state_matches(self):
        self.assertEqual(compute_run_state([]), "IDLE")
        self.assertEqual(compute_run_state([_outcome(1, published=True)]), "OK")
        self.assertEqual(compute_run_state([_outcome(None, published=False)]), "ERROR")
        self.assertEqual(
            compute_run_state([_outcome(1, published=True), _outcome(None, published=False)]),
            "PARTIAL",
        )


class PublicationOrphanEvents(unittest.TestCase):
    def test_an_orphan_id_produces_an_event(self):
        events = detect_events(outcomes=[], orphan_question_ids=[555], run_id="r1",
                                state=NotificationState())
        self.assertIn("publication_orphan", [e.kind for e in events])


class SelectHeadlinePriority(unittest.TestCase):
    def test_new_question_outranks_run_error(self):
        events = [
            Event("run_error:r1", "P0", "run_error", "err"),
            Event("new_question:1", "P0", "new_question", "new"),
        ]
        self.assertEqual(select_headline(events).kind, "new_question")

    def test_multiple_new_questions_collapse_into_one_batch_headline(self):
        events = [
            Event("new_question:1", "P0", "new_question", "Q1"),
            Event("new_question:2", "P0", "new_question", "Q2"),
        ]
        headline = select_headline(events)
        self.assertIn("2", headline.headline)

    def test_empty_events_has_no_headline(self):
        self.assertIsNone(select_headline([]))


# =====================================================  STATE: persistence


class StatePersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="notif-state-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = os.path.join(self.tmp, "state.json")

    def test_missing_file_loads_a_fresh_state(self):
        state = load_state(self.path)
        self.assertEqual(state.notified_event_ids, [])

    def test_corrupt_file_degrades_to_fresh_state_not_a_crash(self):
        with open(self.path, "w") as fh:
            fh.write("{not valid json")
        state = load_state(self.path)
        self.assertEqual(state.notified_event_ids, [])

    def test_round_trip_across_two_independent_load_calls(self):
        """Simulates two separate GitHub Actions runs: save in one process
        (conceptually), then a FRESH `load_state` call reads it back --
        nothing here is in-memory carryover."""
        state = NotificationState(notified_event_ids=["new_question:1", "forecast_published:1"])
        save_state(state, self.path)

        reloaded = load_state(self.path)
        self.assertEqual(sorted(reloaded.notified_event_ids),
                          ["forecast_published:1", "new_question:1"])

    def test_save_is_atomic_no_temp_file_left_behind(self):
        state = NotificationState(notified_event_ids=["x"])
        save_state(state, self.path)
        leftovers = [f for f in os.listdir(self.tmp) if f != "state.json"]
        self.assertEqual(leftovers, [])

    def test_unknown_fields_in_the_file_are_ignored_not_fatal(self):
        with open(self.path, "w") as fh:
            json.dump({"notified_event_ids": ["a"], "some_field_from_the_future": 42}, fh)
        state = load_state(self.path)
        self.assertEqual(state.notified_event_ids, ["a"])


# =====================================================  DASHBOARD: no invented metrics


class DashboardNeverInventsMetrics(unittest.TestCase):
    def test_zero_new_questions_omits_the_pendientes_line(self):
        headline = Event("forecast_published:1", "P0", "forecast_published", "🔮 Q1")
        text = dashboard.build_dashboard_text(
            headline=headline, run_state="OK", tournament_label="Summer AIB",
            new_question_count=0, forecasts_published_count=1, now=FIXED_NOW,
        )
        self.assertNotIn("Pendientes", text)

    def test_zero_forecasts_omits_the_forecasts_line(self):
        headline = Event("new_question:1", "P0", "new_question", "🆕 Q1")
        text = dashboard.build_dashboard_text(
            headline=headline, run_state="OK", tournament_label="Summer AIB",
            new_question_count=1, forecasts_published_count=0, now=FIXED_NOW,
        )
        self.assertNotIn("Forecasts", text)

    def test_no_probability_field_ever_appears(self):
        headline = Event("forecast_published:1", "P0", "forecast_published", "🔮 Q1 — title")
        text = dashboard.build_dashboard_text(
            headline=headline, run_state="OK", tournament_label="Summer AIB",
            new_question_count=0, forecasts_published_count=1, now=FIXED_NOW,
        )
        self.assertNotIn("Forecast:", text)  # the field name from the original mockup
        self.assertNotIn("0.", text)  # no stray decimal probability

    def test_message_stays_within_the_length_budget(self):
        headline = Event("new_question:1", "P0", "new_question",
                          "🆕 Q1 — " + "x" * 200)  # a pathologically long title
        text = dashboard.build_dashboard_text(
            headline=headline, run_state="OK", tournament_label="Summer AIB",
            new_question_count=1, forecasts_published_count=1, now=FIXED_NOW,
        )
        self.assertLessEqual(len(text), dashboard.MAX_MESSAGE_LENGTH)

    def test_error_state_uses_the_red_indicator(self):
        headline = Event("run_error:r1", "P0", "run_error", "🚨 2 de 2 fallaron")
        text = dashboard.build_dashboard_text(
            headline=headline, run_state="ERROR", tournament_label="Summer AIB",
            new_question_count=0, forecasts_published_count=0, now=FIXED_NOW,
        )
        self.assertIn("🔴", text)
        self.assertIn("ERROR", text)

    def test_timestamp_is_unambiguously_labelled_utc(self):
        headline = Event("new_question:1", "P0", "new_question", "🆕 Q1")
        text = dashboard.build_dashboard_text(
            headline=headline, run_state="OK", tournament_label="Summer AIB",
            new_question_count=1, forecasts_published_count=0, now=FIXED_NOW,
        )
        self.assertIn("UTC", text)


# =====================================================  INTEGRATION: end to end


class _Question:
    def __init__(self, qid, text):
        self.id_of_question = qid
        self.question_text = text


class _Report:
    def __init__(self, qid, text, errors=None):
        self.question = _Question(qid, text)
        self.errors = errors or []


class _Client:
    def __init__(self, orphans=None):
        self.orphans = orphans or []


class _Orphan:
    def __init__(self, question_id):
        self.question_id = question_id


class _Record:
    def __init__(self, state):
        self.state = state  # a plain string, matching PublicationState's own .value


class _ClientWithRecords:
    """A `PublishingClient`-shaped fake: `record_for` always returns
    SOMETHING (lazily "pending" by default), matching the real
    `PublishingClient.record_for`'s own lazy-create behaviour."""

    def __init__(self, states=None, orphans=None):
        self._states = states or {}
        self.orphans = orphans or []

    def record_for(self, question_id):
        return _Record(self._states.get(question_id, "pending"))


class HandleRunEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="notif-integration-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.state_path = os.path.join(self.tmp, "state.json")
        self.sent: list[str] = []

    def _send(self, message: str) -> bool:
        self.sent.append(message)
        return True

    def test_no_novelty_sends_nothing(self):
        result = handle_run(
            forecast_reports=[], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(result.events_detected, 0)
        self.assertFalse(result.message_sent)
        self.assertEqual(self.sent, [])

    def test_one_new_question_sends_exactly_one_message(self):
        result = handle_run(
            forecast_reports=[_Report(1, "Q1")], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertTrue(result.message_sent)
        self.assertEqual(len(self.sent), 1)

    def test_the_same_question_again_sends_nothing(self):
        handle_run(forecast_reports=[_Report(1, "Q1")], client=_Client(), tournament_label="AIB",
                   state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW)
        self.sent.clear()
        result2 = handle_run(
            forecast_reports=[_Report(1, "Q1")], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r2", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(result2.events_detected, 0)
        self.assertEqual(self.sent, [])

    def test_several_new_questions_send_exactly_one_message(self):
        reports = [_Report(i, "Q{0}".format(i)) for i in range(1, 6)]
        result = handle_run(
            forecast_reports=reports, client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(len(self.sent), 1)
        self.assertIn("5", self.sent[0])  # batch headline mentions the count

    def test_several_forecasts_send_one_aggregated_message(self):
        # Same call as above ALREADY published 5 forecasts too (new + published
        # fire together on a first sighting) -- so this asserts on a second
        # batch of forecasts for ALREADY-known questions never happens in this
        # notifier's scope (forecast_published only fires alongside new_question
        # here); the aggregate-in-one-message property is covered by the
        # previous test and this one's single `self.sent` count.
        reports = [_Report(i, "Q{0}".format(i)) for i in range(10, 13)]
        result = handle_run(
            forecast_reports=reports, client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(result.message_sent)

    def test_a_total_failure_sends_one_error_message(self):
        result = handle_run(
            forecast_reports=[RuntimeError("boom")], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(len(self.sent), 1)
        self.assertIn("🚨", self.sent[0])

    def test_questions_plus_forecasts_plus_error_is_one_summary_message(self):
        reports = [_Report(1, "Q1"), RuntimeError("boom")]
        result = handle_run(
            forecast_reports=reports, client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(len(self.sent), 1)
        self.assertGreaterEqual(result.events_detected, 2)  # new_question + partial_run at least

    def test_a_retry_with_the_same_run_id_does_not_duplicate(self):
        """Simulates a GitHub Actions retry: same run_id, same inputs, state
        already saved from the first attempt."""
        reports = [_Report(1, "Q1")]
        handle_run(forecast_reports=reports, client=_Client(), tournament_label="AIB",
                   state_path=self.state_path, run_id="SAME_RUN_ID", send=self._send, now=FIXED_NOW)
        self.sent.clear()
        result = handle_run(
            forecast_reports=reports, client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="SAME_RUN_ID", send=self._send, now=FIXED_NOW,
        )
        self.assertEqual(self.sent, [])
        self.assertEqual(result.events_detected, 0)

    def test_state_persists_across_two_separate_handle_run_calls(self):
        handle_run(forecast_reports=[_Report(42, "Q42")], client=_Client(), tournament_label="AIB",
                   state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW)
        state = load_state(self.state_path)
        self.assertIn("new_question:42", state.notified_event_ids)
        self.assertIn("forecast_published:42", state.notified_event_ids)

    def test_callmebot_failure_does_not_raise_and_bot_continues(self):
        def failing_send(message: str) -> bool:
            return False

        result = handle_run(
            forecast_reports=[_Report(1, "Q1")], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=failing_send, now=FIXED_NOW,
        )
        self.assertFalse(result.message_sent)
        self.assertIsNone(result.error)  # a failed SEND is not an integration error

    def test_a_failed_send_does_not_persist_state_so_it_retries_next_run(self):
        def failing_send(message: str) -> bool:
            return False

        handle_run(forecast_reports=[_Report(1, "Q1")], client=_Client(), tournament_label="AIB",
                   state_path=self.state_path, run_id="r1", send=failing_send, now=FIXED_NOW)
        state = load_state(self.state_path)
        self.assertEqual(state.notified_event_ids, [])

        # Next run, CallMeBot recovers -- the SAME event must still be sent.
        result = handle_run(
            forecast_reports=[_Report(1, "Q1")], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r2", send=self._send, now=FIXED_NOW,
        )
        self.assertTrue(result.message_sent)

    def test_an_exception_deep_inside_never_escapes_handle_run(self):
        class ExplodingClient:
            @property
            def orphans(self):
                raise RuntimeError("boom inside client")

        result = handle_run(
            forecast_reports=[_Report(1, "Q1")], client=ExplodingClient(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertIsNotNone(result.error)
        self.assertFalse(result.message_sent)

    def test_publication_orphan_produces_a_message(self):
        result = handle_run(
            forecast_reports=[], client=_Client(orphans=[_Orphan(999)]), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertTrue(result.message_sent)
        self.assertIn("999", self.sent[0])

    def test_error_report_field_never_contains_an_env_var_style_secret(self):
        """`RunOutcome.error` is the one field that could accidentally carry
        exception text. Simulate an integration-level crash whose message
        includes something secret-shaped, and confirm handle_run's own error
        field only ever carries the exception TYPE name (see integration.py),
        never str(exc)."""
        secret = "sk-should-never-appear-000111"

        class ExplodingClient:
            @property
            def orphans(self):
                raise RuntimeError(secret)

        result = handle_run(
            forecast_reports=[_Report(1, "Q1")], client=ExplodingClient(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertNotIn(secret, result.error or "")


# =====================================================  REGRESSION: forecast_published
# must mean CONFIRMED published (adversarial audit finding, see integration.py's
# _confirm_published)


class ForecastPublishedRequiresConfirmedPublication(unittest.TestCase):
    """Before this fix, `forecast_published` fired for ANY non-exception
    ForecastReport, regardless of whether the prediction was actually
    confirmed posted by `publication.PublishingClient`'s own bookkeeping.
    A report that LOOKS successful but whose PublicationRecord never left
    "pending" (the prediction POST itself did not confirm) used to produce
    a false "FORECAST PUBLICADO" WhatsApp."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="notif-confirm-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.state_path = os.path.join(self.tmp, "state.json")
        self.sent: list[str] = []

    def _send(self, message: str) -> bool:
        self.sent.append(message)
        return True

    def test_a_report_whose_record_is_still_pending_does_not_fire_forecast_published(self):
        client = _ClientWithRecords(states={123: "pending"})
        result = handle_run(
            forecast_reports=[_Report(123, "Q123")], client=client, tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertTrue(result.message_sent)  # new_question still fires
        self.assertNotIn("🔮", self.sent[0])  # but no forecast_published headline/body

    def test_a_report_confirmed_predicted_does_fire_forecast_published(self):
        client = _ClientWithRecords(states={123: "predicted"})
        result = handle_run(
            forecast_reports=[_Report(123, "Q123")], client=client, tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertIn("🔮", self.sent[0])

    def test_a_report_confirmed_complete_does_fire_forecast_published(self):
        client = _ClientWithRecords(states={123: "complete"})
        result = handle_run(
            forecast_reports=[_Report(123, "Q123")], client=client, tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertIn("🔮", self.sent[0])

    def test_an_orphaned_record_still_counts_as_published_the_prediction_is_live(self):
        client = _ClientWithRecords(states={123: "orphaned"})
        result = handle_run(
            forecast_reports=[_Report(123, "Q123")], client=client, tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertIn("🔮", self.sent[0])

    def test_a_pending_report_counts_toward_run_error_not_silently_ok(self):
        """new_question still outranks run_error for the HEADLINE (both fire
        here -- Q123 is new AND its only outcome is unconfirmed), but the
        dashboard's own Estado line must reflect ERROR, not OK -- a silent
        "OK" would be the same false-confidence bug from the opposite
        direction."""
        client = _ClientWithRecords(states={123: "pending"})
        result = handle_run(
            forecast_reports=[_Report(123, "Q123")], client=client, tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertIn("🔴", self.sent[0])
        self.assertIn("ERROR", self.sent[0])
        self.assertNotIn("🔮", self.sent[0])  # still must not claim forecast_published

    def test_a_client_without_record_for_falls_back_to_the_older_weaker_signal(self):
        """A caller that does not expose PublishingClient's bookkeeping
        (e.g. a bare test double) must not be treated as "everything
        failed" -- it degrades to trusting the report object, same as
        before this fix."""
        result = handle_run(
            forecast_reports=[_Report(123, "Q123")], client=_Client(), tournament_label="AIB",
            state_path=self.state_path, run_id="r1", send=self._send, now=FIXED_NOW,
        )
        self.assertIn("🔮", self.sent[0])


if __name__ == "__main__":
    unittest.main()
