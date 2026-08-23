"""Discovery must read every page, and must never truncate in silence.

forecasting_tools 0.2.90 fetches offset=0 only when num_questions is None
(metaculus_client.py:761-766), which is what
get_all_open_questions_from_tournament passes. The 101st open post in a
tournament is therefore invisible, and the run still reports success.

These tests drive the real fetch_all_open_questions against a fake pager, so
what is under test is the loop, its termination condition, its deduplication
and its reporting -- not a restatement of them.
"""

from __future__ import annotations

import unittest

from tests._real_forecasting_tools import real_forecasting_tools

# discovery imports the real MetaculusClient/ApiFilter at import time, which
# the suite's stubs would otherwise shadow. See the helper for why.
with real_forecasting_tools():
    from discovery import (
        MAX_PAGES,
        fetch_all_open_questions,
        install_complete_pagination,
    )


class FakeQuestion:
    def __init__(self, question_id: int, post_id: int):
        self.id_of_question = question_id
        self.id_of_post = post_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Q({self.id_of_question}@{self.id_of_post})"


class FakePagingClient:
    """Serves a fixed corpus in pages, the way the real API does.

    Mirrors the real return contract exactly: (questions_after_local_filter,
    server_had_rows_before_local_filter). Those two differ, and conflating
    them is the bug this fake exists to expose.
    """

    MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST = 100

    def __init__(self, corpus, page_size=100, locally_filtered_pages=()):
        self.corpus = list(corpus)
        self.page_size = page_size
        self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST = page_size
        # Pages whose rows all vanish in local filtering but which the SERVER
        # did return rows for.
        self.locally_filtered_pages = set(locally_filtered_pages)
        self.offsets_requested: list[int] = []

    def _grab_filtered_questions_with_offset(self, api_filter, offset):
        self.offsets_requested.append(offset)
        page = self.corpus[offset : offset + self.page_size]
        server_had_rows = len(page) > 0
        page_index = offset // self.page_size
        if page_index in self.locally_filtered_pages:
            return [], server_had_rows
        return page, server_had_rows


def corpus(n: int, start: int = 1000):
    return [FakeQuestion(start + i, 5000 + i) for i in range(n)]


class PageCountTests(unittest.TestCase):
    def fetch(self, client):
        return fetch_all_open_questions(client, "minibench")

    def test_zero_questions(self):
        client = FakePagingClient(corpus(0))
        self.assertEqual(self.fetch(client), [])
        self.assertEqual(client.offsets_requested, [0])

    def test_one_question(self):
        client = FakePagingClient(corpus(1))
        self.assertEqual(len(self.fetch(client)), 1)
        self.assertEqual(client.offsets_requested, [0, 100])

    def test_ninety_nine_questions_take_one_full_page(self):
        client = FakePagingClient(corpus(99))
        self.assertEqual(len(self.fetch(client)), 99)

    def test_exactly_one_full_page(self):
        """100 is where the upstream implementation stops being correct."""
        client = FakePagingClient(corpus(100))
        self.assertEqual(len(self.fetch(client)), 100)
        self.assertEqual(client.offsets_requested, [0, 100])

    def test_one_hundred_and_one_questions_is_the_regression_case(self):
        client = FakePagingClient(corpus(101))
        found = self.fetch(client)
        self.assertEqual(len(found), 101,
                         "the 101st post is what upstream silently drops")
        self.assertEqual(client.offsets_requested, [0, 100, 200])

    def test_two_hundred_and_fifty_questions_across_three_pages(self):
        client = FakePagingClient(corpus(250))
        self.assertEqual(len(self.fetch(client)), 250)
        self.assertEqual(client.offsets_requested, [0, 100, 200, 300])

    def test_ids_are_preserved_in_order(self):
        client = FakePagingClient(corpus(150))
        found = self.fetch(client)
        self.assertEqual([q.id_of_question for q in found],
                         [1000 + i for i in range(150)])


class TerminationTests(unittest.TestCase):
    def test_an_empty_final_page_ends_the_walk(self):
        client = FakePagingClient(corpus(200))
        fetch_all_open_questions(client, "t")
        self.assertEqual(client.offsets_requested[-1], 200)

    def test_a_page_emptied_by_local_filters_does_not_end_the_walk(self):
        """The termination condition is 'the server had rows', not 'we kept
        some'. Using the latter loses every page after the first that local
        filtering happens to empty."""
        client = FakePagingClient(corpus(250), locally_filtered_pages={1})
        found = fetch_all_open_questions(client, "t")
        self.assertEqual(len(found), 150, "page 1 filtered out, pages 0 and 2 kept")
        self.assertIn(200, client.offsets_requested,
                      "must have kept paging past the filtered page")

    def test_the_page_limit_is_enforced_and_reported_as_an_error(self):
        # A corpus that never runs out: every offset returns a full page.
        class Endless(FakePagingClient):
            def _grab_filtered_questions_with_offset(self, api_filter, offset):
                self.offsets_requested.append(offset)
                return [FakeQuestion(offset + i, offset + i) for i in range(100)], True

        client = Endless(corpus(0))
        with self.assertLogs("discovery", level="ERROR") as captured:
            found = fetch_all_open_questions(client, "t", max_pages=5)
        self.assertEqual(len(found), 500)
        self.assertEqual(len(client.offsets_requested), 5)
        blob = "\n".join(captured.output)
        self.assertIn("discovery_page_limit_reached", blob)
        self.assertIn("discovery_may_be_incomplete", blob)

    def test_the_default_page_limit_is_explicit_and_generous(self):
        self.assertEqual(MAX_PAGES, 200)


class DeduplicationTests(unittest.TestCase):
    def test_a_question_served_on_two_pages_is_kept_once_and_reported(self):
        """order_by is not stable while questions open and close underneath a
        paging walk, so the same row genuinely can appear twice."""
        repeated = corpus(100) + [FakeQuestion(1000, 5000)] + corpus(5, start=2000)
        client = FakePagingClient(repeated)
        with self.assertLogs("discovery", level="WARNING") as captured:
            found = fetch_all_open_questions(client, "t")
        ids = [q.id_of_question for q in found]
        self.assertEqual(len(ids), len(set(ids)), "no duplicate question reached the bot")
        self.assertEqual(len(found), 105)
        self.assertIn("discovery_duplicate_questions", "\n".join(captured.output))

    def test_group_subquestions_sharing_a_post_are_all_kept(self):
        """Dedup is on the QUESTION id. Group subquestions legitimately share
        one post id, and dropping them would lose real forecasts."""
        group = [FakeQuestion(1000 + i, 5000) for i in range(4)]
        client = FakePagingClient(group)
        found = fetch_all_open_questions(client, "t")
        self.assertEqual(len(found), 4)
        self.assertEqual({q.id_of_post for q in found}, {5000})


class ObservabilityTests(unittest.TestCase):
    def test_the_completion_line_carries_the_numbers_an_operator_needs(self):
        client = FakePagingClient(corpus(150))
        with self.assertLogs("discovery", level="INFO") as captured:
            fetch_all_open_questions(client, "minibench")
        blob = "\n".join(captured.output)
        self.assertIn("discovery_complete", blob)
        self.assertIn("pages=3", blob)
        self.assertIn("questions=150", blob)
        self.assertIn("truncated=False", blob)

    def test_the_upstream_retrieved_line_is_preserved(self):
        """Every existing audit of this bot greps for it."""
        client = FakePagingClient(corpus(2))
        with self.assertLogs("discovery", level="INFO") as captured:
            fetch_all_open_questions(client, 33022)
        self.assertIn("Retrieved 2 questions from tournament 33022",
                      "\n".join(captured.output))


class InstallationTests(unittest.TestCase):
    def test_install_replaces_the_truncating_method(self):
        client = FakePagingClient(corpus(150))
        client.get_all_open_questions_from_tournament = lambda *a, **k: ["truncated"]
        install_complete_pagination(client)
        self.assertEqual(len(client.get_all_open_questions_from_tournament("t")), 150)

    def test_the_publishing_client_pages_and_records_post_ids(self):
        """The bot's real client must both page fully and learn the
        question -> post mapping the publication state machine needs."""
        from publication import PublishingClient

        client = PublishingClient()
        client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST = 100
        pager = FakePagingClient(corpus(150))
        client._grab_filtered_questions_with_offset = (
            pager._grab_filtered_questions_with_offset
        )
        found = client.get_all_open_questions_from_tournament("minibench")
        self.assertEqual(len(found), 150)
        self.assertEqual(client._question_to_post[1000], 5000)
        self.assertEqual(client._question_to_post[1149], 5149)


if __name__ == "__main__":
    unittest.main()


class HostilePageTests(unittest.TestCase):
    """Malformed and failing pages, which the API can genuinely produce."""

    def test_questions_without_an_id_are_kept_not_dropped(self):
        """A question the parser could not assign an id to is still a real
        question. Dropping it would be silent coverage loss -- the exact class
        of bug this module exists to remove."""
        page = [FakeQuestion(1000, 5000), FakeQuestion(None, 5001),
                FakeQuestion(1002, 5002)]
        client = FakePagingClient(page)
        found = fetch_all_open_questions(client, "t")
        self.assertEqual(len(found), 3)

    def test_several_id_less_questions_are_not_deduplicated_against_each_other(self):
        """None is not an identity. Two unidentified questions are two
        questions, not one seen twice."""
        page = [FakeQuestion(None, 5000), FakeQuestion(None, 5001)]
        client = FakePagingClient(page)
        self.assertEqual(len(fetch_all_open_questions(client, "t")), 2)

    def test_an_api_failure_mid_pagination_fails_the_run(self):
        """Fail closed. Returning pages 0-1 because page 2 errored would be a
        silent partial result, which is what R7 was about. The upstream
        request is already wrapped in @retry_with_exponential_backoff, so a
        failure that reaches here has survived three retries and the run
        should surface it rather than quietly forecast a subset."""

        class FailsOnThirdPage(FakePagingClient):
            def _grab_filtered_questions_with_offset(self, api_filter, offset):
                if offset >= 200:
                    raise ConnectionError("upstream gave up after its own retries")
                return super()._grab_filtered_questions_with_offset(api_filter, offset)

        client = FailsOnThirdPage(corpus(250))
        with self.assertRaises(ConnectionError):
            fetch_all_open_questions(client, "t")

    def test_a_page_of_the_wrong_shape_does_not_crash_the_walk(self):
        """`server_had_rows` is what terminates the loop; an empty list with a
        truthy flag must keep paging, not spin forever."""

        class EmptyButClaimsRows(FakePagingClient):
            def __init__(self):
                super().__init__([])
                self.calls = 0

            def _grab_filtered_questions_with_offset(self, api_filter, offset):
                self.calls += 1
                self.offsets_requested.append(offset)
                return [], self.calls <= 3

        client = EmptyButClaimsRows()
        with self.assertLogs("discovery", level="INFO") as captured:
            found = fetch_all_open_questions(client, "t", max_pages=10)
        self.assertEqual(found, [])
        self.assertEqual(client.calls, 4, "stops when the server stops claiming rows")
        self.assertIn("discovery_complete", "\n".join(captured.output))
