"""The publication state machine: one forecast, one prediction, one comment.

Why this exists
---------------
``publish_report_to_metaculus`` (forecasting_tools 0.2.90,
``binary_report.py:66-71`` and the three sibling report types) does exactly
two things, in this order, with nothing tying them together::

    metaculus_client.post_binary_question_prediction(id_of_question, value)
    metaculus_client.post_question_comment(id_of_post, explanation)

Both are wrapped in ``@retry_with_exponential_backoff(max_retries=3)``. Neither
is idempotent, and there is no transaction. Three consequences, all real:

R1  Prediction lands, comment exhausts its retries. The prediction is now on
    Metaculus. ``MetaculusQuestion.already_forecasted`` is derived from
    ``question["my_forecasts"]["history"]`` (``questions.py:138-139``), so it
    is now True, and ``ForecastBot.forecast_questions`` (``forecast_bot.py:233``)
    drops the question from every future run. The comment is never written.
    Metaculus' AI Benchmark rules require "a comment response (including a
    display of its forecast) under each question" for eligibility, so this is
    an eligibility failure that cannot self-heal.

R3  The retry decorator retries on ``requests.exceptions.Timeout`` and
    ``ConnectionError``. A POST the server accepted but whose response was
    lost is retried, producing a duplicate. For a comment that is visible
    duplicate content.

R6  ``_unpack_group_question`` (``metaculus_client.py:685``) deep-copies the
    parent post once per subquestion, so N subquestions share ONE
    ``id_of_post`` while having distinct ``id_of_question``. Publication posts
    the prediction to the question and the comment to the POST, and
    ``forecast_questions`` runs every question concurrently through a bare
    ``asyncio.gather``. N subquestions therefore write N comments onto one
    parent post.

Where it intervenes, and why there
----------------------------------
``PublishingClient`` subclasses ``MetaculusClient`` and is injected via
``ForecastBot(metaculus_client=...)`` (``forecast_bot.py:89``, used at ``:166``
for discovery and ``:416`` for publication).

The client, not the bot, is the seam. ``_run_individual_question`` offers no
hook around publication, and the report type -- not the caller -- fixes the
order of the two POSTs. But both POSTs are *client methods*, so overriding
them intercepts every question type, every report class and every future
upstream refactor of the calling code, without reimplementing any of it.

``question_id`` and ``post_id`` are associated during discovery, which this
same client performs, so by publication time the mapping is already known and
nothing has to be threaded through upstream code that does not know about it.

It enforces, per process:

* one prediction per ``question_id``            -- R3, in-run half
* one comment per ``post_id``                   -- R3 and R6, in-run half
* an extended, bounded retry budget for the comment specifically, because the
  comment is the eligibility-critical half and the SDK spends the same three
  tries on it as on anything else
* an explicit state machine, so "prediction written, comment not" is a named
  state rather than an absence
* an unmistakable, greppable marker when that state is reached

What it deliberately does NOT do
--------------------------------
It never changes a forecast value, never adds or removes a prediction, and
never invents a comment body. It decides *whether* a POST is issued and *how
often*, nothing else. Aggregation, the success threshold and the report text
are all decided upstream and are untouched.

The ordering question, and why the order is unchanged
-----------------------------------------------------
Reversing to comment-then-prediction WOULD make R1 structural rather than
probabilistic: the prediction would only ever be written after the comment
succeeded, so "a prediction exists" would imply "a comment exists", and
``already_forecasted`` -- the very flag that makes R1 permanent -- would
become a correct completion marker. The residual failure inverts into a
recoverable one (comment written, prediction not, question still
``already_forecasted == False``, so the next run retries it in full).

It is NOT done here, for two reasons that outweigh it:

1. The order is fixed inside each report type's ``publish_report_to_metaculus``.
   Changing it means reimplementing publication for four report types, which
   is a far larger blast radius than the bug.
2. ``post_question_comment`` defaults to ``included_forecast=True``, and
   whether Metaculus accepts that on a question the author has not yet
   forecast could not be verified without production credentials. Putting an
   unverified server interaction into the critical publication path, when the
   current order has published successfully in production three times out of
   three (runs 32239144510, 32268972325, 32366841649), trades a rare failure
   for an unmeasured one.

So R1 is reduced here, not eliminated. See ``docs/publication.md``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from forecasting_tools.helpers.metaculus_client import MetaculusClient

logger = logging.getLogger(__name__)


class PublicationState(str, Enum):
    """Where one question got to.

    ORPHANED is the state R1 describes and the only one that needs an
    operator: a prediction is live on Metaculus with no comment beside it.
    """

    PENDING = "pending"
    PREDICTED = "predicted"
    COMPLETE = "complete"
    ORPHANED = "orphaned"


#: Greppable in a workflow log and in `gh run view --log`. Deliberately a
#: token, not prose: an operator alert must not break because someone
#: rephrased a sentence.
ORPHAN_MARKER = "PUBLICATION_ORPHAN"

#: Attempts the comment gets from THIS module, each of which already contains
#: the SDK's own three tries with exponential backoff. 3 x 3 = up to nine
#: chances for the eligibility-critical half, against the three it had before.
COMMENT_ATTEMPTS = 3

#: Seconds between our own comment attempts. Fixed and short: the SDK already
#: applies exponential backoff with jitter inside each attempt, so this is a
#: settling pause between independent attempts, not a second backoff ladder.
COMMENT_RETRY_DELAY_SECONDS = 5.0


@dataclass
class PublicationRecord:
    question_id: int
    post_id: int | None = None
    state: PublicationState = PublicationState.PENDING
    comment_attempts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_orphaned(self) -> bool:
        return self.state is PublicationState.ORPHANED


class PublishingClient(MetaculusClient):
    """A MetaculusClient that publishes at most once and never silently."""

    def __init__(
        self,
        *args: Any,
        comment_attempts: int = COMMENT_ATTEMPTS,
        sleep: Callable[[float], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.comment_attempts = comment_attempts
        # Injectable so tests prove the retry pacing without sleeping.
        self._sleep = sleep if sleep is not None else time.sleep
        self._records: dict[int, PublicationRecord] = {}
        self._question_to_post: dict[int, int] = {}
        self._commented_posts: set[int] = set()
        self._predicted_questions: set[int] = set()

    # ---------------------------------------------------------------- state

    def note_question(self, question_id: int, post_id: int) -> None:
        """Associate a question with its post, at discovery time.

        Group subquestions legitimately share one post, so this is many-to-one
        and must never be inverted into a dict keyed by post.
        """
        self._question_to_post[question_id] = post_id

    def record_for(self, question_id: int) -> PublicationRecord:
        record = self._records.get(question_id)
        if record is None:
            record = PublicationRecord(
                question_id=question_id,
                post_id=self._question_to_post.get(question_id),
            )
            self._records[question_id] = record
        return record

    @property
    def records(self) -> list[PublicationRecord]:
        return list(self._records.values())

    @property
    def orphans(self) -> list[PublicationRecord]:
        return [r for r in self._records.values() if r.is_orphaned]

    def publication_summary(self) -> dict[str, int]:
        """Counts only. Never a forecast value, never a comment body."""
        counts = {state.value: 0 for state in PublicationState}
        for record in self._records.values():
            counts[record.state.value] += 1
        return counts

    def _questions_predicted_on_post(self, post_id: int) -> list[int]:
        return sorted(
            question_id
            for question_id in self._predicted_questions
            if self._question_to_post.get(question_id) == post_id
        )

    # ------------------------------------------------------------ discovery

    def get_all_open_questions_from_tournament(
        self, tournament_id: int | str, *args: Any, **kwargs: Any
    ) -> list[Any]:
        """Upstream discovery, plus the question -> post association.

        Deliberately a pass-through for everything else: pagination is fixed
        in ``discovery.py``, which this method delegates to, so the two
        concerns stay separable.
        """
        from discovery import fetch_all_open_questions

        questions = fetch_all_open_questions(
            self, tournament_id, *args, **kwargs
        )
        for question in questions:
            if question.id_of_question is not None and question.id_of_post is not None:
                self.note_question(question.id_of_question, question.id_of_post)
        return questions

    # ----------------------------------------------------------- publication

    def _post_question_prediction(
        self, question_id: int, forecast_payload: dict
    ) -> None:
        """One prediction per question per process.

        Guards the in-run half of R3: a retry that duplicates because a
        response was lost is still a second POST from the same process, and
        the caller has no way to tell. It does NOT guard the cross-run case,
        which is what ``already_forecasted`` is for.
        """
        if question_id in self._predicted_questions:
            logger.info(
                "prediction_suppressed_duplicate question_id=%d "
                "reason=already_predicted_this_process",
                question_id,
            )
            return
        super()._post_question_prediction(question_id, forecast_payload)
        self._predicted_questions.add(question_id)
        record = self.record_for(question_id)
        record.state = PublicationState.PREDICTED
        logger.info(
            "publication_prediction_written question_id=%d post_id=%s",
            question_id, record.post_id,
        )

    def post_question_comment(
        self,
        post_id: int,
        comment_text: str,
        is_private: bool = True,
        included_forecast: bool = True,
    ) -> None:
        """One comment per post per process, retried beyond the SDK's budget.

        The duplicate suppression is not an optimisation. Without it, a group
        post with N open subquestions receives N copies of N different
        reports, and because ``forecast_questions`` gathers questions
        concurrently they can interleave. Metaculus' rule asks for "a comment
        response ... under each question"; a post carrying one comment
        satisfies it, a post carrying N near-identical ones is duplicate
        content on a single page.
        """
        if post_id in self._commented_posts:
            logger.info(
                "comment_suppressed_duplicate post_id=%d "
                "reason=already_commented_this_process",
                post_id,
            )
            self._mark_complete_for_post(post_id)
            return

        last_error: BaseException | None = None
        for attempt in range(1, self.comment_attempts + 1):
            try:
                super().post_question_comment(
                    post_id,
                    comment_text,
                    is_private=is_private,
                    included_forecast=included_forecast,
                )
            except Exception as exc:  # noqa: BLE001 - every failure gets retried
                last_error = exc
                self._note_comment_error(post_id, attempt, exc)
                if attempt < self.comment_attempts:
                    self._sleep(COMMENT_RETRY_DELAY_SECONDS)
                continue
            self._commented_posts.add(post_id)
            logger.info(
                "publication_comment_written post_id=%d attempts=%d",
                post_id, attempt,
            )
            self._mark_complete_for_post(post_id)
            return

        self._mark_orphaned_for_post(post_id)
        assert last_error is not None
        raise last_error

    def _note_comment_error(
        self, post_id: int, attempt: int, exc: BaseException
    ) -> None:
        reason = _safe_error(exc)
        for question_id in self._questions_predicted_on_post(post_id):
            record = self.record_for(question_id)
            record.comment_attempts = attempt
            record.errors.append(reason)
        logger.warning(
            "publication_comment_attempt_failed post_id=%d attempt=%d/%d reason=%s",
            post_id, attempt, self.comment_attempts, reason,
        )

    def _mark_complete_for_post(self, post_id: int) -> None:
        for question_id in self._questions_predicted_on_post(post_id):
            self.record_for(question_id).state = PublicationState.COMPLETE

    def _mark_orphaned_for_post(self, post_id: int) -> None:
        """Every question already predicted on this post is now an orphan.

        Plural on purpose: a group post's subquestions each have their own
        live prediction and all of them are missing the same comment.
        """
        orphaned = self._questions_predicted_on_post(post_id)
        for question_id in orphaned:
            self.record_for(question_id).state = PublicationState.ORPHANED
        if not orphaned:
            # The comment failed without any prediction on this post having
            # been written by us -- nothing is half-published, so this is a
            # plain failure and must not fire the operator alert.
            logger.warning(
                "publication_comment_failed post_id=%d "
                "state=no_prediction_written_by_this_process",
                post_id,
            )
            return
        # The one log line an operator must never miss. Ids and counts only:
        # the forecast itself is on Metaculus, which is where it belongs.
        # Putting it here would trade an R1 for an R2.
        logger.error(
            "%s post_id=%d question_ids=%s comment_attempts=%d "
            "state=prediction_published_without_comment "
            "action=comment must be reconciled for this post",
            ORPHAN_MARKER,
            post_id,
            ",".join(str(q) for q in orphaned),
            self.comment_attempts,
        )


def print_publication_report(client: PublishingClient) -> int:
    """End-of-run publication state, on stdout, and the orphan count.

    print() rather than logging on purpose, for the same reason
    print_run_summary_banner uses it: this must survive the redaction filter
    and any handler reconfiguration. It is safe to print because it contains
    nothing but ids, states and counts.
    """
    summary = client.publication_summary()
    orphans = client.orphans
    interesting = {k: v for k, v in summary.items() if v}
    if interesting:
        print(f"\n  Publication: {interesting}")
    if not orphans:
        return 0
    print(
        f"\n  {ORPHAN_MARKER}: {len(orphans)} forecast(s) are published on "
        "Metaculus WITHOUT their required comment."
    )
    for record in orphans:
        print(
            f"    • post_id={record.post_id} question_id={record.question_id} "
            f"comment_attempts={record.comment_attempts}"
        )
    print(
        "    Metaculus requires a comment under each forecast question for\n"
        "    eligibility. These need the comment posting before the question\n"
        "    closes; the bot will NOT retry them on its own, because\n"
        "    already_forecasted is now true for them."
    )
    return len(orphans)


def _safe_error(exc: BaseException) -> str:
    """A short, log-safe label for a failure.

    Mirrors ``backtest.fallback_llm._reason``: never the full response body.
    A Metaculus error body can echo the comment text back, and that text is
    the forecast rationale, so this must stay bounded and value-free.
    """
    from bot_helpers import scrub

    return type(exc).__name__ + ":" + scrub(exc, limit=80)
