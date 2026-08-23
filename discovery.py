"""Question discovery that reads every page, and says so.

Why this exists
---------------
``MetaculusClient.get_all_open_questions_from_tournament`` (forecasting_tools
0.2.90, ``metaculus_client.py:425``) calls
``get_questions_matching_filter(api_filter)`` with ``num_questions=None``.
That routes into ``_filter_sequential_strategy``, whose first branch is::

    if num_questions is None:
        questions, _ = self._grab_filtered_questions_with_offset(api_filter, 0)
        return questions

One request. ``offset=0``, ``limit=100``. The library's own docstring is
explicit about it -- "If num questions is not set, it will only grab the first
page of questions from API" -- so this is documented upstream behaviour, not a
bug. It is still wrong for a tournament runner: the 101st open post is
invisible, no warning is emitted, and the run reports success.

The cap is on POSTS, not questions. A group post unpacks into N subquestions
*after* the page is fetched (``_unpack_group_question``), so 100 posts can
legitimately yield far more than 100 questions -- and conversely, 101 open
posts silently lose everything past the first hundred however few questions
they carry.

What this does
--------------
Walks ``offset`` until the server stops returning rows, then reports what it
saw. Every number an operator needs to notice truncation is logged: pages
read, posts seen before local filtering, questions after unpacking, and
whether the walk stopped because the data ran out or because it hit the
safety limit.

The safety limit is explicit and observable, which is the whole point. An
unbounded ``while True`` against a paging API is how a scheduled job turns
into an incident; a silent ``break`` is how coverage loss hides. So there is a
ceiling, and reaching it is an ERROR, not a shrug.

Deliberately built on the client's own primitives
-------------------------------------------------
``_grab_filtered_questions_with_offset`` already applies
``_apply_local_filters`` and already unpacks group questions, and its second
return value is precisely "did the server have rows here, before local
filtering" -- which is the correct loop condition. Reimplementing the request
would mean reimplementing the filter semantics, and the two would drift.
"""

from __future__ import annotations

import logging
from typing import Any

from forecasting_tools.helpers.metaculus_client import ApiFilter, MetaculusClient
from forecasting_tools.data_models.questions import MetaculusQuestion

logger = logging.getLogger(__name__)

#: Pages, not posts. At the API's 100-per-page ceiling this is 20,000 posts in
#: one tournament, which is far past anything Metaculus has ever run and far
#: short of a runaway loop. Reaching it is reported as an error rather than
#: treated as the end of the data.
MAX_PAGES = 200


def fetch_all_open_questions(
    client: MetaculusClient,
    tournament_id: int | str,
    group_question_mode: str = "unpack_subquestions",
    max_pages: int = MAX_PAGES,
) -> list[MetaculusQuestion]:
    """Every open question in a tournament, across all pages.

    Signature-compatible with
    ``MetaculusClient.get_all_open_questions_from_tournament`` so it can stand
    in for it directly.
    """
    logger.info("Retrieving questions from tournament %s", tournament_id)
    api_filter = ApiFilter(
        allowed_tournaments=[tournament_id],
        allowed_statuses=["open"],
        group_question_mode=group_question_mode,  # type: ignore[arg-type]
    )

    page_size = client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
    questions: list[MetaculusQuestion] = []
    seen_question_ids: set[int] = set()
    duplicates = 0
    pages_read = 0
    hit_page_limit = False

    for page_index in range(max_pages):
        offset = page_index * page_size
        page_questions, server_had_rows = client._grab_filtered_questions_with_offset(
            api_filter, offset
        )
        pages_read += 1

        for question in page_questions:
            # Dedup on the QUESTION id, not the post id: group subquestions
            # share a post and are legitimately distinct questions. A repeat
            # here means the server paged us the same row twice, which
            # ``order_by`` instability can genuinely do while questions are
            # opening and closing underneath the walk.
            question_id = question.id_of_question
            if question_id is not None and question_id in seen_question_ids:
                duplicates += 1
                continue
            if question_id is not None:
                seen_question_ids.add(question_id)
            questions.append(question)

        if not server_had_rows:
            # `server_had_rows` is "the server returned posts at this offset",
            # measured BEFORE local filtering. A page that is entirely removed
            # by local filters is therefore not mistaken for the end of the
            # data -- which is exactly the bug an `if not page_questions`
            # condition would introduce.
            break
    else:
        hit_page_limit = True

    if duplicates:
        logger.warning(
            "discovery_duplicate_questions tournament=%s duplicates=%d "
            "reason=server_returned_a_question_on_more_than_one_page",
            tournament_id, duplicates,
        )
    if hit_page_limit:
        logger.error(
            "discovery_page_limit_reached tournament=%s pages=%d page_size=%d "
            "state=discovery_may_be_incomplete",
            tournament_id, max_pages, page_size,
        )

    logger.info(
        "discovery_complete tournament=%s pages=%d questions=%d "
        "duplicates_dropped=%d truncated=%s",
        tournament_id, pages_read, len(questions), duplicates, hit_page_limit,
    )
    logger.info(
        "Retrieved %d questions from tournament %s", len(questions), tournament_id
    )
    return questions


def install_complete_pagination(client: MetaculusClient) -> MetaculusClient:
    """Bind full pagination onto an existing client instance.

    Provided for callers holding a plain ``MetaculusClient``; the bot uses
    ``publication.PublishingClient``, which overrides the method directly.
    """

    def _paged(
        tournament_id: int | str, *args: Any, **kwargs: Any
    ) -> list[MetaculusQuestion]:
        return fetch_all_open_questions(client, tournament_id, *args, **kwargs)

    client.get_all_open_questions_from_tournament = _paged  # type: ignore[method-assign]
    return client
