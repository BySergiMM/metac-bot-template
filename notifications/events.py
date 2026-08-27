"""Event detection: what changed since the last recorded state.

WHAT COUNTS AS AN EVENT, AND WHY
----------------------------------
Every event kind here maps to a signal `main.py` (via bot_helpers.py's own
existing summary code, `print_run_summary_banner`) or `publication.py`
already computes reliably in production, TODAY, without any new Metaculus
call:

  new_question         a question_id in this run's outcomes. Reliable ONLY
                        when the caller comes from a run with
                        skip_previously_forecasted_questions=True (`main.py`
                        default "tournament" mode) -- see
                        notifications/state.py's docstring for why that flag
                        is what makes "appears in outcomes" mean "new".
  forecast_published    a successful ForecastReport with no minor errors
                        (bot_helpers.py's own "valid, no note" case).
                        Contains question_id and a timestamp. Deliberately
                        does NOT contain the published probability -- see
                        "WHY NO PROBABILITY" below.
  publication_orphan    a question in `PublishingClient.orphans` -- a
                        forecast published with no required comment
                        (publication.py, already tracked in production)
  partial_run           this run attempted more than one question and SOME
                        but not all failed
  run_error             this run attempted at least one question and ALL of
                        them failed

Deliberately NOT implemented, because it cannot be done reliably from what is
in scope here:

  - per-question identity of a forecasting FAILURE. `return_exceptions=True`
    hands back bare exceptions; `print_run_summary_banner` -- the existing,
    shipped code this repository already trusts -- does not attempt to
    recover a question id from one either, which is the strongest evidence
    available that doing so is not reliable. `partial_run`/`run_error`
    report a COUNT, never a fabricated identity.
  - total open questions in the tournament universe, or any notion of
    "tournament state" beyond this bot's own run. Nothing at this hook point
    calls discovery a second time to learn it, and adding that call was
    explicitly out of scope for this feature.
  - a workflow that crashes BEFORE this code runs (a step earlier than "Run
    bot" failing, or an unhandled exception before `forecast_reports`
    exists). That case produces no in-process event at all -- catching it
    needs a separate `if: failure()` workflow step, which needs no persisted
    state (see the report delivered alongside this module).

WHY NO PROBABILITY
-------------------
`bot_helpers.RedactForecastContent` already keeps forecast VALUES out of this
repository's logs, for a documented reason that is not a style preference:
Metaculus requires private comments, and FutureEval forbids a bot maker from
previewing their own bot's forecast on an OPEN question. Sending the exact
probability to WhatsApp for a still-open question is exactly that preview,
through a channel the existing redaction was never built to cover. Confirmed
with the user before writing this: the dashboard reports THAT a forecast was
published, never the number.

INPUT SHAPE
-----------
This module takes already-extracted primitives (`ForecastOutcome`), not SDK
objects -- so it has zero dependency on `forecasting_tools` and is fully
testable with plain data. `notifications/integration.py` is the one place
that reaches into real `ForecastReport`/`PublicationRecord` objects, and it
does so defensively (`getattr(..., None)`), because this repository cannot
install `forecasting_tools` to verify the exact attribute set some of those
objects expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from notifications.state import NotificationState

P0 = "P0"
P1 = "P1"

# Fixed so a batch of mixed events always leads with the most important one.
_HEADLINE_PRIORITY = (
    "new_question",
    "forecast_published",
    "publication_orphan",
    "run_error",
    "partial_run",
)


@dataclass(frozen=True)
class ForecastOutcome:
    """One question this run touched. `published=False` means the attempt
    raised (an entry in the exceptions list, not a ForecastReport) --
    `question_id`/`title` are `None` for those in the overwhelming majority
    of cases, per the module docstring."""

    question_id: int | None
    title: str | None
    published: bool
    published_at: str | None = None
    had_minor_errors: bool = False


@dataclass(frozen=True)
class Event:
    event_id: str
    priority: str
    kind: str
    headline: str
    detail: dict[str, Any] = field(default_factory=dict)


def _truncate(text: str | None, limit: int = 60) -> str:
    if not text:
        return "(sin título)"
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def compute_run_state(outcomes: list[ForecastOutcome]) -> str:
    """OK / PARTIAL / ERROR / IDLE -- IDLE when nothing was attempted at
    all, distinct from OK (something was attempted and none of it failed)."""
    if not outcomes:
        return "IDLE"
    failed = sum(1 for o in outcomes if not o.published)
    if failed == 0:
        return "OK"
    if failed == len(outcomes):
        return "ERROR"
    return "PARTIAL"


def detect_events(
    *,
    outcomes: list[ForecastOutcome],
    orphan_question_ids: list[int],
    run_id: str,
    state: NotificationState,
) -> list[Event]:
    """Pure function: given this run's outcomes and the state carried over
    from the last one, return every NEW event. Does not mutate `state` --
    the caller decides when (and whether) to persist that the returned
    events were actually sent.

    CALLER CONTRACT: `outcomes` must come from a run with
    `skip_previously_forecasted_questions=True` (`main.py`'s default
    "tournament" mode) -- see `notifications/state.py`'s module docstring.
    This function does not and cannot check the flag itself, since it never
    sees the bot instance; `notifications/integration.py` is responsible for
    only calling this in that mode.
    """
    events: list[Event] = []
    notified_ids = state.notified_event_id_set()

    def _emit(event_id: str, priority: str, kind: str, headline: str, **detail: Any) -> None:
        if event_id in notified_ids:
            return
        events.append(Event(event_id=event_id, priority=priority, kind=kind,
                             headline=headline, detail=detail))

    for outcome in outcomes:
        if outcome.question_id is None:
            continue
        # No local "have we seen this id" check: in scope (tournament mode,
        # skip_previously_forecasted_questions=True) EVERY id here is new to
        # Metaculus by construction. The only memory that matters is whether
        # WE already sent a WhatsApp about it -- `_emit`'s dedup, above.
        _emit(
            "new_question:{0}".format(outcome.question_id), P0, "new_question",
            "🆕 Q{0} — {1}".format(outcome.question_id, _truncate(outcome.title)),
            question_id=outcome.question_id, title=outcome.title,
        )

    for outcome in outcomes:
        if outcome.question_id is None or not outcome.published:
            continue
        # Deliberately no probability in the event detail -- see "WHY NO
        # PROBABILITY" in the module docstring.
        _emit(
            "forecast_published:{0}".format(outcome.question_id), P0, "forecast_published",
            "🔮 Q{0} — {1}".format(outcome.question_id, _truncate(outcome.title)),
            question_id=outcome.question_id, title=outcome.title,
            published_at=outcome.published_at, had_minor_errors=outcome.had_minor_errors,
        )

    for question_id in orphan_question_ids:
        _emit(
            "publication_orphan:{0}".format(question_id), P0, "publication_orphan",
            "🚫 Q{0} publicado sin comentario obligatorio".format(question_id),
            question_id=question_id,
        )

    run_state = compute_run_state(outcomes)
    failed_count = sum(1 for o in outcomes if not o.published)
    if run_state == "ERROR":
        _emit(
            "run_error:{0}".format(run_id), P0, "run_error",
            "🚨 {0} de {1} intento(s) de forecast fallaron".format(failed_count, len(outcomes)),
            count=failed_count, total=len(outcomes),
        )
    elif run_state == "PARTIAL":
        _emit(
            "partial_run:{0}".format(run_id), P1, "partial_run",
            "⚠️ {0} de {1} intento(s) de forecast fallaron".format(failed_count, len(outcomes)),
            count=failed_count, total=len(outcomes),
        )

    return events


def select_headline(events: list[Event]) -> Event | None:
    """The single event a batch's WhatsApp message leads with. Order is
    fixed (`_HEADLINE_PRIORITY`), not "most recent" or "highest count" --
    a fixed rule is auditable, a heuristic on counts is one more thing to
    get subtly wrong."""
    if not events:
        return None
    by_kind: dict[str, list[Event]] = {}
    for event in events:
        by_kind.setdefault(event.kind, []).append(event)
    for kind in _HEADLINE_PRIORITY:
        if kind in by_kind:
            group = by_kind[kind]
            if len(group) == 1:
                return group[0]
            return Event(
                event_id="batch:{0}:{1}".format(kind, len(group)),
                priority=group[0].priority,
                kind=kind,
                headline=_batch_headline(kind, len(group)),
                detail={"count": len(group)},
            )
    return events[0]


def _batch_headline(kind: str, count: int) -> str:
    labels = {
        "new_question": "🆕 {0} preguntas nuevas".format(count),
        "forecast_published": "🔮 {0} forecasts publicados".format(count),
        "publication_orphan": "🚫 {0} forecasts sin comentario".format(count),
    }
    return labels.get(kind, "{0} evento(s) de tipo {1}".format(count, kind))
