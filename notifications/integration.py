"""The one thing `main.py` imports. Ties detection + state + dashboard +
notifier together, and guarantees none of it can affect the bot's own
result.

WHY THIS FILE EXISTS SEPARATELY FROM main.py
-----------------------------------------------
"No quiero una clase gigante metida en main.py": `main.py` gets exactly one
call, `notifications.integration.handle_run(...)`, wrapped in its own
try/except at the call site AND again in here -- belt and braces, because a
notification system that can turn a successful forecasting run into a failed
one would be worse than not having it at all.

WHAT THIS DOES NOT DO
----------------------
It never touches Metaculus (all its inputs are things `main.py` already
computed this run). It never opens `outcomes.jsonl` or anything under
`research/` -- unrelated subsystem, not imported. It never imports
`forecasting_tools` directly; it reaches into `ForecastReport`/
`PublicationRecord`-shaped objects defensively (`getattr(..., None)`) because
this repository cannot install the real package to verify their exact shape.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from notifications.dashboard import build_dashboard_text
from notifications.events import ForecastOutcome, compute_run_state, detect_events, select_headline
from notifications.state import DEFAULT_STATE_PATH, load_state, save_state

NOTIFY_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "scripts", "notify.py")


@dataclass
class RunOutcome:
    """What a single `handle_run` call decided and did. Returned so a caller
    (or a test) can observe the outcome without depending on side effects."""

    events_detected: int
    message_sent: bool
    message: str | None
    error: str | None = None


#: publication.PublicationState's own values (it is a `str, Enum`). Compared
#: as plain strings deliberately -- importing PublicationState itself would
#: import publication.py, which imports forecasting_tools at module scope,
#: exactly the dependency this package must not gain.
_CONFIRMED_PUBLISHED_STATES = frozenset({"predicted", "complete", "orphaned"})


def _confirm_published(client: Any, question_id: int | None) -> bool | None:
    """Ground truth for "was a prediction actually posted", straight from
    `publication.PublishingClient`'s own bookkeeping -- `_post_question_
    prediction` only advances a record past PENDING after the real POST
    call returns without raising (see publication.py). This is a stronger
    signal than "forecasting_tools handed back a ForecastReport instead of
    an exception": main.py's own docstring says `forecast_on_tournament`'s
    per-question pipeline includes the submit step before returning a
    report, which is why the weaker signal was used originally, but this
    repository's OWN publication-tracking code is the more direct answer
    and was sitting unused.

    Returns `None` (not True/False) if `client` does not expose
    `record_for` at all, so `_extract_outcome` can fall back to the older,
    weaker signal rather than wrongly reporting "not published" for a
    caller that never claimed to track this."""
    if question_id is None:
        return False
    record_for = getattr(client, "record_for", None)
    if record_for is None:
        return None
    try:
        record = record_for(question_id)
        state = getattr(record, "state", None)
        state_value = getattr(state, "value", state)
        return str(state_value) in _CONFIRMED_PUBLISHED_STATES
    except Exception:  # noqa: BLE001 - a broken lookup must not crash the bot
        return None


def _extract_outcome(report_or_exception: Any, now_iso: str, client: Any) -> ForecastOutcome:
    """Defensive: `getattr(..., None)` throughout, because `ForecastReport`
    comes from `forecasting_tools`, which this repository cannot install to
    verify against. A report whose shape does not match what
    `bot_helpers.print_run_summary_banner` already relies on degrades to
    `question_id=None` rather than raising.

    `published` is `True` only when confirmed via `_confirm_published` (or,
    when the client cannot confirm at all, the older weaker signal: this
    report is not an exception). `new_question` still fires whenever a real
    `question_id` is known, whether or not publication is confirmed --
    "we saw a new question" and "we confirmed a forecast is live for it"
    are different claims, and only the second one requires the stronger
    check."""
    if isinstance(report_or_exception, BaseException):
        return ForecastOutcome(question_id=None, title=None, published=False)

    question = getattr(report_or_exception, "question", None)
    question_id = getattr(question, "id_of_question", None)
    title = getattr(question, "question_text", None)
    errors = getattr(report_or_exception, "errors", None)

    confirmed = _confirm_published(client, question_id)
    if confirmed is None:
        confirmed = True  # client does not support the stronger check

    return ForecastOutcome(
        question_id=question_id,
        title=title,
        published=confirmed,
        published_at=now_iso,
        had_minor_errors=bool(errors),
    )


def _extract_orphan_ids(client: Any) -> list[int]:
    orphans = getattr(client, "orphans", None)
    if not orphans:
        return []
    ids = []
    for record in orphans:
        qid = getattr(record, "question_id", None)
        if qid is not None:
            ids.append(qid)
    return ids


def _send_whatsapp(message: str) -> bool:
    """Shells out to `scripts/notify.py whatsapp` -- reuses that mechanism
    exactly rather than re-implementing the CallMeBot call here. Never
    raises, never prints stdout/stderr (defence in depth: notify.py already
    never prints the API key, but this file does not echo its output
    either, on principle)."""
    try:
        result = subprocess.run(
            [sys.executable, NOTIFY_SCRIPT, "whatsapp", "--message", message],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 - a notifier failure must never propagate
        return False


def handle_run(
    *,
    forecast_reports: Sequence[Any],
    client: Any,
    tournament_label: str,
    state_path: str = DEFAULT_STATE_PATH,
    run_id: str | None = None,
    send: Any = _send_whatsapp,
    now: datetime | None = None,
) -> RunOutcome:
    """The single entry point `main.py` calls. Every parameter beyond the
    first three has a production default; `send`/`now` are overridable so
    tests never touch a real clock or a real network call."""
    try:
        # Explicit UTC, not naive datetime.now(): a naive timestamp happens
        # to equal UTC on a GitHub Actions runner, but making it explicit
        # means this stays correct wherever it runs, and dashboard.py can
        # label it unambiguously rather than relying on that coincidence.
        now = now or datetime.now(timezone.utc)
        run_id = run_id or os.environ.get("GITHUB_RUN_ID", "local")
        now_iso = now.isoformat()

        outcomes = [_extract_outcome(item, now_iso, client) for item in forecast_reports]
        orphan_ids = _extract_orphan_ids(client)

        state = load_state(state_path)
        events = detect_events(
            outcomes=outcomes, orphan_question_ids=orphan_ids, run_id=run_id, state=state,
        )
        if not events:
            return RunOutcome(events_detected=0, message_sent=False, message=None)

        headline = select_headline(events)
        run_state = compute_run_state(outcomes)
        new_question_count = sum(1 for e in events if e.kind == "new_question")
        forecasts_published_count = sum(1 for e in events if e.kind == "forecast_published")
        message = build_dashboard_text(
            headline=headline,
            run_state=run_state,
            tournament_label=tournament_label,
            new_question_count=new_question_count,
            forecasts_published_count=forecasts_published_count,
            now=now,
        )

        sent = send(message)
        if sent:
            # Only mark as notified on confirmed send -- a failed send
            # leaves these events eligible again next run, which is the
            # retry semantics this system wants (see
            # notifications/state.py's module docstring).
            state.notified_event_ids = sorted(
                set(state.notified_event_ids) | {e.event_id for e in events}
            )
            try:
                save_state(state, state_path)
            except Exception as exc:  # noqa: BLE001 - persistence failing must not fail the bot
                return RunOutcome(
                    events_detected=len(events), message_sent=True, message=message,
                    error="state not saved: {0}".format(type(exc).__name__),
                )
        return RunOutcome(events_detected=len(events), message_sent=sent, message=message)

    except Exception as exc:  # noqa: BLE001 - the whole point of this file
        return RunOutcome(
            events_detected=0, message_sent=False, message=None,
            error="{0}: notifications disabled for this run".format(type(exc).__name__),
        )
