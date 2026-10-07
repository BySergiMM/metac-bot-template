"""Which Market Pulse questions to forecast on this run.

Market Pulse is spot scored: "only your predictions standing at the close
time of each question matter" (tournament page). Forecasting each question
once, when first seen -- what every other tournament here does -- would leave
a forecast that can be weeks old standing at the one instant that is scored.
Re-forecasting every open question on every run would fix that and multiply
LLM spend by the ~6 runs a day GitHub actually delivers.

So a question is forecast when either:

* we have never forecast it, or
* it closes within ``REFRESH_WINDOW`` and our latest forecast is older than
  ``MIN_AGE`` -- which, at the observed run cadence, refreshes each question
  about once or twice in its last day and a half.

Standard library only, so the selection is testable without the SDK; the
questions only need ``already_forecasted``, ``close_time`` and
``timestamp_of_my_last_forecast``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

logger = logging.getLogger(__name__)

REFRESH_WINDOW = timedelta(hours=36)
MIN_AGE = timedelta(hours=18)


def _utc(moment: datetime | None) -> datetime | None:
    """The SDK parses API dates with pendulum (aware, UTC); a naive value is
    taken as UTC rather than letting the subtraction below raise."""
    if moment is not None and moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _last_forecast(question: Any) -> datetime | None:
    try:
        return _utc(question.timestamp_of_my_last_forecast)
    except Exception:  # noqa: BLE001 - an unreadable timestamp means "unknown"
        return None


def select(questions: Sequence[Any], now: datetime) -> list[Any]:
    """The questions worth an LLM call now, new ones first."""
    now = _utc(now)
    new: list[Any] = []
    refresh: list[Any] = []
    for question in questions:
        if not question.already_forecasted:
            new.append(question)
            continue
        close = _utc(question.close_time)
        last = _last_forecast(question)
        if close is None or last is None:
            continue
        if close - now <= REFRESH_WINDOW and now - last >= MIN_AGE:
            refresh.append(question)
    logger.info(
        "market_pulse_selection open=%d new=%d refresh=%d",
        len(questions), len(new), len(refresh),
    )
    return new + refresh
