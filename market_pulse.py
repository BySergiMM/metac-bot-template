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
#: The forecast that is scored (owner, 2026-10-23: Market Pulse is one of the
#: tournaments the budget goes to). Only the forecast standing at close counts,
#: so since 2026-11-16 it is the one Opus forecast a question gets
#: (main.py, via split()): a question closing within FINAL_WINDOW that has no
#: forecast made inside that window is forecast once more, and only once.
#: Dagu dispatches every 30 minutes, so it lands about 3.5 hours before close.
#: Everything earlier -- the first forecast, the 36-hour refresh -- is a cheap
#: safety net that stands only if the final one never happens.
FINAL_WINDOW = timedelta(hours=4)

#: Forecasts made before this moment were produced without the group option
#: in the prompt (main._question_text), so a group subquestion was forecast
#: as the generic group question -- e.g. Meta's EPS forecast as some other
#: company's. Each is redone once, whatever its age or close.
REFORECAST_BEFORE = datetime(2026, 10, 21, 8, 0, tzinfo=timezone.utc)


def _predates_label_fix(question: Any, last: datetime | None) -> bool:
    return bool(getattr(question, "group_question_option", None)) and last is not None \
        and last < REFORECAST_BEFORE


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
        if _predates_label_fix(question, last):
            refresh.append(question)
            continue
        if close is None or last is None:
            continue
        left, age = close - now, now - last
        final_due = left <= FINAL_WINDOW and last < close - FINAL_WINDOW
        if (left <= REFRESH_WINDOW and age >= MIN_AGE) or final_due:
            refresh.append(question)
    logger.info(
        "market_pulse_selection open=%d new=%d refresh=%d",
        len(questions), len(new), len(refresh),
    )
    return new + refresh


def is_final(question: Any, now: datetime) -> bool:
    """Whether forecasting ``question`` now makes the forecast that stands at
    its close: it closes within FINAL_WINDOW."""
    close = _utc(question.close_time)
    return close is not None and close - _utc(now) <= FINAL_WINDOW


def split(questions: Sequence[Any], now: datetime) -> tuple[list[Any], list[Any]]:
    """select(), split into the forecasts that will be scored (closing
    within FINAL_WINDOW: Opus) and the safety net before them (Haiku)."""
    chosen = select(questions, now)
    final = [q for q in chosen if is_final(q, now)]
    early = [q for q in chosen if not is_final(q, now)]
    logger.info("market_pulse_split final=%d early=%d", len(final), len(early))
    return final, early


#: The Metaculus Cup is scored against the human crowd over each question's
#: life, so a forecast left untouched for weeks loses ground as news arrives
#: (Metaculus flagged 9 of the bot's Cup forecasts "might be stale" on
#: 2026-10-16). A week is the refresh age: the Cup opens a batch weekly and
#: pays bots nothing, so more often is spend without a prize behind it.
CUP_MAX_AGE = timedelta(days=7)
#: The refresh age while spend.boost_active(): the credit left before it
#: renews buys fresher Cup forecasts (owner, 2026-10-24).
CUP_BOOST_MAX_AGE = timedelta(days=2)


def select_stale(
    questions: Sequence[Any], now: datetime, max_age: timedelta = CUP_MAX_AGE
) -> list[Any]:
    """New questions, plus those whose latest forecast is ``max_age`` old.

    Unlike :func:`select` this ignores close time: every open question is
    refreshed on the same cadence. Unknown forecast times are left alone.
    """
    now = _utc(now)
    new: list[Any] = []
    stale: list[Any] = []
    for question in questions:
        if not question.already_forecasted:
            new.append(question)
            continue
        last = _last_forecast(question)
        if _predates_label_fix(question, last) or (last is not None and now - last >= max_age):
            stale.append(question)
    logger.info(
        "stale_selection open=%d new=%d stale=%d max_age_days=%.1f",
        len(questions), len(new), len(stale), max_age.total_seconds() / 86400,
    )
    return new + stale
