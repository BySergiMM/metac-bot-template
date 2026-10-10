"""Guards on what a forecast may say, and the dates every prompt must carry.

Why these exist
---------------
On 141 resolved questions (2026-10-07) the bot's mean peer score was -0.81,
and most of the deficit came from a handful of confident misses near -50 each:
an option the bot all but excluded won a multiple-choice question, and two
event questions went the way the bot had called at the extremes. Peer score is
100 * ln(p / G) against the other forecasters' geometric mean G, so the loss
from a probability near zero on what happens is unbounded in practice, while
the gain from being extreme and right is small. Capping is the one technique
Metaculus' Fall 2025 survey found separating prize winners from the rest
(r=+0.48); extremizing went the other way in Spring 2026.

The prompts also never said when a question closes or resolves, only today's
date, so "time left" -- the first thing each prompt asks the model to state --
had to be guessed from the question text.

Standard library only, so research/ and the tests can import it without the
SDK.
"""

from __future__ import annotations

from datetime import datetime, timezone

#: Binary forecasts stay inside [BINARY_MIN, 1 - BINARY_MIN]. At 3%, a miss
#: against a field at 50% costs 100*ln(0.03/0.5) = -281 instead of -391 at 1%;
#: a correct call gives up 100*ln(0.97/0.99) = -2. The SDK's own clamp is 1%.
BINARY_MIN = 0.03

#: Every multiple-choice option keeps at least this much. Giving 1% to an
#: option the field puts at 15% costs -271 when it wins; 2.5% costs -179.
OPTION_FLOOR = 0.025


def clip_binary(probability: float) -> float:
    return max(BINARY_MIN, min(1 - BINARY_MIN, probability))


def floor_options(probabilities: list[float], floor: float = OPTION_FLOOR) -> list[float]:
    """Mix the forecast with a floor on every option, keeping the sum at 1.

    ``p' = f + (1 - n*f) * p / sum(p)`` rather than clamp-then-renormalise:
    dividing after clamping can push an option back under the floor. The
    mixture keeps the ranking and sums to exactly 1, which matters because
    the SDK's PredictedOptionList validator renormalises whatever it is given
    and rejects the result if that moved any option by more than 0.05; an
    input that already sums to 1 inside [0.01, 0.99] passes it unchanged.
    The floor shrinks for questions with so many options that it would
    otherwise eat half the mass.
    """
    n = len(probabilities)
    if n == 0:
        return []
    total = sum(probabilities)
    if total <= 0:
        return [1.0 / n] * n
    floor = min(floor, 0.5 / n)
    scale = 1 - n * floor
    out = [floor + scale * p / total for p in probabilities]
    out[-1] += 1.0 - sum(out)
    return out


def _date(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def timing_context(
    now: datetime,
    close_time: datetime | None,
    resolution_time: datetime | None,
) -> str:
    """The dates a forecaster needs, stated rather than left to be inferred.

    Ends with the stale-knowledge warning: the forecasting models were trained
    well before today, and the research step may be answering from the same
    memory, so an event "has not happened yet" in the model's head says
    nothing about whether it happened last week.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    lines = [f"Today is {_date(now)}."]
    for label, moment in (
        ("Forecasting on this question closes", close_time),
        ("The question is scheduled to resolve", resolution_time),
    ):
        stated = _date(moment)
        if stated is None:
            continue
        aware = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        days = (aware - now).total_seconds() / 86400
        lines.append(f"{label} on {stated} ({days:.1f} days from now).")
    lines.append(
        "Your training data ends well before today. Treat any event in the "
        "question window as unknown unless the research gives a dated source "
        "for it, and never assume the question has already resolved."
    )
    return "\n".join(lines)
