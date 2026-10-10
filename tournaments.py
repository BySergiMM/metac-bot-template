"""The Metaculus projects this bot forecasts on, pinned in one place.

Why the ids live here and are not read from forecasting_tools
--------------------------------------------------------------
``MetaculusClient.CURRENT_AI_COMPETITION_ID`` and its siblings only move when
the SDK is upgraded. On 2026-10-01 the locked forecasting-tools 0.2.90 still
named summer-futureeval-2026 (33022) three days into the fall season, so every
poll logged ``discovery_complete tournament=33022 questions=0`` while
fall-futureeval-2026 (33121) opened and closed its first questions without
this bot. Nothing failed, because "no open questions" is a normal state.

Which tournament the bot competes in is a decision, so it is a one-line diff in
this file rather than a side effect of a dependency bump. Each id below was
read from the tournament's own page ("The project ID is 33121 (or
"fall-futureeval-2026")") and matches the constant forecasting-tools 0.3.2
ships for the same season.

Standard library only: research/ and the tests import this without the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Project:
    """One Metaculus project the bot forecasts on."""

    #: What ``forecast_on_tournament`` and the posts API accept: the numeric
    #: project id, or an alias Metaculus resolves itself.
    id: int | str
    #: Shown in the end-of-run banner. None when there is no single page.
    url: str | None
    #: The season's last day as its tournament page states it. None for an
    #: alias Metaculus rotates on its own, which therefore cannot go stale.
    ends_on: date | None
    #: Whether the tournament pays bots. The Metaculus Cup ranks them but
    #: does not pay them (FutureEval resources page).
    prize_eligible: bool


#: Fall 2026 FutureEval Bot Tournament: 28 Sep 2026 to 6 Jan 2027, $50k.
FUTUREEVAL = Project(
    id=33121,
    url="https://www.metaculus.com/tournament/fall-futureeval-2026/",
    ends_on=date(2027, 1, 6),
    prize_eligible=True,
)

#: "The minibench project ID for the currently active minibench is always
#: 'minibench'" (FutureEval resources page). An alias, so no end date.
MINIBENCH = Project(
    id="minibench",
    url=None,
    ends_on=None,
    prize_eligible=True,
)

#: Metaculus Cup Fall 2026: 28 Aug 2026 to 1 Jan 2027. Practice for bots:
#: scored against the human crowd, not prize-eligible.
METACULUS_CUP = Project(
    id=33108,
    url="https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    ends_on=date(2027, 1, 1),
    prize_eligible=False,
)

#: Market Pulse Challenge 26Q4: $7,500, ends 31 Dec 2026. Its tournament
#: page: "You are welcome to enter this tournament and compete for prizes
#: using a forecasting bot account, but you can only enter the competition
#: once" -- so the owner's human account must not forecast here. Spot scored
#: at each question's close, which is why market_pulse.py refreshes forecasts
#: shortly before close instead of forecasting once.
MARKET_PULSE = Project(
    id="market-pulse-26q4",
    url="https://www.metaculus.com/tournament/market-pulse-26q4/",
    ends_on=date(2026, 12, 31),
    prize_eligible=True,
)

ALL = (FUTUREEVAL, MINIBENCH, METACULUS_CUP, MARKET_PULSE)


def ended(projects: tuple[Project, ...], today: date) -> list[Project]:
    """The projects whose season is over on ``today``.

    A project with no end date never appears: it is an alias Metaculus keeps
    current. The end date itself still counts as inside the season.
    """
    return [p for p in projects if p.ends_on is not None and today > p.ends_on]


def season_warnings(today: date) -> list[str]:
    """One GitHub Actions ``::warning::`` line per pinned season that is over.

    Printed at startup so a stale id shows up as an annotation on the run,
    instead of as a run that quietly finds nothing, which is what happened
    with 33022.
    """
    return [
        f"::warning::{p.url or p.id} ended on {p.ends_on.isoformat()}; "
        "pin the next season's project id in tournaments.py"
        for p in ended(ALL, today)
        if p.ends_on is not None
    ]
