"""Dashboard builder: one short WhatsApp-ready snapshot, plus the event that
triggered it.

FIELDS, AND WHERE EACH ONE ACTUALLY COMES FROM
-------------------------------------------------
    🏆 Torneo      main.py's own TOURNAMENT_URLS/run_mode label -- already
                   hardcoded there, not fetched.
    ❓ Pendientes   COUNT of `new_question` events THIS RUN (questions the bot
                   just started tracking). Not the tournament's total open
                   question count -- that needs a second discovery call this
                   feature does not make. Named "Pendientes" rather than
                   "Nuevas" to match the requested mockup, but it means the
                   same thing: new to us, as of this run.
    🔮 Forecasts    COUNT of `forecast_published` events THIS RUN. NOT a
                   lifetime total: no reliable local counter exists (an
                   earlier draft invented one and it was removed on audit --
                   see notifications/state.py), and the real lifetime number
                   lives only in Metaculus' own track-record API, which is a
                   separate, token-gated call this feature does not make.
    🟢 Estado       OK / PARTIAL / ERROR / IDLE, from
                   `events.compute_run_state`.
    🕐 timestamp    real wall-clock time of THIS run, always UTC and always
                   labelled as such -- `datetime.now()` (naive, system-local)
                   happens to equal UTC on a GitHub Actions runner, but a
                   string reading "26/08 17:42" with no timezone is
                   ambiguous to a human reading it on their own phone in
                   their own timezone. `integration.py` passes an explicit
                   `datetime.now(timezone.utc)`, not the naive form, so this
                   is correct even if the code is ever run somewhere whose
                   system clock is not UTC. Unlike the replay harness's
                   production_llm variant, this is a live run, not a
                   simulated past cutoff -- there is no lookahead concern
                   here at all.

Deliberately absent, because no reliable source exists at this hook point:
tournament-wide pending question count, any Metaculus-side track record
number, and (per explicit instruction after auditing
`bot_helpers.RedactForecastContent`) the published probability itself.
"""

from __future__ import annotations

from datetime import datetime

from notifications.events import Event

MAX_MESSAGE_LENGTH = 500  # WhatsApp allows far more; this keeps it a snapshot, not a report.


def build_dashboard_text(
    *,
    headline: Event,
    run_state: str,
    tournament_label: str,
    new_question_count: int,
    forecasts_published_count: int,
    now: datetime,
) -> str:
    state_emoji = {"OK": "🟢", "PARTIAL": "🟡", "ERROR": "🔴", "IDLE": "⚪"}.get(run_state, "⚪")

    lines = [
        "🤖 *METACULUS BOT*",
        "━━━━━━━━━━━━",
        headline.headline,
        "",
        "🏆 Torneo: {0}".format(tournament_label),
    ]
    if new_question_count:
        lines.append("❓ Pendientes: {0}".format(new_question_count))
    if forecasts_published_count:
        lines.append("🔮 Forecasts: {0}".format(forecasts_published_count))
    lines.append("{0} Estado: {1}".format(state_emoji, run_state))
    lines.append("🕐 {0} UTC".format(now.strftime("%d/%m %H:%M")))

    text = "\n".join(lines)
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[: MAX_MESSAGE_LENGTH - 1].rstrip() + "…"
    return text
