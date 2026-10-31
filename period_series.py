"""The resolution source's own history for Market Pulse's biweekly series.

Why this exists
---------------
On 2026-10-31 eight of Market Pulse's twelve open posts were one quantity
over biweekly periods of Q4 2026, 48 sub-questions in all:

* the level of a FRED series on the period's last day: the 10-year Treasury
  yield (DGS10) and the ICE BofA US High Yield OAS (BAMLH0A0HYM2);
* the difference of two percent price returns over the period, on Yahoo
  closes (P0 the close before the period starts, P1 its last close): crude,
  gold and Nasdaq-100 futures against S&P 500 futures, Nvidia against Apple
  and against Microsoft;
* the VIX's highest intraday value over the period (Cboe's daily HIGH).

The 2026-10-30 comparison with futuresearch found our numeric ranges too
wide, set by feel; futuresearch starts from the resolution source and sizes
ranges from how much the quantity moves. This reads the same sources the
questions resolve on and states, for the period asked: the latest value, what
the period has done so far when it has started, and the spread of the move
still to come from the last three months (one year for the VIX maximum).

Normal approximations, weekdays as sessions (holidays ignored): context for
the forecasters, never a forecast. Never raises; "" when the question is not
one of these series or a source fails.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

Fetch = Callable[[str, dict], bytes]

FRED = {
    "DGS10": ("10-year constant maturity", "ust 10y", "10-year treasury"),
    "BAMLH0A0HYM2": ("high yield oas", "high yield option-adjusted spread", "bamlh0a0hym2"),
}
#: Names in Market Pulse's return-difference questions, to Yahoo symbols.
YAHOO = (
    ("crude oil futures", "CL=F"), ("(cl)", "CL=F"),
    ("gold futures", "GC=F"), ("(gc)", "GC=F"),
    ("nasdaq-100 futures", "NQ=F"), ("(nq)", "NQ=F"),
    ("s&p 500 futures", "ES=F"), ("(es)", "ES=F"),
    ("nvidia", "NVDA"), ("apple", "AAPL"), ("microsoft", "MSFT"),
)
MONTHS = {m: i + 1 for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}
PERIOD = re.compile(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\s*[-–]\s*([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\b")
LOOKBACK_DAYS = 92


def _text(question: Any, *fields: str) -> str:
    return "\n".join(str(getattr(question, f, "") or "") for f in fields).lower()


def period(question: Any) -> tuple[date, date] | None:
    """The biweekly period a sub-question asks about, from its group option
    ("Oct 26 - Nov 6"); the year from its scheduled resolution."""
    found = PERIOD.search(str(getattr(question, "group_question_option", "") or ""))
    when = getattr(question, "scheduled_resolution_time", None) or getattr(question, "close_time", None)
    if not found or not isinstance(when, datetime):
        return None
    m1, d1, m2, d2 = MONTHS.get(found.group(1).lower()), int(found.group(2)), \
        MONTHS.get(found.group(3).lower()), int(found.group(4))
    if not (m1 and m2):
        return None
    # It resolves shortly after the period ends: "Dec 21 - Jan 1" in January.
    year = when.year if m2 <= when.month else when.year - 1
    try:
        end = date(year, m2, d2)
        start = date(year if m1 <= m2 else year - 1, m1, d1)
    except ValueError:
        return None
    return start, end


def _sessions(after: date, through: date) -> int:
    """Weekdays after ``after`` up to and including ``through``."""
    return sum(1 for k in range(1, (through - after).days + 1)
               if (after + timedelta(days=k)).weekday() < 5)


def _sd(xs: list[float]) -> float | None:
    if len(xs) < 20:
        return None
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


# ------------------------------------------------------------------- FRED

def _fred(fetch: Fetch, series: str, since: date) -> list[tuple[date, float]]:
    raw = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?"
                + urllib.parse.urlencode({"id": series, "cosd": since.isoformat()}), {})
    rows = []
    for row in csv.reader(io.StringIO(raw.decode("utf-8", "replace"))):
        if len(row) < 2:
            continue
        try:
            rows.append((date.fromisoformat(row[0]), float(row[1])))
        except ValueError:
            continue  # the header, and "." for a missing day
    return rows


def fred_level(question: Any, fetch: Fetch, now: datetime) -> str:
    text = _text(question, "question_text", "resolution_criteria", "background_info")
    series = next((s for s, words in FRED.items() if any(w in text for w in words)), None)
    span = period(question)
    if not series or not span:
        return ""
    rows = _fred(fetch, series, now.date() - timedelta(days=LOOKBACK_DAYS))
    if len(rows) < 21:
        return ""
    start, end = span
    last_day, last = rows[-1]
    changes = [b - a for (_, a), (_, b) in zip(rows, rows[1:])]
    sd = _sd(changes)
    line = (f"- FRED {series}: latest {last:.2f} on {last_day.isoformat()}; 3-month range "
            f"{min(v for _, v in rows):.2f}-{max(v for _, v in rows):.2f}")
    if end <= last_day:
        on_end = [v for d, v in rows if d <= end]
        return line + (f"; value for the period's last day ({end.isoformat()}): {on_end[-1]:.2f}"
                       if on_end else "") + " [FRED, now]"
    if sd is not None:
        n = _sessions(last_day, end)
        move = sd * math.sqrt(max(n, 1))
        line += (f"; daily change s.d. {sd:.3f} over the last {len(changes)} days; "
                 f"{n} sessions to {end.isoformat()}: 1 s.d. change ±{move:.2f}, so a normal's "
                 f"10-90 range is {last - 1.28 * move:.2f} to {last + 1.28 * move:.2f}")
    return line + " [FRED, now]"


# ------------------------------------------------------------------ Yahoo

def _yahoo(fetch: Fetch, symbol: str, rng: str = "3mo") -> list[tuple[date, float, float]]:
    """(day, close, high) per session, New York dates."""
    data = json.loads(fetch("https://query1.finance.yahoo.com/v8/finance/chart/"
                            + urllib.parse.quote(symbol) + f"?range={rng}&interval=1d", {}))
    res = data["chart"]["result"][0]
    quote = res["indicators"]["quote"][0]
    offset = int((res.get("meta") or {}).get("gmtoffset") or -4 * 3600)
    out = []
    for ts, close, high in zip(res.get("timestamp") or [], quote.get("close") or [],
                               quote.get("high") or [None] * len(quote.get("close") or [])):
        if close is None:
            continue
        day = datetime.fromtimestamp(ts + offset, tz=timezone.utc).date()
        out.append((day, float(close), float(high) if high is not None else float(close)))
    return out


def _pair(question: Any) -> tuple[str, str] | None:
    """The two Yahoo symbols of "How much will A ... exceed B", in that order."""
    text = _text(question, "question_text")
    if "exceed" not in text:
        return None
    first, _, second = text.partition("exceed")
    a = next((sym for name, sym in YAHOO if name in first), None)
    b = next((sym for name, sym in YAHOO if name in second), None)
    return (a, b) if a and b and a != b else None


def return_difference(question: Any, fetch: Fetch, now: datetime) -> str:
    pair, span = _pair(question), period(question)
    if not pair or not span:
        return ""
    a, b = (_yahoo(fetch, s) for s in pair)
    closes_a, closes_b = dict((d, c) for d, c, _ in a), dict((d, c) for d, c, _ in b)
    days = sorted(set(closes_a) & set(closes_b))
    if len(days) < 21:
        return ""
    diffs = [100 * (math.log(closes_a[y] / closes_a[x]) - math.log(closes_b[y] / closes_b[x]))
             for x, y in zip(days, days[1:])]
    sd = _sd(diffs)
    start, end = span
    full = _sessions(start - timedelta(days=1), end)
    line = f"- {pair[0]} minus {pair[1]}, daily return difference s.d. {sd:.2f} pp over {len(diffs)} sessions" \
        if sd is not None else f"- {pair[0]} minus {pair[1]}"
    if sd is not None:
        line += (f"; over a whole {full}-session period that is ±{sd * math.sqrt(full):.1f} pp "
                 f"for 1 s.d. (10-90 about ±{1.28 * sd * math.sqrt(full):.1f} pp, centred near 0)")
    before = [d for d in days if d < start]
    last = days[-1]
    if before and last >= start:
        p0 = before[-1]
        upto = max(d for d in days if d <= end)
        so_far = 100 * (closes_a[upto] / closes_a[p0] - 1) - 100 * (closes_b[upto] / closes_b[p0] - 1)
        line += (f"; this period so far (closes {p0.isoformat()} to {upto.isoformat()}): "
                 f"{so_far:+.2f} pp")
        if upto < end and sd is not None:
            left = _sessions(upto, end)
            line += f", {left} sessions left (±{sd * math.sqrt(max(left, 1)):.1f} pp for 1 s.d. more)"
        elif upto >= end:
            line += " (the period is over: this is the outcome on Yahoo closes)"
    return line + " [Yahoo, now]"


def vix_maximum(question: Any, fetch: Fetch, now: datetime) -> str:
    text = _text(question, "question_text", "resolution_criteria")
    span = period(question)
    if "vix" not in text or "max" not in text or not span:
        return ""
    rows = _yahoo(fetch, "^VIX", "1y")
    if len(rows) < 60:
        return ""
    start, end = span
    n = max(_sessions(start - timedelta(days=1), end), 1)
    highs = [h for _, _, h in rows]
    windows = sorted(max(highs[i:i + n]) for i in range(len(highs) - n + 1))

    def q(p: float) -> float:
        return windows[min(len(windows) - 1, int(p * len(windows)))]

    last_day, last_close, _ = rows[-1]
    line = (f"- VIX: last close {last_close:.2f} on {last_day.isoformat()}; the highest intraday "
            f"value over {n} consecutive sessions in the last year: 10th/50th/90th percentile "
            f"{q(0.1):.1f}/{q(0.5):.1f}/{q(0.9):.1f}")
    inside = [h for d, _, h in rows if start <= d <= end]
    if inside:
        line += f"; highest so far this period: {max(inside):.2f} (the outcome cannot be lower)"
    return line + " [Yahoo, now]"


def build(question: Any, fetch: Fetch, now: datetime | None = None) -> str:
    """The lines that apply to ``question``, or ""."""
    now = now or datetime.now(timezone.utc)
    lines = []
    for source in (fred_level, return_difference, vix_maximum):
        try:
            line = source(question, fetch, now)
        except Exception:  # noqa: BLE001 - one bad source never costs the others
            line = ""
        if line:
            lines.append(line)
    return "\n".join(lines)
