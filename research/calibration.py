"""Calibration of the current system on its resolved questions: counts only.

Why this exists
---------------
The 2026-10-30 comparison with futuresearch found the old system's numeric
ranges too wide (the outcome fell inside its interquartile range 68% of the
time, against an ideal 50%) and the SDK's 5% past every open bound costly; the
fixes of that day (numeric_shape's 2% past a bound, the width line, the live
volatility) are unvalidated. This measures, on questions whose scored forecast
the current system made (from ``NEW_SYSTEM_START``), what those fixes should
move:

* binary: how many, and the mean Brier score;
* numeric, discrete and date: how many, the share whose outcome landed inside
  the forecast's 25-75 and 10-90 ranges (ideal 50% and 80%), how many landed
  past the range, and the mean mass the forecast left past its open bounds.

Only aggregates leave this module: the repository and its Actions logs are
public, and a per-question figure would publish a forecast.

Standard library only, like the rest of research/.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

#: First run of the current system (models, research and prompts of
#: 2026-10-10; see the bot-profile memory): forecasts made from then on.
NEW_SYSTEM_START = datetime(2026, 10, 10, 15, 17, tzinfo=timezone.utc)
CONTINUOUS = ("numeric", "discrete", "date")


def _parse(ts: Any) -> datetime | None:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if not isinstance(ts, str) or not ts:
        return None
    try:
        out = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo else out.replace(tzinfo=timezone.utc)


def scored_forecast(question: dict[str, Any]) -> dict[str, Any] | None:
    """The bot's last forecast made before the question closed, when the
    current system made it; None otherwise."""
    history = ((question.get("my_forecasts") or {}).get("history")) or []
    close = _parse(question.get("actual_close_time")) or _parse(
        question.get("scheduled_close_time"))
    best, best_start = None, None
    for record in history:
        if not isinstance(record, dict):
            continue
        start = _parse(record.get("start_time"))
        if start is None or (close is not None and start >= close):
            continue
        if best_start is None or start > best_start:
            best, best_start = record, start
    if best is None or best_start < NEW_SYSTEM_START:
        return None
    return best


def location(question: dict[str, Any]) -> float | None:
    """Where the resolution sits on the question's [0, 1] axis (below 0 or
    above 1 when past a bound), or None when it cannot be read safely."""
    resolution = str(question.get("resolution") or "").strip()
    if resolution == "below_lower_bound":
        return -math.inf
    if resolution == "above_upper_bound":
        return math.inf
    scaling = question.get("scaling") or {}
    lo, hi, zero = scaling.get("range_min"), scaling.get("range_max"), scaling.get("zero_point")
    try:
        lo, hi = float(lo), float(hi)
        if question.get("type") == "date":
            when = _parse(resolution)
            if when is None:
                return None
            x = when.timestamp()
        else:
            x = float(resolution)
        if not hi > lo:
            return None
        if zero is None:
            return (x - lo) / (hi - lo)
        zero = float(zero)
        return math.log((x - zero) / (lo - zero)) / math.log((hi - zero) / (lo - zero))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def cdf_at(cdf: list[float], loc: float) -> float:
    """The forecast CDF at ``loc``, linear between its evenly spaced points."""
    if loc <= 0:
        return float(cdf[0])
    if loc >= 1:
        return float(cdf[-1])
    pos = loc * (len(cdf) - 1)
    i = min(int(pos), len(cdf) - 2)
    return float(cdf[i]) + (float(cdf[i + 1]) - float(cdf[i])) * (pos - i)


def summarise(questions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates over resolved questions the current system forecast."""
    briers: list[float] = []
    n_cont = in_iqr = in_80 = outside = 0
    past_open: list[float] = []
    # Why resolved questions were left out, by type: counts, for the log.
    skipped: dict[str, int] = {}

    def skip(q: dict[str, Any], reason: str) -> None:
        key = f"{q.get('type')}:{reason}"
        skipped[key] = skipped.get(key, 0) + 1

    for q in questions:
        resolution = str(q.get("resolution") or "").strip().lower()
        if not resolution or resolution in ("annulled", "ambiguous"):
            continue
        if not isinstance((q.get("my_forecasts") or {}).get("history"), list):
            skip(q, "no_history")
            continue
        forecast = scored_forecast(q)
        if forecast is None:
            skip(q, "not_current_system")
            continue
        values = forecast.get("forecast_values")
        if not isinstance(values, list) or not values:
            skip(q, "no_values:" + ",".join(sorted(forecast)))
            continue
        qtype = q.get("type")
        if qtype == "binary" and resolution in ("yes", "no") and len(values) >= 2:
            p = float(values[1])
            briers.append((p - (1.0 if resolution == "yes" else 0.0)) ** 2)
        elif qtype in CONTINUOUS and len(values) >= 2:
            loc = location(q)
            if loc is None:
                skip(q, "unreadable_resolution")
                continue
            n_cont += 1
            if not 0.0 <= loc <= 1.0:
                outside += 1
            else:
                f = cdf_at(values, loc)
                in_iqr += 0.25 <= f <= 0.75
                in_80 += 0.10 <= f <= 0.90
            if q.get("open_lower_bound"):
                past_open.append(float(values[0]))
            if q.get("open_upper_bound"):
                past_open.append(1.0 - float(values[-1]))
    return {
        "binary": len(briers),
        "brier": sum(briers) / len(briers) if briers else None,
        "continuous": n_cont,
        "in_iqr": in_iqr,
        "in_80": in_80,
        "outside": outside,
        "past_open_bound": sum(past_open) / len(past_open) if past_open else None,
        "skipped": skipped,
    }


def render(summary: dict[str, Any]) -> list[str]:
    """WhatsApp lines (Spanish, like the rest of the weekly message)."""
    lines = ["Calibración del sistema actual (desde 10/10, resueltas):"]
    if summary.get("binary"):
        lines.append(f"   binarias: {summary['binary']} · Brier {summary['brier']:.3f}")
    n = summary.get("continuous") or 0
    if n:
        past = summary.get("past_open_bound")
        past_text = f" · masa fuera de límites abiertos {past:.1%}" if past is not None else ""
        lines.append(
            f"   numéricas: {n} · dentro 25–75 {summary['in_iqr'] / n:.0%} (ideal 50%) · "
            f"dentro 10–90 {summary['in_80'] / n:.0%} (ideal 80%) · "
            f"fuera de rango {summary['outside']}{past_text}")
    if len(lines) == 1:
        lines.append("   aún ninguna")
    return lines
