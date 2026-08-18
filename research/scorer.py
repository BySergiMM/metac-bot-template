"""Offline scoring of our own forecasts.

Scope discipline: this module scores forecasts that already exist. It never
produces one. It imports nothing from the forecaster.

Three tiers of claim, kept rigorously separate, because conflating them is how
a laboratory starts lying to itself:

``EXACT``
    Recomputed from the same inputs Metaculus used, with the same formula
    transcribed from ``scoring/score_math.py``. An exact result is expected to
    match Metaculus' published number to floating-point tolerance, and the
    validator checks that it does.

``PROXY``
    A defensible stand-in whose value is *not* the competition metric. Useful
    for ranking variants against each other, never for predicting points.

``UNAVAILABLE``
    We know the formula and we know we lack an input. Reported as unavailable
    with the specific missing term named. Never silently replaced by a proxy.

The spot peer score, transcribed from ``evaluate_forecasts_peer_spot_forecast``::

    score = 100 * (N / (N - 1)) * ln(p / gmp)      # halved if continuous
    coverage = 1.0 if a forecast was active at the spot instant else 0.0

where ``p`` is our probability mass on the resolved outcome, ``gmp`` is the
same quantity for the geometric mean of every eligible forecaster, and ``N`` is
the number of forecasters in that geometric mean. ``p`` we have. ``gmp`` and
``N`` require every other forecaster's distribution, which is the term the API
does not give an ordinary account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from research.track_record import (
    CONTINUOUS_TYPES,
    ForecastRow,
    QuestionRow,
    TrackRecord,
)

EXACT = "EXACT"
PROXY = "PROXY"
UNAVAILABLE = "UNAVAILABLE"

SPOT_PEER = "spot_peer"
PEER = "peer"
BASELINE = "baseline"
SPOT_BASELINE = "spot_baseline"

GEOMETRIC_MEAN_METHOD = "geometric_mean"

# Metaculus clamps forecasts to [0.001, 0.999] on submission, so a true zero
# cannot occur; this floor only guards against a malformed CSV cell turning
# into -inf and poisoning an entire mean.
MIN_PROBABILITY = 1e-12


def peer_factor(num_forecasters: int | None) -> float:
    """``N / (N - 1)``, with Metaculus' own degenerate-case handling.

    ``get_geometric_means`` stores ``num_forecasters = predictors if
    predictors > 1 else 0``, so N is never exactly 1 and the factor is 0 when
    we were the only forecaster -- a peer score against nobody is 0, not
    undefined.
    """
    if num_forecasters is None:
        return 0.0
    if num_forecasters <= 1:
        return 0.0
    return num_forecasters / (num_forecasters - 1.0)


def log_score(probability: float | None) -> float | None:
    """``ln(p)`` on the resolved outcome. Defined for every question type,
    because ``p`` is the server-computed probability mass at the resolution
    bucket rather than anything we derive ourselves."""
    if probability is None:
        return None
    return math.log(max(probability, MIN_PROBABILITY))


def brier_score(probability_yes: float | None, resolved_yes: bool | None) -> float | None:
    """Binary Brier. Deliberately not generalised to other question types.

    Brier is a *secondary* metric here. The competition is scored on a
    log-based rule, and Metaculus states plainly that the log score avoids the
    problems Brier has. Reporting Brier for continuous questions would invite
    optimising the wrong thing in a place where it is hardest to notice.
    """
    if probability_yes is None or resolved_yes is None:
        return None
    outcome = 1.0 if resolved_yes else 0.0
    return (probability_yes - outcome) ** 2


@dataclass
class SpotScore:
    """Our reconstruction for one question."""

    question_id: int
    question_type: str
    question_weight: float
    spot_timestamp: float | None

    # our forecast at the spot instant
    active_forecast: bool = False
    probability_of_resolution: float | None = None
    probability_yes: float | None = None
    reproduced_coverage: float = 0.0

    # scores we can always compute
    spot_log_score: float | None = None
    spot_brier: float | None = None

    # the competition metric
    spot_peer: float | None = None
    spot_peer_tier: str = UNAVAILABLE
    spot_peer_missing: list[str] = field(default_factory=list)

    # geometric-mean inputs, when present
    gm_probability: float | None = None
    gm_forecaster_count: int | None = None

    notes: list[str] = field(default_factory=list)


def select_forecast_at_spot(
    forecasts: list[ForecastRow], spot_timestamp: float | None
) -> ForecastRow | None:
    """The forecast row whose interval contains the spot instant.

    Matches ``start <= spot < end`` from the server. There should be at most
    one; if the export ever produced overlapping intervals we take the latest
    starting one, which is what a "last forecast standing" rule implies.
    """
    if spot_timestamp is None:
        return None
    active = [f for f in forecasts if f.is_active_at(spot_timestamp)]
    if not active:
        return None
    active.sort(key=lambda f: f.start_time.timestamp() if f.start_time else 0.0)
    return active[-1]


def select_geometric_mean_at_spot(
    aggregates: list[ForecastRow], spot_timestamp: float | None
) -> ForecastRow | None:
    """Last geometric-mean entry *strictly before* the spot instant.

    Transcribed from::

        for gm in geometric_mean_forecasts[::-1]:
            if gm.timestamp < spot_forecast_timestamp:
                g = gm.pmf
                break

    The strict inequality matters: an aggregate recomputed exactly at the spot
    instant is not the one used.
    """
    if spot_timestamp is None:
        return None
    candidates = [
        row
        for row in aggregates
        if row.start_time is not None and row.start_time.timestamp() < spot_timestamp
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.start_time.timestamp())  # type: ignore[union-attr]
    return candidates[-1]


def score_question(
    question: QuestionRow,
    own_forecasts: list[ForecastRow],
    geometric_means: list[ForecastRow] | None = None,
) -> SpotScore:
    """Reconstruct every score we can for one resolved question."""
    spot_ts = question.spot_timestamp()
    result = SpotScore(
        question_id=question.question_id,
        question_type=question.question_type,
        question_weight=question.question_weight if question.question_weight is not None else 1.0,
        spot_timestamp=spot_ts,
    )

    if not question.is_resolved:
        result.notes.append("question has no usable resolution (unresolved or annulled)")
        result.spot_peer_missing.append("resolution")
        return result

    if spot_ts is None:
        result.notes.append("no spot scoring time could be derived from the question timestamps")
        result.spot_peer_missing.append("spot_scoring_time")
        return result

    forecast = select_forecast_at_spot(own_forecasts, spot_ts)
    if forecast is None:
        # This is the coverage-0 case and it is *the* thing to measure: under a
        # summed leaderboard a missing forecast is a forfeited question.
        result.notes.append("no forecast of ours was active at the spot instant")
        result.reproduced_coverage = 0.0
        result.spot_peer = 0.0
        result.spot_peer_tier = EXACT
        return result

    result.active_forecast = True
    result.reproduced_coverage = 1.0
    result.probability_of_resolution = forecast.probability_of_resolution
    result.probability_yes = forecast.probability_yes

    p = forecast.probability_of_resolution
    if p is None:
        result.notes.append(
            "forecast row has no 'Probability of Resolution'; the server only "
            "fills it once the question is resolved"
        )
        result.spot_peer_missing.append("probability_of_resolution")
        return result

    result.spot_log_score = log_score(p)
    if question.question_type == "binary":
        resolved_yes = (question.resolution or "").strip().lower() == "yes"
        result.spot_brier = brier_score(forecast.probability_yes, resolved_yes)

    gm_row = select_geometric_mean_at_spot(geometric_means or [], spot_ts)
    if gm_row is None or gm_row.probability_of_resolution is None:
        result.spot_peer_tier = UNAVAILABLE
        result.spot_peer_missing.append("geometric_mean_of_other_forecasters")
        result.notes.append(
            "spot peer needs the geometric mean of all eligible forecasters at "
            "the spot instant; no geometric_mean aggregate rows are present in "
            "this dataset"
        )
        return result

    result.gm_probability = gm_row.probability_of_resolution
    result.gm_forecaster_count = gm_row.forecaster_count
    factor = peer_factor(gm_row.forecaster_count)
    if factor == 0.0:
        result.spot_peer = 0.0
        result.spot_peer_tier = EXACT
        result.notes.append("fewer than two forecasters in the aggregate; peer score is 0 by definition")
        return result

    gmp = max(gm_row.probability_of_resolution, MIN_PROBABILITY)
    raw = 100.0 * factor * math.log(max(p, MIN_PROBABILITY) / gmp)
    if question.question_type in CONTINUOUS_TYPES:
        raw /= 2.0
    result.spot_peer = raw
    result.spot_peer_tier = EXACT
    return result


def implied_geometric_mean(
    metaculus_spot_peer: float,
    probability_of_resolution: float,
    num_forecasters: int | None,
    is_continuous: bool,
) -> float | None:
    """Invert the spot peer formula to recover the peer aggregate.

    DIAGNOSTIC, not a score. Given Metaculus' published ``spot_peer`` and our
    own ``p``, the only unknowns are ``gmp`` and ``N``::

        S = 100 * (N/(N-1)) * ln(p/gmp)   [/2 if continuous]
        =>  gmp = p * exp(-S * k / (100 * N/(N-1)))       k = 2 if continuous

    Its use is a falsifiability check: a correct pipeline yields an implied
    ``gmp`` inside (0, 1]. A value outside that range proves something upstream
    is wrong -- the wrong forecast selected, the wrong spot instant, or the
    wrong ``p``. Since ``N`` is usually unknown we evaluate the large-``N``
    limit (factor -> 1), which is a lower bound on the correction.
    """
    factor = peer_factor(num_forecasters) if num_forecasters else 1.0
    if factor == 0.0:
        return None
    scale = 2.0 if is_continuous else 1.0
    exponent = -metaculus_spot_peer * scale / (100.0 * factor)
    try:
        return probability_of_resolution * math.exp(exponent)
    except OverflowError:
        return None


@dataclass
class ScoreSummary:
    n_questions_in_dataset: int = 0
    n_resolved: int = 0
    n_with_own_forecast: int = 0
    n_scored: int = 0
    n_covered_at_spot: int = 0
    n_missed_at_spot: int = 0

    mean_spot_log_score: float | None = None
    mean_spot_brier: float | None = None
    coverage_rate: float | None = None

    spot_peer_tier: str = UNAVAILABLE
    total_spot_peer: float | None = None
    weighted_total_spot_peer: float | None = None
    mean_spot_peer: float | None = None

    by_question_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_inputs: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_questions_in_dataset": self.n_questions_in_dataset,
            "n_resolved": self.n_resolved,
            "n_with_own_forecast": self.n_with_own_forecast,
            "n_scored": self.n_scored,
            "n_covered_at_spot": self.n_covered_at_spot,
            "n_missed_at_spot": self.n_missed_at_spot,
            "mean_spot_log_score": self.mean_spot_log_score,
            "mean_spot_brier": self.mean_spot_brier,
            "coverage_rate": self.coverage_rate,
            "spot_peer_tier": self.spot_peer_tier,
            "total_spot_peer": self.total_spot_peer,
            "weighted_total_spot_peer": self.weighted_total_spot_peer,
            "mean_spot_peer": self.mean_spot_peer,
            "by_question_type": self.by_question_type,
            "missing_inputs": self.missing_inputs,
        }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def score_track_record(
    record: TrackRecord, user_id: int | None = None
) -> tuple[list[SpotScore], ScoreSummary]:
    """Score every resolved question in a dataset.

    Returns per-question results and an aggregate summary. The summary's
    ``spot_peer_tier`` is EXACT only if *every* scored question had the
    geometric-mean input available; one missing term downgrades the whole
    aggregate, because a total summed over a subset is not the total.
    """
    if user_id is None:
        user_id = record.infer_user_id()

    by_question = record.forecasts_by_question(user_id=user_id)
    aggregates_by_question: dict[int, list[ForecastRow]] = {}
    for row in record.aggregate_forecasts(method=GEOMETRIC_MEAN_METHOD):
        aggregates_by_question.setdefault(row.question_id, []).append(row)

    results: list[SpotScore] = []
    summary = ScoreSummary(n_questions_in_dataset=len(record.questions))

    for question_id, question in sorted(record.questions.items()):
        own = by_question.get(question_id, [])
        if question.is_resolved:
            summary.n_resolved += 1
        if own:
            summary.n_with_own_forecast += 1
        if not question.is_resolved:
            continue
        result = score_question(
            question, own, geometric_means=aggregates_by_question.get(question_id)
        )
        results.append(result)

    summary.n_scored = len(results)
    summary.n_covered_at_spot = sum(1 for r in results if r.reproduced_coverage >= 1.0)
    summary.n_missed_at_spot = summary.n_scored - summary.n_covered_at_spot
    if summary.n_scored:
        summary.coverage_rate = summary.n_covered_at_spot / summary.n_scored

    summary.mean_spot_log_score = _mean(
        [r.spot_log_score for r in results if r.spot_log_score is not None]
    )
    summary.mean_spot_brier = _mean(
        [r.spot_brier for r in results if r.spot_brier is not None]
    )

    exact = [r for r in results if r.spot_peer_tier == EXACT and r.spot_peer is not None]
    if exact and len(exact) == len(results):
        summary.spot_peer_tier = EXACT
        summary.total_spot_peer = sum(r.spot_peer for r in exact)  # type: ignore[misc]
        summary.weighted_total_spot_peer = sum(
            r.spot_peer * r.question_weight for r in exact  # type: ignore[operator]
        )
        summary.mean_spot_peer = summary.total_spot_peer / len(exact)
    else:
        summary.spot_peer_tier = UNAVAILABLE

    for result in results:
        bucket = summary.by_question_type.setdefault(
            result.question_type,
            {"n": 0, "covered": 0, "log_scores": [], "mean_log_score": None},
        )
        bucket["n"] += 1
        if result.reproduced_coverage >= 1.0:
            bucket["covered"] += 1
        if result.spot_log_score is not None:
            bucket["log_scores"].append(result.spot_log_score)
        for missing in result.spot_peer_missing:
            summary.missing_inputs[missing] = summary.missing_inputs.get(missing, 0) + 1

    for bucket in summary.by_question_type.values():
        bucket["mean_log_score"] = _mean(bucket.pop("log_scores"))

    return results, summary
