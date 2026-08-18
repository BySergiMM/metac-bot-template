"""Validate our offline reconstruction against Metaculus' published scores.

This is the scientific core of the milestone. Everything else in the lab is
plumbing; this module answers the only question that matters before we start
making changes to the forecaster:

    *Does our instrument measure what the competition measures?*

Two independent checks, in increasing order of strength:

1. **Coverage reproduction.** Metaculus publishes a ``Coverage`` column per
   score row: 1.0 if a forecast of ours was live at the spot instant, 0.0
   otherwise. We derive the same flag from the forecast intervals and the
   question timestamps. This is an exact, binary, falsifiable test of our spot
   instant and of our forecast-selection rule -- and it works even when the
   peer aggregate is unavailable, which is the situation we expect to be in.

2. **Spot peer reproduction.** Only possible when the dataset carries
   ``geometric_mean`` aggregate rows. Reports MAE, max error, Pearson
   correlation and the fraction inside tolerance.

When check 2 is impossible we do NOT substitute a proxy and call it a match.
We run the inversion diagnostic instead: solve Metaculus' own score for the
peer aggregate it implies, and assert that the implied value is a probability.
An implied ``gmp`` outside (0, 1] falsifies the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from research.scorer import (
    EXACT,
    SPOT_PEER,
    SpotScore,
    implied_geometric_mean,
)
from research.track_record import CONTINUOUS_TYPES, TrackRecord

DEFAULT_TOLERANCE = 1e-6


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r without numpy. ``None`` when undefined (n < 2, or either
    series is constant -- correlation with a constant is not 0, it is
    meaningless, and returning 0 would read as "no relationship")."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


@dataclass
class CoverageCheck:
    n_compared: int = 0
    n_matching: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def match_rate(self) -> float | None:
        if not self.n_compared:
            return None
        return self.n_matching / self.n_compared

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_compared": self.n_compared,
            "n_matching": self.n_matching,
            "match_rate": self.match_rate,
            "mismatches": self.mismatches[:50],
            "n_mismatches": len(self.mismatches),
        }


@dataclass
class ScoreCheck:
    tier: str = "UNAVAILABLE"
    n_compared: int = 0
    mean_absolute_error: float | None = None
    max_absolute_error: float | None = None
    correlation: float | None = None
    within_tolerance: int = 0
    tolerance: float = DEFAULT_TOLERANCE
    worst: list[dict[str, Any]] = field(default_factory=list)
    blocked_reason: str | None = None
    missing_terms: list[str] = field(default_factory=list)

    @property
    def within_tolerance_rate(self) -> float | None:
        if not self.n_compared:
            return None
        return self.within_tolerance / self.n_compared

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "n_compared": self.n_compared,
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "correlation": self.correlation,
            "within_tolerance": self.within_tolerance,
            "within_tolerance_rate": self.within_tolerance_rate,
            "tolerance": self.tolerance,
            "worst": self.worst,
            "blocked_reason": self.blocked_reason,
            "missing_terms": sorted(set(self.missing_terms)),
        }


@dataclass
class InversionDiagnostic:
    """Falsifiability check used when exact reproduction is impossible."""

    n_evaluated: int = 0
    n_valid_probability: int = 0
    n_invalid_probability: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_evaluated": self.n_evaluated,
            "n_valid_probability": self.n_valid_probability,
            "n_invalid_probability": self.n_invalid_probability,
            "pass_rate": (
                self.n_valid_probability / self.n_evaluated if self.n_evaluated else None
            ),
            "examples": self.examples[:20],
            "interpretation": self.interpretation,
        }


@dataclass
class ValidationReport:
    coverage: CoverageCheck = field(default_factory=CoverageCheck)
    spot_peer: ScoreCheck = field(default_factory=ScoreCheck)
    inversion: InversionDiagnostic = field(default_factory=InversionDiagnostic)
    metaculus_score_types_present: dict[str, int] = field(default_factory=dict)
    n_metaculus_spot_peer_rows: int = 0
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage.to_dict(),
            "spot_peer": self.spot_peer.to_dict(),
            "inversion": self.inversion.to_dict(),
            "metaculus_score_types_present": self.metaculus_score_types_present,
            "n_metaculus_spot_peer_rows": self.n_metaculus_spot_peer_rows,
            "verdict": self.verdict,
        }


def validate(
    record: TrackRecord,
    results: list[SpotScore],
    user_id: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValidationReport:
    report = ValidationReport()
    if user_id is None:
        user_id = record.infer_user_id()

    for row in record.scores:
        key = row.score_type or "<blank>"
        report.metaculus_score_types_present[key] = (
            report.metaculus_score_types_present.get(key, 0) + 1
        )

    metaculus_rows = {
        row.question_id: row
        for row in record.own_scores(user_id=user_id, score_type=SPOT_PEER)
    }
    report.n_metaculus_spot_peer_rows = len(metaculus_rows)

    by_id = {r.question_id: r for r in results}

    # ------------------------------------------------------ coverage check
    for question_id, metaculus_row in sorted(metaculus_rows.items()):
        ours = by_id.get(question_id)
        if ours is None or metaculus_row.coverage is None:
            continue
        report.coverage.n_compared += 1
        theirs = 1.0 if metaculus_row.coverage >= 0.5 else 0.0
        if abs(ours.reproduced_coverage - theirs) < 1e-9:
            report.coverage.n_matching += 1
        else:
            report.coverage.mismatches.append(
                {
                    "question_id": question_id,
                    "metaculus_coverage": metaculus_row.coverage,
                    "reconstructed_coverage": ours.reproduced_coverage,
                    "spot_timestamp": ours.spot_timestamp,
                    "note": (
                        "our spot instant or forecast-interval selection "
                        "disagrees with the server; most likely cause is a "
                        "per-question spot_scoring_time override, which the "
                        "CSV export does not expose"
                    ),
                }
            )

    # ---------------------------------------------------- spot peer check
    reproducible = [
        (by_id[qid], row)
        for qid, row in metaculus_rows.items()
        if qid in by_id
        and by_id[qid].spot_peer_tier == EXACT
        and by_id[qid].spot_peer is not None
        and row.score is not None
    ]
    all_missing: list[str] = []
    for result in results:
        all_missing.extend(result.spot_peer_missing)

    # An exact comparison is only meaningful when the geometric mean was
    # actually present. A question we scored 0 purely because we had no
    # forecast is trivially reproducible and would inflate the match rate, so
    # those are excluded from the numeric comparison (they are already covered
    # by the coverage check).
    numeric = [
        (ours, theirs)
        for ours, theirs in reproducible
        if ours.gm_probability is not None
    ]

    if numeric:
        report.spot_peer.tier = EXACT
        errors = [abs(ours.spot_peer - theirs.score) for ours, theirs in numeric]  # type: ignore[operator,arg-type]
        report.spot_peer.n_compared = len(numeric)
        report.spot_peer.mean_absolute_error = sum(errors) / len(errors)
        report.spot_peer.max_absolute_error = max(errors)
        report.spot_peer.within_tolerance = sum(1 for e in errors if e <= tolerance)
        report.spot_peer.tolerance = tolerance
        report.spot_peer.correlation = pearson(
            [ours.spot_peer for ours, _ in numeric],  # type: ignore[misc]
            [theirs.score for _, theirs in numeric],  # type: ignore[misc]
        )
        ranked = sorted(
            zip(errors, numeric), key=lambda pair: pair[0], reverse=True
        )[:10]
        report.spot_peer.worst = [
            {
                "question_id": ours.question_id,
                "question_type": ours.question_type,
                "reconstructed": ours.spot_peer,
                "metaculus": theirs.score,
                "absolute_error": error,
                "p": ours.probability_of_resolution,
                "gm_p": ours.gm_probability,
                "n_forecasters": ours.gm_forecaster_count,
            }
            for error, (ours, theirs) in ranked
        ]
    else:
        report.spot_peer.tier = "UNAVAILABLE"
        report.spot_peer.missing_terms = all_missing
        report.spot_peer.blocked_reason = (
            "no question in this dataset carried the geometric-mean aggregate "
            "needed for the spot peer denominator; the formula is known and "
            "implemented, the input is not available to this account"
        )

    # ------------------------------------------------------ inversion check
    if not numeric:
        for question_id, metaculus_row in sorted(metaculus_rows.items()):
            ours = by_id.get(question_id)
            if ours is None or metaculus_row.score is None:
                continue
            if ours.probability_of_resolution is None or not ours.active_forecast:
                continue
            implied = implied_geometric_mean(
                metaculus_row.score,
                ours.probability_of_resolution,
                ours.gm_forecaster_count,
                ours.question_type in CONTINUOUS_TYPES,
            )
            if implied is None:
                continue
            report.inversion.n_evaluated += 1
            valid = 0.0 < implied <= 1.0
            if valid:
                report.inversion.n_valid_probability += 1
            else:
                report.inversion.n_invalid_probability += 1
            if len(report.inversion.examples) < 20 or not valid:
                report.inversion.examples.append(
                    {
                        "question_id": question_id,
                        "question_type": ours.question_type,
                        "metaculus_spot_peer": metaculus_row.score,
                        "our_p": ours.probability_of_resolution,
                        "implied_peer_probability": implied,
                        "valid_probability": valid,
                    }
                )
        report.inversion.interpretation = (
            "Given Metaculus' own spot_peer S and our probability p on the "
            "resolved outcome, the peer aggregate is pinned to "
            "gmp = p * exp(-S*k / (100 * N/(N-1))). Evaluated in the large-N "
            "limit. Every implied value must be a probability in (0, 1]. Any "
            "value outside that range proves our p, our spot instant, or our "
            "forecast selection is wrong. This does not prove the pipeline "
            "correct -- it is a necessary condition, not a sufficient one."
        )

    report.verdict = _verdict(report)
    return report


def _verdict(report: ValidationReport) -> str:
    if report.coverage.n_compared == 0:
        return (
            "NO GROUND TRUTH: the dataset contains no Metaculus spot_peer rows "
            "for our account, so nothing could be validated. Either the bot has "
            "no scored questions yet, or the fetch did not request scores."
        )
    coverage_rate = report.coverage.match_rate or 0.0
    if report.spot_peer.tier == EXACT:
        rate = report.spot_peer.within_tolerance_rate or 0.0
        if rate >= 0.999 and coverage_rate >= 0.999:
            return (
                "EXACT REPRODUCTION: coverage and spot peer both reproduce "
                "Metaculus to tolerance. The offline scorer measures the "
                "competition metric."
            )
        return (
            "PARTIAL REPRODUCTION: the formula runs but {0:.1%} of spot peer "
            "values and {1:.1%} of coverage flags match. Investigate before "
            "trusting any comparison built on this scorer.".format(rate, coverage_rate)
        )
    if coverage_rate >= 0.999:
        return (
            "COVERAGE REPRODUCED, PEER SCORE UNAVAILABLE: our spot instant and "
            "forecast selection agree with Metaculus on every scored question, "
            "so the timing half of the reconstruction is verified. The peer "
            "denominator requires other forecasters' distributions, which this "
            "account cannot retrieve. Offline comparisons must use the log "
            "score plus a frozen reference panel, clearly labelled PROXY."
        )
    return (
        "RECONSTRUCTION DISAGREES: only {0:.1%} of coverage flags match "
        "Metaculus. The spot instant or the forecast-interval selection is "
        "wrong; fix that before drawing any conclusion from offline "
        "scores.".format(coverage_rate)
    )
