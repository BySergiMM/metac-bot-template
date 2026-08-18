"""Coverage analysis: how much of the competition are we actually playing?

Under a summed spot-peer leaderboard a question we never forecast scores
exactly 0 -- not negative, but not positive either, and the median empirical
peer score is positive. Coverage is therefore not hygiene, it is the first
lever on the final total, and it is measurable without any modelling.

Two distinct notions, kept apart on purpose:

**Production coverage**
    Of the questions the tournament actually posed, how many did the live bot
    forecast, and how many did it forecast in time to be standing at the spot
    instant? This is a fact about the deployed system.

**Benchmark coverage**
    Of the questions in a research dataset, how many can the offline lab score
    at all? This is a fact about the instrument. A lab that silently evaluates
    only binary questions while the tournament is 35% continuous is measuring
    the wrong population, however clean its arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from research.scorer import SPOT_PEER, SpotScore
from research.track_record import ALL_QUESTION_TYPES, TrackRecord


@dataclass
class UniverseQuestion:
    """A question the tournament posed, as seen from ``/api/posts/``."""

    post_id: int
    question_id: int | None
    title: str
    question_type: str
    status: str | None
    tournament: str | None
    open_time: str | None = None
    close_time: str | None = None

    @classmethod
    def from_post_json(cls, post: dict[str, Any], tournament: str | None = None) -> list["UniverseQuestion"]:
        """Expand one post into its questions.

        A post may hold a single question, a group of subquestions, or a
        conditional pair. Counting posts instead of questions would understate
        the tournament, because group subquestions are scored individually.
        """
        out: list["UniverseQuestion"] = []
        post_id = post.get("id")
        title = post.get("title") or ""
        status = post.get("status") or post.get("curation_status")

        question = post.get("question")
        if isinstance(question, dict):
            out.append(
                cls(
                    post_id=post_id,
                    question_id=question.get("id"),
                    title=title,
                    question_type=(question.get("type") or "unknown"),
                    status=status,
                    tournament=tournament,
                    open_time=question.get("open_time"),
                    close_time=question.get("scheduled_close_time"),
                )
            )
            return out

        group = post.get("group_of_questions")
        if isinstance(group, dict):
            for sub in group.get("questions") or []:
                out.append(
                    cls(
                        post_id=post_id,
                        question_id=sub.get("id"),
                        title="{0} / {1}".format(title, sub.get("label") or sub.get("title") or ""),
                        question_type=(sub.get("type") or "unknown"),
                        status=status,
                        tournament=tournament,
                        open_time=sub.get("open_time"),
                        close_time=sub.get("scheduled_close_time"),
                    )
                )
            if out:
                return out

        conditional = post.get("conditional")
        if isinstance(conditional, dict):
            for key in ("question_yes", "question_no"):
                sub = conditional.get(key)
                if isinstance(sub, dict):
                    out.append(
                        cls(
                            post_id=post_id,
                            question_id=sub.get("id"),
                            title="{0} [{1}]".format(title, key),
                            question_type=(sub.get("type") or "unknown"),
                            status=status,
                            tournament=tournament,
                        )
                    )
            if out:
                return out

        # A post we cannot decompose still counts: dropping it would flatter
        # the coverage number, which is the opposite of what this report is for.
        out.append(
            cls(
                post_id=post_id,
                question_id=None,
                title=title,
                question_type="unknown",
                status=status,
                tournament=tournament,
            )
        )
        return out


@dataclass
class ProductionCoverage:
    tournament: str | None = None
    n_universe_questions: int = 0
    n_forecasted: int = 0
    n_not_forecasted: int = 0
    coverage_rate: float | None = None

    n_scored_by_metaculus: int = 0
    n_scored_with_coverage_1: int = 0
    n_scored_with_coverage_0: int = 0

    universe_by_type: dict[str, int] = field(default_factory=dict)
    forecasted_by_type: dict[str, int] = field(default_factory=dict)
    uncovered_by_type: dict[str, int] = field(default_factory=dict)

    forfeited_points_estimate: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tournament": self.tournament,
            "n_universe_questions": self.n_universe_questions,
            "n_forecasted": self.n_forecasted,
            "n_not_forecasted": self.n_not_forecasted,
            "coverage_rate": self.coverage_rate,
            "n_scored_by_metaculus": self.n_scored_by_metaculus,
            "n_scored_with_coverage_1": self.n_scored_with_coverage_1,
            "n_scored_with_coverage_0": self.n_scored_with_coverage_0,
            "universe_by_type": self.universe_by_type,
            "forecasted_by_type": self.forecasted_by_type,
            "uncovered_by_type": self.uncovered_by_type,
            "forfeited_points_estimate": self.forfeited_points_estimate,
            "caveats": self.caveats,
        }


@dataclass
class BenchmarkCoverage:
    n_questions: int = 0
    n_resolved: int = 0
    n_evaluable: int = 0
    n_discarded: int = 0
    evaluable_rate: float | None = None
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    discard_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_questions": self.n_questions,
            "n_resolved": self.n_resolved,
            "n_evaluable": self.n_evaluable,
            "n_discarded": self.n_discarded,
            "evaluable_rate": self.evaluable_rate,
            "by_type": self.by_type,
            "discard_reasons": self.discard_reasons,
        }


def production_coverage(
    universe: Iterable[UniverseQuestion],
    record: TrackRecord,
    results: list[SpotScore] | None = None,
    user_id: int | None = None,
    tournament: str | None = None,
) -> ProductionCoverage:
    """Compare what the tournament posed against what we forecast."""
    if user_id is None:
        user_id = record.infer_user_id()

    report = ProductionCoverage(tournament=tournament)
    forecasted_question_ids = {
        row.question_id for row in record.own_forecasts(user_id=user_id)
    }

    seen: set[Any] = set()
    for question in universe:
        key = question.question_id if question.question_id is not None else ("post", question.post_id)
        if key in seen:
            continue
        seen.add(key)
        report.n_universe_questions += 1
        qtype = question.question_type or "unknown"
        report.universe_by_type[qtype] = report.universe_by_type.get(qtype, 0) + 1
        if question.question_id is not None and question.question_id in forecasted_question_ids:
            report.n_forecasted += 1
            report.forecasted_by_type[qtype] = report.forecasted_by_type.get(qtype, 0) + 1
        else:
            report.uncovered_by_type[qtype] = report.uncovered_by_type.get(qtype, 0) + 1

    report.n_not_forecasted = report.n_universe_questions - report.n_forecasted
    if report.n_universe_questions:
        report.coverage_rate = report.n_forecasted / report.n_universe_questions

    spot_rows = record.own_scores(user_id=user_id, score_type=SPOT_PEER)
    report.n_scored_by_metaculus = len(spot_rows)
    for row in spot_rows:
        if row.coverage is not None and row.coverage >= 0.5:
            report.n_scored_with_coverage_1 += 1
        else:
            report.n_scored_with_coverage_0 += 1

    report.forfeited_points_estimate = _forfeit_estimate(report, spot_rows)
    report.caveats = [
        "A question we never forecast scores exactly 0, and the leaderboard is "
        "a weighted SUM over questions. Missing questions cannot lose points, "
        "but they forfeit every point they could have won.",
        "The forfeited-points figure below is an ESTIMATE built from our own "
        "realised mean on the questions we did cover. It assumes the missed "
        "questions were no harder than the covered ones, which is optimistic if "
        "misses cluster on bursty question drops.",
        "The universe count comes from the posts API at fetch time. Questions "
        "created and resolved between fetches are invisible to it, so true "
        "coverage may be lower than reported, never higher.",
    ]
    return report


def _forfeit_estimate(report: ProductionCoverage, spot_rows: list[Any]) -> dict[str, Any]:
    realised = [row.score for row in spot_rows if row.score is not None and row.coverage and row.coverage >= 0.5]
    if not realised:
        return {
            "available": False,
            "reason": "no scored questions with coverage yet, so there is no realised mean to extrapolate from",
        }
    mean_realised = sum(realised) / len(realised)
    missed = report.n_not_forecasted + report.n_scored_with_coverage_0
    return {
        "available": True,
        "mean_spot_peer_on_covered_questions": mean_realised,
        "n_covered_scored": len(realised),
        "n_missed": missed,
        "estimated_forfeited_points": mean_realised * missed,
        "method": "mean realised spot peer on covered questions x number of missed questions",
        "tier": "PROXY",
    }


def benchmark_coverage(record: TrackRecord, results: list[SpotScore]) -> BenchmarkCoverage:
    """What fraction of the dataset can the lab actually score, and why not the rest."""
    report = BenchmarkCoverage(n_questions=len(record.questions))
    by_id = {r.question_id: r for r in results}

    for question_id, question in record.questions.items():
        qtype = question.question_type or "unknown"
        bucket = report.by_type.setdefault(
            qtype, {"total": 0, "resolved": 0, "evaluable": 0, "discarded": 0}
        )
        bucket["total"] += 1

        if not question.is_resolved:
            report.n_discarded += 1
            bucket["discarded"] += 1
            report.discard_reasons["unresolved_or_annulled"] = (
                report.discard_reasons.get("unresolved_or_annulled", 0) + 1
            )
            continue

        report.n_resolved += 1
        bucket["resolved"] += 1

        result = by_id.get(question_id)
        if result is None:
            report.n_discarded += 1
            bucket["discarded"] += 1
            report.discard_reasons["not_scored"] = report.discard_reasons.get("not_scored", 0) + 1
            continue
        if result.spot_log_score is None and result.active_forecast:
            report.n_discarded += 1
            bucket["discarded"] += 1
            report.discard_reasons["no_probability_of_resolution"] = (
                report.discard_reasons.get("no_probability_of_resolution", 0) + 1
            )
            continue

        report.n_evaluable += 1
        bucket["evaluable"] += 1

    if report.n_questions:
        report.evaluable_rate = report.n_evaluable / report.n_questions

    # Make silence visible: a type present in the competition but absent here
    # is exactly the blind spot this report exists to surface.
    for qtype in sorted(ALL_QUESTION_TYPES):
        report.by_type.setdefault(qtype, {"total": 0, "resolved": 0, "evaluable": 0, "discarded": 0})

    return report
