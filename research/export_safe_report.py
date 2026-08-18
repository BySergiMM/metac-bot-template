#!/usr/bin/env python3
"""Build the publishable summary of a track-record analysis.

    python3 research/export_safe_report.py --in full_report.json \
        --out-json milestone2_summary.json --out-text milestone2_report.txt

Why this exists
---------------
The repository is public, so a workflow artifact is downloadable by anyone who
can see the run. Two things must therefore never reach an artifact:

1. **Metaculus content.** Their Terms of Use forbid redistributing API data.
   Question titles, URLs, resolution values and community aggregates are theirs.
2. **Anything that reveals a resolution.** This is the subtle one. Our own
   ``Probability of Resolution`` looks like our data, but it is the probability
   we placed *on the outcome that happened* -- publishing it next to our
   ``Probability Yes`` reveals which outcome that was. Per-question
   probabilities are therefore excluded even though the forecasts themselves
   are ours.

Allowlist, not denylist
-----------------------
This module does not take the full report and strip fields out. It builds a new
document from an explicit list of permitted keys. A denylist fails open: a
field added upstream later would sail straight through. An allowlist fails
closed, which is the correct direction when the failure mode is publishing
somebody else's data on a public URL.

``assert_safe`` then re-checks the finished document against a list of banned
key names and value shapes, so a mistake in the allowlist is still caught
before anything is written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Key names that must never appear anywhere in the published document,
# whatever nests them. Checked recursively after the document is built.
BANNED_KEYS = frozenset(
    {
        "token",
        "authorization",
        "api_key",
        "secret",
        "title",
        "question_title",
        "url",
        "question_url",
        "resolution",
        "resolution_string",
        "raw",
        "community_prediction",
        "probability_yes",
        "probability_of_resolution",
        "p",
        "our_p",
        "gm_probability",
        "gm_p",
        "implied_peer_probability",
        "metaculus_spot_peer",
        "score",
        "reconstructed",
        "metaculus",
        "per_question",
        "examples",
        "worst",
        "body",
    }
)

# A Metaculus token is 40 hex characters. Any long hex-or-base64-ish run in the
# output is treated as a possible credential and aborts the export.
_SECRET_SHAPE = re.compile(r"\b[A-Fa-f0-9]{32,}\b")


class UnsafeReportError(RuntimeError):
    """The document failed the safety check and must not be written."""


def _hashes_only(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """File provenance without file contents: enough to prove which dataset a
    number came from, not enough to reconstruct any of it."""
    return [
        {
            "name": entry.get("name"),
            "sha256": entry.get("sha256"),
            "bytes": entry.get("bytes"),
            "rows": entry.get("rows"),
        }
        for entry in files or []
    ]


def build_safe_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Construct the publishable document from permitted fields only."""
    dataset = report.get("dataset") or {}
    account = dataset.get("account") or {}
    git = dataset.get("git") or {}
    summary = dataset.get("summary") or {}
    scoring = report.get("scoring") or {}
    validation = report.get("validation") or {}
    coverage_check = validation.get("coverage") or {}
    peer_check = validation.get("spot_peer") or {}
    inversion = validation.get("inversion") or {}
    benchmark = report.get("benchmark_coverage") or {}

    safe: dict[str, Any] = {
        "milestone": "1-2 track record",
        "generated_from": {
            "dataset_id": dataset.get("dataset_id"),
            "created_at": dataset.get("created_at"),
            "kind": dataset.get("kind"),
            "integrity": report.get("integrity"),
            "files": _hashes_only(dataset.get("files") or []),
            "git_commit": git.get("commit"),
            "git_branch": git.get("branch"),
            "git_dirty": git.get("dirty"),
        },
        "account": {
            # The bot's public identity. Never its token.
            "user_id": account.get("user_id"),
            "username": account.get("username"),
            "is_bot": account.get("is_bot"),
            "bot_id": account.get("bot_id"),
            "has_bot_benchmarking_tier": _tier_flag(account.get("data_access_status")),
        },
        "dataset_summary": {
            "n_questions": summary.get("n_questions"),
            "n_resolved_questions": summary.get("n_resolved_questions"),
            "n_own_forecasts": summary.get("n_own_forecasts"),
            "n_questions_with_own_forecast": summary.get("n_questions_with_own_forecast"),
            "n_scored_forecasts": summary.get("n_scored_forecasts"),
            "own_score_types": summary.get("own_score_types") or {},
            "n_aggregate_rows": summary.get("n_aggregate_rows"),
            "aggregation_methods_present": summary.get("aggregation_methods_present") or [],
            "question_types": summary.get("question_types") or {},
            "first_forecast": (summary.get("date_range") or {}).get("first_forecast"),
            "last_forecast": (summary.get("date_range") or {}).get("last_forecast"),
        },
        "offline_scoring": {
            "n_questions_in_dataset": scoring.get("n_questions_in_dataset"),
            "n_resolved": scoring.get("n_resolved"),
            "n_with_own_forecast": scoring.get("n_with_own_forecast"),
            "n_scored": scoring.get("n_scored"),
            "n_covered_at_spot": scoring.get("n_covered_at_spot"),
            "n_missed_at_spot": scoring.get("n_missed_at_spot"),
            "coverage_rate": scoring.get("coverage_rate"),
            "mean_spot_log_score": suppress_small_n(
                scoring.get("mean_spot_log_score"), scoring.get("n_scored")
            ),
            "mean_spot_brier_binary_only": suppress_small_n(
                scoring.get("mean_spot_brier"), scoring.get("n_scored")
            ),
            "spot_peer_tier": scoring.get("spot_peer_tier"),
            "total_spot_peer": suppress_small_n(
                scoring.get("total_spot_peer"), scoring.get("n_scored")
            ),
            "weighted_total_spot_peer": suppress_small_n(
                scoring.get("weighted_total_spot_peer"), scoring.get("n_scored")
            ),
            "mean_spot_peer": suppress_small_n(
                scoring.get("mean_spot_peer"), scoring.get("n_scored")
            ),
            "missing_inputs": scoring.get("missing_inputs") or {},
            "by_question_type": _safe_by_type(scoring.get("by_question_type") or {}),
        },
        "validation": {
            "n_metaculus_spot_peer_rows": validation.get("n_metaculus_spot_peer_rows"),
            "metaculus_score_types_present": validation.get("metaculus_score_types_present") or {},
            "coverage_reproduction": {
                "n_compared": coverage_check.get("n_compared"),
                "n_matching": coverage_check.get("n_matching"),
                "match_rate": coverage_check.get("match_rate"),
                "n_mismatches": coverage_check.get("n_mismatches"),
                # Question ids are public identifiers and carry no content;
                # without them a mismatch is undebuggable.
                "mismatch_question_ids": [
                    entry.get("question_id")
                    for entry in (coverage_check.get("mismatches") or [])
                ][:50],
            },
            "spot_peer_reproduction": {
                "tier": peer_check.get("tier"),
                "n_compared": peer_check.get("n_compared"),
                "mean_absolute_error": peer_check.get("mean_absolute_error"),
                "max_absolute_error": peer_check.get("max_absolute_error"),
                "correlation": peer_check.get("correlation"),
                "within_tolerance": peer_check.get("within_tolerance"),
                "within_tolerance_rate": peer_check.get("within_tolerance_rate"),
                "tolerance": peer_check.get("tolerance"),
                "blocked_reason": peer_check.get("blocked_reason"),
                "missing_terms": peer_check.get("missing_terms") or [],
            },
            "inversion_diagnostic": {
                # Counts only. The raw (score, probability) pairs would reveal
                # both our performance per question and its resolution.
                "n_evaluated": inversion.get("n_evaluated"),
                "n_valid_probability": inversion.get("n_valid_probability"),
                "n_invalid_probability": inversion.get("n_invalid_probability"),
                "pass_rate": inversion.get("pass_rate"),
            },
            "verdict": validation.get("verdict"),
        },
        "production_coverage": {
            tournament: {
                "n_universe_questions": block.get("n_universe_questions"),
                "n_forecasted": block.get("n_forecasted"),
                "n_not_forecasted": block.get("n_not_forecasted"),
                "coverage_rate": block.get("coverage_rate"),
                "n_scored_by_metaculus": block.get("n_scored_by_metaculus"),
                "n_scored_with_coverage_1": block.get("n_scored_with_coverage_1"),
                "n_scored_with_coverage_0": block.get("n_scored_with_coverage_0"),
                "universe_by_type": block.get("universe_by_type") or {},
                "forecasted_by_type": block.get("forecasted_by_type") or {},
                "uncovered_by_type": block.get("uncovered_by_type") or {},
                "forfeited_points_estimate": _safe_forfeit(
                    block.get("forfeited_points_estimate") or {}
                ),
            }
            for tournament, block in (report.get("production_coverage") or {}).items()
        },
        "benchmark_coverage": {
            "n_questions": benchmark.get("n_questions"),
            "n_resolved": benchmark.get("n_resolved"),
            "n_evaluable": benchmark.get("n_evaluable"),
            "n_discarded": benchmark.get("n_discarded"),
            "evaluable_rate": benchmark.get("evaluable_rate"),
            "discard_reasons": benchmark.get("discard_reasons") or {},
            "by_type": benchmark.get("by_type") or {},
        },
        "limitations": list(dataset.get("limitations") or []),
        "excluded_from_this_artifact": [
            "question titles, URLs and resolution values (Metaculus content)",
            "per-question probabilities (they reveal which outcome resolved)",
            "raw CSVs and the dataset itself (kept on the runner, discarded with it)",
            "the API token (never leaves the fetch step's environment)",
            "any mean computed over fewer than {0} observations, shown as "
            "'{1}' -- a mean over one observation is that observation".format(MIN_CELL, SUPPRESSED),
        ],
    }
    return safe


def _tier_flag(status: Any) -> Any:
    """Reduce the data-access probe to a boolean-ish flag.

    The raw payload is echoed back from the server and could grow fields we
    have not reviewed, so only a summarised verdict is published.
    """
    if not isinstance(status, dict):
        return None
    if "error" in status:
        return "probe_failed"
    for key in ("has_data_access", "bot_benchmarking", "data_access"):
        if key in status:
            return bool(status[key])
    return "unknown_shape"


MIN_CELL = 3
SUPPRESSED = "suppressed_small_n"


def suppress_small_n(value: Any, n: Any, minimum: int = MIN_CELL) -> Any:
    """Small-cell suppression.

    A "mean" over one observation is that observation. Publishing it as an
    aggregate would disclose a single question's Metaculus-computed score while
    looking like a summary statistic -- which is exactly the leak this function
    exists to stop, and which the test suite caught on the first run.

    Counts are never suppressed. They carry the coverage story, which is the
    finding that matters, and a count discloses nothing on its own.
    """
    if value is None:
        return None
    if n is None or not isinstance(n, int) or n < minimum:
        return SUPPRESSED
    return value


def _safe_by_type(by_type: dict[str, Any]) -> dict[str, Any]:
    return {
        qtype: {
            "n": bucket.get("n"),
            "covered": bucket.get("covered"),
            "mean_log_score": suppress_small_n(bucket.get("mean_log_score"), bucket.get("n")),
        }
        for qtype, bucket in by_type.items()
    }


def _safe_forfeit(forfeit: dict[str, Any]) -> dict[str, Any]:
    n_covered = forfeit.get("n_covered_scored")
    return {
        "available": forfeit.get("available"),
        "reason": forfeit.get("reason"),
        "tier": forfeit.get("tier"),
        "n_covered_scored": n_covered,
        "n_missed": forfeit.get("n_missed"),
        "mean_spot_peer_on_covered_questions": suppress_small_n(
            forfeit.get("mean_spot_peer_on_covered_questions"), n_covered
        ),
        "estimated_forfeited_points": suppress_small_n(
            forfeit.get("estimated_forfeited_points"), n_covered
        ),
        "method": forfeit.get("method"),
    }


def assert_safe(document: Any, path: str = "$") -> None:
    """Recursively refuse banned key names and credential-shaped strings.

    A second line of defence: if the allowlist above is ever edited carelessly,
    this still stops the artifact from being written.
    """
    if isinstance(document, dict):
        for key, value in document.items():
            lowered = str(key).lower()
            if lowered in BANNED_KEYS:
                raise UnsafeReportError(
                    "banned key {0!r} at {1}; this document must not be published".format(key, path)
                )
            assert_safe(value, "{0}.{1}".format(path, key))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            assert_safe(value, "{0}[{1}]".format(path, index))
    elif isinstance(document, str):
        if _SECRET_SHAPE.search(document) and "git_commit" not in path and "sha256" not in path:
            raise UnsafeReportError(
                "credential-shaped string at {0}; refusing to publish".format(path)
            )


def render_text(safe: dict[str, Any]) -> str:
    """A human-readable rendering of the same allowlisted document.

    Rendered from ``safe``, never from the full report, so the text file cannot
    contain anything the JSON does not.
    """
    lines: list[str] = []
    add = lines.append
    bar = "=" * 78

    def fmt(value: Any, spec: str = ".4f") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return format(value, spec)
        return str(value)

    def pct(value: Any) -> str:
        return "n/a" if value is None else format(value * 100, ".1f") + "%"

    src = safe["generated_from"]
    add(bar)
    add("MILESTONE 2 - TRACK RECORD (aggregates only; safe for a public repo)")
    add(bar)
    add("dataset_id : {0}".format(src["dataset_id"]))
    add("created_at : {0}".format(src["created_at"]))
    add("integrity  : {0}".format(src["integrity"]))
    # Short SHA on purpose: a full 40-char hash trips the credential-shape
    # check that guards the rendered text, and 12 chars identify a commit.
    add("git commit : {0}{1}".format(
        (src["git_commit"] or "unknown")[:12], "  [DIRTY]" if src["git_dirty"] else ""
    ))
    add("account    : {0} (user_id={1}, bot_id={2})".format(
        safe["account"]["username"], safe["account"]["user_id"], safe["account"]["bot_id"]
    ))
    add("data tier  : {0}".format(safe["account"]["has_bot_benchmarking_tier"]))
    add("")
    add("files (hashes only, contents not published):")
    for entry in src["files"]:
        add("  {0:<28} {1} bytes  rows={2}  sha256={3}".format(
            entry["name"], entry["bytes"], entry["rows"], (entry["sha256"] or "")[:16] + "..."
        ))
    add("")

    ds = safe["dataset_summary"]
    add("-- DATASET " + "-" * 67)
    add("questions                     : {0}".format(ds["n_questions"]))
    add("resolved                      : {0}".format(ds["n_resolved_questions"]))
    add("our forecasts (rows)          : {0}".format(ds["n_own_forecasts"]))
    add("questions we forecast on      : {0}".format(ds["n_questions_with_own_forecast"]))
    add("scores Metaculus gave us      : {0}".format(ds["n_scored_forecasts"]))
    add("  by score type               : {0}".format(json.dumps(ds["own_score_types"])))
    add("aggregate rows                : {0}".format(ds["n_aggregate_rows"]))
    add("aggregation methods present   : {0}".format(ds["aggregation_methods_present"] or "none"))
    add("question types                : {0}".format(json.dumps(ds["question_types"])))
    add("forecast window               : {0} -> {1}".format(ds["first_forecast"], ds["last_forecast"]))
    add("")

    sc = safe["offline_scoring"]
    add("-- OFFLINE SCORING " + "-" * 59)
    add("scored offline                : {0}".format(sc["n_scored"]))
    add("  covered at spot instant     : {0}".format(sc["n_covered_at_spot"]))
    add("  MISSED at spot instant      : {0}".format(sc["n_missed_at_spot"]))
    add("coverage rate                 : {0}".format(pct(sc["coverage_rate"])))
    add("mean spot log score  [EXACT]  : {0}".format(fmt(sc["mean_spot_log_score"])))
    add("mean spot Brier (binary only) : {0}".format(fmt(sc["mean_spot_brier_binary_only"])))
    add("spot peer tier                : {0}".format(sc["spot_peer_tier"]))
    if sc["spot_peer_tier"] == "EXACT":
        add("total spot peer               : {0}".format(fmt(sc["total_spot_peer"], ".2f")))
        add("weighted total spot peer      : {0}".format(fmt(sc["weighted_total_spot_peer"], ".2f")))
        add("mean spot peer                : {0}".format(fmt(sc["mean_spot_peer"], ".2f")))
    if sc["missing_inputs"]:
        add("missing inputs                : {0}".format(json.dumps(sc["missing_inputs"])))
    if sc["by_question_type"]:
        add("")
        add("    {0:<16} {1:>5} {2:>9} {3:>16}".format("type", "n", "covered", "mean log score"))
        for qtype, bucket in sorted(sc["by_question_type"].items()):
            add("    {0:<16} {1:>5} {2:>9} {3:>16}".format(
                qtype, bucket["n"], bucket["covered"], fmt(bucket["mean_log_score"])
            ))
    add("")

    val = safe["validation"]
    cov = val["coverage_reproduction"]
    peer = val["spot_peer_reproduction"]
    add("-- VALIDATION AGAINST METACULUS " + "-" * 46)
    add("Metaculus score rows          : {0}".format(json.dumps(val["metaculus_score_types_present"])))
    add("our spot_peer rows            : {0}".format(val["n_metaculus_spot_peer_rows"]))
    add("")
    add("coverage reproduction [EXACT TEST]")
    add("  compared / matching         : {0} / {1}  ({2})".format(
        cov["n_compared"], cov["n_matching"], pct(cov["match_rate"])
    ))
    add("  mismatching question ids    : {0}".format(cov["mismatch_question_ids"] or "none"))
    add("")
    add("spot peer reproduction        : {0}".format(peer["tier"]))
    if peer["tier"] == "EXACT":
        add("  compared                    : {0}".format(peer["n_compared"]))
        add("  mean absolute error         : {0}".format(fmt(peer["mean_absolute_error"], ".8f")))
        add("  max  absolute error         : {0}".format(fmt(peer["max_absolute_error"], ".8f")))
        add("  correlation                 : {0}".format(fmt(peer["correlation"], ".8f")))
        add("  within tolerance            : {0}  ({1})".format(
            peer["within_tolerance"], pct(peer["within_tolerance_rate"])
        ))
    else:
        add("  blocked                     : {0}".format(peer["blocked_reason"]))
        add("  missing terms               : {0}".format(", ".join(peer["missing_terms"]) or "n/a"))
        inv = val["inversion_diagnostic"]
        add("  inversion diagnostic        : {0} evaluated, {1} valid, {2} INVALID ({3})".format(
            inv["n_evaluated"], inv["n_valid_probability"],
            inv["n_invalid_probability"], pct(inv["pass_rate"]),
        ))
    add("")
    add("VERDICT: {0}".format(val["verdict"]))
    add("")

    add("-- COVERAGE " + "-" * 66)
    for tournament, block in sorted(safe["production_coverage"].items()):
        add("production / {0}".format(tournament))
        add("  questions posed               : {0}".format(block["n_universe_questions"]))
        add("  forecasted by our bot         : {0}".format(block["n_forecasted"]))
        add("  NOT forecasted                : {0}".format(block["n_not_forecasted"]))
        add("  coverage rate                 : {0}".format(pct(block["coverage_rate"])))
        add("  scored by Metaculus           : {0} (coverage 1: {1}, coverage 0: {2})".format(
            block["n_scored_by_metaculus"], block["n_scored_with_coverage_1"],
            block["n_scored_with_coverage_0"],
        ))
        add("  universe by type              : {0}".format(json.dumps(block["universe_by_type"])))
        add("  uncovered by type             : {0}".format(json.dumps(block["uncovered_by_type"])))
        forfeit = block["forfeited_points_estimate"]
        if forfeit.get("available"):
            add("  forfeited points [PROXY]      : {0} over {1} missed".format(
                fmt(forfeit["estimated_forfeited_points"], ".1f"), forfeit["n_missed"]
            ))
        else:
            add("  forfeited points              : not estimable ({0})".format(forfeit.get("reason")))
        add("")

    bench = safe["benchmark_coverage"]
    add("benchmark coverage (what the lab can score)")
    add("  evaluable / questions         : {0} / {1}  ({2})".format(
        bench["n_evaluable"], bench["n_questions"], pct(bench["evaluable_rate"])
    ))
    add("  discard reasons               : {0}".format(json.dumps(bench["discard_reasons"])))
    add("    {0:<16} {1:>7} {2:>9} {3:>10} {4:>10}".format(
        "type", "total", "resolved", "evaluable", "discarded"
    ))
    for qtype, bucket in sorted(bench["by_type"].items()):
        add("    {0:<16} {1:>7} {2:>9} {3:>10} {4:>10}".format(
            qtype, bucket.get("total"), bucket.get("resolved"),
            bucket.get("evaluable"), bucket.get("discarded"),
        ))
    add("")

    if safe["limitations"]:
        add("-- LIMITATIONS " + "-" * 63)
        for note in safe["limitations"]:
            add("  * {0}".format(note))
        add("")
    add("-- DELIBERATELY NOT IN THIS ARTIFACT " + "-" * 41)
    for note in safe["excluded_from_this_artifact"]:
        add("  * {0}".format(note))
    add(bar)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input", required=True, help="full analysis JSON")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-text", required=True)
    args = parser.parse_args(argv)

    with open(args.input) as handle:
        report = json.load(handle)

    safe = build_safe_summary(report)
    assert_safe(safe)
    text = render_text(safe)
    assert_safe(text, "$.rendered_text")

    with open(args.out_json, "w") as handle:
        json.dump(safe, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(args.out_text, "w") as handle:
        handle.write(text + "\n")

    print(text)
    print("\nsafety check passed; wrote {0} and {1}".format(args.out_json, args.out_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
