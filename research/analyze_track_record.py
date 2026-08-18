#!/usr/bin/env python3
"""Score, validate and report on a track-record dataset.

    python3 research/analyze_track_record.py                    # latest dataset
    python3 research/analyze_track_record.py --dataset DIR
    python3 research/analyze_track_record.py --json report.json

Runs three passes over an immutable dataset and prints a report:

1. offline scoring       (research/scorer.py)
2. validation vs Metaculus' own published scores (research/validate.py)
3. coverage, production and benchmark (research/coverage.py)

Reads only. Writes nothing except the optional ``--json`` report, and never
into the dataset directory -- a dataset that changed after being analysed
would invalidate the analysis that cited it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.coverage import (  # noqa: E402
    UniverseQuestion,
    benchmark_coverage,
    production_coverage,
)
from research.provenance import (  # noqa: E402
    latest_dataset_dir,
    read_manifest,
    verify_dataset,
)
from research.scorer import EXACT, score_track_record  # noqa: E402
from research.track_record import load_track_record  # noqa: E402
from research.validate import validate  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_ROOT = os.path.join(REPO_ROOT, "research", "datasets")
UNIVERSE_FILE = "tournament_questions.json"


def load_universe(dataset_dir: str) -> dict[str, list[UniverseQuestion]]:
    path = os.path.join(dataset_dir, UNIVERSE_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        raw = json.load(handle)
    out: dict[str, list[UniverseQuestion]] = {}
    for tournament, posts in raw.items():
        questions: list[UniverseQuestion] = []
        for post in posts:
            questions.extend(UniverseQuestion.from_post_json(post, tournament=tournament))
        out[tournament] = questions
    return out


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return format(value * 100, ".1f") + "%"


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    bar = "=" * 78

    manifest = report["dataset"]
    add(bar)
    add("TRACK RECORD REPORT")
    add(bar)
    add("dataset_id : {0}".format(manifest["dataset_id"]))
    add("created_at : {0}".format(manifest["created_at"]))
    add("account    : {0} (user_id={1})".format(
        manifest["account"].get("username"), manifest["account"].get("user_id")
    ))
    add("git commit : {0}{1}".format(
        (manifest.get("git") or {}).get("commit"),
        "  [DIRTY TREE]" if (manifest.get("git") or {}).get("dirty") else "",
    ))
    add("integrity  : {0}".format(report["integrity"]))
    add("")

    scoring = report["scoring"]
    add("-- 1. OFFLINE SCORING " + "-" * 56)
    add("questions in dataset          : {0}".format(scoring["n_questions_in_dataset"]))
    add("resolved                      : {0}".format(scoring["n_resolved"]))
    add("with a forecast of ours       : {0}".format(scoring["n_with_own_forecast"]))
    add("scored offline                : {0}".format(scoring["n_scored"]))
    add("  covered at spot instant     : {0}".format(scoring["n_covered_at_spot"]))
    add("  MISSED at spot instant      : {0}".format(scoring["n_missed_at_spot"]))
    add("coverage rate                 : {0}".format(_pct(scoring["coverage_rate"])))
    add("mean spot log score  [EXACT]  : {0}".format(_fmt(scoring["mean_spot_log_score"])))
    add("mean spot Brier (binary only) : {0}".format(_fmt(scoring["mean_spot_brier"])))
    add("spot peer tier                : {0}".format(scoring["spot_peer_tier"]))
    if scoring["spot_peer_tier"] == EXACT:
        add("total spot peer               : {0}".format(_fmt(scoring["total_spot_peer"], ".2f")))
        add("weighted total spot peer      : {0}".format(_fmt(scoring["weighted_total_spot_peer"], ".2f")))
        add("mean spot peer                : {0}".format(_fmt(scoring["mean_spot_peer"], ".2f")))
    if scoring["missing_inputs"]:
        add("missing inputs                : {0}".format(json.dumps(scoring["missing_inputs"])))
    add("")
    if scoring["by_question_type"]:
        add("  by question type:")
        add("    {0:<16} {1:>5} {2:>9} {3:>16}".format("type", "n", "covered", "mean log score"))
        for qtype, bucket in sorted(scoring["by_question_type"].items()):
            add("    {0:<16} {1:>5} {2:>9} {3:>16}".format(
                qtype, bucket["n"], bucket["covered"], _fmt(bucket["mean_log_score"])
            ))
        add("")

    validation = report["validation"]
    add("-- 2. VALIDATION AGAINST METACULUS " + "-" * 43)
    add("Metaculus score rows present  : {0}".format(json.dumps(validation["metaculus_score_types_present"])))
    add("our spot_peer rows            : {0}".format(validation["n_metaculus_spot_peer_rows"]))
    cov = validation["coverage"]
    add("")
    add("coverage reproduction [EXACT TEST]")
    add("  compared                    : {0}".format(cov["n_compared"]))
    add("  matching                    : {0}  ({1})".format(cov["n_matching"], _pct(cov["match_rate"])))
    add("  mismatches                  : {0}".format(cov["n_mismatches"]))
    for mismatch in cov["mismatches"][:5]:
        add("    q{0}: metaculus={1} ours={2}".format(
            mismatch["question_id"], mismatch["metaculus_coverage"], mismatch["reconstructed_coverage"]
        ))
    peer = validation["spot_peer"]
    add("")
    add("spot peer reproduction: {0}".format(peer["tier"]))
    if peer["tier"] == EXACT:
        add("  compared                    : {0}".format(peer["n_compared"]))
        add("  mean absolute error         : {0}".format(_fmt(peer["mean_absolute_error"], ".6f")))
        add("  max  absolute error         : {0}".format(_fmt(peer["max_absolute_error"], ".6f")))
        add("  correlation                 : {0}".format(_fmt(peer["correlation"], ".6f")))
        add("  within tolerance {0:<10}: {1}  ({2})".format(
            peer["tolerance"], peer["within_tolerance"], _pct(peer["within_tolerance_rate"])
        ))
        for worst in peer["worst"][:5]:
            add("    q{0} ({1}): ours={2} metaculus={3} err={4}".format(
                worst["question_id"], worst["question_type"],
                _fmt(worst["reconstructed"], ".4f"), _fmt(worst["metaculus"], ".4f"),
                _fmt(worst["absolute_error"], ".6f"),
            ))
    else:
        add("  blocked: {0}".format(peer["blocked_reason"]))
        add("  missing terms: {0}".format(", ".join(peer["missing_terms"]) or "n/a"))
        inv = validation["inversion"]
        add("")
        add("  inversion diagnostic (falsifiability check)")
        add("    evaluated                 : {0}".format(inv["n_evaluated"]))
        add("    implied peer prob valid   : {0}  ({1})".format(
            inv["n_valid_probability"], _pct(inv["pass_rate"])
        ))
        add("    implied peer prob INVALID : {0}".format(inv["n_invalid_probability"]))
        for example in inv["examples"][:5]:
            add("      q{0}: S={1} p={2} -> implied gmp={3} {4}".format(
                example["question_id"], _fmt(example["metaculus_spot_peer"], ".3f"),
                _fmt(example["our_p"], ".4f"), _fmt(example["implied_peer_probability"], ".4f"),
                "OK" if example["valid_probability"] else "<<< INVALID",
            ))
    add("")
    add("VERDICT: {0}".format(validation["verdict"]))
    add("")

    add("-- 3. COVERAGE " + "-" * 63)
    for tournament, block in sorted(report["production_coverage"].items()):
        add("production coverage / {0}".format(tournament))
        add("  questions posed by tournament : {0}".format(block["n_universe_questions"]))
        add("  forecasted by our bot         : {0}".format(block["n_forecasted"]))
        add("  NOT forecasted                : {0}".format(block["n_not_forecasted"]))
        add("  coverage rate                 : {0}".format(_pct(block["coverage_rate"])))
        add("  scored by Metaculus           : {0}".format(block["n_scored_by_metaculus"]))
        add("    with coverage 1             : {0}".format(block["n_scored_with_coverage_1"]))
        add("    with coverage 0             : {0}".format(block["n_scored_with_coverage_0"]))
        add("  universe by type              : {0}".format(json.dumps(block["universe_by_type"])))
        add("  uncovered by type             : {0}".format(json.dumps(block["uncovered_by_type"])))
        forfeit = block["forfeited_points_estimate"]
        if forfeit.get("available"):
            add("  forfeited points [PROXY]      : {0} over {1} missed questions".format(
                _fmt(forfeit["estimated_forfeited_points"], ".1f"), forfeit["n_missed"]
            ))
            add("    (mean realised spot peer on covered questions = {0})".format(
                _fmt(forfeit["mean_spot_peer_on_covered_questions"], ".2f")
            ))
        else:
            add("  forfeited points              : not estimable ({0})".format(forfeit.get("reason")))
        add("")

    bench = report["benchmark_coverage"]
    add("benchmark coverage (what the lab can score)")
    add("  questions in dataset          : {0}".format(bench["n_questions"]))
    add("  resolved                      : {0}".format(bench["n_resolved"]))
    add("  evaluable offline             : {0}  ({1})".format(bench["n_evaluable"], _pct(bench["evaluable_rate"])))
    add("  discarded                     : {0}".format(bench["n_discarded"]))
    add("  discard reasons               : {0}".format(json.dumps(bench["discard_reasons"])))
    add("    {0:<16} {1:>7} {2:>9} {3:>10} {4:>10}".format("type", "total", "resolved", "evaluable", "discarded"))
    for qtype, bucket in sorted(bench["by_type"].items()):
        add("    {0:<16} {1:>7} {2:>9} {3:>10} {4:>10}".format(
            qtype, bucket["total"], bucket["resolved"], bucket["evaluable"], bucket["discarded"]
        ))
    add("")

    if manifest.get("limitations"):
        add("-- LIMITATIONS RECORDED WITH THIS DATASET " + "-" * 36)
        for note in manifest["limitations"]:
            add("  * {0}".format(note))
        add("")
    add(bar)
    return "\n".join(lines)


def analyze(dataset_dir: str) -> dict[str, Any]:
    manifest = read_manifest(dataset_dir)
    problems = verify_dataset(dataset_dir)
    record = load_track_record(dataset_dir)
    user_id = (manifest.account or {}).get("user_id") or record.infer_user_id()

    results, summary = score_track_record(record, user_id=user_id)
    validation = validate(record, results, user_id=user_id)

    universe = load_universe(dataset_dir)
    production: dict[str, Any] = {}
    for tournament, questions in universe.items():
        production[tournament] = production_coverage(
            questions, record, results=results, user_id=user_id, tournament=tournament
        ).to_dict()

    return {
        "dataset": manifest.to_dict(),
        "integrity": "OK" if not problems else problems,
        "scoring": summary.to_dict(),
        "per_question": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "question_weight": r.question_weight,
                "active_forecast": r.active_forecast,
                "reproduced_coverage": r.reproduced_coverage,
                "p": r.probability_of_resolution,
                "spot_log_score": r.spot_log_score,
                "spot_brier": r.spot_brier,
                "spot_peer": r.spot_peer,
                "spot_peer_tier": r.spot_peer_tier,
                "notes": r.notes,
            }
            for r in results
        ],
        "validation": validation.to_dict(),
        "production_coverage": production,
        "benchmark_coverage": benchmark_coverage(record, results).to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="dataset directory (default: newest under --root)")
    parser.add_argument("--root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--json", default=None, help="also write the full report as JSON here")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset or latest_dataset_dir(args.root, kind="track-record")
    if not dataset_dir:
        print(
            "No dataset found under {0}.\n"
            "Run: python3 research/fetch_own_track_record.py".format(args.root),
            file=sys.stderr,
        )
        return 1

    report = analyze(dataset_dir)
    print(render(report))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print("wrote {0}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
