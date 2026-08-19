#!/usr/bin/env python3
"""Milestone 2.5 -- reconstruct the forecaster's real throughput from CI logs.

    python3 research/throughput_audit.py --cache DIR [--refresh]

ANALYSIS ONLY. Reads GitHub Actions logs via `gh` and counts what happened. It
never imports the forecaster, never calls OpenRouter, never calls Metaculus,
and never writes anything outside its cache and report.

Deliberately not wired into anything: this answers one question -- where does
coverage actually get lost -- and is disposable once answered.

Every counter below maps to a log line the bot or the SDK actually emits:

  "Retrieved N questions from tournament T"   metaculus_client.py:437
  "Skipping N previously forecasted questions" forecast_bot.py:236
  "Found Research for URL ..."                 main.py:176
  "Summarizing research for question: ..."     forecast_bot.py:273
  "Forecasted URL ... with prediction: ..."    main.py:241 (one per prediction)
  "Posting prediction on question N"           metaculus_client.py:261
  "Posted comment on post N"                   metaculus_client.py:255
  "Bot submitted N forecast(s)."               bot_helpers.py:124
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any

WORKFLOWS = [
    "run_bot_on_tournament.yaml",
    "test_bot.yaml",
    "run_bot_on_metaculus_cup.yaml",
]

PATTERNS = {
    "retrieved": re.compile(r"Retrieved (\d+) questions from tournament (\S+)"),
    "skipping": re.compile(r"Skipping (\d+) previously forecasted questions"),
    "research": re.compile(r"Found Research for URL (\S+)"),
    "summarizing": re.compile(r"Summarizing research for question: (\S+)"),
    "forecasted": re.compile(r"Forecasted URL (\S+) with prediction: (.+?)\."),
    "posting": re.compile(r"Posting prediction on question (\d+)"),
    "posted_comment": re.compile(r"Posted comment on post (\d+)"),
    "submitted": re.compile(r"Bot (?:submitted|produced \(dry run\)) (\d+) forecast"),
    "no_new": re.compile(r"No new questions to forecast"),
    "ratelimit_headers": re.compile(
        r'"X-RateLimit-Limit":"(\d+)","X-RateLimit-Remaining":"(\d+)"'
    ),
    "question_url": re.compile(r"metaculus\.com/questions/(\d+)"),
}

ERROR_KINDS = {
    "rate_limit_day": re.compile(r"free-models-per-day"),
    "rate_limit_other": re.compile(r"RateLimitError"),
    "model_not_found": re.compile(r"NotFoundError.*No endpoints found"),
    "timeout": re.compile(r"litellm\.Timeout|Timeout.*timeout passed"),
    "structure_output": re.compile(r"Sampled outputs from structure_output are not the same"),
    "expected_at_least": re.compile(r"Expected at least .* successful predictions"),
    "api_connection": re.compile(r"APIConnectionError|ConnectionError"),
}


def run_gh(args: list[str]) -> str:
    result = subprocess.run(
        ["gh"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180
    )
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", "replace")


def list_runs(workflow: str) -> list[dict[str, Any]]:
    raw = run_gh(
        [
            "run", "list", "--workflow", workflow, "--limit", "200",
            "--json", "databaseId,createdAt,conclusion,event,displayTitle",
        ]
    )
    if not raw:
        return []
    runs = json.loads(raw)
    for entry in runs:
        entry["workflow"] = workflow
    return runs


def get_log(run_id: int, cache_dir: str, refresh: bool = False) -> str:
    path = os.path.join(cache_dir, "{0}.log".format(run_id))
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    text = run_gh(["run", "view", str(run_id), "--log"])
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def analyse_log(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "retrieved": {},
        "retrieved_total": 0,
        "skipped_previously": 0,
        "research_urls": [],
        "summarize_calls": 0,
        "prediction_lines": 0,
        "questions_with_predictions": [],
        "post_prediction_calls": 0,
        "post_comment_calls": 0,
        "submitted": 0,
        "no_new_questions": False,
        "errors": Counter(),
        "rate_limit_headers": [],
        "question_ids": set(),
        "pinned_models": None,
        "has_pin_step": "Pin LLM models" in text or "Pin and verify LLM models" in text,
    }

    for match in PATTERNS["retrieved"].finditer(text):
        count, tournament = int(match.group(1)), match.group(2)
        out["retrieved"][tournament] = out["retrieved"].get(tournament, 0) + count
        out["retrieved_total"] += count

    for match in PATTERNS["skipping"].finditer(text):
        out["skipped_previously"] += int(match.group(1))

    for match in PATTERNS["research"].finditer(text):
        out["research_urls"].append(match.group(1))

    out["summarize_calls"] = len(PATTERNS["summarizing"].findall(text))

    for match in PATTERNS["forecasted"].finditer(text):
        out["prediction_lines"] += 1
        out["questions_with_predictions"].append(match.group(1))

    out["post_prediction_calls"] = len(PATTERNS["posting"].findall(text))
    out["post_comment_calls"] = len(PATTERNS["posted_comment"].findall(text))

    submitted = PATTERNS["submitted"].findall(text)
    out["submitted"] = sum(int(value) for value in submitted)
    out["no_new_questions"] = bool(PATTERNS["no_new"].search(text))

    for kind, pattern in ERROR_KINDS.items():
        hits = len(pattern.findall(text))
        if hits:
            out["errors"][kind] = hits

    for match in PATTERNS["ratelimit_headers"].finditer(text):
        out["rate_limit_headers"].append(
            {"limit": int(match.group(1)), "remaining": int(match.group(2))}
        )

    for match in PATTERNS["question_url"].finditer(text):
        out["question_ids"].add(int(match.group(1)))

    if "nemotron" in text:
        models = re.findall(r"(default|researcher|summarizer|parser)\s+(openrouter/\S+)", text)
        if models:
            out["pinned_models"] = dict(models)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    all_runs: list[dict[str, Any]] = []
    for workflow in WORKFLOWS:
        all_runs.extend(list_runs(workflow))
    all_runs.sort(key=lambda entry: entry["createdAt"])
    print("runs found: {0}".format(len(all_runs)), file=sys.stderr)

    results = []
    for index, entry in enumerate(all_runs):
        text = get_log(entry["databaseId"], args.cache, refresh=args.refresh)
        analysis = analyse_log(text)
        analysis["question_ids"] = sorted(analysis["question_ids"])
        analysis["errors"] = dict(analysis["errors"])
        analysis.update(
            {
                "run_id": entry["databaseId"],
                "created_at": entry["createdAt"],
                "conclusion": entry["conclusion"],
                "event": entry["event"],
                "workflow": entry["workflow"],
                "log_bytes": len(text),
            }
        )
        results.append(analysis)
        print(
            "  [{0}/{1}] {2} {3} retrieved={4} preds={5} submitted={6} errors={7}".format(
                index + 1, len(all_runs), entry["createdAt"], entry["workflow"][:22],
                analysis["retrieved_total"], analysis["prediction_lines"],
                analysis["submitted"], analysis["errors"] or "-",
            ),
            file=sys.stderr,
        )

    payload = {"runs": results}
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
    else:
        print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
