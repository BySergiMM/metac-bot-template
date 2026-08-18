#!/usr/bin/env python3
"""Download this bot's own Metaculus track record into an immutable dataset.

    python3 research/fetch_own_track_record.py --check      # auth + access probe only
    python3 research/fetch_own_track_record.py              # full fetch
    python3 research/fetch_own_track_record.py --verify DIR # re-hash an existing dataset

What it fetches and why it is allowed
-------------------------------------
Metaculus grants every account unrestricted access to *its own* data, and
resolution values for "every question it has forecasted at least once"
(metaculus.com/api, "All Authenticated Accounts"). This script uses nothing
else. The server enforces it independently: ``export_data_for_questions`` in
``utils/csv_utils.py`` applies ``user_forecasts.filter(author=user)`` for any
account without the data-access tier, so another forecaster's private rows
cannot reach us even if we asked.

Competition safety
------------------
Every request is a GET. The client class physically refuses other verbs. No
forecast is produced, no question is written to, ``main.py`` is never imported.
Reading our own already-submitted forecasts on already-closed questions is
explicitly the sanctioned use: Metaculus offers this data so bot makers can
perform "retrospective assessment of performance".

The one live-question caution: ``--include-open`` also pulls questions still
open. That is off by default, because reading our own live forecast and then
changing the bot is precisely the human-in-the-loop pattern the tournament
rules prohibit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import __version__
from research.metaculus_read_api import (  # noqa: E402
    DEFAULT_BASE_URL,
    DOWNLOAD_POST_CHUNK,
    MetaculusReadClient,
    MetaculusReadError,
    chunked,
)
from research.provenance import (  # noqa: E402
    FileRecord,
    Manifest,
    content_digest,
    git_info,
    make_dataset_id,
    utc_now_iso,
    verify_dataset,
    write_manifest,
)
from research.track_record import (  # noqa: E402
    FORECAST_CSV,
    QUESTION_CSV,
    SCORE_CSV,
    extract_zip,
    merge_csv_texts,
    read_forecast_csv,
    read_question_csv,
    read_score_csv,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_ROOT = os.path.join(REPO_ROOT, "research", "datasets")
DATASET_KIND = "track-record"

# Tournaments whose question universe defines production coverage.
DEFAULT_UNIVERSE_TOURNAMENTS = ["33022", "minibench"]

# Closed statuses only. "open" is added solely by --include-open.
CLOSED_STATUSES = ["resolved", "closed"]

UNIVERSE_FILE = "tournament_questions.json"
ACCOUNT_FILE = "account.json"
FETCH_LOG_FILE = "fetch_log.json"


def read_token(args: argparse.Namespace) -> str:
    if args.token_file:
        with open(os.path.expanduser(args.token_file)) as handle:
            token = handle.read().strip()
        if token:
            return token
        raise SystemExit("token file {0} is empty".format(args.token_file))
    token = os.environ.get("METACULUS_TOKEN", "").strip()
    if token:
        return token
    # Fall back to a .env in the repo root, same file the bot itself reads.
    env_path = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("METACULUS_TOKEN="):
                    value = line.split("=", 1)[1].strip().strip("'\"")
                    if value and value != "REPLACE_ME":
                        return value
    raise SystemExit(
        "No Metaculus token found.\n"
        "  Set METACULUS_TOKEN in the environment, put it in .env, or pass\n"
        "  --token-file PATH. The token is read-only here: this script only\n"
        "  ever issues GET requests.\n"
        "  Get one at https://www.metaculus.com/accounts/settings/account/"
    )


def probe(client: MetaculusReadClient) -> dict[str, Any]:
    """Identify the account and record what data tier it has."""
    account: dict[str, Any] = {}
    user = client.get_current_user()
    account["user_id"] = user.get("id")
    account["username"] = user.get("username")
    account["is_bot"] = user.get("is_bot")
    try:
        account["data_access_status"] = client.get_data_access_status()
    except MetaculusReadError as exc:
        # Not fatal: the endpoint is newer than some deployments and the tier
        # can be inferred from what the download actually returns.
        account["data_access_status"] = {
            "error": str(exc),
            "status": exc.status,
            "body": exc.body,
        }
    return account


def fetch_universe(client: MetaculusReadClient, tournaments: list[str], statuses: list[str]) -> dict[str, list[dict[str, Any]]]:
    universe: dict[str, list[dict[str, Any]]] = {}
    for tournament in tournaments:
        try:
            posts = client.get_tournament_posts(tournament, statuses=statuses)
        except MetaculusReadError as exc:
            print("  ! tournament {0}: {1}".format(tournament, exc))
            universe[tournament] = []
            continue
        universe[tournament] = posts
        print("  tournament {0}: {1} posts".format(tournament, len(posts)))
    return universe


def download_all(
    client: MetaculusReadClient,
    post_ids: list[int],
    try_geometric_mean: bool,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Download every post's data, chunked, and collect the CSV texts.

    Attempts the geometric-mean aggregation first. That aggregate is the exact
    denominator of the spot peer score, so if this account can retrieve it the
    whole competition metric becomes exactly reproducible offline. It is
    expected to fail for an ordinary account -- we record precisely how it
    fails rather than assuming.
    """
    texts: dict[str, list[str]] = {QUESTION_CSV: [], FORECAST_CSV: [], SCORE_CSV: []}
    log: list[dict[str, Any]] = []
    aggregation_methods: str | None = "geometric_mean" if try_geometric_mean else None
    geometric_mean_supported = try_geometric_mean

    staging = tempfile.mkdtemp(prefix="metac-dl-")
    try:
        for index, chunk in enumerate(chunked(post_ids, DOWNLOAD_POST_CHUNK)):
            entry: dict[str, Any] = {
                "chunk": index,
                "n_posts": len(chunk),
                "aggregation_methods": aggregation_methods,
            }
            data: bytes | None = None
            try:
                data = client.download_data_zip(
                    chunk, aggregation_methods=aggregation_methods
                )
                entry["status"] = "ok"
            except MetaculusReadError as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
                entry["http_status"] = exc.status
                entry["body"] = exc.body
                if aggregation_methods and geometric_mean_supported:
                    # Retry once without the aggregation request: losing the
                    # exact peer denominator is bad, losing the whole track
                    # record because of it would be worse.
                    print(
                        "  ! geometric_mean aggregation rejected "
                        "(HTTP {0}); retrying without it".format(exc.status)
                    )
                    entry["fallback"] = "retried without aggregation_methods"
                    geometric_mean_supported = False
                    aggregation_methods = None
                    try:
                        data = client.download_data_zip(chunk, aggregation_methods=None)
                        entry["status"] = "ok_after_fallback"
                    except MetaculusReadError as exc2:
                        entry["status"] = "error_after_fallback"
                        entry["error"] = str(exc2)
                        entry["http_status"] = exc2.status
                        entry["body"] = exc2.body
                        # Print it: a failure recorded only in a file that
                        # lives on a discarded runner is a failure nobody sees.
                        print(
                            "  ! fallback download also failed (HTTP {0}): {1}".format(
                                exc2.status, (exc2.body or "")[:300]
                            )
                        )
                else:
                    print(
                        "  ! download failed (HTTP {0}): {1}".format(
                            exc.status, (exc.body or "")[:300]
                        )
                    )

            if data is None:
                log.append(entry)
                continue

            chunk_dir = os.path.join(staging, "chunk-{0}".format(index))
            written = extract_zip(data, chunk_dir)
            entry["files"] = [os.path.basename(path) for path in written]
            for name in texts:
                path = os.path.join(chunk_dir, name)
                if os.path.exists(path):
                    with open(path, newline="") as handle:
                        texts[name].append(handle.read())
            log.append(entry)
            print(
                "  chunk {0}: {1} posts -> {2}".format(
                    index, len(chunk), ", ".join(entry["files"]) or "no files"
                )
            )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return texts, log


def summarise(
    question_text: str, forecast_text: str, score_text: str, user_id: int | None
) -> dict[str, Any]:
    questions = read_question_csv(question_text) if question_text else []
    forecasts = read_forecast_csv(forecast_text) if forecast_text else []
    scores = read_score_csv(score_text) if score_text else []

    own = [f for f in forecasts if f.forecaster_id is not None]
    if user_id is not None:
        own = [f for f in own if f.forecaster_id == user_id]
    own_scores = [s for s in scores if s.user_id is not None]
    if user_id is not None:
        own_scores = [s for s in own_scores if s.user_id == user_id]

    by_type: dict[str, int] = {}
    for question in questions:
        by_type[question.question_type] = by_type.get(question.question_type, 0) + 1

    times = [f.start_time for f in own if f.start_time is not None]
    date_range = {
        "first_forecast": min(times).isoformat() if times else None,
        "last_forecast": max(times).isoformat() if times else None,
    }

    score_types: dict[str, int] = {}
    for row in own_scores:
        score_types[row.score_type] = score_types.get(row.score_type, 0) + 1

    return {
        "n_questions": len(questions),
        "n_resolved_questions": sum(1 for q in questions if q.is_resolved),
        "n_forecasts_total_rows": len(forecasts),
        "n_own_forecasts": len(own),
        "n_questions_with_own_forecast": len({f.question_id for f in own}),
        "n_scored_forecasts": len(own_scores),
        "own_score_types": score_types,
        "n_aggregate_rows": sum(1 for f in forecasts if f.forecaster_id is None),
        "aggregation_methods_present": sorted(
            {f.forecaster_username for f in forecasts if f.forecaster_id is None and f.forecaster_username}
        ),
        "question_types": by_type,
        "date_range": date_range,
    }


def build_limitations(summary: dict[str, Any], fetch_log: list[dict[str, Any]]) -> list[str]:
    limitations: list[str] = []
    if "geometric_mean" not in summary.get("aggregation_methods_present", []):
        limitations.append(
            "No geometric_mean aggregate rows: the spot peer denominator is "
            "absent, so the competition metric cannot be reproduced exactly "
            "from this dataset. Log score and coverage are exact; peer score is "
            "UNAVAILABLE."
        )
    if not summary.get("n_scored_forecasts"):
        limitations.append(
            "No per-question scores for this account: either nothing has "
            "resolved yet, or scores were not requested. Without them there is "
            "no ground truth to validate the offline scorer against."
        )
    if any(entry.get("status", "").startswith("error") for entry in fetch_log):
        limitations.append(
            "At least one download chunk failed; this dataset is incomplete. "
            "See fetch_log.json."
        )
    limitations.append(
        "Question-level spot_scoring_time overrides are not exported by "
        "Metaculus. Where one exists our derived spot instant is wrong; the "
        "validator detects this via the Coverage column."
    )
    limitations.append(
        "Metaculus Terms of Use govern this data: it may not be used to train "
        "or evaluate AI models without prior written permission. Keep it "
        "local; do not commit or redistribute it."
    )
    return limitations


def cmd_fetch(args: argparse.Namespace) -> int:
    token = read_token(args)
    client = MetaculusReadClient(token, base_url=args.base_url)

    print("Identifying account ...")
    account = probe(client)
    user_id = account.get("user_id")
    print("  user_id={0} username={1} is_bot={2}".format(
        user_id, account.get("username"), account.get("is_bot")
    ))
    access = account.get("data_access_status")
    print("  data access: {0}".format(json.dumps(access)[:300]))

    if args.check:
        print("\n--check: authentication works and the account is identified.")
        print("No files written. Re-run without --check to build a dataset.")
        return 0

    if user_id is None:
        raise SystemExit("could not determine our own user id; refusing to guess")

    statuses = list(CLOSED_STATUSES)
    if args.include_open:
        statuses.append("open")
        print(
            "\n!! --include-open requested. Open competition questions will be\n"
            "   included. Do NOT use the resulting forecasts to inform changes\n"
            "   to the bot: that is human-in-the-loop and against tournament\n"
            "   rules.\n"
        )

    print("\nListing posts this account has forecast on ...")
    posts = client.get_posts_forecasted_by(user_id, statuses=statuses)
    post_ids = sorted({post["id"] for post in posts if post.get("id") is not None})
    print("  {0} posts".format(len(post_ids)))

    if not post_ids:
        print(
            "\nThis account has no forecasts on questions with status "
            + ",".join(statuses)
            + ".\nNothing to build a track record from yet."
        )
        return 1

    print("\nFetching the question universe for coverage ...")
    universe = fetch_universe(client, args.tournaments, statuses=statuses)

    print("\nDownloading own forecast/score data ...")
    texts, fetch_log = download_all(client, post_ids, try_geometric_mean=not args.no_geometric_mean)

    question_text = merge_csv_texts(texts[QUESTION_CSV])
    forecast_text = merge_csv_texts(texts[FORECAST_CSV])
    score_text = merge_csv_texts(texts[SCORE_CSV])
    if not question_text:
        raise SystemExit("no question data returned; refusing to write an empty dataset")

    summary = summarise(question_text, forecast_text, score_text, user_id)

    created_at = utc_now_iso()
    staging = tempfile.mkdtemp(prefix="metac-dataset-")
    try:
        records: list[FileRecord] = []
        for name, text in (
            (QUESTION_CSV, question_text),
            (FORECAST_CSV, forecast_text),
            (SCORE_CSV, score_text),
        ):
            if not text:
                continue
            path = os.path.join(staging, name)
            with open(path, "w", newline="") as handle:
                handle.write(text)
            rows = max(0, len(text.splitlines()) - 1)
            records.append(FileRecord.from_path(path, rows=rows))

        universe_path = os.path.join(staging, UNIVERSE_FILE)
        with open(universe_path, "w") as handle:
            json.dump(universe, handle, indent=1, sort_keys=True)
        records.append(FileRecord.from_path(universe_path))

        account_path = os.path.join(staging, ACCOUNT_FILE)
        with open(account_path, "w") as handle:
            json.dump(account, handle, indent=2, sort_keys=True)
        records.append(FileRecord.from_path(account_path))

        log_path = os.path.join(staging, FETCH_LOG_FILE)
        with open(log_path, "w") as handle:
            json.dump({"chunks": fetch_log, "api_calls": client.call_log}, handle, indent=2)
        records.append(FileRecord.from_path(log_path))

        digest = content_digest(records)
        dataset_id = make_dataset_id(DATASET_KIND, created_at, digest)
        dataset_dir = os.path.join(args.out, dataset_id)
        if os.path.exists(dataset_dir):
            raise SystemExit(
                "dataset {0} already exists; datasets are immutable".format(dataset_dir)
            )
        os.makedirs(args.out, exist_ok=True)
        shutil.copytree(staging, dataset_dir)

        manifest = Manifest(
            dataset_id=dataset_id,
            kind=DATASET_KIND,
            created_at=created_at,
            tool={
                "name": "research/fetch_own_track_record.py",
                "version": __version__,
                "python": sys.version.split()[0],
            },
            source={
                "base_url": args.base_url,
                "endpoints": [
                    "GET /users/me/",
                    "GET /get-data-access-status/",
                    "GET /posts/ (forecaster_id, tournaments)",
                    "GET /data/download/",
                ],
                "statuses_requested": statuses,
                "tournaments_for_universe": args.tournaments,
                "n_posts_requested": len(post_ids),
                "read_only": True,
            },
            account={
                "user_id": user_id,
                "username": account.get("username"),
                "is_bot": account.get("is_bot"),
                "bot_id": args.bot_id,
                "data_access_status": account.get("data_access_status"),
            },
            request={
                "include_user_data": True,
                "include_scores": True,
                "include_comments": False,
                "aggregation_methods_attempted": (
                    None if args.no_geometric_mean else "geometric_mean"
                ),
                "chunk_size": DOWNLOAD_POST_CHUNK,
            },
            git=git_info(REPO_ROOT),
            files=records,
            summary=summary,
            limitations=build_limitations(summary, fetch_log),
        )
        write_manifest(dataset_dir, manifest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print("\nDataset written: {0}".format(dataset_dir))
    print(json.dumps(summary, indent=2))
    problems = verify_dataset(dataset_dir)
    print("\nintegrity check: {0}".format("OK" if not problems else problems))
    for note in manifest.limitations:
        print("  limitation: {0}".format(note))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Probe the download endpoint's parameter space against a single post.

    Written after the first real run returned 403 twice and the script reported
    only "no question data returned". Guessing which parameter the gateway
    dislikes would cost one push and one workflow run per guess; one matrix
    answers it in a single run. Prints statuses only -- no question content.
    """
    token = read_token(args)
    client = MetaculusReadClient(token, base_url=args.base_url)

    account = probe(client)
    user_id = account.get("user_id")
    print("account   : user_id={0} username={1}".format(user_id, account.get("username")))
    print("data tier : {0}".format(json.dumps(account.get("data_access_status"))))

    posts = client.get_posts_forecasted_by(user_id, statuses=CLOSED_STATUSES)
    print("\nposts we have forecast on (closed/resolved): {0}".format(len(posts)))
    if not posts:
        print("nothing to probe with")
        return 1

    # What the posts listing itself already gives us. If it carries our own
    # forecast history, the download endpoint is not the only possible source.
    sample = posts[0]
    question = sample.get("question") or {}
    print("\nshape of a post we forecast on (keys only, no content):")
    print("  post keys     : {0}".format(sorted(sample.keys())))
    print("  question keys : {0}".format(sorted(question.keys())))
    for key in ("my_forecasts", "aggregations", "resolution", "type", "status"):
        present = key in question
        print("  question.{0:<14} present={1}".format(key, present))
    my_forecasts = question.get("my_forecasts")
    if isinstance(my_forecasts, dict):
        print("  my_forecasts keys: {0}".format(sorted(my_forecasts.keys())))
        history = my_forecasts.get("history")
        print("  my_forecasts.history entries: {0}".format(
            len(history) if isinstance(history, list) else "n/a"
        ))
        if isinstance(history, list) and history:
            print("  history[0] keys: {0}".format(sorted(history[0].keys())))
        scores = my_forecasts.get("scores")
        if scores:
            print("  my_forecasts.scores entries: {0}".format(len(scores)))
            if isinstance(scores, list) and scores:
                print("  scores[0] keys: {0}".format(sorted(scores[0].keys())))

    post_id = sample.get("id")
    question_id = question.get("id")

    print("\n/api/data/download/ parameter matrix (post_id={0}):".format(post_id))
    matrix: list[tuple[str, dict[str, Any]]] = [
        ("post_ids", {"post_ids": [post_id]}),
        ("question_id", {"question_id": question_id}),
        ("project_id 33022 (FE summer)", {"project_id": 33022}),
        ("project_id + user data + scores", {
            "project_id": 33022, "include_user_data": True, "include_scores": True,
        }),
        ("project_id + geometric_mean", {
            "project_id": 33022, "include_user_data": True, "include_scores": True,
            "aggregation_methods": "geometric_mean",
        }),
        ("project_id of this post", {"project_id": sample.get("projects", {}).get("default_project", {}).get("id")}),
    ]
    for label, params in matrix:
        if any(value is None for value in params.values()):
            print("  {0:<34} skipped (no id available)".format(label))
            continue
        try:
            body, headers = client.get_bytes("/data/download/", params)
            if body[:2] == b"PK":
                import io as _io
                import zipfile as _zip

                with _zip.ZipFile(_io.BytesIO(body)) as archive:
                    names = archive.namelist()
                    sizes = {n: archive.getinfo(n).file_size for n in names}
                print("  {0:<34} 200  ZIP {1} bytes  {2}".format(
                    label, len(body), json.dumps(sizes)
                ))
            else:
                print("  {0:<34} 200  {1}".format(
                    label, (headers.get("Content-Type") or "?")
                ))
        except MetaculusReadError as exc:
            print("  {0:<34} {1}  {2}".format(
                label, exc.status, (exc.body or "").replace("\n", " ")[:150]
            ))

    print("\npost detail endpoint /api/posts/{0}/ :".format(post_id))
    try:
        detail = client.get_json("/posts/{0}/".format(post_id))
        detail_question = detail.get("question") or {}
        print("  question keys : {0}".format(sorted(detail_question.keys())))
        mine = detail_question.get("my_forecasts")
        if isinstance(mine, dict):
            print("  my_forecasts keys : {0}".format(sorted(mine.keys())))
            for field in ("history", "latest", "score_data"):
                value = mine.get(field)
                if isinstance(value, list):
                    print("    {0}: list of {1}".format(field, len(value)))
                    if value and isinstance(value[0], dict):
                        print("      [0] keys: {0}".format(sorted(value[0].keys())))
                elif isinstance(value, dict):
                    print("    {0} keys: {1}".format(field, sorted(value.keys())))
        else:
            print("  my_forecasts : absent")
        aggregations = detail_question.get("aggregations") or {}
        print("  aggregations methods : {0}".format(sorted(aggregations.keys())))
        for method, block in aggregations.items():
            if isinstance(block, dict):
                populated = {k: (len(v) if isinstance(v, list) else bool(v)) for k, v in block.items()}
                print("    {0}: {1}".format(method, json.dumps(populated)))
    except MetaculusReadError as exc:
        print("  {0}  {1}".format(exc.status, (exc.body or "")[:150]))

    print("\nleaderboard /api/leaderboards/project/33022/ :")
    try:
        board = client.get_json("/leaderboards/project/33022/")
        if isinstance(board, dict):
            print("  keys: {0}".format(sorted(board.keys())))
            entries = board.get("entries")
            if isinstance(entries, list):
                print("  entries: {0}".format(len(entries)))
                mine = [e for e in entries if (e.get("user") or {}).get("id") == user_id]
                if entries and isinstance(entries[0], dict):
                    print("  entry keys: {0}".format(sorted(entries[0].keys())))
                print("  our entry present: {0}".format(bool(mine)))
    except MetaculusReadError as exc:
        print("  {0}  {1}".format(exc.status, (exc.body or "")[:150]))

    print("\nno question content was printed by this probe")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    problems = verify_dataset(args.verify)
    if problems:
        print("INTEGRITY FAILURE")
        for problem in problems:
            print("  - {0}".format(problem))
        return 1
    print("dataset verified: every file matches the manifest hashes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_DATASET_ROOT, help="dataset root directory")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token-file", default=None, help="file containing the Metaculus token")
    parser.add_argument("--bot-id", default=306913, type=int, help="recorded in the manifest for traceability")
    parser.add_argument(
        "--tournaments",
        nargs="*",
        default=DEFAULT_UNIVERSE_TOURNAMENTS,
        help="tournaments whose questions form the coverage denominator",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="probe authentication and data access, write nothing",
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="ALSO include still-open questions (human-in-the-loop risk; off by default)",
    )
    parser.add_argument(
        "--no-geometric-mean",
        action="store_true",
        help="skip the geometric_mean aggregation attempt",
    )
    parser.add_argument("--verify", default=None, help="verify an existing dataset directory and exit")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="probe the download endpoint's parameter space and exit (prints statuses only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        return cmd_verify(args)
    if args.diagnose:
        return cmd_diagnose(args)
    return cmd_fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
