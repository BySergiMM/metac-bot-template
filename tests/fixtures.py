"""Synthetic Metaculus exports, built to the server's exact CSV schema.

Column names and orders are copied from ``utils/csv_utils.py::generate_data``
in the Metaculus server source. Keeping the fixtures faithful is the whole
point: a test that passes against an invented schema proves nothing about the
real download.

Values are chosen so the expected scores can be worked out by hand, which is
what lets the scorer tests assert against an independently derived number
rather than against whatever the code happens to produce.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

QUESTION_HEADER = [
    "Question ID",
    "Question URL",
    "Question Title",
    "Post ID",
    "Post Curation Status",
    "Post Published Time",
    "Default Project",
    "Default Project ID",
    "Categories",
    "Leaderboard Tags",
    "Label",
    "Question Type",
    "MC Options (Current)",
    "MC Options (All)",
    "MC Options History",
    "Lower Bound",
    "Open Lower Bound",
    "Upper Bound",
    "Open Upper Bound",
    "Continuous Range",
    "Open Time",
    "CP Reveal Time",
    "Scheduled Close Time",
    "Actual Close Time",
    "Resolution",
    "Resolution Known Time",
    "Include Bots in Aggregates",
    "Question Weight",
]

FORECAST_HEADER = [
    "Question ID",
    "Forecaster ID",
    "Forecaster Username",
    "Is Bot",
    "Start Time",
    "End Time",
    "Forecaster Count",
    "Probability Yes",
    "Probability Yes Per Category",
    "Continuous CDF",
    "Probability Below Lower Bound",
    "Probability Above Upper Bound",
    "5th Percentile",
    "25th Percentile",
    "Median",
    "75th Percentile",
    "95th Percentile",
    "Probability of Resolution",
    "PDF at Resolution",
]

SCORE_HEADER = [
    "Question ID",
    "User ID",
    "User Username",
    "Score Type",
    "Score",
    "Coverage",
]

# Django's csv writer stringifies datetimes like this.
T_OPEN = "2026-06-01 12:00:00+00:00"
T_MID = "2026-06-03 12:00:00+00:00"
T_REVEAL = "2026-06-05 12:00:00+00:00"
T_CLOSE = "2026-06-10 12:00:00+00:00"
T_RESOLVED = "2026-06-11 09:00:00+00:00"

OUR_USER_ID = 987654
OUR_USERNAME = "seergiii-bot"
OTHER_USER_ID = 111111


def _csv(header: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if value is None else value for value in row])
    return buffer.getvalue()


def question_row(
    question_id: int,
    question_type: str = "binary",
    resolution: str | None = "yes",
    open_time: str | None = T_OPEN,
    cp_reveal_time: str | None = T_REVEAL,
    scheduled_close_time: str | None = T_CLOSE,
    actual_close_time: str | None = T_CLOSE,
    weight: float = 1.0,
    post_id: int | None = None,
) -> list[Any]:
    return [
        question_id,
        "https://www.metaculus.com/questions/{0}/q/".format(post_id or question_id),
        "Synthetic question {0}".format(question_id),
        post_id or question_id,
        "approved",
        T_OPEN,
        "Synthetic Tournament",
        33022,
        "[]",
        "[]",
        None,
        question_type,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        open_time,
        cp_reveal_time,
        scheduled_close_time,
        actual_close_time,
        resolution,
        T_RESOLVED if resolution else None,
        True,
        weight,
    ]


def forecast_row(
    question_id: int,
    probability_of_resolution: float | None,
    start_time: str | None = T_OPEN,
    end_time: str | None = None,
    forecaster_id: int | None = OUR_USER_ID,
    forecaster_username: str = OUR_USERNAME,
    probability_yes: float | None = None,
    forecaster_count: int | None = None,
    is_bot: bool | None = True,
) -> list[Any]:
    return [
        question_id,
        forecaster_id,
        forecaster_username,
        is_bot,
        start_time,
        end_time,
        forecaster_count,
        probability_yes,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        probability_of_resolution,
        None,
    ]


def geometric_mean_row(
    question_id: int,
    probability_of_resolution: float,
    forecaster_count: int,
    start_time: str = T_OPEN,
    end_time: str | None = None,
) -> list[Any]:
    """Aggregate rows carry a null Forecaster ID and the method name in the
    username column (``row.extend([None, aggregate_forecast.method])``)."""
    return forecast_row(
        question_id,
        probability_of_resolution,
        start_time=start_time,
        end_time=end_time,
        forecaster_id=None,
        forecaster_username="geometric_mean",
        forecaster_count=forecaster_count,
        is_bot=None,
    )


def score_row(
    question_id: int,
    score: float | None,
    coverage: float | None = 1.0,
    score_type: str = "spot_peer",
    user_id: int | None = OUR_USER_ID,
    user_username: str = OUR_USERNAME,
) -> list[Any]:
    return [question_id, user_id, user_username, score_type, score, coverage]


def questions_csv(rows: list[list[Any]]) -> str:
    return _csv(QUESTION_HEADER, rows)


def forecasts_csv(rows: list[list[Any]]) -> str:
    return _csv(FORECAST_HEADER, rows)


def scores_csv(rows: list[list[Any]]) -> str:
    return _csv(SCORE_HEADER, rows)


def post_json(
    post_id: int,
    question_id: int,
    question_type: str = "binary",
    status: str = "resolved",
) -> dict[str, Any]:
    return {
        "id": post_id,
        "title": "Synthetic post {0}".format(post_id),
        "status": status,
        "question": {
            "id": question_id,
            "type": question_type,
            "open_time": T_OPEN,
            "scheduled_close_time": T_CLOSE,
        },
    }


def group_post_json(post_id: int, question_ids: list[int], question_type: str = "numeric") -> dict[str, Any]:
    return {
        "id": post_id,
        "title": "Synthetic group {0}".format(post_id),
        "status": "resolved",
        "group_of_questions": {
            "questions": [
                {"id": qid, "type": question_type, "label": "sub {0}".format(qid)}
                for qid in question_ids
            ]
        },
    }


def write_dataset(
    directory: str,
    question_rows: list[list[Any]],
    forecast_rows: list[list[Any]],
    score_rows: list[list[Any]] | None = None,
    universe: dict[str, list[dict[str, Any]]] | None = None,
    account: dict[str, Any] | None = None,
) -> str:
    """Write a dataset directory *without* a manifest.

    Manifest creation is the fetch script's job and is tested separately; tests
    that only need parsing/scoring should not have to fake provenance.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "question_data.csv"), "w", newline="") as handle:
        handle.write(questions_csv(question_rows))
    with open(os.path.join(directory, "forecast_data.csv"), "w", newline="") as handle:
        handle.write(forecasts_csv(forecast_rows))
    if score_rows is not None:
        with open(os.path.join(directory, "score_data.csv"), "w", newline="") as handle:
            handle.write(scores_csv(score_rows))
    if universe is not None:
        with open(os.path.join(directory, "tournament_questions.json"), "w") as handle:
            json.dump(universe, handle)
    if account is not None:
        with open(os.path.join(directory, "account.json"), "w") as handle:
            json.dump(account, handle)
    return directory
