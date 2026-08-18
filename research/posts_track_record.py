"""Build a track record from the posts API instead of the CSV download.

Why this exists
---------------
``/api/data/download/`` is closed to an ordinary bot account. Probed directly
(see ``--diagnose``), every form is refused:

    post_ids / post_id / question_id
        403 "This endpoint is restricted to project-scoped exports, so the data
        must be selected by project alone."
    project_id
        403 "You can only export data for projects you've been granted access
        to."

and ``/api/get-data-access-status/`` returns ``has_data_access: false``, so
there is no project we may export. That path needs the Bot Benchmarking tier.

The posts API, however, gives an account its own data, and the question payload
turned out to carry more than expected:

    /api/posts/?forecaster_id=<us>   resolution, spot_scoring_time,
                                     question_weight, type, timing fields
    /api/posts/<id>/                 my_forecasts.history: our own forecasts
                                     with start_time, end_time, forecast_values

That is everything the spot rule needs except the peer denominator. Notably it
includes ``spot_scoring_time`` *explicitly*, which the CSV export omits -- so
this route is strictly more accurate about the scoring instant than the one it
replaces.

Design choice
-------------
Rather than teach the scorer a second input format, this module emits the exact
same CSV schema the lab already reads. Every downstream module -- scorer,
validator, coverage, exporter -- works unchanged, and the two sources stay
interchangeable if the download endpoint ever opens up.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from research.track_record import CONTINUOUS_TYPES

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
    # Additive column the CSV export does not have. The reader treats it as
    # optional, so datasets from either source stay readable.
    "Spot Scoring Time",
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


def resolution_bucket_index(question: dict[str, Any]) -> int | None:
    """Index into ``forecast_values`` of the outcome that actually happened.

    Mirrors the server's ``string_location_to_bucket_index`` for the cases we
    can do safely:

    - binary: ``forecast_values`` is ``[P(no), P(yes)]``.
    - multiple choice: the index of the resolved option in ``options``.

    Continuous questions return ``None``. Their bucket index depends on the
    question's ``scaling`` (range_min/range_max/zero_point, log or linear) and
    on out-of-bounds handling, and getting that subtly wrong would produce a
    plausible-looking but wrong probability -- worse than reporting nothing.
    The lab already handles a missing value by naming it as missing.
    """
    resolution = (question.get("resolution") or "").strip()
    if not resolution or resolution.lower() in ("annulled", "ambiguous"):
        return None

    qtype = question.get("type")
    if qtype == "binary":
        lowered = resolution.lower()
        if lowered == "yes":
            return 1
        if lowered == "no":
            return 0
        return None

    if qtype == "multiple_choice":
        options = question.get("options") or []
        if resolution in options:
            return options.index(resolution)
        return None

    return None


def probability_of_resolution(
    question: dict[str, Any], forecast_values: Any
) -> float | None:
    """The probability mass our forecast put on the resolved outcome.

    This is the quantity the server calls ``Probability of Resolution`` and
    uses for scoring. Computing it ourselves is only safe where the bucket
    index is unambiguous, which is why continuous questions yield ``None``.
    """
    index = resolution_bucket_index(question)
    if index is None:
        return None
    if not isinstance(forecast_values, (list, tuple)):
        return None
    if index >= len(forecast_values):
        return None
    value = forecast_values[index]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _question_row(post: dict[str, Any], question: dict[str, Any]) -> list[Any]:
    projects = post.get("projects") or {}
    default_project = projects.get("default_project") or {}
    scaling = question.get("scaling") or {}
    return [
        question.get("id"),
        "https://www.metaculus.com/questions/{0}/".format(post.get("id")),
        question.get("title") or post.get("title") or "",
        post.get("id"),
        post.get("curation_status"),
        post.get("published_at"),
        default_project.get("name"),
        default_project.get("id"),
        [],
        [],
        question.get("label"),
        question.get("type"),
        question.get("options"),
        question.get("all_options_ever"),
        question.get("options_history"),
        scaling.get("range_min"),
        question.get("open_lower_bound"),
        scaling.get("range_max"),
        question.get("open_upper_bound"),
        None,
        question.get("open_time"),
        question.get("cp_reveal_time"),
        question.get("scheduled_close_time"),
        question.get("actual_close_time"),
        question.get("resolution"),
        question.get("actual_resolve_time") or question.get("resolution_set_time"),
        question.get("include_bots_in_aggregates"),
        question.get("question_weight"),
        question.get("spot_scoring_time"),
    ]


def _forecast_rows(
    question: dict[str, Any], user_id: int, username: str | None
) -> list[list[Any]]:
    mine = question.get("my_forecasts") or {}
    history = mine.get("history")
    if not isinstance(history, list):
        return []

    rows: list[list[Any]] = []
    qtype = question.get("type")
    for record in history:
        if not isinstance(record, dict):
            continue
        values = record.get("forecast_values")
        p_resolution = probability_of_resolution(question, values)

        probability_yes = None
        per_category = None
        continuous_cdf = None
        if qtype == "binary" and isinstance(values, (list, tuple)) and len(values) >= 2:
            probability_yes = values[1]
        elif qtype == "multiple_choice":
            per_category = values
        elif qtype in CONTINUOUS_TYPES:
            continuous_cdf = values

        rows.append(
            [
                question.get("id"),
                record.get("author_id") or user_id,
                username,
                True,
                record.get("start_time"),
                record.get("end_time"),
                None,
                probability_yes,
                per_category,
                continuous_cdf,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                p_resolution,
                None,
            ]
        )
    return rows


def _write_csv(header: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if value is None else value for value in row])
    return buffer.getvalue()


def build_csvs(
    post_details: list[dict[str, Any]], user_id: int, username: str | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Turn post-detail payloads into ``question_data.csv`` / ``forecast_data.csv``.

    Returns the two CSV texts plus a stats dict recording what could and could
    not be derived, so the gaps end up in the dataset manifest rather than
    being discovered later as a silent zero.
    """
    question_rows: list[list[Any]] = []
    forecast_rows: list[list[Any]] = []
    stats: dict[str, Any] = {
        "n_posts": len(post_details),
        "n_questions": 0,
        "n_with_my_forecasts": 0,
        "n_forecast_records": 0,
        "n_with_probability_of_resolution": 0,
        "n_resolved": 0,
        "unsupported_probability_types": {},
        "n_with_spot_scoring_time": 0,
    }

    for post in post_details:
        question = post.get("question")
        if not isinstance(question, dict):
            # Group and conditional posts have no single question; they are
            # counted in coverage from the listing, not scored here.
            continue
        stats["n_questions"] += 1
        if question.get("spot_scoring_time"):
            stats["n_with_spot_scoring_time"] += 1
        resolution = (question.get("resolution") or "").strip().lower()
        if resolution and resolution not in ("annulled", "ambiguous"):
            stats["n_resolved"] += 1

        question_rows.append(_question_row(post, question))
        rows = _forecast_rows(question, user_id, username)
        if rows:
            stats["n_with_my_forecasts"] += 1
        stats["n_forecast_records"] += len(rows)
        for row in rows:
            if row[17] is not None:
                stats["n_with_probability_of_resolution"] += 1
            elif resolution:
                qtype = question.get("type") or "unknown"
                stats["unsupported_probability_types"][qtype] = (
                    stats["unsupported_probability_types"].get(qtype, 0) + 1
                )
        forecast_rows.extend(rows)

    return (
        _write_csv(QUESTION_HEADER, question_rows),
        _write_csv(FORECAST_HEADER, forecast_rows),
        stats,
    )
