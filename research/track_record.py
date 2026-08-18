"""Typed reader for the CSVs inside a Metaculus data-download ZIP.

Column names and value formats are transcribed from the Metaculus server's own
exporter (``utils/csv_utils.py::generate_data``), not inferred from a sample
file. If Metaculus renames a column this module raises immediately with the
missing name rather than silently producing ``None`` everywhere -- the previous
backtest harness lost four commits to exactly that failure mode.

Three tables matter:

``question_data.csv``  one row per question, with the timing fields the spot
                       score depends on (Open Time, CP Reveal Time, Actual
                       Close Time) and ``Question Weight``.
``forecast_data.csv``  one row per forecast *interval*. Our own forecasts have
                       a ``Forecaster ID``; aggregate rows have a null
                       ``Forecaster ID`` and carry the aggregation method's
                       name in ``Forecaster Username``.
``score_data.csv``     one row per (question, user, score type). This is the
                       ground truth we validate against: Metaculus' own
                       ``spot_peer`` value and its ``Coverage``.
"""

from __future__ import annotations

import ast
import csv
import io
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

# Mirrors questions/models.py::QUESTION_CONTINUOUS_TYPES
CONTINUOUS_TYPES = frozenset({"numeric", "date", "discrete"})
ALL_QUESTION_TYPES = frozenset({"binary", "multiple_choice", "numeric", "date", "discrete"})

QUESTION_CSV = "question_data.csv"
FORECAST_CSV = "forecast_data.csv"
SCORE_CSV = "score_data.csv"
README_FILE = "README.md"

# Cancelled/void outcomes. Metaculus stores these in the same `resolution`
# field as real outcomes; treating them as data would silently score
# un-scoreable questions.
NON_RESOLUTIONS = frozenset({"annulled", "ambiguous", ""})


class SchemaError(RuntimeError):
    """A CSV did not have the columns the server is documented to emit."""


# --------------------------------------------------------------------- values


def parse_dt(value: Any) -> datetime | None:
    """Parse the timestamps Django's csv writer emits.

    ``csv.writer`` stringifies a ``datetime`` with ``str()``, giving
    ``'2026-08-18 11:04:49.472000+00:00'``. The API elsewhere emits ISO-8601
    with a ``Z``. Both are accepted; anything naive is assumed UTC, because
    every timestamp Metaculus stores is UTC and guessing local time here would
    silently shift spot-score windows by hours.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "nan"):
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "nan", ""):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def parse_list(value: Any) -> list[Any] | None:
    """Python list literals -- the exporter writes lists through ``str()``,
    producing ``"[0.1, 0.2]"``. ``literal_eval`` only accepts literals, so a
    malformed cell cannot execute anything."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return None


def normalise_resolution(value: Any) -> str | None:
    """``None`` for anything that is not a real outcome, so callers cannot
    accidentally score an annulled question."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in NON_RESOLUTIONS or text.lower() in ("none", "null"):
        return None
    return text


# --------------------------------------------------------------------- models


@dataclass
class QuestionRow:
    question_id: int
    post_id: int | None
    title: str
    url: str
    question_type: str
    default_project: str | None
    default_project_id: int | None
    open_time: datetime | None
    cp_reveal_time: datetime | None
    scheduled_close_time: datetime | None
    actual_close_time: datetime | None
    resolution: str | None
    resolution_known_time: datetime | None
    include_bots_in_aggregates: bool | None
    question_weight: float | None
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def is_continuous(self) -> bool:
        return self.question_type in CONTINUOUS_TYPES

    @property
    def is_resolved(self) -> bool:
        return self.resolution is not None

    def spot_scoring_time(self) -> datetime | None:
        """Reimplementation of ``Question.get_spot_scoring_time()``.

        Server source (``questions/models.py``)::

            if self.spot_scoring_time:            return self.spot_scoring_time
            elif cp_reveal_time and open_time and cp_reveal_time > open_time:
                                                  return self.cp_reveal_time
            elif self.actual_close_time:          return self.actual_close_time
            elif self.scheduled_close_time:       return self.scheduled_close_time
            return None

        CAVEAT, and it is a real one: the first branch reads a per-question
        ``spot_scoring_time`` override which the CSV export does **not**
        contain. If a question carries an override we silently fall through to
        the CP-reveal rule and get the wrong instant. This is why the validator
        checks reproduced coverage against Metaculus' own Coverage column
        rather than trusting this function.
        """
        if self.cp_reveal_time and self.open_time and self.cp_reveal_time > self.open_time:
            return self.cp_reveal_time
        if self.actual_close_time:
            return self.actual_close_time
        return self.scheduled_close_time

    def spot_timestamp(self) -> float | None:
        """``min(spot_scoring_time, actual_close_time)`` -- the clamp applied in
        ``scoring/score_math.py::evaluate_question``."""
        spot = self.spot_scoring_time()
        if spot is None:
            return None
        value = spot.timestamp()
        if self.actual_close_time:
            value = min(value, self.actual_close_time.timestamp())
        return value


@dataclass
class ForecastRow:
    question_id: int
    forecaster_id: int | None
    forecaster_username: str | None
    is_bot: bool | None
    start_time: datetime | None
    end_time: datetime | None
    forecaster_count: int | None
    probability_yes: float | None
    probability_yes_per_category: list[Any] | None
    continuous_cdf: list[float] | None
    probability_of_resolution: float | None
    pdf_at_resolution: float | None
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def is_aggregate(self) -> bool:
        """Aggregate rows are written with a null Forecaster ID and the method
        name in the username column (``csv_utils.py``: ``row.extend([None,
        aggregate_forecast.method])``)."""
        return self.forecaster_id is None

    @property
    def aggregation_method(self) -> str | None:
        return self.forecaster_username if self.is_aggregate else None

    def is_active_at(self, timestamp: float) -> bool:
        """``start <= t < end`` with an open-ended end, matching
        ``evaluate_forecasts_peer_spot_forecast`` exactly."""
        if self.start_time is None:
            return False
        start = self.start_time.timestamp()
        end = float("inf") if self.end_time is None else self.end_time.timestamp()
        return start <= timestamp < end


@dataclass
class ScoreRow:
    question_id: int
    user_id: int | None
    user_username: str | None
    score_type: str
    score: float | None
    coverage: float | None

    @property
    def is_aggregate(self) -> bool:
        return self.user_id is None


@dataclass
class TrackRecord:
    questions: dict[int, QuestionRow]
    forecasts: list[ForecastRow]
    scores: list[ScoreRow]
    source_dir: str | None = None

    # ------------------------------------------------------------- selectors

    def own_forecasts(self, user_id: int | None = None) -> list[ForecastRow]:
        rows = [f for f in self.forecasts if not f.is_aggregate]
        if user_id is not None:
            rows = [f for f in rows if f.forecaster_id == user_id]
        return rows

    def aggregate_forecasts(self, method: str | None = None) -> list[ForecastRow]:
        rows = [f for f in self.forecasts if f.is_aggregate]
        if method is not None:
            rows = [f for f in rows if f.aggregation_method == method]
        return rows

    def own_scores(self, user_id: int | None = None, score_type: str | None = None) -> list[ScoreRow]:
        rows = [s for s in self.scores if not s.is_aggregate]
        if user_id is not None:
            rows = [s for s in rows if s.user_id == user_id]
        if score_type is not None:
            rows = [s for s in rows if s.score_type == score_type]
        return rows

    def forecasts_by_question(self, user_id: int | None = None) -> dict[int, list[ForecastRow]]:
        out: dict[int, list[ForecastRow]] = {}
        for row in self.own_forecasts(user_id=user_id):
            out.setdefault(row.question_id, []).append(row)
        for rows in out.values():
            rows.sort(key=lambda r: (r.start_time or datetime.min.replace(tzinfo=timezone.utc)))
        return out

    def infer_user_id(self) -> int | None:
        """The forecaster id appearing on the most rows.

        Only used as a fallback when the caller did not record its own account
        id; a dataset produced by ``fetch_own_track_record.py`` always states it
        explicitly in the manifest.
        """
        counts: dict[int, int] = {}
        for row in self.forecasts:
            if row.forecaster_id is None:
                continue
            counts[row.forecaster_id] = counts.get(row.forecaster_id, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------- readers


def _require(header: list[str], needed: Iterable[str], filename: str) -> None:
    missing = [name for name in needed if name not in header]
    if missing:
        raise SchemaError(
            "{0} is missing expected column(s) {1}; got {2}".format(
                filename, missing, header
            )
        )


def read_question_csv(text: str) -> list[QuestionRow]:
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    _require(
        header,
        ["Question ID", "Question Type", "Resolution", "Open Time", "Scheduled Close Time"],
        QUESTION_CSV,
    )
    rows: list[QuestionRow] = []
    for raw in reader:
        question_id = parse_int(raw.get("Question ID"))
        if question_id is None:
            continue
        rows.append(
            QuestionRow(
                question_id=question_id,
                post_id=parse_int(raw.get("Post ID")),
                title=(raw.get("Question Title") or "").strip(),
                url=(raw.get("Question URL") or "").strip(),
                question_type=(raw.get("Question Type") or "").strip(),
                default_project=(raw.get("Default Project") or "").strip() or None,
                default_project_id=parse_int(raw.get("Default Project ID")),
                open_time=parse_dt(raw.get("Open Time")),
                cp_reveal_time=parse_dt(raw.get("CP Reveal Time")),
                scheduled_close_time=parse_dt(raw.get("Scheduled Close Time")),
                actual_close_time=parse_dt(raw.get("Actual Close Time")),
                resolution=normalise_resolution(raw.get("Resolution")),
                resolution_known_time=parse_dt(raw.get("Resolution Known Time")),
                include_bots_in_aggregates=parse_bool(raw.get("Include Bots in Aggregates")),
                question_weight=parse_float(raw.get("Question Weight")),
                raw=dict(raw),
            )
        )
    return rows


def read_forecast_csv(text: str) -> list[ForecastRow]:
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    _require(header, ["Question ID", "Start Time", "Probability of Resolution"], FORECAST_CSV)
    anonymized = "Forecaster (Anonymized)" in header
    rows: list[ForecastRow] = []
    for raw in reader:
        question_id = parse_int(raw.get("Question ID"))
        if question_id is None:
            continue
        if anonymized:
            # An anonymised export cannot be attributed to our account, which
            # makes it useless for a *track record*. Surface it loudly instead
            # of quietly analysing someone else's numbers.
            raise SchemaError(
                "forecast_data.csv is anonymised; a track record needs "
                "attributable rows. Re-fetch without anonymisation."
            )
        rows.append(
            ForecastRow(
                question_id=question_id,
                forecaster_id=parse_int(raw.get("Forecaster ID")),
                forecaster_username=(raw.get("Forecaster Username") or "").strip() or None,
                is_bot=parse_bool(raw.get("Is Bot")),
                start_time=parse_dt(raw.get("Start Time")),
                end_time=parse_dt(raw.get("End Time")),
                forecaster_count=parse_int(raw.get("Forecaster Count")),
                probability_yes=parse_float(raw.get("Probability Yes")),
                probability_yes_per_category=parse_list(raw.get("Probability Yes Per Category")),
                continuous_cdf=parse_list(raw.get("Continuous CDF")),
                probability_of_resolution=parse_float(raw.get("Probability of Resolution")),
                pdf_at_resolution=parse_float(raw.get("PDF at Resolution")),
                raw=dict(raw),
            )
        )
    return rows


def read_score_csv(text: str) -> list[ScoreRow]:
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    _require(header, ["Question ID", "Score Type", "Score"], SCORE_CSV)
    rows: list[ScoreRow] = []
    for raw in reader:
        question_id = parse_int(raw.get("Question ID"))
        if question_id is None:
            continue
        rows.append(
            ScoreRow(
                question_id=question_id,
                user_id=parse_int(raw.get("User ID")),
                user_username=(raw.get("User Username") or "").strip() or None,
                score_type=(raw.get("Score Type") or "").strip(),
                score=parse_float(raw.get("Score")),
                coverage=parse_float(raw.get("Coverage")),
            )
        )
    return rows


def load_track_record(dataset_dir: str) -> TrackRecord:
    """Load the three CSVs from an unpacked dataset directory.

    ``score_data.csv`` is optional: the server omits it when
    ``include_scores=false``, and a dataset without scores is still a valid
    forecast record -- it just cannot be validated against Metaculus.
    """
    questions: dict[int, QuestionRow] = {}
    question_path = os.path.join(dataset_dir, QUESTION_CSV)
    if not os.path.exists(question_path):
        raise FileNotFoundError("no {0} in {1}".format(QUESTION_CSV, dataset_dir))
    with open(question_path, newline="") as handle:
        for row in read_question_csv(handle.read()):
            questions[row.question_id] = row

    forecasts: list[ForecastRow] = []
    forecast_path = os.path.join(dataset_dir, FORECAST_CSV)
    if os.path.exists(forecast_path):
        with open(forecast_path, newline="") as handle:
            forecasts = read_forecast_csv(handle.read())

    scores: list[ScoreRow] = []
    score_path = os.path.join(dataset_dir, SCORE_CSV)
    if os.path.exists(score_path):
        with open(score_path, newline="") as handle:
            scores = read_score_csv(handle.read())

    return TrackRecord(
        questions=questions, forecasts=forecasts, scores=scores, source_dir=dataset_dir
    )


def extract_zip(data: bytes, dest_dir: str) -> list[str]:
    """Unpack a data-download ZIP, refusing path traversal.

    The archive comes from Metaculus, but an unpacker that trusts member names
    is a liability regardless of source.
    """
    os.makedirs(dest_dir, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            safe = os.path.basename(name)
            if not safe or safe != name:
                raise ValueError("unexpected path in archive: {0!r}".format(name))
            target = os.path.join(dest_dir, safe)
            with archive.open(name) as source, open(target, "wb") as sink:
                sink.write(source.read())
            written.append(target)
    return written


def merge_csv_texts(texts: list[str]) -> str:
    """Concatenate CSVs that share a header, keeping one header.

    Needed because ``/api/data/download/`` is scoped per request: a track
    record spanning hundreds of posts arrives as several ZIPs which have to be
    stitched into one table. Raises if the headers disagree, since silently
    merging mismatched schemas would corrupt every downstream number.
    """
    header: str | None = None
    body: list[str] = []
    for text in texts:
        if not text.strip():
            continue
        lines = text.splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise SchemaError(
                "cannot merge CSVs with different headers:\n  {0}\n  {1}".format(header, lines[0])
            )
        body.extend(line for line in lines[1:] if line.strip())
    if header is None:
        return ""
    return "\n".join([header] + body) + "\n"
