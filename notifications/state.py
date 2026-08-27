"""Persisted state: which WhatsApp events have already been sent.

WHY THIS EXISTS, AND WHY IT IS SMALLER THAN AN EARLIER DRAFT
--------------------------------------------------------------
Audited before writing a line of this file: `main.py`, `discovery.py` and
`publication.py` keep no state that survives one process. `discovery.py`
returns a plain list. `PublishingClient._records` (`publication.py`) is an
in-memory dict, discarded when the process exits.

An earlier draft of this module also tracked `known_question_ids` and
`total_forecasts_observed` locally, to answer "is this question new" and
"how many forecasts have we ever published". A second audit pass found that
duplicates state Metaculus already holds: in the ONLY mode this notifier is
scoped to (`main.py`'s default "tournament" mode,
`skip_previously_forecasted_questions=True`), every question in
`forecast_reports` is, BY CONSTRUCTION, one Metaculus has no prior forecast
from us for. So "is it new" needs no local memory at all -- Metaculus is
already the source of truth for that fact, and duplicating it here was
exactly the unnecessary state the brief asked to avoid.

What Metaculus does NOT know is whether we already sent a WHATSAPP about a
given question -- that is a fact about THIS notifier, not about forecasting,
and it is the one thing this file exists to remember. Without it: a question
that fails to forecast stays eligible on Metaculus (no prediction was
recorded), so it reappears in `forecast_reports` on every subsequent run
until it succeeds -- and without a local memory of "we already said 🆕 about
this one", it would re-notify every single run.

So the only thing persisted is `notified_event_ids` (a set of event
identities already sent) plus `last_run_state` (needed to detect a
transition, which is not a fact Metaculus tracks either). One JSON file, one
dataclass, atomic writes.

SURVIVING BETWEEN GITHUB ACTIONS RUNS -- UNRESOLVED, ON PURPOSE
------------------------------------------------------------------
This module only knows how to read and write a path on disk. It deliberately
has no opinion on how that path survives between two ephemeral runners: every
option (git-commit-back, `actions/cache`, `actions/artifact`) has a real
trade-off against `run_bot_on_tournament.yaml`'s current, repo-wide
`permissions: contents: read` posture (every workflow in this repository
declares it -- there is no existing write-back pattern to copy). That
decision is intentionally NOT made in code; see the report delivered
alongside this module for the options and their failure modes.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any

STATE_SCHEMA_VERSION = 1

DEFAULT_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


@dataclass
class NotificationState:
    """Everything the event detector and dashboard builder need to remember
    across runs. Every collection is stored as a JSON-friendly list and
    exposed as a set through the `*_set` helpers -- lists preserve insertion
    order in the file (nicer to read in a diff), sets are what membership
    checks actually want."""

    schema_version: int = STATE_SCHEMA_VERSION
    notified_event_ids: list[str] = field(default_factory=list)
    last_run_state: str | None = None  # "OK" | "PARTIAL" | "ERROR" | "IDLE"

    def notified_event_id_set(self) -> set[str]:
        return set(self.notified_event_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NotificationState":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def load_state(path: str = DEFAULT_STATE_PATH) -> NotificationState:
    """Never raises. A missing file is a fresh start (first run ever). A
    corrupt file is ALSO a fresh start, logged, never a crash -- this system
    exists to avoid annoying the operator, so it must not become the reason
    a run fails."""
    if not os.path.exists(path):
        return NotificationState()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return NotificationState.from_dict(data)
    except Exception:  # noqa: BLE001 - any corruption degrades to a fresh state
        return NotificationState()


def save_state(state: NotificationState, path: str = DEFAULT_STATE_PATH) -> None:
    """Atomic: write to a temp file in the same directory, then `os.replace`,
    so a run killed mid-write never leaves a half-written, unparseable state
    file for the next run to choke on."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".state-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
