"""Which open questions in the prize tournaments are siblings, and whether
the bot's standing forecasts on them already contradict each other.

Why this exists
---------------
futuresearch's production forecaster ends with a "world modeling" pass that
reconciles each forecast with related questions (about 0.005 Brier where it
acts, 2026-07). Ours forecasts every question alone. Before building
anything, this measures what there is to reconcile:

* per tournament, the open posts: single questions and groups, the group
  sizes and sub-question types, and how many groups are ladders (options
  that read as ordered numbers or dates);
* titles that share a company ticker or a leading name (same-entity
  clusters, e.g. Market Pulse's revenue, EPS and share-price questions);
* on ladders of binaries the bot has forecast, how many are not monotone in
  either direction (an incoherence a rule could fix), and on ladders of
  numerics, how many medians run against the options' order.

Titles and options are public; forecasts never leave this module, only the
counts of incoherent groups (the repository and its logs are public).

Read only (MetaculusReadClient raises on any verb but GET); standard
library only, like the rest of research/.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.metaculus_read_api import MetaculusReadClient, MetaculusReadError  # noqa: E402

TOURNAMENTS = (("futureeval", "33121"), ("market_pulse", "market-pulse-26q4"))

NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|[kmbt]n?|billion|million|trillion)?", re.I)
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
TICKER = re.compile(r"\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)")


def option_key(option: str) -> float | None:
    """A sortable value for a group option that reads as a number or a
    month/year; None for a name ("Tesla") or anything else."""
    text = (option or "").strip().lower()
    for i, month in enumerate(MONTHS):
        found = re.search(rf"\b{month}[a-z]*\.?\s+(\d{{4}})\b", text)
        if found:
            return int(found.group(1)) * 12 + i
    found = re.search(r"\b(19|20)\d{2}\b", text)
    if found and not NUMBER.fullmatch(text.replace(found.group(0), "").strip() or "x"):
        return float(found.group(0))
    found = NUMBER.search(text)
    if not found:
        return None
    raw = found.group(0).lower().replace("$", "").replace(",", "").strip()
    scale = 1.0
    for suffix, factor in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("tn", 1e12),
                           ("bn", 1e9), ("t", 1e12), ("b", 1e9), ("m", 1e6), ("k", 1e3), ("%", 1.0)):
        if raw.endswith(suffix):
            raw, scale = raw[: -len(suffix)].strip(), factor
            break
    try:
        return float(raw) * scale
    except ValueError:
        return None


def questions_of(post: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(post.get("question"), dict):
        return [post["question"]]
    group = post.get("group_of_questions") or {}
    return [q for q in (group.get("questions") or []) if isinstance(q, dict)]


def is_ladder(questions: list[dict[str, Any]]) -> bool:
    keys = [option_key(str(q.get("label") or q.get("title") or "")) for q in questions]
    return len(keys) >= 3 and all(k is not None for k in keys) and len(set(keys)) == len(keys)


def latest_values(question: dict[str, Any]) -> list[float] | None:
    latest = ((question.get("my_forecasts") or {}).get("latest")) or {}
    values = latest.get("forecast_values")
    return [float(v) for v in values] if isinstance(values, list) and values else None


def monotone(xs: list[float], tol: float = 0.005) -> bool:
    up = all(b >= a - tol for a, b in zip(xs, xs[1:]))
    down = all(b <= a + tol for a, b in zip(xs, xs[1:]))
    return up or down


def median_of(cdf: list[float]) -> float:
    for i, f in enumerate(cdf):
        if f >= 0.5:
            return i / (len(cdf) - 1)
    return 1.0


def ladder_coherence(questions: list[dict[str, Any]]) -> str:
    """'coherent', 'incoherent' or 'unforecast' for a ladder group, from the
    bot's latest forecasts: binaries must be monotone along the options'
    order (either direction); numerics' medians must not run against it."""
    ordered = sorted(questions, key=lambda q: option_key(str(q.get("label") or q.get("title") or "")))
    vals = [latest_values(q) for q in ordered]
    if sum(v is not None for v in vals) < 3:
        return "unforecast"
    pts = []
    for q, v in zip(ordered, vals):
        if v is None:
            continue
        if q.get("type") == "binary":
            pts.append(v[-1])
        elif q.get("type") in ("numeric", "discrete", "date") and len(v) > 2:
            scaling = q.get("scaling") or {}
            lo, hi = scaling.get("range_min"), scaling.get("range_max")
            try:
                pts.append(float(lo) + (float(hi) - float(lo)) * median_of(v))
            except (TypeError, ValueError):
                return "unforecast"
        else:
            return "unforecast"
    return "coherent" if monotone(pts) else "incoherent"


def entity(title: str) -> str | None:
    """A ticker in parentheses, else the first capitalised name."""
    found = TICKER.search(title or "")
    if found:
        return found.group(1)
    words = re.findall(r"\b([A-Z][A-Za-z&.\-]{2,})\b", title or "")
    skip = {"Will", "What", "How", "When", "Which", "Who", "The", "Before", "After", "Between", "Does"}
    names = [w for w in words if w not in skip]
    return names[0] if names else None


def survey(client: MetaculusReadClient, tournament: str, log: Callable[[str], None]) -> dict[str, Any]:
    posts = client.get_tournament_posts(tournament, statuses=["open"])
    shapes: Counter = Counter()
    sizes: list[int] = []
    ladders, coherence = 0, Counter()
    by_entity: dict[str, list[str]] = defaultdict(list)
    for post in posts:
        qs = questions_of(post)
        title = str(post.get("title") or "")
        key = entity(title)
        if key:
            by_entity[key].append(title[:90])
        if isinstance(post.get("question"), dict):
            shapes[f"single:{qs[0].get('type')}"] += 1
            continue
        sizes.append(len(qs))
        types = sorted({str(q.get("type")) for q in qs})
        shapes[f"group:{'+'.join(types)}"] += 1
        options = [str(q.get("label") or "") for q in qs]
        log(f"  group post={post.get('id')} n={len(qs)} types={types} title={title[:90]!r} "
            f"options={options[:8]}")
        if is_ladder(qs):
            ladders += 1
            try:
                detail = client.get_json(f"/posts/{post['id']}/")
            except MetaculusReadError:
                coherence["unreadable"] += 1
                continue
            coherence[ladder_coherence(questions_of(detail))] += 1
    clusters = {k: v for k, v in by_entity.items() if len(v) >= 2}
    for k, titles in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        log(f"  cluster {k!r} posts={len(titles)}: " + " | ".join(titles[:6]))
    return {"posts": len(posts), "shapes": dict(shapes), "group_sizes": sorted(sizes),
            "ladders": ladders, "ladder_coherence": dict(coherence),
            "clusters": len(clusters), "posts_in_clusters": sum(len(v) for v in clusters.values())}


def main() -> int:
    client = MetaculusReadClient(os.environ.get("METACULUS_TOKEN", ""))
    for name, tournament in TOURNAMENTS:
        print(f"== {name} ({tournament})")
        try:
            summary = survey(client, tournament, print)
        except MetaculusReadError as exc:
            print(f"survey_failed tournament={name} status={exc.status}")
            continue
        print(f"siblings_summary tournament={name} " +
              " ".join(f"{k}={v}" for k, v in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
