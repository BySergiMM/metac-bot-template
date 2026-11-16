"""Recent news from AskNews, as one more research section.

Why this exists
---------------
On the Fall 2026 FutureEval board (2026-11-16) the two best Metaculus
reference bots both read AskNews: metac-asknews-deepnews[research-only] 595
and metac-claude-fable-5-1-high+asknews 587 points, against our 105. The
leaders research found the number of distinct research sources to be the most
robust correlate of score (r=0.42); ours was one, Anthropic web search.
AskNews is free for registered FutureEval bot makers: 1,000 calls a month and
4,000 a tournament, a "latest news" search costing 1 call and a "news
knowledge" (archive) search 5 (Metaculus' AI benchmark resources page). Its
terms keep it to the competition, so it never runs for the Cup.

It is added to the research, never a replacement for it, and only for the
forecasts that are scored:

* ``full``: latest news and the archive -- FutureEval, 6 calls a question;
* ``latest``: latest news only -- MiniBench and the Market Pulse final.

At about 90 FutureEval and 155 other questions a month that is about 700
calls, under the 1,000. MONTHLY_CALLS keeps it there: the calls are written
to the spend ledger (provider "asknews", $0) and the archive search stops
first, then everything, as the month's count nears the allowance.

The free tier allows one call every 10 seconds, so calls are spaced by a lock
shared by the whole process. Credentials: ASKNEWS_API_KEY, or
ASKNEWS_CLIENT_ID with ASKNEWS_SECRET (the SDK rejects both at once). Without
them nothing happens. Never raises: "" on any failure. Article text is never
logged (public repo).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import spend

logger = logging.getLogger(__name__)

#: The free allowance, and the margin kept from it.
MONTHLY_CALLS = 1000
STOP_ARCHIVE_AT = 800
STOP_ALL_AT = 950
CALLS = {"latest news": 1, "news knowledge": 5}
SPACING_SECONDS = 10.5
N_LATEST = 8
N_ARCHIVE = 10
SUMMARY_CHARS = 600
MAX_CHARS = 9000

_lock: asyncio.Lock | None = None
_last_call = 0.0


def credentials(environ=None) -> dict:
    """The SDK arguments, or {} when no usable credential is set."""
    env = os.environ if environ is None else environ
    if env.get("ASKNEWS_API_KEY"):
        return {"api_key": env["ASKNEWS_API_KEY"]}
    if env.get("ASKNEWS_CLIENT_ID") and env.get("ASKNEWS_SECRET"):
        return {"client_id": env["ASKNEWS_CLIENT_ID"], "client_secret": env["ASKNEWS_SECRET"]}
    return {}


def enabled() -> bool:
    return bool(credentials())


def calls_this_month(lines: list[dict], now: datetime) -> int:
    """AskNews calls recorded since the 1st of this calendar month (AskNews'
    allowance is monthly; the ledger's own cycle is Claude's)."""
    first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = 0
    for line in spend.calls(lines):
        if line.get("provider") != "asknews":
            continue
        try:
            when = datetime.fromisoformat(str(line.get("t")))
        except ValueError:
            continue
        if when >= first:
            total += int(line.get("searches", 0) or 0)
    return total


def plan(mode: str | None, used: int) -> list[str]:
    """The strategies to run for one question under ``mode``, given the
    month's calls so far."""
    if mode not in ("latest", "full") or used >= STOP_ALL_AT:
        return []
    if mode == "full" and used < STOP_ARCHIVE_AT:
        return ["latest news", "news knowledge"]
    return ["latest news"]


def _format(articles) -> str:
    rows = []
    for a in sorted(articles, key=lambda x: getattr(x, "pub_date", None) or datetime.min, reverse=True):
        when = getattr(a, "pub_date", None)
        day = when.strftime("%Y-%m-%d") if when else "undated"
        title = getattr(a, "eng_title", "") or getattr(a, "title", "") or ""
        summary = " ".join(str(getattr(a, "summary", "") or "").split())[:SUMMARY_CHARS]
        source = getattr(a, "source_id", "") or ""
        url = getattr(a, "article_url", "") or ""
        rows.append(f"- [{day}] {title} ({source}) {url}\n  {summary}")
    return "\n".join(rows)


async def _spaced(call):
    global _lock, _last_call
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        wait = _last_call + SPACING_SECONDS - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            return await call()
        finally:
            _last_call = time.monotonic()


async def news(query: str, mode: str | None, sdk_factory=None) -> str:
    """The news section for ``query`` under ``mode`` ("latest", "full" or
    None), or ""."""
    try:
        return await _news(query, mode, sdk_factory)
    except Exception as exc:  # noqa: BLE001 - a news failure costs only itself
        logger.warning("asknews_failed reason=%s", type(exc).__name__)
        return ""


async def _news(query: str, mode: str | None, sdk_factory) -> str:
    creds = credentials()
    if not creds or not mode:
        return ""
    strategies = plan(mode, calls_this_month(spend.load(), datetime.now(timezone.utc)))
    if not strategies:
        logger.info("asknews_skipped reason=monthly_allowance")
        return ""
    if sdk_factory is None:
        from asknews_sdk import AsyncAskNewsSDK

        def sdk_factory():
            return AsyncAskNewsSDK(scopes={"news"}, **creds)

    parts, found = [], 0
    async with sdk_factory() as ask:
        for strategy in strategies:
            n = N_LATEST if strategy == "latest news" else N_ARCHIVE
            try:
                response = await _spaced(lambda s=strategy, n=n: ask.news.search_news(
                    query=query, n_articles=n, return_type="dicts", strategy=s))
            finally:
                # Billed whether or not it returned anything.
                spend.record(provider="asknews", model=strategy.replace(" ", "-"), usd=0.0,
                             searches=CALLS[strategy])
            articles = list(getattr(response, "as_dicts", None) or [])
            found += len(articles)
            if articles:
                label = "Last days" if strategy == "latest news" else "Earlier months"
                parts.append(f"### {label}\n{_format(articles)}")
    logger.info("asknews strategies=%d articles=%d", len(strategies), found)
    if not parts:
        return ""
    text = "\n\n".join(parts)[:MAX_CHARS]
    return ("## Recent news (AskNews, dated; a second source beside the web research)\n"
            f"{text}")
