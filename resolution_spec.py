"""Resolution forensics: what exactly resolves the question, read from the
question and its resolution sources, before anyone forecasts.

Why this exists
---------------
The 2026-11-16 research into the tournament leaders found one practice every
described top bot shares (killian, SwampBot, Panshul42, GreeneiBot2): before
forecasting they pin down what counts as YES and read the resolution source
itself, in full. Preseen lost points by missing resolution clarifications;
bot makers name "before date X" handling as their worst failure. Ours read
three linked pages cut to 1,250 characters each (live_context), and the
research rundown reads search-result snippets.

One Haiku 5.5 request per question, separate from the search loop:

* the question, its resolution criteria, fine print and timing go in as the
  user message, so every link in them is fetchable by the web fetch tool
  (it only opens URLs that appear in the conversation);
* ``web_fetch_20250910`` (the basic version, the one Haiku 5.5 takes), at
  most MAX_FETCHES pages of about MAX_CONTENT_TOKENS each; fetching has no
  fee beyond tokens. A PDF is not capped by max_content_tokens, so a large one
  can push the request past Haiku's 100k-token price step; the cost is still
  cents and is recorded like any research call;
* the answer is a short spec: what resolves YES (or the exact quantity), the
  event deadline as opposed to the question's close, the official source and
  what it shows now, whether the outcome is already effectively determined,
  and the traps (units, alternative readings, revisions).

Never raises to the caller's benefit: build() returns "" on any failure, so a
failed spec costs only itself. Nothing it writes is logged (public repo).
"""

from __future__ import annotations

import logging
import re

import claude_research
import spend

logger = logging.getLogger(__name__)

ENGINE_KEY = "haiku"
MAX_FETCHES = 3
MAX_CONTENT_TOKENS = 8000
MAX_TOKENS = 4000
EFFORT = "medium"
MAX_RESUMES = 2
URL = re.compile(r"https?://[^\s)\]>\"']+")

SYSTEM = (
    "You check exactly how a forecasting question resolves, for a professional "
    "forecaster. Read the question, its resolution criteria and fine print. Use "
    "the web_fetch tool to open the resolution source and any page the criteria "
    "link to (not other pages), and read what they publish now. Do not forecast "
    "and do not say how likely any outcome is.\n\n"
    "Answer in at most 220 words, under these headings, each one or two lines:\n"
    "RESOLVES YES IF / THE QUANTITY: the exact condition or number that counts, "
    "with its units and threshold, as the criteria define it.\n"
    "DEADLINE: the date by which the event must happen or the value be measured; "
    "say when it differs from the question's close.\n"
    "SOURCE NOW: the official source and the latest value or status it shows, "
    "with its date; or 'could not read' and why.\n"
    "ALREADY DETERMINED?: yes, no, or nearly, and on what evidence. A question "
    "is open unless the source shows otherwise.\n"
    "TRAPS: ambiguous wording, alternative readings, unit or revision pitfalls, "
    "or 'none seen'."
)


def links(*texts: str) -> list[str]:
    """The non-Metaculus links in ``texts``, in order, each once."""
    found = [u.rstrip(".,;") for t in texts for u in URL.findall(t or "")]
    return [u for u in dict.fromkeys(found) if "metaculus.com" not in u]


def prompt(question_text: str, criteria: str, fine_print: str, timing: str) -> str:
    urls = links(criteria, fine_print, question_text)
    sources = "\n".join(f"- {u}" for u in urls) or "- none linked"
    return (
        f"{timing}\n\nQuestion:\n{question_text}\n\nResolution criteria:\n{criteria}\n\n"
        f"Fine print:\n{fine_print}\n\nLinks in the criteria (fetch the resolution "
        f"source first):\n{sources}"
    )


async def build(question_text: str, criteria: str, fine_print: str, timing: str) -> str:
    """The spec as a research section, or "" when it fails."""
    try:
        return await _build(question_text, criteria, fine_print, timing)
    except Exception as exc:  # noqa: BLE001 - a failed spec costs only itself
        logger.warning("resolution_spec_failed reason=%s", type(exc).__name__)
        return ""


async def _build(question_text: str, criteria: str, fine_print: str, timing: str) -> str:
    import anthropic

    engine = claude_research.ENGINES[ENGINE_KEY]
    text_in = prompt(question_text, criteria, fine_print, timing)
    tools = [{"type": "web_fetch_20250910", "name": "web_fetch",
              "max_uses": MAX_FETCHES, "max_content_tokens": MAX_CONTENT_TOKENS}]
    client = anthropic.AsyncAnthropic()
    tally = claude_research.Tally(engine)
    messages = [{"role": "user", "content": text_in}]
    content: list = []
    response = None
    fetches = 0
    try:
        for attempt in range(MAX_RESUMES + 1):
            response = await client.beta.messages.create(
                model=engine.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                messages=[*messages, {"role": "assistant", "content": content}] if content else messages,
                tools=tools,
                output_config={"effort": EFFORT},
            )
            tally.add(response.usage)
            used = getattr(response.usage, "server_tool_use", None)
            fetches += int(getattr(used, "web_fetch_requests", 0) or 0) if used is not None else 0
            content = [*content, *response.content]
            if response.stop_reason != "pause_turn":
                break
    finally:
        if tally.usd:
            spend.record(provider="anthropic", model=engine.model, usd=tally.usd,
                         tokens_in=tally.tokens_in, tokens_out=tally.out, searches=0)
    if response is None or response.stop_reason == "refusal":
        return ""
    text = claude_research._text(content)
    logger.info("resolution_spec ok=%d fetches=%d tokens_in=%d est_cost_usd=%.4f",
                int(bool(text)), fetches, tally.tokens_in, tally.usd)
    if not text:
        return ""
    return ("## How this question resolves (read from its criteria and resolution "
            f"source just now)\n{text}")
