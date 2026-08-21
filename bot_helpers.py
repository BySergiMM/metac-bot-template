"""
Runtime helpers for the template bot: environment validation, startup/result
banners, and suppression of noisy upstream warnings.

Kept separate from main.py so that file can focus on the bot's forecasting
logic. main_with_no_framework.py keeps its own inline copies on purpose --
it's meant to be a single-file reference implementation.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from typing import Any, Sequence


# Placeholder values shipped in .env.template. If a real env var still equals
# one of these the user forgot to replace it; we'd rather fail loudly here than
# inside the SDK three layers down.
_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def _is_real_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip() and val.strip() not in _PLACEHOLDER_ENV_VALUES)


def silence_noisy_dependencies() -> None:
    """
    Quiet warnings from transitive deps that fire on import and confuse new
    users. Must be called *before* importing forecasting_tools.
    """
    warnings.filterwarnings(
        "ignore", message=r".*does not support cost tracking.*"
    )
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(
        logging.ERROR
    )
    # Streamlit installs its own logger hierarchy; suppress via its own API.
    try:
        from streamlit.logger import set_log_level

        set_log_level("error")
    except ImportError:
        pass
    # LiteLLM is verbose at INFO; its WARNING level is enough for us.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


# Log records whose CONTENT is a forecast on a live tournament question.
# Everything else -- discovery counts, provider/bucket lines, rate-limit waits,
# "Posted prediction/comment" -- carries no forecast content and is left alone,
# because that is what every audit of this bot has relied on.
_FORECAST_CONTENT_PATTERNS = (
    # main.py: the full research body, logged with the question URL
    re.compile(r"^(Found Research for URL \S+)", re.S),
    # main.py: the individual prediction value, for every question type
    re.compile(r"^(Forecasted URL \S+) with prediction: .*", re.S),
)
# forecast_bot.log_report_summary: summaries and first rationales, in one record
_SUMMARY_MARKERS = ("<<<<<<<<<<<<<<<<<<<< Summary", "First Rationale")


class RedactForecastContent(logging.Filter):
    """Keep forecast reasoning out of the process's log stream.

    Why this exists
    ---------------
    This repository is a PUBLIC fork, so GitHub Actions logs are world
    readable. A tournament run used to emit, in the clear: the full research
    body, the report summary and first rationale, and each of the five
    individual predictions -- for a question that was still OPEN.

    Two distinct problems, one cause:

      * Metaculus requires bots to leave PRIVATE comments, published later at
        Metaculus' own intervals. Publishing the same reasoning immediately in
        a public log defeats that, whatever the log's medium.
      * FutureEval forbids a bot maker from previewing their bot's forecasts on
        open questions and then updating the bot. Printing the predictions puts
        that preview in front of the maker on every run. Removing it makes the
        guarantee structural instead of a promise -- the same reasoning that
        keeps research/ limited to closed questions.

    It redacts CONTENT, never the fact that work happened: the question URL,
    the call counts, the provider and bucket lines and the publication
    confirmations all survive, so operational auditing is unaffected.

    Not applied in test_questions mode: bot-testing-area is an unscored
    practice area, and full output there is what makes it useful.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not kill the run
            return True

        if any(marker in message for marker in _SUMMARY_MARKERS):
            record.msg = (
                "[redacted: per-question summary and rationale withheld from a "
                "public log while the question may still be open]"
            )
            record.args = ()
            return True

        for pattern in _FORECAST_CONTENT_PATTERNS:
            match = pattern.match(message)
            if match:
                record.msg = match.group(1) + " [content redacted]"
                record.args = ()
                return True
        return True


def install_forecast_redaction(run_mode: str) -> bool:
    """Attach the redaction filter for scored modes. Returns whether it was."""
    if run_mode == "test_questions":
        return False
    log_filter = RedactForecastContent()
    root = logging.getLogger()
    root.addFilter(log_filter)
    # Filters on the root logger are not consulted for records emitted by child
    # loggers, so the handlers get it too - that is the path every record takes.
    for handler in root.handlers:
        handler.addFilter(log_filter)
    return True


def check_environment(strict: bool = True) -> None:
    """
    Verify METACULUS_TOKEN is set; warn if no LLM key is configured. On
    failure with strict=True, exits the process with a non-zero status.
    """
    problems: list[str] = []

    if not _is_real_env("METACULUS_TOKEN"):
        problems.append(
            "METACULUS_TOKEN is missing or still a placeholder. "
            "Get one at https://www.metaculus.com/futureeval/participate/"
        )

    has_llm_key = any(
        _is_real_env(k)
        for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_llm_key:
        print(
            "⚠️  No LLM key set (OPENROUTER/OPENAI/ANTHROPIC). The bot will fall back\n"
            "    to the Metaculus LLM proxy. Free OpenRouter credits: "
            "https://forms.gle/aQdYMq9Pisrf1v7d8\n"
        )

    if problems:
        print("❌  Setup problems:")
        for p in problems:
            print(f"    • {p}")
        if strict:
            sys.exit(1)


def print_startup_banner(run_mode: str, will_publish: bool) -> None:
    publish = "publish=yes" if will_publish else "publish=no (dry run)"
    print(f"🤖  Running mode={run_mode}, {publish}\n")


def print_run_summary_banner(
    forecast_reports: Sequence[Any],
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    """
    End-of-run summary printed via print() (not logger) so it survives log
    filtering. Shows count, per-question URLs, and any failure tracebacks.
    If tournament_url is given, it's included as a footer link.
    """
    # Lazy import so this module is usable in contexts where forecasting_tools
    # isn't installed (e.g. unit tests of the banner format).
    from forecasting_tools import ForecastReport

    valid = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]
    banner = "=" * 80

    print()
    print(banner)

    if not forecast_reports:
        print("ℹ️   No new questions to forecast on this run.")
        print(banner)
        print()
        return

    if valid and not exceptions:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"🎉  Bot {verb} {len(valid)} forecast(s).")
    elif valid and exceptions:
        print(
            f"⚠️   Partial — {len(valid)} succeeded, {len(exceptions)} failed."
        )
    else:
        print(f"❌  All {len(exceptions)} attempt(s) failed.")

    if valid:
        print()
        for r in valid:
            note = f"  (with {len(r.errors)} minor error(s))" if r.errors else ""
            print(f"  ✅ {r.question.page_url}{note}")
        if will_publish and tournament_url:
            print(f"\n  Tournament: {tournament_url}")

    if exceptions:
        print()
        for exc in exceptions:
            msg = str(exc)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"  ❌ {type(exc).__name__}: {msg}")

    print(banner)
    print()
