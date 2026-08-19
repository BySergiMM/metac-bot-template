"""FallbackLlm: a GeneralLlm that tries several backends in order.

Why this exists
----------------
A free-tier daily rate limit (OpenRouter: 50 req/day per account) is a wall,
not a retry-able hiccup: retrying the same account does nothing until the
next UTC reset. The fix is a second account on a different provider, tried
only when the first one is actually exhausted.

This subclasses GeneralLlm (not a bare duck-typed object) because
forecast_bots code does `isinstance(llm, GeneralLlm)` in several places
(forecast_bot.py get_llm, template_bot_2026_summer.py) - anything that
fails that check gets silently downgraded to a plain string re-wrap, which
would throw away every kwarg (timeout, temperature, allowed_tries) and defeat
the point.

Only __init__ and invoke() are overridden. Everything else (cost tracking,
retry decorator, token limits) is inherited from GeneralLlm and applies to
whichever backend actually served the request.

Rate limiting
-------------
Every backend call goes through a process-wide limiter for that model first.
The E2E run proved this is not optional: without it the bot fired ~185 calls in
18 seconds against a Gemini quota of 15 per MINUTE and produced zero forecasts.

The limiter is fetched from a registry keyed by model, never stored per
instance, because get_llm() hands out one object per ROLE - four of them, each
with its own GeneralLlm for the same Gemini model. Per-instance limiters would
silently permit four times the quota.

What this deliberately does NOT do
----------------------------------
It changes availability and pacing, nothing else. One call in gives one string
out, of the same type, whichever provider produced it and however long it
waited. It does not know what a forecast is, cannot retry a forecast, and
cannot make one question look like two. The number of predictions, the success
threshold, the aggregation and the POST are all decided upstream and are
untouched. A call delayed by the limiter is still exactly one call.

Metrics discipline: provider attempts and rate-limit waits are INFRASTRUCTURE.
They live in rate_limiter.COUNTERS and are never read by the forecast metric,
which is derived from Metaculus' own my_forecasts records.
"""

from __future__ import annotations

import logging
import time

from forecasting_tools.ai_models.general_llm import GeneralLlm

from backtest.rate_limiter import COUNTERS, RateLimitTimeout, get_limiter

logger = logging.getLogger(__name__)


class FallbackLlm(GeneralLlm):
    def __init__(self, backends: list[GeneralLlm]) -> None:
        if not backends:
            raise ValueError("FallbackLlm needs at least one backend")
        self._backends = backends
        # Present as the primary backend for cost tracking / isinstance checks/
        # logging that read self.model before invoke() has run.
        primary = backends[0]
        super().__init__(
            model=primary.model,
            temperature=primary.litellm_kwargs.get("temperature"),
            timeout=primary.litellm_kwargs.get("timeout"),
            allowed_tries=primary.allowed_tries,
        )

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        """Try each backend once, in order, and return the first success.

        Note this overrides GeneralLlm.invoke, so the RetryableModel decorator
        does NOT wrap the chain: the chain is attempted exactly once. Each
        backend keeps its own allowed_tries, which pin_models.py sets to 1 for
        chain members so a dead provider costs one attempt, not three plus
        exponential backoff.
        """
        errors = []
        for depth, backend in enumerate(self._backends):
            limiter = get_limiter(backend.model)

            # Pace before calling. A wait here is not a failure and not a
            # retry: it is the same single call, starting later.
            try:
                waited = await limiter.acquire()
            except RateLimitTimeout as exc:
                errors.append(f"{backend.model}: {exc}")
                logger.warning(
                    "llm_ratelimit_giveup provider_index=%d provider=%s reason=%s",
                    depth, backend.model, exc,
                )
                continue

            COUNTERS.llm_attempts_total += 1
            logger.info(
                "llm_attempt provider_index=%d provider=%s wait_ms=%d",
                depth, backend.model, int(waited * 1000),
            )

            started = time.monotonic()
            try:
                result = await backend.invoke(prompt, system_prompt)
            except Exception as exc:  # noqa: BLE001 - must try the next backend regardless of cause
                elapsed = time.monotonic() - started
                errors.append(f"{backend.model}: {exc}")
                logger.warning(
                    "llm_failure provider_index=%d provider=%s latency_s=%.2f reason=%r",
                    depth, backend.model, elapsed, _reason(exc),
                )
                continue

            elapsed = time.monotonic() - started
            COUNTERS.llm_success_total += 1
            if depth > 0:
                COUNTERS.llm_fallback_total += 1
            logger.info(
                "llm_success provider_index=%d provider=%s latency_s=%.2f "
                "wait_ms=%d fallback_used=%s",
                depth, backend.model, elapsed, int(waited * 1000), depth > 0,
            )
            # NOTE: self.model is deliberately NOT reassigned to the serving
            # backend. forecast_questions runs every question concurrently
            # through one shared llm object, so mutating instance state here is
            # a race that mis-attributes models across questions. Which backend
            # answered is recorded in the log line above instead, where it is
            # per-call and cannot be clobbered.
            return result

        logger.error(
            "llm_exhausted backends=%d - this call produced no text; the "
            "prediction it belonged to will not exist",
            len(self._backends),
        )
        raise RuntimeError("All backends failed:\n" + "\n".join(errors))


def _reason(exc: Exception) -> str:
    """A short, log-safe label. Never the full response body: prompts and model
    output can be long and are not ours to spray into CI logs."""
    name = type(exc).__name__
    text = str(exc)
    for marker in ("RESOURCE_EXHAUSTED", "rate_limit", "RateLimit", "429"):
        if marker in text:
            return name + ":rate_limited"
    return name + ":" + text[:80].replace("\n", " ")
