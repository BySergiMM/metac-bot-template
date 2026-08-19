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

What this deliberately does NOT do
----------------------------------
It changes availability and nothing else. One call in gives one string out, of
the same type, whichever provider produced it. It does not know what a
forecast is, cannot retry a forecast, and cannot make one question look like
two. The number of predictions, the success threshold, the aggregation and the
POST are all decided upstream and are untouched.

Metrics discipline: a provider attempt is INFRASTRUCTURE, not a prediction.
Switching provider mid-question is one prediction served by the second
provider, never two predictions. This class emits structured logs so provider
behaviour can be measured separately, and increments no forecast counter -
there is none here to increment.
"""

from __future__ import annotations

import logging
import time

from forecasting_tools.ai_models.general_llm import GeneralLlm

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
            started = time.monotonic()
            # INFRASTRUCTURE metric. Deliberately not a forecast counter.
            logger.info(
                "llm_attempt provider_index=%d model=%s", depth, backend.model
            )
            try:
                result = await backend.invoke(prompt, system_prompt)
            except Exception as exc:  # noqa: BLE001 - must try the next backend regardless of cause
                elapsed = time.monotonic() - started
                errors.append(f"{backend.model}: {exc}")
                logger.warning(
                    "llm_failure provider_index=%d model=%s latency_s=%.2f error=%r",
                    depth, backend.model, elapsed, exc,
                )
                continue

            elapsed = time.monotonic() - started
            logger.info(
                "llm_success provider_index=%d model=%s latency_s=%.2f fallback_used=%s",
                depth, backend.model, elapsed, depth > 0,
            )
            if depth > 0:
                logger.warning(
                    "Primary backend(s) failed, served by fallback #%d: %s",
                    depth, backend.model,
                )
            # NOTE: self.model is deliberately NOT reassigned to the serving
            # backend. forecast_questions runs every question concurrently
            # through one shared llm object, so mutating instance state here is
            # a race that mis-attributes models across questions. Which backend
            # answered is recorded in the log line above instead, where it is
            # per-call and cannot be clobbered.
            return result

        logger.error(
            "llm_exhausted backends=%d - question will have no prediction from this call",
            len(self._backends),
        )
        raise RuntimeError("All backends failed:\n" + "\n".join(errors))
