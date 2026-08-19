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
"""

from __future__ import annotations

import logging

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
        errors = []
        for i, backend in enumerate(self._backends):
            try:
                result = await backend.invoke(prompt, system_prompt)
                if i > 0:
                    logger.warning(
                        f"Primary backend(s) failed, served by fallback #{i}: {backend.model}"
                    )
                # Mirror whichever backend actually answered, so cost
                # tracking/logging downstream reflects reality.
                self.model = backend.model
                return result
            except Exception as exc:  # noqa: BLE001 - must try the next backend regardless of cause
                errors.append(f"{backend.model}: {exc}")
                logger.warning(f"Backend {backend.model} failed ({exc!r}), trying next")
        raise RuntimeError(
            "All backends failed:\n" + "\n".join(errors)
        )
