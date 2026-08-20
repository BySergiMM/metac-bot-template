"""BalancedLlm: spread calls across several independent Gemini quota buckets.

Why this exists
---------------
Gemini meters PerProjectPerModel. Four credentials from four projects are four
independent 15 RPM allowances -- measured by interference, not assumed (runs
32380381980 and 32381181256: all six pairs independent, each quoting
quotaValue "15").

Appending a second Gemini to a FallbackLlm chain does NOT use that allowance.
FallbackLlm advances only on an *exception*, and a saturated limiter does not
raise -- it waits. Simulated over 60 questions, a second Gemini added as a
fourth link served 0 of 1020 calls and the run took exactly as long as with
one. Extra quota is only reachable by choosing a chain at ADMISSION time,
which is what this class does.

What it deliberately does NOT change
------------------------------------
Each chain keeps the production order OpenRouter -> Gemini -> Groq, untouched.
This class sits above them and only decides *which* chain a call enters. One
call in still gives one string out; it cannot retry a forecast, change how many
predictions exist, or make one question look like two. The success threshold,
the aggregation and the POST are all decided upstream.

Selection policy
----------------
Least-loaded by admitted requests currently inside each bucket's 60-second
window -- the same quantity the limiter itself rations. Ties go to the earliest
bucket, which keeps the choice deterministic and makes tests reproducible.

Cross-chain failover
--------------------
If the chosen chain raises, the remaining chains are tried in load order. This
is not a second fallback ladder: the chains are interchangeable, so it is a
retry of the same call against an equivalent route. It exists for one concrete
failure: the parser chain excludes Groq (Groq cannot emit schema-constrained
JSON), so a parser chain whose Gemini credential is dead has no third link. A
prediction must not disappear because a *secondary* key was mistyped.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from forecasting_tools.ai_models.general_llm import GeneralLlm

from backtest.rate_limiter import get_limiter

logger = logging.getLogger(__name__)


def bucket_backend(
    model: str,
    api_key_env: str,
    limiter_key: str,
    timeout: int,
    allowed_tries: int,
    temperature: float | None = 0.3,
) -> GeneralLlm:
    """One Gemini backend bound to a specific credential and quota bucket.

    The credential is read from ``api_key_env`` here, at bot runtime, rather
    than being baked into the generated source -- a secret must never be
    written into a file. An absent or empty variable falls through to litellm's
    own environment lookup, which is exactly the pre-bucket behaviour.

    ``limiter_key`` is attached as an ATTRIBUTE rather than passed to the
    constructor on purpose: GeneralLlm forwards unknown kwargs straight into
    ``acompletion``, so a ``limiter_key=`` kwarg would travel to litellm as a
    request parameter. ``api_key`` is a real litellm parameter and is passed
    normally.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
        "allowed_tries": allowed_tries,
    }
    key = (os.environ.get(api_key_env) or "").strip()
    if key:
        kwargs["api_key"] = key
    backend = GeneralLlm(**kwargs)
    backend.limiter_key = limiter_key  # type: ignore[attr-defined]
    return backend


class BalancedLlm(GeneralLlm):
    """Route each call to the least-loaded of several equivalent chains."""

    def __init__(self, chains: list[GeneralLlm], bucket_keys: list[str]) -> None:
        if not chains:
            raise ValueError("BalancedLlm needs at least one chain")
        if len(chains) != len(bucket_keys):
            raise ValueError(
                "each chain must name the bucket it is balanced on: "
                f"{len(chains)} chains, {len(bucket_keys)} bucket keys"
            )
        self._chains = chains
        self._bucket_keys = bucket_keys
        primary = chains[0]
        super().__init__(
            model=primary.model,
            temperature=primary.litellm_kwargs.get("temperature"),
            timeout=primary.litellm_kwargs.get("timeout"),
            allowed_tries=primary.allowed_tries,
        )

    def _load_order(self) -> list[int]:
        """Chain indices, least-loaded first. Ties keep registry order."""
        loads = []
        for index, key in enumerate(self._bucket_keys):
            limiter = get_limiter(key)
            limiter._prune(limiter._clock())
            loads.append((len(limiter._requests), index))
        loads.sort()
        return [index for _load, index in loads]

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        errors: list[str] = []
        order = self._load_order()
        for position, index in enumerate(order):
            try:
                result = await self._chains[index].invoke(prompt, system_prompt)
            except Exception as exc:  # noqa: BLE001 - an equivalent chain remains
                errors.append(f"bucket[{index}]: {exc}")
                logger.warning(
                    "balanced_chain_failed bucket_index=%d position=%d remaining=%d",
                    index, position, len(order) - position - 1,
                )
                continue
            if position > 0:
                logger.info(
                    "balanced_recovered bucket_index=%d after=%d failed chain(s)",
                    index, position,
                )
            return result

        logger.error(
            "balanced_exhausted chains=%d - this call produced no text; the "
            "prediction it belonged to will not exist",
            len(self._chains),
        )
        raise RuntimeError("All balanced chains failed:\n" + "\n".join(errors))
