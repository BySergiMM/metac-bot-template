"""BalancedLlm: route each call to the least-loaded of several fallback chains.

Why this exists
---------------
backtest/pin_models.py wraps the forecaster (the "default" role) in this class,
holding one FallbackLlm chain PER PRIMARY MODEL. That role is invoked
predictions_per_research_report times per question and the answers are
aggregated, so if every call went to one model the five answers would be
correlated samples whose shared blind spots survive the average.
FutureSearch publishes the finding ("Run agents twice for fun and profit",
2026-05-01): ensembling across different models cuts Brier score, not just
repeating one model. Choosing a chain at ADMISSION time spreads the calls over
the distinct models instead.

It has to be done at admission, not by appending models to one chain:
FallbackLlm advances only on an *exception*, and a saturated limiter does not
raise -- it waits, up to its ceiling. A model placed later in a chain is
therefore not reached while the one ahead of it is merely busy.

The chains are interchangeable in VALIDITY only: any of them can serve a call
and every one returns a usable answer, deliberately not the same answer,
because each leads with a different model.

What it deliberately does NOT change
------------------------------------
Inside each chain the models behind the leading one keep their production
order (OpenRouter -> Gemini -> Groq); only the lead differs between chains.
This class sits above them and only decides *which* chain a call enters. One
call in still gives one string out; it cannot retry a forecast, change how many
predictions exist, or make one question look like two. The success threshold,
the aggregation and the POST are all decided upstream.

Selection policy
----------------
Least-loaded by admitted requests currently inside each chain's 60-second
window -- the same quantity the limiter itself rations. The window measured is
the one registered under the chain's entry in ``bucket_keys``: a key of
backtest.rate_limiter's registry, which pin_models sets to the chain's primary
model. Ties go to the earliest chain, which keeps the choice deterministic and
makes tests reproducible.

Cross-chain failover
--------------------
If the chosen chain raises, the remaining chains are tried in load order, so a
call is lost only when every chain has failed.
"""

from __future__ import annotations

import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm

from backtest.rate_limiter import get_limiter

logger = logging.getLogger(__name__)


class BalancedLlm(GeneralLlm):
    """Route each call to the least-loaded of several chains.

    Any of them can serve the call, but they lead with different models on
    purpose, so they are not interchangeable in WHAT they answer -- see the
    module docstring.
    """

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
            except Exception as exc:  # noqa: BLE001 - another chain remains
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
