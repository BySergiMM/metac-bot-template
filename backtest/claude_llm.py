"""ClaudeLlm: one Claude model called directly with the owner's Anthropic key.

Why this exists
---------------
Since 2026-11-03 every role runs on the Anthropic key and no other (owner's
decision). On 2026-10-25 the sponsored OpenRouter key hit its limit and, for nine
days, the fallback chain had Gemini Flash Lite forecasting in Opus's place
without anyone noticing.

Not GeneralLlm's litellm path: the locked litellm (1.80.10) rejects effort
"xhigh" client-side and sends extra_body as a literal field of the request
(checked against a local stand-in for the API). This is the same SDK call
claude_research already makes, so the two price and record alike.

It is a GeneralLlm for everything that reads one (FallbackLlm reads .model and
.litellm_kwargs; structure_output calls invoke), and overrides only invoke.

Caching across a question's forecasts
-------------------------------------
The forecasters of one question send the same prompt (the research included) at
once, so none could read what another wrote. The first caller of a prompt goes
alone and the others wait for it: their input then costs 0.05x instead of 1x on
Opus, about half of what a question's three forecasts spend on input. A failed
first call still releases the others.
"""
from __future__ import annotations

import asyncio
import hashlib

from forecasting_tools.ai_models.general_llm import GeneralLlm

import claude_research
import spend

MAX_TOKENS = 16000

#: Prompts with a first call in flight, per event loop: key -> Event.
_in_flight: dict[tuple[int, str], asyncio.Event] = {}


class ClaudeLlm(GeneralLlm):
    """``model`` is "anthropic/<api id>"; ``effort`` asks for adaptive thinking
    at that effort, and None leaves the model's default without thinking."""

    def __init__(self, model: str, effort: str | None = None, timeout: float = 180,
                 max_tokens: int = MAX_TOKENS, allowed_tries: int = 1) -> None:
        super().__init__(model=model, temperature=None, timeout=timeout, allowed_tries=allowed_tries)
        self.api_model = model.partition("/")[2]
        self.effort = effort
        self.max_tokens = max_tokens
        self.request_timeout = timeout
        self.engine = next(e for e in claude_research.ENGINES.values() if e.model == self.api_model)

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        key = (id(asyncio.get_running_loop()), hashlib.sha256(
            f"{self.api_model}\0{system_prompt}\0{prompt}".encode()).hexdigest())
        first = _in_flight.get(key)
        if first is not None:
            await first.wait()
            return await self._call(prompt, system_prompt)
        done = _in_flight[key] = asyncio.Event()
        try:
            return await self._call(prompt, system_prompt)
        finally:
            del _in_flight[key]
            done.set()

    async def _call(self, prompt, system_prompt: str | None) -> str:
        import anthropic

        extra = {}
        if self.effort:
            extra = {"thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort}}
        if system_prompt:
            extra["system"] = system_prompt
        async with anthropic.AsyncAnthropic(timeout=self.request_timeout) as client:
            response = await client.beta.messages.create(
                model=self.api_model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": str(prompt)}],
                cache_control=claude_research.CACHE,
                **extra,
            )
        tally = claude_research.Tally(self.engine)
        tally.add(response.usage)
        spend.record(provider="anthropic", model=self.api_model, usd=tally.usd,
                     tokens_in=tally.tokens_in, tokens_out=tally.out)
        if response.stop_reason == "refusal":
            raise RuntimeError("claude refused")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise RuntimeError(f"claude returned no text (stop_reason={response.stop_reason})")
        return text
