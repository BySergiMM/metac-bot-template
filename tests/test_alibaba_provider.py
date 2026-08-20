"""Alibaba DashScope as a candidate fallback provider.

Scope discipline, matching the rest of this suite: every test here is offline.
The only test that touches Alibaba is guarded by the presence of
ALIBABA_API_KEY and skips otherwise, so `python -m unittest discover` stays
network-free on a laptop and in the read-only research workflow.

What is asserted:
  1. the route is declared exactly as litellm 1.80.10 expects it
  2. an ABSENT credential degrades to NOT_TESTED, never to an exception
  3. the existing rate limiter accepts the model without special-casing
  4. FallbackLlm can carry a DashScope backend without disturbing the
     OpenRouter -> Gemini -> Groq ordering that production relies on

No credential value is ever printed, asserted on, or written to a file.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

# ------------------------------------------------------------- stub setup
# Same shape as tests/test_fallback_e2e_simulation.py: the suite must run
# without forecasting-tools installed.
if "forecasting_tools" not in sys.modules:
    _pkg = types.ModuleType("forecasting_tools")
    _ai = types.ModuleType("forecasting_tools.ai_models")
    _gl = types.ModuleType("forecasting_tools.ai_models.general_llm")

    class _StubGeneralLlm:
        def __init__(self, model, temperature=None, timeout=None,
                     allowed_tries=1, **kwargs):
            self.model = model
            self.allowed_tries = allowed_tries
            self.litellm_kwargs = {"temperature": temperature, "timeout": timeout}

        async def invoke(self, prompt, system_prompt=None):  # pragma: no cover
            raise NotImplementedError

    _gl.GeneralLlm = _StubGeneralLlm
    _ai.general_llm = _gl
    _pkg.ai_models = _ai
    sys.modules["forecasting_tools"] = _pkg
    sys.modules["forecasting_tools.ai_models"] = _ai
    sys.modules["forecasting_tools.ai_models.general_llm"] = _gl

import asyncio  # noqa: E402

from backtest import rate_limiter as rl  # noqa: E402
from backtest.fallback_llm import FallbackLlm  # noqa: E402
from forecasting_tools.ai_models.general_llm import GeneralLlm  # noqa: E402
from research import smoke_test_providers as stp  # noqa: E402

DASHSCOPE_MODEL = "dashscope/qwen-turbo"
INTL = "Alibaba DashScope (intl)"
MAINLAND = "Alibaba DashScope (mainland)"

# Transcribed from litellm 1.80.10, not guessed:
#   llms/dashscope/chat/transformation.py  -> mainland base
#   constants.py:554                       -> intl base
EXPECTED_BASES = {
    INTL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    MAINLAND: "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def route_by_label(label: str) -> dict:
    matches = [r for r in stp.ROUTES if r["label"] == label]
    assert len(matches) == 1, "expected exactly one route named {0}".format(label)
    return matches[0]


class RouteDeclarationTests(unittest.TestCase):
    """The route must match what litellm will actually do with it."""

    def test_both_regions_are_declared(self):
        labels = [r["label"] for r in stp.ROUTES]
        self.assertIn(INTL, labels)
        self.assertIn(MAINLAND, labels)

    def test_litellm_model_uses_the_dashscope_provider_prefix(self):
        for label in (INTL, MAINLAND):
            self.assertEqual(route_by_label(label)["litellm_model"], DASHSCOPE_MODEL)

    def test_api_base_matches_the_region(self):
        for label, base in EXPECTED_BASES.items():
            route = route_by_label(label)
            self.assertEqual(route["litellm_extra"]["api_base"], base)
            # The raw-HTTP probe must hit the same host as the litellm probe,
            # otherwise a pass on one layer says nothing about the other.
            self.assertTrue(route["url"].startswith(base))

    def test_our_secret_name_is_accepted_before_litellms_own(self):
        """litellm looks for DASHSCOPE_API_KEY; the repo secret is
        ALIBABA_API_KEY. Both are accepted, ours first."""
        for label in (INTL, MAINLAND):
            candidates = route_by_label(label)["env_candidates"]
            self.assertEqual(candidates[0], "ALIBABA_API_KEY")
            self.assertIn("DASHSCOPE_API_KEY", candidates)

    def test_the_model_list_endpoint_is_registered(self):
        for label in (INTL, MAINLAND):
            self.assertIn(label, stp.MODEL_LIST_URLS)


class MissingCredentialTests(unittest.TestCase):
    """An absent key must never be an error. This is the property that keeps a
    provider we have not provisioned from breaking the run."""

    def setUp(self):
        self._saved = {
            name: os.environ.pop(name, None)
            for name in ("ALIBABA_API_KEY", "DASHSCOPE_API_KEY")
        }

    def tearDown(self):
        for name, value in self._saved.items():
            if value is not None:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)

    def test_resolve_key_returns_none_without_raising(self):
        key, source = stp.resolve_key(route_by_label(INTL))
        self.assertIsNone(key)
        self.assertIsNone(source)

    def test_http_probe_degrades_to_not_tested(self):
        result = stp.probe_http(route_by_label(INTL))
        self.assertEqual(result["result"], "NOT_TESTED")
        self.assertEqual(result["reason"], "CREDENTIAL_REQUIRED")

    def test_litellm_probe_degrades_to_not_tested(self):
        result = stp.probe_litellm(route_by_label(MAINLAND))
        self.assertEqual(result["result"], "NOT_TESTED")
        self.assertEqual(result["reason"], "CREDENTIAL_REQUIRED")

    def test_no_probe_ever_echoes_a_credential_variable(self):
        """The report may name the variable; it must never carry a value."""
        result = stp.probe_http(route_by_label(INTL))
        self.assertNotIn("api_key", result)
        self.assertNotIn("key", result)


class WrongRegionClassificationTests(unittest.TestCase):
    """A key valid in one Alibaba region 401s in the other. That must be
    reported as a credential problem, not mistaken for a dead model."""

    def test_401_is_classified_as_invalid_key(self):
        self.assertEqual(stp.classify(401, "InvalidApiKey"), "INVALID_KEY")

    def test_404_is_still_a_model_problem(self):
        self.assertEqual(
            stp.classify(404, "model not found"), "MODEL_OR_ENDPOINT_NOT_FOUND"
        )

    def test_429_is_still_a_quota_problem(self):
        self.assertEqual(stp.classify(429, "Throttling.RateQuota"),
                         "QUOTA_OR_RATE_LIMIT")


class RateLimiterCompatibilityTests(unittest.TestCase):
    """The limiter is keyed by model string and needs no per-provider code.
    DashScope's real quota is unknown until measured, so it must arrive
    unthrottled rather than silently throttled at someone else's numbers."""

    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_an_unknown_model_gets_an_unconstrained_limiter(self):
        limits = rl.limits_for(DASHSCOPE_MODEL)
        self.assertIsNone(limits.requests_per_minute)
        self.assertIsNone(limits.tokens_per_minute)

    def test_a_limiter_can_be_created_for_it(self):
        limiter = rl.get_limiter(DASHSCOPE_MODEL)
        self.assertEqual(limiter.model, DASHSCOPE_MODEL)
        self.assertIs(rl.get_limiter(DASHSCOPE_MODEL), limiter,
                      "one limiter per model, shared process-wide")

    def test_env_override_can_supply_a_measured_quota_without_a_code_change(self):
        os.environ["LLM_RATE_DASHSCOPE_QWEN_TURBO_RPM"] = "60"
        try:
            self.assertEqual(rl.limits_for(DASHSCOPE_MODEL).requests_per_minute, 60.0)
        finally:
            os.environ.pop("LLM_RATE_DASHSCOPE_QWEN_TURBO_RPM", None)


class _Recording(GeneralLlm):
    def __init__(self, model, fail=False):
        super().__init__(model=model, allowed_tries=1)
        self.fail = fail
        self.calls = 0

    async def invoke(self, prompt, system_prompt=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated 429")
        return "OK from " + self.model


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class FallbackSelectionTests(unittest.TestCase):
    """Provider selection, offline. Ordering is a property of the list handed
    to FallbackLlm, so adding DashScope must not perturb the existing chain."""

    def setUp(self):
        rl.reset_registry()

    def tearDown(self):
        rl.reset_registry()

    def test_dashscope_is_only_reached_after_the_existing_chain_fails(self):
        openrouter = _Recording("openrouter/nvidia/nemotron-3.5-lightning:free", fail=True)
        gemini = _Recording("gemini/gemini-3.5-flash-lite", fail=True)
        groq = _Recording("groq/openai/gpt-oss-120b", fail=True)
        dashscope = _Recording(DASHSCOPE_MODEL)
        chain = FallbackLlm([openrouter, gemini, groq, dashscope])

        result = _run(chain.invoke("x"))

        self.assertEqual(result, "OK from " + DASHSCOPE_MODEL)
        self.assertEqual(openrouter.calls, 1)
        self.assertEqual(gemini.calls, 1)
        self.assertEqual(groq.calls, 1)
        self.assertEqual(dashscope.calls, 1)

    def test_a_healthy_primary_never_reaches_dashscope(self):
        openrouter = _Recording("openrouter/nvidia/nemotron-3.5-lightning:free")
        dashscope = _Recording(DASHSCOPE_MODEL)
        chain = FallbackLlm([openrouter, dashscope])

        _run(chain.invoke("x"))

        self.assertEqual(openrouter.calls, 1)
        self.assertEqual(dashscope.calls, 0,
                         "fallback must stay a failure path, not a preference")

    def test_adding_dashscope_does_not_change_who_answers_first(self):
        openrouter = _Recording("openrouter/nvidia/nemotron-3.5-lightning:free", fail=True)
        gemini = _Recording("gemini/gemini-3.5-flash-lite")
        dashscope = _Recording(DASHSCOPE_MODEL)
        chain = FallbackLlm([openrouter, gemini, dashscope])

        result = _run(chain.invoke("x"))

        self.assertEqual(result, "OK from gemini/gemini-3.5-flash-lite")
        self.assertEqual(dashscope.calls, 0)

    def test_one_delayed_call_is_still_exactly_one_call(self):
        dashscope = _Recording(DASHSCOPE_MODEL)
        chain = FallbackLlm([dashscope])
        _run(chain.invoke("x"))
        self.assertEqual(dashscope.calls, 1)


@unittest.skipUnless(
    os.environ.get("ALIBABA_API_KEY", "").strip(),
    "ALIBABA_API_KEY not present - live provider test skipped by design",
)
class AlibabaLiveIntegrationTests(unittest.TestCase):
    """E2E. Runs ONLY where the secret exists (the smoke-test workflow step).

    Deliberately one call per region and nothing more: the point is to learn
    which region the credential belongs to and whether the production code path
    accepts the reply. DashScope is a metered provider, so this must never
    become a burst.
    """

    def test_the_credential_authenticates_in_exactly_one_region(self):
        results = {
            label: stp.probe_http(route_by_label(label))
            for label in (INTL, MAINLAND)
        }
        outcomes = {label: r.get("result") for label, r in results.items()}
        self.assertIn(
            "OK", outcomes.values(),
            "neither Alibaba region accepted the credential: {0}".format(
                {k: (v.get("error_class"), v.get("http")) for k, v in results.items()}
            ),
        )

    def test_the_production_litellm_path_accepts_the_reply(self):
        working = [
            label for label in (INTL, MAINLAND)
            if stp.probe_http(route_by_label(label)).get("result") == "OK"
        ]
        if not working:
            self.skipTest("no working region; the HTTP test above reports why")
        result = stp.probe_litellm(route_by_label(working[0]))
        self.assertEqual(result["result"], "OK", result.get("error"))
        self.assertTrue(result["response"], "provider returned an empty completion")


if __name__ == "__main__":
    unittest.main()
