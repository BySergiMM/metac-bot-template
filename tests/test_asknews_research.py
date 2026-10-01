"""Live news for the researcher, gated on an AskNews credential.

Two halves, matching the two places the feature lives:

  pin_models   with a credential at patch time the researcher's chain leads
               with asknews/news-summaries and the LLM researcher stands
               directly behind it; without one the generated block is
               byte-for-byte what it was before the feature existed.

  runtime      an AskNews failure of any kind costs one attempt and the LLM
               produces the research -- proven with the REAL GeneralLlm and
               the REAL FallbackLlm, never a mock of either, because the
               property depends on GeneralLlm routing that model string to
               AskNewsSearcher and raising from it.

pin_models reads os.environ at import time, so the generator tests reload it
under a controlled environment (test_pin_models_chains.load / pinned_env).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import unittest

from tests._real_forecasting_tools import real_forecasting_tools
from tests.test_pin_models_chains import (
    GEMINI,
    GROQ_OSS,
    HAIKU,
    chains_by_role,
    load,
    pinned_env,
)

ASKNEWS = "asknews/news-summaries"
PRODUCTION = dict(OPENROUTER_API_KEY="o", GEMINI_API_KEY="g", GROQ_API_KEY="q")


# ------------------------------------------------------------- generation


class GatingTests(unittest.TestCase):
    """The feature is on exactly when AskNewsSearcher could authenticate."""

    def test_off_without_any_asknews_variable(self):
        pm = load(**PRODUCTION)
        self.assertFalse(pm.ACTIVE_ASKNEWS)

    def test_on_with_an_api_key(self):
        pm = load(ASKNEWS_API_KEY="k")
        self.assertTrue(pm.ACTIVE_ASKNEWS)

    def test_on_with_the_oauth_pair(self):
        pm = load(ASKNEWS_CLIENT_ID="id", ASKNEWS_SECRET="s")
        self.assertTrue(pm.ACTIVE_ASKNEWS)

    def test_half_an_oauth_pair_is_off(self):
        """AskNewsSearcher needs both halves; leading with a backend that is
        certain to raise would be a wasted attempt on every question."""
        for env in (dict(ASKNEWS_CLIENT_ID="id"), dict(ASKNEWS_SECRET="s")):
            with self.subTest(env=sorted(env)):
                self.assertFalse(load(**env).ACTIVE_ASKNEWS)

    def test_an_empty_value_is_off(self):
        """The workflows pass `${{ secrets.X }}` for a secret that does not
        exist, which arrives as an empty string, not an absent variable."""
        pm = load(ASKNEWS_API_KEY="", ASKNEWS_CLIENT_ID="", ASKNEWS_SECRET="")
        self.assertFalse(pm.ACTIVE_ASKNEWS)

    def test_both_credential_styles_at_once_is_off(self):
        """AskNewsSearcher refuses the combination outright."""
        pm = load(ASKNEWS_API_KEY="k", ASKNEWS_CLIENT_ID="id", ASKNEWS_SECRET="s")
        self.assertFalse(pm.ACTIVE_ASKNEWS)

    def test_the_predicate_matches_the_searcher_it_gates(self):
        """Same answer as the REAL AskNewsSearcher constructor, for every
        combination of the three variables being set or not: the gate opens
        exactly when the searcher would authenticate."""
        with real_forecasting_tools():
            from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
        pm = load()
        for bits in range(8):
            env = {
                name: ("v" if bits & (1 << index) else "")
                for index, name in enumerate(pm.ASKNEWS_ENV_VARS)
            }
            with self.subTest(env={k: bool(v) for k, v in env.items()}):
                with _without_asknews_env():
                    try:
                        AskNewsSearcher(
                            api_key=env["ASKNEWS_API_KEY"] or None,
                            client_id=env["ASKNEWS_CLIENT_ID"] or None,
                            client_secret=env["ASKNEWS_SECRET"] or None,
                        )
                    except ValueError:
                        searcher_authenticates = False
                    else:
                        searcher_authenticates = True
                self.assertEqual(
                    pm._asknews_credential_present(env), searcher_authenticates
                )


class GeneratedChainTests(unittest.TestCase):
    def test_without_a_credential_the_block_is_unchanged(self):
        """Byte-for-byte: the no-AskNews path is today's production path."""
        for label, env in (
            ("production", PRODUCTION),
            ("no keys at all", {}),
            ("gemini and groq", dict(GEMINI_API_KEY="g", GROQ_API_KEY="q")),
        ):
            with self.subTest(env=label):
                pm = load(**env)
                block = pm.build_block(pm.DEFAULTS)
                self.assertNotIn("asknews", block)

    def test_with_a_credential_the_researcher_leads_with_asknews(self):
        pm = load(ASKNEWS_API_KEY="k", **PRODUCTION)
        chains = chains_by_role(pm, pm.DEFAULTS)
        self.assertEqual(chains["researcher"], [[ASKNEWS, HAIKU, GEMINI, GROQ_OSS]])

    def test_the_llm_researcher_is_the_first_fallback(self):
        """Research must degrade to exactly what it was, not to a different
        model: the configured researcher stands directly behind the feed."""
        pm = load(ASKNEWS_API_KEY="k", **PRODUCTION)
        chains = chains_by_role(pm, pm.DEFAULTS)
        self.assertEqual(chains["researcher"][0][1], pm.DEFAULTS["researcher"])

    def test_only_the_researcher_changes(self):
        """News is not reasoning. The forecaster ensemble, the summarizer and
        the parser must be exactly what they are without the credential."""
        with_news = chains_by_role(load(ASKNEWS_API_KEY="k", **PRODUCTION),
                                   load(**PRODUCTION).DEFAULTS)
        without = chains_by_role(load(**PRODUCTION), load(**PRODUCTION).DEFAULTS)
        for role in ("default", "summarizer", "parser"):
            with self.subTest(role=role):
                self.assertEqual(with_news[role], without[role])
                for chain in with_news[role]:
                    self.assertNotIn(ASKNEWS, chain)

    def test_the_chain_exists_with_no_fallback_key_at_all(self):
        """The case the import logic had to change for: an AskNews failure
        must still land on the LLM when there is no provider fallback."""
        pm = load(ASKNEWS_API_KEY="k")
        chains = chains_by_role(pm, pm.DEFAULTS)
        self.assertEqual(chains["researcher"], [[ASKNEWS, pm.DEFAULTS["researcher"]]])
        self.assertEqual(chains["default"], [[pm.DEFAULTS["default"]]])
        block = pm.build_block(pm.DEFAULTS)
        self.assertIn("FallbackLlm([", block)
        self.assertNotIn("BalancedLlm", block)
        # The parser keeps the bare-string form the no-fallback path promises.
        self.assertIn('"parser": "{0}",'.format(pm.DEFAULTS["parser"]), block)

    def test_the_fallback_import_follows_the_chain(self):
        """FallbackLlm( without its import is a NameError at startup."""
        src = _main_src()
        for env in (dict(ASKNEWS_API_KEY="k"), dict(ASKNEWS_API_KEY="k", **PRODUCTION)):
            with self.subTest(env=sorted(env)):
                pm = load(**env)
                generated = pm.patch(src, pm.DEFAULTS)
                self.assertIn("from backtest.fallback_llm import FallbackLlm", generated)
                self.assertEqual(pm.patch(generated, pm.DEFAULTS), generated,
                                 "patch must be idempotent")

    def test_adding_and_removing_the_credential_converges(self):
        """A main.py patched under one credential set must be rewritten, in
        place, when the set changes -- in both directions."""
        src = _main_src()
        pm_plain = load(**PRODUCTION)
        plain = pm_plain.patch(src, pm_plain.DEFAULTS)
        self.assertNotIn(ASKNEWS, _llms_block(pm_plain, plain))

        pm_news = load(ASKNEWS_API_KEY="k", **PRODUCTION)
        gained = pm_news.patch(plain, pm_news.DEFAULTS)
        self.assertIn(ASKNEWS, _llms_block(pm_news, gained))
        self.assertEqual(gained, pm_news.patch(src, pm_news.DEFAULTS))

        pm_plain = load(**PRODUCTION)
        lost = pm_plain.patch(gained, pm_plain.DEFAULTS)
        self.assertNotIn(ASKNEWS, _llms_block(pm_plain, lost))
        self.assertEqual(lost, plain)

    def test_no_credential_value_reaches_the_block(self):
        pm = load(ASKNEWS_API_KEY="SECRET-NEWS", ASKNEWS_CLIENT_ID="SECRET-ID",
                  ASKNEWS_SECRET="SECRET-S", **PRODUCTION)
        generated = pm.patch(_main_src(), pm.DEFAULTS)
        for value in ("SECRET-NEWS", "SECRET-ID", "SECRET-S"):
            self.assertNotIn(value, generated)

    def test_an_override_naming_asknews_itself_does_not_repeat_it(self):
        """models.txt may set `researcher: asknews/news-summaries` by hand,
        which was already possible before this feature."""
        pm = load(ASKNEWS_API_KEY="k", **PRODUCTION)
        chains = chains_by_role(pm, dict(pm.DEFAULTS, researcher=ASKNEWS))
        self.assertEqual(chains["researcher"], [[ASKNEWS, HAIKU, GEMINI, GROQ_OSS]])

    def test_the_asknews_bucket_key_is_registered(self):
        """FallbackLlm acquires a limiter keyed by the model string; an
        unregistered key is an unthrottled limiter that looks controlled."""
        from backtest import rate_limiter

        self.assertIn(ASKNEWS, rate_limiter.DEFAULT_LIMITS)
        self.assertIn(ASKNEWS, load().RATE_LIMITED_MODELS)


class SelftestTests(unittest.TestCase):
    """pin_models.selftest() runs before every patch, under whatever
    credentials the step has. It must pass with AskNews present, in every
    combination the workflows can produce, or the pin step crashes."""

    def test_selftest_passes_with_asknews_in_every_combination(self):
        for label, env in (
            ("api key only", dict(ASKNEWS_API_KEY="k")),
            ("oauth pair only", dict(ASKNEWS_CLIENT_ID="i", ASKNEWS_SECRET="s")),
            ("api key + production", dict(ASKNEWS_API_KEY="k", **PRODUCTION)),
            ("api key + pin step", dict(ASKNEWS_API_KEY="k", GEMINI_API_KEY="g",
                                        GROQ_API_KEY="q")),
            ("api key + openrouter only", dict(ASKNEWS_API_KEY="k",
                                               OPENROUTER_API_KEY="o")),
        ):
            with self.subTest(env=label):
                with pinned_env(**env) as pm:
                    pm.selftest()


class PreflightTests(unittest.TestCase):
    """`pin_models.py --check` must ping AskNews through AskNews, and must not
    declare the researcher dead when only the feed is."""

    def test_check_pings_asknews_only_when_active(self):
        for env, expect in ((PRODUCTION, False),
                            (dict(ASKNEWS_API_KEY="k", **PRODUCTION), True)):
            with self.subTest(active=expect):
                with pinned_env(**env) as pm:
                    pinged = []
                    pm._ping = lambda model: pinged.append(model)  # all alive
                    pm.check_models(pm.DEFAULTS)
                    self.assertEqual(ASKNEWS in pinged, expect)

    def test_a_dead_feed_with_a_live_llm_is_not_a_dead_role(self):
        with pinned_env(ASKNEWS_API_KEY="k", **PRODUCTION) as pm:
            pm._ping = lambda model: "unreachable" if model == ASKNEWS else None
            self.assertEqual(pm.check_models(pm.DEFAULTS), [])

    def test_a_dead_feed_and_dead_llms_is_a_dead_researcher(self):
        with pinned_env(ASKNEWS_API_KEY="k") as pm:
            pm._ping = lambda model: "unreachable"
            self.assertIn("researcher", pm.check_models(pm.DEFAULTS))

    def test_the_asknews_ping_does_not_go_through_litellm(self):
        """litellm has no AskNews provider: routed there, the ping would fail
        on every run and say nothing about the credential."""
        with pinned_env(ASKNEWS_API_KEY="k") as pm:
            asked = []
            pm._ping_asknews = lambda model: asked.append(model) or None
            self.assertIsNone(pm._ping(ASKNEWS))
            self.assertEqual(asked, [ASKNEWS])


# ---------------------------------------------------------------- runtime


with real_forecasting_tools():
    from forecasting_tools.ai_models import general_llm as _general_llm
    from forecasting_tools.ai_models.general_llm import GeneralLlm

    from backtest.fallback_llm import FallbackLlm


class _LlmResearcher(GeneralLlm):
    """Stands in for the Haiku researcher: a real GeneralLlm subclass whose
    invoke never touches the network."""

    def __init__(self):
        super().__init__(model=HAIKU, temperature=0.3, timeout=180, allowed_tries=1)
        self.calls = 0

    async def invoke(self, prompt, system_prompt=None):  # noqa: ARG002
        self.calls += 1
        return "LLM RESEARCH"


class _FakeSearcher:
    """AskNewsSearcher stand-in, installed where GeneralLlm looks it up."""

    outcome: object = None

    async def call_preconfigured_version(self, preset, prompt):
        assert preset == ASKNEWS
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@contextlib.contextmanager
def _asknews_returns(outcome):
    """Route GeneralLlm's AskNews call to a fake returning `outcome`."""
    original = _general_llm.AskNewsSearcher
    _FakeSearcher.outcome = outcome
    _general_llm.AskNewsSearcher = _FakeSearcher
    try:
        yield
    finally:
        _general_llm.AskNewsSearcher = original


@contextlib.contextmanager
def _without_asknews_env():
    saved = {name: os.environ.pop(name, None)
             for name in ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET")}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def _researcher_chain():
    """[asknews, llm], built exactly as the generated main.py builds it."""
    with real_forecasting_tools():
        feed = GeneralLlm(model=ASKNEWS, temperature=0.3, timeout=180, allowed_tries=1)
        llm = _LlmResearcher()
        return FallbackLlm([feed, llm]), llm


class RuntimeFallbackTests(unittest.TestCase):
    def setUp(self):
        from backtest import rate_limiter

        rate_limiter.reset_registry()

    def test_the_model_string_is_routed_to_asknews_by_the_real_generalllm(self):
        """The whole design rests on this: no new class, because the
        installed forecasting_tools already treats the string as AskNews."""
        with real_forecasting_tools():
            feed = GeneralLlm(model=ASKNEWS, temperature=0.3, timeout=180,
                              allowed_tries=1)
        self.assertTrue(feed._use_asknews)

    def test_news_is_used_when_asknews_answers(self):
        chain, llm = _researcher_chain()
        with _asknews_returns("Here are the relevant news articles:\n\nNEWS BODY"):
            result = asyncio.run(chain.invoke("question"))
        self.assertIn("NEWS BODY", result)
        self.assertEqual(llm.calls, 0, "the LLM must not run when the feed answered")

    def test_the_llm_answers_when_asknews_raises(self):
        """Any exception: the SDK's own errors, a rate limit, a timeout."""
        chain, llm = _researcher_chain()
        for exc in (RuntimeError("AskNews 429"), TimeoutError("slow"),
                    ValueError("Cannot use both OAuth and API key")):
            with self.subTest(exc=type(exc).__name__):
                llm.calls = 0
                with _asknews_returns(exc):
                    result = asyncio.run(chain.invoke("question"))
                self.assertEqual(result, "LLM RESEARCH")
                self.assertEqual(llm.calls, 1)

    def test_the_llm_answers_when_no_credential_is_present_at_runtime(self):
        """The real AskNewsSearcher, not a fake: with no credential it raises
        from its constructor, and that raise must land on the LLM too. This
        is the shape of a credential that reaches the pin step but not the
        run step."""
        chain, llm = _researcher_chain()
        with _without_asknews_env():
            result = asyncio.run(chain.invoke("question"))
        self.assertEqual(result, "LLM RESEARCH")
        self.assertEqual(llm.calls, 1)

    def test_one_call_in_one_string_out(self):
        """The fallback changes availability, nothing else: the chain is still
        a GeneralLlm subclass, so main.py's isinstance() branch takes it and
        the research summary, prediction count and aggregation are untouched.

        Read from the source: earlier test modules install a GeneralLlm stub
        before importing backtest.fallback_llm, so which class OBJECT the
        chain descends from depends on module order, while the declaration
        does not."""
        import io
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = io.open(os.path.join(root, "backtest", "fallback_llm.py"),
                      encoding="utf-8").read()
        self.assertIn("class FallbackLlm(GeneralLlm):", src)
        chain, _llm = _researcher_chain()
        self.assertEqual(chain.model, ASKNEWS)

    def test_a_feed_failure_is_logged_bounded_and_content_free(self):
        """Logs are public. The failure line names the provider and a short
        reason; it never carries the prompt, the research or the full error
        body (which an SDK can fill with the request it made)."""
        chain, _llm = _researcher_chain()
        body = "x" * 500 + " SECRET-PROMPT-ECHO"
        with self.assertLogs("backtest.fallback_llm", level=logging.INFO) as logs:
            with _asknews_returns(RuntimeError(body)):
                asyncio.run(chain.invoke("THE QUESTION PROMPT"))
        text = "\n".join(logs.output)
        self.assertIn("llm_failure provider_index=0 provider=" + ASKNEWS, text)
        self.assertIn("llm_success provider_index=1 provider=" + HAIKU, text)
        self.assertNotIn("SECRET-PROMPT-ECHO", text)
        self.assertNotIn("THE QUESTION PROMPT", text)
        self.assertNotIn("LLM RESEARCH", text)


def _llms_block(pm, generated: str) -> str:
    """The active llms={...} block only. main.py itself mentions the AskNews
    preset in its docstring and in run_research, so a whole-file search says
    nothing about what was generated."""
    span = pm._llms_block_span(generated)
    assert span is not None, "no active llms block"
    return generated[span[0]:span[1]]


def _main_src() -> str:
    import io
    import os.path

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return io.open(os.path.join(root, "main.py"), encoding="utf-8").read()


if __name__ == "__main__":
    unittest.main()
