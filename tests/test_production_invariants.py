"""Guarantees this bot makes to production, asserted as guarantees.

Every other test file checks that a component behaves. This one checks that
the SYSTEM still promises what an operator was told it promises. The tests are
deliberately written against properties rather than implementations, so a
refactor that preserves the guarantee passes and a refactor that quietly drops
it fails.

Grouped by the promise, not by the module:

  Security      no secret, rationale, probability or provider body in a log
  Integrity     five predictions stay five; nothing partial is ever published
  Publication   one prediction, one comment, orphans named out loud
  Concurrency   loop-safe semaphore, bounded research, isolated quota buckets
  Discovery     every page read, no question processed twice
  Workflow      scored tournaments only from the one production workflow
  Deployment    the code CI generates still keeps all of the above
"""

from __future__ import annotations

import ast
import io
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def parse(*parts: str) -> ast.Module:
    return ast.parse(read(*parts))


def yaml_without_comments(*parts: str) -> str:
    """Workflow text with `#` comment lines removed.

    Necessary because these workflows document what they deliberately do NOT
    contain -- research_fallback_e2e.yaml literally says "No `schedule:` key
    anywhere in this file, on purpose" -- so a plain substring search finds
    the prose rather than the key.
    """
    kept = []
    for line in read(*parts).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        kept.append(line.split(" #", 1)[0])
    return "\n".join(kept)


def find_function(tree: ast.AST, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


PRODUCTION_SOURCES = (
    ("main.py",),
    ("bot_helpers.py",),
    ("publication.py",),
    ("discovery.py",),
    ("backtest", "fallback_llm.py"),
    ("backtest", "balanced_llm.py"),
    ("backtest", "rate_limiter.py"),
    ("backtest", "pin_models.py"),
)


# ===========================================================  SECURITY


class NoCredentialMaterialInProductionCode(unittest.TestCase):
    """Credentials are referenced by env var NAME, never by value."""

    #: Shapes of real provider credentials. Deliberately requires enough
    #: trailing characters that a prose mention ("an AIza key") cannot match.
    CREDENTIAL_SHAPES = (
        re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
        re.compile(r"\bgsk_[0-9A-Za-z]{20,}"),
        re.compile(r"\bsk-or-v1-[0-9A-Za-z]{20,}"),
        re.compile(r"\bsk-[0-9A-Za-z]{32,}"),
    )

    def test_no_production_file_contains_credential_shaped_text(self):
        for parts in PRODUCTION_SOURCES:
            src = read(*parts)
            for shape in self.CREDENTIAL_SHAPES:
                self.assertIsNone(
                    shape.search(src),
                    "{0} contains credential-shaped text".format("/".join(parts)),
                )

    def test_the_generator_emits_env_var_names_not_values(self):
        """_bucket_expr must interpolate the env var NAME. The behavioural
        proof is GeneratedCodeInvariants; this pins the mechanism, because a
        generator that switched to os.environ.get() there would bake a live
        credential into a file."""
        tree = parse("backtest", "pin_models.py")
        node = find_function(tree, "_bucket_expr")
        self.assertIsNotNone(node, "_bucket_expr is gone")
        body = ast.unparse(node)
        self.assertIn("env_var", body, "the env var NAME must be interpolated")
        self.assertNotIn("os.environ", body)
        self.assertNotIn("os.getenv", body)

    def test_bucket_backend_reads_the_credential_at_runtime(self):
        src = read("backtest", "balanced_llm.py")
        self.assertIn("os.environ.get(api_key_env)", src)


class NoForecastContentReachesLogs(unittest.TestCase):
    """The full proof lives in test_forecast_content_leakage; these are the
    structural preconditions that make it hold."""

    def test_both_redaction_layers_exist(self):
        src = read("bot_helpers.py")
        self.assertIn("def log_forecast_content", src, "layer 1 (source)")
        self.assertIn("class RedactForecastContent", src, "layer 2 (filter)")

    def test_the_source_layer_fails_closed(self):
        src = read("bot_helpers.py")
        self.assertIn("_withhold_forecast_content = True", src)

    def test_the_banner_scrubs_rather_than_truncates(self):
        src = read("bot_helpers.py")
        self.assertIn("scrub(exc)", src)
        self.assertNotIn("msg = msg[:200]", src,
                         "the unchecked truncation must not come back")

    def test_provider_errors_are_bounded_everywhere_they_are_logged(self):
        self.assertIn("def _reason", read("backtest", "fallback_llm.py"))
        self.assertIn("def _safe_error", read("publication.py"))


# ===========================================================  INTEGRITY


class ForecastIntegrityInvariants(unittest.TestCase):
    def test_five_predictions_per_research_report(self):
        self.assertIn("predictions_per_research_report=5,", read("main.py"))

    def test_one_research_report_per_question(self):
        self.assertIn("research_reports_per_question=1,", read("main.py"))

    def test_publishing_is_enabled(self):
        self.assertIn("publish_to_metaculus = True", read("main.py"))

    def test_the_success_threshold_is_not_overridden(self):
        """required_successful_predictions defaults to 0.5 upstream, so 5
        predictions need at least 3 successes or nothing is published at all.
        Overriding it downward would let a thinner forecast reach Metaculus."""
        self.assertNotIn("required_successful_predictions", read("main.py"))

    def test_the_fallback_layers_return_one_string_per_call(self):
        """Neither wrapper may turn one logical call into two predictions."""
        for module in (("backtest", "fallback_llm.py"), ("backtest", "balanced_llm.py")):
            tree = parse(*module)
            invoke = find_function(tree, "invoke")
            self.assertIsNotNone(invoke, module)
            returns = [n for n in ast.walk(invoke) if isinstance(n, ast.Return)]
            self.assertTrue(returns, module)
            for node in returns:
                self.assertIsInstance(
                    node.value, ast.Name,
                    "{0}.invoke must return the single backend result".format(module),
                )

    def test_neither_wrapper_knows_what_a_forecast_is(self):
        for module in (("backtest", "fallback_llm.py"), ("backtest", "balanced_llm.py"),
                       ("backtest", "rate_limiter.py")):
            src = read(*module)
            for forbidden in ("post_question_comment", "post_binary_question_prediction",
                              "publish_report_to_metaculus", "aggregate_predictions"):
                self.assertNotIn(forbidden, src, "/".join(module))


# ===========================================================  PUBLICATION


class PublicationInvariants(unittest.TestCase):
    def test_the_bot_uses_the_publishing_client(self):
        src = read("main.py")
        self.assertIn("client = PublishingClient()", src)
        self.assertIn("metaculus_client=client", src)

    def test_duplicate_guards_exist_on_both_write_paths(self):
        src = read("publication.py")
        self.assertIn("_commented_posts", src)
        self.assertIn("_predicted_questions", src)

    def test_the_orphan_marker_is_a_stable_token(self):
        from publication import ORPHAN_MARKER

        self.assertEqual(ORPHAN_MARKER, "PUBLICATION_ORPHAN")

    def test_the_comment_gets_more_attempts_than_the_sdk_alone(self):
        from publication import COMMENT_ATTEMPTS

        self.assertGreater(COMMENT_ATTEMPTS, 1)

    def test_comments_are_never_forced_public(self):
        """post_question_comment defaults is_private=True; production must not
        override it. Metaculus publishes bot comments on its own schedule."""
        for parts in PRODUCTION_SOURCES:
            self.assertNotIn("is_private=False", read(*parts), "/".join(parts))

    def test_the_orphan_report_runs_at_the_end_of_every_run(self):
        self.assertIn("print_publication_report(client)", read("main.py"))


# ===========================================================  CONCURRENCY


class ConcurrencyInvariants(unittest.TestCase):
    """R5's intended properties, asserted structurally.

    The behavioural proof is tests/test_concurrency_limiter_loops.py. These
    catch the two mistakes that would reintroduce the bug while still passing
    a behavioural test on a single event loop.
    """

    def _limiter_property(self) -> ast.AST:
        tree = parse("main.py")
        node = find_function(tree, "_concurrency_limiter")
        self.assertIsNotNone(node, "the limiter property is gone")
        return node

    def test_the_property_contains_no_await_or_yield(self):
        """An await inside would let two tasks interleave between the check
        and the assignment, producing two semaphores for one loop."""
        node = self._limiter_property()
        for child in ast.walk(node):
            self.assertNotIsInstance(child, ast.Await)
            self.assertNotIsInstance(child, ast.Yield)
            self.assertNotIsInstance(child, ast.YieldFrom)

    def test_no_semaphore_is_constructed_at_import_time(self):
        """A Semaphore built at class-definition time binds to whichever loop
        first waits on it, which is the original R5 bug."""
        tree = parse("main.py")
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                # A CALL, not a mention. The declaration carries the quoted
                # annotation "asyncio.Semaphore | None", which is exactly what
                # the lazy version is supposed to look like.
                for child in ast.walk(statement):
                    if not isinstance(child, ast.Call):
                        continue
                    called = getattr(child.func, "attr", None) or getattr(
                        child.func, "id", None
                    )
                    self.assertNotEqual(
                        called, "Semaphore",
                        "class-level Semaphore() rebinds the original R5 bug",
                    )

    def test_the_property_binds_to_the_running_loop(self):
        src = ast.unparse(self._limiter_property())
        self.assertIn("get_running_loop", src)

    def test_it_does_not_replace_a_live_semaphore_outside_a_loop(self):
        src = ast.unparse(self._limiter_property())
        self.assertIn("_concurrency_limiter_instance is not None", src)

    def test_research_concurrency_stays_at_one(self):
        self.assertIn("_max_concurrent_questions = (\n        1", read("main.py"))

    def test_load_order_contains_no_await_or_yield(self):
        """R9: bucket selection must be atomic under asyncio. A suspension
        point between reading the loads and using them would let every
        concurrent caller pick the same bucket."""
        node = find_function(parse("backtest", "balanced_llm.py"), "_load_order")
        self.assertIsNotNone(node)
        for child in ast.walk(node):
            self.assertNotIsInstance(child, ast.Await)
            self.assertNotIsInstance(child, ast.Yield)

    def test_every_gemini_bucket_has_an_explicit_quota(self):
        """An unregistered bucket key resolves to a limiter with NO limit."""
        from backtest.rate_limiter import DEFAULT_LIMITS, GEMINI_BUCKET_KEYS

        for key in GEMINI_BUCKET_KEYS:
            self.assertIn(key, DEFAULT_LIMITS)
            self.assertEqual(DEFAULT_LIMITS[key].requests_per_minute, 15.0)


# ===========================================================  DISCOVERY


class DiscoveryInvariants(unittest.TestCase):
    def test_the_bot_does_not_use_the_truncating_upstream_discovery(self):
        """PublishingClient must override the single-page method."""
        src = read("publication.py")
        self.assertIn("def get_all_open_questions_from_tournament", src)
        self.assertIn("fetch_all_open_questions", src)

    def test_truncation_is_reported_and_never_silent(self):
        src = read("discovery.py")
        self.assertIn("discovery_page_limit_reached", src)
        self.assertIn("logger.error", src)

    def test_the_page_ceiling_is_explicit(self):
        from discovery import MAX_PAGES

        self.assertIsInstance(MAX_PAGES, int)
        self.assertGreater(MAX_PAGES, 1)

    def test_deduplication_is_on_the_question_not_the_post(self):
        """Group subquestions share a post id; deduping on it loses forecasts."""
        src = read("discovery.py")
        self.assertIn("seen_question_ids", src)
        self.assertNotIn("seen_post_ids", src)


# ===========================================================  WORKFLOW


class WorkflowInvariants(unittest.TestCase):
    PRODUCTION = (".github", "workflows", "run_bot_on_tournament.yaml")
    CUP = (".github", "workflows", "run_bot_on_metaculus_cup.yaml")
    TEST_BOT = (".github", "workflows", "test_bot.yaml")
    E2E = (".github", "workflows", "research_fallback_e2e.yaml")

    def test_production_runs_the_scored_tournament_mode(self):
        src = read(*self.PRODUCTION)
        self.assertIn("python main.py", src)
        self.assertNotIn("--mode test_questions", src)
        self.assertNotIn("--mode metaculus_cup", src)

    def test_production_is_least_privilege(self):
        self.assertIn("permissions:\n  contents: read", read(*self.PRODUCTION))

    def test_production_serialises_itself(self):
        src = read(*self.PRODUCTION)
        self.assertIn("concurrency:", src)
        self.assertIn("cancel-in-progress: false", src)

    def test_the_unscored_workflows_target_only_the_practice_area(self):
        for parts in (self.TEST_BOT, self.E2E):
            src = read(*parts)
            self.assertIn("--mode test_questions", src, "/".join(parts))

    def test_no_unscored_workflow_is_scheduled(self):
        for parts in (self.TEST_BOT, self.E2E):
            src = yaml_without_comments(*parts)
            self.assertNotIn("schedule:", src, "/".join(parts))
            self.assertNotIn("cron:", src, "/".join(parts))

    def test_exactly_one_workflow_publishes_to_a_scored_tournament(self):
        """The Cup workflow also publishes, so it must stay off the schedule
        of anything CI can start by itself. It is disabled at the GitHub level
        (disabled_fork); this asserts the repository-side half."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        scored = []
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            runs_bot = "python main.py" in src
            unscored = "--mode test_questions" in src
            scheduled = "cron:" in src
            if runs_bot and not unscored and scheduled:
                scored.append(name)
        self.assertEqual(
            scored, ["run_bot_on_tournament.yaml"],
            "exactly one scheduled workflow may publish to a scored tournament",
        )

    def test_the_cup_workflow_is_manual_only(self):
        """It publishes to a scored tournament under the same token, in a
        different concurrency group, without the model-pinning step."""
        src = yaml_without_comments(*self.CUP)
        self.assertIn("--mode metaculus_cup", src)
        self.assertNotIn("schedule:", src)
        self.assertNotIn("cron:", src)

    def test_every_workflow_declares_least_privilege(self):
        """Without an explicit grant a job inherits the repository-wide
        default, which is broader than any job here needs."""
        for name in sorted(os.listdir(os.path.join(ROOT, ".github", "workflows"))):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            self.assertIn("permissions:", src, name)
            self.assertIn("contents: read", src, name)

    def test_a_ci_gate_runs_the_suite_on_push_and_pull_request(self):
        src = yaml_without_comments(".github", "workflows", "ci.yaml")
        self.assertIn("push:", src)
        self.assertIn("pull_request:", src)
        self.assertIn("unittest discover -s tests -t .", src)

    def test_the_ci_gate_covers_both_supported_python_versions(self):
        src = yaml_without_comments(".github", "workflows", "ci.yaml")
        self.assertIn('"3.11"', src)
        self.assertIn('"3.12"', src)

    def test_the_ci_gate_holds_no_credentials_and_runs_no_bot(self):
        """A test job that could authenticate could also publish.

        Comment-stripped: ci.yaml documents in prose that a future edit adding
        `env: secrets.*` must be caught, and that mention is not a grant."""
        src = yaml_without_comments(".github", "workflows", "ci.yaml")
        self.assertNotIn("secrets.", src)
        self.assertNotIn("python main.py", src)

    def test_production_uses_exactly_one_gemini_credential(self):
        """R11. Four credentials from four Google projects give one
        application 60 RPM where one project allows 15. Google documents that
        limits are per project, and the APIs ToS 2.d forbids attempting to
        circumvent them, but no Google source says whether one workload across
        several projects is circumvention. Unresolved, so production carries
        one bucket. The code path and the secrets both remain; only the
        production wiring is reduced. See the note in the workflow."""
        src = yaml_without_comments(*self.PRODUCTION)
        self.assertIn("GEMINI_API_KEY:", src, "the fallback leg must stay")
        for extra in ("GEMINI2_API_KEY:", "GEMINI3_API_KEY:", "GEMINI4_API_KEY:"):
            self.assertNotIn(
                extra, src,
                "additional Gemini projects must not reach production while "
                "the quota-circumvention question is unresolved",
            )

    def test_one_credential_generates_the_pre_bucket_block(self):
        """The mitigation must be a real reduction, not a cosmetic one:
        with a single credential pin_models must emit no BalancedLlm."""
        import importlib
        import sys

        saved = {
            key: os.environ.get(key)
            for key in ("GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY",
                        "GEMINI4_API_KEY", "GROQ_API_KEY")
        }
        for key in saved:
            os.environ.pop(key, None)
        os.environ["GEMINI_API_KEY"] = "one"
        os.environ["GROQ_API_KEY"] = "g"
        try:
            sys.modules.pop("backtest.pin_models", None)
            pin_models = importlib.import_module("backtest.pin_models")
            self.assertFalse(pin_models.BALANCED)
            generated = pin_models.patch(read("main.py"), pin_models.DEFAULTS)
            self.assertNotIn("BalancedLlm(", generated)
            self.assertIn("FallbackLlm([", generated)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
            sys.modules.pop("backtest.pin_models", None)

    def test_every_workflow_that_can_publish_to_a_scored_tournament_pins_models(self):
        """Without pin_models, forecasting-tools assigns the researcher role to
        a model OpenRouter no longer serves and every question 404s -- after
        the run has already been dispatched against real questions under the
        real token. The Cup is scored, so it needs the same protection as the
        tournament even though it is manual-only."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            if "python main.py" not in src:
                continue
            if "--mode test_questions" in src:
                continue  # bot-testing-area is unscored
            self.assertIn("pin_models.py", src,
                          name + " can publish to a scored tournament without "
                          "pinning models")

    def test_no_scored_publisher_carries_extra_gemini_credentials(self):
        """R11 applies to every workflow that can publish, not just the
        scheduled one."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            if "python main.py" not in src or "--mode test_questions" in src:
                continue
            for extra in ("GEMINI2_API_KEY:", "GEMINI3_API_KEY:", "GEMINI4_API_KEY:"):
                self.assertNotIn(extra, src, name)

    def test_no_workflow_grants_write_permissions_to_the_bot(self):
        for name in sorted(os.listdir(os.path.join(ROOT, ".github", "workflows"))):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = read(".github", "workflows", name)
            self.assertNotIn("contents: write", src, name)
            self.assertNotIn("permissions: write-all", src, name)


# ===========================================================  DEPLOYMENT


class GeneratedCodeInvariants(unittest.TestCase):
    """pin_models.py rewrites main.py inside the runner, before the bot runs.

    That makes it the closest thing this repo has to a supply-chain step: the
    code that executes in production is not the code in git. Everything the
    other tests assert about main.py has to survive the rewrite.
    """

    def _generate(self, **env):
        """Patch a real main.py with a real environment and return the result."""
        import importlib
        import sys

        saved = {k: os.environ.get(k) for k in (
            "GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY",
            "GEMINI4_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
        )}
        for key in saved:
            os.environ.pop(key, None)
        os.environ.update(env)
        try:
            sys.modules.pop("backtest.pin_models", None)
            pin_models = importlib.import_module("backtest.pin_models")
            return pin_models.patch(read("main.py"), pin_models.DEFAULTS)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
            sys.modules.pop("backtest.pin_models", None)

    def test_generated_code_is_valid_python(self):
        ast.parse(self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g"))

    def test_generation_is_deterministic(self):
        env = dict(GEMINI_API_KEY="a", GEMINI2_API_KEY="b", GROQ_API_KEY="g")
        self.assertEqual(self._generate(**env), self._generate(**env))

    def test_generation_is_idempotent(self):
        import importlib
        import sys

        once = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        saved = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY")}
        os.environ.update(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        try:
            sys.modules.pop("backtest.pin_models", None)
            pin_models = importlib.import_module("backtest.pin_models")
            self.assertEqual(pin_models.patch(once, pin_models.DEFAULTS), once)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
            sys.modules.pop("backtest.pin_models", None)

    def test_generated_code_names_every_bucket_and_no_secret_value(self):
        generated = self._generate(
            GEMINI_API_KEY="secret-one", GEMINI2_API_KEY="secret-two",
            GEMINI3_API_KEY="secret-three", GEMINI4_API_KEY="secret-four",
            GROQ_API_KEY="secret-groq",
        )
        for env_var in ("GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY",
                        "GEMINI4_API_KEY"):
            self.assertIn('"{0}"'.format(env_var), generated)
        for value in ("secret-one", "secret-two", "secret-three", "secret-four",
                      "secret-groq"):
            self.assertNotIn(value, generated,
                             "a credential VALUE was written into main.py")

    def test_generated_code_keeps_the_redaction_wiring(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("install_forecast_redaction(run_mode)", generated)
        self.assertEqual(generated.count("log_forecast_content("), 9)

    def test_generated_code_keeps_the_publishing_client(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("client = PublishingClient()", generated)
        self.assertIn("metaculus_client=client", generated)
        self.assertIn("print_publication_report(client)", generated)

    def test_generated_code_keeps_the_event_loop_safe_limiter(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("_concurrency_limiter_instance", generated)
        self.assertIn("get_running_loop", generated)

    def test_generated_code_keeps_the_forecast_integrity_settings(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("predictions_per_research_report=5,", generated)
        self.assertIn("skip_previously_forecasted_questions=True,", generated)

    def test_the_generator_only_touches_the_llms_block(self):
        """Everything outside llms={...} must be byte-identical."""
        original = read("main.py")
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")

        def strip_llms(src: str) -> str:
            """Remove the bot's llms= block, whether commented or generated.

            Anchored PAST the bot instantiation on purpose: main.py's module
            docstring contains an uncommented `llms={...}` EXAMPLE at the same
            indentation, and an unanchored search strips that instead -- the
            exact trap pin_models._llms_block_span documents.
            """
            init_at = src.find("template_bot = SummerTemplateBot2026(")
            if init_at == -1:
                return src
            commented = src.find("        # llms={", init_at)
            if commented != -1:
                end = src.find("        # },\n", commented)
                return src[:commented] + src[end + len("        # },\n"):]
            start = src.find("        llms={", init_at)
            if start == -1:
                return src
            depth, index = 0, src.index("{", start)
            for position in range(index, len(src)):
                if src[position] == "{":
                    depth += 1
                elif src[position] == "}":
                    depth -= 1
                    if depth == 0:
                        end = position + 1
                        if src[end:end + 2] == ",\n":
                            end += 2
                        return src[:start] + src[end:]
            return src

        # The generator also inserts its own imports, which are the only other
        # legitimate difference.
        cleaned = strip_llms(generated)
        for import_line in ("from backtest.fallback_llm import FallbackLlm\n",
                            "from backtest.balanced_llm import BalancedLlm, bucket_backend\n"):
            cleaned = cleaned.replace(import_line, "")
        self.assertEqual(cleaned, strip_llms(original))


if __name__ == "__main__":
    unittest.main()


# ===========================================================  TEST HARNESS


class TestHarnessSafetyInvariants(unittest.TestCase):
    """The suite must be incapable of touching production.

    These guard the harness itself. A module-caching bug in
    _real_forecasting_tools once produced two MetaculusClient classes -- the
    fake transport went on one, publication.PublishingClient was bound to the
    other, and _post_question_prediction fell through to the real
    implementation and began issuing live requests. It showed up only as a
    hang. Both properties below would have named it immediately.
    """

    def test_there_is_exactly_one_metaculus_client_class(self):
        from tests._real_forecasting_tools import (
            real_forecasting_tools,
            the_real_metaculus_client,
        )

        canonical = the_real_metaculus_client()
        with real_forecasting_tools():
            import publication

            self.assertIs(publication.MetaculusClient, canonical,
                          "publication is bound to a different MetaculusClient "
                          "class than the tests patch")
        # And re-entering must not mint a third one.
        self.assertIs(the_real_metaculus_client(), canonical)

    def test_the_network_kill_switch_is_armed(self):
        import requests

        from tests import NetworkAccessDeniedInTests

        with self.assertRaises(NetworkAccessDeniedInTests):
            requests.post("https://www.metaculus.com/api/questions/forecast/",
                          json=[{"question": 1}], timeout=1)

    def test_the_kill_switch_covers_plain_get_too(self):
        import requests

        from tests import NetworkAccessDeniedInTests

        with self.assertRaises(NetworkAccessDeniedInTests):
            requests.get("https://example.com", timeout=1)

    def test_the_kill_switch_names_the_target(self):
        import requests

        from tests import NetworkAccessDeniedInTests

        with self.assertRaises(NetworkAccessDeniedInTests) as caught:
            requests.post("https://www.metaculus.com/api/comments/create/", timeout=1)
        self.assertIn("metaculus.com", str(caught.exception))
