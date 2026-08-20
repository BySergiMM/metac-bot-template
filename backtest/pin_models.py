#!/usr/bin/env python3
"""Activate the pinned `llms=` block in main.py before the bot runs.

Why this exists
---------------
forecasting-tools picks default models from whichever provider env vars are
present. With only OPENROUTER_API_KEY set it assigns the researcher role to
`openai/gpt-4o-search-preview`, a model OpenRouter no longer serves, so every
question dies with:

    litellm.NotFoundError: OpenrouterException -
    {"error":{"message":"No endpoints found for openai/gpt-4o-search-preview","code":404}}

main.py ships the fix as a commented-out block. This rewrites it into an active
one, so the template stays a clean fork (no edits to main.py in git) while the
run gets models that exist.

Changing models
---------------
Put one OpenRouter model id per line in `backtest/models.txt`:

    default:    openrouter/<model>     # the forecaster - this is the one that matters
    researcher: openrouter/<model>
    summarizer: openrouter/<model>
    parser:     openrouter/<model>     # must support structured outputs

Any role you leave out falls back to the defaults below. Delete the file to go
back to the defaults entirely. No workflow edits needed.

Cross-provider fallback
------------------------
Free tiers cap by the day, not just by the minute (OpenRouter: 50 free-model
requests/24h per account). A model that is fine at 9am can 429 every question
by 6pm, and the run does not recover until the provider's own daily reset -
retrying the same account is a no-op.

When GEMINI_API_KEY and/or GROQ_API_KEY are present as well as
OPENROUTER_API_KEY, every role - including parser - is wrapped in
backtest.fallback_llm.FallbackLlm, which tries OpenRouter first and falls
through to Gemini, then Groq, only on an actual failure (rate limit, timeout,
provider outage) - never as a quality preference. Each backend keeps its own
timeout/allowed_tries. If only OPENROUTER_API_KEY is set, behavior is
unchanged: a single GeneralLlm per role, same as before fallback existed.

The parser is included even though it is a few hundred tokens per call and
was never at risk of a per-model or per-minute limit: OpenRouter's free tier
caps the whole account per day (50 requests/24h), not each model separately,
so once that account-wide cap is hit every role sharing the OPENROUTER_API_KEY
dies at once - parser included. Excluding it would leave the exact failure
mode this feature exists for (an OpenRouter-wide outage) able to kill the run
through the one role left unprotected.

Why the defaults are what they are
----------------------------------
Metaculus's own FutureEval writeup found model choice to be the single largest
differentiator between winning and losing bots, and that plain one-shot bots on
frontier models placed top-5. So the reasoning roles get the best model that
costs nothing.

The obvious pick, nemotron-3-ultra-550b, was measured and rejected: on OpenRouter's
free tier it queued past 60s and timed out on 8 of 9 test questions. A model that
does not answer inside the window scores zero, however good it is. nemotron-3.5-lightning
is the free model built for latency, so that is what runs.

The parser has to emit structured output reliably, which the free Nvidia endpoints
do not advertise, so that role alone uses a paid model - at roughly a thousandth of
a cent per question. It stays a bare string, exactly as the official template ships
it: parsing is a few hundred tokens and never came close to the default timeout, so
there is nothing to gain by varying it.

Preflight
---------
`python backtest/pin_models.py --check` additionally sends one tiny completion to
every distinct model and exits 1 if any of them errors. That is what caught the
dead `gpt-4o-search-preview` id: a model that 404s costs a whole tournament run,
and the run is the scarce resource, not the token.

Exits 1 if the anchor block is missing, so an upstream template change fails
loudly instead of silently restoring the broken defaults.
"""

from __future__ import annotations

import ast
import pathlib
import sys

DEFAULTS = {
    "default": "openrouter/nvidia/nemotron-3.5-lightning:free",
    "researcher": "openrouter/nvidia/nemotron-3.5-lightning:free",
    "summarizer": "openrouter/nvidia/nemotron-3.5-lightning:free",
    "parser": "openrouter/openai/gpt-4o-mini",
}

# Fallback chain for the free OpenRouter model, tried in order only on
# failure (rate limit, timeout, outage). Each needs its own API key as an
# env var (GEMINI_API_KEY / GROQ_API_KEY) - present in the workflow only if
# the corresponding secret is set. Absent keys are skipped, never an error:
# fallback is an upgrade, not a requirement.
# Every entry here was verified end-to-end by research/smoke_test_providers.py
# (workflow run 32293364075): the provider's own /models catalogue was listed,
# then a real completion returned 200 OK through BOTH raw HTTP and litellm.
# Model names are NOT taken from litellm's price registry - that is a price
# table, not an entitlement check, and three of three guesses taken from it
# turned out to be unavailable to these accounts.
#
# Deliberately excluded, both verified as unusable at EUR 0:
#   Cerebras  402 "Payment required to access this resource" - no free tier
#   Metaculus 400 "You don't have an allowance for model <gpt-4o-mini>"
FALLBACK_CHAIN = [
    ("gemini/gemini-3.5-flash-lite", "GEMINI_API_KEY"),
    ("groq/openai/gpt-oss-120b", "GROQ_API_KEY"),
]

# Which of those can serve the PARSER. The parser is the one role whose output
# must be machine-readable: structure_output() requests a JSON schema and
# raises when two samples disagree, so a backend that answers in prose turns a
# survived outage into a discarded prediction.
#
# Measured, not assumed (same smoke-test run):
#   gemini/gemini-3.5-flash-lite  returned {"probability": 0.42}          OK
#   groq/openai/gpt-oss-120b      json_validate_failed, failed_generation="" 
# So Groq stays in the reasoning chain, where prose is exactly what we want,
# and is excluded from the parser chain. A reasoning role that falls all the
# way to Groq still gets parsed by OpenRouter or Gemini.
STRUCTURED_OUTPUT_CAPABLE = {"gemini/gemini-3.5-flash-lite"}

# forecasting-tools only applies a custom timeout to roles given as GeneralLlm;
# a bare model string silently gets its 60s default. The free tier queues past
# that, which showed up as `litellm.Timeout ... timeout passed=60.0` on 8 of 9
# test questions. Hence: every role is a GeneralLlm, and the budget is generous.
TIMEOUT_SECONDS = 180

# Which fallback keys are actually available *right now*. Read once, at patch
# time, via os.environ - not deferred to the bot's own runtime - so the
# generated main.py is an honest record of what ran, and --check can verify
# exactly the chain that will be used.
import os

ACTIVE_FALLBACKS = [
    (model, env_var) for model, env_var in FALLBACK_CHAIN if os.getenv(env_var)
]

# Additional Gemini credentials, each backed by its own Google project and so
# its own 15 RPM allowance. Proven independent by interference, not assumed:
# runs 32380381980 and 32381181256 saturated one key and found every other key
# still admitted inside the same 60s window, for all six pairs.
#
# Order is fixed and meaningful: index i maps to rate_limiter's bucket i, and
# GEMINI_API_KEY is always bucket 0, so a single-key install produces exactly
# the registry it produced before buckets existed.
# Deliberately NOT imported from backtest.rate_limiter. This file runs as a
# bare script -- the workflow calls `python backtest/pin_models.py`, so the
# repo root is not on sys.path and any `backtest.*` import raises
# ModuleNotFoundError before the bot ever starts. The values are duplicated
# here and test_pin_models_buckets asserts they never drift apart.
GEMINI_BUCKET_MODEL = "gemini/gemini-3.5-flash-lite"
GEMINI_BUCKET_KEYS = (
    GEMINI_BUCKET_MODEL,
    GEMINI_BUCKET_MODEL + "#b1",
    GEMINI_BUCKET_MODEL + "#b2",
    GEMINI_BUCKET_MODEL + "#b3",
)

GEMINI_BUCKET_ENV_VARS = (
    "GEMINI_API_KEY",
    "GEMINI2_API_KEY",
    "GEMINI3_API_KEY",
    "GEMINI4_API_KEY",
)

# (env_var, limiter_key) for the credentials present RIGHT NOW. Read at patch
# time, like ACTIVE_FALLBACKS, so the generated main.py is an honest record of
# what ran. A key present here contributes a bucket; a key absent contributes
# nothing and the run simply has fewer buckets.
ACTIVE_GEMINI_BUCKETS = [
    (env_var, GEMINI_BUCKET_KEYS[i])
    for i, env_var in enumerate(GEMINI_BUCKET_ENV_VARS)
    if os.getenv(env_var)
]

# Balancing is only worth anything with two or more buckets. With one, the
# generated block must be the pre-bucket block, unchanged.
BALANCED = len(ACTIVE_GEMINI_BUCKETS) > 1

OVERRIDE_FILE = pathlib.Path(__file__).with_name("models.txt")

ANCHOR = '''        # llms={
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
'''

# All roles now get fallback-wrapped when active (see docstring: OpenRouter's
# daily cap is account-wide, so parser shares the exact failure mode this
# feature exists to protect against).
REASONING_ROLES = ("default", "summarizer", "researcher", "parser")


# A provider inside a fallback chain gets exactly ONE attempt. GeneralLlm.invoke
# is wrapped by RetryableModel with wait_random_exponential(min=5, max=60), so
# allowed_tries=3 on a chain member means up to ~2 minutes of backoff before the
# next provider is even tried - and a daily-quota 429 cannot succeed on retry
# anyway. Retrying is the fallback's job now, and it retries by switching
# provider, not by asking the same exhausted account again.
#
# Standalone (no fallback keys present) keeps allowed_tries=3, so the
# no-fallback path stays byte-for-byte identical to before this feature.
TRIES_IN_CHAIN = 1
TRIES_STANDALONE = 3


def _backend_expr(model: str, allowed_tries: int = TRIES_STANDALONE) -> str:
    """One GeneralLlm(...) call, single line, for use inside a FallbackLlm list
    or standalone."""
    return (
        f'GeneralLlm(model="{model}", temperature=0.3, '
        f"timeout={TIMEOUT_SECONDS}, allowed_tries={allowed_tries})"
    )


def _fallbacks_for(role: str) -> list[str]:
    """Active fallback models this role may actually use.

    Every role gets the full chain except the parser, which only gets backends
    verified to emit schema-constrained JSON."""
    models = [model for model, _env in ACTIVE_FALLBACKS]
    if role == "parser":
        return [m for m in models if m in STRUCTURED_OUTPUT_CAPABLE]
    return models


GEMINI_MODEL = "gemini/gemini-3.5-flash-lite"


def _bucket_expr(env_var: str, limiter_key: str) -> str:
    """One Gemini link bound to a specific credential and quota bucket.

    The env var NAME is emitted, never its value: a secret must not be written
    into main.py. bucket_backend reads it at bot runtime.
    """
    return (
        f'bucket_backend("{GEMINI_MODEL}", "{env_var}", "{limiter_key}", '
        f"{TIMEOUT_SECONDS}, {TRIES_IN_CHAIN})"
    )


def _chain_expr(
    model: str,
    fallbacks: list[str],
    bucket: tuple[str, str] | None,
    indent: int = 12,
) -> str:
    """One FallbackLlm, in the production order OpenRouter -> Gemini -> Groq.

    When `bucket` is given, the Gemini link is bound to that credential and
    quota bucket; every other link is byte-identical to the unbalanced form.
    The ORDER is never altered -- balancing chooses between whole chains, it
    does not reorder the links inside one.
    """
    backends = [_backend_expr(model, TRIES_IN_CHAIN)]
    for fb_model in fallbacks:
        if bucket is not None and fb_model == GEMINI_MODEL:
            backends.append(_bucket_expr(*bucket))
        else:
            backends.append(_backend_expr(fb_model, TRIES_IN_CHAIN))
    # `indent` keeps the unbalanced output byte-identical to the pre-bucket
    # generator; a chain nested inside BalancedLlm sits one level deeper.
    inner = " " * (indent + 4)
    outer = " " * indent
    joined = (",\n" + inner).join(backends)
    return f"FallbackLlm([\n{inner}{joined},\n{outer}])"


def _role_expr(role: str, model: str) -> str:
    """The right-hand side for one llms={} entry. Reasoning roles get wrapped
    in FallbackLlm when at least one fallback API key is present; otherwise
    (or for the parser) a single GeneralLlm, unchanged from before fallback
    existed.

    With two or more Gemini credentials present, the FallbackLlm is replicated
    once per bucket and the copies are wrapped in a BalancedLlm. With one
    credential the output is exactly what it was before buckets existed --
    there is no BalancedLlm around a single chain.
    """
    if role not in REASONING_ROLES:
        return _backend_expr(model)
    fallbacks = _fallbacks_for(role)
    if not fallbacks:
        return _backend_expr(model)

    if not (BALANCED and GEMINI_MODEL in fallbacks):
        return _chain_expr(model, fallbacks, bucket=None)

    chains = [
        _chain_expr(model, fallbacks, bucket=(env_var, limiter_key), indent=16)
        for env_var, limiter_key in ACTIVE_GEMINI_BUCKETS
    ]
    joined = ",\n                ".join(chains)
    keys = ", ".join(f'"{limiter_key}"' for _env, limiter_key in ACTIVE_GEMINI_BUCKETS)
    return (
        f"BalancedLlm([\n                {joined},\n            ], "
        f"[{keys}])"
    )


TEMPLATE_HEADER = '''        llms={
'''
TEMPLATE_FOOTER = '''        },
'''


def build_block(models: dict[str, str]) -> str:
    lines = [TEMPLATE_HEADER.rstrip("\n")]
    for role in ("default", "summarizer", "researcher", "parser"):
        if role == "parser" and not ACTIVE_FALLBACKS:
            # Unwrapped: stay a bare model string, exactly as before fallback
            # existed, so the no-fallback path is byte-for-byte unchanged.
            lines.append(f'            "parser": "{models["parser"]}",')
        else:
            lines.append(f'            "{role}": {_role_expr(role, models[role])},')
    lines.append(TEMPLATE_FOOTER.rstrip("\n"))
    return "\n".join(lines) + "\n"


def read_overrides(path: pathlib.Path) -> dict[str, str]:
    """Parse `role: model` lines. Unknown roles are an error, not a silent no-op."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path.name}:{lineno}: expected 'role: model', got {raw!r}")
        role, model = (part.strip() for part in line.split(":", 1))
        if role not in DEFAULTS:
            raise ValueError(
                f"{path.name}:{lineno}: unknown role {role!r}; "
                f"expected one of {sorted(DEFAULTS)}"
            )
        if not model:
            raise ValueError(f"{path.name}:{lineno}: empty model for role {role!r}")
        out[role] = model
    return out


IMPORT_ANCHOR = "silence_noisy_dependencies()\n"
IMPORT_LINE = "from backtest.fallback_llm import FallbackLlm\n"
BALANCED_IMPORT_LINE = (
    "from backtest.balanced_llm import BalancedLlm, bucket_backend\n"
)


def _ensure_fallback_import(src: str) -> str:
    """Add the FallbackLlm import once, only when a fallback chain is active.
    Idempotent and independent of whether the llms= block itself is patched
    yet, so re-running after an env change (a key added or removed) converges
    to the right import state on the next invocation either way."""
    has_import = IMPORT_LINE in src
    if ACTIVE_FALLBACKS and not has_import:
        if IMPORT_ANCHOR not in src:
            raise LookupError(
                "import anchor not found in main.py; upstream template changed"
            )
        # Assign rather than return: the balanced import below must be
        # considered in the SAME pass, or the first patch would emit
        # BalancedLlm(...) without importing it and only a second run would
        # fix it -- which also breaks the idempotency the selftest asserts.
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
    elif not ACTIVE_FALLBACKS and has_import:
        src = src.replace(IMPORT_LINE, "", 1)

    # The balanced import follows the same converge-either-way rule, so adding
    # or removing a secondary key and re-running lands in the right state.
    has_balanced = BALANCED_IMPORT_LINE in src
    if BALANCED and ACTIVE_FALLBACKS and not has_balanced:
        if IMPORT_LINE not in src:
            raise LookupError(
                "FallbackLlm import missing; cannot anchor the balanced import"
            )
        src = src.replace(IMPORT_LINE, IMPORT_LINE + BALANCED_IMPORT_LINE, 1)
    elif (not BALANCED or not ACTIVE_FALLBACKS) and has_balanced:
        src = src.replace(BALANCED_IMPORT_LINE, "", 1)
    return src


# main.py's own module docstring contains a *documentation example* of an
# llms={...} block, at the same indentation, to show users the syntax. A bare
# text search for "llms={" matches that example first and silently patches
# nothing real. Anchoring the search to start after the actual bot
# instantiation call is what makes this unambiguous.
BOT_INIT_ANCHOR = "template_bot = SummerTemplateBot2026(\n"


def _llms_block_span(src: str) -> tuple[int, int] | None:
    """Locate the active (uncommented) llms={...} block inside the real bot
    instantiation - not the docstring example above it - by brace counting.
    Generated code has a known shape, not arbitrary Python, so this is safe."""
    init_at = src.find(BOT_INIT_ANCHOR)
    if init_at == -1:
        return None  # upstream renamed/restructured the bot init; caller falls through to ANCHOR/LookupError
    start = src.find("        llms={", init_at)
    if start == -1 or src[: start + 8].rstrip().endswith("#"):
        return None
    depth = 0
    i = src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                # include the trailing comma + newline the template emits
                end = j + 1
                if src[end : end + 2] == ",\n":
                    end += 2
                return start, end
    return None  # pragma: no cover - malformed input, caller re-raises via LookupError


def patch(src: str, models: dict[str, str]) -> str:
    src = _ensure_fallback_import(src)

    span = _llms_block_span(src)
    if span is not None:
        current_block = src[span[0] : span[1]]
        wants_fallback = bool(ACTIVE_FALLBACKS)
        has_fallback = "FallbackLlm(" in current_block
        if wants_fallback == has_fallback:
            return src  # already active and already matches the current fallback state
        # Stale: fallback keys were added/removed since this block was written.
        # Rebuild it in place rather than leaving an import/usage mismatch.
        return src[: span[0]] + build_block(models) + src[span[1] :]

    if ANCHOR not in src:
        raise LookupError("anchor block not found in main.py; upstream template changed")
    return src.replace(ANCHOR, build_block(models))


def _eval_llms_dict(generated: str, active_fallbacks: list) -> dict:
    """Actually execute the generated `llms={...}` block against stand-ins for
    GeneralLlm/FallbackLlm and return the resulting dict. ast.parse only proves
    the code is syntactically valid Python - it does NOT prove `llms={{...}}`
    (doubled braces, a leftover from an earlier .format()-based template)
    parses as a dict rather than a set-containing-a-dict, which is valid
    syntax but crashes at runtime with "unhashable type: dict". This caught
    exactly that bug; keep it, do not downgrade it back to just ast.parse."""
    span = _llms_block_span(generated)
    assert span is not None, "generated code has no llms={...} block to evaluate"
    block_src = generated[span[0] : span[1]].strip().rstrip(",")

    class _StubGeneralLlm:
        def __init__(self, model, **kw):
            self.model = model

    class _StubFallbackLlm:
        def __init__(self, backends):
            self.backends = backends

    class _StubBalancedLlm:
        def __init__(self, chains, bucket_keys):
            self.chains = chains
            self.bucket_keys = bucket_keys

    def _stub_bucket_backend(model, api_key_env, limiter_key, timeout, tries):
        backend = _StubGeneralLlm(model)
        backend.limiter_key = limiter_key
        backend.api_key_env = api_key_env
        return backend

    namespace = {
        "GeneralLlm": _StubGeneralLlm,
        "FallbackLlm": _StubFallbackLlm,
        "BalancedLlm": _StubBalancedLlm,
        "bucket_backend": _stub_bucket_backend,
    }
    exec(f"result = {block_src[len('llms='):]}", namespace)  # noqa: S102 - trusted, self-generated code
    result = namespace["result"]
    assert isinstance(result, dict), f"llms= evaluated to {type(result)}, not dict"
    assert set(result) == {"default", "summarizer", "researcher", "parser"}
    for role in ("default", "summarizer", "researcher", "parser"):
        # Chain length is per-role now: the parser only accepts backends
        # verified to emit schema-constrained JSON, so it can legitimately be
        # shorter than the reasoning chain (or absent entirely, if the only
        # active fallback cannot parse).
        expected_fallbacks = _fallbacks_for(role) if active_fallbacks else []
        if expected_fallbacks:
            entry = result[role]
            if BALANCED and GEMINI_BUCKET_MODEL in expected_fallbacks:
                # One chain per credential, wrapped. The bucket keys must be
                # exactly the registered ones, in order: an unregistered key
                # would resolve to a limiter with NO rate limit at all.
                assert isinstance(entry, _StubBalancedLlm), f"{role} should be balanced"
                assert len(entry.chains) == len(ACTIVE_GEMINI_BUCKETS), (
                    f"{role}: {len(entry.chains)} chains for "
                    f"{len(ACTIVE_GEMINI_BUCKETS)} credentials"
                )
                assert entry.bucket_keys == [k for _e, k in ACTIVE_GEMINI_BUCKETS], (
                    f"{role}: bucket keys {entry.bucket_keys} do not match the "
                    "registered buckets"
                )
                for key in entry.bucket_keys:
                    assert key in GEMINI_BUCKET_KEYS, (
                        f"{role}: {key!r} is not a registered bucket, so its "
                        "limiter would be created with no rate limit"
                    )
                seen_envs = []
                for chain, expected_key in zip(entry.chains, entry.bucket_keys):
                    assert isinstance(chain, _StubFallbackLlm)
                    served = [b.model for b in chain.backends[1:]]
                    assert served == expected_fallbacks, (
                        f"{role} chain order changed: {served}"
                    )
                    gem = chain.backends[1]
                    assert gem.limiter_key == expected_key, (
                        f"{role}: chain bound to {gem.limiter_key}, expected "
                        f"{expected_key}"
                    )
                    seen_envs.append(gem.api_key_env)
                assert len(set(seen_envs)) == len(seen_envs), (
                    f"{role}: two chains share a credential ({seen_envs}); "
                    "they would contend for one quota"
                )
                continue
            assert isinstance(entry, _StubFallbackLlm), f"{role} should be wrapped"
            assert len(entry.backends) == 1 + len(expected_fallbacks), (
                f"{role}: expected {1 + len(expected_fallbacks)} backends, "
                f"got {len(entry.backends)}"
            )
            served = [b.model for b in entry.backends[1:]]
            assert served == expected_fallbacks, f"{role} chain is {served}"
        elif active_fallbacks and role == "parser":
            # Fallbacks exist but none can parse: the parser stays unwrapped
            # rather than gaining a backend that would answer in prose.
            assert isinstance(result[role], _StubGeneralLlm), (
                "parser should stay a single backend when no fallback can parse"
            )
        elif role == "parser":
            assert isinstance(result[role], str), "parser should stay a bare model string"
        else:
            assert isinstance(result[role], _StubGeneralLlm), f"{role} should be a plain backend"
    return result


def selftest() -> None:
    """Runs on every invocation. A patch step that silently no-ops is worse than
    one that crashes, so prove the three behaviours that matter before touching
    the real file. Must not assume anything about this process' own environment -
    ACTIVE_FALLBACKS reflects whatever keys are actually set when this module was
    imported, so every check below reads that real value rather than hardcoding [].
    """
    global ACTIVE_FALLBACKS, ACTIVE_GEMINI_BUCKETS, BALANCED
    stub = IMPORT_ANCHOR + "\n" + BOT_INIT_ANCHOR + "        a=1,\n" + ANCHOR + "    )\n"
    once = patch(stub, DEFAULTS)
    assert "# llms={" not in once, "comment markers survived the patch"
    assert DEFAULTS["default"] in once
    assert DEFAULTS["parser"] in once
    ast.parse(once)                                   # the result still has to be valid Python
    _eval_llms_dict(once, ACTIVE_FALLBACKS)           # ...and evaluate to a real dict, not a set
    assert patch(once, DEFAULTS) == once              # idempotent

    try:
        patch("bot = Bot()\n", DEFAULTS)
    except LookupError:
        pass
    else:  # pragma: no cover - guards against a silently permissive matcher
        raise AssertionError("missing anchor should raise")

    # Regression guard: a real main.py has a documentation example of
    # `llms={...}` inside its module docstring, at the same indentation,
    # BEFORE the real bot instantiation. A naive whole-file text search for
    # "llms={" finds that example first and reports success while patching
    # nothing - this is what actually happened during development.
    doc_example = ANCHOR.replace("        # ", "        ")  # same block, not commented
    docstring_trap = (
        chr(39)*3 + "\n    Example:\n    ```python\n" + doc_example + "    ```\n    " + chr(39)*3 + "\n"
        + IMPORT_ANCHOR + "\n"
        + BOT_INIT_ANCHOR + "        a=1,\n" + ANCHOR + "    )\n"
    )
    trapped = patch(docstring_trap, DEFAULTS)
    # The docstring's own example model (gpt-4o) must survive untouched,
    # and the real block below it must be the one that got patched.
    assert 'model="openrouter/openai/gpt-4o"' in trapped  # docstring example untouched
    # Three reasoning roles are patched. When balancing is active each role
    # emits one chain per credential, and every chain opens with the same
    # OpenRouter backend, so the model string appears once per chain per role.
    # The parser is excluded from this count: it uses a different default.
    chains_per_role = len(ACTIVE_GEMINI_BUCKETS) if BALANCED else 1
    assert trapped.count(DEFAULTS["default"]) == 3 * chains_per_role, (
        "expected {0} occurrences ({1} reasoning roles x {2} chain(s)), got {3}".format(
            3 * chains_per_role, 3, chains_per_role,
            trapped.count(DEFAULTS["default"]))
    )
    real_block_start = trapped.index(BOT_INIT_ANCHOR)
    assert 'model="openrouter/openai/gpt-4o"' not in trapped[real_block_start:], (
        "the real block still has the old example model - docstring text leaked into the match"
    )

    # Fallback wiring: simulate both keys present, independent of this
    # process's real environment, and check the generated code is what it
    # claims to be - the import appears, and now every role including parser
    # gets every backend, since parser shares OpenRouter's account-wide daily
    # cap with the reasoning roles.
    saved_fallbacks = ACTIVE_FALLBACKS
    saved_buckets = ACTIVE_GEMINI_BUCKETS
    saved_balanced = BALANCED
    try:
        # Exercise the real configured chain rather than a hardcoded copy, so
        # this check cannot drift away from FALLBACK_CHAIN.
        ACTIVE_FALLBACKS = list(FALLBACK_CHAIN)
        # Pin the bucket count too. Without this the counts below silently
        # depend on how many GEMINI*_API_KEY happen to be set in the calling
        # environment, which is what broke CI run 32385415823: the block was
        # written for one chain per role and a second credential makes it two.
        ACTIVE_GEMINI_BUCKETS = [(GEMINI_BUCKET_ENV_VARS[0], GEMINI_BUCKET_KEYS[0])]
        BALANCED = False
        parser_fallbacks = _fallbacks_for("parser")
        reasoning_fallbacks = _fallbacks_for("default")
        with_fb = patch(stub, DEFAULTS)
        assert IMPORT_LINE in with_fb
        # default/researcher/summarizer share one model string in DEFAULTS
        # (3 occurrences as FallbackLlm primaries); parser has its own
        # distinct model string, so it is not part of this count.
        assert with_fb.count(DEFAULTS["default"]) == 3
        assert "FallbackLlm([\n                GeneralLlm(model=\"" + DEFAULTS["default"] in with_fb
        # Reasoning roles get every active fallback; the parser gets only the
        # structured-output-capable subset, so counts differ by design.
        for model in reasoning_fallbacks:
            expected = 3 + (1 if model in parser_fallbacks else 0)
            assert with_fb.count(model) == expected, (
                f"{model}: expected {expected} occurrences, got {with_fb.count(model)}"
            )
        wrapped_roles = 3 + (1 if parser_fallbacks else 0)
        assert with_fb.count("FallbackLlm([") == wrapped_roles
        if parser_fallbacks:
            assert "FallbackLlm([\n                GeneralLlm(model=\"" + DEFAULTS["parser"] in with_fb
        ast.parse(with_fb)
        _eval_llms_dict(with_fb, ACTIVE_FALLBACKS)  # real dict, correct backend count per role
        assert patch(with_fb, DEFAULTS) == with_fb  # idempotent with fallback active too

        # Balanced wiring: the same roles, one chain per credential. Counts
        # scale by the number of buckets, and the balanced import must appear.
        ACTIVE_GEMINI_BUCKETS = [
            (env, key) for env, key
            in zip(GEMINI_BUCKET_ENV_VARS, GEMINI_BUCKET_KEYS)
        ]
        BALANCED = True
        balanced_src = patch(stub, DEFAULTS)
        buckets = len(ACTIVE_GEMINI_BUCKETS)
        assert BALANCED_IMPORT_LINE in balanced_src
        assert balanced_src.count(DEFAULTS["default"]) == 3 * buckets
        assert balanced_src.count("BalancedLlm([") == 3 + (1 if parser_fallbacks else 0)
        # Every bucket must be named exactly once per role that uses it.
        for _env, key in ACTIVE_GEMINI_BUCKETS:
            occurrences = balanced_src.count('"{0}"'.format(key))
            assert occurrences > 0, "bucket {0} never appears".format(key)
        ast.parse(balanced_src)
        _eval_llms_dict(balanced_src, ACTIVE_FALLBACKS)
        assert patch(balanced_src, DEFAULTS) == balanced_src  # idempotent
        ACTIVE_GEMINI_BUCKETS = [(GEMINI_BUCKET_ENV_VARS[0], GEMINI_BUCKET_KEYS[0])]
        BALANCED = False

        # Import must disappear again if the chain goes back to empty (keys removed).
        ACTIVE_FALLBACKS = []
        reverted = patch(with_fb, DEFAULTS)
        assert IMPORT_LINE not in reverted
    finally:
        ACTIVE_FALLBACKS = saved_fallbacks
        ACTIVE_GEMINI_BUCKETS = saved_buckets
        BALANCED = saved_balanced

    tmp = pathlib.Path("/tmp/_pin_models_selftest.txt")
    tmp.write_text("parser: x/y  # trailing comment\n\n# whole-line comment\n")
    assert read_overrides(tmp) == {"parser": "x/y"}
    tmp.write_text("nonsense: z\n")
    try:
        read_overrides(tmp)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown role should raise")
    tmp.unlink()


def _ping(model: str) -> str | None:
    """One minimal completion. Returns None on success, the error text on failure."""
    import litellm

    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
            timeout=60,
        )
        print(f"  ok    {model}")
        return None
    except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
        head = str(exc).replace("\n", " ")[:300]
        print(f"  FAIL  {model}: {head}")
        return head


def check_models(models: dict[str, str]) -> list[str]:
    """Ping every distinct model that could actually be called at runtime -
    primaries and every fallback backend - and return the roles that have NO
    working backend at all. A role with a dead primary but a live fallback is
    NOT a failure: that is exactly the scenario fallback exists for. Testing
    only `models` (the primaries) would make --check block on a transient
    OpenRouter outage that FallbackLlm would have silently absorbed."""
    fallback_models = [m for m, _env in ACTIVE_FALLBACKS]
    all_models = sorted(set(models.values()) | set(fallback_models))
    results = {m: _ping(m) for m in all_models}

    dead_roles = []
    for role in ("default", "researcher", "summarizer", "parser"):
        # When fallback is active, parser is wrapped exactly like the
        # reasoning roles (see REASONING_ROLES / docstring). With no fallback
        # keys present, fallback_models is empty and this degrades to "dead
        # primary = dead role", the pre-fallback behavior.
        chain = [models[role]] + (_fallbacks_for(role) if role in REASONING_ROLES else [])
        if all(results[m] is not None for m in chain):
            dead_roles.append(role)
    return dead_roles


def main() -> int:
    selftest()

    path = pathlib.Path("main.py")
    if not path.exists():
        print("main.py not found; run this from the repo root", file=sys.stderr)
        return 1

    models = dict(DEFAULTS)
    try:
        overrides = read_overrides(OVERRIDE_FILE)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    models.update(overrides)

    src = path.read_text()
    try:
        patched = patch(src, models)
    except LookupError as exc:
        print(exc, file=sys.stderr)
        return 1

    if patched == src:
        print("llms block already active, nothing to do")
    else:
        ast.parse(patched)
        path.write_text(patched)
        for role in ("default", "researcher", "summarizer", "parser"):
            mark = " (from models.txt)" if role in overrides else ""
            print(f"  {role:<11} {models[role]}{mark}")

    # Deliberately outside the branch above: an already-patched tree still has to
    # prove its models answer, otherwise --check silently passes on a rerun.
    if "--check" in sys.argv:
        print("checking every model answers before the bot burns a run on them")
        if ACTIVE_FALLBACKS:
            chain = " -> ".join([m for m, _e in [(models["default"], None)] + ACTIVE_FALLBACKS])
            print(f"  fallback chain per role (default/researcher/summarizer/parser): {chain}")
        dead_roles = check_models(models)
        if dead_roles:
            print(f"role(s) with no working backend at all: {', '.join(dead_roles)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
