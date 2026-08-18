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

Why the defaults are what they are
----------------------------------
Metaculus's own FutureEval writeup found model choice to be the single largest
differentiator between winning and losing bots, and that plain one-shot bots on
frontier models placed top-5. So the forecaster role gets the strongest model
that costs nothing: nemotron-3-ultra is a 550B reasoning model on OpenRouter's
free tier. The parser has to emit structured output reliably, which the free
Nvidia endpoints do not advertise, so that role alone uses a paid model - at
roughly a thousandth of a cent per question.

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
    "default": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "researcher": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "summarizer": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "parser": "openrouter/openai/gpt-4o-mini",
}

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

TEMPLATE = '''        llms={{
            "default": GeneralLlm(
                model="{default}",
                temperature=0.3,
                timeout=120,
                allowed_tries=3,
            ),
            "summarizer": "{summarizer}",
            "researcher": "{researcher}",
            "parser": "{parser}",
        }},
'''


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


def build_block(models: dict[str, str]) -> str:
    return TEMPLATE.format(**models)


def patch(src: str, models: dict[str, str]) -> str:
    if "llms={" in src and "# llms={" not in src:
        return src  # already active
    if ANCHOR not in src:
        raise LookupError("anchor block not found in main.py; upstream template changed")
    return src.replace(ANCHOR, build_block(models))


def selftest() -> None:
    """Runs on every invocation. A patch step that silently no-ops is worse than
    one that crashes, so prove the three behaviours that matter before touching
    the real file."""
    stub = "bot = Bot(\n        a=1,\n" + ANCHOR + "    )\n"
    once = patch(stub, DEFAULTS)
    assert "# llms={" not in once, "comment markers survived the patch"
    assert DEFAULTS["default"] in once
    assert DEFAULTS["parser"] in once
    ast.parse(once)                       # the result still has to be valid Python
    assert patch(once, DEFAULTS) == once  # idempotent

    try:
        patch("bot = Bot()\n", DEFAULTS)
    except LookupError:
        pass
    else:  # pragma: no cover - guards against a silently permissive matcher
        raise AssertionError("missing anchor should raise")

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


def check_models(models: dict[str, str]) -> list[str]:
    """One minimal completion per distinct model. Returns the failures."""
    import litellm

    failures = []
    for model in sorted(set(models.values())):
        try:
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=16,
                timeout=60,
            )
            print(f"  ok    {model}")
        except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
            head = str(exc).replace("\n", " ")[:300]
            print(f"  FAIL  {model}: {head}")
            failures.append(model)
    return failures


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
        failures = check_models(models)
        if failures:
            print(f"unusable model(s): {', '.join(failures)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
