#!/usr/bin/env python3
"""Pin the LLM models before the bot runs.

forecasting-tools picks defaults from whichever provider env vars are present.
With only OPENROUTER_API_KEY set it assigns the researcher role to
openai/gpt-4o-search-preview, and OpenRouter serves no endpoint for that model,
so every question dies with:

    litellm.NotFoundError: OpenrouterException -
    {"error":{"message":"No endpoints found for openai/gpt-4o-search-preview","code":404}}

main.py ships that config as a commented-out block. This rewrites it into an
active one. Run from the repo root before main.py.

Exits 1 if the anchor block is missing, so an upstream change fails loudly
instead of silently leaving the broken defaults in place.
"""
import pathlib
import sys

MODEL = "openrouter/openai/gpt-4o-mini"

OLD = '''        # llms={
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

NEW = '''        llms={
            "default": GeneralLlm(
                model="__MODEL__",
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
            "summarizer": "__MODEL__",
            "researcher": "__MODEL__",
            "parser": "__MODEL__",
        },
'''.replace("__MODEL__", MODEL)


def main():
    p = pathlib.Path("main.py")
    if not p.exists():
        print("main.py not found; run this from the repo root", file=sys.stderr)
        return 1
    src = p.read_text()
    if "llms={" in src and "# llms={" not in src:
        print("llms block already active, nothing to do")
        return 0
    if OLD not in src:
        print("anchor block not found in main.py; upstream template changed", file=sys.stderr)
        return 1
    p.write_text(src.replace(OLD, NEW))
    print("pinned default, summarizer, researcher and parser to " + MODEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
