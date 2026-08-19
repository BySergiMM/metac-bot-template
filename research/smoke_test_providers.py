#!/usr/bin/env python3
"""Isolated smoke test for candidate LLM providers. ANALYSIS ONLY.

    python3 research/smoke_test_providers.py            # stdlib HTTP probes
    python3 research/smoke_test_providers.py --litellm   # also test the production path

What it does NOT do: import the forecaster, touch Metaculus questions, post
anything, run a forecast, call OpenRouter, or call xAI.

Deliberate exclusions
---------------------
OpenRouter  already known to work; testing it would burn one of the ~50 free
            calls/day that the audit showed is the scarce resource.
xAI (Grok)  every xai/* model in litellm's registry is paid (cheapest $0.20/M).
            The goal is EUR 0, so the route is not present in this file at all.

Secrets are read from the environment and NEVER printed: only their variable
name, presence, and the provider's response.

Two probe layers, because a route is only trustworthy if both pass:
  1. raw HTTP (stdlib)  the credential + endpoint + model name are valid
  2. litellm            the exact path production uses

One call per provider. No retries: a failure is data, and retrying a bad
credential or a dead model just wastes quota.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

PROMPT = "Return exactly: OK"
MAX_TOKENS = 10

# Every model name below was verified to exist in litellm 1.80.10's registry
# (model_prices_and_context_window_backup.json) before this file was written.
#
# `env_candidates` records which env vars each route accepts. litellm looks for:
# which matches NEITHER env var litellm looks for:
#   Groq -> GROQ_API_KEY   (get_llm_provider_logic.py:204)
# The earlier GROK_API_KEY secret turned out not to be a Groq key (401 Invalid
# API Key on /models and /chat/completions); it has been replaced by a real
# GROQ_API_KEY. The key is still passed explicitly rather than relying on env
# discovery, so a future rename cannot silently disable a route.
ROUTES = [
    {
        "label": "Groq",
        "env_candidates": ["GROQ_API_KEY"],
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "auth_scheme": "Bearer",
        "http_model": "openai/gpt-oss-120b",
        "litellm_model": "groq/openai/gpt-oss-120b",
        "litellm_extra": {},
    },
    {
        "label": "Cerebras",
        "env_candidates": ["CEREBRAS_API_KEY"],
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "auth_scheme": "Bearer",
        "http_model": "gpt-oss-120b",
        "litellm_model": "cerebras/gpt-oss-120b",
        "litellm_extra": {},
    },
    {
        "label": "Google Gemini",
        "env_candidates": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "auth_scheme": "Bearer",
        "http_model": "gemini-3.5-flash-lite",
        "litellm_model": "gemini/gemini-3.5-flash-lite",
        "litellm_extra": {},
    },
    {
        # NOT a litellm provider. Verified: "metaculus" is absent from litellm's
        # LlmProviders enum, so `model="metaculus/..."` is NOT invocable.
        # forecasting-tools implements the prefix itself (general_llm.py:169-215):
        # it strips "metaculus/", sets base_url to the proxy and injects an
        # `Authorization: Token <METACULUS_TOKEN>` header. We replicate exactly
        # that here rather than assuming the prefix works.
        "label": "Metaculus LLM proxy",
        "env_candidates": ["METACULUS_TOKEN"],
        "url": "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions",
        "auth_scheme": "Token",  # Metaculus style, not Bearer
        "http_model": "gpt-4o-mini",
        "litellm_model": "openai/gpt-4o-mini",
        "litellm_extra": {
            "api_base": "https://llm-proxy.metaculus.com/proxy/openai/v1",
            "extra_headers_from": "METACULUS_TOKEN",
        },
    },
]


def classify(status: int | None, body: str) -> str:
    """Map a failure onto the category that decides what fallback should do."""
    low = (body or "").lower()
    if status in (401, 403) or "invalid api key" in low or "unauthor" in low or "authenticat" in low:
        return "INVALID_KEY"
    if status == 402 or "insufficient" in low or "billing" in low or "payment" in low:
        return "NO_CREDITS"
    if status == 429 or "rate limit" in low or "quota" in low:
        return "QUOTA_OR_RATE_LIMIT"
    if status == 404 or "does not exist" in low or "not found" in low or "no endpoints" in low:
        return "MODEL_OR_ENDPOINT_NOT_FOUND"
    if status and status >= 500:
        return "PROVIDER_5XX"
    if status == 400 and ("context" in low or "too long" in low):
        return "CONTEXT"
    if status == 400:
        return "BAD_REQUEST"
    return "OTHER"


def resolve_key(route: dict) -> tuple[str | None, str | None]:
    for name in route["env_candidates"]:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value, name
    return None, None


def probe_http(route: dict, timeout: int = 45) -> dict:
    key, source = resolve_key(route)
    out: dict = {
        "layer": "http",
        "provider": route["label"],
        "endpoint": route["url"],
        "model": route["http_model"],
        "credential_env": source or "CREDENTIAL_REQUIRED",
    }
    if not key:
        out.update({"result": "NOT_TESTED", "reason": "CREDENTIAL_REQUIRED"})
        return out

    payload = json.dumps({
        "model": route["http_model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        route["url"],
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "{0} {1}".format(route["auth_scheme"], key),
            "User-Agent": "smoke-test/1.0",
            "Accept": "application/json",
        },
        method="POST",
    )

    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            elapsed = round(time.time() - started, 2)
            data = json.loads(body)
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage") or {}
            out.update({
                "result": "OK",
                "http": response.status,
                "latency_s": elapsed,
                "response": (text or "").strip()[:60],
                "tokens": {"prompt": usage.get("prompt_tokens"),
                           "completion": usage.get("completion_tokens")},
                "rate_limit_headers": {
                    k: v for k, v in response.headers.items()
                    if k.lower().startswith(("x-ratelimit", "ratelimit"))
                },
            })
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        out.update({
            "result": "FAIL", "http": exc.code,
            "latency_s": round(time.time() - started, 2),
            "error_class": classify(exc.code, body),
            "error": body[:300].replace("\n", " "),
        })
    except Exception as exc:  # noqa: BLE001
        out.update({
            "result": "FAIL", "http": None,
            "latency_s": round(time.time() - started, 2),
            "error_class": classify(None, str(exc)),
            "error": str(exc)[:300],
        })
    return out


def probe_litellm(route: dict) -> dict:
    """Exercise the exact path production uses (GeneralLlm -> acompletion)."""
    out = {"layer": "litellm", "provider": route["label"],
           "litellm_model": route["litellm_model"]}
    key, _source = resolve_key(route)
    if not key:
        out.update({"result": "NOT_TESTED", "reason": "CREDENTIAL_REQUIRED"})
        return out
    try:
        import litellm
    except ImportError:
        out.update({"result": "NOT_TESTED", "reason": "litellm not installed"})
        return out

    litellm.suppress_debug_info = True
    litellm.drop_params = True

    kwargs = dict(route.get("litellm_extra") or {})
    # Reproduce forecasting-tools' Metaculus-proxy header injection.
    if kwargs.pop("extra_headers_from", None):
        kwargs["extra_headers"] = {
            "Content-Type": "application/json",
            "Authorization": "Token {0}".format(key),
        }

    started = time.time()
    try:
        response = litellm.completion(
            model=route["litellm_model"],
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            api_key=key,   # explicit: the secret name may not match what
            timeout=45,    # litellm would look for on its own
            **kwargs,
        )
        out.update({
            "result": "OK",
            "latency_s": round(time.time() - started, 2),
            "response": (response.choices[0].message.content or "").strip()[:60],
        })
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        out.update({
            "result": "FAIL",
            "latency_s": round(time.time() - started, 2),
            "error_class": classify(getattr(exc, "status_code", None), text),
            "error": text[:300].replace("\n", " "),
        })
    return out


MODEL_LIST_URLS = {
    "Groq": "https://api.groq.com/openai/v1/models",
    "Cerebras": "https://api.cerebras.ai/v1/models",
    "Google Gemini": "https://generativelanguage.googleapis.com/v1beta/openai/models",
    "Metaculus LLM proxy": "https://llm-proxy.metaculus.com/proxy/openai/v1/models",
}


def list_models(route: dict, timeout: int = 30) -> dict:
    """Ask the provider what it serves. Beats guessing: the first run failed on
    three routes purely because of model names, and one provider replied with
    the exact replacement to use."""
    url = MODEL_LIST_URLS.get(route["label"])
    out = {"layer": "models", "provider": route["label"]}
    key, _ = resolve_key(route)
    if not url or not key:
        out.update({"result": "NOT_TESTED"})
        return out
    request = urllib.request.Request(
        url,
        headers={"Authorization": "{0} {1}".format(route["auth_scheme"], key),
                 "User-Agent": "smoke-test/1.0", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        ids = sorted(m.get("id", "?") for m in (data.get("data") or []))
        out.update({"result": "OK", "n_models": len(ids), "models": ids[:40]})
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        out.update({"result": "FAIL", "http": exc.code,
                    "error_class": classify(exc.code, body), "error": body[:200].replace("\n", " ")})
    except Exception as exc:  # noqa: BLE001
        out.update({"result": "FAIL", "error": str(exc)[:200]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litellm", action="store_true")
    args = parser.parse_args()

    print("Excluded on purpose: OpenRouter (known good, would burn free quota), "
          "xAI/Grok (paid, target is EUR 0)\n")
    print("credential presence (names only, values never printed):")
    for name in sorted({n for r in ROUTES for n in r["env_candidates"]}):
        present = bool((os.environ.get(name) or "").strip())
        print("  {0:<20} {1}".format(name, "present" if present else "absent"))
    print()

    results = []
    for route in ROUTES:
        catalogue = list_models(route)
        results.append(catalogue)
        print(json.dumps(catalogue, indent=2, sort_keys=True))
        http = probe_http(route)
        results.append(http)
        print(json.dumps(http, indent=2, sort_keys=True))
        if args.litellm:
            lite = probe_litellm(route)
            results.append(lite)
            print(json.dumps(lite, indent=2, sort_keys=True))
        print("-" * 70)

    print("=" * 70)
    print("SUMMARY")
    for r in results:
        print("  {0:<24} {1:<8} {2:<8} {3}".format(
            r.get("provider", "?")[:23], r.get("layer", "?"), r.get("result", "?"),
            r.get("error_class") or r.get("reason") or r.get("response", "")[:40],
        ))
    print("  totals: {0} OK / {1} FAIL / {2} NOT_TESTED".format(
        sum(1 for r in results if r.get("result") == "OK"),
        sum(1 for r in results if r.get("result") == "FAIL"),
        sum(1 for r in results if r.get("result") == "NOT_TESTED"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
