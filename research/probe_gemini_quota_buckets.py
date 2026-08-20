#!/usr/bin/env python3
"""Are the four GEMINI*_API_KEY credentials independent quota buckets? ANALYSIS ONLY.

    python3 research/probe_gemini_quota_buckets.py            # validity only
    python3 research/probe_gemini_quota_buckets.py --interference

What this does NOT do: import the forecaster, touch Metaculus, post anything,
run a forecast, or modify DEFAULT_LIMITS / the fallback chain.

Why an interference test is required
------------------------------------
The quota Google enforces on us was read off 344 real 429 payloads:

    quotaId         "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    quotaValue      "15"
    quotaDimensions {"location": "global", "model": "gemini-3.5-flash-lite"}

The scope is PROJECT x MODEL. An API key is not a quota dimension, and a single
project can mint many keys. So four *distinct* keys prove nothing on their own:
they may all draw on one 15 RPM bucket.

The only decisive evidence available to us is interference. Saturate one key
until Google refuses it, then -- inside the same 60s window -- ask another key
for one completion:

    other key SUCCEEDS  -> separate bucket   (independent quota)
    other key 429s      -> same bucket       (shared quota)

Rounds are pairwise-closing: key i is saturated and every key j>i is probed, so
the full relation over four keys costs three rounds rather than twelve.

Credential hygiene
------------------
Keys are read from the environment and never printed. Each is identified by
`sha256(key)[:8]`, which is stable across runs (so two secrets holding the SAME
value are visibly identical) and non-reversible. Google's error bodies are
scanned and any substring matching a live key is redacted before display.

Cost
----
Free-tier calls with max_tokens=1. A saturation round is ~20 refusals, which
cost nothing. This DOES consume the production GEMINI_API_KEY's per-minute
quota for the duration, so it must not run while the bot is forecasting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-3.5-flash-lite"
PROMPT = "Return exactly: OK"
KEY_VARS = ["GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY", "GEMINI4_API_KEY"]

# Free tier is 15 RPM (measured). Enough attempts to cross it, capped so a
# misconfigured tier cannot turn this into an unbounded burst.
SATURATION_ATTEMPTS = 22
WINDOW_SECONDS = 60.0


def safe_id(key: str) -> str:
    """Stable, non-reversible identity. Equal ids mean equal secrets."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def load_keys() -> list[dict]:
    out = []
    for name in KEY_VARS:
        value = (os.environ.get(name) or "").strip()
        out.append({
            "env": name,
            "present": bool(value),
            "id": safe_id(value) if value else None,
            "_key": value,
        })
    return out


def redact(text: str, keys: list[dict]) -> str:
    for k in keys:
        if k["_key"]:
            text = text.replace(k["_key"], "<REDACTED>")
    return text


def call(key: str, timeout: int = 30) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "quota-bucket-probe/1.0",
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return {"ok": True, "http": response.status,
                    "latency_s": round(time.time() - started, 2)}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "http": exc.code,
                "latency_s": round(time.time() - started, 2),
                "body": body}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "http": None,
                "latency_s": round(time.time() - started, 2),
                "body": str(exc)}


def quota_facts(body: str) -> dict:
    """Pull the quota block out of a 429. Returns {} when absent, never guesses."""
    facts: dict = {}
    try:
        data = json.loads(body)
    except Exception:  # noqa: BLE001
        return facts
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("quotaId", "quotaValue", "quotaDimensions", "quotaMetric"):
                    facts[key] = value
                # Google sometimes names the billed project here. Not a secret,
                # and it is the only direct evidence of project identity.
                if key == "consumer":
                    facts["consumer"] = value
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return facts


def phase_validity(keys: list[dict]) -> None:
    print("=" * 74)
    print("PHASE 1 - VALIDITY  (one minimal call per credential)")
    print("=" * 74)
    for k in keys:
        if not k["present"]:
            print(f"  {k['env']:<20} ABSENT")
            k["valid"] = None
            continue
        result = call(k["_key"])
        k["valid"] = result["ok"]
        status = "OK" if result["ok"] else "FAIL"
        line = (f"  {k['env']:<20} id={k['id']}  {status:<5} "
                f"http={result['http']}  {result['latency_s']}s")
        if not result["ok"]:
            facts = quota_facts(result.get("body", ""))
            snippet = redact(result.get("body", ""), keys)[:160].replace("\n", " ")
            line += f"\n      error: {snippet}"
            if facts:
                line += f"\n      quota: {json.dumps(facts)}"
        print(line)

    ids = [k["id"] for k in keys if k["id"]]
    duplicates = {i for i in ids if ids.count(i) > 1}
    print()
    print(f"  distinct credential values: {len(set(ids))} of {len(ids)} present")
    if duplicates:
        print(f"  !! DUPLICATE SECRETS share id(s) {sorted(duplicates)} - "
              "same value stored twice, so they cannot be separate buckets")


def saturate(k: dict, keys: list[dict]) -> dict:
    """Fire until Google refuses. Returns the first 429's quota facts."""
    refusals = 0
    first_429: dict = {}
    for attempt in range(SATURATION_ATTEMPTS):
        result = call(k["_key"])
        if not result["ok"]:
            refusals += 1
            if result["http"] == 429 and not first_429:
                first_429 = quota_facts(result.get("body", ""))
                first_429["_attempt"] = attempt + 1
            if refusals >= 2:
                break
    return first_429


def phase_interference(keys: list[dict]) -> None:
    usable = [k for k in keys if k["present"] and k.get("valid")]
    print()
    print("=" * 74)
    print("PHASE 2 - INTERFERENCE  (does saturating one refuse another?)")
    print("=" * 74)
    if len(usable) < 2:
        print("  need at least two valid keys; skipping")
        return

    for i, saturated in enumerate(usable[:-1]):
        others = usable[i + 1:]
        print(f"\n  ROUND {i + 1}: saturating {saturated['env']} (id={saturated['id']})")
        facts = saturate(saturated, keys)
        if facts:
            print(f"    refused at attempt {facts.pop('_attempt', '?')}; "
                  f"quota={json.dumps(facts)}")
        else:
            print("    NOT REFUSED within "
                  f"{SATURATION_ATTEMPTS} attempts - quota higher than expected; "
                  "this round proves nothing")
            continue

        for other in others:
            result = call(other["_key"])
            if result["ok"]:
                verdict = "SEPARATE bucket"
            elif result["http"] == 429:
                verdict = "SAME bucket (shared quota)"
            else:
                verdict = f"INCONCLUSIVE (http={result['http']})"
            print(f"    while saturated -> {other['env']:<20} id={other['id']}  "
                  f"http={result['http']:<5} => {verdict}")

        # Let the window drain so the next round starts from a clean bucket.
        if i < len(usable) - 2:
            print(f"    cooling down {WINDOW_SECONDS:.0f}s before the next round")
            time.sleep(WINDOW_SECONDS + 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interference", action="store_true",
                        help="also run the saturation test that proves independence")
    args = parser.parse_args()

    keys = load_keys()
    if not any(k["present"] for k in keys):
        print("no GEMINI*_API_KEY present in the environment; nothing to probe")
        return 2

    phase_validity(keys)
    if args.interference:
        phase_interference(keys)
    else:
        print("\n  (validity only. --interference runs the test that actually "
              "decides whether the buckets are independent.)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
