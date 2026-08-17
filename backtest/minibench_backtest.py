#!/usr/bin/env python3
"""MiniBench backtest harness.

fetch     -> pull every RESOLVED MiniBench question via the Metaculus API
evaluate  -> score strategies against that dataset vs the community baseline

Token comes from METACULUS_TOKEN (a GitHub secret). Never logged.
Only ever touches resolved questions, so it cannot break the no-human-in-the-loop rule.
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, sys, time
import urllib.error, urllib.parse, urllib.request
from collections import defaultdict

API = "https://www.metaculus.com/api"
PAGE = 100
UA = {"User-Agent": "minibench-backtest/1.0", "Accept": "application/json"}

def call(path, token, params):
    """Return (status, parsed_or_none, raw_body). Never raises on HTTP errors."""
    url = API + path + "?" + urllib.parse.urlencode(params, doseq=True)
    h = {"Authorization": "Token " + token}
    h.update(UA)
    r = urllib.request.Request(url, headers=h)
    for i in range(4):
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                body = resp.read().decode()
                return resp.status, json.loads(body), body
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:600]
            except Exception:
                pass
            if e.code == 429:
                time.sleep(2 ** i)
                continue
            return e.code, None, body
        except Exception as e:
            return -1, None, str(e)
    return -1, None, "retries exhausted"

# Candidate query shapes. The API rejected the first guess with a bare 400, so we
# try progressively simpler ones and keep whichever the server accepts.
VARIANTS = [
    ("tournaments+statuses+cp", lambda s, o: {"tournaments": s, "statuses": "resolved", "limit": PAGE, "offset": o, "with_cp": "true"}),
    ("tournaments+statuses",    lambda s, o: {"tournaments": s, "statuses": "resolved", "limit": PAGE, "offset": o}),
    ("tournaments only",        lambda s, o: {"tournaments": s, "limit": PAGE, "offset": o}),
    ("tournament (singular)",   lambda s, o: {"tournament": s, "limit": PAGE, "offset": o}),
    ("projects",                lambda s, o: {"projects": s, "limit": PAGE, "offset": o}),
    ("search by slug",          lambda s, o: {"search": s, "limit": PAGE, "offset": o}),
]
CHOSEN = {"name": None, "fn": None}

def probe(slug, token):
    print("probing query shapes against " + slug)
    for name, fn in VARIANTS:
        code, data, body = call("/posts/", token, fn(slug, 0))
        n = len(data.get("results", [])) if isinstance(data, dict) else 0
        print("  " + name.ljust(24) + " -> http " + str(code) + "  results=" + str(n))
        if code >= 400:
            print("      body: " + body.replace(chr(10), " ")[:300])
        if code == 200 and n > 0:
            CHOSEN["name"], CHOSEN["fn"] = name, fn
            print("  using: " + name)
            return True
    return False

def posts(slug, token):
    out, off = [], 0
    while True:
        code, data, body = call("/posts/", token, CHOSEN["fn"](slug, off))
        if code != 200 or not isinstance(data, dict):
            return out
        res = data.get("results", [])
        if not res:
            return out
        out += res
        if not data.get("next"):
            return out
        off += PAGE

def rounds():
    out = ["minibench"]
    d = dt.date(2026, 8, 10)
    for _ in range(26):
        d -= dt.timedelta(days=14)
        out.append("minibench-" + d.isoformat())
    return out

def flatten(p, slug):
    q = p.get("question") or {}
    if not q: return None
    r = q.get("resolution")
    if r in (None, "", "annulled", "ambiguous"): return None
    cp = None
    latest = ((q.get("aggregations") or {}).get("recency_weighted") or {}).get("latest") or {}
    c = latest.get("centers") or []
    if c: cp = c[0]
    return {"id": p.get("id"), "round": slug, "title": p.get("title"), "type": q.get("type"),
            "resolution": r, "open_time": q.get("open_time"),
            "close_time": q.get("scheduled_close_time"), "community_prediction": cp}

def do_fetch(a):
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN missing", file=sys.stderr); return 2
    if not probe("minibench-2026-07-27", token):
        print("no query shape worked; see bodies above"); return 1
    rows, skipped = [], 0
    for s in rounds():
        ps = posts(s, token)
        keep = [x for x in (flatten(p, s) for p in ps) if x]
        if not keep:
            skipped += 1; continue
        rows += keep
        print("  " + s + ": " + str(len(keep)))
    json.dump(rows, open(a.out, "w"), indent=1)
    nb = sum(1 for r in rows if r["type"] == "binary")
    ncp = sum(1 for r in rows if r.get("community_prediction") is not None)
    print("skipped " + str(skipped) + " empty rounds")
    print("wrote " + str(len(rows)) + " questions (" + str(nb) + " binary, " + str(ncp) + " with community prediction)")
    return 0 if rows else 1

def brier(p, o): return (p - o) ** 2
def logs(p, o):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p) if o else math.log(1 - p)

class Res:
    def __init__(s, name):
        s.name, s.n, s.b, s.l = name, 0, 0.0, 0.0
        s.buck, s.band = defaultdict(list), defaultdict(list)
    def add(s, p, o):
        s.n += 1; s.b += brier(p, o); s.l += logs(p, o)
        s.buck[min(int(p * 10), 9)].append(o)
        k = "low" if p < 0.2 else "high" if p > 0.8 else "mid"
        s.band[k].append(brier(p, o))
    def show(s):
        if not s.n:
            print(s.name + " - nothing scored"); return
        print(s.name)
        print("  n=" + str(s.n) + "  Brier=" + format(s.b / s.n, ".4f") + "  log=" + format(s.l / s.n, ".4f"))
        for b in range(10):
            o = s.buck.get(b, [])
            if o:
                print("    " + str(b * 10) + "-" + str(b * 10 + 9) + "% -> actual " +
                      format(sum(o) / len(o) * 100, ".1f") + "%  n=" + str(len(o)))
        for k in sorted(s.band):
            v = s.band[k]
            print("    band " + k + ": Brier " + format(sum(v) / len(v), ".4f") + "  n=" + str(len(v)))
        print("")

def do_eval(a):
    rows = json.load(open(a.dataset))
    bins = [r for r in rows if r["type"] == "binary" and r["resolution"] in ("yes", "no")]
    print(str(len(rows)) + " questions, " + str(len(bins)) + " scorable binary")
    if not bins: return 1
    base = sum(1 for r in bins if r["resolution"] == "yes") / len(bins)
    print("observed YES base rate: " + format(base, ".3f"))
    print("")
    strategies = [("community prediction (baseline)", lambda r: r.get("community_prediction")),
                  ("constant base rate", lambda r: base),
                  ("flat 25 percent status quo", lambda r: 0.25)]
    for name, fn in strategies:
        res = Res(name)
        for r in bins:
            p = fn(r)
            if p is None: continue
            res.add(float(p), 1 if r["resolution"] == "yes" else 0)
        res.show()
    return 0

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
f = sub.add_parser("fetch"); f.add_argument("--out", default="dataset.json"); f.set_defaults(fn=do_fetch)
e = sub.add_parser("evaluate"); e.add_argument("--dataset", default="dataset.json"); e.set_defaults(fn=do_eval)
if __name__ == "__main__":
    args = ap.parse_args()
    raise SystemExit(args.fn(args))
