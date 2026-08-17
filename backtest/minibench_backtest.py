#!/usr/bin/env python3
"""MiniBench backtest harness.

fetch     -> pull every RESOLVED MiniBench question via the Metaculus API
evaluate  -> score strategies against that dataset vs the community baseline

Token comes from the METACULUS_TOKEN env var (a GitHub secret). Never logged.
Only ever touches resolved questions, so it cannot violate the no-human-in-the-loop rule.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, urllib.error, urllib.parse, urllib.request
from collections import defaultdict

API = "https://www.metaculus.com/api"
PAGE = 100

def req(path, token, params=None):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    r = urllib.request.Request(url, headers={"Authorization": "Token " + token, "User-Agent": "minibench-backtest/1.0", "Accept": "application/json"})
    for i in range(5):
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** i); continue
            raise
    raise RuntimeError("gave up on " + path)

def rounds(token):
    slugs, off = set(), 0
    while True:
        d = req("/tournaments/", token, {"limit": PAGE, "offset": off})
        res = d.get("results", d if isinstance(d, list) else [])
        if not res: break
        for t in res:
            s = t.get("slug") or ""
            if s.startswith("minibench"): slugs.add(s)
        if not d.get("next"): break
        off += PAGE
    return sorted(slugs)

def posts(slug, token):
    out, off = [], 0
    while True:
        d = req("/posts/", token, {"tournaments": slug, "statuses": "resolved",
                                   "limit": PAGE, "offset": off, "with_cp": "true"})
        res = d.get("results", [])
        if not res: break
        out += res
        if not d.get("next"): break
        off += PAGE
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
    rows = []
    rs = rounds(token)
    print("rounds found:", len(rs))
    for s in rs:
        try: ps = posts(s, token)
        except Exception as e:
            print("  " + s + ": FAILED " + str(e)); continue
        keep = [x for x in (flatten(p, s) for p in ps) if x]
        rows += keep
        print("  " + s + ": " + str(len(keep)))
    json.dump(rows, open(a.out, "w"), indent=1)
    nb = sum(1 for r in rows if r["type"] == "binary")
    print("wrote " + str(len(rows)) + " questions (" + str(nb) + " binary) to " + a.out)
    return 0

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
        if not s.n: print(s.name, "- nothing scored"); return
        print(s.name)
        print("  n=" + str(s.n) + "  Brier=" + format(s.b / s.n, ".4f") + "  log=" + format(s.l / s.n, ".4f"))
        for b in range(10):
            o = s.buck.get(b, [])
            if o: print("    " + str(b * 10) + "-" + str(b * 10 + 9) + "% -> actual " +
                        format(sum(o) / len(o) * 100, ".1f") + "%  n=" + str(len(o)))
        for k in sorted(s.band):
            v = s.band[k]
            print("    band " + k + ": Brier " + format(sum(v) / len(v), ".4f") + "  n=" + str(len(v)))
        print()

def do_eval(a):
    rows = json.load(open(a.dataset))
    bins = [r for r in rows if r["type"] == "binary" and r["resolution"] in ("yes", "no")]
    print(str(len(rows)) + " questions, " + str(len(bins)) + " scorable binary")
    if not bins: return 1
    base = sum(1 for r in bins if r["resolution"] == "yes") / len(bins)
    print("observed YES base rate: " + format(base, ".3f") + chr(10))
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
    raise SystemExit(ap.parse_args().fn(ap.parse_args()))
