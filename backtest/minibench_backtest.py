#!/usr/bin/env python3
"""MiniBench backtest harness. Resolved questions only."""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, sys, time
import urllib.error, urllib.parse, urllib.request
from collections import defaultdict

API = "https://www.metaculus.com/api"
PAGE = 100
UA = {"User-Agent": "minibench-backtest/1.0", "Accept": "application/json"}
DUMPED = {"done": False}

def call(path, token, params):
    url = API + path + "?" + urllib.parse.urlencode(params, doseq=True)
    h = {"Authorization": "Token " + token}
    h.update(UA)
    r = urllib.request.Request(url, headers=h)
    for i in range(4):
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode()), ""
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:400]
            except Exception:
                pass
            if e.code in (429, 400, 502, 503):
                time.sleep(1.5 * (i + 1))
                continue
            return e.code, None, body
        except Exception as e:
            time.sleep(1.0)
            continue
    return -1, None, "retries exhausted"

def params_for(slug, off):
    return {"tournaments": slug, "statuses": "resolved", "limit": PAGE, "offset": off, "with_cp": "true"}

def posts(slug, token):
    out, off = [], 0
    while True:
        code, data, body = call("/posts/", token, params_for(slug, off))
        if code != 200 or not isinstance(data, dict):
            if code != 404:
                print("    " + slug + " http " + str(code) + " " + body.replace(chr(10), " ")[:160])
            return out
        res = data.get("results", [])
        if not res:
            return out
        out += res
        if not data.get("next"):
            return out
        off += PAGE
        time.sleep(0.4)
    return out

def rounds():
    out = ["minibench"]
    d = dt.date(2026, 8, 10)
    for _ in range(26):
        d -= dt.timedelta(days=14)
        out.append("minibench-" + d.isoformat())
    return out

def deep_find(obj, key, depth=0):
    """Find the first value for key anywhere in a nested dict, for shape-agnostic parsing."""
    if depth > 4 or not isinstance(obj, dict):
        return None
    if key in obj and obj[key] not in (None, ""):
        return obj[key]
    for v in obj.values():
        if isinstance(v, dict):
            r = deep_find(v, key, depth + 1)
            if r is not None:
                return r
    return None

def flatten(p, slug):
    q = p.get("question") or {}
    if not DUMPED["done"]:
        DUMPED["done"] = True
        print("  post keys: " + ",".join(sorted(p.keys()))[:300])
        print("  question keys: " + ",".join(sorted(q.keys()))[:300])
        print("  sample resolution: " + repr(deep_find(p, "resolution")))
        print("  sample type: " + repr(q.get("type") or p.get("type")))
    res = q.get("resolution") or p.get("resolution") or deep_find(p, "resolution")
    if res in (None, "", "annulled", "ambiguous"):
        return None
    qtype = q.get("type") or p.get("type") or "unknown"
    cp = None
    latest = ((q.get("aggregations") or {}).get("recency_weighted") or {}).get("latest") or {}
    c = latest.get("centers") or []
    if c:
        cp = c[0]
    if cp is None:
        cp = deep_find(q, "community_prediction")
    return {"id": p.get("id"), "round": slug, "title": p.get("title"), "type": qtype,
            "resolution": str(res).lower(),
            "open_time": q.get("open_time"), "close_time": q.get("scheduled_close_time"),
            "community_prediction": cp}

def do_fetch(a):
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN missing", file=sys.stderr); return 2
    rows, skipped, raw_total = [], 0, 0
    for s in rounds():
        ps = posts(s, token)
        raw_total += len(ps)
        keep = [x for x in (flatten(p, s) for p in ps) if x]
        if not keep:
            skipped += 1
        else:
            rows += keep
            print("  " + s + ": " + str(len(keep)) + " of " + str(len(ps)))
        time.sleep(0.6)
    json.dump(rows, open(a.out, "w"), indent=1)
    nb = sum(1 for r in rows if r["type"] == "binary")
    ncp = sum(1 for r in rows if r.get("community_prediction") is not None)
    print("raw posts seen: " + str(raw_total) + ", skipped rounds: " + str(skipped))
    print("wrote " + str(len(rows)) + " questions (" + str(nb) + " binary, " + str(ncp) + " with CP)")
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
    types = defaultdict(int)
    for r in rows:
        types[r["type"]] += 1
    print("types: " + json.dumps(dict(types)))
    bins = [r for r in rows if r["type"] == "binary" and r["resolution"] in ("yes", "no")]
    print(str(len(rows)) + " questions, " + str(len(bins)) + " scorable binary")
    if not bins: return 1
    base = sum(1 for r in bins if r["resolution"] == "yes") / len(bins)
    print("observed YES base rate: " + format(base, ".3f"))
    print("")
    for name, fn in [("community prediction (baseline)", lambda r: r.get("community_prediction")),
                     ("constant base rate", lambda r: base),
                     ("flat 25 percent status quo", lambda r: 0.25)]:
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
