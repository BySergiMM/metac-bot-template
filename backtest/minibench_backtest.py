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
    for i in range(3):
        try:
            with urllib.request.urlopen(r, timeout=45) as resp:
                return resp.status, json.loads(resp.read().decode()), ""
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            if e.code in (429, 400):
                time.sleep(1.5 * (i + 1)); continue
            return e.code, None, body
        except Exception as e:
            return -1, None, str(e)[:120]
    return -2, None, "throttled"

def posts(slug, token):
    out, off = [], 0
    while True:
        code, data, body = call("/posts/", token, {"tournaments": slug, "statuses": "resolved", "limit": PAGE, "offset": off, "with_cp": "true"})
        if code != 200 or not isinstance(data, dict):
            return out, code
        res = data.get("results", [])
        out += res
        if not res or not data.get("next"):
            return out, 200
        off += PAGE
        time.sleep(0.4)

def rounds():
    out = ["minibench"]
    d = dt.date(2026, 8, 10)
    for _ in range(14):
        d -= dt.timedelta(days=14)
        out.append("minibench-" + d.isoformat())
    return out

def resolution_of(p):
    q = p.get("question") or {}
    for src in (q, p):
        for k, v in src.items():
            if "resol" in k.lower() and isinstance(v, (str, int, float)) and str(v) not in ("", "None", "True", "False"):
                return v
    return None

def flatten(p, slug):
    q = p.get("question") or {}
    if not DUMPED["done"]:
        DUMPED["done"] = True
        print("FULL question keys: " + ",".join(sorted(q.keys())))
        print("resol-ish in question: " + json.dumps({k: str(v)[:60] for k, v in q.items() if "resol" in k.lower()}))
        print("resol-ish in post: " + json.dumps({k: str(v)[:60] for k, v in p.items() if "resol" in k.lower()}))
        print("type=" + repr(q.get("type")) + " post.resolved=" + repr(p.get("resolved")))
    res = resolution_of(p)
    if res is None:
        return None
    cp = None
    latest = ((q.get("aggregations") or {}).get("recency_weighted") or {}).get("latest") or {}
    c = latest.get("centers") or []
    if c: cp = c[0]
    return {"id": p.get("id"), "round": slug, "title": p.get("title"),
            "type": q.get("type") or "unknown", "resolution": str(res).lower(),
            "open_time": q.get("open_time"), "close_time": q.get("scheduled_close_time"),
            "community_prediction": cp}

def do_fetch(a):
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN missing", file=sys.stderr); return 2
    rows, raw = [], 0
    for s in rounds():
        ps, code = posts(s, token)
        raw += len(ps)
        if code != 200:
            print("  " + s + ": http " + str(code)); continue
        keep = [x for x in (flatten(p, s) for p in ps) if x]
        print("  " + s + ": " + str(len(keep)) + " kept of " + str(len(ps)))
        rows += keep
        time.sleep(0.5)
    json.dump(rows, open(a.out, "w"), indent=1)
    types = defaultdict(int)
    for r in rows: types[r["type"]] += 1
    print("raw posts: " + str(raw))
    print("types kept: " + json.dumps(dict(types)))
    print("wrote " + str(len(rows)) + " questions")
    return 0 if rows else 1

def brier(p, o): return (p - o) ** 2
def logs(p, o):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p) if o else math.log(1 - p)

class Res:
    def __init__(s, name):
        s.name, s.n, s.b, s.l = name, 0, 0.0, 0.0
        s.buck = defaultdict(list)
    def add(s, p, o):
        s.n += 1; s.b += brier(p, o); s.l += logs(p, o)
        s.buck[min(int(p * 10), 9)].append(o)
    def show(s):
        if not s.n:
            print(s.name + " - nothing scored"); return
        print(s.name)
        print("  n=" + str(s.n) + "  Brier=" + format(s.b / s.n, ".4f") + "  log=" + format(s.l / s.n, ".4f"))
        for b in range(10):
            o = s.buck.get(b, [])
            if o: print("    " + str(b*10) + "-" + str(b*10+9) + "% -> actual " + format(sum(o)/len(o)*100, ".1f") + "%  n=" + str(len(o)))
        print("")

def do_eval(a):
    rows = json.load(open(a.dataset))
    types = defaultdict(int)
    for r in rows: types[r["type"]] += 1
    print("types: " + json.dumps(dict(types)))
    vals = defaultdict(int)
    for r in rows: vals[r["resolution"]] += 1
    print("top resolutions: " + json.dumps(dict(sorted(vals.items(), key=lambda x: -x[1])[:8])))
    bins = [r for r in rows if r["resolution"] in ("yes", "no")]
    print(str(len(rows)) + " questions, " + str(len(bins)) + " yes/no scorable")
    if not bins: return 1
    base = sum(1 for r in bins if r["resolution"] == "yes") / len(bins)
    print("observed YES base rate: " + format(base, ".3f") + chr(10))
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
