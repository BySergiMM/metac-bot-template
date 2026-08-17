#!/usr/bin/env python3
"""MiniBench backtest harness.

fetch     -> download every resolved MiniBench question via the Metaculus API
evaluate  -> score baseline strategies against the community prediction

METACULUS_TOKEN comes from the environment (a GitHub secret). Never logged.
Touches resolved questions only, so it cannot breach the no-human-in-the-loop rule.
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, sys, time
import urllib.error, urllib.parse, urllib.request
from collections import defaultdict

API = "https://www.metaculus.com/api"
PAGE = 100
UA = {"User-Agent": "minibench-backtest/1.0", "Accept": "application/json"}

def call(path, token, params=None):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    h = {"Authorization": "Token " + token}
    h.update(UA)
    r = urllib.request.Request(url, headers=h)
    for i in range(3):
        try:
            with urllib.request.urlopen(r, timeout=45) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 400):
                time.sleep(1.5 * (i + 1)); continue
            return e.code, None
        except Exception:
            return -1, None
    return -2, None

def posts(slug, token):
    out, off = [], 0
    while True:
        code, data = call("/posts/", token, {"tournaments": slug, "statuses": "resolved",
                                             "limit": PAGE, "offset": off, "with_cp": "true"})
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

def community_prediction(q):
    """Aggregations live under the method the question declares, not always
    recency_weighted. Reading the wrong key silently yields None for everything."""
    agg = q.get("aggregations") or {}
    for m in [q.get("default_aggregation_method"), "recency_weighted", "unweighted"]:
        if not m:
            continue
        latest = (agg.get(m) or {}).get("latest") or {}
        centers = latest.get("centers") or []
        if centers:
            return centers[0]
    return None

def flatten(p, slug):
    q = p.get("question") or {}
    if p.get("status") != "resolved":
        return None
    res = q.get("resolution")
    if res in (None, "", "annulled", "ambiguous"):
        return None
    return {"id": p.get("id"), "round": slug, "title": p.get("title"),
            "type": q.get("type") or "unknown",
            "resolution": str(res).strip().lower(),
            "open_time": q.get("open_time"),
            "close_time": q.get("scheduled_close_time"),
            "nr_forecasters": p.get("nr_forecasters"),
            "community_prediction": community_prediction(q)}

def do_fetch(a):
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN missing", file=sys.stderr); return 2
    rows, raw, dropped = [], 0, defaultdict(int)
    for s in rounds():
        ps, code = posts(s, token)
        raw += len(ps)
        if code != 200:
            continue
        for p in ps:
            r = flatten(p, s)
            if r:
                rows.append(r)
            else:
                dropped[(p.get("question") or {}).get("type") or "unknown"] += 1
        time.sleep(0.5)
    json.dump(rows, open(a.out, "w"), indent=1)
    kept = defaultdict(int)
    for r in rows:
        kept[r["type"]] += 1
    print("raw posts: " + str(raw) + "  kept: " + str(len(rows)))
    print("kept by type: " + json.dumps(dict(kept)))
    print("dropped by type: " + json.dumps(dict(dropped)))
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
        print("  n=" + str(s.n) + "  Brier=" + format(s.b / s.n, ".4f") +
              "  log=" + format(s.l / s.n, ".4f"))
        for b in range(10):
            o = s.buck.get(b, [])
            if o:
                print("    " + str(b*10) + "-" + str(b*10+9) + "% -> actual " +
                      format(sum(o)/len(o)*100, ".1f") + "%  n=" + str(len(o)))
        print("")

def do_eval(a):
    rows = json.load(open(a.dataset))
    types = defaultdict(int)
    for r in rows:
        types[r["type"]] += 1
    print("types: " + json.dumps(dict(types)))
    bins = [r for r in rows if r["type"] == "binary" and r["resolution"] in ("yes", "no")]
    withcp = [r for r in bins if r.get("community_prediction") is not None]
    print(str(len(rows)) + " questions, " + str(len(bins)) + " binary yes/no, " +
          str(len(withcp)) + " with community prediction")
    if not bins:
        return 1
    base = sum(1 for r in bins if r["resolution"] == "yes") / len(bins)
    print("observed YES base rate: " + format(base, ".3f"))
    print("")
    for name, fn in [("community prediction (baseline)", lambda r: r.get("community_prediction")),
                     ("constant base rate", lambda r: base),
                     ("flat 25 percent status quo", lambda r: 0.25),
                     ("always 0.5", lambda r: 0.5)]:
        res = Res(name)
        for r in bins:
            p = fn(r)
            if p is None:
                continue
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
