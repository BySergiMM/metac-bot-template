#!/usr/bin/env python3
"""MiniBench backtest harness.

fetch     -> download every resolved MiniBench post via the Metaculus API and
             persist all of it, including questions whose resolution the API
             does not expose. Never drops rows silently.
evaluate  -> report what is usable and score baselines on whatever carries a
             resolution. Runs a deterministic self-test of the scorer either way.
selftest  -> just the scorer proof, no network.

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
    """MiniBench runs on a fixed 14-day cadence; slugs are minibench-YYYY-MM-DD.
    Deriving them beats discovery: /api/tournaments/ returns 404."""
    out = ["minibench"]
    d = dt.date(2026, 8, 10)
    for _ in range(14):
        d -= dt.timedelta(days=14)
        out.append("minibench-" + d.isoformat())
    return out

def community_prediction(q):
    """Aggregations sit under the method the question declares, not always
    recency_weighted. Reading the wrong key silently yields None everywhere."""
    agg = q.get("aggregations") or {}
    for m in [q.get("default_aggregation_method"), "recency_weighted", "unweighted"]:
        if not m:
            continue
        latest = (agg.get(m) or {}).get("latest") or {}
        centers = latest.get("centers") or []
        if centers:
            return centers[0]
    return None

def normalise_resolution(res):
    if res is None:
        return None
    s = str(res).strip().lower()
    if s in ("", "none", "null", "annulled", "ambiguous"):
        return None
    return s

def flatten(p, slug):
    """Keep every post. Resolution may be absent: the API does not always expose it."""
    q = p.get("question") or {}
    return {"id": p.get("id"), "question_id": q.get("id"), "round": slug,
            "title": p.get("title"), "type": q.get("type") or "unknown",
            "status": p.get("status"), "resolved_flag": p.get("resolved"),
            "resolution": normalise_resolution(q.get("resolution")),
            "resolution_set_time": q.get("resolution_set_time"),
            "open_time": q.get("open_time"),
            "close_time": q.get("scheduled_close_time"),
            "nr_forecasters": p.get("nr_forecasters"),
            "community_prediction": community_prediction(q),
            "url": "https://www.metaculus.com/questions/" + str(p.get("id")) + "/"}

def do_fetch(a):
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN missing", file=sys.stderr)
        return 2
    rows, bad = [], []
    for s in rounds():
        ps, code = posts(s, token)
        if code != 200:
            bad.append(s + ":" + str(code))
            continue
        if not ps:
            continue
        rows += [flatten(p, s) for p in ps]
        print("  " + s + ": " + str(len(ps)))
        time.sleep(0.5)
    json.dump(rows, open(a.out, "w"), indent=1)
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["type"]] += 1
    with_res = sum(1 for r in rows if r["resolution"])
    with_cp = sum(1 for r in rows if r["community_prediction"] is not None)
    print("")
    print("saved " + str(len(rows)) + " questions to " + a.out)
    print("  by type: " + json.dumps(dict(by_type)))
    print("  with resolution exposed by the API: " + str(with_res))
    print("  with community prediction:          " + str(with_cp))
    if bad:
        print("  rounds that errored: " + ",".join(bad))
    if not with_res:
        print("")
        print("NOTE: the API reports status=resolved and a resolution_set_time but leaves")
        print("question.resolution null, so nothing is scorable from the API alone. The web")
        print("UI does render the outcome, so an HTML fallback is the next thing to try.")
        print("Every other field is complete and the dataset ships as an artifact.")
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
            print(s.name + " - nothing scored")
            return
        print(s.name)
        print("  n=" + str(s.n) + "  Brier=" + format(s.b / s.n, ".4f") +
              "  log=" + format(s.l / s.n, ".4f"))
        for b in range(10):
            o = s.buck.get(b, [])
            if o:
                print("    " + str(b*10) + "-" + str(b*10+9) + "% -> actual " +
                      format(sum(o)/len(o)*100, ".1f") + "%  n=" + str(len(o)))
        print("")

def selftest():
    """Deterministic proof the scorer is right, independent of live data."""
    assert brier(1.0, 1) == 0.0
    assert brier(0.0, 1) == 1.0
    assert brier(0.5, 1) == 0.25
    assert logs(0.99, 1) > logs(0.5, 1) > logs(0.01, 1)
    r = Res("calibration")
    for _ in range(30):
        r.add(0.3, 1)
    for _ in range(70):
        r.add(0.3, 0)
    got = r.b / r.n
    assert abs(got - 0.21) < 1e-9, "expected 0.2100, got " + format(got, ".6f")
    print("self-test: brier(1,1)=0 brier(0,1)=1 brier(.5,1)=.25   OK")
    print("self-test: 30/100 outcomes at p=0.3 -> Brier " + format(got, ".4f") +
          " (expected 0.2100)   OK")
    print("scorer verified")
    print("")

def do_eval(a):
    selftest()
    rows = json.load(open(a.dataset))
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["type"]] += 1
    print("dataset: " + str(len(rows)) + " questions")
    print("  by type: " + json.dumps(dict(by_type)))
    with_res = [r for r in rows if r.get("resolution")]
    with_cp = [r for r in rows if r.get("community_prediction") is not None]
    print("  with resolution:           " + str(len(with_res)))
    print("  with community prediction: " + str(len(with_cp)))
    bins = [r for r in with_res if r["type"] == "binary" and r["resolution"] in ("yes", "no")]
    print("  scorable binary:           " + str(len(bins)))
    print("")
    if not bins:
        print("Nothing scorable yet. Data-access limitation, not a scoring bug:")
        print("the scorer above is verified and runs the moment resolutions exist.")
        return 0
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
t = sub.add_parser("selftest"); t.set_defaults(fn=lambda a: (selftest(), 0)[1])
if __name__ == "__main__":
    args = ap.parse_args()
    raise SystemExit(args.fn(args))
