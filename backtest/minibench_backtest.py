#!/usr/bin/env python3
"""MiniBench backtest harness."""
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
                return resp.status, json.loads(resp.read().decode()), ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 400):
                time.sleep(1.5 * (i + 1)); continue
            return e.code, None, ""
        except Exception as e:
            return -1, None, str(e)[:120]
    return -2, None, "throttled"

def posts(slug, token):
    out, off = [], 0
    while True:
        code, data, _ = call("/posts/", token, {"tournaments": slug, "statuses": "resolved", "limit": PAGE, "offset": off, "with_cp": "true"})
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

def do_probe(a):
    """Dump the real shape of one list item and its detail record."""
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN missing", file=sys.stderr); return 2
    ps, code = posts("minibench-2026-07-27", token)
    print("http " + str(code) + "  posts=" + str(len(ps)))
    if not ps:
        return 1
    p = ps[0]
    print("=== LIST ITEM (truncated) ===")
    print(json.dumps(p)[:2500])
    print("=== DETAIL ===")
    c2, det, _ = call("/posts/" + str(p.get("id")) + "/", token)
    print("detail http " + str(c2))
    if isinstance(det, dict):
        dq = det.get("question") or {}
        print("detail question keys: " + ",".join(sorted(dq.keys())))
        print("detail resolution: " + repr(dq.get("resolution")))
        print("detail type: " + repr(dq.get("type")))
        print("detail post resolved: " + repr(det.get("resolved")))
        print("=== DETAIL QUESTION (truncated) ===")
        print(json.dumps(dq)[:2000])
    return 0

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
pr = sub.add_parser("probe"); pr.set_defaults(fn=do_probe)
f = sub.add_parser("fetch"); f.add_argument("--out", default="dataset.json"); f.set_defaults(fn=do_probe)
e = sub.add_parser("evaluate"); e.add_argument("--dataset", default="dataset.json"); e.set_defaults(fn=lambda a: 0)
if __name__ == "__main__":
    args = ap.parse_args()
    raise SystemExit(args.fn(args))
