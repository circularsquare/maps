"""T-1536: per period, before and after: boardings, turned away, trains full, lines with busy platforms,
and how many lines have Medium's dot above High's.  usage: python periods.py before.json after.json"""
import json
import sys

PN = ["6-10", "10-16", "16-20", "20-24", "0-6"]


def load(path):
    return json.load(open(path))


def per_period(d):
    out = [dict(board=0.0, gone=0.0, full=0.0, trains=0.0, busy=0) for _ in range(5)]
    for r in d["rows"]:
        for p in range(5):
            x = r["per"][p]
            a = x.get("acct")
            if not a:
                continue
            out[p]["board"] += x.get("board", 0)
            out[p]["gone"] += a[8]
            out[p]["full"] += a[1]
            out[p]["trains"] += a[0]
            out[p]["busy"] += a[10] > 0
    return out


def dots(d):
    """per line, the level's dot in minutes: pooled over the level's periods by those riders"""
    inv = 0
    for r in d["rows"]:
        lv = {}
        for p in range(5):
            a = r["per"][p].get("acct")
            if not a:
                continue
            L = lv.setdefault(d["periods"][p]["demand"], [0.0, 0.0])
            L[0] += a[9]
            L[1] += a[10]
        dot = {k: (v[0] / v[1] / 60 if v[1] else 0.0) for k, v in lv.items()}
        if dot.get("medium", 0) > dot.get("high", 0) + 0.05:
            inv += 1
    return inv


for path in sys.argv[1:]:
    d = load(path)
    print(path.split("/")[-1])
    for q, v in zip(PN, per_period(d)):
        print(f"  {q:6s} boardings {v['board']:11,.0f}  turned away {v['gone']:9,.0f}  trains full {v['full']:6,.0f}/{v['trains']:6,.0f}  lines with busy platforms {v['busy']}")
    print("  lines with Medium's dot above High's:", dots(d))
