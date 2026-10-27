"""T-1536: network totals from probe.mjs output, one or two files (before, after).

usage: python analyse.py before.json [after.json]
Running cost a day is tph x 2 x km x cost a train-km, summed over the clock hours at each hour's
level (as notes/t1523/analyse3.py did); `runDaily` is the game's own figure (moneyState).
"""
import json
import sys

LEV = {"high": "high", "medium": "medium", "low": "low"}


def figs(path):
    d = json.load(open(path))
    cost = trips = gone = arrivals = 0.0
    n = 0
    for r in d["rows"]:
        if not r["per"][0].get("acct"):
            continue
        n += 1
        cost += sum(r["tph"][LEV[lv]] for lv in d["levelOfHour"]) * 2 * (r["km"] or 0) * r["perTrainKm"]
        for p in r["per"]:
            trips += p.get("board", 0)
            a = p.get("acct") or [0] * 11
            gone += a[8]
            arrivals += a[2]
    return dict(lines=n, cost=cost / 1e6, run_daily=d["runDaily"], boardings=trips, gone=gone, arrivals=arrivals,
                rail=d["rail"], trips=d["trips"], missing=d["missing"], levels=d["levelOfHour"], weekday=d["weekday"])


a = figs(sys.argv[1])
b = figs(sys.argv[2]) if len(sys.argv) > 2 else None
keys = ["lines", "cost", "run_daily", "boardings", "gone", "arrivals", "rail", "trips", "missing", "weekday"]
for k in keys:
    if b:
        print(f"{k:10s} {a[k]:>16,.2f} {b[k]:>16,.2f}" if isinstance(a[k], float) else f"{k:10s} {a[k]!s:>16} {b[k]!s:>16}")
    else:
        print(f"{k:10s} {a[k]}")
print("levels by hour:", "".join(x[0].upper() for x in a["levels"]), ("-> " + "".join(x[0].upper() for x in b["levels"])) if b else "")
if b:
    da, db = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    ta = {r["num"]: r["tph"] for r in da["rows"]}
    same = sum(1 for r in db["rows"] if ta.get(r["num"]) == r["tph"])
    print(f"lines with the same three counts before and after: {same} of {len(db['rows'])}")
