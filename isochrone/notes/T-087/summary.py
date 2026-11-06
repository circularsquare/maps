"""T-087: one line a rule: mean and worst of the median |100 m - 1 km| / 1 km and of |ratio - 1| over tiles.
    python isochrone/notes/T-087/summary.py out.json rule,rule,... [exclude,tile,...]
"""
import json
import sys

data = json.load(open(sys.argv[1]))
ex = sys.argv[3].split(",") if len(sys.argv) > 3 else []
rules = sys.argv[2].split(",") if sys.argv[2] != "all" else list(data[-1]["rules"])
data = [t for t in data if t["name"] not in ex]
print("%d tiles" % len(data))
print("%-18s %9s %9s %9s %9s %9s" % ("rule", "mean rel", "max rel", "mean|r-1|", "max|r-1|", "mean bias"))
for r in rules:
    rel = [t["rules"][r]["rel"] for t in data if r in t["rules"]]
    rr = [abs(t["rules"][r]["ratio"] - 1) for t in data if r in t["rules"]]
    bi = [t["rules"][r]["bias"] for t in data if r in t["rules"]]
    print("%-18s %8.1f%% %8.1f%% %8.1f%% %8.1f%% %+8.1f%%" % (r, 100 * sum(rel) / len(rel), 100 * max(rel), 100 * sum(rr) / len(rr),
                                                           100 * max(rr), 100 * sum(bi) / len(bi)))
