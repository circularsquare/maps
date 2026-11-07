"""T-007: world totals of count.py's per-tile json.

    python isochrone/notes/T-007/sum.py <count dir>
"""
import glob
import json
import os
import sys

tot = {}
n = 0
for fn in glob.glob(os.path.join(sys.argv[1], "count_*.json")):
    r = json.load(open(fn))
    n += 1
    for lv, d in r.items():
        if lv == "sec":
            continue
        t = tot.setdefault(lv, {})
        for k, v in d.items():
            t[k] = t.get(k, 0) + v
print("tiles", n)
for lv in ("L10_on", "L10_off", "L3_on", "L3_off"):
    t = tot.get(lv)
    if not t:
        continue
    c = t["cells"]
    pc = lambda x: "%d (%.2f%%)" % (x, 100.0 * x / c)
    print(lv, "land cells", c)
    print("  multi (2+ linked pieces)", pc(t["multi"]), " further nodes", t["extra_nm"],
          " by water: " + ", ".join("%s %d" % (k, t["multi_" + k]) for k in ("fjord", "strait", "river", "lake")))
    print("  barrier (pieces not joined in the 3x3 block)", pc(t["barrier"]), " further nodes", t["extra"],
          " by water: " + ", ".join("%s %d" % (k, t["barrier_" + k]) for k in ("fjord", "strait", "river", "lake")))
    print("  neighbour pairs", t["edge_pairs"], " no land across", t["edge_noadj"], " not joined in 3x3", t["edge_barrier"],
          " by water: " + ", ".join("%s %d" % (k, t["edge_" + k]) for k in ("fjord", "strait", "river", "lake")))
    print("  listed (multi, or a pair with no land across)", pc(t["listed"]))
