"""One table line per country and variant: zones, rings, rows changed."""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
for iso in sys.argv[1].split(","):
    R = pickle.load(open(os.path.join(HERE, f"{iso}_out.pkl"), "rb"))
    bsets = {frozenset(v) for v in R["before"]["groups"].values()}
    out = []
    for v in ("before", "near_cut", "repair_only"):
        if v not in R:
            continue
        r = R[v]["res"]
        ch = sum(frozenset(x) not in bsets for x in R[v]["groups"].values())
        gone = sum(frozenset(x) not in {frozenset(y) for y in R[v]["groups"].values()} for x in R["before"]["groups"].values())
        out.append(f"{v}: {r['zones']} zones, median {r['median'] / 1e6:.2f}M, rings {r['ring_zones']} "
                   f"({r['rings_zz']} zone, {r['rings_zm']} metro), over cap {r['over_cap']}, "
                   f"plan rows changed {gone}, new rows {ch}")
    print(iso, " | ".join(out))
