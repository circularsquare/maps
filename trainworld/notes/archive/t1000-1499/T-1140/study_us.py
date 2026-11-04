"""T-1140 "very big regions": the planner's rule on the US with a zone counting as full at ZONE_MIN_POP
over `big` km2, for several `big`. Prints the zones under the floor each makes, and the West's zones.

    python study_us.py 0 30000 50000 80000
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
import time

import common as C
import world_plan as wp

sys.stdout.reconfigure(encoding="utf-8")
WEST = {"Washington", "Oregon", "California", "Nevada", "Idaho", "Montana", "Wyoming", "Utah", "Colorado", "Arizona",
        "New Mexico", "Alaska", "Hawaii"}
D = C.load("us_data.pkl")
P = C.load("us_prep.pkl")
d = D["USA"]
U = d["units"]
km2 = {i: u.geom.area / 1e6 for i, u in U.items()}
region = {i: u.parents[0] for i, u in U.items()}
fixed_units = {f["id"]: f["units"] for f in d["fixed"]}
F = C.acs_sym(d["flows"], list(U))
out = {}
for arg in sys.argv[1:]:
    mode, big = arg.split(":") if ":" in arg else ("stop", arg)
    big = float(big)
    wp.BIG_MODE = mode
    t0 = time.time()
    g, _ = wp.zone_groups(list(U.values()), list(d["free"]), region, 500_000, 1_500_000, True, d["fixed"], P["USA"]["acs"],
                          fixed_units, big)
    g = {k: v for k, v in g.items() if sum(U[i].free for i in v) >= wp.ZONE_ALONE_MIN}
    out[arg] = g
    own = {i: k for k, v in g.items() for i in v}
    for f in d["fixed"]:
        for i in f["units"]:
            own.setdefault(i, f["id"])
    tot = kept = 0.0
    for k, ids in g.items():
        for h in ids:
            for w, n in (d["flows"] or {}).get(h, {}).items():
                if w != h:
                    tot += n
                    kept += n * (own.get(w) == k)
    rows = []
    for k, ids in g.items():
        p = sum(U[i].free for i in ids)
        a = sum(km2[i] for i in ids)
        st = U[max(ids, key=lambda i: U[i].free)].parents[0][1]
        rows.append((st, U[max(ids, key=lambda i: U[i].free)].name, p, a, len(ids)))
    under = [r for r in rows if r[2] < 500_000]
    west = [r for r in rows if r[0] in WEST]
    print(f"\n=== {mode} {big:,.0f} km2: {len(g)} zones, {len(under)} under 0.5M, leavers kept {kept / tot:.1%}, "
          f"West {len(west)} zones ({time.time() - t0:.0f}s)")
    for r in sorted(under, key=lambda r: -r[3]):
        print(f"  under  {r[0][:12]:12} {r[1][:28]:28} {r[2] / 1e6:5.2f}M {r[3]:9,.0f} km2 n{r[4]}")
    for r in sorted(west):
        print(f"  west   {r[0][:12]:12} {r[1][:28]:28} {r[2] / 1e6:5.2f}M {r[3]:9,.0f} km2 n{r[4]}")
pickle.dump(out, open(os.path.join(C.HERE, f"study_us_{len(sys.argv)}_{sys.argv[1].replace(':', '-')}.pkl"), "wb"))
