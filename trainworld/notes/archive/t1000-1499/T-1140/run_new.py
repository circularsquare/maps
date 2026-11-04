"""T-1140 port check: the planner's own zone_groups (tree t1150) on T-1140's US data, against the
design's `rbp` run. Writes new_runs.pkl here.

    python run_new.py <tag> <big_km2>
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8a3f6fa5-349e-4cb4-90bd-f8d565510cd9\scratchpad\t1140"
sys.path.insert(0, r"C:\Users\anita\projects\maps\.tw-trees\t1150\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import world_plan as wp  # noqa: E402

t0 = time.time()
tag, big = sys.argv[1], float(sys.argv[2])
D = pickle.load(open(os.path.join(OLD, "us_data.pkl"), "rb"))
P = pickle.load(open(os.path.join(OLD, "us_prep.pkl"), "rb"))
out = {}
for iso, d in D.items():
    U = d["units"]
    rule = wp.ZONE_RULES[iso]
    region = {i: u.parents[0] for i, u in U.items()}
    flows = P[iso]["acs"] or P[iso]["grav"]
    groups, lines = wp.zone_groups(list(U.values()), list(d["free"]), region, rule["floor"], rule["cap"], True, d["fixed"],
                                   flows, {f["id"]: f["units"] for f in d["fixed"]}, big)
    for lead in [g for g, ids in groups.items() if sum(U[i].free for i in ids) < wp.ZONE_ALONE_MIN]:
        groups.pop(lead)
    out[iso] = {"groups": groups, "lines": lines, "tests": []}
    print(f"[{time.time() - t0:5.0f}s] {tag} {iso}: {len(groups)} zones", flush=True)
path = os.path.join(HERE, "new_runs.pkl")
allr = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
allr[tag] = out
pickle.dump(allr, open(path, "wb"))
ref = pickle.load(open(os.path.join(OLD, "us_runs.pkl"), "rb"))["rbp"]
for iso in out:
    a = {frozenset(v) for v in out[iso]["groups"].values()}
    b = {frozenset(v) for v in ref[iso]["groups"].values()}
    print(f"{iso}: {len(a)} zones, design's rbp {len(b)}, identical {len(a & b)}")
