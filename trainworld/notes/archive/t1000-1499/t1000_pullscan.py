"""Every pack: mean commute, the share of jobs in zones whose balance pull passes 10x (either way),
cross-pack commuters out, province (solve_region). Writes artifacts/trainworld/t1000-pullscan.json
(T-1000)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import glob
import json
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1000lib import load_pack

P = r"C:/Users/anita/projects/maps/trainworld/data/packs/"
plan = json.load(open(r"C:/Users/anita/projects/maps/trainworld/pipeline/world_plan.json", encoding="utf-8"))
info = {}
for r in plan["metros"] + [r for v in plan["zones"].values() if isinstance(v, list) for r in v if isinstance(r, dict)]:
    info[r.get("id")] = (r.get("iso2"), r.get("solve_region") or r.get("region"), r.get("name"), r.get("status"), r.get("kind"))
out = []
for f in sorted(glob.glob(P + "*.json")):
    city = os.path.basename(f)[:-5]
    if "." in city:
        continue
    try:
        h = json.load(open(f, encoding="utf-8"))
        g = h.get("gravity")
        if not isinstance(g, dict) or "mean_trip_km" not in g:
            continue
        h, a = load_pack(city)
    except Exception as e:
        continue
    nz = h["zones"]
    zone = a["zone"].astype(np.int64)
    ww = a["commute_work"] if "commute_work" in a else a["jobs"]
    J = np.bincount(zone, weights=ww.astype(np.float64), minlength=nz)
    col = a["zone_col"].astype(np.float64)
    jm = (J > 0) & (col > 0)
    b = col[jm] / J[jm]
    gm = np.exp((J[jm] * np.log(b)).sum() / J[jm].sum())
    b = b / gm
    wj = J[jm]
    row = {"id": city, "country": h.get("country") or (info.get(city, ("?",))[0]), "region": info.get(city, (None, None))[1],
           "name": info.get(city, (None, None, city))[2], "status": info.get(city, (None,) * 4)[3], "kind": info.get(city, (None,) * 5)[4],
           "zones": nz, "workers": float(J.sum()), "mean_km": g["mean_trip_km"], "joint": "zone_home" in a,
           "pull_p99": float(np.percentile(b, 99)), "pull_max": float(b.max()), "pull_min": float(b.min())}
    for m in (10, 20):
        row[f"jobs_past_{m}"] = float(wj[(b > m) | (b < 1 / m)].sum() / wj.sum())
    j = g.get("joint")
    if isinstance(j, dict):
        row["out"] = j.get("commuters_out")
        row["in"] = j.get("commuters_in")
    out.append(row)
json.dump(out, open(r"C:/Users/anita/projects/maps/artifacts/trainworld/t1000-pullscan.json", "w"), indent=0)
print(len(out), "packs")
