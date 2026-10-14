"""T-288 stage a: sanity figures for the London tier 0 pack (notes/T-288.md).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t288_figures.py

Population and jobs by ITL3 group (Greater London = TLI*, inner London = TLI3* and TLI4*, the
ring = the rest), jobs within a few km of Bank, Census 2021 district sums for the same groups, and
the commute's straight-line mean and median under the pack's curve over 2 km zones (tier0's model).
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import pandas as pd
import shapely
from shapely import union_all
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
import tier0  # noqa: E402
from read_pack import load_pack  # noqa: E402

CITY = "lon"


def main():
    h, a = load_pack(CITY)
    lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
    x, y = a["x_m"].astype(np.float64), a["y_m"].astype(np.float64)
    lon = lon0 + np.degrees(x / (b.R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / b.R_EARTH)
    pop, jobs = a["pop"].astype(np.float64), a["jobs"].astype(np.float64)
    with open(england.raw("lon_itl3.geojson"), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    groups = {
        "Greater London": [f for f in feats if f["properties"]["ITL325CD"].startswith("TLI")],
        "inner London": [f for f in feats if f["properties"]["ITL325CD"][:4] in ("TLI3", "TLI4")],
        "Westminster and City (TLI35)": [f for f in feats if f["properties"]["ITL325CD"] == "TLI35"],
        "the ring": [f for f in feats if not f["properties"]["ITL325CD"].startswith("TLI")],
    }
    out = {"cells": int(h["cells"]), "zones": int(h["zones"]), "pop": float(pop.sum()), "jobs": float(jobs.sum()),
           "commuters": float(a["commute_work"].sum()), "groups": {}}
    lads = england.lad_totals(union_all([shape(f["geometry"]) for f in feats]).buffer(0))
    for name, fs in groups.items():
        g = union_all([shape(f["geometry"]) for f in fs]).buffer(0)
        shapely.prepare(g)
        sel = shapely.contains_xy(g, lon, lat)
        census = sum(p for _, lg, p in lads if g.contains(lg.representative_point()))
        out["groups"][name] = {"pop": float(pop[sel].sum()), "census_2021": census, "jobs": float(jobs[sel].sum()),
                               "jobs_share": float(jobs[sel].sum() / jobs.sum())}
    # jobs near Bank
    bx = b.R_EARTH * np.radians(-0.0886 - lon0) * np.cos(np.radians(lat0))
    by = b.R_EARTH * np.radians(51.5133 - lat0)
    d = np.hypot(x - bx, y - by)
    out["jobs_near_bank"] = {f"{r} km": float(jobs[d < r * 1000].sum()) for r in (1, 2, 3, 5, 10)}
    # the commute under the shipped curve
    g = h["gravity"]
    zone, gx, gy = tier0.zones_of(x, y)
    H = np.bincount(zone, weights=a["commute_home"].astype(np.float64))
    J = np.bincount(zone, weights=a["commute_work"].astype(np.float64))
    dz = tier0.zone_distances(gx, gy)
    mean, med = tier0.trip_stats(tier0.gravity(H, J, dz, g.get("decay_pow", 1.0), g["decay_km"]), dz)
    out["commute_straight_km"] = {"curve": f"d^-{g.get('decay_pow')} exp(-d/{g['decay_km']} km)", "mean": mean, "median": med}
    print(json.dumps(out, indent=1))
    with open(os.path.join(ROOT, "..", "artifacts", "trainworld", "T-288-figures.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
