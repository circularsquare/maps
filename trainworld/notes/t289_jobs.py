"""T-289: tier 0's jobs proxy against INSEE's 2022 workers by commune of work in Ile-de-France
(demand-worldwide.md asked for Paris's core against a published count).

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t289_jobs.py

Needs the pack's cached signals (data/work/par.tier0_signals.npz). Prints totals for Paris, the
inner ring and the outer, jobs within 1/2/5 km of the origin, and jobs misplaced between
communes, départements and 2 km zones; writes artifacts/trainworld/T-289-jobs.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
import france  # noqa: E402
import tier0  # noqa: E402

CITY = "par"


def miss(a, c):
    return 0.5 * float(np.abs(a / a.sum() - c / c.sum()).sum())


def main():
    cfg = b.CITIES[CITY]
    sig = dict(np.load(os.path.join(b.WORK, f"{CITY}.tier0_signals.npz")))
    cs = france.communes(cfg["departements"])
    pop, _ = france.commune_population(CITY, None, sig, cfg["departements"])
    sig["pop"] = pop
    jpr = tier0.worldbank(cfg["iso2"])[0]
    proxy = tier0.jobs_from(sig, pop.sum() * jpr)
    insee, _ = france.insee_jobs(CITY, None, sig, tier0.jobs_from(sig, 1.0), cfg["departements"])
    li, _ = england.cell_units(sig["lat"], sig["lon"], [(c[0], c[3]) for c in cs])
    li = np.where(li < 0, 0, li)
    dep = np.array([c[2] for c in cs])[li]
    lon0, lat0 = cfg["origin"]
    x = b.R_EARTH * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0))
    y = b.R_EARTH * np.radians(sig["lat"] - lat0)
    r = np.hypot(x, y) / 1000
    zone, _, _ = tier0.zones_of(x, y)
    groups = {"Paris (75)": dep == "75", "inner ring (92, 93, 94)": np.isin(dep, ["92", "93", "94"]),
              "outer ring": np.isin(dep, ["77", "78", "91", "95"])}
    res = {"jobs_per_resident": jpr, "totals": {"proxy": proxy.sum(), "insee": insee.sum()}}
    for k, m in groups.items():
        res[k] = {"proxy": proxy[m].sum(), "insee": insee[m].sum()}
    for km in (1, 2, 5):
        res[f"within {km} km of the origin"] = {"proxy": proxy[r <= km].sum(), "insee": insee[r <= km].sum()}
    res["misplaced"] = {k: miss(np.bincount(v, weights=proxy), np.bincount(v, weights=insee))
                        for k, v in {"communes": li, "départements": np.unique(dep, return_inverse=True)[1], "2 km zones": zone}.items()}
    by = np.bincount(li, weights=proxy, minlength=len(cs)), np.bincount(li, weights=insee, minlength=len(cs))
    worst = sorted(range(len(cs)), key=lambda i: by[0][i] - by[1][i])[:8]
    res["most short"] = [(cs[i][1], round(by[0][i]), round(by[1][i])) for i in worst]
    for k, v in res.items():
        print(k, json.dumps(v, default=lambda o: round(float(o), 3)))
    with open(os.path.join(ROOT, "..", "artifacts", "trainworld", "T-289-jobs.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float, ensure_ascii=False)


if __name__ == "__main__":
    main()
