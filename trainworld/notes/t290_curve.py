"""T-290: Berlin's commute distance curve fitted on the Pendlerrechnung der Länder's 2024 flows
between Gemeinden (Pendleratlas), against Paris's, London's, Toronto's and the US curve. The
method of notes/t289_curve.py, on the flows the atlas publishes: each Gemeinde's ten largest
out-flows and ten largest in-flows, and everyone working in their home Gemeinde.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t290_curve.py

Makes the pack's tier 0 signals first if they are not cached (boundary, water mask, GHSL, OSM),
so the build that follows reuses them. The game's gravity (tier0.gravity) on the pack's 2 km
zones with the pack's own margins (Zensus population, Pendlerrechnung jobs x (1 - work from home));
a Gemeinde's flows go to the zone of its population-weighted centre. Score: on the published
pairs only (both ends inside), half the L1 distance between model and published counts over the
published total (the model scaled up by 1 / (1 - work from home), since the flows count everyone);
also without Berlin's own 1.67M, which the margins all but fix.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
import germany  # noqa: E402
import tier0  # noqa: E402

CITY = "ber"
OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-290-curve.json")
NAMED = {"Paris d^-1.3 exp(-d/20)": (1.3, 20.0), "London d^-1.85 exp(-d/30)": (1.85, 30.0),
         "Toronto d^-1.2 exp(-d/20)": (1.2, 20.0), "US d^-0.8 exp(-d/30)": (0.8, 30.0)}


def published_pairs(ci, lands):
    """{(home index, work index): workers} for the published pairs with both ends inside."""
    v = germany.pendler("Verfl", lands)
    pairs = {}
    for h, w, n in list(zip(v.ARS, v.ARS_AO, v.AUSP_AO)) + list(zip(v.ARS_WO, v.ARS, v.EIP_WO)):
        n = pd.to_numeric(n, errors="coerce")
        if h in ci and w in ci and h != w and n == n:
            pairs[(ci[h], ci[w])] = float(n)
    ez = germany.pendler("EZ", lands)
    for a, iop in zip(ez.ARS, ez.IOP):
        if a in ci:
            pairs[(ci[a], ci[a])] = float(iop)
    return pairs


def main():
    cfg = b.CITIES[CITY]
    kreise = cfg["kreise"]
    boundary = tier0.build_boundary(CITY, cfg)
    cache = os.path.join(b.WORK, f"{CITY}.tier0_signals.npz")
    mask = None if os.path.exists(cache) else tier0.build_water(cfg, boundary)[0]
    sig = tier0.signals(CITY, cfg, boundary, mask)
    lon0, lat0 = cfg["origin"]
    sx = b.R_EARTH * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0))
    sy = b.R_EARTH * np.radians(sig["lat"] - lat0)

    gs = germany.gemeinden(CITY, kreise)
    units = [(g[0], g[2]) for g in gs]
    nl = len(gs)
    ci = {g[0]: i for i, g in enumerate(gs)}
    li, near = england.cell_units(sig["lat"], sig["lon"], units)
    pop, _ = germany.zensus_population(CITY, boundary, sig, kreise)
    jobs, _ = germany.pendler_jobs(CITY, boundary, sig, tier0.jobs_from(sig, 1.0), kreise)
    wfh = cfg.get("wfh", 0.0)
    work_c = jobs * (1 - wfh)
    home_c = pop * work_c.sum() / pop.sum()
    li = np.where(li < 0, 0, li)

    pairs = published_pairs(ci, sorted({k[:2] for k in kreise}))
    keys = np.array(list(pairs.keys()))
    obs = np.array(list(pairs.values()))
    berlin = ci["110000000000"]
    not_berlin_own = ~((keys[:, 0] == berlin) & (keys[:, 1] == berlin))
    print(f"published pairs inside: {len(obs):,} ({obs.sum():,.0f} workers; {obs[not_berlin_own].sum():,.0f} without Berlin's own)", flush=True)

    zone, gx, gy = tier0.zones_of(sx, sy)
    nz = len(gx)
    d = tier0.zone_distances(gx, gy)
    H = np.bincount(zone, weights=home_c, minlength=nz)
    J = np.bincount(zone, weights=work_c, minlength=nz)
    P = np.zeros((nz, nl)); Q = np.zeros((nz, nl))
    np.add.at(P, (zone, li), home_c); np.add.at(Q, (zone, li), work_c)
    P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
    print(f"{nz:,} zones, {nl} Gemeinden", flush=True)

    def score(pw, km):
        T = tier0.gravity(H, J, d, pw, km)
        C = (P.T @ T.astype(np.float64) @ Q) / (1 - wfh)
        m = C[keys[:, 0], keys[:, 1]]
        mean, med = tier0.trip_stats(T, d)
        r = dict(pow=pw, km=km, misplaced=0.5 * float(np.abs(m - obs).sum() / obs.sum()),
                 misplaced_ring=0.5 * float(np.abs(m - obs)[not_berlin_own].sum() / obs[not_berlin_own].sum()),
                 model_on_pairs=float(m.sum() / obs.sum()), berlin_own=float(C[berlin, berlin] / obs[~not_berlin_own].sum()),
                 mean_km=mean, median_km=med)
        print("  " + json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
        return r

    rows = [score(pw, km) for pw in (0.8, 1.2, 1.6, 2.0) for km in (10.0, 20.0, 40.0, 80.0)]
    best = min(rows, key=lambda r: r["misplaced"])
    for pw in np.round(best["pow"] + np.array([-0.2, 0.0, 0.2]), 2):
        for km in np.round(best["km"] * np.array([0.7, 1.0, 1.4]), 1):
            if pw > 0 and not any(abs(r["pow"] - pw) < 1e-9 and abs(r["km"] - km) < 1e-9 for r in rows):
                rows.append(score(float(pw), float(km)))
    print("--- named", flush=True)
    named = {k: score(*v) for k, v in NAMED.items()}
    rows.sort(key=lambda r: r["misplaced"])
    print("--- best", flush=True)
    for r in rows[:5]:
        print("  " + json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    with open(OUT, "w") as f:
        json.dump(dict(pairs=len(obs), workers=float(obs.sum()), best=rows[:10], named=named,
                       ring_best=sorted(rows, key=lambda r: r["misplaced_ring"])[:5]), f, indent=1, default=float)
    print("curve done", flush=True)


if __name__ == "__main__":
    main()
