"""T-289: Paris's commute distance curve fitted on INSEE's 2022 home to work flows (commune to
commune, Paris by arrondissement, both ends inside Ile-de-France), against London's, Toronto's,
the US curve and tier 0's fallback. The method of pipeline/curves/t360_curve.py.

    ..\\venv\\Scripts\\python.exe -u trainworld/pipeline/curves/t289_curve.py

Makes the pack's tier 0 signals first if they are not cached (boundary, water mask, GHSL, OSM).
The game's gravity (tier0.gravity) on the pack's 2 km zones, with two sets of margins: the flows'
own and the pack's (legal population by commune, INSEE workers by commune of work x (1 - work from
home)). A commune's flows go to the zone of its population-weighted centre (from the cells).
Scores: communes misplaced (half the L1 distance between shares), the distance histogram
misplaced, mean and median straight-line km.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
import france  # noqa: E402
import tier0  # noqa: E402

CITY = "par"
OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-289-curve.json")
BINS = [0, 1.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 1e9]
NAMED = {"London d^-1.85 exp(-d/30)": (1.85, 30.0), "Toronto d^-1.2 exp(-d/20)": (1.2, 20.0), "US d^-0.8 exp(-d/30)": (0.8, 30.0)}


def miss(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    return 0.5 * float(np.abs(a / a.sum() - b_ / b_.sum()).sum())


def main():
    cfg = b.CITIES[CITY]
    deps = cfg["departements"]
    boundary = tier0.build_boundary(CITY, cfg)
    cache = os.path.join(b.WORK, f"{CITY}.tier0_signals.npz")
    mask = None if os.path.exists(cache) else tier0.build_water(cfg, boundary)[0]
    sig = tier0.signals(CITY, cfg, boundary, mask)
    lon0, lat0 = cfg["origin"]
    sx = b.R_EARTH * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0))
    sy = b.R_EARTH * np.radians(sig["lat"] - lat0)

    cs = france.communes(deps)
    units = [(c[0], c[3]) for c in cs]
    nl = len(cs)
    ci = {c[0]: i for i, c in enumerate(cs)}
    li, near = england.cell_units(sig["lat"], sig["lon"], units)
    pop, _ = france.commune_population(CITY, boundary, sig, deps)
    jobs, _ = france.insee_jobs(CITY, boundary, sig, tier0.jobs_from(sig, 1.0), deps)
    wfh = cfg.get("wfh", 0.0)
    work_c = jobs * (1 - wfh)
    home_c = pop * work_c.sum() / pop.sum()

    # each commune's population-weighted centre, from its cells (a commune with none: its nearest cell)
    w = np.bincount(li[li >= 0], weights=pop[li >= 0] + 1e-6, minlength=nl)
    cx = np.bincount(li[li >= 0], weights=(sx * (pop + 1e-6))[li >= 0], minlength=nl) / np.maximum(w, 1e-12)
    cy = np.bincount(li[li >= 0], weights=(sy * (pop + 1e-6))[li >= 0], minlength=nl) / np.maximum(w, 1e-12)
    for u, p in near.items():
        cx[u], cy[u] = sx[p], sy[p]
    li = np.where(li < 0, 0, li)

    fl = france.flux()
    fl = fl[fl.h.isin(ci) & fl.w.isin(ci)]
    allx, ally = np.concatenate([sx, cx]), np.concatenate([sy, cy])
    zone, gx, gy = tier0.zones_of(allx, ally)
    zs, zm = zone[:len(sx)], zone[len(sx):]
    nz = len(gx)
    d = tier0.zone_distances(gx, gy)
    h_i, w_i = fl.h.map(ci).to_numpy(), fl.w.map(ci).to_numpy()
    n = fl.n.to_numpy(float)
    print(f"INSEE 2022 flows: {nl:,} communes, {n.sum():,.0f} workers with both ends inside, {nz:,} zones", flush=True)
    Cz = np.zeros((nz, nz))
    np.add.at(Cz, (zm[h_i], zm[w_i]), n)
    Cl = np.zeros((nl, nl))
    np.add.at(Cl, (h_i, w_i), n)

    def hist(T):
        return np.histogram(d.ravel(), bins=BINS, weights=T.ravel().astype(np.float64))[0]
    ch = hist(Cz)
    cmean, cmed = tier0.trip_stats(Cz, d)
    print(f"census: mean {cmean:.2f} km, median {cmed:.2f} km (zone distances); own commune {np.trace(Cl) / Cl.sum():.1%}", flush=True)

    margins = {"census 2022": (Cz.sum(1), Cz.sum(0)),
               "pack (legal pop, INSEE jobs)": (np.bincount(zs, weights=home_c, minlength=nz), np.bincount(zs, weights=work_c, minlength=nz))}
    Pm = np.zeros((nz, nl)); Qm = np.zeros((nz, nl))
    np.add.at(Pm, (zm[h_i], h_i), n); np.add.at(Qm, (zm[w_i], w_i), n)
    Pp = np.zeros((nz, nl)); Qp = np.zeros((nz, nl))
    np.add.at(Pp, (zs, li), home_c); np.add.at(Qp, (zs, li), work_c)
    shares = {"census 2022": (Pm, Qm), "pack (legal pop, INSEE jobs)": (Pp, Qp)}

    res = {"census": dict(mean_km=cmean, median_km=cmed, own_commune=float(np.trace(Cl) / Cl.sum()), commuters=float(n.sum()),
                          hist=(ch / ch.sum()).tolist(), bins=BINS)}
    for mname, (H, J) in margins.items():
        P, Q = shares[mname]
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)

        def score(pw, km):
            T = tier0.gravity(H, J, d, pw, km)
            C = P.T @ T.astype(np.float64) @ Q
            mean, med = tier0.trip_stats(T, d)
            return dict(pow=pw, km=km, commune=miss(C, Cl), hist=miss(hist(T), ch), mean_km=mean, median_km=med,
                        own_commune=float(np.trace(C) / C.sum()))
        rows = [score(pw, km) for pw in (0.6, 1.0, 1.4, 1.8, 2.2) for km in (5.0, 10.0, 20.0, 40.0, 100.0)]
        best = min(rows, key=lambda r: r["commune"])
        for pw in np.round(np.arange(best["pow"] - 0.3, best["pow"] + 0.31, 0.1), 2):
            for km in np.round(best["km"] * np.array([0.6, 0.8, 1.0, 1.25, 1.6]), 1):
                if pw > 0:
                    rows.append(score(float(pw), float(km)))
        named = {k: score(*v) for k, v in NAMED.items()}
        lo, hi = np.log(2.0), np.log(300.0)
        for _ in range(14):
            km = float(np.exp((lo + hi) / 2))
            if score(tier0.DEFAULT_CURVE_POW, km)["median_km"] < cmed:
                lo = np.log(km)
            else:
                hi = np.log(km)
        named[f"tier 0 fallback, power {tier0.DEFAULT_CURVE_POW} fitted to the census median"] = score(tier0.DEFAULT_CURVE_POW, round(float(np.exp((lo + hi) / 2)), 1))
        rows.sort(key=lambda r: r["commune"])
        res[mname] = dict(best=rows[:10], named=named, best_hist=sorted(rows, key=lambda r: r["hist"])[:5])
        print(f"--- margins: {mname}")
        for r in rows[:5]:
            print("  best " + json.dumps({k: round(v, 3) for k, v in r.items()}))
        for r in sorted(rows, key=lambda r: r["hist"])[:2]:
            print("  hist " + json.dumps({k: round(v, 3) for k, v in r.items()}))
        for k, r in named.items():
            print(f"  {k}: " + json.dumps({k2: round(v, 3) for k2, v in r.items()}))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
