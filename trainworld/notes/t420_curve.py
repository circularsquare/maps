"""T-420: Greater Tokyo's commute distance curve fitted on the 2020 census's commuting table
(municipality of residence to municipality of work, designated cities by ward, both ends inside
the four prefectures, at-home workers out), against Paris's, London's, Toronto's, the US curve
and tier 0's fallback. notes/t289_curve.py's method.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t420_curve.py [city] [task id for the output name]

Makes the pack's tier 0 signals first if they are not cached, so the build that follows reuses
them. Prints the census's at-home share too (the config's wfh starts from it).
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
import japan  # noqa: E402
import tier0  # noqa: E402

CITY = sys.argv[1] if len(sys.argv) > 1 else "tyo"
TASK = sys.argv[2] if len(sys.argv) > 2 else "T-426"
OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", f"{TASK}-{CITY}-curve.json")
BINS = [0, 1.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 1e9]
NAMED = {"Tokyo stage 1 d^-2.2 exp(-d/80)": (2.2, 80.0),
         "Paris d^-1.3 exp(-d/20)": (1.3, 20.0), "London d^-1.85 exp(-d/30)": (1.85, 30.0),
         "Toronto d^-1.2 exp(-d/20)": (1.2, 20.0), "US d^-0.8 exp(-d/30)": (0.8, 30.0)}


def miss(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    return 0.5 * float(np.abs(a / a.sum() - b_ / b_.sum()).sum())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = b.CITIES[CITY]
    prefs, nb = cfg["prefectures"], cfg["neighbours"]
    boundary = tier0.build_boundary(CITY, cfg)
    cache = os.path.join(b.WORK, f"{CITY}.tier0_signals.npz")
    mask = None if os.path.exists(cache) else tier0.build_water(cfg, boundary)[0]
    sig = tier0.signals(CITY, cfg, boundary, mask)
    lon0, lat0 = cfg["origin"]
    sx = b.R_EARTH * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0))
    sy = b.R_EARTH * np.radians(sig["lat"] - lat0)

    ms = japan.municipalities(prefs)
    units = [(m[0], m[3]) for m in ms]
    nl = len(ms)
    ci = {m[0]: i for i, m in enumerate(ms)}
    li, near = england.cell_units(sig["lat"], sig["lon"], units)
    pop, _ = japan.mesh_population(CITY, boundary, sig, prefs, nb)
    jobs, _ = japan.census_jobs(CITY, boundary, sig, tier0.jobs_from(sig, 1.0), prefs, nb)
    _, home_share = japan.workers_at_work(prefs)
    print(f"census: {home_share:.2%} of workers by place of work work at home", flush=True)
    wfh = cfg.get("wfh", 0.0)
    work_c = jobs * (1 - wfh)
    home_c = pop * work_c.sum() / pop.sum()

    w = np.bincount(li[li >= 0], weights=pop[li >= 0] + 1e-6, minlength=nl)
    cx = np.bincount(li[li >= 0], weights=(sx * (pop + 1e-6))[li >= 0], minlength=nl) / np.maximum(w, 1e-12)
    cy = np.bincount(li[li >= 0], weights=(sy * (pop + 1e-6))[li >= 0], minlength=nl) / np.maximum(w, 1e-12)
    for u, p in near.items():
        cx[u], cy[u] = sx[p], sy[p]
    li = np.where(li < 0, 0, li)

    fl = japan.flows(prefs)
    fl = fl[fl.h.isin(ci) & fl.w.isin(ci)]
    allx, ally = np.concatenate([sx, cx]), np.concatenate([sy, cy])
    zone, gx, gy = tier0.zones_of(allx, ally)
    zs, zm = zone[:len(sx)], zone[len(sx):]
    nz = len(gx)
    d = tier0.zone_distances(gx, gy)
    h_i, w_i = fl.h.map(ci).to_numpy(), fl.w.map(ci).to_numpy()
    n = fl.n.to_numpy(float)
    print(f"census 2020 flows: {nl:,} municipalities, {n.sum():,.0f} workers with both ends inside, {nz:,} zones", flush=True)
    Cz = np.zeros((nz, nz))
    np.add.at(Cz, (zm[h_i], zm[w_i]), n)
    Cl = np.zeros((nl, nl))
    np.add.at(Cl, (h_i, w_i), n)

    def hist(T):
        return np.histogram(d.ravel(), bins=BINS, weights=T.ravel().astype(np.float64))[0]
    ch = hist(Cz)
    cmean, cmed = tier0.trip_stats(Cz, d)
    print(f"census: mean {cmean:.2f} km, median {cmed:.2f} km (zone distances); own municipality {np.trace(Cl) / Cl.sum():.1%}", flush=True)

    margins = {"census 2020": (Cz.sum(1), Cz.sum(0)),
               "pack (mesh pop, census jobs)": (np.bincount(zs, weights=home_c, minlength=nz), np.bincount(zs, weights=work_c, minlength=nz))}
    Pm = np.zeros((nz, nl)); Qm = np.zeros((nz, nl))
    np.add.at(Pm, (zm[h_i], h_i), n); np.add.at(Qm, (zm[w_i], w_i), n)
    Pp = np.zeros((nz, nl)); Qp = np.zeros((nz, nl))
    np.add.at(Pp, (zs, li), home_c); np.add.at(Qp, (zs, li), work_c)
    shares = {"census 2020": (Pm, Qm), "pack (mesh pop, census jobs)": (Pp, Qp)}

    res = {"census": dict(mean_km=cmean, median_km=cmed, own_municipality=float(np.trace(Cl) / Cl.sum()), commuters=float(n.sum()),
                          home_share=home_share, hist=(ch / ch.sum()).tolist(), bins=BINS)}
    for mname, (H, J) in margins.items():
        P, Q = shares[mname]
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)

        def score(pw, km):
            T = tier0.gravity(H, J, d, pw, km)
            C = P.T @ T.astype(np.float64) @ Q
            mean, med = tier0.trip_stats(T, d)
            return dict(pow=pw, km=km, municipality=miss(C, Cl), hist=miss(hist(T), ch), mean_km=mean, median_km=med,
                        own_municipality=float(np.trace(C) / C.sum()))
        rows = [score(pw, km) for pw in (0.6, 1.0, 1.4, 1.8, 2.2) for km in (5.0, 10.0, 20.0, 40.0, 100.0)]
        best = min(rows, key=lambda r: r["municipality"])
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
        rows.sort(key=lambda r: r["municipality"])
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
