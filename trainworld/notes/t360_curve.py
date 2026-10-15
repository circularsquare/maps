"""T-360: London's commute distance curve fitted on the 2011 census flows (WU01EW, MSOA to MSOA,
both ends inside the pack), against Toronto's borrowed d^-1.2 exp(-d/20 km) and the US curve.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t360_curve.py

The game's gravity (tier0.gravity) on the pack's 2 km zones, with two sets of margins: the census
flows' own (homes and workplaces of the 2011 flows, which isolates the curve) and the stage b pack's
(Census 2021 population, BRES jobs x (1 - work from home)). Census flows go to zones by their MSOA's
population-weighted centroid. Scores: districts misplaced (75 x 75 pairs, half the L1 distance
between shares), the distance histogram misplaced, mean and median straight-line km.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")

import json
import sys

import numpy as np
from shapely import make_valid, union_all
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
import tier0  # noqa: E402

OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-360-curve.json")
BINS = [0, 1.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 1e9]
NAMED = {"Toronto d^-1.2 exp(-d/20)": (1.2, 20.0), "US d^-0.8 exp(-d/30)": (0.8, 30.0)}


def miss(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    return 0.5 * float(np.abs(a / a.sum() - b_ / b_.sum()).sum())


def main():
    origin = b.CITIES["lon"]["origin"]
    sig = dict(np.load(os.path.join(b.WORK, "lon.tier0_signals.npz")))
    with open(england.raw("lon_itl3.geojson"), encoding="utf-8") as f:
        boundary = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    with open(england.raw("lad_dec2023_uk_bgc.geojson"), encoding="utf-8") as f:
        lads = [(ft["properties"]["LAD23NM"], make_valid(shape(ft["geometry"]))) for ft in json.load(f)["features"]]
    lads = [x for x in lads if boundary.contains(x[1].representative_point())]
    nl = len(lads)

    def xy(lat, lon):
        lon0, lat0 = origin
        return (b.R_EARTH * np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0)), b.R_EARTH * np.radians(np.asarray(lat) - lat0))

    # census flows, both ends inside
    pts = england.msoa11_points("lon", boundary)
    fl = england.wu01ew()
    fl = fl[fl.h.isin(pts) & fl.w.isin(pts)]
    codes = sorted(pts)
    ci = {c: i for i, c in enumerate(codes)}
    mll = np.array([pts[c] for c in codes])
    mx, my = xy(mll[:, 0], mll[:, 1])
    ml, _ = england.cell_units(mll[:, 0], mll[:, 1], lads)
    ml = np.where(ml < 0, 0, ml)
    print(f"WU01EW: {len(codes):,} MSOAs inside, {fl.n.sum():,.0f} commuters with both ends inside", flush=True)

    # one zone grid over the signal cells and the MSOA centroids
    sx, sy = xy(sig["lat"], sig["lon"])
    allx, ally = np.concatenate([sx, mx]), np.concatenate([sy, my])
    zone, gx, gy = tier0.zones_of(allx, ally)
    zs, zm = zone[:len(sx)], zone[len(sx):]
    nz = len(gx)
    d = tier0.zone_distances(gx, gy)
    h_i, w_i = fl.h.map(ci).to_numpy(), fl.w.map(ci).to_numpy()
    n = fl.n.to_numpy(float)
    Cz = np.zeros((nz, nz))
    np.add.at(Cz, (zm[h_i], zm[w_i]), n)
    Cl = np.zeros((nl, nl))
    np.add.at(Cl, (ml[h_i], ml[w_i]), n)

    def hist(T):
        w = T.ravel().astype(np.float64)
        return np.histogram(d.ravel(), bins=BINS, weights=w)[0]
    ch = hist(Cz)
    cmean, cmed = tier0.trip_stats(Cz, d)
    print(f"census: mean {cmean:.2f} km, median {cmed:.2f} km (zone distances); own district {np.trace(Cl) / Cl.sum():.1%}", flush=True)

    # margins
    li, _ = england.cell_units(sig["lat"], sig["lon"], lads)
    li = np.where(li < 0, 0, li)
    bres, _ = england.bres_jobs("lon", boundary, sig, tier0.jobs_from(sig, 1.0))
    work_c = bres * (1 - b.CITIES["lon"]["wfh"])
    home_c = sig["pop"] * work_c.sum() / sig["pop"].sum()
    margins = {"census 2011": (Cz.sum(1), Cz.sum(0), None),
               "pack (2021 pop, BRES 2024)": (np.bincount(zs, weights=home_c, minlength=nz), np.bincount(zs, weights=work_c, minlength=nz), (home_c, work_c))}
    # zone -> district shares for the pack margins (census margins: by MSOA)
    Pm = np.zeros((nz, nl)); Qm = np.zeros((nz, nl))
    np.add.at(Pm, (zm[h_i], ml[h_i]), n); np.add.at(Qm, (zm[w_i], ml[w_i]), n)
    Pp = np.zeros((nz, nl)); Qp = np.zeros((nz, nl))
    np.add.at(Pp, (zs, li), home_c); np.add.at(Qp, (zs, li), work_c)
    shares = {"census 2011": (Pm, Qm), "pack (2021 pop, BRES 2024)": (Pp, Qp)}

    res = {"census": dict(mean_km=cmean, median_km=cmed, own_district=float(np.trace(Cl) / Cl.sum()), commuters=float(n.sum()),
                          hist=(ch / ch.sum()).tolist(), bins=BINS)}
    for mname, (H, J, _) in margins.items():
        P, Q = shares[mname]
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)

        def score(pw, km):
            T = tier0.gravity(H, J, d, pw, km)
            C = P.T @ T.astype(np.float64) @ Q
            mean, med = tier0.trip_stats(T, d)
            return dict(pow=pw, km=km, district=miss(C, Cl), hist=miss(hist(T), ch), mean_km=mean, median_km=med,
                        own_district=float(np.trace(C) / C.sum()))
        rows = []
        for pw in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
            for km in (5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0):
                rows.append(score(pw, km))
        best = min(rows, key=lambda r: r["district"])
        # refine around the best
        for pw in np.round(np.arange(best["pow"] - 0.15, best["pow"] + 0.16, 0.05), 2):
            for km in np.round(best["km"] * np.array([0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]), 1):
                rows.append(score(float(pw), float(km)))
        named = {k: score(*v) for k, v in NAMED.items()}
        # tier 0's fallback: power 1, length fitted to the census median
        lo, hi = np.log(2.0), np.log(300.0)
        for _ in range(16):
            km = float(np.exp((lo + hi) / 2))
            if score(1.0, km)["median_km"] < cmed:
                lo = np.log(km)
            else:
                hi = np.log(km)
        named["tier 0 fallback, power 1 fitted to the census median"] = score(1.0, round(float(np.exp((lo + hi) / 2)), 1))
        rows.sort(key=lambda r: r["district"])
        res[mname] = dict(best=rows[:10], named=named, best_hist=sorted(rows, key=lambda r: r["hist"])[:5])
        print(f"--- margins: {mname}")
        for r in rows[:6]:
            print("  best " + json.dumps({k: round(v, 3) for k, v in r.items()}))
        for r in sorted(rows, key=lambda r: r["hist"])[:3]:
            print("  hist " + json.dumps({k: round(v, 3) for k, v in r.items()}))
        for k, r in named.items():
            print(f"  {k}: " + json.dumps({k2: round(v, 3) for k2, v in r.items()}))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
