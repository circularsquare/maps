"""T-006: fit the zone gravity's distance decay against LODES OD (the tier-2 check of SPEC 9).

    python pipeline/fit_decay.py nyc [--plot out.png] [--final A B]

Observed: LODES 8 OD 2023 (JT00, all jobs), home and work block both in the city's counties
(the states' main files plus the aux files for workers living in another state). Each block goes
to the 2 km zone holding its internal point, and distances are zone-centre distances, so both
sides are measured exactly as the game's gravity measures them (intrazonal 0.52 x side).

Model: the game's own doubly constrained gravity on the pack's zones (workers = population scaled
to jobs, jobs from the pack), solved here in numpy with a dense decay matrix for each candidate
decay, and compared on the trip-length distribution (2 km bands) and county-to-county flows.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_city as bc
from read_pack import load_pack

LODES_OD = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/od/{st}_od_{part}_JT00_{year}.csv.gz"
BAND_KM = 2.0
MAX_KM = 120.0


def lodes_od(city, year=2023):
    cfg = bc.CITIES[city]
    counties = set(cfg["counties"])
    out = []
    for fips in sorted({c[:2] for c in counties}):
        st = bc.STATES[fips]
        for part in ("main", "aux"):
            p = bc.fetch(LODES_OD.format(st=st, part=part, year=year))
            for ch in pd.read_csv(p, usecols=["w_geocode", "h_geocode", "S000"],
                                  dtype={"w_geocode": str, "h_geocode": str}, chunksize=2_000_000):
                ch = ch[ch["w_geocode"].str[:5].isin(counties) & ch["h_geocode"].str[:5].isin(counties)]
                out.append(ch)
    return pd.concat(out, ignore_index=True)


def local_xy(lat, lon, lon0, lat0):
    x = bc.R_EARTH * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = bc.R_EARTH * np.radians(lat - lat0)
    return x, y


class Zones:
    def __init__(self, header, a):
        g = header["gravity"]
        self.zm = g["zone_m"]
        self.x0, self.y0 = g["grid_origin_m"]
        self.w, self.h = g["grid"]
        self.gx = a["zone_gx"].astype(np.int64)
        self.gy = a["zone_gy"].astype(np.int64)
        self.n = len(self.gx)
        zone = a["zone"].astype(np.int64)
        hw = a["commute_home"] if "commute_home" in a else a["pop"]  # T-042: commuters only
        ww = a["commute_work"] if "commute_work" in a else a["jobs"]
        self.pop = np.bincount(zone, weights=hw.astype(np.float64), minlength=self.n)
        self.jobs = np.bincount(zone, weights=ww.astype(np.float64), minlength=self.n)
        self.lookup = -np.ones(self.w * self.h, np.int64)
        self.lookup[self.gx * self.h + self.gy] = np.arange(self.n)
        zk = self.zm / 1000.0
        dx = (self.gx[:, None] - self.gx[None, :]).astype(np.float32)
        dy = (self.gy[:, None] - self.gy[None, :]).astype(np.float32)
        self.d = zk * np.sqrt(dx * dx + dy * dy)
        self.d[self.d == 0] = 0.52 * zk
        self.intra = 0.52 * zk

    def of_xy(self, x, y):
        gx = np.floor((x - self.x0) / self.zm).astype(np.int64)
        gy = np.floor((y - self.y0) / self.zm).astype(np.int64)
        ok = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)
        z = -np.ones(len(x), np.int64)
        z[ok] = self.lookup[gx[ok] * self.h + gy[ok]]
        return z


def furness(f, o, dj, iters=300, tol=1e-3):
    """Doubly constrained: T = row_i col_j f_ij with row sums o and column sums dj (Furness, the
    same balancing as the sim crate's kernel::gravity; 1e-3 is plenty to compare shapes)."""
    col = dj.astype(np.float32)
    row = np.zeros(len(o), np.float32)
    has = o > 0
    e = np.inf
    for it in range(iters):
        s = (f @ col).astype(np.float64)
        if it > 0:
            e = np.max(np.abs(row[has] * s[has] / o[has] - 1))
            if e < tol:
                break
        row = np.where(has, o / np.maximum(s, 1e-30), 0.0).astype(np.float32)
        t = (row @ f).astype(np.float64)
        col = np.where(dj > 0, dj / np.maximum(t, 1e-30), 0.0).astype(np.float32)
    return row, col, it + 1, e


def bands(d, w):
    edges = np.arange(0, MAX_KM + BAND_KM, BAND_KM)
    h = np.histogram(np.minimum(d, MAX_KM - 1e-6), bins=edges, weights=w)[0]
    return h / h.sum()


def model(z, f):
    o = z.pop * z.jobs.sum() / z.pop.sum()
    row, col, it, err = furness(f, o, z.jobs)
    T = (row[:, None] * f) * col[None, :]
    return T, it, err


def decay(kind, d, p):
    if kind == "exp":
        return np.exp(-d / p[0])
    if kind == "gamma":  # d^-a exp(-d/b), the "combined" or Tanner function
        return d ** (-p[0]) * np.exp(-d / p[1])
    if kind == "power":
        return d ** (-p[0])
    raise ValueError(kind)


def summary(T, z, P, Q):
    w = T.ravel()
    tl = bands(z.d.ravel(), w)
    mean = float((T * z.d).sum() / T.sum())
    C = P.T @ T @ Q
    return tl, mean, C


def label(r):
    if r["kind"] == "exp":
        return f"exp(-d/{r['p'][0]:g} km)"
    return f"d^-{r['p'][0]:g} exp(-d/{r['p'][1]:g} km)"


def main(city, plot=None, final=None):
    t0 = time.time()
    header, a = load_pack(city)
    z = Zones(header, a)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    blocks = pd.read_csv(os.path.join(bc.WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str})
    bx, by = local_xy(blocks["lat"].to_numpy(), blocks["lon"].to_numpy(), lon0, lat0)
    bzone = z.of_xy(bx, by)
    counties = sorted(bc.CITIES[city]["counties"])
    cidx = {c: i for i, c in enumerate(counties)}
    bcounty = blocks["geoid"].str[:5].map(cidx).to_numpy()
    # zone -> county shares, homes by population, work by jobs (block internal points)
    nz, nc = z.n, len(counties)
    ok = bzone >= 0
    P = np.zeros((nz, nc)); Q = np.zeros((nz, nc))
    np.add.at(P, (bzone[ok], bcounty[ok]), blocks["pop"].to_numpy()[ok])
    np.add.at(Q, (bzone[ok], bcounty[ok]), blocks["jobs"].to_numpy()[ok])
    P /= np.maximum(P.sum(1, keepdims=True), 1e-9)
    Q /= np.maximum(Q.sum(1, keepdims=True), 1e-9)

    od = lodes_od(city)
    print(f"LODES OD in the county set: {len(od):,} block pairs, {od['S000'].sum():,} jobs "
          f"(the pack has {z.jobs.sum():,.0f} commuting jobs); {time.time() - t0:.0f} s", flush=True)
    bi = {g: i for i, g in enumerate(blocks["geoid"])}
    hi = od["h_geocode"].map(bi)
    wi = od["w_geocode"].map(bi)
    miss = hi.isna() | wi.isna()
    if miss.any():
        print(f"  {int(od['S000'][miss].sum()):,} jobs with a block not in the pack's table, dropped")
    od = od[~miss]
    hi, wi = hi[~miss].to_numpy(np.int64), wi[~miss].to_numpy(np.int64)
    n = od["S000"].to_numpy(np.float64)
    crow = np.hypot(bx[hi] - bx[wi], by[hi] - by[wi]) / 1000.0
    zh, zw = bzone[hi], bzone[wi]
    zok = (zh >= 0) & (zw >= 0)
    dz = z.d[zh[zok], zw[zok]]
    obs = {"crow_mean": float((crow * n).sum() / n.sum()), "crow_median": float(np.median(np.repeat(crow, n.astype(int)))),
           "zone_mean": float((dz * n[zok]).sum() / n[zok].sum()), "tl": bands(dz, n[zok]),
           "C": np.zeros((nc, nc))}
    np.add.at(obs["C"], (bcounty[hi], bcounty[wi]), n)
    within = {k: float(n[crow < k].sum() / n.sum()) for k in (5, 10, 20, 40)}
    print(f"LODES: mean crow-fly {obs['crow_mean']:.2f} km (median {obs['crow_median']:.1f}), zone-centre "
          f"{obs['zone_mean']:.2f} km; within 5/10/20/40 km "
          + ", ".join(f"{v:.1%}" for v in within.values()), flush=True)

    def score(kind, p):
        f = decay(kind, z.d, p).astype(np.float32)
        T, it, err = model(z, f)
        tl, mean, C = summary(T, z, P, Q)
        # trip-length fit: sum of absolute band differences / 2 (share of trips misplaced)
        tld = 0.5 * float(np.abs(tl - obs["tl"]).sum())
        co, cm = obs["C"] / obs["C"].sum(), C / C.sum()
        m = (co > 0) & (cm > 0)
        r_log = float(np.corrcoef(np.log(co[m]), np.log(cm[m]))[0, 1])
        cmis = 0.5 * float(np.abs(co - cm).sum())
        same = float(np.trace(cm)), float(np.trace(co))
        return {"kind": kind, "p": list(p), "mean": mean, "tld_off": tld, "county_r_log": r_log,
                "county_off": cmis, "own_county_model": same[0], "own_county_obs": same[1],
                "tl": tl, "C": cm, "iters": it, "err": err}

    results = []
    def run(kind, p):
        r = score(kind, p)
        results.append(r)
        print(f"  {kind} {', '.join(f'{v:g}' for v in p):>12}: mean {r['mean']:5.2f} km, trips misplaced by "
              f"length {r['tld_off']:.1%}, county pairs log r {r['county_r_log']:.3f}, county share misplaced "
              f"{r['county_off']:.1%}, own county {r['own_county_model']:.1%} vs {r['own_county_obs']:.1%} "
              f"({r['iters']} it, {time.time() - t0:.0f} s)", flush=True)
        if final:
            cum = np.cumsum(r["tl"])
            ocum = np.cumsum(obs["tl"])
            print("      within 4/10/20/40 km (zone-centre): model " + ", ".join(f"{cum[int(k / BAND_KM) - 1]:.1%}" for k in (4, 10, 20, 40))
                  + "; LODES " + ", ".join(f"{ocum[int(k / BAND_KM) - 1]:.1%}" for k in (4, 10, 20, 40)))
        return r

    base = run("exp", [9.0])
    if final:  # only the old and the chosen decay, for the record and the plot
        r = run("gamma", final)
        if plot:
            draw(plot, obs, base, None, r, counties)
        return
    for b in (6.0, 8.0, 10.0, 11.0, 12.0, 14.0):
        run("exp", [b])
    for a_ in (0.5, 0.7, 0.9, 1.1, 1.3):
        for b in (15.0, 20.0, 30.0, 45.0, 70.0):
            run("gamma", [a_, b])
    # refine each family by Nelder-Mead on the same objective (decay length on a log scale)
    from scipy.optimize import minimize

    def obj(kind):
        def f(v):
            p = [float(np.exp(v[0]))] if kind == "exp" else [float(v[0]), float(np.exp(v[1]))]
            if kind == "gamma" and not (0.0 <= p[0] <= 3.0):
                return 9.0
            r = run(kind, [round(x, 4) for x in p])
            return r["tld_off"] + r["county_off"]
        return f
    for kind in ("exp", "gamma"):
        rb = min((r for r in results if r["kind"] == kind), key=lambda r: r["tld_off"] + r["county_off"])
        x0 = [np.log(rb["p"][0])] if kind == "exp" else [rb["p"][0], np.log(rb["p"][1])]
        minimize(obj(kind), x0, method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-4, "maxfev": 40})
    best_exp = min((r for r in results if r["kind"] == "exp"), key=lambda r: r["tld_off"] + r["county_off"])
    best_all = min(results, key=lambda r: r["tld_off"] + r["county_off"])
    out = {"lodes": {k: v for k, v in obs.items() if k not in ("tl", "C")}, "within_km": within,
           "results": [{k: v for k, v in r.items() if k not in ("tl", "C")} for r in results],
           "base": base["p"], "best_exp": best_exp["p"], "best": [best_all["kind"], best_all["p"]]}
    with open(os.path.join(bc.WORK, f"{city}.fit_decay.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"best exp {best_exp['p']}, best overall {best_all['kind']} {best_all['p']}")
    if plot:
        draw(plot, obs, base, best_exp, best_all, counties)


def draw(path, obs, base, bexp, ball, counties):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), dpi=130, facecolor="white")
    x = np.arange(0, MAX_KM, BAND_KM) + BAND_KM / 2
    ax[0].plot(x, obs["tl"], color="#222222", lw=2, label=f"LODES OD 2023 (mean {obs['zone_mean']:.1f} km)")
    curves = [(base, "#3b7dd8"), (ball, "#d8613b")] + ([(bexp, "#888888")] if bexp and bexp["p"] != base["p"] else [])
    for r, c in curves:
        lab = f"{label(r)} (mean {r['mean']:.1f} km)"
        ax[0].plot(x, r["tl"], color=c, lw=1.3, label=lab)
    ax[0].set_xlabel("home to work, zone-centre km"); ax[0].set_ylabel("share of commutes per 2 km band")
    ax[0].legend(fontsize=8); ax[0].set_xlim(0, 80)
    ax[1].semilogy(x, obs["tl"], color="#222222", lw=2)
    for r, c in curves:
        ax[1].semilogy(x, r["tl"], color=c, lw=1.3)
    ax[1].set_xlabel("home to work, zone-centre km"); ax[1].set_title("same, log scale", fontsize=10)
    co = obs["C"] / obs["C"].sum()
    for r, c, mk in ((base, "#3b7dd8", "o"), (ball, "#d8613b", ".")):
        m = (co > 0) & (r["C"] > 0)
        ax[2].loglog(co[m], r["C"][m], mk, color=c, ms=3, label=f"{label(r)}: log r {r['county_r_log']:.3f}")
    lim = [1e-7, 0.2]
    ax[2].plot(lim, lim, color="#888888", lw=0.6)
    ax[2].set_xlabel("LODES share of commutes, county pair"); ax[2].set_ylabel("model share")
    ax[2].legend(fontsize=8); ax[2].set_title(f"{len(counties)}² county pairs", fontsize=10)
    fig.tight_layout(); fig.savefig(path, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    plot = final = None
    if "--plot" in args:
        i = args.index("--plot"); plot = args[i + 1]; del args[i:i + 2]
    if "--final" in args:  # --final A B: score only exp 9 km and d^-A exp(-d/B)
        i = args.index("--final"); final = [float(args[i + 1]), float(args[i + 2])]; del args[i:i + 3]
    main(args[0] if args else "nyc", plot, final)
