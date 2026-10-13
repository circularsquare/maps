"""T-162: fit the work gravity's decay jointly with the region border factor k against LODES OD,
on the New York and Philadelphia packs with their zones split by region (as shipped). Read-only
on the packs; LODES and ACS are the fit target and checks only, never a runtime input.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t162_fit.py [--grid] [--final POW KM K]

Model: the game's doubly constrained gravity on the pack's zones (commuters at both ends),
f(d) = d^-pow exp(-d / km), x k when the home and work zones are in different regions (US states),
solved here in numpy (Furness). Observed: LODES 8 OD 2023 S000, home and work blocks both in the
pack's counties, each block in the zone of its internal point and its state.

--grid: for each (pow, km), k by bisection so the model's share of commutes between states equals
LODES's in each city; scored on trips misplaced by length (2 km zone-centre bands) plus the share
misplaced between county pairs (pipeline/fit_decay.py's objective). --final scores one setting.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")

import json
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as bc  # noqa: E402
import fit_decay as fd  # noqa: E402
from read_pack import load_pack  # noqa: E402

BAND_KM, MAX_KM = 2.0, 120.0


class City:
    def __init__(self, city, packs=bc.PACKS):
        t0 = time.time()
        header, a = load_pack(city, packs)
        g = header["gravity"]
        self.name = city
        zm = g["zone_m"]
        x0, y0 = g["grid_origin_m"]
        w, h = g["grid"]
        gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
        nz = len(gx)
        zr = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(nz, np.int64)
        ids = header.get("regions", {}).get("ids", ["?"])
        self.zr = zr
        zone = a["zone"].astype(np.int64)
        self.pop = np.bincount(zone, weights=a["commute_home"].astype(np.float64), minlength=nz)
        self.jobs = np.bincount(zone, weights=a["commute_work"].astype(np.float64), minlength=nz)
        zk = zm / 1000.0
        dx = (gx[:, None] - gx[None, :]).astype(np.float32)
        dy = (gy[:, None] - gy[None, :]).astype(np.float32)
        self.d = np.float32(zk) * np.sqrt(dx * dx + dy * dy)
        del dx, dy
        self.d[self.d == 0] = np.float32(0.52 * zk)
        self.other = zr[:, None] != zr[None, :]
        self.band = np.minimum(self.d / BAND_KM, MAX_KM / BAND_KM - 1).astype(np.int16)
        # blocks -> zones: square of the internal point, the block's own state where that square has it
        b = pd.read_csv(os.path.join(bc.WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str},
                        usecols=["geoid", "lat", "lon", "pop", "jobs"])
        lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
        bx, by = fd.local_xy(b["lat"].to_numpy(), b["lon"].to_numpy(), lon0, lat0)
        bgx = np.floor((bx - x0) / zm).astype(np.int64)
        bgy = np.floor((by - y0) / zm).astype(np.int64)
        rid = {c: i for i, c in enumerate(ids)}
        breg = b["geoid"].str[:2].map(rid).fillna(-1).astype(np.int64).to_numpy()
        lookup = {}
        anyz = {}
        for i in range(nz):
            lookup[(gx[i] * h + gy[i], zr[i])] = i
            anyz.setdefault(gx[i] * h + gy[i], i)
        sq = bgx * h + bgy
        inside = (bgx >= 0) & (bgx < w) & (bgy >= 0) & (bgy < h)
        bzone = np.array([lookup.get((s, r), anyz.get(s, -1)) if ok else -1 for s, r, ok in zip(sq, breg, inside)], np.int64)
        self.blocks = b
        self.bzone = bzone
        self.bstate = b["geoid"].str[:2].to_numpy()
        counties = sorted(bc.CITIES[city]["counties"])
        self.counties = counties
        cidx = {c: i for i, c in enumerate(counties)}
        bcounty = b["geoid"].str[:5].map(cidx).to_numpy()
        self.bcounty = bcounty
        nc = len(counties)
        ok = bzone >= 0
        P = np.zeros((nz, nc)); Q = np.zeros((nz, nc))
        np.add.at(P, (bzone[ok], bcounty[ok]), b["pop"].to_numpy()[ok])
        np.add.at(Q, (bzone[ok], bcounty[ok]), b["jobs"].to_numpy()[ok])
        self.P = (P / np.maximum(P.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        self.Q = (Q / np.maximum(Q.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        print(f"{city}: {nz:,} zones ({len(set(gx * h + gy)):,} squares), regions {ids}; "
              f"{int((~ok).sum()):,} blocks outside every zone; {time.time() - t0:.0f} s", flush=True)
        self.observe()

    def observe(self):
        od = fd.lodes_od(self.name)
        bi = {g: i for i, g in enumerate(self.blocks["geoid"])}
        hi = od["h_geocode"].map(bi)
        wi = od["w_geocode"].map(bi)
        miss = hi.isna() | wi.isna()
        od, hi, wi = od[~miss], hi[~miss].to_numpy(np.int64), wi[~miss].to_numpy(np.int64)
        n = od["S000"].to_numpy(np.float64)
        zh, zw = self.bzone[hi], self.bzone[wi]
        ok = (zh >= 0) & (zw >= 0)
        self.obs_cross = float(n[self.bstate[hi] != self.bstate[wi]].sum() / n.sum())
        S = pd.DataFrame({"h": self.bstate[hi], "w": self.bstate[wi], "n": n}).pivot_table(index="h", columns="w", values="n", aggfunc="sum", fill_value=0)
        self.obs_states = (S.div(S.sum(1), axis=0)).round(3)
        tl = np.bincount(self.band[zh[ok], zw[ok]], weights=n[ok], minlength=int(MAX_KM / BAND_KM))
        self.obs_tl = tl / tl.sum()
        nc = len(self.counties)
        C = np.zeros((nc, nc))
        np.add.at(C, (self.bcounty[hi], self.bcounty[wi]), n)
        self.obs_C = C / C.sum()
        self.obs_mean = float((self.d[zh[ok], zw[ok]] * n[ok]).sum() / n[ok].sum())
        print(f"  LODES: {n.sum():,.0f} jobs, between states {self.obs_cross:.2%}, own county {np.trace(self.obs_C):.1%}, "
              f"mean zone distance {self.obs_mean:.2f} km", flush=True)

    def solve(self, pw, km, k, iters=300, tol=1e-4):
        F = np.exp(-self.d / np.float32(km)) * self.d ** np.float32(-pw)
        F[self.other] *= np.float32(k)
        o = self.pop * self.jobs.sum() / self.pop.sum()
        dj = self.jobs
        col = dj.astype(np.float32)
        has = o > 0
        for it in range(iters):
            s = (F @ col).astype(np.float64)
            if it > 0 and np.max(np.abs(row[has] * s[has] / o[has] - 1)) < tol:
                break
            row = np.where(has, o / np.maximum(s, 1e-30), 0.0).astype(np.float32)
            t = (row @ F).astype(np.float64)
            col = np.where(dj > 0, dj / np.maximum(t, 1e-30), 0.0).astype(np.float32)
        F *= row[:, None]
        F *= col[None, :]
        return F  # T, float32

    def score(self, T):
        tot = float(T.sum(dtype=np.float64))
        cross = float(T[self.other].sum(dtype=np.float64)) / tot
        tl = np.bincount(self.band.ravel(), weights=T.ravel(), minlength=int(MAX_KM / BAND_KM))
        tl = tl / tl.sum()
        C = (self.P.T @ T @ self.Q).astype(np.float64)
        C /= C.sum()
        mean = float((T * self.d).sum(dtype=np.float64)) / tot
        return {"cross": cross, "tld_off": 0.5 * float(np.abs(tl - self.obs_tl).sum()),
                "county_off": 0.5 * float(np.abs(C - self.obs_C).sum()), "own": float(np.trace(C)),
                "mean": mean, "C": C}

    def fit_k(self, pw, km, steps=12):
        lo, hi = np.log(0.03), np.log(1.0)
        for _ in range(steps):
            m = 0.5 * (lo + hi)
            c = self.score(self.solve(pw, km, np.exp(m)))["cross"]
            lo, hi = (lo, m) if c > self.obs_cross else (m, hi)
        return float(np.exp(0.5 * (lo + hi)))


def line(c, pw, km, k, r):
    return (f"  {c.name} d^-{pw:g} exp(-d/{km:g}) k {k:.3f}: between states {r['cross']:.2%} (LODES {c.obs_cross:.2%}), "
            f"length misplaced {r['tld_off']:.1%}, county pairs misplaced {r['county_off']:.1%}, own county "
            f"{r['own']:.1%} (LODES {np.trace(c.obs_C):.1%}), mean {r['mean']:.2f} km (LODES {c.obs_mean:.2f})")


def main(args):
    out = os.path.join(bc.WORK, "t162_fit.json")
    if "--final" in args:
        # --final POW KM K [city ...]: score k = 1, K and the city's own k at that decay
        i = args.index("--final")
        pw, km, k = (float(v) for v in args[i + 1:i + 4])
        names = args[i + 4:] or ["nyc", "phl"]
        res = {}
        for name in names:
            c = City(name)
            own = c.fit_k(pw, km)
            for kk in (1.0, k, own):
                r = c.score(c.solve(pw, km, kk))
                print(line(c, pw, km, kk, r) + (" (own fit)" if kk == own else ""), flush=True)
                res[f"{c.name} k={kk:.3f}"] = {x: v for x, v in r.items() if x != "C"}
            if c.zr.max() > 0:
                # by region pair: model k=1, model K, LODES (share of all in-pack commutes)
                for kk in (1.0, k):
                    T = c.solve(pw, km, kk)
                    nr = c.zr.max() + 1
                    R = np.zeros((nr, nr))
                    for a_ in range(nr):
                        for b_ in range(nr):
                            R[a_, b_] = T[np.ix_(c.zr == a_, c.zr == b_)].sum(dtype=np.float64)
                    print(f"    model k={kk:g} by region pair, share of the home region's trips:\n" + str(np.round(R / R.sum(1, keepdims=True), 3)))
                print(f"    LODES by state pair:\n{c.obs_states}")
        with open(os.path.join(bc.WORK, "t162_final.json"), "w") as f:
            json.dump(res, f, indent=1)
        return
    # --grid [POWS KMS], comma-separated (default the coarse grid); results append to OUT
    pows, kms = (0.8, 1.0, 1.2), (15.0, 20.0, 29.0, 40.0, 60.0)
    i = args.index("--grid") if "--grid" in args else -1
    if i >= 0 and len(args) > i + 2:
        pows = [float(v) for v in args[i + 1].split(",")]
        kms = [float(v) for v in args[i + 2].split(",")]
        out = os.path.join(bc.WORK, "t162_fit_fine.json")
    cities = [City("nyc"), City("phl")]
    rows = []
    t0 = time.time()
    for pw in pows:
        for km in kms:
            row = {"pow": pw, "km": km}
            for c in cities:
                k = c.fit_k(pw, km)
                r = c.score(c.solve(pw, km, k))
                print(line(c, pw, km, k, r) + f" ({time.time() - t0:.0f} s)", flush=True)
                row[c.name] = {"k": k, **{x: v for x, v in r.items() if x != "C"}}
            row["objective"] = sum(row[c.name]["tld_off"] + row[c.name]["county_off"] for c in cities)
            rows.append(row)
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    best = min(rows, key=lambda r: r["objective"])
    print(f"best: d^-{best['pow']} exp(-d/{best['km']}), k nyc {best['nyc']['k']:.3f}, phl {best['phl']['k']:.3f}, "
          f"objective {best['objective']:.4f}; -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
