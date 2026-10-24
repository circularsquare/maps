"""T-277: fit Toronto's work gravity curve, f(d) = d^-pow exp(-d / km), on the 2021 census
commuting flows between census subdivisions (StatCan 98-10-0459-01), as T-162 and T-198 fitted the
US curve on LODES. Read-only on the pack; the flows are the fit target only.

    ..\\venv\\Scripts\\python.exe trainworld/pipeline/curves/t277_fit.py [--final POW KM]

Model: the game's doubly constrained gravity on the pack's 2 km zones (commuters at both ends),
solved here in numpy (Furness). Each zone's homes and jobs are split between census subdivisions
(CSDs) by the dissemination areas whose points fall in it. Observed: census flows with both ends
in the pack's CSDs (usual place of work, May 2021). Score: the share of commutes misplaced between
CSD pairs (half the L1 distance of the two shares), and each CSD's share of its residents working in
their own CSD.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")

import json
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as bc  # noqa: E402
import canada  # noqa: E402
from read_pack import load_pack  # noqa: E402

CITY = "tor"
OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-277-fit.json")


class Fit:
    def __init__(self):
        t0 = time.time()
        header, a = load_pack(CITY)
        g = header["gravity"]
        zm = g["zone_m"]
        x0, y0 = g["grid_origin_m"]
        w, h = g["grid"]
        gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
        nz = len(gx)
        zone = a["zone"].astype(np.int64)
        self.home = np.bincount(zone, weights=a["commute_home"].astype(np.float64), minlength=nz)
        self.work = np.bincount(zone, weights=a["commute_work"].astype(np.float64), minlength=nz)
        zk = zm / 1000.0
        dx = (gx[:, None] - gx[None, :]).astype(np.float32)
        dy = (gy[:, None] - gy[None, :]).astype(np.float32)
        self.d = np.float32(zk) * np.sqrt(dx * dx + dy * dy)
        self.d[self.d == 0] = np.float32(g.get("intrazonal_km", 0.52 * zk))
        # DAs -> zones by their points; CSD shares of each zone's homes and jobs
        b = pd.read_csv(os.path.join(bc.WORK, f"{CITY}.blocks.csv.gz"), dtype={"geoid": str, "csd": str})
        lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
        bx = bc.R_EARTH * np.radians(b.lon - lon0) * np.cos(np.radians(lat0))
        by = bc.R_EARTH * np.radians(b.lat - lat0)
        sq = np.floor((bx - x0) / zm).astype(np.int64) * h + np.floor((by - y0) / zm).astype(np.int64)
        zs = {int(gx[i] * h + gy[i]): i for i in range(nz)}
        bz = np.array([zs.get(int(s), -1) for s in sq])
        self.csds = sorted(b.csd.unique())
        ci = {c: i for i, c in enumerate(self.csds)}
        bcsd = b.csd.map(ci).to_numpy()
        ok = bz >= 0
        nc = len(self.csds)
        P = np.zeros((nz, nc)); Q = np.zeros((nz, nc))
        np.add.at(P, (bz[ok], bcsd[ok]), b.home_c.to_numpy()[ok])
        np.add.at(Q, (bz[ok], bcsd[ok]), b.work_c.to_numpy()[ok])
        # a zone no DA point falls in: its nearest neighbour square's split (rare, edge zones)
        for M in (P, Q):
            empty = M.sum(1) == 0
            if empty.any():
                full = np.nonzero(~empty)[0]
                for i in np.nonzero(empty)[0]:
                    j = full[np.argmin((gx[full] - gx[i]) ** 2 + (gy[full] - gy[i]) ** 2)]
                    M[i] = M[j]
        self.P = (P / P.sum(1, keepdims=True)).astype(np.float32)
        self.Q = (Q / Q.sum(1, keepdims=True)).astype(np.float32)
        f = canada.csd_flows(bc.CITIES[CITY])
        f = f[f.h.isin(ci) & f.w.isin(ci)]
        O = np.zeros((nc, nc))
        np.add.at(O, (f.h.map(ci).to_numpy(), f.w.map(ci).to_numpy()), f.n.to_numpy(float))
        self.O = O / O.sum()
        self.names = {c: c for c in self.csds}
        print(f"{nz:,} zones, {nc} CSDs, {int((~ok).sum())} DAs outside every zone; census flows inside the pack "
              f"{O.sum():,.0f}, own CSD {np.trace(self.O):.1%}; {time.time() - t0:.0f} s", flush=True)

    def solve(self, pw, km, iters=300, tol=1e-4):
        F = np.exp(-self.d / np.float32(km)) * self.d ** np.float32(-pw)
        o = self.home * self.work.sum() / self.home.sum()
        dj = self.work
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
        return F

    def score(self, T):
        C = (self.P.T @ T @ self.Q).astype(np.float64)
        C /= C.sum()
        own_m = np.diag(C) / C.sum(1)
        own_o = np.diag(self.O) / self.O.sum(1)
        wts = self.O.sum(1)
        mean = float((T * self.d).sum(dtype=np.float64) / T.sum(dtype=np.float64))
        return {"csd_off": 0.5 * float(np.abs(C - self.O).sum()), "own": float(np.trace(C)),
                "own_err": float((np.abs(own_m - own_o) * wts).sum()), "mean_km": mean, "C": C}


def main(args):
    fit = Fit()
    res = []
    if "--final" in args:
        i = args.index("--final")
        grid = [(float(args[i + 1]), float(args[i + 2]))]
    else:
        grid = [(p, k) for p in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5) for k in (10, 15, 20, 30, 45, 70, 120)]
    for pw, km in grid:
        r = fit.score(fit.solve(pw, km))
        res.append({"pow": pw, "km": km, **{k: v for k, v in r.items() if k != "C"}})
        print(f"  d^-{pw:g} exp(-d/{km:g}): CSD pairs misplaced {r['csd_off']:.1%}, own CSD {r['own']:.1%} "
              f"(census {np.trace(fit.O):.1%}), own-CSD error weighted {r['own_err']:.1%}, mean {r['mean_km']:.1f} km", flush=True)
    best = min(res, key=lambda r: r["csd_off"])
    print(f"best: d^-{best['pow']:g} exp(-d/{best['km']:g}), {best['csd_off']:.1%} misplaced")
    if len(grid) == 1:
        r = fit.score(fit.solve(*grid[0]))
        C = r["C"]
        rows = []
        for i, c in enumerate(fit.csds):
            rows.append((c, fit.O[i].sum(), np.diag(fit.O)[i] / fit.O[i].sum(), C[i, i] / C[i].sum(),
                         fit.O[i, fit.csds.index("3520005")] / fit.O[i].sum(), C[i, fit.csds.index("3520005")] / C[i].sum()))
        t = pd.DataFrame(rows, columns=["csd", "share_of_commuters", "own_census", "own_model", "to_toronto_census", "to_toronto_model"])
        print(t.sort_values("share_of_commuters", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"results": res, "best": best}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
