"""T-185: shared pieces for the reverse-commuting diagnosis. Read-only on packs and LODES.

City185 extends T-162's City (notes/t162_fit.py: the game's zones, the numpy Furness solve and
fit_decay's scores) with:

- zone classes by job density: zones ranked by commuting jobs per 2 km zone; the densest zones
  holding the first 25% of the pack's jobs are "centre", the next 25% "middle", the rest
  (half the jobs) "suburb". Relative, so it works the same in New York and New Haven, and it needs
  only a jobs layer (tier 1 jobs, or tier 0's GHSL proxy), which every city has.
- LODES 8 OD 2023 as a sparse zone-pair list, cached in the scratchpad (not beside the data).
- evaluate(T): class table (share of each home class's commuters by work class), mean trip length
  per home class, the same split by same-state / other-state work, plus T-162's scores.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t162_fit as tf  # noqa: E402
from t162_fit import bc, fd  # noqa: E402

CACHE = os.environ.get("T185_CACHE", os.path.join(os.environ.get("TEMP", "."), "claude", "t185_cache"))
PACK_K = {"wdc": 0.49}  # shipped border factors (T-162, T-181); 0.23 elsewhere
BORDER_US = 0.23
os.makedirs(CACHE, exist_ok=True)
CITIES = ["nyc", "phl", "bos", "wdc", "bal", "pvd", "nhv", "hfd", "spf"]
CLASSES = ["centre", "middle", "suburb"]
CUTS = (0.25, 0.50)


class City185(tf.City):
    def observe(self):
        p = os.path.join(CACHE, f"{self.name}.obs.npz")
        if os.path.exists(p):
            z = np.load(p)
            self.oh, self.ow, self.on = z["h"], z["w"], z["n"]
            self.obs_C, self.obs_tl, self.obs_cross = z["C"], z["tl"], float(z["cross"])
        else:
            od = fd.lodes_od(self.name)
            bi = {g: i for i, g in enumerate(self.blocks["geoid"])}
            hi, wi = od["h_geocode"].map(bi), od["w_geocode"].map(bi)
            miss = hi.isna() | wi.isna()
            od, hi, wi = od[~miss], hi[~miss].to_numpy(np.int64), wi[~miss].to_numpy(np.int64)
            n = od["S000"].to_numpy(np.float64)
            self.obs_cross = float(n[self.bstate[hi] != self.bstate[wi]].sum() / n.sum())
            nc = len(self.counties)
            C = np.zeros((nc, nc))
            np.add.at(C, (self.bcounty[hi], self.bcounty[wi]), n)
            self.obs_C = C / C.sum()
            zh, zw = self.bzone[hi], self.bzone[wi]
            ok = (zh >= 0) & (zw >= 0)
            nz = len(self.pop)
            key, inv = np.unique(zh[ok] * nz + zw[ok], return_inverse=True)
            self.on = np.bincount(inv, weights=n[ok])
            self.oh, self.ow = key // nz, key % nz
            tl = np.bincount(self.band[self.oh, self.ow], weights=self.on, minlength=int(tf.MAX_KM / tf.BAND_KM))
            self.obs_tl = tl / tl.sum()
            np.savez(p, h=self.oh, w=self.ow, n=self.on, C=self.obs_C, tl=self.obs_tl, cross=self.obs_cross)
        self.obs_mean = float((self.d[self.oh, self.ow] * self.on).sum() / self.on.sum())
        self.classify()

    def classify(self, cuts=CUTS):
        order = np.argsort(-self.jobs, kind="stable")
        cum = np.cumsum(self.jobs[order]) / self.jobs.sum()
        before = cum - self.jobs[order] / self.jobs.sum()  # share of jobs in denser zones
        cls = np.full(len(self.jobs), 2, np.int64)
        cls[order[before < cuts[1]]] = 1
        cls[order[before < cuts[0]]] = 0
        self.cls = cls
        self.onehot = np.zeros((len(cls), 3), np.float32)
        self.onehot[np.arange(len(cls)), cls] = 1
        self.density_cut = [float(self.jobs[order][np.searchsorted(before, c) - 1]) / 4.0 for c in cuts]  # jobs/km2 of the last zone in

    # ---- relative yardstick (Anita, 2026-10-12): no absolute density classes
    RATIO_EDGES = (0.5, 0.8, 1.25, 2.0)  # local jobs per resident commuter, within RADIUS_KM
    RADIUS_KM = 5.0
    S_EDGES = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6)  # intervening jobs (nearer to home than work)

    def rel_prepare(self):
        if getattr(self, "rbin", None) is not None:
            return
        o = self.pop * self.jobs.sum() / self.pop.sum()
        near = (self.d <= self.RADIUS_KM).astype(np.float32)
        self.ratio = (near @ self.jobs.astype(np.float32)) / np.maximum(near @ o.astype(np.float32), 1.0)
        del near
        self.rbin = np.digitize(self.ratio, self.RATIO_EDGES).astype(np.int64)  # 0..4
        s = self.opportunities()
        sb = np.digitize(s, self.S_EDGES).astype(np.int8)
        sb[np.eye(len(self.jobs), dtype=bool)] = 0
        self.sbin = sb  # 0..8
        self.downhill = (self.ratio[None, :] < self.ratio[:, None])

    def rel(self, T):
        """per origin ratio bin: commuters, mean km, share downhill (to a zone with a lower local
        job ratio than home: the relative reverse commute); joint (ratio bin x opportunity bin)."""
        self.rel_prepare()
        nb, ns = len(self.RATIO_EDGES) + 1, len(self.S_EDGES) + 1
        oh = np.zeros((len(self.rbin), nb), np.float32); oh[np.arange(len(self.rbin)), self.rbin] = 1
        tot = oh.T.astype(np.float64) @ T.sum(1, dtype=np.float64)
        km = oh.T.astype(np.float64) @ np.einsum("ij,ij->i", T, self.d, dtype=np.float64)
        dh = oh.T.astype(np.float64) @ np.where(self.downhill, T, 0).sum(1, dtype=np.float64)
        J = np.zeros((nb, ns))
        for b in range(ns):
            J[:, b] = oh.T.astype(np.float64) @ np.where(self.sbin == b, T, 0).sum(1, dtype=np.float64)
        return {"n": tot.tolist(), "km": (km / np.maximum(tot, 1e-9)).tolist(),
                "down": (dh / np.maximum(tot, 1e-9)).tolist(), "J": J.tolist()}

    def rel_obs(self):
        self.rel_prepare()
        nb, ns = len(self.RATIO_EDGES) + 1, len(self.S_EDGES) + 1
        h, w, n = self.oh, self.ow, self.on
        rb = self.rbin[h]
        tot = np.bincount(rb, weights=n, minlength=nb)
        km = np.bincount(rb, weights=n * self.d[h, w], minlength=nb)
        dh = np.bincount(rb, weights=n * self.downhill[h, w], minlength=nb)
        J = np.zeros((nb, ns)); np.add.at(J, (rb, self.sbin[h, w]), n)
        return {"n": tot.tolist(), "km": (km / np.maximum(tot, 1e-9)).tolist(),
                "down": (dh / np.maximum(tot, 1e-9)).tolist(), "J": J.tolist()}

    def rel_off(self, r):
        """trips misplaced over (home ratio bin x opportunity bin), shares within each home bin
        weighted by LODES's home-bin sizes (so margins do not count)."""
        o = self.rel_o if getattr(self, "rel_o", None) else self.rel_obs()
        self.rel_o = o
        Jm, Jo = np.array(r["J"]), np.array(o["J"])
        wm = Jm / np.maximum(Jm.sum(1, keepdims=True), 1e-9)
        wo = Jo / np.maximum(Jo.sum(1, keepdims=True), 1e-9)
        hw = Jo.sum(1) / Jo.sum()
        return 0.5 * float((np.abs(wm - wo).sum(1) * hw).sum())

    # ---- the observed (LODES) tables
    def obs_tables(self):
        h, w, n = self.oh, self.ow, self.on
        M = np.zeros((3, 3)); np.add.at(M, (self.cls[h], self.cls[w]), n)
        L = np.zeros(3); np.add.at(L, self.cls[h], n * self.d[h, w])
        o = self.zr[h] != self.zr[w]
        X = np.zeros((3, 3)); np.add.at(X, (self.cls[h][o], self.cls[w][o]), n[o])
        return M, L, X

    def model_tables(self, T):
        oh = self.onehot
        M = (oh.T @ T @ oh).astype(np.float64)
        Ld = np.einsum("ij,ij->i", T, self.d, dtype=np.float64)
        L = oh.T.astype(np.float64) @ Ld
        if self.zr.max() > 0:
            X = (oh.T @ (T * self.other) @ oh).astype(np.float64)
        else:
            X = np.zeros((3, 3))
        return M, L, X

    def evaluate(self, T, margins=True):
        r = self.score(T)
        r.pop("C")
        M, L, X = self.model_tables(T)
        oM, oL, oX = self.obs_tables()
        r["cls"] = M.tolist()
        r["cls_len"] = (L / M.sum(1)).tolist()
        r["cls_cross"] = X.tolist()
        r["class_off"] = 0.5 * float(np.abs(M / M.sum() - oM / oM.sum()).sum())
        # share of each home class's commuters going to each work class, model / LODES
        sm, so = M / M.sum(1, keepdims=True), oM / oM.sum(1, keepdims=True)
        r["ratio"] = (sm / so).tolist()
        if margins:
            arr = T.sum(0, dtype=np.float64)
            r["job_margin_off"] = 0.5 * float(np.abs(arr - self.jobs).sum() / self.jobs.sum())
        r["rel"] = self.rel(T)
        r["rel_off"] = self.rel_off(r["rel"])
        return r

    def lodes_summary(self):
        M, L, X = self.obs_tables()
        return {"cls": M.tolist(), "cls_len": (L / M.sum(1)).tolist(), "cls_cross": X.tolist(),
                "cross": self.obs_cross, "mean": self.obs_mean, "own": float(np.trace(self.obs_C))}

    # ---- solvers beyond T-162's
    def furness(self, F, iters=400, tol=1e-4, alpha=1.0):
        """Doubly constrained on F (modified in place into T). alpha < 1: the work end only partly
        constrained: after the full solve, destination factors B^alpha, then rows rebalanced."""
        o = self.pop * self.jobs.sum() / self.pop.sum()
        dj = self.jobs
        has = o > 0
        col = dj.astype(np.float32)
        row = None
        for it in range(iters):
            s = (F @ col).astype(np.float64)
            if it > 0 and np.max(np.abs(row[has] * s[has] / o[has] - 1)) < tol:
                break
            row = np.where(has, o / np.maximum(s, 1e-30), 0.0).astype(np.float32)
            t = (row @ F).astype(np.float64)
            col = np.where(dj > 0, dj / np.maximum(t, 1e-30), 0.0).astype(np.float32)
        if alpha != 1.0:
            # B_j = col_j / jobs_j (attraction per job); keep jobs * B^alpha
            B = np.where(dj > 0, col / np.maximum(dj, 1e-30), 0.0)
            B = B / np.exp(np.average(np.log(np.maximum(B[dj > 0], 1e-30)), weights=dj[dj > 0]))
            col = (dj * np.where(dj > 0, B, 0.0) ** alpha).astype(np.float32)
            s = (F @ col).astype(np.float64)
            row = np.where(has, o / np.maximum(s, 1e-30), 0.0).astype(np.float32)
        F *= row[:, None]
        F *= col[None, :]
        return F

    def gravity_F(self, pw=1.0, km=40.0, k=1.0, km_i=None):
        """d^-pw exp(-d/km) (km per origin zone if km_i given) x k across regions."""
        if km_i is None:
            F = np.exp(-self.d / np.float32(km))
        else:
            F = np.exp(-self.d / km_i.astype(np.float32)[:, None])
        F *= self.d ** np.float32(-pw)
        if k != 1.0:
            F[self.other] *= np.float32(k)
        return F

    def opportunities(self):
        """s_ij: jobs in zones nearer to i than j (ties count half), excluding i and j. float32."""
        if getattr(self, "_s", None) is not None:
            return self._s
        nz = len(self.jobs)
        S = np.empty((nz, nz), np.float32)
        n = self.jobs
        for i in range(nz):
            di = self.d[i]
            o = np.argsort(di, kind="stable")
            ds = di[o]
            c = np.concatenate([[0.0], np.cumsum(n[o])])
            lt = c[np.searchsorted(ds, di, side="left")]
            le = c[np.searchsorted(ds, di, side="right")]
            s = 0.5 * (lt + le - n) - n[i]
            s[i] = 0.0
            S[i] = np.maximum(s, 0.0)
        self._s = S
        return S


def load(name):
    t0 = time.time()
    c = City185(name)
    print(f"  {name} ready, {time.time() - t0:.0f} s; centre {int((c.cls == 0).sum())} zones, "
          f"middle {int((c.cls == 1).sum())}, suburb {int((c.cls == 2).sum())}", flush=True)
    return c
