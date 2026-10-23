"""T-1000: rebuild a pack's zone commute matrix from its shipped gravity, cluster its settlements,
and re-solve it under alternative balancing rules. Read-only on packs."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import json
import numpy as np
import h3
from scipy import ndimage

PACKS = r"C:/Users/anita/projects/maps/trainworld/data/packs"
DT = {"u64": "<u8", "f32": "<f4", "u32": "<u4", "i32": "<i4", "f64": "<f8", "u16": "<u2", "u8": "u1"}


def load_pack(city):
    with open(os.path.join(PACKS, f"{city}.json"), encoding="utf-8") as f:
        header = json.load(f)
    with open(os.path.join(PACKS, f"{city}.bin"), "rb") as f:
        buf = f.read()
    arrays = {}
    for a in header["arrays"]:
        if a["dtype"] in DT:
            arrays[a["name"]] = np.frombuffer(buf, dtype=np.dtype(DT[a["dtype"]]), count=a["count"], offset=a["offset"])
    return header, arrays


class Pack:
    def __init__(self, city):
        self.city = city
        h, a = load_pack(city)
        self.h, self.a = h, a
        g = h["gravity"]
        self.g = g
        nz = self.nz = h["zones"]
        self.zone = a["zone"].astype(np.int64)
        self.gx = a["zone_gx"].astype(np.int64)
        self.gy = a["zone_gy"].astype(np.int64)
        self.side = a["zone_size"].astype(np.int64) if "zone_size" in a else np.ones(nz, np.int64)
        self.zr = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(nz, np.int64)
        hw = a["commute_home"] if "commute_home" in a else a["pop"]
        ww = a["commute_work"] if "commute_work" in a else a["jobs"]
        zpop = np.bincount(self.zone, weights=hw.astype(np.float64), minlength=nz)
        self.J = np.bincount(self.zone, weights=ww.astype(np.float64), minlength=nz)
        self.W = zpop * self.J.sum() / zpop.sum()
        self.pop = np.bincount(self.zone, weights=a["pop"].astype(np.float64), minlength=nz)
        self.row = a["zone_row"].astype(np.float64)
        self.col = a["zone_col"].astype(np.float64)
        self.joint = "zone_home" in a
        self.zk = g["zone_m"] / 1000.0
        self.bf = float(g.get("border_factor", 1.0))
        self.cx = (self.gx + self.side / 2.0) * self.zk
        self.cy = (self.gy + self.side / 2.0) * self.zk
        self._build()

    def _build(self):
        g, nz, zk = self.g, self.nz, self.zk
        a_pow = g.get("decay_pow", 0.0) if g["decay"] == "pow_exp" else 0.0
        gx, gy, side = self.gx, self.gy, self.side
        F = np.zeros((nz, nz))
        D = np.zeros((nz, nz), np.float32)
        subs = [(0, 0)] if side.max() == 1 else [(u, v) for u in (0, 1) for v in (0, 1)]
        wsub = {q: np.where(side == 2, 0.25, 1.0 if q == (0, 0) else 0.0) for q in subs}
        for s in range(0, nz, 400):
            e = min(nz, s + 400)
            dx = gx[s:e, None] - gx[None, :]
            dy = gy[s:e, None] - gy[None, :]
            f = 0.0
            for qi in subs:
                for qj in subs:
                    wq = wsub[qi][s:e, None] * wsub[qj][None, :]
                    if not wq.any():
                        continue
                    ex, ey = dx + qi[0] - qj[0], dy + qi[1] - qj[1]
                    dq = zk * np.sqrt(ex * ex + ey * ey)
                    dq[dq == 0] = g["intrazonal_km"]
                    dq32 = dq.astype(np.float32)
                    f = f + wq * (np.exp(-dq32 / np.float32(g["decay_km"])) * dq32 ** np.float32(-a_pow)).astype(np.float64)
            if len(subs) == 1:
                d = dq
            else:
                hx, hy = 2 * dx + side[s:e, None] - side[None, :], 2 * dy + side[s:e, None] - side[None, :]
                d = zk * 0.5 * np.sqrt(hx * hx + hy * hy)
                d = np.where(d == 0, g["intrazonal_km"] * side[s:e, None], d)
            other = self.zr[s:e, None] != self.zr[None, :]
            F[s:e] = f * np.where(other, self.bf, 1.0)
            D[s:e] = d
        self.F, self.D = F, D

    # ---- settlement clusters ----
    def clusters(self, gap_km=25.0, seed_min=100.0):
        """Zones whose commuters (homes + jobs) reach seed_min are seeds; seeds within gap_km of each
        other (square edges, by dilation of gap/2) form one cluster; every other zone joins its
        nearest seed's cluster."""
        w, hgt = self.g["grid"]
        occ = np.zeros((w, hgt), bool)
        ends = self.W + self.J
        seed = ends >= seed_min
        for z in np.nonzero(seed)[0]:
            s = self.side[z]
            occ[self.gx[z]:self.gx[z] + s, self.gy[z]:self.gy[z] + s] = True
        r = gap_km / 2.0 / self.zk
        ri = int(np.ceil(r))
        yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
        disk = (xx * xx + yy * yy) <= r * r
        dil = ndimage.binary_dilation(occ, structure=disk)
        lab, n = ndimage.label(dil)
        lab_seed = np.where(occ, lab, 0)
        # nearest seed square for every grid square
        _, (ix, iy) = ndimage.distance_transform_edt(lab_seed == 0, return_indices=True)
        cl = lab_seed[ix[self.gx, self.gy], iy[self.gx, self.gy]]
        # renumber by workers, biggest first
        ids = np.unique(cl)
        wsum = np.array([self.W[cl == i].sum() for i in ids])
        order = ids[np.argsort(-wsum)]
        remap = {old: k for k, old in enumerate(order)}
        return np.array([remap[c] for c in cl])

    def cluster_names(self, cl):
        try:
            with open(os.path.join(PACKS, f"{self.city}.places.json"), encoding="utf-8") as f:
                places = [p for p in json.load(f)["places"] if p[1] <= 1]
        except OSError:
            places = []
        names = {}
        h3s = self.a["h3"]
        cellz = self.zone
        cellw = (self.a["commute_home"] if "commute_home" in self.a else self.a["pop"]).astype(np.float64)
        zc = cl[cellz]
        for c in range(cl.max() + 1):
            idx = np.nonzero(zc == c)[0]
            if len(idx) == 0:
                names[c] = f"#{c}"
                continue
            best = idx[np.argmax(cellw[idx])]
            lat, lon = h3.cell_to_latlng(h3.int_to_str(int(h3s[best])))
            nm, bd = f"#{c} ({lat:.2f},{lon:.2f})", 25.0
            for p in places:
                dkm = 111.2 * np.hypot((p[2] - lon) * np.cos(np.radians(lat)), p[3] - lat)
                if dkm < bd:
                    bd, nm = dkm, p[0]
            names[c] = nm
        return names

    # ---- solves ----
    def furness(self, J=None, F=None, clamp=None, iters=4000, tol=1e-4, W=None):
        """Doubly constrained by default. clamp=m: rows exact, each column's pull relative to
        distance alone (B_j / J_j, normalised by its job-weighted geometric mean) held within
        [1/m, m]; m=1 is singly (origin) constrained."""
        F = self.F if F is None else F
        J = self.J if J is None else J
        W = self.W if W is None else W
        jm = J > 0
        B = J.copy()
        A = np.zeros_like(W)
        err = np.inf
        for it in range(iters):
            rs = F @ B
            A = np.where(rs > 0, W / np.where(rs > 0, rs, 1), 0)
            cs = F.T @ A
            if clamp is None:
                Bn = np.where(cs > 0, J / np.where(cs > 0, cs, 1), 0)
            else:
                b = np.where(cs > 0, 1.0 / np.where(cs > 0, cs, 1), 0)
                lb = np.log(b[jm])
                gm = np.exp((J[jm] * lb).sum() / J[jm].sum())
                b = np.clip(b / gm, 1.0 / clamp, clamp)
                Bn = np.where(jm, J * b, 0)
            colsum = cs * B
            err = np.max(np.abs(colsum[jm] / J[jm] - 1)) if clamp is None else np.max(np.abs(Bn - B) / np.maximum(B, 1e-300))
            B = Bn
            if err < tol:
                break
        rs = F @ B
        A = np.where(rs > 0, W / np.where(rs > 0, rs, 1), 0)
        return A, B, it + 1, err

    def metrics(self, A, B, cl=None, F=None, J=None):
        F = self.F if F is None else F
        J = self.J if J is None else J
        T = A[:, None] * F * B[None, :]
        D = self.D
        tot = T.sum()
        out = {"trips": tot, "mean": float((T * D).sum() / tot)}
        for x in (25, 50, 100):
            out[f">{x}"] = float(T[D > x].sum() / tot)
        # median
        hist, edges = np.histogram(D, bins=np.arange(0, 2001, 1.0), weights=T)
        c = np.cumsum(hist) / tot
        out["median"] = float(edges[np.searchsorted(c, 0.5) + 1])
        col = T.sum(axis=0)
        out["jobs_unmatched"] = float(np.abs(col - self.J).sum() / 2 / self.J.sum())
        if cl is not None:
            out["cross_cluster"] = float(T[cl[:, None] != cl[None, :]].sum() / tot)
        self._T = T
        return out


def tstats(p, T, base=None):
    D = p.D
    tot = T.sum()
    out = {"trips": tot, "mean": float((T * D).sum() / tot)}
    for x in (25, 50, 100, 150):
        out[f">{x}"] = float(T[D > x].sum() / tot)
    hist, edges = np.histogram(D, bins=np.arange(0, 2001, 1.0), weights=T)
    c = np.cumsum(hist) / tot
    out["median"] = float(edges[np.searchsorted(c, 0.5) + 1])
    col = T.sum(axis=0)
    out["jobs_unmatched"] = float(np.abs(col - p.J).sum() / 2 / p.J.sum())
    if base is not None:
        out["moved"] = float(np.abs(T - base).sum() / 2 / base.sum())
    return out


def fmt2(m):
    s = (f"mean {m['mean']:6.1f} km, median {m['median']:4.0f}, >25 {m['>25']:6.2%}, >50 {m['>50']:6.2%}, "
         f">100 {m['>100']:6.2%}, >150 {m['>150']:6.2%}, work ends off the jobs {m['jobs_unmatched']:5.1%}")
    if "moved" in m:
        s += f", trips moved {m['moved']:6.2%}"
    return s


def components(p, X):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    act = np.nonzero((p.W + p.J) > 0)[0]
    adj = csr_matrix(p.D[np.ix_(act, act)] <= X)
    n, lab = connected_components(adj, directed=False)
    comp = np.full(p.nz, -1, np.int64)
    comp[act] = lab
    return comp, n


def lead(p, X, s=0.0, xmax=150.0, tol=1e-3):
    """The lead option: the local layer reaches X km (decay 0 beyond), each group of zones joined by
    pairs within X has its jobs scaled to its workers, and a share s of every zone's workers is added
    back as long commutes to jobs between X and xmax km, by jobs x the curve, unbalanced."""
    W, J = p.W, p.J
    far = (p.D > X) & (p.D <= xmax)
    L = np.zeros_like(p.F)
    if s > 0:
        Fl = np.where(far, p.F, 0.0)
        wl = Fl @ J
        k = np.where(wl > 0, s * W / np.where(wl > 0, wl, 1), 0)
        L = k[:, None] * Fl * J[None, :]
        del Fl
    Wl = W - L.sum(axis=1)
    Jl = np.maximum(J - L.sum(axis=0), 0)
    comp, n = components(p, X)
    for c in range(n):
        m = comp == c
        wc, jc = Wl[m].sum(), Jl[m].sum()
        if jc > 0:
            Jl[m] = Jl[m] * wc / jc
        else:
            Jl[m] = Wl[m]
    Fx = np.where(p.D <= X, p.F, 0.0)
    A, B, it, err = p.furness(J=Jl, F=Fx, W=Wl, tol=tol)
    T = A[:, None] * Fx * B[None, :] + L
    del Fx, L
    return T, it, err, n


def pull_spread(p, B, J=None):
    """B_j / J_j over its job-weighted geometric mean: how hard each zone's jobs pull beyond their count."""
    J = p.J if J is None else J
    jm = J > 0
    b = B[jm] / J[jm]
    gm = np.exp((J[jm] * np.log(b)).sum() / J[jm].sum())
    return b / gm, J[jm]


def lead2(p, m, X=None, s=0.0, xmax=150.0, tol=1e-6, r0=None):
    """The lead option: the balance's pull capped at m (rows exact, jobs past the cap unfilled), the
    local layer cut at X km (None: no cut), and a share s of every zone's workers added back as
    commutes to jobs between r0 (default X) and xmax km by jobs x the curve, unbalanced."""
    W, J = p.W, p.J
    L = None
    r0 = X if r0 is None else r0
    if s > 0:
        far = (p.D > r0) & (p.D <= xmax)
        Fl = np.where(far, p.F, 0.0)
        wl = Fl @ J
        k = np.where(wl > 0, s * W / np.where(wl > 0, wl, 1), 0)
        L = k[:, None] * Fl * J[None, :]
        del Fl, far
        Wl = W - L.sum(axis=1)
        Jl = np.maximum(J - L.sum(axis=0), 0)
    else:
        Wl, Jl = W, J
    Fx = p.F if X is None else np.where(p.D <= X, p.F, 0.0)
    A, B, it, err = p.furness(J=Jl, F=Fx, W=Wl, clamp=m, tol=tol)
    T = A[:, None] * Fx * B[None, :]
    lost = 1 - T.sum() / Wl.sum()
    if L is not None:
        T += L
    return T, it, lost


def fmt(m):
    s = f"mean {m['mean']:6.1f} km, median {m['median']:5.0f}, >25 {m['>25']:6.1%}, >50 {m['>50']:6.1%}, >100 {m['>100']:6.1%}"
    if "cross_cluster" in m:
        s += f", other cluster {m['cross_cluster']:6.1%}"
    s += f", jobs unmatched {m['jobs_unmatched']:5.1%}"
    return s
