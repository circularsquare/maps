"""T-159: re-solve the Philadelphia zone gravity (doubly constrained, Furness, in numpy) under
candidate fixes and compare the cross-river flows with LODES. Read-only on the pack.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t159_variants.py SCRATCH

SCRATCH is t159_flows.py's output folder (zh.npy, zw.npy, lodes_class_flows.npy).

Units are (zone, state) parts, so a 2 km square that straddles the Delaware is two units at the
zone's own intrazonal distance; with no factor this equals the shipped zone gravity exactly.

Variants:
  base      crow-fly zone-centre km, d^-1 exp(-d/29)  (the shipped gravity)
  bridges   NJ <-> non-NJ pairs measured via the best Delaware crossing (approximate bridge
            points below; a stand-in for what an OSM road distance would do at this river)
  border b  crow-fly, decay x exp(-b) when the two units are in different states
  both      bridges and border b
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys

import h3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t159_flows as tf  # noqa: E402

OUT = sys.argv[1]
# Delaware River road crossings in or next to the pack, (lat, lon), approximate (+-1 km); diagnostic only
BRIDGES = {
    "Delaware Memorial": (39.689, -75.512), "Commodore Barry": (39.826, -75.372),
    "Walt Whitman": (39.906, -75.130), "Ben Franklin + PATCO": (39.953, -75.134),
    "Betsy Ross": (39.991, -75.064), "Tacony-Palmyra": (40.011, -75.042),
    "Burlington-Bristol": (40.087, -74.859), "Turnpike": (40.114, -74.818), "Trenton bridges": (40.205, -74.765),
}


def furness(W, J, F, iters=200, tol=1e-6):
    """T_ij = a_i b_j F_ij, rows sum to W, columns to J."""
    b = J.copy()
    for it in range(iters):
        r = F @ b
        a_ = np.where(W > 0, W / np.maximum(r, 1e-300), 0.0)
        c = a_ @ F
        b = np.where(J > 0, J / np.maximum(c, 1e-300), 0.0)
        rs = a_ * (F @ b)
        err = np.max(np.abs(rs[W > 0] / W[W > 0] - 1))
        if err < tol:
            break
    return a_[:, None] * F * b[None, :]


def main():
    header, a = tf.read_pack.load_pack("phl")
    g = header["gravity"]
    zh, zw = np.load(os.path.join(OUT, "zh.npy")), np.load(os.path.join(OUT, "zw.npy"))
    L = np.load(os.path.join(OUT, "lodes_class_flows.npy"))
    nz = header["zones"]
    nc4 = zh.shape[1]
    state_of_class = np.array([tf.CF[c // 4][:2] for c in range(nc4)])
    states = sorted(set(state_of_class))
    S = np.array([[1.0 if state_of_class[c] == s else 0.0 for s in states] for c in range(nc4)])
    # units: (zone, state) with any weight
    zhs, zws = zh @ S, zw @ S
    uz, us = np.nonzero((zhs + zws) > 0)
    nu = len(uz)
    uh = zh[uz] * S[:, us].T  # class weights of the unit's state only
    uw = zw[uz] * S[:, us].T
    print(f"{nz} zones -> {nu} (zone, state) units")
    gx, gy = a["zone_gx"].astype(np.float64)[uz], a["zone_gy"].astype(np.float64)[uz]
    zk = g["zone_m"] / 1000.0
    d = zk * np.hypot(gx[:, None] - gx[None, :], gy[:, None] - gy[None, :])
    d[d == 0] = g["intrazonal_km"]
    # zone centres in km of the pack frame; bridges into that frame by an affine fit on cells
    x0, y0 = g["grid_origin_m"]
    cx, cy = (x0 + (gx + 0.5) * g["zone_m"]) / 1000, (y0 + (gy + 0.5) * g["zone_m"]) / 1000
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(i))) for i in a["h3"][::50]])
    X = np.c_[ll, np.ones(len(ll))]
    kx = np.linalg.lstsq(X, a["x_m"][::50].astype(np.float64) / 1000, rcond=None)[0]
    ky = np.linalg.lstsq(X, a["y_m"][::50].astype(np.float64) / 1000, rcond=None)[0]
    bxy = np.array([[np.r_[p, 1] @ kx, np.r_[p, 1] @ ky] for p in BRIDGES.values()])
    nj = np.array([states[s] == "34" for s in us])
    cross = nj[:, None] != nj[None, :]
    db = np.full((nu, nu), np.inf)
    for bx, by in bxy:
        r = np.hypot(cx - bx, cy - by)
        db = np.minimum(db, r[:, None] + r[None, :])
    d_br = np.where(cross, np.maximum(db, g["intrazonal_km"]), d)
    border = us[:, None] != us[None, :]

    W = uh.sum(1); J = uw.sum(1)
    W = W * J.sum() / W.sum()
    f = lambda dd: np.exp(-dd / g["decay_km"]) * dd ** -g["decay_pow"]
    uhs = uh / np.maximum(uh.sum(1, keepdims=True), 1e-12)
    uws = uw / np.maximum(uw.sum(1, keepdims=True), 1e-12)
    C2G = tf.class_to_groups()
    LG = C2G.T @ L @ C2G
    homeL = LG[:, :tf.NC].sum(1)
    P, CM, BU, GL, DE, MO = (tf.CIDX[x] for x in ["42101", "34007", "34005", "34015", "42045", "42091"])
    CC, CAM = tf.GCC, tf.GCAM
    pa = [tf.CIDX[x] for x in tf.CF if x.startswith("42")]
    njc = [tf.CIDX[x] for x in tf.CF if x.startswith("34")]
    pairs = [(CC, CM, "Center City -> Camden Co"), (CAM, CC, "Camden city -> Center City"), (CM, CC, "Camden Co -> Center City"),
             (P, CM, "Philadelphia -> Camden Co"), (CM, P, "Camden Co -> Philadelphia"), (P, BU, "Philadelphia -> Burlington"),
             (P, GL, "Philadelphia -> Gloucester"), (P, DE, "Philadelphia -> Delaware Co"), (P, MO, "Philadelphia -> Montgomery"),
             (P, P, "Philadelphia own county"), (CM, CM, "Camden own county")]

    def summarise(T):
        MG = C2G.T @ (uhs.T @ T @ uws) @ C2G
        homeM = MG[:, :tf.NC].sum(1)
        out = {lab: (MG[h, w], MG[h, w] / homeM[h] / (LG[h, w] / homeL[h])) for h, w, lab in pairs}
        pn = MG[np.ix_(pa, njc)].sum() + MG[np.ix_(njc, pa)].sum()
        own = np.trace(MG[:tf.NC, :tf.NC]) / MG[:tf.NC, :tf.NC].sum()
        mean_km = None
        return out, pn / MG[:tf.NC, :tf.NC].sum(), own, MG

    lodes_cross = (LG[np.ix_(pa, njc)].sum() + LG[np.ix_(njc, pa)].sum()) / LG[:tf.NC, :tf.NC].sum()
    print(f"LODES PA<->NJ share of in-region jobs {lodes_cross:.2%}; own county {np.trace(LG[:tf.NC, :tf.NC]) / LG[:tf.NC, :tf.NC].sum():.1%}")
    T0 = W[:, None] * 0  # placeholder
    w_cross = None
    results = {}

    def run(name, F):
        T = furness(W, J, F)
        out, pn, own, MG = summarise(T)
        km = (T * np.where(F > 0, d, 0)).sum() / T.sum()
        results[name] = (out, pn, own, km)
        print(f"  {name}: PA<->NJ {pn:.2%}, own county {own:.1%}, mean crow-fly km {km:.2f}")
        return pn

    run("base", f(d))
    T = furness(W, J, f(d))
    det = (T * np.where(cross, d_br / d, 0)).sum() / (T * cross).sum()
    print(f"  base cross-river trips: mean detour via best bridge {det:.2f}x crow-fly")
    run("bridges", f(d_br))
    # border factor: bisection on exp(-b) to match LODES PA<->NJ share
    for name, dd in [("border", d), ("both", d_br)]:
        lo, hi = 0.0, 4.0
        for _ in range(18):
            b = (lo + hi) / 2
            T = furness(W, J, f(dd) * np.where(border, np.exp(-b), 1.0))
            out, pn, own, MG = summarise(T)
            if pn > lodes_cross:
                lo = b
            else:
                hi = b
        b = (lo + hi) / 2
        run(f"{name} b={b:.2f} (x{np.exp(-b):.3f})", f(dd) * np.where(border, np.exp(-b), 1.0))

    names = list(results)
    print("\n| pair | " + " | ".join(names) + " | LODES jobs |")
    print("|---|" + "---|" * (len(names) + 1))
    for h, w, lab in pairs:
        cells = [f"{results[n][0][lab][0]:,.0f} ({results[n][0][lab][1]:.2f}x)" for n in names]
        print(f"| {lab} | " + " | ".join(cells) + f" | {LG[h, w]:,.0f} |")
    print("| PA<->NJ share | " + " | ".join(f"{results[n][1]:.2%}" for n in names) + f" | {lodes_cross:.2%} |")
    print("| own-county share | " + " | ".join(f"{results[n][2]:.1%}" for n in names) + f" | {np.trace(LG[:tf.NC, :tf.NC]) / LG[:tf.NC, :tf.NC].sum():.1%} |")
    print("| mean crow-fly km | " + " | ".join(f"{results[n][3]:.2f}" for n in names) + " | |")


if __name__ == "__main__":
    main()
