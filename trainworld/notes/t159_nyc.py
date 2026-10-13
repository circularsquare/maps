"""T-159: the state-border factor on the New York pack. Re-solves the zone gravity in numpy with
(zone, state) units: base (the shipped gravity), Philadelphia's factor carried over, and a factor
fitted to LODES NY<->NJ. Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t159_nyc.py [PHL_B]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import read_pack  # noqa: E402
import realnet  # noqa: E402

PHL_B = float(sys.argv[1]) if len(sys.argv) > 1 else 1.41


def furness(W, J, F, iters=300, tol=1e-6):
    b = J.copy()
    for _ in range(iters):
        a_ = np.where(W > 0, W / np.maximum(F @ b, 1e-30), 0.0)
        b = np.where(J > 0, J / np.maximum(a_ @ F, 1e-30), 0.0)
        rs = a_ * (F @ b)
        if np.max(np.abs(rs[W > 0] / W[W > 0] - 1)) < tol:
            break
    return a_, b


def main():
    header, a = read_pack.load_pack("nyc")
    g = header["gravity"]
    order = realnet.COUNTY_ORDER
    nc = len(order)
    cc = np.fromfile(os.path.join(realnet.WORK, "cell_county.u8"), np.uint8).astype(np.int64)
    assert len(cc) == header["cells"], "cell_county.u8 does not match the pack"
    zone = a["zone"].astype(np.int64)
    nz = header["zones"]
    zh = np.zeros((nz, nc)); zw = np.zeros((nz, nc))
    np.add.at(zh, (zone, cc), a["commute_home"].astype(np.float64))
    np.add.at(zw, (zone, cc), a["commute_work"].astype(np.float64))
    st_of = np.array([f[:2] for f in order])
    states = sorted(set(st_of))
    S = np.array([[1.0 if st_of[c] == s else 0.0 for s in states] for c in range(nc)])
    uz, us = np.nonzero(((zh + zw) @ S) > 0)
    uh = zh[uz] * S[:, us].T
    uw = zw[uz] * S[:, us].T
    nu = len(uz)
    print(f"{nz} zones -> {nu} (zone, state) units; states {states}")
    gx, gy = a["zone_gx"].astype(np.float32)[uz], a["zone_gy"].astype(np.float32)[uz]
    d = np.float32(g["zone_m"] / 1000.0) * np.hypot(gx[:, None] - gx[None, :], gy[:, None] - gy[None, :])
    d[d == 0] = g["intrazonal_km"]
    F0 = (np.exp(-d / np.float32(g["decay_km"])) * d ** np.float32(-g.get("decay_pow", 0.0))).astype(np.float32)
    del d
    border = us[:, None] != us[None, :]
    W = uh.sum(1); J = uw.sum(1); W = W * J.sum() / W.sum()
    uhs = uh / np.maximum(uh.sum(1, keepdims=True), 1e-12)
    uws = uw / np.maximum(uw.sum(1, keepdims=True), 1e-12)
    L = np.load(os.path.join(realnet.WORK, "lodes_county_flows.npy"))
    I = {realnet.COUNTIES[f]: i for i, f in enumerate(order)}
    ny = [i for i, f in enumerate(order) if f.startswith("36")]
    nj = [i for i, f in enumerate(order) if f.startswith("34")]

    def county(F):
        a_, b = furness(W, J, F.astype(np.float64) if F.dtype != np.float64 else F)
        M = (uhs * a_[:, None]).T @ (F @ (uws * b[:, None]))
        return M

    def cross(M):
        return (M[np.ix_(ny, nj)].sum() + M[np.ix_(nj, ny)].sum()) / M.sum()

    target = cross(L)
    res = {}
    res["base"] = county(F0)
    res[f"PHL b={PHL_B:.2f}"] = county(np.where(border, F0 * np.float32(np.exp(-PHL_B)), F0))
    lo, hi = 0.0, 4.0
    for _ in range(14):
        bb = (lo + hi) / 2
        c = cross(county(np.where(border, F0 * np.float32(np.exp(-bb)), F0)))
        lo, hi = (bb, hi) if c > target else (lo, bb)
    bb = (lo + hi) / 2
    res[f"fit b={bb:.2f}"] = county(np.where(border, F0 * np.float32(np.exp(-bb)), F0))
    names = list(res)
    man = I["New York"]
    pairs = [("Hudson", "New York"), ("Bergen", "New York"), ("Essex", "New York"), ("Passaic", "New York"), ("Union", "New York"),
             ("Rockland", "Bergen"), ("Richmond", "Union"), ("Westchester", "Bergen"), ("Kings", "New York"), ("Westchester", "New York")]
    print("\nshare of home county's commuters, model/LODES in brackets")
    print("| pair | " + " | ".join(names) + " | LODES |")
    print("|---|" + "---|" * (len(names) + 1))
    for h, w in pairs:
        i, j = I[h], I[w]
        lv = L[i, j] / L[i].sum()
        print(f"| {h} -> {w} | " + " | ".join(f"{res[n][i, j] / res[n][i].sum():.1%} ({res[n][i, j] / res[n][i].sum() / lv:.2f}x)" for n in names) + f" | {lv:.1%} |")
    print("| NY<->NJ share | " + " | ".join(f"{cross(res[n]):.2%}" for n in names) + f" | {target:.2%} |")
    print("| own county | " + " | ".join(f"{np.trace(res[n]) / res[n].sum():.1%}" for n in names) + f" | {np.trace(L) / L.sum():.1%} |")
    for n in names:
        print(f"  {n}: to Manhattan all {res[n][:, man].sum() / res[n].sum():.1%} (LODES {L[:, man].sum() / L.sum():.1%})")


if __name__ == "__main__":
    main()
