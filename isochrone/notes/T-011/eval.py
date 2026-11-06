"""T-011: crossing rules for L1 scored against a cached 100 m truth.

    python isochrone/notes/T-011/eval.py truth <dir> <tile> ...   # 100 m walker, 12 starts, both grids
    python isochrone/notes/T-011/eval.py cross <dir> <tile> ...   # crossings: 1, 3 and 5 cells along
    python isochrone/notes/T-011/eval.py score <dir> <tile> ...   # L1 variants against the truth
    python isochrone/notes/T-011/eval.py all <dir> <tile> ...     # the three (truth kept if there)

truth: <tile>_truth.npz, the 100 m times at every third fine cell (rows and columns from 1) from each
start, with and without roads, and the elevation there. cross: <tile>_cross.npz, crossing slowness
(h/km) per direction and grid for strips of 1, 3, 5 cells. score: <tile>_score.json.

L1 variants (served land mask and fallbacks as pilot.py): today; symN and dirN (one value from the
four directions' geometric mean, or the four) from N-cell strips; symRN / dirRN: each direction's
one-cell crossing times its N-cell strip over the sum of the strip's one-cell crossings (the strip's
consistency, the cell's own character). With roads: four directed values or the plain four and one
road byte (suffix 5).
"""
import json
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools", "terrain"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glo90 as G      # noqa: E402
import walk as Wk      # noqa: E402
import pilot as Pi     # noqa: E402

HALO = 24
NL = 120
CAP = None
STRIPS = (1, 3, 5)
SWEEPS = int(os.environ.get("T011_SWEEPS", "4"))


def load_inputs(name):
    z = G.read_fine(G.raw_path(name), name)
    cls, bits, zz, north_h = Wk.inputs(name, HALO, z)
    return z, cls, bits, zz, north_h


def truth(d, name, ns=12):
    z, cls, bits, zz, north_h = load_inputs(name)
    spd, rb = Pi.served(name)
    land = spd > 0
    out = {"z": z[1::3, 1::3]}
    starts = []
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for grid in ("plain", "roads"):
        costs, s = Wk.move_costs(cls, bits, zz, north_h, roads=(grid == "roads"), cap=CAP)
        if not starts:
            si = s[HALO:HALO + 1200, HALO:HALO + 1200]
            tries = 0
            while len(starts) < ns and tries < 2000:
                r0, c0 = rng.integers(180, 1020, 2)
                tries += 1
                if np.isfinite(si[r0, c0]) and land[r0 // 10, c0 // 10]:
                    starts.append((int(r0), int(c0)))
        g = Wk.csr(costs, (HALO, HALO + 1200, HALO, HALO + 1200))
        del costs
        ts = []
        for (r0, c0) in starts:
            t = dijkstra(g, directed=True, indices=r0 * 1200 + c0, limit=Pi.MAXH)
            ts.append(t.reshape(1200, 1200)[1::3, 1::3].astype(np.float32))
        out[grid] = np.stack(ts)
        del g
    out["starts"] = np.array(starts)
    out["zstart"] = np.array([z[r, c] for r, c in starts])
    np.savez_compressed(os.path.join(d, name + "_truth.npz"), **out)


CROSS = [("1", 1, 1), ("3", 3, 1), ("5", 5, 1), ("3w", 3, 3)]     # key, cells along, margin


def cross(d, name):
    p = os.path.join(d, name + "_cross.npz")
    out = dict(np.load(p)) if os.path.exists(p) else {}
    todo = [c for c in CROSS if "plain_%s_E" % c[0] not in out]
    if not todo:
        return
    z, cls, bits, zz, north_h = load_inputs(name)
    tm = {}
    for grid in ("plain", "roads"):
        t0 = time.time()
        costs, s = Wk.move_costs(cls, bits, zz, north_h, roads=(grid == "roads"), cap=CAP)
        tm["costs_" + grid] = time.time() - t0
        cells = Wk.land_cells(s, HALO)
        for key, n, m in todo:
            t0 = time.time()
            X, _, its = Wk.crossings(costs, s, north_h, HALO, m, cells=cells, along=n, max_sweeps=SWEEPS)
            tm["x%s_%s" % (key, grid)] = time.time() - t0
            for dd in "EWSN":
                a = np.full((NL, NL), np.inf, np.float32)
                a[cells] = X[dd]
                out["%s_%s_%s" % (grid, key, dd)] = a
        del costs
    np.savez_compressed(p, **out)
    print(name, {k: round(v, 2) for k, v in tm.items()}, flush=True)


def strip_ratio(X1, Xn, n, dd):
    """Each cell's one-cell crossing times its n-strip over the sum of the strip's one-cell crossings
    (the neighbours along the direction); where a neighbour is missing, the cell's own."""
    axis = 1 if dd in "EW" else 0
    h = (n - 1) // 2
    tot = np.zeros_like(X1)
    good = np.isfinite(X1)
    for k in range(-h, h + 1):
        sh = np.roll(X1, -k, axis=axis)
        # no wrap at the tile's edge
        if k > 0:
            sl = [slice(None)] * 2; sl[axis] = slice(NL - k, NL); sh[tuple(sl)] = np.inf
        elif k < 0:
            sl = [slice(None)] * 2; sl[axis] = slice(0, -k); sh[tuple(sl)] = np.inf
        good &= np.isfinite(sh)
        tot = tot + np.where(np.isfinite(sh), sh, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(good & np.isfinite(Xn), X1 * Xn * n / tot, X1)
    return r


def variants(C, spd, rb):
    """L1 grids (uint8 km/h x 40; dicts for directed) per grid."""
    today_plain, today_roads = spd, Pi.fold(spd, rb)
    land, land_r = spd > 0, today_roads > 0
    V = {"plain": {"today": today_plain}, "roads": {"today": today_roads}}
    rules = {}
    for key, n, m in CROSS:
        if "plain_%s_E" % key not in C:
            continue
        rules[key] = {g: {dd: C["%s_%s_%s" % (g, key, dd)] for dd in "EWSN"} for g in ("plain", "roads")}
        if n > 1:
            rules["R" + key] = {g: {dd: strip_ratio(C["%s_1_%s" % (g, dd)], C["%s_%s_%s" % (g, key, dd)], n, dd)
                                    for dd in "EWSN"} for g in ("plain", "roads")}
    # strip-scaled without roads, one cell with roads
    for key in ("R3", "R3w"):
        if key in rules:
            rules["mix" + key[1:]] = {"plain": rules[key]["plain"], "roads": rules["1"]["roads"]}
            # the strip-scaled plain times the one-cell road ratio (roadless cells: the plain value)
            q_ = {}
            for dd in "EWSN":
                p1, r1, pr = rules["1"]["plain"][dd], rules["1"]["roads"][dd], rules[key]["plain"][dd]
                with np.errstate(invalid="ignore", divide="ignore"):
                    q_[dd] = np.where(np.isfinite(p1) & np.isfinite(r1) & np.isfinite(pr), pr * np.minimum(1, r1 / p1), r1)
            rules["mixQ" + key[1:]] = {"plain": rules[key]["plain"], "roads": q_}
    # narrow strips without roads; wide-lane strips with roads where roads change a one-cell
    # crossing (or make the cell crossable), the plain value elsewhere
    if "R3" in rules and "R3w" in rules:
        h_ = {}
        for dd in "EWSN":
            p1, r1 = rules["1"]["plain"][dd], rules["1"]["roads"][dd]
            near = (r1 < p1 * 0.999) | (~np.isfinite(p1) & np.isfinite(r1))
            pr, rw = rules["R3"]["plain"][dd], rules["R3w"]["roads"][dd]
            h_[dd] = np.where(near, np.where(np.isfinite(pr), np.minimum(rw, pr), rw), pr)
        rules["hyb"] = {"plain": rules["R3"]["plain"], "roads": h_}
    gm = lambda D_: np.exp(np.mean([np.log(np.maximum(D_[dd], 1e-9)) for dd in "EWSN"], axis=0))
    info = {}
    for rn, R in rules.items():
        dp, nf, _ = Pi.fill(R["plain"], land, today_plain / 40.0)
        dr_, _, _ = Pi.fill(R["roads"], land_r, today_roads / 40.0)
        info[rn] = nf
        qp = {dd: Pi.q(dp[dd]) for dd in "EWSN"}
        qr = {dd: Pi.q(dr_[dd]) for dd in "EWSN"}
        sym_p = Pi.q(np.where(land, gm(dp), 0))
        sym_r_kmh = np.where(land_r, gm(dr_), 0)
        ratio = np.where(land & (sym_r_kmh > 0), np.maximum(sym_p, 1) / 40.0 / np.maximum(sym_r_kmh, 1e-9), 1)
        rbyte = np.where(land, np.clip(np.rint(ratio * 250), 1, 250),
                         np.where(land_r, np.clip(np.rint(sym_r_kmh / 5 * 250), 1, 250), 0))
        rbyte = np.where(land & (rbyte == 250), 0, rbyte).astype(np.uint8)
        sym_r = Pi.fold(sym_p, rbyte)
        dir5 = {dd: np.where(land, Pi.fold(qp[dd], np.where(land, rbyte, 0).astype(np.uint8)), sym_r) for dd in "EWSN"}
        V["plain"]["sym" + rn] = sym_p
        V["plain"]["dir" + rn] = qp
        V["roads"]["sym" + rn] = sym_r
        V["roads"]["dir" + rn] = qr
        V["roads"]["dir%s_5" % rn] = dir5
    return V, info


def score(d, name):
    T = np.load(os.path.join(d, name + "_truth.npz"))
    C = np.load(os.path.join(d, name + "_cross.npz"))
    C = {k: C[k] for k in C.files}
    spd, rb = Pi.served(name)
    lat, lon = Wk.tile_latlon(name)
    V, info = variants(C, spd, rb)
    rows = np.arange(1, 1200, 3) // 10
    res = {"name": name, "fallback": info}
    for grid in ("plain", "roads"):
        res[grid] = {}
        tt = T[grid]
        graphs = {k: Pi.l1_graph(v, lat + 1) for k, v in V[grid].items()}
        t1 = {}
        for k, (g1, l1land) in graphs.items():
            t1[k] = []
            for (r0, c0) in T["starts"]:
                i1 = (r0 // 10) * NL + c0 // 10
                t1[k].append(dijkstra(g1, directed=True, indices=i1, limit=Pi.MAXH).reshape(NL, NL)[np.ix_(rows, rows)]
                             if l1land.ravel()[i1] else np.full((400, 400), np.inf))
        base = np.stack(t1["today"])
        for k in graphs:
            b = np.stack(t1[k])
            a = tt
            ok = np.isfinite(a) & np.isfinite(b) & (a <= Pi.MAXH) & (b <= Pi.MAXH) & (b >= 0.5) & (a > 0)
            okt = ok & np.isfinite(base) & (base >= 0.5) & (base <= Pi.MAXH)
            dz = (T["z"][None] - T["zstart"][:, None, None])
            A_, B_ = a[ok], b[ok]
            rel = np.abs(A_ - B_) / B_
            r = {"n": int(ok.sum()), "rel": float(np.median(rel)), "rel90": float(np.quantile(rel, 0.9)),
                 "ratio": float(A_.sum() / B_.sum()), "shift": float(b[okt].sum() / base[okt].sum())}
            for cname, lo, hi in Pi.CLASSES:
                sel = ok & (dz >= lo) & (dz < hi)
                if sel.sum() > 100:
                    r[cname] = {"n": int(sel.sum()), "rel": float(np.median(np.abs(a[sel] - b[sel]) / b[sel])),
                                "ratio": float(a[sel].sum() / b[sel].sum())}
            res[grid][k] = r
    json.dump(res, open(os.path.join(d, name + "_score.json"), "w"), indent=1)
    print(name, " ".join("%s:%.3f/%.2f" % (k, v["rel"], v["ratio"]) for k, v in res["plain"].items()), flush=True)
    print(name, " ".join("%s:%.3f/%.2f" % (k, v["rel"], v["ratio"]) for k, v in res["roads"].items()), flush=True)


if __name__ == "__main__":
    what, d, tiles = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(d, exist_ok=True)
    stages = ["truth", "cross", "score"] if what == "all" else [what]
    for t in tiles:
        for st in stages:
            if st == "truth" and what == "all" and os.path.exists(os.path.join(d, t + "_truth.npz")):
                continue
            t0 = time.time()
            {"truth": truth, "cross": cross, "score": score}[st](d, t)
            print(st, t, "%.0f s" % (time.time() - t0), flush=True)
