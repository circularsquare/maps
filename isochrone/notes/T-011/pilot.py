"""T-011 pilot: L1 by directed crossing times of the 100 m walker on GLO-90, against the full 100 m
walker and against today's L1.

    python isochrone/notes/T-011/pilot.py <out dir> <tile> [<tile> ...] [--m 1] [--starts 12]

Per tile (GLO-90 in data/raw/glo90/, roads bits in data/roads/bits/, today's L1 in data/tiles/):
  - 100 m costs with roads and without (tools/terrain/walk.py), crossings E, W, S, N per L1 cell
    (forward sweep only, and swept until nothing changes; the difference is printed)
  - truth: Dijkstra over the tile's 100 m grid from `starts` random land cells, both grids
  - L1 grids (served land mask): today (data/tiles l1 + r1), sym (one speed a cell, the geometric
    mean of the four crossings; road byte their ratio), dir8 (four directed speeds with and four
    without roads), dir5 (four without, one road byte), dir8e (dir8 costed on the ellipse)
  - pairs as notes/T-087/compare.mjs: every third 100 m cell reached at both with L1 time >= 0.5 h;
    rel = median |t100 - t1| / t1, ratio = sum t100 / sum t1, shift = sum t1 / sum t1(today)
Writes <out>/<tile>.json and <out>/<tile>_x.npz (crossings).
"""
import gzip
import json
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools", "terrain"))
import glo90 as G      # noqa: E402
import walk as Wk      # noqa: E402

TILES = os.path.join(HERE, "data", "tiles")
HALO = 4
MAXH = 1024.0
NL = 120


def served(name):
    """Today's L1 speed and road bytes of the tile (120 x 120)."""
    lat, lon = Wk.tile_latlon(name)
    r0, c0 = (85 - (lat + 1)) * 120, (lon + 180) * 120
    tr, tc = r0 // 600, c0 // 600
    out = []
    for kind in ("l1", "r1"):
        p = os.path.join(TILES, kind, "%d_%d.bin.gz" % (tr, tc))
        if not os.path.exists(p):
            out.append(np.zeros((NL, NL), np.uint8))
            continue
        t = np.frombuffer(gzip.open(p).read(), np.uint8)[:360000].reshape(600, 600)
        out.append(t[r0 - tr * 600:r0 - tr * 600 + NL, c0 - tc * 600:c0 - tc * 600 + NL].copy())
    return out


def fold(spd, rb):
    """applyRoads (js/tiles.js)."""
    s = spd.astype(np.float64)
    v = np.where(spd > 0, np.rint(s * 250 / np.maximum(rb, 1)), np.rint(rb * 0.8))
    v = np.where(rb > 0, np.clip(v, 1, 255), spd)
    return v.astype(np.uint8)


def q(kmh):
    return np.where(np.isfinite(kmh) & (kmh > 0), np.clip(np.rint(kmh * 40), 1, 255), 0).astype(np.uint8)


def fill(X, land, fallback_kmh):
    """Four directed slownesses (h/km, inf where not crossed) to km/h on the land cells: a missing
    direction takes the geometric mean of those found; none found, the fallback."""
    S = np.stack([X[d] for d in "EWSN"])
    fin = np.isfinite(S)
    lg = np.where(fin, np.log(np.where(fin, S, 1)), 0)
    nf = fin.sum(0)
    gm = np.exp(lg.sum(0) / np.maximum(nf, 1))
    S = np.where(fin, S, gm[None])
    fb = 1 / np.maximum(fallback_kmh, 1e-3)
    S = np.where((nf == 0)[None], fb[None], S)
    return {d: np.where(land, 1 / S[i], 0) for i, d in enumerate("EWSN")}, int((land & (nf == 0)).sum()), \
        int((land & (nf > 0) & (nf < 4)).sum())


def l1_graph(sp, north, ellipse=False):
    """CSR over the 120 x 120 L1 cells. sp: {d: uint8 km/h x 40} directed, or one array (undirected).
    A move costs its length times the mean over the cells it touches of each cell's hours per km in
    the move's direction: linear in the move's east and north shares, or on the ellipse."""
    if not isinstance(sp, dict):
        sp = {d: sp for d in "EWSN"}
    land = sp["E"] > 0
    inv = {d: np.where(sp[d] > 0, 40.0 / np.maximum(sp[d], 1), np.inf) for d in "EWSN"}
    lat = north - (np.arange(NL) + 0.5) / 120
    dxr = Wk.KM_DEG / 120 * np.cos(np.radians(lat))
    dy = Wk.KM_DEG / 120
    n = NL * NL
    rr, cc = np.divmod(np.arange(n), NL)
    src, dst, w = [], [], []
    for mv in Wk.MOVES:
        dc, dr = mv
        r2, c2 = rr + dr, cc + dc
        okm = (r2 >= 0) & (r2 < NL) & (c2 >= 0) & (c2 < NL)
        r2c, c2c = np.clip(r2, 0, NL - 1), np.clip(c2, 0, NL - 1)
        ddx = abs(dc) * 0.5 * (dxr[rr] + dxr[r2c])
        ddy = abs(dr) * dy
        L = np.hypot(ddx, ddy)
        sx = inv["E"] if dc > 0 else inv["W"]
        sy = inv["S"] if dr > 0 else inv["N"]

        def cell(i):
            a, b = sx.ravel()[i], sy.ravel()[i]
            if ellipse:
                return np.sqrt((a * ddx) ** 2 + (b * ddy) ** 2)
            return L * (ddx * a + ddy * b) / (ddx + ddy)
        cells = [rr * NL + cc, r2c * NL + c2c]
        if mv in Wk.BETWEEN:
            (a0, b0), (a1, b1) = Wk.BETWEEN[mv]
            cells += [np.clip(rr + b0, 0, NL - 1) * NL + np.clip(cc + a0, 0, NL - 1),
                      np.clip(rr + b1, 0, NL - 1) * NL + np.clip(cc + a1, 0, NL - 1)]
        cost = sum(cell(i) for i in cells) / len(cells)
        good = okm & np.isfinite(cost)
        src.append(np.arange(n)[good]); dst.append((r2c * NL + c2c)[good]); w.append(cost[good])
    src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
    return csr_matrix((w, (src, dst)), shape=(n, n)), land


def pairs(t100, t1):
    """Every third 100 m cell (rows, columns from 1) reached at both, L1 time >= 0.5 h."""
    a = t100.reshape(1200, 1200)[1::3, 1::3]
    rows = np.arange(1, 1200, 3) // 10
    b = t1.reshape(NL, NL)[np.ix_(rows, rows)]
    ok = np.isfinite(a) & np.isfinite(b) & (a <= MAXH) & (b <= MAXH) & (b >= 0.5) & (a > 0)
    return a[ok], b[ok], ok


CLASSES = (("up", 200, 1e9), ("down", -1e9, -200), ("level", -200, 200))


def main(out, tiles, m, ns, cap):
    os.makedirs(out, exist_ok=True)
    for name in tiles:
        T = {"name": name}
        tt = time.time()
        lat, lon = Wk.tile_latlon(name)
        north = lat + 1
        z = G.read_fine(G.raw_path(name), name)
        cls, bits, zz, north_h = Wk.inputs(name, HALO, z)
        tm = {}
        X, X1, its = {}, {}, {}
        S = {}
        for grid in ("plain", "roads"):
            t0 = time.time()
            costs, s = Wk.move_costs(cls, bits, zz, north_h, roads=(grid == "roads"), cap=cap)
            tm["costs_" + grid] = time.time() - t0
            S[grid] = s
            t0 = time.time()
            x1, cells, _ = Wk.crossings(costs, s, north_h, HALO, m, max_sweeps=1)
            tm["sweep1_" + grid] = time.time() - t0
            t0 = time.time()
            x, cells, it = Wk.crossings(costs, s, north_h, HALO, m, cells=cells, max_sweeps=12)
            tm["sweepall_" + grid] = time.time() - t0
            I, J = cells
            full = {d: np.full((NL, NL), np.inf, np.float32) for d in "EWSN"}
            full1 = {d: np.full((NL, NL), np.inf, np.float32) for d in "EWSN"}
            for d in "EWSN":
                full[d][I, J] = x[d]
                full1[d][I, J] = x1[d]
            X[grid], X1[grid], its[grid] = full, full1, it
            # forward-only against converged
            dif = [np.abs(full1[d] - full[d])[np.isfinite(full[d])] / full[d][np.isfinite(full[d])] for d in "EWSN"]
            dif = np.concatenate(dif)
            T["sweep1_" + grid] = {"mean": float(dif.mean()), "p99": float(np.quantile(dif, 0.99)),
                                   "max": float(dif.max()), "n": int(dif.size), "sweeps": it}
            if grid == "plain":
                truth_costs_plain = costs
            else:
                truth_costs_roads = costs
            del costs
        np.savez_compressed(os.path.join(out, name + "_x.npz"),
                            **{"%s_%s" % (g, d): X[g][d] for g in X for d in "EWSN"})
        # ---- L1 grids
        spd, rb = served(name)
        today_plain, today_roads = spd, fold(spd, rb)
        land = spd > 0
        land_r = today_roads > 0
        fb_plain = spd / 40.0
        fb_roads = today_roads / 40.0
        dp, nofb_p, part_p = fill(X["plain"], land, fb_plain)
        dr_, nofb_r, part_r = fill(X["roads"], land_r, fb_roads)
        T["fallback"] = {"plain_none": nofb_p, "plain_some": part_p, "roads_none": nofb_r, "roads_some": part_r,
                         "land": int(land.sum()), "land_roads": int(land_r.sum())}
        qp = {d: q(dp[d]) for d in "EWSN"}
        qr = {d: q(dr_[d]) for d in "EWSN"}
        gm = lambda D_: np.exp(np.mean([np.log(np.maximum(D_[d], 1e-9)) for d in "EWSN"], axis=0))
        sym_p = q(np.where(land, gm(dp), 0))
        sym_r_kmh = np.where(land_r, gm(dr_), 0)
        # road byte as the page would store it: time with roads over time without, folded back
        ratio = np.where(land & (sym_r_kmh > 0), np.maximum(sym_p, 1) / 40.0 / np.maximum(sym_r_kmh, 1e-9), 1)
        rbyte = np.where(land, np.clip(np.rint(ratio * 250), 1, 250), np.where(land_r, np.clip(np.rint(sym_r_kmh / 5 * 250), 1, 250), 0))
        rbyte = np.where(land & (rbyte == 250), 0, rbyte).astype(np.uint8)
        sym_r = fold(sym_p, rbyte)
        # dir5: the four plain directions with the one road byte; bridges as sym
        dir5 = {}
        for d in "EWSN":
            dir5[d] = fold(qp[d], np.where(land, rbyte, 0).astype(np.uint8))
            dir5[d] = np.where(land, dir5[d], sym_r)
        variants = {
            "plain": {"today": today_plain, "sym": sym_p, "dir": qp, "dire": (qp, True)},
            "roads": {"today": today_roads, "sym": sym_r, "dir8": qr, "dir5": dir5, "dir8e": (qr, True)},
        }
        # gzip sizes of the byte layers, for the format choice
        gz = lambda a: len(zlib.compress(np.ascontiguousarray(a).tobytes(), 6))
        lr = lambda a, b: np.where((a > 0) & (b > 0), np.clip(np.rint(64 * np.log(a.astype(float) / np.maximum(b, 1))), -127, 127), 0).astype(np.int8)
        T["bytes"] = {"today_l1": gz(spd), "today_r1": gz(rb), "sym": gz(sym_p), "rbyte": gz(rbyte),
                      "dir4_raw": gz(np.stack([qp[d] for d in "EWSN"])),
                      "dir4_logratio": gz(np.stack([lr(qp[d], sym_p) for d in "EWSN"])),
                      "dir4_roads_logratio": gz(np.stack([lr(qr[d], qp[d]) for d in "EWSN"]))}
        # directedness
        fe = land & (qp["E"] > 0) & (qp["W"] > 0)
        fs = land & (qp["S"] > 0) & (qp["N"] > 0)
        T["asym"] = {"ew_med": float(np.median(np.abs(np.log(qp["E"][fe] / qp["W"][fe])))),
                     "ns_med": float(np.median(np.abs(np.log(qp["S"][fs] / qp["N"][fs])))),
                     "ew_p90": float(np.quantile(np.abs(np.log(qp["E"][fe] / qp["W"][fe])), 0.9)),
                     "aniso_med": float(np.median(np.abs(np.log(np.sqrt(qp["E"][fe & fs] * qp["W"][fe & fs].astype(float)) / np.sqrt(qp["S"][fe & fs] * qp["N"][fe & fs].astype(float)))))),
                     "kmh_today": float(np.mean(spd[land]) / 40), "kmh_sym": float(np.mean(sym_p[land]) / 40)}
        # ---- truth and L1 searches
        inner = (HALO, HALO + 1200, HALO, HALO + 1200)
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        sp_inner = S["plain"][HALO:HALO + 1200, HALO:HALO + 1200]
        starts = []
        tries = 0
        while len(starts) < ns and tries < 2000:
            r0, c0 = rng.integers(180, 1020, 2)
            tries += 1
            if np.isfinite(sp_inner[r0, c0]) and land[r0 // 10, c0 // 10]:
                starts.append((int(r0), int(c0)))
        T["starts"] = starts
        res = {}
        for grid, costs in (("plain", truth_costs_plain), ("roads", truth_costs_roads)):
            t0 = time.time()
            g100 = Wk.csr(costs, inner)
            tm["csr_" + grid] = time.time() - t0
            graphs = {}
            for k, v in variants[grid].items():
                if isinstance(v, tuple):
                    graphs[k] = l1_graph(v[0], north, ellipse=True)
                else:
                    graphs[k] = l1_graph(v, north)
            acc = {k: {"a": [], "b": [], "bt": [], "dz": []} for k in graphs}
            zs = z[1::3, 1::3]
            t0 = time.time()
            for (r0, c0) in starts:
                t100 = dijkstra(g100, directed=True, indices=r0 * 1200 + c0, limit=MAXH)
                i1 = (r0 // 10) * NL + c0 // 10
                t1s = {}
                for k, (g1, l1land) in graphs.items():
                    if not l1land.ravel()[i1]:
                        continue
                    t1s[k] = dijkstra(g1, directed=True, indices=i1, limit=MAXH)
                if "today" not in t1s:
                    continue
                for k, t1 in t1s.items():
                    a, b, ok = pairs(t100, t1)
                    _, bt, okt = pairs(t100, t1s["today"])
                    both = okt[ok]          # pairs also reached by today, for the shift
                    acc[k]["a"].append(a); acc[k]["b"].append(b)
                    acc[k]["dz"].append(zs[ok] - z[r0, c0])
                    acc[k]["bt"].append((b[both], bt[ok[okt]]))
            tm["truth_" + grid] = time.time() - t0
            res[grid] = {}
            for k, A in acc.items():
                if not A["a"]:
                    continue
                a, b = np.concatenate(A["a"]), np.concatenate(A["b"])
                rel = np.abs(a - b) / b
                sb = sum(x[0].sum() for x in A["bt"]); st = sum(x[1].sum() for x in A["bt"])
                res[grid][k] = {"n": int(a.size), "rel": float(np.median(rel)), "rel90": float(np.quantile(rel, 0.9)),
                                "ratio": float(a.sum() / b.sum()), "bias": float(np.median(np.log(a / b))),
                                "shift": float(sb / st)}
                dz = np.concatenate(A["dz"])
                for cname, lo, hi in CLASSES:
                    sel = (dz >= lo) & (dz < hi)
                    if sel.sum() > 100:
                        res[grid][k][cname] = {"n": int(sel.sum()), "rel": float(np.median(rel[sel])),
                                               "ratio": float(a[sel].sum() / b[sel].sum())}
            del g100
        T["res"] = res
        T["time"] = tm
        T["total_s"] = time.time() - tt
        json.dump(T, open(os.path.join(out, name + ".json"), "w"), indent=1)
        print(name, "%.0f s" % T["total_s"], json.dumps({g: {k: "%.3f/%.2f/%.2f" % (v["rel"], v["ratio"], v["shift"])
                                                            for k, v in r.items()} for g, r in res.items()}),
              "sweep1 %.5f/%.5f" % (T["sweep1_plain"]["max"], T["sweep1_roads"]["max"]), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    m = int(a[a.index("--m") + 1]) if "--m" in a else 1
    ns = int(a[a.index("--starts") + 1]) if "--starts" in a else 12
    cap = float(a[a.index("--cap") + 1]) if "--cap" in a else None
    tiles = [x for i, x in enumerate(a[1:], 1) if not x.startswith("--") and a[i - 1] not in ("--m", "--starts", "--cap")]
    main(a[0], tiles, m, ns, cap)
