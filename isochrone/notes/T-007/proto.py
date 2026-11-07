"""T-007 prototype data: L10 cells split into their land groups, with links and per-group crossings.

    python isochrone/notes/T-007/proto.py <out dir> on|off <tile id>...

on/off: roads. Reads the served data/tiles (read only) through tools/build_tiles.py's own crossing
code. Per 5 degree tile, L10 (f = 10):

- groups: 8-connected L1 land components of a coarse cell that touch a coarse land neighbour's
  land ("linked"), those joined within the cell's 3 x 3 coarse block counted as one. A cell is
  split when it has two or more groups; slot 0 is the group with the most L1 cells, the cell's
  other land (unlinked islands) belongs to no slot. Unsplit cells have one slot, all their land.
- links: for each slot and plain move k (js/search.js order), the slots of the neighbour it
  touches: L1 8-neighbour pixels across the border; a diagonal through the corner pixels, or
  through a common side cell's slot by two straight links.
- crossings: a split cell's slots crossed one by one (the cell's other pixels blocked, margin and
  neighbours as they are), strip-scaled against the neighbours' own one-cell crossings; a slot
  crossed in no direction takes the mean rule over its pixels.
- also each cell's directions not crossed (variant A of the note).

Writes <out>/p_<on|off>_<tile>.json: split cells [cell, [[4 hours per km] per slot],
[[8 masks] per slot], slot map (100, 255 = none)], listed cells (unsplit, links not the default)
[cell, [8 masks]], blocked [cell, bits: 1 E 2 W 4 S 8 N, 16 = no direction crossed], and a check
of the recomputed speeds against the served L10 grid.
"""
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np                      # noqa: E402
from scipy import ndimage               # noqa: E402
import build_tiles as BT                # noqa: E402

F = int(os.environ.get("T007_F", "10"))     # 10: L10, 3: L3
GW = 43200 // F                        # the level's world width in cells
BT.HALO = 21                            # 2 coarse cells + margin: the ring's own 3 x 3 blocks
S2 = np.zeros((3, 3, 3), bool)
S2[1] = True
S8 = np.ones((3, 3), bool)
MV = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]   # js/search.js plain moves (dc, dr)
KOF = {m: k for k, m in enumerate(MV)}
NOMERGE = os.environ.get("T007_NOMERGE") == "1"     # every linked component its own slot (no 3 x 3 join)


def crossings_masked(costs, passable, I, J, h, f, along, inner, m=BT.CROSS_M):
    """BT.crossings with the middle cell's pixels outside inner (k, f, f) blocked."""
    A, Bw = along * f + 2 * m, f + 2 * m
    lead = -((along - 1) // 2) * f - m
    a0 = ((along - 1) // 2) * f + m
    out = {}
    nI = len(I)
    for ew in (True, False):
        args = (-m, Bw, lead, A, False) if ew else (lead, A, -m, Bw, True)
        blocked = np.zeros((A, Bw, nI), bool)
        notin = ~inner                                  # (k, rows, cols)
        if ew:
            blocked[a0:a0 + f, m:m + f] = notin.transpose(2, 1, 0)     # [col, row, k]
        else:
            blocked[a0:a0 + f, m:m + f] = notin.transpose(1, 2, 0)     # [row, col, k]

        def sh(da, db):
            o = np.zeros_like(blocked)
            sa = slice(max(0, -da), A - max(0, da))
            sb = slice(max(0, -db), Bw - max(0, db))
            ta = slice(max(0, da), A - max(0, -da))
            tb = slice(max(0, db), Bw - max(0, -db))
            o[sa, sb] = blocked[ta, tb]
            return o
        G = {}
        for mv in BT.MOVES:
            dc, dr = mv
            da, db = (dc, dr) if ew else (dr, dc)
            g = BT.gather(costs[mv], I, J, h, f, *args)
            if da > 0:
                g[A - da:] = np.inf
            elif da < 0:
                g[:-da] = np.inf
            if db > 0:
                g[:, Bw - db:] = np.inf
            elif db < 0:
                g[:, :-db] = np.inf
            bad = blocked | sh(da, db)
            if mv in BT.BETWEEN:
                for (x, y) in BT.BETWEEN[mv]:
                    pa, pb = (x, y) if ew else (y, x)
                    bad |= sh(pa, pb)
            g[bad] = np.inf
            G[mv] = g
        okg = BT.gather(passable, I, J, h, f, *args)
        for d in ("EW" if ew else "SN"):
            flip = d in "WN"
            cb = {}
            for (da, db) in BT.MOVES:
                sa = -da if flip else da
                mv = (sa, db) if ew else (db, sa)
                cb[(da, db)] = G[mv][::-1] if flip else G[mv]
            ok = okg[::-1] if flip else okg
            D = BT.solve_blocks(cb, ok, m, f)
            out[d] = D[m + along * f - 1, m:m + f].min(axis=0)
    return out


def cross_parts(costs, land, land_c, h, f, n):
    """BT.cross_grid's one-cell (with a ring) and 3-cell crossings, kept apart (times)."""
    x1 = {d: np.full((n + 2, n + 2), np.inf, np.float32) for d in BT.DIRS}
    x3 = {d: np.full((n, n), np.inf, np.float32) for d in BT.DIRS}
    for along, store, off, lm in ((1, x1, -1, land_c), (BT.CROSS_STRIP, x3, 0, land_c[1:-1, 1:-1])):
        chunk = max(1, int(1.5e6 // ((along * f + 2 * BT.CROSS_M) * (f + 2 * BT.CROSS_M))))
        I, J = np.nonzero(lm)
        for s in range(0, len(I), chunk):
            ii, jj = I[s:s + chunk], J[s:s + chunk]
            x = BT.crossings(costs, land, ii + off, jj + off, h, f, along)
            for d in BT.DIRS:
                store[d][ii, jj] = x[d]
    return x1, x3


def spans(row0, n, f):
    lat = 85 - (row0 + (np.arange(n) + 0.5) * f) / 120
    return (f * BT.KM_DEG / 120 * np.cos(np.radians(lat))), f * BT.KM_DEG / 120


def tile(tid, roads, out):
    tr, tc = divmod(tid, BT.TX)
    p = os.path.join(out, "p_%s_%d.json" % (roads, tid))
    if os.path.exists(p):
        return
    t0 = time.time()
    f, n, h = F, BT.SIZE[1] // F, BT.HALO
    wn = BT.Win(BT.DATA, tr, tc)
    on = roads == "on"
    if on:
        costs1 = wn.costs1()
        land = wn.land1
    else:
        land = wn.land0
    land0 = wn.land0
    # coarse cells over the window from pixel o0: A coords, cell IA covers rows o0 + f IA ..; tile cell I = IA - 2
    nA = n + 4
    o0 = h - 2 * f
    sub = land[o0:o0 + f * nA, o0:o0 + f * nA]
    sub0 = land0[o0:o0 + f * nA, o0:o0 + f * nA]
    q0 = sub0.reshape(nA, f, nA, f).sum(axis=(1, 3))
    q1 = sub.reshape(nA, f, nA, f).sum(axis=(1, 3))
    cl0 = q0 >= BT.MIN_LAND * f * f
    cl = cl0 | (q1 >= BT.MIN_LAND * f * f)
    blocks = sub.reshape(nA, f, nA, f).transpose(0, 2, 1, 3)
    lab, nl = ndimage.label(blocks.reshape(nA * nA, f, f), structure=S2)
    lab = lab.reshape(nA, nA, f, f)
    flat = lab.reshape(nA * nA, f * f)
    cell_of = np.zeros(nl + 1, np.int64)
    rows = np.repeat(np.arange(nA * nA), f * f)
    v = flat.ravel()
    cell_of[v[v > 0]] = rows[v > 0]

    # label pairs across borders, with the plain move from the first to the second
    pairs = []

    def add(a, b, k):
        m = (a > 0) & (b > 0)
        pairs.append(np.stack([a[m], b[m], np.full(int(m.sum()), k)]))
        ko = KOF[(-MV[k][0], -MV[k][1])]
        pairs.append(np.stack([b[m], a[m], np.full(int(m.sum()), ko)]))
    a, b = lab[:, :-1, :, f - 1], lab[:, 1:, :, 0]
    for d in (-1, 0, 1):
        add(a[:, :, max(0, -d):f - max(0, d)], b[:, :, max(0, d):f - max(0, -d)], 0)
    a, b = lab[:-1, :, f - 1, :], lab[1:, :, 0, :]
    for d in (-1, 0, 1):
        add(a[:, :, max(0, -d):f - max(0, d)], b[:, :, max(0, d):f - max(0, -d)], 2)
    add(lab[:-1, :-1, f - 1, f - 1], lab[1:, 1:, 0, 0], 4)       # SE
    add(lab[:-1, 1:, f - 1, 0], lab[1:, :-1, 0, f - 1], 6)       # SW
    pr = np.unique(np.concatenate(pairs, axis=1), axis=1)
    clf = cl.ravel()
    keep = clf[cell_of[pr[0]]] & clf[cell_of[pr[1]]]
    pr = pr[:, keep]
    linked = np.zeros(nl + 1, bool)
    linked[pr[0]] = True

    # slots for cells IA 1..n+2 (tile and ring)
    slot_of = np.full(nl + 1, 255, np.int32)
    nslots = np.zeros((nA, nA), np.int32)
    split = {}
    for IA in range(1, n + 3):
        for JA in range(1, n + 3):
            if not cl[IA, JA]:
                continue
            own = lab[IA, JA]
            ls = np.unique(own[own > 0])
            lk = [L for L in ls if linked[L]]
            groups = None
            if len(lk) >= 2:
                r0, c0 = o0 + (IA - 1) * f, o0 + (JA - 1) * f
                blk, _ = ndimage.label(land[r0:r0 + 3 * f, c0:c0 + 3 * f], structure=S8)
                gmap = {}
                for L in lk:
                    ys, xs = np.nonzero(own == L)
                    gmap.setdefault(int(L) if NOMERGE else int(blk[f + ys[0], f + xs[0]]), []).append(L)
                if len(gmap) >= 2:
                    groups = sorted(gmap.values(), key=lambda g: -sum(int((own == L).sum()) for L in g))
            if groups is None:
                slot_of[ls] = 0
                nslots[IA, JA] = 1
            else:
                for s, g in enumerate(groups):
                    slot_of[g] = s
                nslots[IA, JA] = len(groups)
                split[(IA, JA)] = groups

    # slot links: (cell u, slot a, k) -> mask of v's slots
    cu, cv = cell_of[pr[0]], cell_of[pr[1]]
    sa, sb = slot_of[pr[0]], slot_of[pr[1]]
    ok = (sa != 255) & (sb != 255)
    links = {}
    for u_, a_, k_, b_ in zip(cu[ok].tolist(), sa[ok].tolist(), pr[2][ok].tolist(), sb[ok].tolist()):
        links[(u_, a_, k_)] = links.get((u_, a_, k_), 0) | (1 << b_)

    def L(u_, a_, k_):
        return links.get((u_, a_, k_), 0)

    def masks(IA, JA, s):
        u_ = IA * nA + JA
        out_m = []
        for k, (dc, dr) in enumerate(MV):
            mk = L(u_, s, k)
            if k >= 4:
                for first, second in (((dc, 0), (0, dr)), ((0, dr), (dc, 0))):
                    s1 = (IA + first[1]) * nA + JA + first[0]
                    m1 = L(u_, s, KOF[first])
                    for x in range(8):
                        if m1 >> x & 1:
                            mk |= L(s1, x, KOF[second])
            out_m.append(mk)
        return out_m

    # baseline crossings (as the build), and per-slot crossings in split cells
    row0 = tr * BT.SIZE[1]
    land_c = cl0[1:n + 3, 1:n + 3]                      # the build crosses where the no-roads quarter rule says land
    x1, x3 = cross_parts(wn.costs0, wn.land0, land_c, h, f, n)
    if on:
        y1, y3 = cross_parts(costs1, wn.land1, land_c, h, f, n)
    span_ew, span_ns = spans(row0, n, f)

    def to_kmh(xs1, xs3):
        o = np.zeros((4, n, n))
        for k, d in enumerate(BT.DIRS):
            x = BT.strip_scaled(xs1[d], xs3[d], d) / (span_ew[:, None] if d in "EW" else span_ns)
            with np.errstate(divide="ignore"):
                o[k] = np.where(np.isfinite(x) & (x > 0), 1 / x, 0)
        return o
    v4 = to_kmh(x1, x3)
    u4 = to_kmh(y1, y3) if on else v4
    # check: geometric mean of the crossed directions against the served L10 speed byte
    if f == 10:
        served = np.frombuffer(gzip.open(os.path.join(BT.DATA, "grid.bin.gz")).read(), np.uint8).reshape(1740, 4320)
        sv = served[tr * n:(tr + 1) * n, tc * n:(tc + 1) * n]
    else:
        pt = os.path.join(BT.DATA, "tiles", "l3", "%d_%d.bin.gz" % (tr, tc))
        sv = np.frombuffer(gzip.open(pt).read(), np.uint8).reshape(n, n) if os.path.exists(pt) else np.zeros((n, n), np.uint8)
    g, vf, uf, gu, crossed = BT.directed(v4.reshape(4, -1), u4.reshape(4, -1))
    okc = crossed.reshape(n, n) & (sv > 0)
    sb_mine = np.clip(np.rint(g.reshape(n, n) * BT.SCALE), 1, 255)
    check = {"crossed": int(okc.sum()), "byte_diff_gt1": int((np.abs(sb_mine - sv) > 1)[okc].sum())}

    blocked = []
    for I in range(n):
        for J in range(n):
            if not cl0[I + 2, J + 2]:
                continue
            bits = 0
            for k in range(4):
                if not (v4[k, I, J] > 0 or u4[k, I, J] > 0):
                    bits |= 1 << k
            if bits == 15:
                bits = 16
            if bits:
                blocked.append([int((tr * n + I) * GW + ((tc * n + J) % GW)), bits])

    def mean_rule(pix_mask, sbyte):
        s = sbyte[pix_mask]
        s = s[s > 0]
        return 1.0 / (len(s) / float((BT.SCALE / s.astype(np.float64)).sum())) if len(s) else None

    out_split, out_listed = [], []
    sp_tile = [(IA, JA) for (IA, JA) in split if 2 <= IA < n + 2 and 2 <= JA < n + 2]
    # every slot of every split cell crossed in one batch per grid and length
    inst = []
    for (IA, JA) in sp_tile:
        own = lab[IA, JA]
        for s, grp in enumerate(split[(IA, JA)]):
            inst.append((IA, JA, s, np.isin(own, grp)))
    batch = {}
    if inst:
        ii = np.array([x[0] - 2 for x in inst])
        jj = np.array([x[1] - 2 for x in inst])
        inn = np.stack([x[3] for x in inst])
        for gk, (costs, lnd) in (("v", (wn.costs0, wn.land0)),) + ((("u", (costs1, wn.land1)),) if on else ()):
            batch[gk] = (crossings_masked(costs, lnd, ii, jj, h, f, 1, inn),
                         crossings_masked(costs, lnd, ii, jj, h, f, BT.CROSS_STRIP, inn))
    pos = {(x[0], x[1], x[2]): q for q, x in enumerate(inst)}
    for (IA, JA) in sp_tile:
        I, J = IA - 2, JA - 2
        own = lab[IA, JA]
        sds = []
        smap = np.full((f, f), 255, np.int32)
        for s, grp in enumerate(split[(IA, JA)]):
            inner = np.isin(own, grp)
            smap[inner] = s
            q = pos[(IA, JA, s)]
            res = []
            for gk, X1 in (("v", x1),) + ((("u", y1),) if on else ()):
                c1, c3 = batch[gk]
                kmh = []
                for k, d in enumerate(BT.DIRS):
                    own1 = c1[d][q]
                    if d in "EW":
                        prev, nxt = X1[d][I + 1, J], X1[d][I + 1, J + 2]
                    else:
                        prev, nxt = X1[d][I, J + 1], X1[d][I + 2, J + 1]
                    tot = prev + own1 + nxt
                    x = own1 * c3[d][q] / tot if np.isfinite(tot) and np.isfinite(c3[d][q]) else own1
                    x /= (span_ew[I] if d in "EW" else span_ns)
                    kmh.append(1 / x if np.isfinite(x) and x > 0 else 0.0)
                res.append(np.array(kmh)[:, None])
            vv = res[0]
            uu = res[1] if on else vv
            g1, vf1, uf1, gu1, cr1 = BT.directed(vv, uu)
            px = np.zeros_like(land, bool)
            r0p, c0p = o0 + IA * f, o0 + JA * f
            px[r0p:r0p + f, c0p:c0p + f] = inner
            if not cr1[0]:
                hpk = mean_rule(px, wn.sr if on else wn.sb)
                sd = [hpk] * 4
            else:
                sd = [float(1 / x) for x in (uf1 if on else vf1)[:, 0]]
            sds.append(sd)
        cell = int((tr * n + I) * GW + ((tc * n + J) % GW))
        out_split.append([cell, sds, [masks(IA, JA, s) for s in range(len(split[(IA, JA)]))], smap.ravel().tolist()])
    for I in range(n):
        for J in range(n):
            IA, JA = I + 2, J + 2
            if not cl[IA, JA] or (IA, JA) in split:
                continue
            mk = masks(IA, JA, 0)
            default = [1 if cl[IA + dr, JA + dc] else 0 for (dc, dr) in MV]
            if mk != default:
                out_listed.append([int((tr * n + I) * GW + ((tc * n + J) % GW)), mk])
    res = {"tile": tid, "roads": roads, "split": out_split, "listed": out_listed, "blocked": blocked,
           "check": check, "sec": round(time.time() - t0, 1)}
    with open(p + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(p + ".tmp", p)
    print(tid, roads, "split", len(out_split), "listed", len(out_listed), "blocked", len(blocked), check, res["sec"], "s", flush=True)


if __name__ == "__main__":
    out, roads = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for t in sys.argv[3:]:
        tile(int(t), roads, out)
