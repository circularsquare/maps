"""T-007: how often a coarse walking cell (L3, L10) joins L1 land that L1 keeps apart.

    python isochrone/notes/T-007/count.py <out dir> [tile ids...]     # all L1 tiles by default

Reads the served data/tiles (l1, r1) and data/canoe/l1 (water kind), read only. For each coarse
level (f = 10, 3) and roads off / on, per coarse land cell u (quarter rule; with roads also the
cells the bridges make land):

- components: 8-connected L1 land inside u; "linked" when it touches the land of a coarse land
  neighbour across the border (8-neighbour pixels; a diagonal through the corner pixels).
- multi: two or more linked components. groups: the linked components joined within the 3 x 3
  coarse block round u count as one; "barrier" when there are two or more groups (L1 must go
  further than the next cell round, or cannot at all).
- edges: neighbour pairs (E, S, SE, SW from u) both coarse land with no L1 pixel adjacency across
  the border (a diagonal: corner pixels, or each of u and v touching a common side cell); "edge
  barrier" when u's and v's land are not joined within u's 3 x 3 block either.
- kind of the water that makes a barrier: the majority canoe kind of the L1 water pixels in u
  (u and v for an edge): sea, river, lake; sea split further by whether the groups are joined
  within the whole tile window (5 degrees + 0.25 degree halo): fjord/inlet if so, strait if not.

Writes <out>/count_<tile>.json per tile (skipped if present, so a run resumes), and
<out>/l10_<tile>.npy (uint8 codes of the L10 cells, roads on, for the plot: low bits the kind
of a multi cell, 1 sea inlet, 2 strait, 3 river, 4 lake; +8 a barrier; +16 an edge barrier on
its E, S, SE or SW side). Also per level: extra_nm (further nodes if every linked component is
its own node), multi_<kind>, listed (cells that are multi or touch a pair with no adjacency).
"""
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                      # noqa: E402
from scipy import ndimage               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TILES = os.path.join(HERE, "data", "tiles")
CANOE = os.path.join(HERE, "data", "canoe", "l1")
T, TX, TY = 600, 72, 29
H = 30                                  # halo, L1 cells (a multiple of 3 and 10)
RB, SCALE = 250, 40
KINDS = ("fjord", "strait", "river", "lake")
S2 = np.zeros((3, 3, 3), bool)
S2[1] = True                            # 8-connected inside each 2-D slab only

_cache = {}


def read(kind, tr, tc, bands=1):
    tc %= TX
    key = (kind, tr, tc)
    if key in _cache:
        return _cache[key]
    if tr < 0 or tr >= TY:
        a = None
    else:
        p = os.path.join(CANOE if kind == "canoe" else os.path.join(TILES, kind), "%d_%d.bin.gz" % (tr, tc))
        a = np.frombuffer(gzip.open(p).read(), np.uint8).reshape(-1, T, T)[0] if os.path.exists(p) else None
    if len(_cache) > 60:
        _cache.pop(next(iter(_cache)))
    _cache[key] = a
    return a


def window(kind, tr, tc, fill):
    P = T + 2 * H
    out = np.full((P, P), fill, np.uint8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            a = read(kind, tr + di, tc + dj)
            if a is None:
                continue
            rs, re = max(0, H + di * T), min(P, H + (di + 1) * T)
            cs, ce = max(0, H + dj * T), min(P, H + (dj + 1) * T)
            if rs < re and cs < ce:
                out[rs:re, cs:ce] = a[rs - H - di * T:re - H - di * T, cs - H - dj * T:ce - H - dj * T]
    return out


def with_roads(sb, rb):
    sb = sb.astype(np.float64)
    rb = rb.astype(np.float64)
    land = np.clip(np.floor(sb * RB / np.where(rb > 0, rb, 1) + 0.5), 1, 255)
    water = np.clip(np.floor(rb * 5 * SCALE / RB + 0.5), 1, 255)
    return np.where(rb > 0, np.where(sb > 0, land, water), sb).astype(np.uint8)


def level(land, land0, kindw, f, tile_lab):
    """Counts for one level and roads setting over the tile window."""
    P = land.shape[0]
    nW, o, n = P // f, H // f, T // f
    cnt0 = land0.reshape(nW, f, nW, f).sum(axis=(1, 3))
    cnt = land.reshape(nW, f, nW, f).sum(axis=(1, 3))
    cl = (cnt0 >= 0.25 * f * f) | (cnt >= 0.25 * f * f)          # coarse land (roads: + bridge cells)
    blocks = land.reshape(nW, f, nW, f).transpose(0, 2, 1, 3)     # (nW, nW, f, f)
    lab, nl = ndimage.label(blocks.reshape(nW * nW, f, f), structure=S2)
    lab = lab.reshape(nW, nW, f, f)
    cell_of = np.zeros(nl + 1, np.int64)
    flat = lab.reshape(nW * nW, f * f)
    rows = np.repeat(np.arange(nW * nW), f * f)
    v = flat.ravel()
    cell_of[v[v > 0]] = rows[v > 0]
    # label pairs across borders (both directions recorded once; links are symmetric)
    pairs = []
    a, b = lab[:, :-1, :, f - 1], lab[:, 1:, :, 0]                # E
    for d in (-1, 0, 1):
        x = a[:, :, max(0, -d):f - max(0, d)]
        y = b[:, :, max(0, d):f - max(0, -d)]
        pairs.append(np.stack([x.ravel(), y.ravel()]))
    a, b = lab[:-1, :, f - 1, :], lab[1:, :, 0, :]                # S
    for d in (-1, 0, 1):
        x = a[:, :, max(0, -d):f - max(0, d)]
        y = b[:, :, max(0, d):f - max(0, -d)]
        pairs.append(np.stack([x.ravel(), y.ravel()]))
    pairs.append(np.stack([lab[:-1, :-1, f - 1, f - 1].ravel(), lab[1:, 1:, 0, 0].ravel()]))      # SE
    pairs.append(np.stack([lab[:-1, 1:, f - 1, 0].ravel(), lab[1:, :-1, 0, f - 1].ravel()]))      # SW
    pr = np.concatenate(pairs, axis=1)
    pr = pr[:, (pr[0] > 0) & (pr[1] > 0)]
    pr = np.unique(pr, axis=1)
    cu, cv = cell_of[pr[0]], cell_of[pr[1]]
    clf = cl.ravel()
    keep = clf[cu] & clf[cv]
    pr, cu, cv = pr[:, keep], cu[keep], cv[keep]
    linked = np.zeros(nl + 1, bool)
    linked[pr[0]] = True
    linked[pr[1]] = True
    # cell-level adjacency, for edges
    adj = set(zip(cu.tolist(), cv.tolist()))
    adj |= set((y, x) for x, y in adj)

    tile_cells = np.zeros((nW, nW), bool)
    tile_cells[o:o + n, o:o + n] = True
    land_cells = tile_cells & cl
    nlinked = np.bincount(cell_of[np.nonzero(linked)[0]], minlength=nW * nW).reshape(nW, nW)
    res = {"cells": int(land_cells.sum()), "multi": 0, "extra_nm": 0, "barrier": 0, "extra": 0,
           "edge_pairs": 0, "edge_noadj": 0, "edge_barrier": 0}
    for k in KINDS:
        res["multi_" + k] = 0
        res["barrier_" + k] = 0
        res["edge_" + k] = 0
    code = np.zeros((nW, nW), np.uint8)

    def kind_of(rs, cs, groups_pix):
        wk = kindw[rs, cs][~land[rs, cs]]
        if not len(wk):
            return "lake"
        sea = int(((wk == 1) | (wk == 0)).sum())
        riv = int((wk == 2).sum())
        lak = int((wk >= 3).sum())
        if sea >= riv and sea >= lak:
            labs = set()
            for pix in groups_pix:
                labs.add(int(tile_lab[pix]))
            return "fjord" if len(labs) == 1 else "strait"
        return "river" if riv >= lak else "lake"

    # cells with two or more linked components
    listed = np.zeros((nW, nW), bool)
    for (i, j) in zip(*np.nonzero(land_cells & (nlinked >= 2))):
        res["multi"] += 1
        res["extra_nm"] += int(nlinked[i, j]) - 1
        listed[i, j] = True
        r0, c0 = (i - 1) * f, (j - 1) * f
        blk, _ = ndimage.label(land[r0:r0 + 3 * f, c0:c0 + 3 * f], structure=np.ones((3, 3), bool))
        own = lab[i, j]
        groups, comps = {}, []
        for L in np.unique(own[own > 0]):
            if not linked[L]:
                continue
            ys, xs = np.nonzero(own == L)
            g = int(blk[f + ys[0], f + xs[0]])
            px = (r0 + f + ys[0], c0 + f + xs[0])
            comps.append(px)
            groups.setdefault(g, px)
        cs = (slice(i * f, (i + 1) * f), slice(j * f, (j + 1) * f))
        km = kind_of(cs[0], cs[1], comps)
        res["multi_" + km] += 1
        code[i, j] = 1 + KINDS.index(km)
        if len(groups) >= 2:
            res["barrier"] += 1
            res["extra"] += len(groups) - 1
            k = kind_of(cs[0], cs[1], list(groups.values()))
            res["barrier_" + k] += 1
            code[i, j] |= 8
    # neighbour pairs with no adjacency
    for (di, dj) in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ii, jj = np.nonzero(land_cells)
        vi, vj = ii + di, jj + dj
        ok = cl[vi, vj]
        for i, j, a_, b_ in zip(ii[ok], jj[ok], vi[ok], vj[ok]):
            res["edge_pairs"] += 1
            u, v = i * nW + j, a_ * nW + b_
            if (u, v) in adj:
                continue
            if di and dj:
                s1, s2 = i * nW + b_, a_ * nW + j
                if ((u, s1) in adj and (s1, v) in adj) or ((u, s2) in adj and (s2, v) in adj):
                    continue
            res["edge_noadj"] += 1
            listed[i, j] = True
            listed[a_, b_] = True
            r0, c0 = (i - 1) * f, (j - 1) * f
            blk, _ = ndimage.label(land[r0:r0 + 3 * f, c0:c0 + 3 * f], structure=np.ones((3, 3), bool))
            bu = set(np.unique(blk[f:2 * f, f:2 * f])) - {0}
            ra, ca = (a_ - i + 1) * f, (b_ - j + 1) * f
            bv = set(np.unique(blk[ra:ra + f, ca:ca + f])) - {0}
            if bu & bv:
                continue
            res["edge_barrier"] += 1
            # the water of both cells
            rs = slice(min(i, a_) * f, (max(i, a_) + 1) * f)
            cs = slice(min(j, b_) * f, (max(j, b_) + 1) * f)
            pu = np.argwhere(land[i * f:(i + 1) * f, j * f:(j + 1) * f])
            pv = np.argwhere(land[a_ * f:(a_ + 1) * f, b_ * f:(b_ + 1) * f])
            gp = [(i * f + pu[0][0], j * f + pu[0][1]), (a_ * f + pv[0][0], b_ * f + pv[0][1])] if len(pu) and len(pv) else []
            k = kind_of(rs, cs, gp)
            res["edge_" + k] += 1
            code[i, j] |= 16
    res["listed"] = int((listed & tile_cells).sum())
    return res, code[o:o + n, o:o + n]


def tile(tid, out):
    tr, tc = divmod(tid, TX)
    p = os.path.join(out, "count_%d.json" % tid)
    if os.path.exists(p):
        return
    t0 = time.time()
    sb = window("l1", tr, tc, 0)
    rb = window("r1", tr, tc, 0)
    kw = window("canoe", tr, tc, 1)
    land0 = sb > 0
    land1 = with_roads(sb, rb) > 0
    res = {}
    for roads, land in (("off", land0), ("on", land1)):
        tl, _ = ndimage.label(land, structure=np.ones((3, 3), bool))
        for f in (10, 3):
            r, code = level(land, land0, kw, f, tl)
            res["L%d_%s" % (f, roads)] = r
            if f == 10 and roads == "on":
                np.save(os.path.join(out, "l10_%d.npy" % tid), code)
    res["sec"] = round(time.time() - t0, 1)
    with open(p + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(p + ".tmp", p)
    print(tid, res["sec"], {k: (v["cells"], v["barrier"], v["edge_barrier"]) for k, v in res.items() if k != "sec"}, flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ids = [int(x) for x in sys.argv[2:]] or json.load(open(os.path.join(TILES, "index.json")))["levels"]["1"]["tiles"]
    for tid in ids:
        tile(tid, out)
    print("done")
