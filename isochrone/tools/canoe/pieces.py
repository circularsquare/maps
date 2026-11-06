"""Water pieces of a coarse level (T-088, docs/canoe.md 4), for rivers.py's contract_level.

A water piece is an 8-connected run of L1 sea or lake pixels inside one coarse cell. Most coarse
cells hold one piece or none; a cell is *split* when it holds two or more, or holds any while it is
river water (kind 2, whose own node has no grid edges). Only split cells need anything beyond the
cell-level water bands, so only their pieces get ids here.

label_level() returns, for a level:
  split      bool per coarse cell
  comp_cell  coarse cell of each piece (int64[G]); the pieces of split cells only
  n_sea, n_lake, lake_code, shore   per piece: pixel counts, largest lake code, least shore code
  blocks     the split cells' L1 blocks: cells (int64[S]), gid (int32[S, k*k], -1 not open water),
             land (bool[S, k*k], L1 land: speed > 0)
  touch      piece pairs whose L1 pixels are 8-neighbours across a cell border, where one side or
             both are in split cells: ua, ub (a piece id >= 0, or -1 - cell for the one water piece
             of a cell that is not split), cell (of a), dir (0 E, 2 S, 4 SE, 5 NE; rivers.py MOVES)
  river_comp the piece of each river pixel asked for, or -1 (not open water, or not in a split cell)
"""
import time

import numpy as np
from scipy import ndimage

W1, H1, STRIP = 43200, 17400, 600
NONE = np.int64(1) << 62
S3 = np.zeros((3, 3, 3), bool)
S3[1] = True                       # 8-connected within one block, never across blocks


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _pairs(a, b, cells, d, out):
    """Keep pixel pairs (a, b) both open water with at least one side in a split cell."""
    m = (a != NONE) & (b != NONE) & ((a >= 0) | (b >= 0))
    if m.any():
        out.append((a[m], b[m], cells[m], np.full(int(m.sum()), d, np.uint8)))


KM1 = 111.195 / 120          # km per L1 pixel north-south
CHUNK = 4000                 # blocks at a time in the access sums


def piece_access(labl, land, nc, lat, k):
    """Access (km) of each piece of each block: the mean straight distance from the block's L1 land
    pixels to the piece's nearest pixel. labl (n, P): the piece's rank in its block, -1 none; land
    (n, P); nc (n,) pieces per block; lat (n,) cell centre latitude. Returns (n, max nc), NaN past nc."""
    P = k * k
    bi, bj = np.divmod(np.arange(P), k)
    out = np.full((len(nc), max(int(nc.max()), 1) if len(nc) else 1), np.nan, np.float32)
    for s in range(0, len(nc), CHUNK):
        e = min(len(nc), s + CHUNK)
        dx = (np.cos(np.radians(lat[s:e])) * KM1).astype(np.float32)
        D = np.hypot((bj[None, :, None] - bj[None, None, :]) * dx[:, None, None],
                     np.float32(KM1) * (bi[None, :, None] - bi[None, None, :])).astype(np.float32)
        ld = land[s:e].astype(np.float32)
        nl = np.maximum(ld.sum(1), 1)
        for q in range(int(nc[s:e].max())):
            m = labl[s:e] == q
            dmin = np.where(m[:, None, :], D, np.inf).min(2)
            a = (np.where(np.isfinite(dmin), dmin, 0) * ld).sum(1) / nl
            out[s:e, q] = np.where(nc[s:e] > q, a, np.nan)
    return out


def label_level(k, kind1, water1, shore1, kc, river_px, land_c=None):
    """k: cells of k x k L1 pixels. kind1, shore1: L1 canoe bands; water1: L1 speed == 0; kc: the
    level's coarse kind (Hc, Wc); river_px: sorted L1 pixel ids (row * W1 + col) of river nodes;
    land_c: the level's land cells (speed > 0), where the access of each piece is worked out."""
    Wc, Hc = W1 // k, H1 // k
    sc, P = STRIP // k, k * k
    acc_cell = np.zeros(Hc * Wc, np.float32)     # access of the one piece of a land cell not split
    acc_comp = []                                # access of each piece of a split land cell, by gid
    kcf = kc.ravel()
    river_comp = np.full(len(river_px), -1, np.int64)
    blk_cells, blk_gid, blk_land, blk_kind, blk_shore = [], [], [], [], []
    touch = []
    G = 0
    prev_row = None                 # uid of the last pixel row of the previous strip
    prev_cells = None
    split_all = np.zeros(Hc * Wc, bool)
    for i in range(H1 // STRIP):
        r0 = i * STRIP
        ks = kind1[r0:r0 + STRIP]
        ow = (ks == 1) | (ks >= 3)
        blocks = np.ascontiguousarray(ow.reshape(sc, k, Wc, k).transpose(0, 2, 1, 3)).reshape(sc * Wc, k, k)
        lab, _ = ndimage.label(blocks, structure=S3)
        del blocks
        labf = lab.reshape(sc * Wc, P)
        hi = labf.max(1).astype(np.int64)
        lo = np.where(labf > 0, labf, np.iinfo(np.int32).max).min(1).astype(np.int64)
        ncomp = np.where(hi > 0, hi - lo + 1, 0)
        # label numbers features in scan order, so a block's pieces are one run of numbers
        nz = hi > 0
        assert (lo[nz][1:] == hi[nz][:-1] + 1).all()
        cells = (i * sc + np.arange(sc * Wc) // Wc) * Wc + np.arange(sc * Wc) % Wc
        split = (ncomp >= 2) | ((ncomp >= 1) & (kcf[cells] == 2))
        split_all[cells] = split
        sb = np.nonzero(split)[0]
        base = G + np.concatenate([[0], np.cumsum(ncomp[sb])[:-1]]).astype(np.int64)
        G += int(ncomp[sb].sum())
        gid = np.where(labf[sb] > 0, labf[sb] - lo[sb, None] + base[:, None], -1)
        uidb = np.where(labf > 0, -1 - cells[:, None], NONE)
        uidb[sb] = np.where(gid >= 0, gid, NONE)
        uid = uidb.reshape(sc, Wc, k, k).transpose(0, 2, 1, 3).reshape(STRIP, W1)
        del uidb
        wb = water1[r0:r0 + STRIP].reshape(sc, k, Wc, k).transpose(0, 2, 1, 3).reshape(sc * Wc, P)
        if land_c is not None:
            ab = np.nonzero((ncomp >= 1) & land_c.ravel()[cells] & (~wb).any(1))[0]
            if len(ab):
                labl = np.where(labf[ab] > 0, labf[ab] - lo[ab, None], -1)
                lat = 85.0 - (cells[ab] // Wc + 0.5) * k / 120.0
                A = piece_access(labl, ~wb[ab], ncomp[ab], lat, k)
                ns = ~split[ab]
                acc_cell[cells[ab][ns]] = A[ns, 0]
                sp_ = np.nonzero(split[ab])[0]
                if len(sp_):
                    # gid of rank q in split block b: base + q
                    bpos = np.searchsorted(sb, ab[sp_])
                    qq = np.arange(A.shape[1])[None, :]
                    ok = qq < ncomp[ab[sp_]][:, None]
                    acc_comp.append(((base[bpos][:, None] + qq)[ok], A[sp_][ok]))
        del labf, lab
        kb = ks.reshape(sc, k, Wc, k).transpose(0, 2, 1, 3).reshape(sc * Wc, P)
        shb = shore1[r0:r0 + STRIP].reshape(sc, k, Wc, k).transpose(0, 2, 1, 3).reshape(sc * Wc, P)
        blk_cells.append(cells[sb])
        blk_gid.append(gid.astype(np.int32))
        blk_land.append(~wb[sb])
        blk_kind.append(kb[sb])
        blk_shore.append(shb[sb])
        del wb, kb, shb
        # river pixels in the strip
        a0, a1 = np.searchsorted(river_px, [r0 * W1, (r0 + STRIP) * W1])
        if a1 > a0:
            px = river_px[a0:a1] - r0 * W1
            u = uid[px // W1, px % W1]
            river_comp[a0:a1] = np.where((u >= 0) & (u != NONE), u, -1)
        # borders inside the strip
        rows = np.arange(STRIP)
        crow = (i * sc + rows // k)[:, None] * Wc            # cell row offset of each pixel row
        cidx = np.arange(Wc)[None, :]
        a = uid[:, k - 1::k]                                 # east: last column of each cell ...
        b = np.roll(uid[:, 0::k], -1, axis=1)                # ... against the first of the next
        cl = np.broadcast_to(crow + cidx, a.shape)
        _pairs(a, b, cl, 0, touch)
        inner = (rows[:-1] % k != k - 1)[:, None]            # pixel pairs in one cell row
        _pairs(np.where(inner, a[:-1], NONE), b[1:], cl[:-1], 0, touch)
        _pairs(np.where(inner, a[1:], NONE), b[:-1], cl[1:], 0, touch)
        if sc > 1:
            a = uid[k - 1:STRIP - 1:k]                       # south: last row of a cell row ...
            b = uid[k::k]                                    # ... against the first of the next
            cl = np.broadcast_to((i * sc + np.arange(sc - 1))[:, None] * Wc + np.arange(W1)[None, :] // k, a.shape)
            _pairs(a, b, cl, 2, touch)
            inner_c = (np.arange(W1 - 1) % k != k - 1)[None, :]
            _pairs(np.where(inner_c, a[:, :-1], NONE), b[:, 1:], cl[:, :-1], 2, touch)
            _pairs(np.where(inner_c, a[:, 1:], NONE), b[:, :-1], cl[:, 1:], 2, touch)
            cl = (i * sc + np.arange(sc - 1))[:, None] * Wc + cidx
            _pairs(uid[k - 1:STRIP - 1:k, k - 1::k], np.roll(uid[k::k, 0::k], -1, axis=1), np.broadcast_to(cl, (sc - 1, Wc)), 4, touch)
            cl = (i * sc + 1 + np.arange(sc - 1))[:, None] * Wc + cidx
            _pairs(uid[k::k, k - 1::k], np.roll(uid[k - 1:STRIP - 1:k, 0::k], -1, axis=1), np.broadcast_to(cl, (sc - 1, Wc)), 5, touch)
        # the border with the previous strip
        first = uid[0]
        this_cells = i * sc * Wc + np.arange(W1) // k
        if prev_row is not None:
            inner_c = np.arange(W1 - 1) % k != k - 1
            _pairs(prev_row, first, prev_cells, 2, touch)
            _pairs(np.where(inner_c, prev_row[:-1], NONE), first[1:], prev_cells[:-1], 2, touch)
            _pairs(np.where(inner_c, prev_row[1:], NONE), first[:-1], prev_cells[1:], 2, touch)
            _pairs(prev_row[k - 1::k], np.roll(first[0::k], -1), prev_cells[k - 1::k], 4, touch)
            _pairs(first[k - 1::k], np.roll(prev_row[0::k], -1), this_cells[k - 1::k], 5, touch)
        prev_row = uid[STRIP - 1].copy()
        prev_cells = ((i + 1) * sc - 1) * Wc + np.arange(W1) // k
        del uid
        if (i + 1) % 6 == 0:
            log("L%d pieces: strip %d / %d, %d pieces in %d split cells" % (k, i + 1, H1 // STRIP, G, int(split_all.sum())))
    cells = np.concatenate(blk_cells)
    gid = np.concatenate(blk_gid)
    land = np.concatenate(blk_land)
    kpx = np.concatenate(blk_kind)
    spx = np.concatenate(blk_shore)
    v = gid >= 0
    g = gid[v].astype(np.int64)
    kv = kpx[v]
    n_sea = np.bincount(g, kv == 1, minlength=G).astype(np.int64)
    n_lake = np.bincount(g, kv >= 3, minlength=G).astype(np.int64)
    lake_code = np.zeros(G, np.uint8)
    np.maximum.at(lake_code, g, np.where(kv >= 3, kv, 0).astype(np.uint8))
    shore = np.full(G, 255, np.uint8)
    np.minimum.at(shore, g, np.where(spx[v] > 0, spx[v], 255).astype(np.uint8))
    comp_cell = np.zeros(G, np.int64)
    comp_cell[g] = np.repeat(cells, P).reshape(-1, P)[v]
    if touch:
        ua, ub, tc, td = (np.concatenate(x) for x in zip(*touch))
        o = np.lexsort((td, tc, ub, ua))
        ua, ub, tc, td = ua[o], ub[o], tc[o], td[o]
        keep = np.ones(len(o), bool)
        keep[1:] = (ua[1:] != ua[:-1]) | (ub[1:] != ub[:-1]) | (tc[1:] != tc[:-1]) | (td[1:] != td[:-1])
        ua, ub, tc, td = ua[keep], ub[keep], tc[keep], td[keep]
    else:
        ua = ub = tc = np.zeros(0, np.int64)
        td = np.zeros(0, np.uint8)
    log("L%d: %d split cells, %d pieces in them, %d piece touches across borders" % (k, int(split_all.sum()), G, len(ua)))
    acc = np.zeros(G, np.float32)
    for gg, aa in acc_comp:
        acc[gg] = aa
    return dict(split=split_all, comp_cell=comp_cell, n_sea=n_sea, n_lake=n_lake, lake_code=lake_code, shore=shore,
                acc=acc, acc_cell=acc_cell,
                blocks=dict(cells=cells, gid=gid, land=land), touch=dict(ua=ua, ub=ub, cell=tc, dir=td),
                river_comp=river_comp, G=G)
