"""T-044: merge sparse res-9 cells into their res-8 parent, the last step of every pack.

    python pipeline/merge_cells.py <id> [<id> ...]   merge these packs
    python pipeline/merge_cells.py --all             every registered pack in data/packs
    python pipeline/merge_cells.py <id> --stats      counts only, nothing written
    python pipeline/merge_cells.py <id> --stats --child 500   the same with another sparse-child threshold
    python pipeline/merge_cells.py <id> --stats --floor 100   counts with the floor rule at 100 trip ends (0 off)
    python pipeline/merge_cells.py <id> --undo       put the res-9 pack back in data/packs (--force: also
                                                     when the merged pack changed after merging)
    python pipeline/merge_cells.py <ids> --through "<module> {id} <args>" [--workers N] [--force]
                                                     per id: --undo, that step on the res-9 pack, merge

A res-8 parent whose present children make under THRESHOLD trip ends a weekday merges; only the
children inside one 2 km zone merge together (a parent cut by a zone edge gives one merged cell
per zone, all with the parent's index), so every zone's sums, the commute gravity and the routes
are unchanged. In a parent over THRESHOLD, children under CHILD trip ends that touch each other and
share a zone merge together (T-466), keeping the res-9 index of their busiest child. A cell under
CHILD left alone by both joins the touching merged cell of its zone with the fewest trip ends, across
its parent's edge, and lone cells touching only each other merge among themselves (T-510), so long as
no member ends up over REACH from the group's centre. With a floor (T-850b, FLOOR or `--floor`), a
merged or single cell still under that many trip ends joins the nearest cell of its zone part at or
over it, which keeps its index and its place; a zone part with no such cell becomes its busiest
cell. A merged cell sums every per-cell weight and
sits at its children's trip-end-weighted centre. A group stays unmerged if that centre would leave
its zone square or stand on water. Where the pack has `walk_part` (T-700), each part of a zone counts
as a zone here: a merged cell never spans two banks (T-725). The merged pack also carries
`zone_cells9`, each zone's res-9 cells, since a merged cell's index no longer gives its land.

The res-9 pack is kept in data/work/res9/<id>.json/.bin: the first merge of a pack copies it there,
and a rerun (or --undo) starts from it. Steps that rewrite a pack (enrichments, routes.py) run on
the res-9 pack: `--undo`, the step, then this again (`--through` does the three); `routes.write_arrays`
and `packio.replace_cell_arrays` refuse a merged pack (T-1261). Cells come out sorted by their index with the
resolution field cleared (a merged cell right after where its children were), then by zone.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import contextlib
import hashlib
import io
import json
import shutil
import sys

import numpy as np

import packio
import read_pack
from read_pack import PACKS, ROOT, load_pack

WORK = os.path.join(ROOT, "data", "work", "res9")
REGISTRY = os.path.join(ROOT, "app", "src", "game", "registry.json")
THRESHOLD = 2500.0  # trip ends a weekday per res-8 parent (DECISIONS 2026-10-16, T-453)
CHILD = THRESHOLD / 2  # per res-9 child of a busier parent (T-466)
REACH = 700.0  # m, a lone cell's group at most this far from its centre: about a merged parent's own (T-510)
# trip ends a weekday under which a cell joins its zone part's nearest cell at or over it (T-850b,
# notes/T-850.md 5); nocar.py remerges by this default too
FLOOR = 50.0
RES_MASK = np.uint64(0xF) << np.uint64(52)
INT_SAME = ("zone", "region", "walk_part")  # integer per-cell arrays, equal within a merged cell


def _bank(a):
    """Each cell's zone and walk part as one key: cells merge only within one part of a zone, so a
    merged cell never spans a river and every zone keeps its parts (the sim counts a zone's parts
    by its cells' highest `walk_part`; T-725)."""
    zone = a["zone"].astype(np.uint64)
    return zone * np.uint64(256) + a["walk_part"].astype(np.uint64) if "walk_part" in a else zone


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def trip_ends(a, n):
    """A weekday's trip ends per cell (notes/perf-survey.md): commutes and the everyday rates of
    everyday.rs, both ends."""
    def g(k):
        return a[k].astype(np.float64) if k in a else np.zeros(n)
    pop = g("pop")
    P = pop.sum()
    e = 2 * (g("commute_home") + g("commute_work")) + 2 * 0.54 * pop
    for k, rate in (("errand_jobs", 0.3), ("medical_visits", 0.04)):
        w = g(k)
        if w.sum() > 0:
            e += 2 * rate * P * w / w.sum()
    lj = g("leisure_jobs")
    e += 2 * 0.2 * P * (0.5 * pop / max(P, 1) + 0.5 * lj / max(lj.sum(), 1))
    e += 2 * 0.9 * (g("school_children") + g("school_enrollment"))
    e += 2 * 0.6 * (g("university_adults") + g("university_enrollment"))
    e += 2 * g("airport_trips")
    return e


def on_water(header, a, x, y):
    wb = header.get("water")
    if wb is None:
        return np.zeros(len(x), bool)
    W, H = wb["size"]
    off, xs = a["water_row"].astype(np.int64), a["water_x"].astype(np.int64)
    c = np.floor((x - wb["origin_m"][0]) / wb["cell_m"]).astype(np.int64)
    r = np.floor((y - wb["origin_m"][1]) / wb["cell_m"]).astype(np.int64)
    out = np.zeros(len(x), bool)
    for i in np.nonzero((c >= 0) & (c < W) & (r >= 0) & (r < H))[0]:
        seg = xs[off[r[i]]:off[r[i] + 1]]
        out[i] = np.searchsorted(seg, c[i], side="right") % 2 == 1
    return out


def _components():
    """For each set of a parent's children (a 7-bit mask of H3 digits), each child's component
    label (its lowest digit) among the set's children that touch, -1 for a child outside the set
    or alone. The centre child (0) touches all six; the ring runs 4-6-2-3-1-5-4."""
    touch = {(0, d) for d in range(1, 7)} | {(4, 6), (2, 6), (2, 3), (1, 3), (1, 5), (4, 5)}
    nb = {d: set() for d in range(7)}
    for i, j in touch:
        nb[i].add(j)
        nb[j].add(i)
    lab = np.full((128, 7), -1, np.int64)
    for mask in range(128):
        seen = set()
        for d in range(7):
            if not mask >> d & 1 or d in seen:
                continue
            comp, todo = {d}, [d]
            while todo:
                for k in nb[todo.pop()]:
                    if mask >> k & 1 and k not in comp:
                        comp.add(k)
                        todo.append(k)
            seen |= comp
            if len(comp) > 1:
                lab[mask, sorted(comp)] = d
    return lab


COMPONENTS = _components()


def _place(header, a, e, cells, inv, cand=None):
    """Trip-end-weighted centres (f32) of the groups `inv` labels over `cells`, each group's first
    cell, and which groups (of `cand`, default all) may not merge: the centre leaves its zone's
    square, or stands on water."""
    ng = int(inv.max()) + 1 if len(inv) else 0
    size = np.bincount(inv, minlength=ng)
    w = e[cells]
    x, y = a["x_m"][cells].astype(np.float64), a["y_m"][cells].astype(np.float64)
    ws = np.bincount(inv, weights=w, minlength=ng)
    # a plain mean for a group without trip ends
    mx = np.where(ws > 0, np.bincount(inv, weights=x * w, minlength=ng) / np.maximum(ws, 1e-30),
                  np.bincount(inv, weights=x, minlength=ng) / np.maximum(size, 1))
    my = np.where(ws > 0, np.bincount(inv, weights=y * w, minlength=ng) / np.maximum(ws, 1e-30),
                  np.bincount(inv, weights=y, minlength=ng) / np.maximum(size, 1))
    mx32, my32 = mx.astype(np.float32), my.astype(np.float32)
    first = np.full(ng, len(cells), np.int64)
    np.minimum.at(first, inv, np.arange(len(cells)))
    first = cells[first]
    # the gravity's own square test, in f32 as read_pack.py makes it
    g = header["gravity"]
    zm, (x0, y0) = g["zone_m"], g["grid_origin_m"]
    zg = a["zone"].astype(np.int64)[first]
    gx, gy = a["zone_gx"].astype(np.int64)[zg], a["zone_gy"].astype(np.int64)[zg]
    side = read_pack.zone_side(a, header["zones"])[zg]  # T-852: a block's four squares
    cx = ((mx32 - np.float32(x0)) / np.float32(zm)).astype(np.int64)
    cy = ((my32 - np.float32(y0)) / np.float32(zm)).astype(np.int64)
    off_sq = (cx < gx) | (cx >= gx + side) | (cy < gy) | (cy >= gy + side)
    wet = np.zeros(ng, bool)
    idx = np.nonzero(~off_sq & (True if cand is None else cand))[0]
    wet[idx] = on_water(header, a, mx32[idx].astype(np.float64), my32[idx].astype(np.float64))
    return mx32, my32, first, size, off_sq, wet


def _touching(h3, cand):
    """Pairs (i, j) of rows: present cell j touches cell i, for each i in `cand`."""
    from h3.api import basic_int as H
    order = np.argsort(h3, kind="stable")
    sh = h3[order]
    src, nb = [], []
    for i in cand:
        ring = H.grid_ring(int(h3[i]), 1)
        src += [i] * len(ring)
        nb += ring
    src, nb = np.array(src, np.int64), np.array(nb, np.uint64)
    k = np.minimum(np.searchsorted(sh, nb), len(sh) - 1)
    hit = sh[k] == nb
    return src[hit], order[k[hit]]


def _spread(a, cells, inv, gx, gy):
    """Each of `cells`' distance (m) from its group's centre."""
    return np.hypot(a["x_m"][cells].astype(np.float64) - gx[inv], a["y_m"][cells].astype(np.float64) - gy[inv])


def _join_lone(header, a, e, h3, head, nx, ny, merged, lone):
    """T-510: a cell under the sparse-child threshold that neither rule merged joins the touching
    merged cell of its zone with the fewest trip ends (which moves demand least), across its
    parent's edge; lone cells that touch only other lone cells of their zone merge among themselves,
    with the busiest one's index. One pass: a cell joins only a group as it was, never another
    joiner. A grown group over REACH sheds its farthest joiner until it fits; a new group over REACH
    stays apart, and either stays as it was if its centre would leave its zone square or stand on
    water. Writes head, nx and ny in place; returns a summary and the joined."""
    n = len(h3)
    li = np.nonzero(lone)[0]
    joined = np.zeros(n, bool)
    why = np.zeros(n, np.int8)  # a lone cell kept alone by: 1 its square, 2 water, 3 reach
    info = {"lone": int(len(li)), "lone_joined": 0, "lone_groups": 0, "lone_grouped": 0, "lone_group_max": 0}
    if not len(li):
        return dict(info, lone_kept_square=0, lone_kept_water=0, lone_kept_reach=0), joined
    zone = _bank(a)
    i, j = _touching(h3, li)
    same = zone[i] == zone[j]
    i, j = i[same], j[same]
    # 1. into a touching merged cell: the lightest, ties to the lowest row
    ue = np.bincount(head, weights=e, minlength=n)
    s = merged[j]
    ia, ua = i[s], head[j[s]]
    o = np.lexsort((ua, ue[ua], ia))
    ia, ua = ia[o], ua[o]
    keep1 = np.r_[True, ia[1:] != ia[:-1]] if len(ia) else np.zeros(0, bool)
    ia, ua = ia[keep1], ua[keep1]
    while len(ia):
        grow = np.zeros(n, bool)
        grow[ua] = True
        members = np.nonzero(grow[head])[0]
        cells = np.r_[members, ia]
        units = np.r_[head[members], ua]
        uniq, inv = np.unique(units, return_inverse=True)
        inv = inv.ravel()
        gx, gy, _, _, sq, wet = _place(header, a, e, cells, inv)
        d = _spread(a, cells, inv, gx, gy)
        reach = np.zeros(len(uniq))
        np.maximum.at(reach, inv, d)
        if not (reach > REACH).any():
            break
        # a group spread too wide gives up its joiner farthest from its centre, then tries again
        gj = inv[len(members):]
        o = np.lexsort((-d[len(members):], gj))
        farthest = o[np.r_[0, np.nonzero(np.diff(gj[o]))[0] + 1]]
        drop = np.zeros(len(ia), bool)
        drop[farthest[reach[gj[farthest]] > REACH]] = True
        why[ia[drop]] = 3
        ia, ua = ia[~drop], ua[~drop]
    if len(ia):
        ok = ~sq & ~wet
        g = np.searchsorted(uniq, ua)
        go = ok[g]
        why[ia] = np.where(sq[g], 1, np.where(wet[g], 2, 0))
        head[ia[go]] = ua[go]
        joined[ia[go]] = True
        nx[uniq[ok]], ny[uniq[ok]] = gx[ok], gy[ok]
    # 2. the rest among themselves, one cell per touching set
    rest = lone & ~joined
    s = rest[i] & rest[j]
    if s.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        _, comp = connected_components(coo_matrix((np.ones(int(s.sum())), (i[s], j[s])), shape=(n, n)), directed=False)
        c2 = np.nonzero(rest)[0]
        _, cinv, cnt = np.unique(comp[c2], return_inverse=True, return_counts=True)
        c2 = c2[cnt[cinv.ravel()] > 1]
        _, sinv = np.unique(comp[c2], return_inverse=True)
        sinv = sinv.ravel()
        gx, gy, _, size, sq, wet = _place(header, a, e, c2, sinv)
        reach = np.zeros(len(gx))
        np.maximum.at(reach, sinv, _spread(a, c2, sinv, gx, gy))
        wide = reach > REACH
        ok = ~sq & ~wet & ~wide
        why[c2] = np.where(sq, 1, np.where(wet, 2, np.where(wide, 3, 0)))[sinv]
        order = np.lexsort((h3[c2], -e[c2], sinv))
        busiest = c2[order][np.r_[0, np.nonzero(np.diff(sinv[order]))[0] + 1]]
        go = ok[sinv]
        head[c2[go]] = busiest[sinv[go]]
        nx[busiest[ok]], ny[busiest[ok]] = gx[ok], gy[ok]
        joined[c2[go]] = True
        info["lone_groups"] = int(ok.sum())
        info["lone_grouped"] = int(go.sum())
        info["lone_group_max"] = int(size[ok].max()) if ok.any() else 0
    info["lone_joined"] = int(joined.sum()) - info["lone_grouped"]
    left = why[lone & ~joined]
    for k, name in ((1, "square"), (2, "water"), (3, "reach")):
        info[f"lone_kept_{name}"] = int((left == k).sum())
    return info, joined


def _floor(a, e, h3, head, nx, ny, floor):
    """T-850b: each unit (a cell and the cells merged into it) under `floor` trip ends joins the
    nearest unit of its zone part at or over the floor, by centre; in a part with none, every unit
    joins the busiest. The unit joined keeps its index and its place (not the group's weighted
    centre: that would move the busier unit too, more trip-end metres in all, and could stand it on
    water). Writes head in place; returns a summary."""
    n = len(head)
    units = np.nonzero(head == np.arange(n))[0]
    ue = np.bincount(head, weights=e, minlength=n)[units]
    bank = _bank(a)[units]
    x, y = nx[units].astype(np.float64), ny[units].astype(np.float64)
    to = np.arange(len(units))
    low, big = ue < floor, ue >= floor
    li = np.nonzero(low)[0]
    found = np.zeros(len(li), bool)
    if len(li) and big.any():
        from scipy.spatial import cKDTree
        bi = np.nonzero(big)[0]
        # a zone part a third axis 1e7 m apart, so the nearest within 1e6 m is in the same part
        zb = bank.astype(np.float64) * 1e7
        d, k = cKDTree(np.c_[x[bi], y[bi], zb[bi]]).query(np.c_[x[li], y[li], zb[li]], distance_upper_bound=1e6)
        found = np.isfinite(d)
        to[li[found]] = bi[k[found]]
    orphan = li[~found]
    if len(orphan):
        o = orphan[np.lexsort((h3[units[orphan]], -ue[orphan], bank[orphan]))]
        start = np.r_[True, bank[o][1:] != bank[o][:-1]]
        to[o] = o[start][np.cumsum(start) - 1]
    moved = to != np.arange(len(units))
    head[:] = units[to][np.searchsorted(units, head)]
    w, dist = ue[moved], np.hypot(x - x[to], y - y[to])[moved]
    o = np.argsort(dist)
    cw = np.cumsum(w[o])
    at = lambda q: float(dist[o][np.searchsorted(cw, q * cw[-1])]) if len(o) and cw[-1] > 0 else 0.0  # noqa: E731
    return {"floor_trip_ends": float(floor), "floor_cells": int(moved.sum()),
            "floor_parts_alone": int(len(np.unique(bank[orphan]))),
            "floor_ends": float(w.sum() / max(e.sum(), 1e-30)),
            "floor_mean_m": float((w * dist).sum() / max(w.sum(), 1e-30)),
            "floor_p50_m": at(0.5), "floor_p95_m": at(0.95), "floor_max_m": float(dist.max()) if len(dist) else 0.0}


def merge(header, a, threshold=THRESHOLD, child=None, floor=None):
    """The merged pack's cell arrays {name: array} and a summary. Three rules (docs/demand.md 4.1):
    a res-8 parent under `threshold` trip ends merges its children within each zone (T-044);
    in a busier parent, children under `child` trip ends that touch merge among themselves
    (T-466), the merged cell keeping the index of its busiest child; such a child left alone
    joins a touching group across its parent's edge (T-510, `_join_lone`); with a floor (None takes
    FLOOR, 0 none), a unit still under it joins its zone part's nearest unit at or over it (T-850b)."""
    child = CHILD if child is None else child
    floor = FLOOR if floor is None else floor
    n = header["cells"]
    assert header["h3_res"] == 9 and "merged" not in header, "merge from a res-9 pack"
    cell = [e for e in header["arrays"] if e.get("per") == "cell"]
    for e in cell:
        assert e["count"] == n, e["name"]
        if e["dtype"] not in ("f32", "f64") and e["name"] not in ("h3",) + INT_SAME:
            sys.exit(f"{header['city']}: per-cell array {e['name']} ({e['dtype']}) has no merge rule")
    h3 = a["h3"]
    par = (h3 & ~RES_MASK) | (np.uint64(8) << np.uint64(52)) | (np.uint64(7) << np.uint64(18))
    zone = _bank(a)
    e = trip_ends(a, n)
    _, pinv = np.unique(par, return_inverse=True)
    pends = np.bincount(pinv.ravel(), weights=e)[pinv.ravel()]
    _, ginv = np.unique(np.stack([par, zone], 1), axis=0, return_inverse=True)
    ginv = ginv.ravel()
    allc = np.arange(n)
    ng = int(ginv.max()) + 1
    want = np.zeros(ng, bool)
    want[ginv[pends < threshold]] = True
    want &= np.bincount(ginv, minlength=ng) > 1
    mx, my, first, size, out_sq, wet = _place(header, a, e, allc, ginv, want)
    kept_sq, kept_wet = int((want & out_sq).sum()), int((want & ~out_sq & wet).sum())
    want &= ~out_sq & ~wet
    m1 = want[ginv]
    # every cell's unit: its own, its parent group's, or its touching sparse siblings'
    head = allc.copy()
    nh3, nx, ny = h3.copy(), a["x_m"].copy(), a["y_m"].copy()
    head[m1] = first[ginv[m1]]
    nh3[m1], nx[m1], ny[m1] = par[m1], mx[ginv[m1]], my[ginv[m1]]
    # T-725: a parent merged in two walk parts of one zone would give that zone its index twice;
    # such groups keep their busiest child's index instead, as the sparse groups below do
    gz = np.zeros(ng, np.uint64)
    gz[ginv] = a["zone"].astype(np.uint64)
    gp = par[first]
    gk = np.stack([gp[want], gz[want]], 1)
    twin = np.zeros(ng, bool)
    if len(gk):
        _, gi, gc = np.unique(gk, axis=0, return_inverse=True, return_counts=True)
        twin[np.nonzero(want)[0][gc[gi.ravel()] > 1]] = True
        if twin.any():
            o = np.lexsort((h3, -e, ginv))
            top = o[np.r_[0, np.nonzero(np.diff(ginv[o]))[0] + 1]]
            t = m1 & twin[ginv]
            nh3[t] = h3[top[ginv[t]]]
    # T-466: sparse children of a parent left unmerged
    digit = ((h3 >> np.uint64(18)) & np.uint64(7)).astype(np.int64)
    sparse = ~m1 & (e < child)
    gmask = np.zeros(ng, np.int64)
    np.bitwise_or.at(gmask, ginv[sparse], np.left_shift(1, digit[sparse]))
    lab = COMPONENTS[gmask[ginv], digit]
    c2 = np.nonzero(sparse & (lab >= 0))[0]
    _, sinv = np.unique(ginv[c2] * 8 + lab[c2], return_inverse=True)
    sinv = sinv.ravel()
    sx, sy, _, ssize, s_sq, s_wet = _place(header, a, e, c2, sinv)
    ok = ~s_sq & ~s_wet
    # the busiest child gives the merged cell its index (ties: the lowest index)
    order = np.lexsort((h3[c2], -e[c2], sinv))
    busiest = c2[order][np.r_[0, np.nonzero(np.diff(sinv[order]))[0] + 1]] if len(c2) else c2
    m2c = c2[ok[sinv]]
    m2g = sinv[ok[sinv]]
    head[m2c] = busiest[m2g]
    nh3[m2c], nx[m2c], ny[m2c] = h3[busiest[m2g]], sx[m2g], sy[m2g]
    m2 = np.zeros(n, bool)
    m2[m2c] = True
    lone_info, m3 = _join_lone(header, a, e, h3, head, nx, ny, m1 | m2, sparse & ~m2)
    floor_info = _floor(a, e, h3, head, nx, ny, floor) if floor else {}
    # one output row per unit
    keep = head == allc
    row = np.cumsum(keep) - 1
    target = row[head]
    m = int(keep.sum())
    out = {}
    for spec in cell:
        k, v = spec["name"], a[spec["name"]]
        if k == "h3":
            nv = nh3[keep]
        elif k in ("x_m", "y_m"):
            nv = (nx if k == "x_m" else ny)[keep]
        elif k in INT_SAME:
            nv = v[keep]
            assert np.array_equal(nv[target], v), f"{k} differs inside a zone"
        else:
            nv = np.bincount(target, weights=v.astype(np.float64), minlength=m).astype(v.dtype)
        out[k] = np.ascontiguousarray(nv)
    order = np.lexsort((out["zone"], out["h3"] & ~RES_MASK))
    out = {k: v[order] for k, v in out.items()}
    # a pack without walk parts keeps the block it always had, so its re-merge (a joint solve) is unchanged
    parts = {"parents_split_by_part": int(twin.sum())} if "walk_part" in a else {}
    # a merged parent that joined a busier cell under the floor is no longer a cell of its own
    parents = int((want & ~twin).sum()) if not floor_info else int((((out["h3"] & RES_MASK) >> np.uint64(52)) == 8).sum())
    info = {"cells_before": n, "cells": m, "parents": parents, **parts,
            "cells_merged": int(m1.sum()),
            "sparse_groups": int(ok.sum()), "sparse_cells_merged": int(m2.sum()),
            "sparse_child_trip_ends": float(child),
            "ends_merged": float(e[(m1 | m2 | m3) if not floor_info else np.bincount(head, minlength=n)[head] > 1].sum()
                                 / max(e.sum(), 1e-30)),
            "kept_square": kept_sq + int(s_sq.sum()), "kept_water": kept_wet + int((~s_sq & s_wet).sum()),
            **lone_info, **floor_info}
    return out, info


def write_pack(stem, header, a, cells, block):
    """The pack at `stem` from `header`/`a` with its per-cell arrays replaced by `cells`, the .bin
    laid out afresh."""
    h = json.loads(json.dumps(header))
    h["cells"] = block["cells"]
    h["merged"] = block
    buf = bytearray()
    for spec in h["arrays"]:
        v = cells[spec["name"]] if spec.get("per") == "cell" else a.get(spec["name"])
        if v is None:  # a dtype the readers skip: copy its bytes as they are
            sys.exit(f"{h['city']}: array {spec['name']} ({spec['dtype']}) cannot be carried over")
        buf += b"\0" * (-len(buf) % 8)
        spec["offset"], spec["count"] = len(buf), int(len(v))
        buf += np.ascontiguousarray(v).tobytes()
    packio.write(stem, h, buf)
    return len(buf)


def res9_copy(city, copy=True):
    """Where the pack's res-9 version is: data/work/res9, copied there first if data/packs holds it."""
    with open(os.path.join(PACKS, f"{city}.json"), encoding="utf-8") as f:
        live = json.load(f)
    if not copy and "merged" not in live:
        return PACKS
    os.makedirs(WORK, exist_ok=True)
    if "merged" not in live:
        for ext in (".json", ".bin"):
            shutil.copyfile(os.path.join(PACKS, city + ext), os.path.join(WORK, city + ext + ".part"))
        for ext in (".bin", ".json"):
            os.replace(os.path.join(WORK, city + ext + ".part"), os.path.join(WORK, city + ext))
    elif not os.path.exists(os.path.join(WORK, f"{city}.json")):
        sys.exit(f"{city}: data/packs holds a merged pack and data/work/res9 has no res-9 copy")
    elif stale(city):
        sys.exit(STALE.format(city=city))
    return WORK


STALE = ("{city}: the merged pack in data/packs was changed after merging (a step run on it?); the res-9 copy "
         "in data/work/res9 lacks that change, so neither is merged nor undone. Rerun that step on the res-9 "
         "pack: merge_cells.py {city} --through \"<module> {{id}} <args>\" --force (T-1261)")


def stale(city, packs=PACKS, work=WORK):
    """Whether data/packs holds a merged pack that no longer matches the stamp merge_cells wrote for
    it, so its res-9 copy lacks a later change (T-1261)."""
    with open(os.path.join(packs, f"{city}.json"), encoding="utf-8") as f:
        if "merged" not in json.load(f):
            return False
    p = os.path.join(work, f"{city}.merged.json")
    if not os.path.exists(p):
        return True
    with open(p) as f:
        return json.load(f) != stamp(packs, city)


def stamp(d, city):
    return {"json_sha256": sha(os.path.join(d, f"{city}.json")), "bin_sha256": sha(os.path.join(d, f"{city}.bin"))}


def written(city):
    p = os.path.join(WORK, f"{city}.merged.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def run(city, stats=False, child=None, floor=None):
    src = res9_copy(city, copy=not stats)
    header, a = load_pack(city, src)
    cells, info = merge(header, a, child=child, floor=floor)
    line = (f"{city}: {info['cells_before']:,} -> {info['cells']:,} cells ({1 - info['cells'] / info['cells_before']:.0%} fewer), "
            f"{info['parents']:,} merged parents from {info['cells_merged']:,}, {info['sparse_groups']:,} merged siblings "
            f"from {info['sparse_cells_merged']:,}; of {info['lone']:,} lone sparse cells {info['lone_joined']:,} joined a "
            f"neighbour and {info['lone_grouped']:,} made {info['lone_groups']:,} groups (largest {info['lone_group_max']}); "
            f"{info['ends_merged']:.1%} of trip ends; "
            f"groups left alone: {info['kept_square']} off their square, {info['kept_water']} on water; lone cells left "
            f"alone: {info['lone_kept_square']} off their square, {info['lone_kept_water']} on water, "
            f"{info['lone_kept_reach']} too wide")
    if "floor_cells" in info:
        line += (f"; under the floor of {info['floor_trip_ends']:,.0f} trip ends {info['floor_cells']:,} cells joined a busier one "
                 f"({info['floor_parts_alone']:,} zone parts with none became one cell), moving {info['floor_ends']:.2%} of trip "
                 f"ends {info['floor_mean_m']:,.0f} m on average (half under {info['floor_p50_m']:,.0f} m, 95% under "
                 f"{info['floor_p95_m']:,.0f} m, at most {info['floor_max_m']:,.0f} m)")
    if stats:
        print(line)
        return info
    block = {"task": "T-044, T-466, T-510" + (", T-850b" if "floor_cells" in info else ""), "rule": "res-8 parents under the threshold of weekday trip ends (commutes and "
             "everyday outings, both ends) merge, children within one zone together, at their trip-end-weighted centre; "
             "in a busier parent, touching children within one zone each under the sparse-child threshold merge "
             "together, with their busiest child's index; a cell under that threshold left alone joins the touching "
             "merged cell of its zone with the fewest trip ends, or merges with touching lone cells of its zone"
             + ("; with walk parts, 'zone' is one walk part of a zone (T-725)" if "walk_part" in a else "")
             + ("; then a cell under the floor of trip ends joins the nearest cell of its zone at or over it, which keeps its "
                "index and place, or the zone's busiest cell where it has none (T-850b)" if "floor_cells" in info else ""),
             "threshold_trip_ends": THRESHOLD, **info,
             "source": stamp(src, city)}
    size9 = os.path.getsize(os.path.join(src, f"{city}.bin"))
    # the zones' land, which a merged cell's index no longer gives (the sim's zone density)
    header = json.loads(json.dumps(header))
    header["arrays"].append({"name": "zone_cells9", "dtype": "u32", "per": "zone"})
    a = dict(a, zone_cells9=np.bincount(a["zone"].astype(np.int64), minlength=header["zones"]).astype(np.uint32))
    # read_pack's invariants on the merged pack before it replaces the live one, so a bad merge rule
    # fails this step rather than the gate later (T-764)
    staged = os.path.join(WORK, "staged")
    os.makedirs(staged, exist_ok=True)
    size = write_pack(os.path.join(staged, city), header, a, cells, block)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            read_pack.check(city, staged)
    except Exception as e:  # as the gate's read_pack check counts it
        sys.exit(f"{city}: the merged pack fails read_pack ({type(e).__name__}: {e}); data/packs left as it was, "
                 f"the merged one in {staged}")
    for ext in (".bin", ".json"):
        os.replace(os.path.join(staged, city + ext), os.path.join(PACKS, city + ext))
    with open(os.path.join(WORK, f"{city}.merged.json"), "w") as f:
        json.dump(stamp(PACKS, city), f)
    print(f"{line}; .bin {size9 / 1e6:.2f} -> {size / 1e6:.2f} MB", flush=True)
    return info


def undo(city, force=False):
    for ext in (".json", ".bin"):
        if not os.path.exists(os.path.join(WORK, city + ext)):
            sys.exit(f"{city}: no res-9 copy in data/work/res9")
    if not force and stale(city):
        sys.exit(STALE.format(city=city) + "; --undo --force drops the merged pack's change")
    for ext in (".bin", ".json"):
        shutil.copyfile(os.path.join(WORK, city + ext), os.path.join(PACKS, city + ext + ".part"))
        os.replace(os.path.join(PACKS, city + ext + ".part"), os.path.join(PACKS, city + ext))
    print(f"{city}: res-9 pack back in data/packs")


def through(ids, template, workers=1, force=False):
    """Per id: --undo, the step `template` names ("<module> {id} <args>", run through world_run.py
    step), then merge again; on a failure no new id starts, and the failed pack's res-9 version stays in
    data/packs. Returns the failed ids."""
    import shlex
    import subprocess
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_run.py")
    stop, lock, failed = threading.Event(), threading.Lock(), []

    def one(cid):
        if stop.is_set():
            return
        t, out = time.time(), []
        steps = [["merge_cells", cid, "--undo"] + (["--force"] if force else []),
                 [x.replace("{id}", cid) for x in shlex.split(template)],
                 ["merge_cells", cid]]
        for st in steps:
            r = subprocess.run([sys.executable, "-I", runner, "step"] + st, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            out.append(f"$ {' '.join(st)}\n{r.stdout}{r.stderr}")
            if r.returncode:
                stop.set()
                break
        ok = not r.returncode
        with lock:
            if not ok:
                failed.append(cid)
            print("".join(out).rstrip() + f"\n{'DONE' if ok else 'FAILED'} {cid} {time.time() - t:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, ids))
    print(f"through: {len(ids)} ids, {len(failed)} failed {' '.join(failed)}"
          + ("; stopped at the first failure, ids not yet started left as they were" if failed else ""), flush=True)
    return failed


def registered():
    with open(REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    return [c["id"] for c in reg["cities"] if os.path.exists(os.path.join(PACKS, f"{c['id']}.bin"))]


if __name__ == "__main__":
    args = sys.argv[1:]
    child = floor = None
    if "--child" in args:  # --stats only: another sparse-child threshold
        i = args.index("--child")
        child = float(args.pop(i + 1))
        args.pop(i)
    if "--floor" in args:
        i = args.index("--floor")
        floor = float(args.pop(i + 1))
        args.pop(i)
    tmpl = workers = None
    if "--through" in args:
        i = args.index("--through")
        tmpl = args.pop(i + 1)
        args.pop(i)
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args.pop(i + 1))
        args.pop(i)
    ids = registered() if "--all" in args else [x for x in args if not x.startswith("--")]
    if not ids:
        sys.exit(__doc__)
    if tmpl is not None:
        sys.exit(1 if through(ids, tmpl, workers or 1, force="--force" in args) else 0)
    if child is not None and "--stats" not in args:
        sys.exit("--child is for --stats")
    if floor is not None and "--stats" not in args and floor != (FLOOR or 0):
        sys.exit("--floor other than FLOOR is for --stats: nocar.py remerges by FLOOR")
    for cid in ids:
        if "--undo" in args:
            undo(cid, force="--force" in args)
        else:
            run(cid, stats="--stats" in args, child=child, floor=floor)
