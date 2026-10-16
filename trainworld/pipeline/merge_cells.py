"""T-044: merge sparse res-9 cells into their res-8 parent, the last step of every pack.

    python pipeline/merge_cells.py <id> [<id> ...]   merge these packs
    python pipeline/merge_cells.py --all             every registered pack in data/packs
    python pipeline/merge_cells.py <id> --stats      counts only, nothing written
    python pipeline/merge_cells.py <id> --stats --child 500   the same with another sparse-child threshold
    python pipeline/merge_cells.py <id> --undo       put the res-9 pack back in data/packs

A res-8 parent whose present children make under THRESHOLD trip ends a weekday merges; only the
children inside one 2 km zone merge together (a parent cut by a zone edge gives one merged cell
per zone, all with the parent's index), so every zone's sums, the commute gravity and the routes
are unchanged. In a parent over THRESHOLD, children under CHILD trip ends that touch each other and
share a zone merge together (T-466), keeping the res-9 index of their busiest child. A merged cell
sums every per-cell weight and sits at its children's trip-end-weighted centre. A group stays
unmerged if that centre would leave its zone square or stand on water. The merged pack also carries
`zone_cells9`, each zone's res-9 cells, since a merged cell's index no longer gives its land.

The res-9 pack is kept in data/work/res9/<id>.json/.bin: the first merge of a pack copies it there,
and a rerun (or --undo) starts from it. Steps that rewrite a pack (enrichments, routes.py) run on
the res-9 pack: `--undo`, the step, then this again. Cells come out sorted by their index with the
resolution field cleared (a merged cell right after where its children were), then by zone.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import hashlib
import json
import shutil
import sys

import numpy as np

from read_pack import PACKS, ROOT, load_pack

WORK = os.path.join(ROOT, "data", "work", "res9")
REGISTRY = os.path.join(ROOT, "app", "src", "game", "registry.json")
THRESHOLD = 2500.0  # trip ends a weekday per res-8 parent (DECISIONS 2026-10-16, T-453)
CHILD = THRESHOLD / 2  # per res-9 child of a busier parent (T-466)
RES_MASK = np.uint64(0xF) << np.uint64(52)
INT_SAME = ("zone", "region")  # integer per-cell arrays, equal within a zone


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
    cx = ((mx32 - np.float32(x0)) / np.float32(zm)).astype(np.int64)
    cy = ((my32 - np.float32(y0)) / np.float32(zm)).astype(np.int64)
    off_sq = (cx != gx) | (cy != gy)
    wet = np.zeros(ng, bool)
    idx = np.nonzero(~off_sq & (True if cand is None else cand))[0]
    wet[idx] = on_water(header, a, mx32[idx].astype(np.float64), my32[idx].astype(np.float64))
    return mx32, my32, first, size, off_sq, wet


def merge(header, a, threshold=THRESHOLD, child=None):
    """The merged pack's cell arrays {name: array} and a summary. Two rules (docs/demand.md 4.1):
    a res-8 parent under `threshold` trip ends merges its children within each zone (T-044);
    in a busier parent, children under `child` trip ends that touch merge among themselves
    (T-466), the merged cell keeping the index of its busiest child."""
    child = CHILD if child is None else child
    n = header["cells"]
    assert header["h3_res"] == 9 and "merged" not in header, "merge from a res-9 pack"
    cell = [e for e in header["arrays"] if e.get("per") == "cell"]
    for e in cell:
        assert e["count"] == n, e["name"]
        if e["dtype"] not in ("f32", "f64") and e["name"] not in ("h3",) + INT_SAME:
            sys.exit(f"{header['city']}: per-cell array {e['name']} ({e['dtype']}) has no merge rule")
    h3 = a["h3"]
    par = (h3 & ~RES_MASK) | (np.uint64(8) << np.uint64(52)) | (np.uint64(7) << np.uint64(18))
    zone = a["zone"].astype(np.uint64)
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
    info = {"cells_before": n, "cells": m, "parents": int(want.sum()), "cells_merged": int(m1.sum()),
            "sparse_groups": int(ok.sum()), "sparse_cells_merged": int(m2.sum()),
            "sparse_child_trip_ends": float(child),
            "ends_merged": float(e[m1 | m2].sum() / max(e.sum(), 1e-30)),
            "kept_square": kept_sq + int(s_sq.sum()), "kept_water": kept_wet + int((~s_sq & s_wet).sum())}
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
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(h, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    return len(buf)


def res9_copy(city, copy=True):
    """Where the pack's res-9 version is: data/work/res9, copied there first if data/packs holds it."""
    with open(os.path.join(PACKS, f"{city}.json")) as f:
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
    elif written(city) != stamp(PACKS, city):
        sys.exit(f"{city}: the merged pack in data/packs was changed after merging (a step run on it?); "
                 f"the res-9 copy in data/work/res9 lacks that change, so neither is merged nor undone")
    return WORK


def stamp(d, city):
    return {"json_sha256": sha(os.path.join(d, f"{city}.json")), "bin_sha256": sha(os.path.join(d, f"{city}.bin"))}


def written(city):
    p = os.path.join(WORK, f"{city}.merged.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def run(city, stats=False, child=None):
    src = res9_copy(city, copy=not stats)
    header, a = load_pack(city, src)
    cells, info = merge(header, a, child=child)
    line = (f"{city}: {info['cells_before']:,} -> {info['cells']:,} cells ({1 - info['cells'] / info['cells_before']:.0%} fewer), "
            f"{info['parents']:,} merged parents from {info['cells_merged']:,}, {info['sparse_groups']:,} merged siblings "
            f"from {info['sparse_cells_merged']:,}, {info['ends_merged']:.1%} of trip ends; "
            f"left alone: {info['kept_square']} off their square, {info['kept_water']} on water")
    if stats:
        print(line)
        return info
    block = {"task": "T-044, T-466", "rule": "res-8 parents under the threshold of weekday trip ends (commutes and everyday "
             "outings, both ends) merge, children within one zone together, at their trip-end-weighted centre; in a "
             "busier parent, touching children within one zone each under the sparse-child threshold merge together, "
             "with their busiest child's index",
             "threshold_trip_ends": THRESHOLD, **info,
             "source": stamp(src, city)}
    size9 = os.path.getsize(os.path.join(src, f"{city}.bin"))
    # the zones' land, which a merged cell's index no longer gives (the sim's zone density)
    header = json.loads(json.dumps(header))
    header["arrays"].append({"name": "zone_cells9", "dtype": "u32", "per": "zone"})
    a = dict(a, zone_cells9=np.bincount(a["zone"].astype(np.int64), minlength=header["zones"]).astype(np.uint32))
    size = write_pack(os.path.join(PACKS, city), header, a, cells, block)
    with open(os.path.join(WORK, f"{city}.merged.json"), "w") as f:
        json.dump(stamp(PACKS, city), f)
    print(f"{line}; .bin {size9 / 1e6:.2f} -> {size / 1e6:.2f} MB", flush=True)
    return info


def undo(city):
    for ext in (".json", ".bin"):
        if not os.path.exists(os.path.join(WORK, city + ext)):
            sys.exit(f"{city}: no res-9 copy in data/work/res9")
    for ext in (".bin", ".json"):
        shutil.copyfile(os.path.join(WORK, city + ext), os.path.join(PACKS, city + ext + ".part"))
        os.replace(os.path.join(PACKS, city + ext + ".part"), os.path.join(PACKS, city + ext))
    print(f"{city}: res-9 pack back in data/packs")


def registered():
    with open(REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    return [c["id"] for c in reg["cities"] if os.path.exists(os.path.join(PACKS, f"{c['id']}.bin"))]


if __name__ == "__main__":
    args = sys.argv[1:]
    child = None
    if "--child" in args:  # --stats only: another sparse-child threshold
        i = args.index("--child")
        child = float(args.pop(i + 1))
        args.pop(i)
    ids = registered() if "--all" in args else [x for x in args if not x.startswith("--")]
    if not ids:
        sys.exit(__doc__)
    if child is not None and "--stats" not in args:
        sys.exit("--child is for --stats")
    for cid in ids:
        if "--undo" in args:
            undo(cid)
        else:
            run(cid, stats="--stats" in args, child=child)
