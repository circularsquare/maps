"""River steps for the canoe at L1, L3 and L10 (docs/canoe.md 4).

    python tools/canoe/rivers.py reaches   # RiverATLAS gdb -> work/reaches.npz (Q >= 1) + work/all_lines.npz   ~3.5 min
    python tools/canoe/rivers.py lakes     # HydroLAKES -> work/lakes_l1.npy (0, 1 lake, 2 reservoir) + reservoir pour points   ~4 min
    python tools/canoe/rivers.py snap      # OSM falls/dams/weirs/rapids + reservoir pour points -> work/barriers.npz   ~1 min
    python tools/canoe/rivers.py build     # -> data/canoe/rivers.bin.gz, data/canoe/rivers/l3, l1, index.json   ~4 min, ~8 GB
    python tools/canoe/rivers.py index     # put the rivers key back into data/canoe/index.json

Work files go in data/raw/rivers_work/. Inputs: data/raw/riveratlas/RiverATLAS_v10.gdb (or the zip,
https://ndownloader.figshare.com/files/20087321; only `reaches` reads it), maps/data/HydroLAKES_polys_v10_shp,
data/raw/osm_barriers/*.tsv (rivers_osm.py), data/surface/elev.tif (GMTED), the speed grid and tiles
(water mask per level).
"""
import glob
import gzip
import json
import os
import sys
import time
from array import array

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
WORK = os.path.join(RAW, "rivers_work")
OUT = os.path.join(HERE, "data", "canoe")
LAKES = os.path.join(os.path.dirname(HERE), "data", "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp")

NORTH, WEST = 85.0, -180.0
L1W, L1H = 43200, 17400
TILE = {1: 600, 3: 200}
QMIN = 1.0
SUB = 0.002                 # degrees, longest sub-segment (a quarter of an L1 cell)
SNAP_KM = 0.5               # fall or rapids to river line, plus half the river's width
SNAP_DAM_KM = 0.3           # dams and weirs: tighter, as many OSM dams are on ponds beside rivers

# step classes (docs/canoe.md 4), as js/canoedata.js reads them
FLAT, RAPIDS, UNRUNNABLE, FALL, DAM, WEIR, RESERVOIR = 0, 1, 2, 3, 4, 5, 6
H_DAM, H_WEIR, O_RAPIDS = 1, 2, 4     # OSM kinds before snapping
NORTH_LAT = 60.0            # HydroSHEDS v1 north of here is HYDRO1k: gradient from GMTED (north_gradient)
NORTH_WIN_KM = 10.0
Q_BIG = 1000.0             # m3/s: on rivers this big, 2 m/km is too steep and OSM rapids are a portage
# the 8 neighbour moves in js/search.js order: (dc, dr), dr > 0 is south
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gdb_path():
    d = os.path.join(RAW, "riveratlas")
    hits = glob.glob(os.path.join(d, "**", "*.gdb"), recursive=True)
    if hits:
        return hits[0]
    z = os.path.join(d, "RiverATLAS_Data_v10.gdb.zip")
    return "/vsizip/" + z.replace("\\", "/") + "/RiverATLAS_v10.gdb"


# ---------------------------------------------------------------- reaches

def stage_reaches():
    import pyogrio
    import shapely
    path = gdb_path()
    info = pyogrio.read_info(path)
    log("reading", path, info["features"], "reaches")
    cols = ["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM", "DIST_UP_KM", "dis_m3_pyr", "dis_m3_pmn", "sgr_dk_rav"]
    meta, _, geom, fields = pyogrio.raw.read(path, columns=cols)
    f = dict(zip(meta["fields"], fields))
    log("read; parsing geometry")
    lines = shapely.from_wkb(geom)
    del geom
    q = np.asarray(f["dis_m3_pyr"], dtype=np.float64)
    co, idx = shapely.get_coordinates(lines, return_index=True)
    del lines
    n = len(q)
    counts = np.bincount(idx, minlength=n)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    log("coords", len(co))
    # every reach, float32, for snapping barriers to the nearest river of any size
    np.savez(os.path.join(WORK, "all_lines.npz"), co=co.astype(np.float32), off=off, q=q.astype(np.float32))
    keep = np.nonzero(q >= QMIN)[0]
    cnt = counts[keep]
    starts = off[keep]
    rep = np.repeat(starts - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
    sel = np.arange(cnt.sum()) + rep
    off2 = np.zeros(len(keep) + 1, np.int64)
    np.cumsum(cnt, out=off2[1:])
    np.savez(os.path.join(WORK, "reaches.npz"), co=co[sel], off=off2, all_index=keep,
             id=np.asarray(f["HYRIV_ID"])[keep], next_down=np.asarray(f["NEXT_DOWN"])[keep],
             length_km=np.asarray(f["LENGTH_KM"], np.float64)[keep],
             dist_up=np.asarray(f["DIST_UP_KM"], np.float64)[keep],
             q=q[keep], qmin=np.asarray(f["dis_m3_pmn"], np.float64)[keep],
             sgr=np.asarray(f["sgr_dk_rav"], np.float64)[keep])
    tot = np.asarray(f["LENGTH_KM"], np.float64)
    log("reaches %d, %.2f M km; Q >= %g: %d, %.2f M km" % (n, tot.sum() / 1e6, QMIN, len(keep), tot[keep].sum() / 1e6))


# ---------------------------------------------------------------- lakes

def stage_lakes():
    import pyogrio
    import shapely
    from rasterio import features
    from rasterio.transform import from_origin
    meta, _, geom, fields = pyogrio.raw.read(LAKES, columns=["Hylak_id", "Lake_type", "Pour_long", "Pour_lat", "Lake_area"])
    f = dict(zip(meta["fields"], fields))
    typ = np.asarray(f["Lake_type"])
    log("lakes", len(typ), "reservoirs", int((typ == 2).sum()))
    polys = shapely.from_wkb(geom)
    del geom
    out = np.zeros((L1H, L1W), np.uint8)
    tr = from_origin(WEST, NORTH, 1 / 120, 1 / 120)
    for val, mask in ((1, typ != 2), (2, typ == 2)):
        shapes = ((p, val) for p in polys[mask])
        features.rasterize(shapes, out=out, transform=tr, merge_alg=features.MergeAlg.replace)
        log("rasterised", val, int((out == val).sum()), "pixels")
    np.save(os.path.join(WORK, "lakes_l1.npy"), out)
    res = typ == 2
    np.savez(os.path.join(WORK, "reservoirs.npz"), id=np.asarray(f["Hylak_id"])[res],
             lon=np.asarray(f["Pour_long"])[res], lat=np.asarray(f["Pour_lat"])[res],
             area=np.asarray(f["Lake_area"])[res])


# ---------------------------------------------------------------- snap

def read_osm():
    lon, lat, kind = [], [], []
    seen = set()
    for p in sorted(glob.glob(os.path.join(RAW, "osm_barriers", "*.tsv"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6 or not parts[2]:
                    continue
                key = (parts[0], parts[1])
                if key in seen:
                    continue
                seen.add(key)
                ww, nat = parts[4], parts[5]
                k = (FALL if (ww == "waterfall" or nat == "waterfall") else H_DAM if ww == "dam" else
                     H_WEIR if ww == "weir" else O_RAPIDS if ww == "rapids" else None)
                if k is None:
                    continue
                lon.append(float(parts[3]))
                lat.append(float(parts[2]))
                kind.append(k)
    return np.array(lon), np.array(lat), np.array(kind, np.int8)


def stage_snap():
    import shapely
    a = np.load(os.path.join(WORK, "all_lines.npz"))
    co, off, qall = a["co"].astype(np.float64), a["off"], a["q"]
    n = len(off) - 1
    lines = shapely.linestrings(co, indices=np.repeat(np.arange(n), np.diff(off)))
    log("lines", n)
    tree = shapely.STRtree(lines)
    olon, olat, okind = read_osm()
    # src: 0 fall, 1 dam, 2 weir, 3 reservoir pour point (a dam), 4 OSM rapids
    src = np.select([okind == FALL, okind == H_DAM, okind == H_WEIR], [0, 1, 2], 4)
    rv = np.load(os.path.join(WORK, "reservoirs.npz"))
    lon = np.concatenate([olon, rv["lon"]])
    lat = np.concatenate([olat, rv["lat"]])
    src = np.concatenate([src, np.full(len(rv["lon"]), 3)])
    log("barriers: falls %d, dams %d, weirs %d, reservoir pour points %d, rapids %d" %
        tuple((src == s).sum() for s in range(5)))
    pts = shapely.points(lon, lat)
    pi, li = tree.query_nearest(pts, max_distance=0.03, all_matches=False)
    # true distance in km, roughly, from the nearest point on the line
    near = shapely.shortest_line(pts[pi], lines[li])
    c = shapely.get_coordinates(near).reshape(-1, 2, 2)
    dx = (c[:, 1, 0] - c[:, 0, 0]) * 111.32 * np.cos(np.radians(c[:, 0, 1]))
    dy = (c[:, 1, 1] - c[:, 0, 1]) * 110.57
    dkm = np.hypot(dx, dy)
    q = qall[li]
    width_km = 7.2 * np.sqrt(np.maximum(q, 0)) / 1000
    base = np.select([src[pi] == 3, (src[pi] == 1) | (src[pi] == 2)], [1.0, SNAP_DAM_KM], SNAP_KM)
    tol = base + width_km / 2
    ok = (dkm <= tol) & (q >= QMIN)
    pi, li, dkm = pi[ok], li[ok], dkm[ok]
    along = shapely.line_locate_point(lines[li], pts[pi])
    r = np.load(os.path.join(WORK, "reaches.npz"))
    pos = np.full(n, -1, np.int64)
    pos[r["all_index"]] = np.arange(len(r["all_index"]))
    np.savez(os.path.join(WORK, "barriers.npz"), reach=pos[li], along=along, src=src[pi],
             lon=lon[pi], lat=lat[pi], dkm=dkm)
    for s, name in enumerate(["falls", "dams", "weirs", "reservoir dams", "rapids"]):
        log("%s on a paddleable river: %d of %d" % (name, (src[pi] == s).sum(), (src == s).sum()))


# ---------------------------------------------------------------- build

def natural_class(q, g):
    """g in m/km."""
    g = np.where(np.isfinite(g) & (g > 0), g, 0)
    lim = np.where(q >= Q_BIG, 2.0, 5.0)
    return np.where(g > lim, UNRUNNABLE, np.where(g >= 1.0, RAPIDS, FLAT)).astype(np.uint8)


def north_gradient(r, down, order):
    """Reach gradient (m/km) north of 60N, where RiverATLAS's sgr_dk_rav comes from HYDRO1k and is
    noise (its drops summed down the lower Yukon come to ~13 km). From GMTED instead: the lowest
    elevation in the 3 x 3 pixels round each reach end, made never to rise downstream, and the drop
    over at least NORTH_WIN_KM downstream of the reach start."""
    import rasterio
    from scipy.ndimage import minimum_filter
    co, off, L = r["co"], r["off"], r["length_km"]
    rows = int((NORTH - NORTH_LAT) * 120) + 2
    with rasterio.open(os.path.join(HERE, "data", "surface", "elev.tif")) as ds:
        z = ds.read(1, window=((0, rows), (0, L1W))).astype(np.float32)
    z[z == -32768] = np.nan
    z = minimum_filter(np.nan_to_num(z, nan=1e4), size=3, mode="wrap")

    def sample(p):
        c = np.floor((p[:, 0] - WEST) * 120).astype(np.int64) % L1W
        rr = np.floor((NORTH - p[:, 1]) * 120).astype(np.int64)
        out = np.full(len(p), np.inf)
        ok = (rr >= 0) & (rr < rows)
        out[ok] = z[rr[ok], c[ok]]
        return out

    zs_raw, ze_raw = sample(co[off[:-1]]), sample(co[off[1:] - 1])
    R = len(L)
    up_min = np.full(R, np.inf)
    zs, ze = np.empty(R), np.empty(R)
    dl = down.tolist()
    for k in order.tolist():
        a = min(zs_raw[k], up_min[k])
        b = min(ze_raw[k], a)
        zs[k], ze[k] = a, b
        d = dl[k]
        if d >= 0 and b < up_min[d]:
            up_min[d] = b
    north = np.nonzero(co[off[:-1], 1] > NORTH_LAT)[0]
    g = np.full(R, np.nan)
    Ll = L.tolist()
    for k0 in north.tolist():
        k, dist = k0, 0.0
        while True:
            dist += Ll[k]
            zend = ze[k]
            if dist >= NORTH_WIN_KM or dl[k] < 0:
                break
            k = dl[k]
        if np.isfinite(zs[k0]) and np.isfinite(zend) and dist > 0:
            g[k0] = (zs[k0] - zend) / dist
    log("north of %gN: %d reaches get a GMTED gradient" % (NORTH_LAT, np.isfinite(g).sum()))
    return g


def densify(r):
    """Sub-segments of every paddleable reach: midpoint, km, reach, degrees along."""
    co, off = r["co"], r["off"]
    R = len(off) - 1
    P = len(co)
    npts = np.diff(off)
    seg_ok = np.ones(P - 1, bool)
    seg_ok[off[1:-1] - 1] = False
    s0 = co[:-1][seg_ok]
    s1 = co[1:][seg_ok]
    seg_reach = np.repeat(np.arange(R), npts - 1)
    d = s1 - s0
    deg = np.hypot(d[:, 0], d[:, 1])
    km = np.hypot(d[:, 0] * 111.32 * np.cos(np.radians((s0[:, 1] + s1[:, 1]) / 2)), d[:, 1] * 110.57)
    ns = np.maximum(1, np.ceil(deg / SUB)).astype(np.int64)
    rep = np.repeat(np.arange(len(ns)), ns)
    j = np.arange(len(rep)) - np.repeat(np.cumsum(ns) - ns, ns)
    t = (j + 0.5) / ns[rep]
    mx = s0[rep, 0] + t * d[rep, 0]
    my = s0[rep, 1] + t * d[rep, 1]
    sub_km = km[rep] / ns[rep]
    sub_deg = deg[rep] / ns[rep]
    sub_reach = seg_reach[rep]
    geo = np.bincount(sub_reach, weights=sub_km, minlength=R)
    scale = np.where(geo > 0, r["length_km"] / np.where(geo > 0, geo, 1), 1.0)
    sub_km = (sub_km * scale[sub_reach]).astype(np.float32)
    first = np.searchsorted(sub_reach, np.arange(R))
    return mx, my, sub_km, sub_reach, sub_deg, first


def load_water(level):
    if level == 10:
        g = np.frombuffer(gzip.open(os.path.join(HERE, "data", "grid.bin.gz")).read(), np.uint8)
        return (g == 0).reshape(L1H // 10, L1W // 10)
    size = TILE[level]
    W, H = L1W // level, L1H // level
    water = np.ones((H, W), bool)
    for p in glob.glob(os.path.join(HERE, "data", "tiles", "l%d" % level, "*.bin.gz")):
        tr, tc = map(int, os.path.basename(p).split(".")[0].split("_"))
        t = np.frombuffer(gzip.open(p).read(), np.uint8).reshape(size, size)
        water[tr * size:(tr + 1) * size, tc * size:(tc + 1) * size] = t == 0
    return water


def _merge(e, e2):
    e[1] += e2[1]; e[2] += e2[2]; e[5] += e2[5]; e[6] += e2[6]
    if e2[3] > e[3]: e[3] = e2[3]
    if e2[4] > e[4]: e[4] = e2[4]


def walk_reach(r, k0, nk, C, L, V, N, Hm, LK, RS, W, corner_km, carry, downl, ql,
               out_a, out_b, out_len, out_vlen, out_nat, out_hum, out_lake, out_res, folded, cornered):
    """One reach's runs of cells to steps. An entry is [cell, km, v*km, natural, human, lake km, reservoir km];
    step i leaves entry i and holds its length. The last entry goes on to the downstream reach."""
    path = []
    where = {}
    for k in range(k0, k0 + nk):
        c = C[k]
        j = where.get(c)
        if j is not None:
            # back to a cell already on this reach's path: fold the loop into it
            e = path[j]
            for e2 in path[j + 1:]:
                _merge(e, e2)
                del where[e2[0]]
            folded += len(path) - j - 1
            del path[j + 1:]
            _merge(e, [c, L[k], V[k], N[k], Hm[k], LK[k], RS[k]])
        else:
            where[c] = len(path)
            path.append([c, L[k], V[k], N[k], Hm[k], LK[k], RS[k]])
        # a cell only clipped at a corner between two diagonal neighbours: step diagonally
        while len(path) >= 3 and path[-2][1] < corner_km:
            a0, a2 = path[-3][0], path[-1][0]
            ddr = a2 // W - a0 // W
            ddc = (a2 % W - a0 % W + W // 2) % W - W // 2
            if not (-1 <= ddr <= 1 and -1 <= ddc <= 1):
                break
            _merge(path[-3], path[-2])
            del where[path[-2][0]]
            del path[-2]
            where[path[-1][0]] = len(path) - 1
            cornered += 1
    cr = carry.pop(r, None)
    if not path:
        if cr is None:
            return folded, cornered
        path = [cr[1]]
    elif cr is not None and cr[1][0] == path[0][0]:
        _merge(path[0], cr[1])
    for i in range(len(path) - 1):
        e = path[i]
        out_a.append(e[0]); out_b.append(path[i + 1][0]); out_len.append(e[1]); out_vlen.append(e[2])
        out_nat.append(e[3]); out_hum.append(e[4]); out_lake.append(e[5]); out_res.append(e[6])
    d = downl[r]
    if d >= 0:
        prev = carry.get(d)
        if prev is None or ql[r] > prev[0]:
            carry[d] = (ql[r], list(path[-1]))
    return folded, cornered


def steps_for_level(level, sub, reach_info):
    mx, my, sub_km, sub_reach, sub_v, sub_nat, sub_hum, sub_lake = sub
    cs = level / 120.0
    W, H = L1W // level, L1H // level
    col = np.floor((mx - WEST) / cs).astype(np.int64) % W
    row = np.floor((NORTH - my) / cs).astype(np.int64)
    ok = (row >= 0) & (row < H)
    idx = np.nonzero(ok)[0]
    cell = row[idx] * W + col[idx]
    rch = sub_reach[idx]
    brk = np.ones(len(idx), bool)
    brk[1:] = (cell[1:] != cell[:-1]) | (rch[1:] != rch[:-1])
    st = np.nonzero(brk)[0]
    run_cell = cell[st]
    run_reach = rch[st]
    km = sub_km[idx].astype(np.float64)
    run_len = np.add.reduceat(km, st)
    run_vlen = np.add.reduceat(km * sub_v[idx], st)
    run_nat = np.maximum.reduceat(sub_nat[idx], st)
    run_hum = np.maximum.reduceat(sub_hum[idx], st)           # 0 none, 1 weir, 2 dam (priority order)
    lk = sub_lake[idx]
    run_lake = np.add.reduceat(km * (lk == 1), st)
    run_res = np.add.reduceat(km * (lk == 2), st)
    log("L%d: %d runs" % (level, len(st)))
    del col, row, ok, idx, cell, rch, brk, km, lk
    water = load_water(level).ravel()

    q, down, order = reach_info
    R = len(q)
    rstart_all = np.searchsorted(run_reach, np.arange(R))
    rcount_all = np.searchsorted(run_reach, np.arange(R), side="right") - rstart_all
    downl = down.tolist()
    ql = q.tolist()
    carry = {}
    out_a, out_b, out_nat, out_hum = array("q"), array("q"), array("B"), array("B")
    out_len, out_vlen, out_lake, out_res = array("d"), array("d"), array("d"), array("d")
    folded = cornered = 0
    corner_km = 0.3 * 0.926 * level
    CH = 200000
    for c0 in range(0, R, CH):
        # the runs of a chunk of reaches in processing order, as Python lists (all at once is too big)
        oc = order[c0:c0 + CH]
        cnt = rcount_all[oc]
        loc0 = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        ix = np.arange(cnt.sum()) + np.repeat(rstart_all[oc] - loc0, cnt)
        C, L, V, N, Hm, LK, RS = (run_cell[ix].tolist(), run_len[ix].tolist(), run_vlen[ix].tolist(),
                                  run_nat[ix].tolist(), run_hum[ix].tolist(), run_lake[ix].tolist(),
                                  run_res[ix].tolist())
        for r, k0, nk in zip(oc.tolist(), loc0.tolist(), cnt.tolist()):
            folded, cornered = walk_reach(r, k0, nk, C, L, V, N, Hm, LK, RS, W, corner_km, carry, downl, ql,
                                          out_a, out_b, out_len, out_vlen, out_nat, out_hum, out_lake, out_res,
                                          folded, cornered)
    a = np.frombuffer(out_a, np.int64)
    b = np.frombuffer(out_b, np.int64)
    ln = np.frombuffer(out_len, np.float64)
    vl = np.frombuffer(out_vlen, np.float64)
    nat = np.frombuffer(out_nat, np.uint8)
    hum = np.frombuffer(out_hum, np.uint8)
    lake = np.frombuffer(out_lake, np.float64)
    res = np.frombuffer(out_res, np.float64)
    ra, ca = a // W, a % W
    rb, cb = b // W, b % W
    dr = rb - ra
    dc = (cb - ca + W // 2) % W - W // 2
    adj = (np.abs(dr) <= 1) & (np.abs(dc) <= 1) & ((dr != 0) | (dc != 0))
    both_water = water[a] & water[b]
    in_lake = both_water & (lake >= 0.5 * ln)
    keep = adj & ~in_lake
    resv = both_water & (res >= 0.5 * ln) & (hum == 0)
    # one class per step: fall, dam, weir, unrunnable, reservoir, rapids, flat. A dam or weir goes
    # before the gradient, since the DEM's drop at a dam is mostly the dam itself.
    cls = np.select([nat == FALL, hum == 2, hum == 1, nat == UNRUNNABLE, resv],
                    [FALL, DAM, WEIR, UNRUNNABLE, RESERVOIR], nat.astype(np.int64)).astype(np.uint8)
    dirs = np.full(len(a), 255, np.int64)
    for k, (mc, mr) in enumerate(MOVES):
        dirs[(dc == mc) & (dr == mr)] = k
    v = np.where(ln > 0, vl / np.where(ln > 0, ln, 1), 0)
    k = np.nonzero(keep)[0]
    # two rivers on the same step (a tributary beside its main stem in one coarse cell): keep one,
    # an open one first, then a dam/weir/reservoir, then a closed one; among those the bigger river
    rank = np.select([cls <= RAPIDS, (cls == RESERVOIR) | (cls == DAM) | (cls == WEIR)], [0, 1], 2)
    o = k[np.lexsort((-v[k], rank[k], dirs[k], a[k]))]
    first = np.ones(len(o), bool)
    first[1:] = (a[o][1:] != a[o][:-1]) | (dirs[o][1:] != dirs[o][:-1])
    k = np.sort(o[first])
    stats = {"runs": int(len(st)), "folded": int(folded), "cornered": int(cornered), "steps": int(len(k)),
             "duplicates_dropped": int((~first).sum()),
             "not_adjacent": int((~adj).sum()), "in_lake_dropped": int((adj & in_lake).sum()),
             "km": float(ln[k].sum())}
    return dict(cell=a[k], row=ra[k], col=ca[k], dir=dirs[k].astype(np.uint8),
                length=np.clip(np.rint(ln[k] * 100), 1, 65535).astype(np.uint16),
                v=np.clip(np.rint(v[k] * 100), 0, 255).astype(np.uint8),
                cls=cls[k], stats=stats, W=W)


def pack(cell, length, dirs, v, cls):
    o = np.lexsort((dirs, cell))
    n = np.array([len(cell)], "<u4")
    return gzip.compress(n.tobytes() + cell[o].astype("<u4").tobytes() + length[o].astype("<u2").tobytes()
                         + dirs[o].tobytes() + v[o].tobytes() + cls[o].tobytes(), compresslevel=9)


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    return len(data)


def merge_index():
    """Put rivers/index.json under the "rivers" key of the shared data/canoe/index.json."""
    with open(os.path.join(OUT, "rivers", "index.json")) as fh:
        rivers = json.load(fh)
    ip = os.path.join(OUT, "index.json")
    index = {}
    if os.path.exists(ip):
        with open(ip) as fh:
            index = json.load(fh)
    index["rivers"] = rivers
    write(ip, json.dumps(index, separators=(",", ":")).encode())
    log("rivers entry written to", ip)


def stage_build(levels=(10, 3, 1)):
    r = dict(np.load(os.path.join(WORK, "reaches.npz")))
    R = len(r["q"])
    log("reaches", R)
    mx, my, sub_km, sub_reach, sub_deg, first = densify(r)
    log("sub-segments", len(mx))
    q = r["q"]
    sub_v = (0.51 * q ** 0.11)[sub_reach].astype(np.float32)
    idpos = {int(i): k for k, i in enumerate(r["id"])}
    down = np.array([idpos.get(int(d), -1) for d in r["next_down"]], np.int64)
    del idpos
    order = np.argsort(r["dist_up"], kind="stable")
    grad = r["sgr"] / 10.0                                   # dm/km -> m/km
    gn = north_gradient(r, down, order)
    grad = np.where(np.isfinite(gn), gn, grad)
    np.save(os.path.join(WORK, "gradient.npy"), grad)
    sub_nat = natural_class(q, grad)[sub_reach]
    sub_hum = np.zeros(len(mx), np.uint8)
    lakes = np.load(os.path.join(WORK, "lakes_l1.npy"), mmap_mode="r")
    lc = np.floor((mx - WEST) * 120).astype(np.int64) % L1W
    lr = np.clip(np.floor((NORTH - my) * 120).astype(np.int64), 0, L1H - 1)
    sub_lake = np.asarray(lakes[lr, lc])
    del lc, lr
    b = np.load(os.path.join(WORK, "barriers.npz"))
    gcum = np.cumsum(sub_deg)
    start_cum = gcum[first] - sub_deg[first]
    last = np.append(first[1:], len(mx)) - 1
    bi = np.searchsorted(gcum, start_cum[b["reach"]] + b["along"])
    bi = np.clip(bi, first[b["reach"]], last[b["reach"]])
    src = b["src"]
    sub_nat[bi[src == 0]] = FALL
    sub_hum[bi[src == 2]] = np.maximum(sub_hum[bi[src == 2]], 1)
    sub_hum[bi[(src == 1) | (src == 3)]] = 2
    ri = bi[src == 4]
    big = q[sub_reach[ri]] >= Q_BIG
    sub_nat[ri[big]] = np.maximum(sub_nat[ri[big]], UNRUNNABLE)
    sub_nat[ri[~big]] = np.maximum(sub_nat[ri[~big]], RAPIDS)
    log("barriers placed: falls %d, dams %d, weirs %d, rapids %d (%d on rivers over %g m3/s)" %
        ((src == 0).sum(), ((src == 1) | (src == 3)).sum(), (src == 2).sum(), len(ri), big.sum(), Q_BIG))

    sub = (mx, my, sub_km, sub_reach, sub_v, sub_nat, sub_hum, sub_lake)
    summary = {"q_min": QMIN, "reaches": int(R), "km": float(r["length_km"].sum()), "levels": {}}
    index = {}
    for level in levels:
        t0 = time.time()
        s = steps_for_level(level, sub, (q, down, order))
        st = s["stats"]
        if level == 10:
            st["bytes"] = write(os.path.join(OUT, "rivers.bin.gz"), pack(s["cell"], s["length"], s["dir"], s["v"], s["cls"]))
            st["files"] = 1
        else:
            size = TILE[level]
            tr, tc = s["row"] // size, s["col"] // size
            local = (s["row"] % size) * size + s["col"] % size
            tid = tr * 72 + tc
            o = np.argsort(tid, kind="stable")
            tids, starts = np.unique(tid[o], return_index=True)
            ends = np.append(starts[1:], len(o))
            nbytes = 0
            for t, a0, a1 in zip(tids.tolist(), starts.tolist(), ends.tolist()):
                k = o[a0:a1]
                nbytes += write(os.path.join(OUT, "rivers", "l%d" % level, "%d_%d.bin.gz" % (t // 72, t % 72)),
                                pack(local[k], s["length"][k], s["dir"][k], s["v"][k], s["cls"][k]))
            st["bytes"] = nbytes
            st["files"] = len(tids)
            index[str(level)] = {"tile": size, "path": "rivers/l%d" % level, "tiles": tids.tolist()}
        cls = s["cls"]
        st["by_class"] = {int(c): int(n) for c, n in zip(*np.unique(cls, return_counts=True))}
        summary["levels"][str(level)] = st
        log("L%d: %d steps, %.2f M km, %.1f MB in %d files (%.0f s)" %
            (level, st["steps"], st["km"] / 1e6, st["bytes"] / 1e6, st["files"], time.time() - t0))
    if 1 in levels and 3 in levels:
        write(os.path.join(OUT, "rivers", "index.json"), json.dumps(index, separators=(",", ":")).encode())
        merge_index()
    with open(os.path.join(WORK, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


def main():
    os.makedirs(WORK, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    if stage == "reaches":
        stage_reaches()
    elif stage == "lakes":
        stage_lakes()
    elif stage == "snap":
        stage_snap()
    elif stage == "build":
        lv = tuple(int(x) for x in sys.argv[2:]) or (10, 3, 1)
        stage_build(lv)
    elif stage == "index":
        merge_index()
    else:
        print(__doc__)
        return 1
    log("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
