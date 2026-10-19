"""The real network as a routing graph, one file per noritetsu region (T-727b, notes/T-727.md).

    python realgraph.py jp us gb fr de      build these regions
    python realgraph.py --check jp ...      read the files back and print what they hold

Reads noritetsu's extract (`noritetsu/data/proc/<cc>/ways.pkl`, `rels.pkl`, `coords.npz`) and its
line model (`noritetsu/dist/data/<cc>/ways.json`); never writes there. Keeps exactly the ways the
game's real-network layer draws (`app/src/map/realNetwork.ts`, the same filter as noritetsu's
build_tiles.py, then SHOWN): subway, and rail that is main line or that a passenger route runs
over, each with some line in ways.json running over it. The track tool snaps only to these
(DECISIONS 2026-10-19, T-727).

Writes `data/packs/realgraph/<cc>.rgr`, gzip of the little-endian layout below; the app reads it
with `game/realGraph.ts` and a production build copies it (vite.config.ts). Coordinates are
degrees times 1e6 (0.1 m). Vertices are thinned to 3 m (Douglas-Peucker), chain ends kept.

    "TWRG" u32 version=1, u32 junctions J, u32 chains C, u32 vertices V, u32 shared S
    i32[J] lon, i32[J] lat                      junctions (OSM nodes where chains meet or end)
    u32[C] a, u32[C] b                          each chain's two junctions, a at its first vertex
    u8[C] kind (0 rail, 2 subway), u8[C] rank (0 main, 1 other), u8[C] flags (1 passenger route,
        2 bridge, 4 tunnel, 8 high speed)
    u32[C + 1] first vertex of each chain
    varint[2V] lon, lat, lon, lat, ...          zigzag LEB128; a chain's first vertex absolute,
                                                the rest as differences from the vertex before
    u32[S] junction, f64[S] OSM node id         chain ends this region shares with a
                                                neighbour's extract, to join their graphs
"""
import argparse
import gzip
import json
import os
import pickle
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = Path(__file__).resolve().parent
NORI = HERE.parents[1] / "noritetsu"
OUT = HERE.parent / "data" / "packs" / "realgraph"
VERSION = 1
THIN_M = 3.0
KIND = {"rail": 0, "subway": 2}
URBAN = {"light_rail", "subway", "tram", "monorail", "funicular"}


def rank_of(kind, tags):
    """noritetsu build_tiles.rank_of: 0 main, 1 branch or unstated, 2 industrial, 3 service."""
    if tags.get("service"):
        return 3
    usage = tags.get("usage")
    if usage == "tourism" and kind == "narrow_gauge":
        return 1
    if usage in ("industrial", "military", "tourism", "freight", "test", "distribution"):
        return 2
    if usage == "main":
        return 0
    if usage == "branch":
        return 1
    return 0 if kind in URBAN else 1


def thin(x, y, tol):
    n = x.size
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    st = [(0, n - 1)]
    while st:
        i, j = st.pop()
        if j <= i + 1:
            continue
        dx, dy = x[j] - x[i], y[j] - y[i]
        L2 = dx * dx + dy * dy
        sx, sy = x[i + 1:j] - x[i], y[i + 1:j] - y[i]
        t = np.clip((sx * dx + sy * dy) / L2, 0, 1) if L2 > 0 else np.zeros(sx.size)
        d = np.hypot(sx - t * dx, sy - t * dy)
        m = int(np.argmax(d))
        if d[m] > tol:
            keep[i + 1 + m] = True
            st += [(i, i + 1 + m), (i + 1 + m, j)]
    return keep


def varints(v):
    """Zigzag LEB128 of a signed integer array."""
    z = ((v << 1) ^ (v >> 63)).astype(np.uint64)
    out = bytearray()
    for x in z.tolist():
        while x >= 0x80:
            out.append((x & 0x7F) | 0x80)
            x >>= 7
        out.append(x)
    return bytes(out)


def unvarints(raw, o, n):
    """n zigzag LEB128 integers from raw[o:]; returns them and the offset after."""
    out = np.empty(n, np.int64)
    for i in range(n):
        x = shift = 0
        while True:
            c = raw[o]
            o += 1
            x |= (c & 0x7F) << shift
            shift += 7
            if c < 0x80:
                break
        out[i] = (x >> 1) ^ -(x & 1)
    return out, o


def region_boxes():
    with open(NORI / "dist" / "regions.json", encoding="utf-8") as f:
        return {cc: r["bbox"] for cc, r in json.load(f)["regions"].items()}


def neighbour_ids(cc, boxes):
    """OSM node ids in the extracts of regions whose box touches this one's."""
    w, s, e, n = boxes[cc]
    ids = []
    for o, (w2, s2, e2, n2) in boxes.items():
        if o == cc or w2 > e or e2 < w or s2 > n or n2 < s:
            continue
        p = NORI / "data" / "proc" / o / "coords.npz"
        if p.exists():
            ids.append(np.load(p)["id"])
    return np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)


def build(cc, boxes):
    t0 = time.time()
    d = NORI / "data" / "proc" / cc
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    with open(NORI / "dist" / "data" / cc / "ways.json", encoding="utf-8") as f:
        on_line = {int(w) for w in json.load(f)["ways"]}
    on_route = set()
    for tags, members in rels.values():
        if tags.get("type") == "route":
            for ty, ref, role in members:
                if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                    on_route.add(ref)
    del rels

    keep = []
    for wid, (tags, nodes) in ways.items():
        rw = tags.get("railway")
        if rw not in KIND or wid not in on_line:
            continue
        rank = rank_of(rw, tags)
        pax = wid in on_route
        if rank >= 2:
            if not pax:
                continue
            rank = 1
        if rw == "rail" and rank == 1 and not pax:
            continue  # the layer draws other rail only where a passenger route runs
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        ok = cid[pos] == nodes
        if not ok.all():
            idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
            starts, ends = idx[0::2], idx[1::2]
            if starts.size == 0:
                continue
            b = np.argmax(ends - starts)
            pos = pos[starts[b]:ends[b]]
        if pos.size < 2:
            continue
        flags = (1 if pax else 0) | (2 if tags.get("bridge") not in (None, "no") else 0) \
            | (4 if tags.get("tunnel") not in (None, "no") else 0) | (8 if tags.get("highspeed") == "yes" else 0)
        keep.append((pos.astype(np.int64), KIND[rw], rank, flags))
    del ways

    allpos = np.concatenate([k[0] for k in keep])
    uniq, counts = np.unique(allpos, return_counts=True)
    isj = np.zeros(cid.size, bool)
    isj[uniq[counts >= 2]] = True
    for k in keep:
        isj[k[0][0]] = isj[k[0][-1]] = True

    lon_all = cx / 1e7
    lat_all = cy / 1e7
    ca, cb, ck, cr, cf, verts = [], [], [], [], [], []
    for pos, kind, rank, flags in keep:
        cut = np.flatnonzero(isj[pos])
        for i in range(len(cut) - 1):
            seg = pos[cut[i]:cut[i + 1] + 1]
            la = lat_all[seg]
            x = lon_all[seg] * np.cos(np.radians(la)) * 111320.0
            y = la * 110950.0
            seg = seg[thin(x, y, THIN_M)]
            ca.append(seg[0])
            cb.append(seg[-1])
            ck.append(kind)
            cr.append(rank)
            cf.append(flags)
            verts.append(seg)
    jpos = np.unique(np.concatenate([ca, cb]))
    a = np.searchsorted(jpos, ca).astype(np.uint32)
    b = np.searchsorted(jpos, cb).astype(np.uint32)
    off = np.concatenate(([0], np.cumsum([v.size for v in verts]))).astype(np.uint32)
    flat = np.concatenate(verts)
    qlon = np.round(cx[flat] / 10).astype(np.int64)  # 1e7 -> 1e6
    qlat = np.round(cy[flat] / 10).astype(np.int64)
    dlon, dlat = np.diff(qlon, prepend=0), np.diff(qlat, prepend=0)
    dlon[off[:-1]], dlat[off[:-1]] = qlon[off[:-1]], qlat[off[:-1]]

    shared_ids = np.intersect1d(cid[jpos], neighbour_ids(cc, boxes))
    shared_j = np.searchsorted(cid[jpos], shared_ids).astype(np.uint32)

    J, C, V, S = jpos.size, a.size, flat.size, shared_ids.size
    body = b"".join([
        b"TWRG", struct.pack("<5I", VERSION, J, C, V, S),
        np.round(cx[jpos] / 10).astype("<i4").tobytes(), np.round(cy[jpos] / 10).astype("<i4").tobytes(),
        a.astype("<u4").tobytes(), b.astype("<u4").tobytes(),
        np.array(ck, np.uint8).tobytes(), np.array(cr, np.uint8).tobytes(), np.array(cf, np.uint8).tobytes(),
        off.astype("<u4").tobytes(), varints(np.column_stack([dlon, dlat]).ravel()),
        shared_j.astype("<u4").tobytes(), shared_ids.astype("<f8").tobytes(),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{cc}.rgr.tmp"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(body, 9))
    os.replace(tmp, OUT / f"{cc}.rgr")
    size = (OUT / f"{cc}.rgr").stat().st_size
    print(f"{cc}: {C:,} chains, {J:,} junctions, {V:,} vertices, {S} shared with neighbours; "
          f"{size / 1e6:.2f} MB in {time.time() - t0:.0f} s", flush=True)


def read(path):
    raw = gzip.decompress(Path(path).read_bytes())
    assert raw[:4] == b"TWRG", "not a real-network graph"
    ver, J, C, V, S = struct.unpack_from("<5I", raw, 4)
    o = 24
    def take(dt, n):
        nonlocal o
        a = np.frombuffer(raw, dt, n, o)
        o += a.nbytes
        return a
    g = {"version": ver}
    g["jlon"], g["jlat"] = take("<i4", J) / 1e6, take("<i4", J) / 1e6
    g["a"], g["b"] = take("<u4", C), take("<u4", C)
    g["kind"], g["rank"], g["flags"] = take("u1", C), take("u1", C), take("u1", C)
    g["off"] = take("<u4", C + 1)
    d, o = unvarints(raw, o, 2 * V)
    dlon, dlat = d[0::2], d[1::2]
    g["shared"], g["shared_id"] = take("<u4", S), take("<f8", S)
    assert o == len(raw), "trailing bytes"
    # undo the differences chain by chain: a running sum restarted at each chain's first vertex
    start = np.zeros(V, bool)
    start[g["off"][:-1]] = True
    grp = np.cumsum(start) - 1
    for d, k in ((dlon, "lon"), (dlat, "lat")):
        cs = np.cumsum(d)
        base = cs[g["off"][:-1]] - d[g["off"][:-1]]
        g[k] = (cs - base[grp]) / 1e6
    return g


def check(cc):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = read(OUT / f"{cc}.rgr")
    J = g["jlon"].size
    first, last = g["off"][:-1], g["off"][1:] - 1
    ends_ok = (np.abs(g["lon"][first] - g["jlon"][g["a"]]) < 1e-6).all() and (np.abs(g["lon"][last] - g["jlon"][g["b"]]) < 1e-6).all()
    la = np.radians(g["lat"])
    seg = np.hypot(np.diff(g["lon"]) * np.cos(la[1:]) * 111.32, np.diff(g["lat"]) * 110.95)
    seg[np.isin(np.arange(seg.size), last)] = 0
    km = seg.sum()
    n, lab = connected_components(coo_matrix((np.ones(g["a"].size), (g["a"], g["b"])), shape=(J, J)), directed=False)
    big = np.bincount(lab).max()
    print(f"{cc}: chain ends on their junctions {ends_ok}; {km:,.0f} km of track; {n:,} pieces, "
          f"largest {big / J:.0%} of junctions; subway chains {np.sum(g['kind'] == 2):,}; shared {g['shared'].size}")
    return ends_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="+")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if all([check(cc) for cc in args.regions]) else 1)
    boxes = region_boxes()
    for cc in args.regions:
        build(cc, boxes)


if __name__ == "__main__":
    main()
