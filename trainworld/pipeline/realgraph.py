"""The real network as a routing graph, one file per noritetsu region (T-727b, notes/T-727.md).

    python realgraph.py jp us gb fr de      build these regions
    python realgraph.py --check jp ...      read the files back and print what they hold
    python realgraph.py                     only rewrite index.json from the files there
    --out DIR                               write or read the files there instead

Reads noritetsu's extract (`noritetsu/data/proc/<cc>/ways.pkl`, `rels.pkl`, `coords.npz`) and its
line model (`noritetsu/dist/data/<cc>/ways.json`); never writes there. Keeps the ways the game's
real-network layer draws under All (`app/src/map/realNetwork.ts`, the same filter as noritetsu's
build_tiles.py): freight track (`freight_way`, from the extract's ways) and abandoned rail (from the region's
`<cc>_extra.pmtiles`, joined to other track within 3 m), and every kind in KIND at rank 0 or 1, whether a line runs over it or not (T-3495,
T-3069). The track tool snaps only to these, each track type to the kinds it may follow
(`app/src/game/realFollow.ts`). Short groups of other track that join drawn track at two points or
more go in too, so a portal or station throat no line runs over does not cut the graph (T-816,
`gap_fillers`; tram sidings and diversions among them, T-3515); they are not drawn, so a click never starts
on one.

Writes `data/packs/realgraph/<cc>.rgr`, and `index.json` listing every region built there (the
track tool tries to snap only inside those, T-913); gzip of the little-endian layout below; the app reads it
with `game/realGraph.ts` and a production build copies it (vite.config.ts). Coordinates are
degrees times 1e6 (0.1 m). Vertices are thinned to 3 m (Douglas-Peucker; tram 0.5 m, T-3515), chain ends kept.

    "TWRG" u32 version=2, u32 junctions J, u32 chains C, u32 vertices V, u32 shared S
    i32[J] lon, i32[J] lat                      junctions (OSM nodes where chains meet or end)
    u32[C] a, u32[C] b                          each chain's two junctions, a at its first vertex
    u8[C] kind (KIND: 0 rail, 1 light rail, 2 subway, 3 tram, 4 monorail, 5 funicular, 6 narrow
        gauge, 7 heritage, 8 freight, 9 abandoned; version 1 had only 0, light rail in it, and 2; an app
        that does not know 8 and 9 follows neither, T-3495), u8[C] rank (0 main, 1 other), u8[C] flags (1 passenger route,
        2 bridge, 4 tunnel, 8 high speed; the top four bits OSM's `layer`, a signed nibble clamped
        to -8..7, for the real networks' levels, T-1773; `layer_of`)
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
# 2 from T-3069: the kinds below; a page that reads only version 1 then finds no graph rather than
# taking tram chains for rail
VERSION = 2
THIN_M = 3.0
# T-3515: tram chains are thinned this finely; at 3 m a street corner's 15 m curve is one or two chords, so the
# router's heading 10 m into a chain reads the corner as a turn of over 60 degrees and refuses it
THIN_TRAM_M = 0.5
# OSM railway=* to the graph's kind; `preserved` is heritage (noritetsu build_tiles.KIND)
KIND = {"rail": 0, "light_rail": 1, "subway": 2, "tram": 3, "monorail": 4, "funicular": 5,
        "narrow_gauge": 6, "preserved": 7}
# T-3495: freight track comes from the extract's ways, abandoned from the region's extra tiles (real_extra.py)
KIND_EXTRA = {"freight": 8, "abandoned": 9}
MAX_Z = 13  # the tiles' last zoom, where geometry is unsimplified
TILE = 4096
JOIN_M = 3.0  # an abandoned line's end this close to a node of the graph's other track is that node
CLUSTER_M = 2.0  # ends of tile pieces this close are one node (each tile rounds its own coordinates)
URBAN = {"light_rail", "subway", "tram", "monorail", "funicular"}
# T-816: track the layer leaves out that may still fill a gap in it; the longest group of such ways
# that goes in, km
SPARE_USAGE = {"industrial", "military", "tourism", "test"}
SPARE_KM = 3.0
# T-3515: tram track of these services may fill a gap too: a city's tram links no line is mapped over (Zurich's
# Bahnhofquai track while its stop is rebuilt) are tagged so; yards (depots) stay out
SPARE_TRAM_SERVICE = {"siding", "diversion", "spur"}


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


def layer_tag(v):
    """OSM's `layer` as an integer clamped to a signed nibble; 0 when absent or unreadable ("-1;0")."""
    try:
        return max(-8, min(7, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def layer_of(flags):
    """The `layer` a chain's flags carry (the top four bits, signed)."""
    n = (int(flags) >> 4) & 15
    return n - 16 if n >= 8 else n


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


def node_positions(cid, nodes):
    """A way's nodes as indices into the extract's coordinates; the longest run the extract holds
    where a way leaves it. None if under two nodes."""
    pos = np.searchsorted(cid, nodes)
    np.clip(pos, 0, cid.size - 1, out=pos)
    ok = cid[pos] == nodes
    if not ok.all():
        idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
        starts, ends = idx[0::2], idx[1::2]
        if starts.size == 0:
            return None
        b = np.argmax(ends - starts)
        pos = pos[starts[b]:ends[b]]
    return pos.astype(np.int64) if pos.size >= 2 else None


def freight_way(tags):
    """Track real_extra.py draws as freight (T-3464), by the extract's tags."""
    if tags.get("railway") != "rail" or tags.get("railway:traffic_mode") == "passenger":
        return False
    return (tags.get("service") in ("yard", "siding", "spur") or tags.get("usage") in ("freight", "industrial")
            or tags.get("railway:traffic_mode") == "freight")


def abandoned_pieces(path):
    """Abandoned rail of a region's extra tiles as lon, lat arrays. The extract keeps no abandoned track, so this reads
    the tiles real_extra.py wrote: the last zoom's lines, each cut to its own tile so the buffer's overlap is not
    counted twice."""
    import math
    import mapbox_vector_tile
    from pmtiles.reader import MmapSource, all_tiles
    from shapely.geometry import LineString, box
    core = box(0, 0, TILE, TILE)
    out = []
    with open(path, "rb") as f:
        for (z, tx, ty), data in all_tiles(MmapSource(f)):
            if z != MAX_Z:
                continue
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            layer = mapbox_vector_tile.decode(data, default_options={"y_coord_down": True}).get("track")
            for ft in layer["features"] if layer else ():
                if ft["properties"].get("k") != "abandoned":
                    continue
                g = ft["geometry"]
                lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"] if g["type"] == "MultiLineString" else []
                for ln in lines:
                    if len(ln) < 2:
                        continue
                    c = LineString(ln).intersection(core)
                    for p in ([c] if c.geom_type == "LineString" else list(c.geoms) if c.geom_type in ("MultiLineString", "GeometryCollection") else []):
                        if p.geom_type != "LineString" or len(p.coords) < 2:
                            continue
                        a = np.asarray(p.coords, np.float64)
                        wx = (tx * TILE + a[:, 0]) / (TILE * 2 ** MAX_Z)
                        wy = (ty * TILE + a[:, 1]) / (TILE * 2 ** MAX_Z)
                        out.append(np.column_stack([wx * 360 - 180, np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * wy))))]))
    return out


def add_abandoned(pieces, keep, cid, cx, cy):
    """The pieces as ways over new nodes appended to the extract's coordinates; an end within JOIN_M of a node of the
    other drawn track is that node, so the graph connects there. Returns (ways, cid, cx, cy)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    planar = lambda lon, lat: np.column_stack([lon * np.cos(np.radians(lat)) * 111320.0, lat * 110950.0])
    n0 = cid.size
    ends = np.array([p[i] for p in pieces for i in (0, -1)])
    em = planar(ends[:, 0], ends[:, 1])
    drawn = np.unique(np.concatenate([k[0] for k in keep]))
    node_of = np.full(len(ends), -1, np.int64)
    d, i = cKDTree(planar(cx[drawn] / 1e7, cy[drawn] / 1e7)).query(em, distance_upper_bound=JOIN_M)
    ok = np.isfinite(d)
    node_of[ok] = drawn[i[ok]]
    pr = np.array(sorted(cKDTree(em).query_pairs(CLUSTER_M)), np.int64).reshape(-1, 2)
    _, lab = connected_components(coo_matrix((np.ones(len(pr), np.int8), (pr[:, 0], pr[:, 1])), shape=(len(ends),) * 2), directed=False)
    comp_node = {}
    for e in np.flatnonzero(node_of >= 0):
        comp_node.setdefault(lab[e], node_of[e])
    new_lon, new_lat, nxt = [], [], n0
    ways = []
    for pi, p in enumerate(pieces):
        idx = np.empty(len(p), np.int64)
        for j, e in ((0, 2 * pi), (len(p) - 1, 2 * pi + 1)):
            if lab[e] not in comp_node:
                comp_node[lab[e]] = nxt
                new_lon.append(ends[e, 0])
                new_lat.append(ends[e, 1])
                nxt += 1
            idx[j] = comp_node[lab[e]]
        inner = len(p) - 2
        if inner > 0:
            idx[1:-1] = np.arange(nxt, nxt + inner)
            new_lon.extend(p[1:-1, 0])
            new_lat.extend(p[1:-1, 1])
            nxt += inner
        if idx[0] != idx[-1] or len(p) > 2:
            ways.append(idx)
    # ids above any OSM node id, so no neighbour's extract shares them
    ids = (2 ** 52 + np.arange(nxt - n0)).astype(cid.dtype)
    return (ways, np.concatenate([cid, ids]), np.concatenate([cx, np.round(np.array(new_lon) * 1e7).astype(cx.dtype)]),
            np.concatenate([cy, np.round(np.array(new_lat) * 1e7).astype(cy.dtype)]))


def gap_fillers(keep, spare, cx, cy, n):
    """The spare ways (track the layer leaves out) that fill a gap in the drawn network (T-816):
    spare ways joined to one another through nodes off the drawn track form a group, and a group
    goes in whole when it meets drawn track at two nodes or more and is at most SPARE_KM long. A
    tunnel portal, a station throat or one track of a double line that no line in ways.json runs
    over would otherwise leave the drawn line in pieces; a yard or a whole undrawn line is longer."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not spare:
        return []
    drawn = np.zeros(n, bool)
    for k in keep:
        drawn[k[0]] = True
    # a drawn node is its own vertex in each spare way, so groups meet only off the drawn track
    verts, extra = [], n
    for pos in (s[0] for s in spare):
        v = pos.copy()
        on = drawn[pos]
        v[on] = extra + np.arange(on.sum())
        extra += on.sum()
        verts.append(v)
    a = np.concatenate([v[:-1] for v in verts])
    b = np.concatenate([v[1:] for v in verts])
    _, lab = connected_components(coo_matrix((np.ones(a.size, np.int8), (a, b)), shape=(extra, extra)), directed=False)
    group = np.array([lab[v[0]] for v in verts])
    km = np.zeros(lab.max() + 1)
    meets = {}
    for g, (pos, *_rest) in zip(group, spare):
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        km[g] += np.sum(np.hypot(np.diff(lon) * np.cos(np.radians(lat[1:])) * 111.32, np.diff(lat) * 110.95))
        meets.setdefault(g, set()).update(pos[drawn[pos]].tolist())
    ok = {g for g, m in meets.items() if len(m) >= 2 and km[g] <= SPARE_KM}
    out = [(pos, kind, 1, flags & ~1) for g, (pos, kind, rank, flags) in zip(group, spare) if g in ok]
    print(f"  gap fillers: {len(out):,} of {len(spare):,} spare ways, {len(ok):,} groups, "
          f"{sum(km[g] for g in ok):,.0f} km", flush=True)
    return out


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
    extra = NORI / "dist" / "data" / cc / "track_extra.json"
    if extra.exists():  # track of a named line no section covers (T-3043), flagged x in the tiles
        on_line |= set(json.loads(extra.read_text(encoding="utf-8")))
    on_route = set()
    for tags, members in rels.values():
        if tags.get("type") == "route":
            for ty, ref, role in members:
                if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                    on_route.add(ref)
    del rels

    keep, spare, freight = [], [], []
    for wid, (tags, nodes) in ways.items():
        rw = tags.get("railway")
        if rw not in KIND:
            continue
        rank = rank_of(rw, tags)
        if rw == "light_rail" and rank == 0:
            rank = 1  # other rail in the layer's classes, whatever its usage tag says
        pax = wid in on_route
        shown = wid in on_line
        if rank >= 2 and pax:
            rank = 1
        if rank >= 2:
            shown = False
        elif not shown:
            shown = True  # track no line runs over is drawn in All, so it is snapped to (T-3495)
        # a way the layer leaves out is kept aside: it may fill a gap between drawn track (T-816)
        service = tags.get("service")
        left_out = not shown and ((service not in (None, "crossover") and not (rw == "tram" and service in SPARE_TRAM_SERVICE))
                                  or tags.get("usage") in SPARE_USAGE)
        is_freight = not shown and freight_way(tags)  # drawn from the extra tiles under All, so snapped to (T-3495)
        if left_out and not is_freight:
            continue
        pos = node_positions(cid, nodes)
        if pos is None:
            continue
        if is_freight:
            fl = (2 if tags.get("bridge") not in (None, "no") else 0) | (4 if tags.get("tunnel") not in (None, "no") else 0) \
                | (layer_tag(tags.get("layer")) & 15) << 4
            freight.append((pos, KIND_EXTRA["freight"], 1, fl))
            if left_out:
                continue
        flags = (1 if pax else 0) | (2 if tags.get("bridge") not in (None, "no") else 0) \
            | (4 if tags.get("tunnel") not in (None, "no") else 0) | (8 if tags.get("highspeed") == "yes" else 0) \
            | (layer_tag(tags.get("layer")) & 15) << 4
        (keep if shown else spare).append((pos, KIND[rw], min(rank, 1), flags))
    del ways
    fill = gap_fillers(keep, spare, cx, cy, cid.size)
    keep += fill
    if not keep:
        print(f"{cc}: no drawn track, no graph", flush=True)
        return
    # the extra file (T-3495): freight not already in as a gap filler, and abandoned rail from the region's tiles; the
    # page loads it only under All. Nodes both files have are junctions in both and listed as shared, which is how the
    # page joins the two
    filled = {id(f[0]) for f in fill}
    extra = [f for f in freight if id(f[0]) not in filled]
    tiles = NORI / "dist" / "data" / f"{cc}_extra.pmtiles"
    if tiles.exists():
        pieces = abandoned_pieces(tiles)
        if pieces:
            ab, cid, cx, cy = add_abandoned(pieces, keep + extra, cid, cx, cy)
            extra += [(w, KIND_EXTRA["abandoned"], 1, 0) for w in ab]
            print(f"  abandoned: {len(ab):,} pieces from {tiles.name}", flush=True)
    both = np.intersect1d(np.concatenate([k[0] for k in keep]), np.concatenate([k[0] for k in extra])) if extra else np.zeros(0, np.int64)
    write_graph(cc, keep, cid, cx, cy, boxes, both, f"{cc}.rgr", t0)
    if extra:
        write_graph(cc, extra, cid, cx, cy, boxes, both, f"{cc}.x.rgr", t0)


def write_graph(cc, keep, cid, cx, cy, boxes, both, name, t0):
    allpos = np.concatenate([k[0] for k in keep])
    uniq, counts = np.unique(allpos, return_counts=True)
    isj = np.zeros(cid.size, bool)
    isj[uniq[counts >= 2]] = True
    for k in keep:
        isj[k[0][0]] = isj[k[0][-1]] = True
    isj[both] = True

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
            seg = seg[thin(x, y, THIN_TRAM_M if kind == KIND["tram"] else THIN_M)]
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

    shared_ids = np.intersect1d(cid[jpos], np.union1d(neighbour_ids(cc, boxes), cid[both]))
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
    tmp = OUT / f"{name}.tmp"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(body, 9))
    os.replace(tmp, OUT / name)
    size = (OUT / name).stat().st_size
    print(f"{name}: {C:,} chains, {J:,} junctions, {V:,} vertices, {S} shared with neighbours; "
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
    kinds = {name: int(np.sum(g["kind"] == k)) for name, k in {**KIND, **KIND_EXTRA}.items() if np.any(g["kind"] == k)}
    print(f"{cc}: version {g['version']}; chain ends on their junctions {ends_ok}; {km:,.0f} km of track; {n:,} pieces, "
          f"largest {big / J:.0%} of junctions; chains by kind {kinds}; shared {g['shared'].size}")
    return ends_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*", help="none: only rewrite index.json")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", help="write or check the files here instead")
    ap.add_argument("--nori", help="the noritetsu folder to read, for a tree that has no data of its own")
    args = ap.parse_args()
    if args.nori:
        global NORI
        NORI = Path(args.nori)
    if args.out:
        global OUT
        OUT = Path(args.out)
    if args.check:
        sys.exit(0 if all([check(cc) for cc in args.regions]) else 1)
    boxes = region_boxes()
    for cc in args.regions:
        build(cc, boxes)
    write_index()


def write_index():
    """index.json: the regions with a graph, so the track tool snaps only where one exists (T-913)."""
    stems = sorted(p.name[:-4] for p in OUT.glob("*.rgr"))
    built = [x for x in stems if not x.endswith(".x")]
    extra = [x[:-2] for x in stems if x.endswith(".x")]
    tmp = OUT / "index.json.tmp"
    tmp.write_text(json.dumps({"regions": built, "extra": extra}), encoding="utf-8")
    os.replace(tmp, OUT / "index.json")
    print(f"index.json: {len(built)} regions, {len(extra)} with an extra file", flush=True)


if __name__ == "__main__":
    main()
