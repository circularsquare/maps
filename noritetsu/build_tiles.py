"""Turn the extracted rail ways into a pmtiles archive: the faint all-lines background.

This builds the layer that is ALWAYS drawn, whether or not anything has been ridden.  Ridden
track is not in here; it is drawn on top from the rider's own data, because it needs its own
colour and width rules and there are only ever a few thousand sections of it.

    python build_tiles.py --region jp

LOD.  The whole point is that this feels smooth on a phone, so what a zoom carries is decided
by `MINZOOM`: main line at z2, branch and unclassified rail at z6, urban rail at z6, industrial
spurs at z9, yards and sidings at z12.  Geometry is Douglas-Peucker'd per zoom to about half a
screen pixel, and features whose whole bounding box is under a pixel are dropped outright.

MAXZOOM IS 13 AND CARRIES UNSIMPLIFIED GEOMETRY, deliberately.  MapLibre overzooms vector
tiles happily, so z13 tiles serve z14-z18 as well; if z13 were simplified at its own one-pixel
tolerance, that error would be sixteen pixels wide by z17 and every curve would visibly
polygonise.  Full OSM detail is cheap here -- Japan is a couple of million vertices.

Encoding uses `mapbox_vector_tile`, not the hand-rolled encoder in ../religiondots/mvt.py.
That one is points-only and exists because a religiondots build encodes 28 MILLION point
features, where the general encoder's per-feature shapely work dominates.  This build encodes
a few hundred thousand line features in total, so the same penalty is worth maybe a minute,
and a second hand-rolled wire format is not worth owning for that.
"""
import argparse
import gzip
import json
import math
import os
import pickle
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from shapely import STRtree, box, simplify
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

import mapbox_vector_tile

ROOT = Path(__file__).resolve().parent

EXTENT = 4096          # MVT tile coordinate space
PER_PX = EXTENT / 512  # MapLibre draws a 512 px tile, so 8 tile units to the screen pixel
TOL = 4.0              # Douglas-Peucker tolerance, half a screen pixel
DROP = 4.0             # a feature whose bbox diagonal is under half a pixel is not drawn
BUFFER = 64            # clip overspill, so a line does not end at the tile seam
MINZ, MAXZ = 0, 13
SPLIT = 10             # below this, merged chains; at and above, individual ways

# railway=* to the kind the map draws.  `preserved` is heritage railway.
KIND = {
    "rail": "rail", "light_rail": "light_rail", "subway": "subway", "tram": "tram",
    "monorail": "monorail", "funicular": "funicular", "narrow_gauge": "narrow_gauge",
    "preserved": "heritage",
}

# THIS MAP IS PASSENGER RAIL ONLY, so only two ranks survive into the tiles: 0 main line and
# 1 everything else a passenger can be carried over.  Yards, sidings, industrial spurs and
# freight-only branches are dropped in `geometries` unless a passenger route relation runs
# over them, which is how a station's own approach tracks stay in.
MINZOOM = {
    ("rail", 0): 0, ("rail", 1): 6,
    ("narrow_gauge", 0): 7, ("narrow_gauge", 1): 7,
    ("heritage", 0): 7, ("heritage", 1): 7,
    ("subway", 0): 6, ("subway", 1): 6,
    ("light_rail", 0): 6, ("light_rail", 1): 6,
    ("monorail", 0): 6, ("monorail", 1): 6,
    ("tram", 0): 7, ("tram", 1): 7,
    ("funicular", 0): 10, ("funicular", 1): 10,
}
URBAN = {"subway", "light_rail", "tram", "monorail", "funicular"}

# Station nodes worth a bubble.  A bare public_transport=stop_position is one per platform
# track, so including those would put five dots on one station.
STATION_RAILWAY = {"station", "halt", "tram_stop"}


def rank_of(kind, tags):
    """0 main, 1 branch or unstated, 2 industrial or military, 3 yard and siding."""
    if tags.get("service"):
        return 3
    usage = tags.get("usage")
    # Tourist NARROW GAUGE carries scheduled passengers often enough to be drawn: Taiwan's
    # Alishan Forest Railway is usage=tourism end to end with no route relation, and ranked
    # 2 it vanished from the map and from its register line. Park trains and rail-bike
    # loops on it are islands, which drop_islands takes out. Standard-gauge usage=tourism
    # stays rank 2: in Korea that is mostly rail bikes on closed main lines.
    if usage == "tourism" and kind == "narrow_gauge":
        return 1
    if usage in ("industrial", "military", "tourism", "freight", "test", "distribution"):
        return 2
    if usage == "main":
        return 0
    if usage == "branch":
        return 1
    # Unstated. Urban rail has no "usage" convention and is all main by purpose; a bare
    # railway=rail with no usage is most often a branch or a connecting curve.
    return 0 if kind in URBAN else 1


def merc(lon, lat):
    """Web Mercator, normalised to [0, 1] with y down, so tile maths is a multiply."""
    x = (lon + 180.0) / 360.0
    s = np.sin(np.radians(np.clip(lat, -85.05112878, 85.05112878)))
    y = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    return x, y


def load(region, log):
    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    c = np.load(d / "coords.npz")

    # Track that a passenger route relation runs over.  OSM has no passenger flag, and this
    # is the strongest evidence there is: a route=train|subway|tram|... relation whose path
    # includes this way.  Members with a role are the furniture (platforms, stops); the
    # route's own path is the roleless way members.
    on_route = set()
    for tags, members in rels.values():
        if tags.get("type") != "route":
            continue
        for ty, ref, role in members:
            if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                on_route.add(ref)

    log(f"loaded {len(ways)} ways, {len(stops)} stops, {c['id'].size} coords, "
        f"{len(on_route)} ways on a passenger route")
    return ways, stops, on_route, c["id"], c["x"], c["y"]


def geometries(ways, on_route, cid, cx, cy, log):
    """Way node ids to normalised mercator linestrings, plus the properties each carries.

    A way can reference nodes outside the extract's cut line.  Rather than drop the whole
    way, keep its longest run of consecutive resolvable nodes: near a border that is the
    part of the line that is actually inside the region.

    PASSENGER TRACK ONLY.  Measured on Japan, a passenger route relation covers 86% of
    main-line km but only 7% of yard and siding km and 12% of industrial km.  So route
    membership alone is much too lossy a filter -- it would delete a seventh of the main
    network, including freight-shared main lines where only one direction got into the
    relation -- while usage and service tags alone would keep every goods yard.  Both
    together: keep main and branch track, keep anything a passenger route actually runs
    over whatever its tags say, drop the rest.  `p` records which of the two let it in, so
    the map can tell confirmed passenger track from track that is merely plausible.
    """
    feats = []
    clipped = dropped = 0
    for wid, (tags, nodes) in ways.items():
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        ok = cid[pos] == nodes
        if not ok.all():
            # longest True run
            idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
            starts, ends = idx[0::2], idx[1::2]
            if starts.size == 0:
                continue
            best = np.argmax(ends - starts)
            sl = slice(starts[best], ends[best])
            nodes, pos = nodes[sl], pos[sl]
            clipped += 1
            if nodes.size < 2:
                continue
        kind = KIND[tags["railway"]]
        rank = rank_of(kind, tags)
        pax = wid in on_route
        if rank >= 2:
            if not pax:
                dropped += 1
                continue
            rank = 1          # a passenger route runs over it, so it is branch track to us
        lon = cx[pos] / 1e7
        lat = cy[pos] / 1e7
        x, y = merc(lon, lat)
        feats.append({
            "wid": wid,
            "nodes": nodes,
            "kind": kind,
            "rank": rank,
            "pax": 1 if pax else 0,
            "minzoom": MINZOOM.get((kind, rank), 7),
            "name": tags.get("name:en") or tags.get("name") or "",
            "xy": np.column_stack([x, y]),
        })
    log(f"built {len(feats)} way geometries ({clipped} truncated at the extract edge, "
        f"{dropped} dropped as yard, siding, industrial or freight-only)")
    return feats


def hex_colour(c):
    """An OSM colour tag as #rrggbb, or "" if it is not one. OSM carries CSS names
    ("SeaGreen"), short hex ("#F00") and the odd "#00 7AC0"; the tiles carry one form."""
    import matplotlib.colors as mc
    s = (c or "").strip().replace(" ", "")
    if not s:
        return ""
    for cand in (s, s.lower(), "#" + s if not s.startswith("#") else s):
        try:
            return mc.to_hex(cand)
        except ValueError:
            continue
    return ""


def way_colours(region, log):
    """OSM way id -> the colour of the line that runs over it, as #rrggbb.

    Read from build_model's output, which is why this build now runs AFTER build_model:
    ways.json already says which lines run over each way, register lines included. The
    register line's colour wins where it has one, since that is the line the map is of; then
    the commonest among the OSM lines on the way, lines before named trains, because the
    Nozomi's colour is not the colour of the Tokaido Shinkansen's track. A way no coloured
    line runs over gets none, and the app falls back to the colour of its kind.
    """
    d = ROOT / "dist" / "data" / region
    try:
        with open(d / "ways.json", encoding="utf-8") as f:
            ways = json.load(f)
        with open(d / "lines.json", encoding="utf-8") as f:
            lines = {l["id"]: l for l in json.load(f)["lines"]}
    except FileNotFoundError:
        log("no model for this region yet, so no line colours in the tiles; "
            "run build_model.py first")
        return {}
    out = {}
    for wid, idxs in ways["ways"].items():
        ls = [lines[ways["lines"][i]] for i in idxs if ways["lines"][i] in lines]
        best = None
        for tier in ([l for l in ls if l.get("src", "osm") != "osm"],
                     [l for l in ls if l.get("src", "osm") == "osm" and not l["service"]],
                     [l for l in ls if l["service"]]):
            got = Counter(h for h in (hex_colour(l["colour"]) for l in tier) if h)
            if got:
                best = got.most_common(1)[0][0]
                break
        if best:
            out[int(wid)] = best
    log(f"{len(out)} of {len(ways['ways'])} ways on a line carry that line's colour")
    return out


def drop_islands(region, feats, log):
    """Leave out every piece of track that touches no line: a set of ways joined to each other
    by shared nodes, none of which any line in the model runs over.

    WHY.  Such track can never be clicked or ridden, since nothing in the model is there, and
    in Japan it is nearly all amusement and tourist rides tagged as railways: the 奥祖谷
    sightseeing monorail looping round a mountainside, Disneyland's Western River Railroad, a
    roller coaster, pedal trolleys on closed branches. Anita saw them as "tiny orange rails
    not connected to anything" (2026-09-30).  Track that joins a line's track stays even when
    no line runs over it (the 武蔵野線 and 東海道 freight lines, the second bore of the 上越線's
    新清水 tunnel), since it reads as part of the network and a gap would read as missing.

    AN OSM PASSENGER ROUTE OVER THE PIECE DOES NOT SAVE IT.  A first version kept those, and
    they turned out to be the same unclickable track: Disneyland's railway and harbour freight
    lines (京葉臨海鉄道, 仙台臨海鉄道) are route=train relations with no stops, exactly like
    the Niesenbahn and Stoosbahn funiculars, and no tag tells them apart. So a real line that
    the model is missing vanishes here too; the log line is where that shows. The fix for one
    of those is to get it into the model, not to draw it unclickable.
    """
    try:
        with open(ROOT / "dist" / "data" / region / "ways.json", encoding="utf-8") as f:
            on_line = {int(w) for w in json.load(f)["ways"]}
    except FileNotFoundError:
        return feats
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in feats:
        ns = f["nodes"].tolist()
        r = find(ns[0])
        for n in ns[1:]:
            s = find(n)
            if s != r:
                parent[s] = r
    touched = {find(int(f["nodes"][0])) for f in feats if f["wid"] in on_line}
    keep, gone = [], Counter()
    for f in feats:
        if find(int(f["nodes"][0])) in touched:
            keep.append(f)
        else:
            gone[f["kind"]] += 1
    log(f"left out {sum(gone.values())} ways on pieces of track no line touches "
        f"({', '.join(f'{n} {k}' for k, n in gone.most_common())})")
    return keep


def merged_chains(feats, log):
    """Ways joined end to end into the longest chains they form, for the low zooms.

    WITHOUT THIS THE WORLD VIEW IS BLANK.  An OSM way is typically a few hundred metres, so
    at z0 every single one of them is a fraction of a pixel wide and the sub-pixel filter
    below throws all of them away -- z0 and z1 came out with zero tiles before this existed.
    Merging also cuts the low-zoom feature count by about an order of magnitude, which is
    most of what makes panning at world scale cheap.

    Merging discards `wid` and `name`, so it is only used below z10, where nothing is
    clickable or labelled and only kind, rank and colour are drawn.  linemerge joins through
    degree-two connections only, so chains break at real junctions, which is what we want,
    and ways of different colours are never merged, so a chain is one line's colour.
    """
    out = []
    keys = sorted({(f["kind"], f["rank"], f["pax"], f["c"]) for f in feats})
    for kind, rank, pax, c in keys:
        group = [LineString(f["xy"]) for f in feats
                 if f["kind"] == kind and f["rank"] == rank and f["pax"] == pax
                 and f["c"] == c]
        if not group:
            continue
        merged = linemerge(group) if len(group) > 1 else group[0]
        parts = (list(merged.geoms) if merged.geom_type == "MultiLineString"
                 else [merged])
        for p in parts:
            out.append({
                "kind": kind, "rank": rank, "pax": pax, "c": c,
                "minzoom": MINZOOM.get((kind, rank), 7),
                "name": "", "wid": 0,
                "xy": np.asarray(p.coords),
            })
    log(f"merged {len(feats)} ways into {len(out)} chains for z<{SPLIT}")
    return out


def tile_features(z, feats, chains, pts):
    """Every tile at zoom z that has anything in it, as {(x, y): (lines, points)}."""
    n = 1 << z
    span = n * EXTENT
    src = feats if z >= SPLIT else chains
    live = [f for f in src if f["minzoom"] <= z]

    geoms, props = [], []
    for f in live:
        g = f["xy"] * span
        if z < MAXZ:
            w = g[:, 0].max() - g[:, 0].min()
            h = g[:, 1].max() - g[:, 1].min()
            if math.hypot(w, h) < DROP:
                continue
        ls = LineString(g)
        if z < MAXZ:
            ls = simplify(ls, TOL, preserve_topology=False)
            if ls.is_empty or len(ls.coords) < 2:
                continue
        geoms.append(ls)
        p = {"k": f["kind"], "r": f["rank"], "p": f["pax"]}
        if f["c"]:
            p["c"] = f["c"]
        if z >= SPLIT:
            p["w"] = f["wid"]
            if f["name"]:
                p["nm"] = f["name"]
        props.append(p)

    out = {}
    if geoms:
        tree = STRtree(geoms)
        cand = set()
        for g in geoms:
            x0, y0, x1, y1 = g.bounds
            for tx in range(max(0, int(x0 // EXTENT)), min(n - 1, int(x1 // EXTENT)) + 1):
                for ty in range(max(0, int(y0 // EXTENT)),
                                min(n - 1, int(y1 // EXTENT)) + 1):
                    cand.add((tx, ty))
        for tx, ty in cand:
            clip = box(tx * EXTENT - BUFFER, ty * EXTENT - BUFFER,
                       (tx + 1) * EXTENT + BUFFER, (ty + 1) * EXTENT + BUFFER)
            hit = tree.query(clip)
            if hit.size == 0:
                continue
            lines = []
            for i in hit:
                piece = geoms[i].intersection(clip)
                if piece.is_empty:
                    continue
                parts = (list(piece.geoms) if piece.geom_type.startswith("Multi")
                         or piece.geom_type == "GeometryCollection" else [piece])
                for part in parts:
                    if part.geom_type != "LineString" or len(part.coords) < 2:
                        continue
                    co = np.asarray(part.coords)
                    co[:, 0] -= tx * EXTENT
                    co[:, 1] -= ty * EXTENT
                    lines.append((np.round(co).astype(np.int64), props[i]))
            if lines:
                out[(tx, ty)] = [lines, []]

    if z >= 8:
        for p in pts:
            gx, gy = p["xy"][0] * span, p["xy"][1] * span
            tx, ty = int(gx // EXTENT), int(gy // EXTENT)
            slot = out.setdefault((tx, ty), [[], []])
            slot[1].append((int(round(gx - tx * EXTENT)), int(round(gy - ty * EXTENT)), p))
    return out


def encode_tile(lines, points, z):
    layers = []
    if lines:
        layers.append({
            "name": "track",
            "features": [{"geometry": LineString(co), "properties": pr}
                         for co, pr in lines],
        })
    if points:
        layers.append({
            "name": "station",
            "features": [{"geometry": Point(x, y),
                          "properties": {"nm": p["name"], "k": p["kind"], "s": p["sid"]}}
                         for x, y, p in points],
        })
    if not layers:
        return None
    blob = mapbox_vector_tile.encode(
        layers, default_options={"extents": EXTENT, "y_coord_down": True,
                                 "check_winding_order": False})
    return gzip.compress(blob, mtime=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--max-zoom", type=int, default=MAXZ)
    args = ap.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    ways, stops, on_route, cid, cx, cy = load(args.region, log)
    feats = drop_islands(args.region, geometries(ways, on_route, cid, cx, cy, log), log)
    colours = way_colours(args.region, log)
    for f in feats:
        f["c"] = colours.get(f["wid"], "")
    chains = merged_chains(feats, log)
    # No station layer any more: bubbles come from the model (stations.json), and the tile
    # layer was one point per OSM station node, which nothing has read since.
    pts = []

    out = ROOT / "dist" / "data" / f"{args.region}.pmtiles"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pmtiles.tmp")

    allxy = np.concatenate([f["xy"] for f in feats])
    lon0 = allxy[:, 0].min() * 360 - 180
    lon1 = allxy[:, 0].max() * 360 - 180
    lat_of = lambda y: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))
    lat1 = lat_of(allxy[:, 1].min())
    lat0 = lat_of(allxy[:, 1].max())

    n_tiles = 0
    with open(tmp, "wb") as fh:
        w = Writer(fh)
        for z in range(MINZ, args.max_zoom + 1):
            tiles = tile_features(z, feats, chains, pts)
            written = 0
            for (tx, ty) in sorted(tiles, key=lambda t: zxy_to_tileid(z, t[0], t[1])):
                lines, points = tiles[(tx, ty)]
                blob = encode_tile(lines, points, z)
                if blob is None:
                    continue
                w.write_tile(zxy_to_tileid(z, tx, ty), blob)
                written += 1
            n_tiles += written
            log(f"  z{z:<2} {written:>7,} tiles")
        w.finalize(
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.GZIP,
                "min_zoom": MINZ,
                "max_zoom": args.max_zoom,
                "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
                "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7),
                "center_zoom": 6,
                "center_lon_e7": int((lon0 + lon1) / 2 * 1e7),
                "center_lat_e7": int((lat0 + lat1) / 2 * 1e7),
            },
            {
                "name": f"noritetsu {args.region}",
                "vector_layers": [
                    {"id": "track", "fields": {"k": "String", "r": "Number",
                                               "p": "Number", "w": "Number",
                                               "nm": "String", "c": "String"}},
                ],
            },
        )

    # The dev server keeps a handle on anything it has served and Windows refuses to replace
    # an open file, which would throw away the whole build over a file lock.
    try:
        os.replace(tmp, out)
    except OSError as e:
        raise SystemExit(f"\nbuilt {tmp.name} but could not put it in place:\n  {e}\n"
                         f"Stop whatever is serving {out.name} and run:  mv {tmp} {out}")
    log(f"wrote {out.name} ({out.stat().st_size/1e6:.1f} MB, {n_tiles:,} tiles)")


if __name__ == "__main__":
    main()
