"""Pull the rail network out of an OSM extract, without interpreting any of it.

THREE PASSES OVER THE .pbf, and the reason is memory.  Building way geometry needs a node
id -> coordinate lookup, and pyosmium's own location index costs about 12 bytes for EVERY
node in the file, the overwhelming majority of which are buildings, trees and footpath
vertices.  Japan is roughly 200M nodes, so that index is ~2.4 GB for the ~3M nodes actually
wanted; a planet-scale run would be ~100 GB, and this machine has 23 GB of disk.  So:

    pass 1   ways + relations   which ways are railway, which nodes they reference,
                                and the members of every rail route relation
    pass 2   tagged nodes       stations, halts, tram stops, stop positions
    pass 3   nodes by id        coordinates for exactly the ids passes 1-2 asked for,
                                selected inside C++ by IdFilter

Passes 1 and 2 skip whole .pbf blocks at the source (`entities=`), so the cost is roughly
one read of the file per pass and almost no Python-level work per object.

This script DOES NOT build edges, lines or sections.  It writes the raw material for
build_model.py, so that re-deciding what a "line" is never means re-reading the .pbf.

    python extract.py --region jp --pbf data/raw/japan-260928.osm.pbf
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# BEFORE importing osmium and numpy, both of which will otherwise help themselves to all 16
# cores: libosmium reads OSMIUM_POOL_THREADS when it lazily builds its decompression pool, and
# numpy's BLAS reads OMP_NUM_THREADS at import.  This box is in use for other things.
os.environ.setdefault("OSMIUM_POOL_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import osmium

ROOT = Path(__file__).resolve().parent

# Track a passenger could be carried over.  `preserved` is heritage railway, which is ridable
# and which the Japanese line-completion hobby does count.  Deliberately absent: abandoned,
# disused, razed, construction, proposed (not ridable), and platform, station, turntable,
# traverser (not track).  `usage=industrial` track IS kept -- it is real track and belongs in
# the faint background layer -- but it will not become a ridable line.
TRACK = {
    "rail", "light_rail", "subway", "tram", "monorail",
    "funicular", "narrow_gauge", "preserved",
}

# Way tags worth carrying forward: identity, then the things a line-detail panel would show.
WAY_TAGS = {
    "railway", "name", "name:en", "ref", "operator", "operator:en", "network",
    "usage", "service", "gauge", "electrified", "voltage", "frequency", "maxspeed",
    "tunnel", "bridge", "layer", "highspeed", "passenger_lines", "tracks",
    # Freight-only track, where no passenger route relations exist to say so (China).
    "railway:traffic_mode",
}

# A node is a stopping place if it carries one of these keys with one of these values.
STOP_RAILWAY = {"station", "halt", "tram_stop", "stop"}
STOP_PT = {"stop_position", "station"}
STOP_TAGS = {
    "railway", "public_transport", "name", "name:en", "name:ja", "name:ko", "name:ja-Latn",
    "ref", "operator", "network", "station", "subway", "light_rail", "train", "tram",
    "monorail", "funicular", "usage", "wikidata",
}

# route_master groups the direction and short-turn variants of one line, which is a far better
# deduplication key than guessing from (network, ref, name).  Where it exists, use it.
ROUTE_KINDS = {"train", "subway", "light_rail", "tram", "monorail", "funicular"}
INFRA_KINDS = {"railway", "tracks"}
REL_TAGS = {
    "type", "route", "route_master", "name", "name:en", "ref", "colour", "color",
    "operator", "operator:en", "network", "from", "to", "via", "service", "roundtrip",
    "public_transport:version", "wikidata", "interval", "duration",
}


def keep(tags, wanted):
    return {k: v for k, v in tags if k in wanted}


def is_station_way(tags):
    """A station mapped only as an area: railway=station/halt (or public_transport=station
    saying train=yes) on a way, with a name. Bulgaria maps Чирпан, Калофер, Силистра and 24
    more stations this way and no node at all."""
    if not tags.get("name"):
        return False
    return (tags.get("railway") in ("station", "halt")
            or (tags.get("public_transport") == "station" and tags.get("train") == "yes"))


AREA_SAME_NAME_M = 500    # a node stop of the area's name this close means it is mapped already


def area_stations(station_ways, stops, nid, nx, ny, log):
    """Station areas as stops at the mean of their nodes, keyed by the NEGATIVE way id so they
    never collide with a node id. Only where no stop node of the same name lies within
    AREA_SAME_NAME_M: an area beside its own station node adds nothing."""
    import math
    by_name = {}
    for tags, lon, lat in stops.values():
        by_name.setdefault(tags.get("name"), []).append((lon, lat))
    added = 0
    for wid, (tags, nodes) in station_ways.items():
        pos = np.searchsorted(nid, nodes)
        np.clip(pos, 0, max(nid.size - 1, 0), out=pos)
        ok = nid[pos] == nodes
        if not ok.any():
            continue
        lon = float(nx[pos[ok]].mean()) / 1e7
        lat = float(ny[pos[ok]].mean()) / 1e7
        if any(math.hypot((lon - x) * math.cos(math.radians(lat)) * 111320, (lat - y) * 110570)
               <= AREA_SAME_NAME_M for x, y in by_name.get(tags["name"], ())):
            continue
        stops[-wid] = (tags, lon, lat)
        added += 1
    log(f"  station areas: {len(station_ways)} named station ways, {added} with no stop node "
        f"of their name within {AREA_SAME_NAME_M} m taken as stops")


def pass_ways_and_relations(pbf, log, station_ways=None):
    """`station_ways`, when a dict is passed (--station-areas), collects stations mapped only
    as an area; see area_stations."""
    """Railway ways (tags + node id list) and rail route relations (tags + members).

    Also the INFRASTRUCTURE line relations, route=railway and route=tracks: named track with
    the national line number in `ref` (France's 830000, Germany's VzG numbers, China's 0002
    京沪线), the same shape as N02. They go to their own dict and file, infra.pkl, so that
    nothing reading rels.pkl ever mistakes one for a passenger route. All tags are kept:
    there are few of them and which ref:* key carries the number varies by country."""
    ways, rels, infra = {}, {}, {}
    building = {}                    # railway=construction track, kept below if a route uses it
    needed = []                      # node ids whose coordinates we will need in pass 3
    fp = osmium.FileProcessor(pbf, osmium.osm.WAY | osmium.osm.RELATION)
    n = 0
    for obj in fp:
        n += 1
        if n % 2_000_000 == 0:
            log(f"  pass 1: {n/1e6:.0f}M objects, {len(ways)} ways, {len(rels)} relations")
        if obj.is_way():
            rw = obj.tags.get("railway")
            if station_ways is not None and is_station_way(obj.tags):
                nodes = np.fromiter((nd.ref for nd in obj.nodes), dtype=np.int64)
                station_ways[obj.id] = (keep(obj.tags, STOP_TAGS), nodes)
                needed.append(nodes)
                continue
            if rw not in TRACK:
                c = obj.tags.get("construction") or "rail"
                if rw == "construction" and c in TRACK:
                    nodes = np.fromiter((nd.ref for nd in obj.nodes), dtype=np.int64)
                    if nodes.size >= 2:
                        t = keep(obj.tags, WAY_TAGS)
                        t["railway"] = c
                        building[obj.id] = (t, nodes)
                continue
            nodes = np.fromiter((nd.ref for nd in obj.nodes), dtype=np.int64)
            if nodes.size < 2:
                continue            # a one-node way has no geometry to contribute
            ways[obj.id] = (keep(obj.tags, WAY_TAGS), nodes)
            needed.append(nodes)
        else:
            t = obj.tags.get("type")
            if t not in ("route", "route_master"):
                continue
            if t == "route" and obj.tags.get("route") in INFRA_KINDS:
                infra[obj.id] = (dict(obj.tags),
                                 [(m.type, m.ref, m.role) for m in obj.members])
                continue
            if obj.tags.get("route", obj.tags.get("route_master")) not in ROUTE_KINDS:
                continue
            members = [(m.type, m.ref, m.role) for m in obj.members]
            rels[obj.id] = (keep(obj.tags, REL_TAGS), members)
            # Stop positions are node members; their coordinates are wanted too.
            needed.append(np.fromiter(
                (r for ty, r, _ in members if ty == "n"), dtype=np.int64))
    # Track mapped railway=construction that a passenger route still runs over is being rebuilt
    # under traffic (Slovenia's line 50 at Preserje, 2026): kept as the track it will be.
    # Decided after the loop because a .pbf has every way before any relation.
    on_route = {r for _tags, ms in rels.values() for ty, r, _ in ms if ty == "w"}
    kept = 0
    for wid, (t, nodes) in building.items():
        if wid in on_route:
            ways[wid] = (t, nodes)
            needed.append(nodes)
            kept += 1
    log(f"  pass 1: {n} objects read, {len(infra)} infrastructure line relations, "
        f"{kept} of {len(building)} construction ways kept as a passenger route runs over them")
    return ways, rels, infra, needed


def pass_stops(pbf, log):
    """Nodes that are stopping places.  Tagged, so a key filter does the work in C++."""
    stops = {}
    fp = (osmium.FileProcessor(pbf, osmium.osm.NODE)
          .with_filter(osmium.filter.KeyFilter("railway", "public_transport")))
    for obj in fp:
        if (obj.tags.get("railway") not in STOP_RAILWAY
                and obj.tags.get("public_transport") not in STOP_PT):
            continue
        stops[obj.id] = (keep(obj.tags, STOP_TAGS), obj.location.lon, obj.location.lat)
        if len(stops) % 50_000 == 0:
            log(f"  pass 2: {len(stops)} stops")
    log(f"  pass 2: {len(stops)} stops")
    return stops


def pass_coords(pbf, ids, log):
    """Coordinates for exactly `ids`, as 1e-7 degree fixed point.

    IdFilter holds the set in C++ and rejects unwanted nodes before they reach Python, which
    is what makes a full node scan affordable: without it this loop is 200M iterations.
    """
    out_id = np.empty(ids.size, dtype=np.int64)
    out_x = np.empty(ids.size, dtype=np.int32)
    out_y = np.empty(ids.size, dtype=np.int32)
    i = 0
    fp = (osmium.FileProcessor(pbf, osmium.osm.NODE)
          .with_filter(osmium.filter.IdFilter(ids.tolist())))
    for obj in fp:
        if i == out_id.size:
            break
        out_id[i] = obj.id
        out_x[i] = obj.location.x        # osmium already stores 1e-7 degree ints
        out_y[i] = obj.location.y
        i += 1
        if i % 500_000 == 0:
            log(f"  pass 3: {i}/{ids.size} coordinates")
    log(f"  pass 3: {i}/{ids.size} coordinates")
    order = np.argsort(out_id[:i], kind="stable")
    return out_id[:i][order], out_x[:i][order], out_y[:i][order]


def clip(ways, rels, stops, nid, nx, ny, bbox, log):
    """Keep a way if any of its nodes is in the box, a stop if it is, and a relation if any
    member survived.  Ways crossing the edge are kept whole, as at a Geofabrik cut line, so a
    border line runs a little way past it.  Coordinates are left alone: build_model reads
    only the ids it is handed."""
    w, s, e, n = bbox
    x, y = nx / 1e7, ny / 1e7
    inside = set(nid[(x >= w) & (x <= e) & (y >= s) & (y <= n)].tolist())
    ways = {k: v for k, v in ways.items() if any(int(i) in inside for i in v[1])}
    stops = {k: v for k, v in stops.items() if w <= v[1] <= e and s <= v[2] <= n}
    kept = {("w", k) for k in ways} | {("n", k) for k in stops}
    # route_masters point at routes, so routes are settled first
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    rels = {k: v for k, v in rels.items()
            if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    log(f"  clipped to {bbox}: {len(ways)} ways, {len(stops)} stops, {len(rels)} relations")
    return ways, rels, stops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="output folder name, e.g. jp")
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--bbox", type=lambda s: [float(v) for v in s.split(",")],
                    help="west,south,east,north: keep only what lies in it, for a country "
                         "that Geofabrik ships inside a bigger extract (Singapore, Hong Kong)")
    ap.add_argument("--station-areas", action="store_true",
                    help="also take stations mapped only as an area (way), for a country "
                         "that maps many that way (Bulgaria)")
    args = ap.parse_args()

    pbf = Path(args.pbf)
    if not pbf.is_absolute():
        pbf = ROOT / pbf
    out = ROOT / "data" / "proc" / args.region
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    log(f"reading {pbf.name} ({pbf.stat().st_size/1e9:.2f} GB)")

    station_ways = {} if args.station_areas else None
    ways, rels, infra, needed = pass_ways_and_relations(str(pbf), log, station_ways)
    if not ways:
        sys.exit("no railway ways found -- wrong file, or the tag filter is wrong")

    stops = pass_stops(str(pbf), log)

    needed.append(np.fromiter(stops.keys(), dtype=np.int64))
    ids = np.unique(np.concatenate(needed))
    log(f"need coordinates for {ids.size} nodes")
    nid, nx, ny = pass_coords(str(pbf), ids, log)
    missing = ids.size - nid.size
    if missing:
        # Normal near an extract's cut line: a way can reference nodes outside the box.
        log(f"  {missing} node ids had no coordinate in this extract")
    if station_ways:
        area_stations(station_ways, stops, nid, nx, ny, log)

    if args.bbox:
        ways, rels, stops = clip(ways, rels, stops, nid, nx, ny, args.bbox, log)
        infra = {k: v for k, v in infra.items()
                 if any(t == "w" and r in ways for t, r, _ in v[1])}

    with open(out / "ways.pkl", "wb") as f:
        pickle.dump(ways, f, protocol=4)
    with open(out / "rels.pkl", "wb") as f:
        pickle.dump(rels, f, protocol=4)
    with open(out / "stops.pkl", "wb") as f:
        pickle.dump(stops, f, protocol=4)
    with open(out / "infra.pkl", "wb") as f:
        pickle.dump(infra, f, protocol=4)
    np.savez_compressed(out / "coords.npz", id=nid, x=nx, y=ny)

    kinds = {}
    for tags, _ in ways.values():
        kinds[tags["railway"]] = kinds.get(tags["railway"], 0) + 1
    masters = sum(1 for tags, _ in rels.values() if tags.get("type") == "route_master")
    log("done")
    print()
    print(f"region {args.region}")
    print(f"  ways       {len(ways)}")
    for k in sorted(kinds, key=lambda k: -kinds[k]):
        print(f"    {k:<14} {kinds[k]}")
    print(f"  stops      {len(stops)}")
    print(f"  relations  {len(rels)}  ({masters} route_master, {len(rels)-masters} route)")
    print(f"  infra      {len(infra)}  (route=railway / route=tracks line relations)")
    print(f"  coords     {nid.size}")
    print(f"  written to {out}")


if __name__ == "__main__":
    main()
