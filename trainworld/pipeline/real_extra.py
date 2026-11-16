"""Freight and abandoned rail tiles for trainworld's Real world network "All" setting (T-3464).

noritetsu's per-region tiles are passenger rail only (build_tiles.py drops yards, sidings and
freight-only track, extract.py never reads railway=abandoned or disused), so this writes a
second, lighter archive per region, `<cc>_extra.pmtiles`, that the game adds only when
Settings > Real world network is All. Passenger never loads it. noritetsu's own files are read
(its tile writer is imported) and never changed.

    python real_extra.py --region us --pbf illinois.osm.pbf --bbox -88.3,41.5,-87.3,42.2 --outdir <dir>

One track layer, `k` = "freight" or "abandoned". What goes in (DECISIONS.md, T-3464):
  freight    railway=rail with usage=freight or industrial, or railway:traffic_mode=freight, or
             service=yard, siding or spur. Zooms: usage=freight 7, industrial 9, spur 10,
             yard and siding 12.
  abandoned  railway=abandoned or railway=disused (the line is still on the ground or its
             course is mapped). Not razed, dismantled, construction or proposed. Zoom 10.
Passenger-served track that also carries freight is not needed here: the game draws this layer
under the passenger layer, so overlap is hidden.
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import osmium
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

NORITETSU = Path(__file__).resolve().parents[2] / "noritetsu"
sys.path.insert(0, str(NORITETSU))
import build_tiles as bt  # noqa: E402  (tile maths and the MVT encoder, read only)

KEPT = {"railway", "usage", "service", "railway:traffic_mode", "name", "name:en"}


def classify(tags):
    """(kind, minzoom) for a way worth drawing here, else None."""
    rw = tags.get("railway")
    if rw in ("abandoned", "disused"):
        return "abandoned", 10
    if rw != "rail":
        return None
    service, usage = tags.get("service"), tags.get("usage")
    if tags.get("railway:traffic_mode") == "passenger":
        return None
    if service in ("yard", "siding"):
        return "freight", 12
    if service == "spur":
        return "freight", 10
    if usage == "freight" or tags.get("railway:traffic_mode") == "freight":
        return "freight", 7
    if usage == "industrial":
        return "freight", 9
    return None


def read(pbf, bbox, log):
    ways, need = {}, []
    fp = osmium.FileProcessor(pbf, osmium.osm.WAY).with_filter(osmium.filter.KeyFilter("railway"))
    for obj in fp:
        c = classify(obj.tags)
        if not c:
            continue
        nodes = np.fromiter((n.ref for n in obj.nodes), dtype=np.int64)
        if nodes.size < 2:
            continue
        ways[obj.id] = (c, obj.tags.get("name:en") or obj.tags.get("name") or "", nodes)
        need.append(nodes)
    log(f"pass 1: {len(ways)} ways")
    ids = np.unique(np.concatenate(need))
    cid = np.empty(ids.size, np.int64); cx = np.empty(ids.size, np.int32); cy = np.empty(ids.size, np.int32)
    i = 0
    fp = osmium.FileProcessor(pbf, osmium.osm.NODE).with_filter(osmium.filter.IdFilter(ids.tolist()))
    for obj in fp:
        if i == cid.size:
            break
        cid[i], cx[i], cy[i] = obj.id, obj.location.x, obj.location.y
        i += 1
    o = np.argsort(cid[:i], kind="stable")
    cid, cx, cy = cid[:i][o], cx[:i][o], cy[:i][o]
    log(f"pass 2: {i} coordinates")
    feats = []
    for wid, ((kind, mz), name, nodes) in ways.items():
        pos = np.clip(np.searchsorted(cid, nodes), 0, cid.size - 1)
        ok = cid[pos] == nodes
        if not ok.all():
            idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
            s, e = idx[0::2], idx[1::2]
            if s.size == 0:
                continue
            b = np.argmax(e - s)
            pos = pos[s[b]:e[b]]
            if pos.size < 2:
                continue
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        if bbox and not ((lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])).any():
            continue
        x, y = bt.merc(lon, lat)
        feats.append({"wid": wid, "nodes": nodes, "kind": kind, "rank": 0, "pax": 0, "c": "",
                      "n": 0, "minzoom": mz, "name": name, "xy": np.column_stack([x, y])})
    n = {k: sum(1 for f in feats if f["kind"] == k) for k in ("freight", "abandoned")}
    log(f"geometries: {n}")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--bbox", help="west,south,east,north: keep only ways touching it")
    ap.add_argument("--outdir", default=str(NORITETSU / "dist" / "data"))
    ap.add_argument("--max-zoom", type=int, default=bt.MAXZ)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    feats = read(a.pbf, bbox, log)
    if not feats:
        log("nothing to draw; no file written")
        return
    chains = bt.merged_chains(feats, log)
    out = Path(a.outdir) / f"{a.region}_extra.pmtiles"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    allxy = np.concatenate([f["xy"] for f in feats])
    lon0, lon1 = allxy[:, 0].min() * 360 - 180, allxy[:, 0].max() * 360 - 180
    lat_of = lambda y: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))
    lat1, lat0 = lat_of(allxy[:, 1].min()), lat_of(allxy[:, 1].max())
    with open(tmp, "wb") as fh:
        w = Writer(fh)
        for z in range(bt.MINZ, a.max_zoom + 1):
            tiles = bt.tile_features(z, feats, chains, [])
            n = 0
            for (tx, ty) in sorted(tiles, key=lambda t: zxy_to_tileid(z, t[0], t[1])):
                blob = bt.encode_tile(tiles[(tx, ty)][0], [], z)
                if blob:
                    w.write_tile(zxy_to_tileid(z, tx, ty), blob)
                    n += 1
            log(f"  z{z:<2} {n:>7,} tiles")
        w.finalize(
            {"tile_type": TileType.MVT, "tile_compression": Compression.GZIP, "min_zoom": bt.MINZ,
             "max_zoom": a.max_zoom, "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
             "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7), "center_zoom": 6,
             "center_lon_e7": int((lon0 + lon1) / 2 * 1e7), "center_lat_e7": int((lat0 + lat1) / 2 * 1e7)},
            {"name": f"trainworld extra {a.region}",
             "vector_layers": [{"id": "track", "fields": {"k": "String", "r": "Number", "p": "Number",
                                                          "w": "Number", "nm": "String"}}]})
    os.replace(tmp, out)
    # extra.json lists the regions that have a file, so the game asks only for those
    man = Path(a.outdir) / "extra.json"
    cur = json.loads(man.read_text()) if man.exists() else {}
    cur[a.region] = out.stat().st_size
    man.write_text(json.dumps(dict(sorted(cur.items()))))
    log(f"wrote {out.name} ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
