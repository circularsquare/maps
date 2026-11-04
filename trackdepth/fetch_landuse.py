"""Fetch the OpenStreetMap areas the terrain colouring needs (Overpass), inside the terrain extent:
piers (they are land over the water polygon), parks and woods (dark green), and lakes, ponds and
reservoirs (light blue).

Writes raw/osm_piers.json, raw/osm_parks.json, raw/osm_lakes.json: closed ways with their geometry
(`out tags geom`). Multipolygon relations are fetched apart, `relations`, into raw/osm_parks_rel.json
and raw/osm_lakes_rel.json, with `out body geom` (without `body` Overpass leaves out a relation's
member list, and with it the lake: The Lake and Harlem Meer in Central Park, Prospect Park Lake).
A relation that two fetch tiles both return is merged member by member. make_water.py and
make_terrain.py read all of them.

The box is asked for in tiles (raw/osm_tiles/ keeps each one, so a rerun carries on), since one
query for all of it times out on the public servers; a tile that keeps failing is split in four.

    python fetch_landuse.py [piers] [parks] [lakes]            # ways
    python fetch_landuse.py relations [parks] [lakes]          # multipolygon relations
"""
import json
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).parent
TILES = ROOT / "raw" / "osm_tiles"
URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
UA = "trackdepth/0.1 (map of NYC subway track depth; personal research)"
W, S, E, N = -74.38, 40.48, -73.60, 41.06
NX, NY = 3, 3

QUERIES = {
    "piers": ['"man_made"="pier"', '"man_made"="breakwater"', '"man_made"="groyne"'],
    "parks": ['"leisure"~"^(park|nature_reserve|garden|golf_course|recreation_ground|common)$"',
              '"landuse"~"^(grass|recreation_ground|village_green|forest|meadow)$"',
              '"natural"~"^(wood|scrub|grassland|wetland|heath)$"'],
    "lakes": ['"natural"="water"', '"landuse"="reservoir"', '"waterway"="riverbank"'],
}


def query(filters, areas_only, bbox, relations):
    parts = []
    for f in filters:
        if relations:
            parts.append(f'relation[{f}]["type"="multipolygon"]({bbox});')
        else:
            parts.append(f"way[{f}]({bbox});")
            if areas_only:
                parts.append(f'relation[{f}]["type"="multipolygon"]({bbox});')
    out = "out body geom;" if relations else "out tags geom;"
    return "[out:json][timeout:300];\n(\n  " + "\n  ".join(parts) + "\n);\n" + out


def post(q, label, tries):
    for attempt in range(tries):        # the public servers are often at capacity: back off and retry
        for url in URLS:
            try:
                r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=600)
            except requests.RequestException as ex:
                print(f"{label} {url}: {ex}", flush=True)
                continue
            if r.ok:
                return json.loads(r.content)["elements"]
            print(f"{label} {url}: {r.status_code}", flush=True)
        time.sleep(20 * 2 ** min(attempt, 3))
    return None


def tile(name, s, w, n, e, label, relations):
    """Elements of one box; split in four if the servers keep refusing it."""
    box = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    els = post(query(QUERIES[name], name != "piers", box, relations), label, 3)
    if els is not None:
        time.sleep(4)
        return els
    if n - s < 0.01:
        raise RuntimeError(f"gave up on {label}")
    ms, mw = (s + n) / 2, (w + e) / 2
    out = []
    for k, (a, b, c, d) in enumerate([(s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)]):
        out += tile(name, a, b, c, d, f"{label}.{k}", relations)
    return out


def merge_relation(a, b):
    """Two copies of one relation from different tiles: per member (by ref), keep the longer geometry."""
    ms = {m["ref"]: m for m in a.get("members", [])}
    for m in b.get("members", []):
        old = ms.get(m["ref"])
        if old is None or len(m.get("geometry") or []) > len(old.get("geometry") or []):
            ms[m["ref"]] = m
    a["members"] = list(ms.values())
    return a


def fetch(name, relations=False):
    TILES.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_rel" if relations else name
    seen, out = {}, []
    for i in range(NX):
        for j in range(NY):
            cache = TILES / f"{tag}_{i}_{j}.json"
            if cache.exists():
                els = json.loads(cache.read_text(encoding="utf-8"))
            else:
                els = tile(name, S + (N - S) * j / NY, W + (E - W) * i / NX, S + (N - S) * (j + 1) / NY,
                           W + (E - W) * (i + 1) / NX, f"{tag} {i},{j}", relations)
                cache.write_text(json.dumps(els), encoding="utf-8")
            for el in els:
                k = (el["type"], el["id"])
                if k in seen:
                    if el["type"] == "relation":
                        merge_relation(seen[k], el)
                else:
                    seen[k] = el
                    out.append(el)
            print(f"{tag} tile {i},{j}: {len(els)} elements, {len(out)} so far", flush=True)
    path = ROOT / "raw" / f"osm_{tag}.json"
    path.write_text(json.dumps({"elements": out}), encoding="utf-8")
    print(f"{path.name}: {len(out)} elements, {path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    rel = "relations" in args
    names = [a for a in args if a != "relations"] or ["parks", "lakes"]
    for name in names:
        fetch(name, relations=rel)
