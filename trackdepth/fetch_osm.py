"""Fetch rail track ways and stations from OpenStreetMap (Overpass) for the map's square.

Writes raw/osm_tracks.json and raw/osm_stations.json (Overpass JSON, `out geom`). Track ways are
cut at the square's edge (a way that leaves and comes back keeps its longest part inside), so
everything downstream (build.py, fetch_routes.py, structure_heights.py) sees the same geometry.
"""
import json
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
URL = "https://overpass-api.de/api/interpreter"
UA = "trackdepth/0.1 (map of NYC rail track depth; research)"
# The map's square (W, S, E, N), about 52 x 52 km centred near Court Sq; decided 2026-11-02.
SQUARE = (-74.26, 40.508, -73.645, 40.975)
BBOX = f"{SQUARE[1]},{SQUARE[0]},{SQUARE[3]},{SQUARE[2]}"   # Overpass order: S, W, N, E

TRACKS = f"""
[out:json][timeout:900];
(
  way["railway"~"^(subway|rail|light_rail|monorail|construction|disused|abandoned)$"]({BBOX});
);
out tags geom;
"""

STATIONS = f"""
[out:json][timeout:300];
(
  node["railway"="station"]({BBOX});
  way["railway"="station"]({BBOX});
  node["railway"="stop"]({BBOX});
  node["public_transport"="stop_position"]["subway"="yes"]({BBOX});
);
out tags center;
"""


def clip_to_square(elements):
    """Cut each way's geometry at the square; drop ways wholly outside. Returns (kept, n_cut)."""
    from shapely.geometry import LineString, box
    sq = box(*SQUARE)
    out, n_cut = [], 0
    for e in elements:
        g = e.get("geometry")
        if e["type"] != "way" or not g or len(g) < 2:
            continue
        pts = [(p["lon"], p["lat"]) for p in g]
        if all(SQUARE[0] < x < SQUARE[2] and SQUARE[1] < y < SQUARE[3] for x, y in pts):
            out.append(e)
            continue
        part = LineString(pts).intersection(sq)
        if part.is_empty:
            continue
        parts = [part] if part.geom_type == "LineString" else [p for p in getattr(part, "geoms", []) if p.geom_type == "LineString"]
        if not parts:
            continue
        best = max(parts, key=lambda p: p.length)
        # keep the original vertices (their exact coordinates join ways at shared nodes); only
        # the cut ends are new points
        e = dict(e)
        e["geometry"] = [{"lon": x, "lat": y} for x, y in best.coords]
        e["tags"] = dict(e.get("tags", {}), **{"trackdepth:cut": "square"})
        out.append(e)
        n_cut += 1
    return out, n_cut


ENDPOINTS = (URL, "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter")


def fetch(q, out, clip=False):
    import time
    r = None
    for attempt in range(12):   # the public servers are often full: back off and go round them
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=1200)
        except requests.RequestException as ex:
            print(f"{url}: {ex}", flush=True)
            r = None
        if r is not None and r.ok:
            break
        if r is not None:
            print(f"{url}: {r.status_code} {r.text[:120]!r}", flush=True)
        time.sleep(min(300, 20 * 2 ** (attempt // len(ENDPOINTS))))
    if r is None:
        raise SystemExit("Overpass unreachable")
    r.raise_for_status()
    js = json.loads(r.content)
    if clip:
        js["elements"], n_cut = clip_to_square(js["elements"])
        print(f"{n_cut} ways cut at the square's edge")
    data = json.dumps(js).encode()
    out.write_bytes(data)
    print(f"{out.name}: {len(js['elements'])} elements, {len(data)/1e6:.1f} MB")


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["tracks", "stations"]
    if "tracks" in which:
        fetch(TRACKS, RAW / "osm_tracks.json", clip=True)
    if "stations" in which:
        fetch(STATIONS, RAW / "osm_stations.json")
