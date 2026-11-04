"""Fetch the OpenStreetMap coastline around the terrain extent (Overpass), for the flat water
polygon the 3D view draws at height 0 (make_water.py).

Writes raw/osm_coast.json (Overpass JSON, `out geom`: every natural=coastline way that touches
the box, whole).
"""
import json
import pathlib

import requests

ROOT = pathlib.Path(__file__).parent
URL = "https://overpass-api.de/api/interpreter"
UA = "trackdepth/0.1 (map of NYC subway track depth; personal research)"
BBOX = "40.40,-74.45,41.10,-73.50"     # a little wider than the terrain extent

Q = f"""
[out:json][timeout:300];
way["natural"="coastline"]({BBOX});
out tags geom;
"""

if __name__ == "__main__":
    for url in (URL, "https://overpass.kumi.systems/api/interpreter"):
        r = requests.post(url, data={"data": Q}, headers={"User-Agent": UA}, timeout=600)
        if r.ok:
            break
        print(f"{url}: {r.status_code} {r.text[:200]!r}")
    r.raise_for_status()
    out = ROOT / "raw" / "osm_coast.json"
    out.write_bytes(r.content)
    print(f"{out.name}: {len(json.loads(r.content)['elements'])} ways, {len(r.content) / 1e6:.1f} MB")
