"""T-302: each registered zone's `box` in app/src/game/registry.json.

    python pipeline/zone_boxes.py

`box` is [west, south, east, north] in degrees: the union of the zone's boundary
(`data/packs/<id>.boundary.geojson`) and its water mask's extent (the pack head's `water`, laid out in
the pack's own equirectangular frame), rounded outward to 0.001 degrees. The app learns where a zone
is from it without fetching any of the zone's files: the clock worker loads a water mask only where
track or the view comes near its box (docs/zones.md 6). Run after adding a zone to the registry.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "app/src/game/registry.json"
PACKS = ROOT / "data/packs"
EARTH_R = 6371008.8  # app/src/game/frame.ts
RAD = math.pi / 180


def coords(geom):
    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            yield from ring
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                yield from ring


def box(cid):
    w = s = math.inf
    e = n = -math.inf
    fc = json.loads((PACKS / f"{cid}.boundary.geojson").read_text(encoding="utf-8"))
    for f in fc["features"]:
        for lon, lat, *_ in coords(f["geometry"]):
            w, s, e, n = min(w, lon), min(s, lat), max(e, lon), max(n, lat)
    head = json.loads((PACKS / f"{cid}.json").read_text(encoding="utf-8"))
    wat = head.get("water")
    if wat:
        lon0, lat0 = head["origin"]["lon"], head["origin"]["lat"]
        kx, ky = EARTH_R * RAD * math.cos(lat0 * RAD), EARTH_R * RAD
        x0, y0 = wat["origin_m"]
        x1, y1 = x0 + wat["size"][0] * wat["cell_m"], y0 + wat["size"][1] * wat["cell_m"]
        w, e = min(w, lon0 + min(x0, x1) / kx), max(e, lon0 + max(x0, x1) / kx)
        s, n = min(s, lat0 + min(y0, y1) / ky), max(n, lat0 + max(y0, y1) / ky)
    return [math.floor(w * 1000) / 1000, math.floor(s * 1000) / 1000, math.ceil(e * 1000) / 1000, math.ceil(n * 1000) / 1000]


def main():
    reg = json.loads(REG.read_text(encoding="utf-8"))
    for c in reg["cities"]:
        c["box"] = box(c["id"])
        print(c["id"], c["box"])
    REG.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
