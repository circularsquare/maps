"""T-1762: every zone row (any status) whose plan origin lies outside the unit it leads with (source.unit_names[0]),
with the distance; all countries whose units are geoBoundaries, NUTS3, ITL3 or Census counties.
python origins_all.py [plan.json] > list"""
import json
import sys
from pathlib import Path

import shapely
from shapely.geometry import Point, shape
from shapely.ops import transform
import pyproj

sys.stdout.reconfigure(encoding="utf-8")
TW = Path(r"C:\Users\anita\projects\maps\trainworld")
planp = Path(sys.argv[1]) if len(sys.argv) > 1 else TW / "pipeline/world_plan.json"
plan = json.loads(planp.read_text(encoding="utf-8"))
real = json.loads((TW / "pipeline/world_plan.json").read_text(encoding="utf-8"))
tr = pyproj.Transformer.from_crs(4326, "ESRI:54009", always_xy=True).transform
cache = {}


def units_of(iso):
    if iso in cache:
        return cache[iso]
    src = real["unit_sources"][iso]
    out = {}
    if src["kind"] == "geoBoundaries":
        p = TW / f"data/raw/worldplan/geoboundaries/{iso}_{src['level']}.geojson"
        if p.exists():
            for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
                out[f["properties"]["shapeID"]] = (f["properties"].get("shapeName"), shape(f["geometry"]))
    elif src["kind"] == "NUTS3":
        for f in json.loads((TW / src["file"]).read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            out[pr.get("NUTS_ID") or pr.get("id")] = (pr.get("NAME_LATN") or pr.get("NUTS_NAME"), shape(f["geometry"]))
    elif src["kind"] == "ITL3":
        for f in json.loads((TW / src["file"]).read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            out[pr.get("ITL325CD")] = (pr.get("ITL325NM"), shape(f["geometry"]))
    cache[iso] = out
    return out


rows = 0
bad = {}
for r in plan["metros"]:
    if r.get("kind") != "zone" or not r.get("units"):
        continue
    iso = list(r["units"])[0]
    if iso == "KOR":  # grouped zones: the origin is in the group it is named after by construction
        continue
    gs = units_of(iso)
    if not gs:
        bad[iso] = bad.get(iso, 0) + 1
        continue
    names = (r.get("source") or {}).get("unit_names") or [r["name"]]
    lead = next((i for i in r["units"][iso] if i in gs and gs[i][0] == names[0]), None)
    if lead is None:
        lead = next((i for i in r["units"][iso] if i in gs), None)
    if lead is None:
        bad[iso] = bad.get(iso, 0) + 1
        continue
    rows += 1
    g = gs[lead][1]
    o = Point(*r["origin"])
    if not g.buffer(1e-9).contains(o):
        d = transform(tr, g).distance(transform(tr, o)) / 1000
        print(f"{d:6.0f} km {iso} {r['id']} {r['name']} ({r['people'] / 1e6:.2f}M) {r.get('status')} lead {names[0]}")
print("checked", rows, "zone rows; units not found:", bad)
