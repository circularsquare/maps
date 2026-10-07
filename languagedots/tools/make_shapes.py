"""Build languagedots/country_shapes.geojson for the viewer's Auto (point-in-country tests).

Every languagedots country with territory: religiondots' data/processed/country_shapes.geojson
(Natural Earth 10m, its ALSO/FROM_UNITS/CLIP rules) where it has the code, else Natural Earth 10m
admin_0_countries by ISO_A2 / ISO_A2_EH. Both read-only. Simplified further to SIMPLIFY degrees,
topology preserved, coordinates rounded to 3 decimals.

    python tools/make_shapes.py [simplify_deg] [min_part_deg2]

Rerun when a country is added, so the viewer's Auto has its outline.
country_shapes.geojson must be published next to index.html.
"""
import json
import sys
from pathlib import Path

import shapely
import shapely.geometry

RD = Path(r"C:\Users\anita\projects\maps\religiondots\data")
LD = Path(r"C:\Users\anita\projects\maps\languagedots")
# the shipped file (2026-10-06) was built with 0.02 deg and islets under ~25 km2 dropped
SIMPLIFY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
MIN_PART = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
NEVER = {"xs"}

counts = json.loads((LD / "data/processed/counts.json").read_text(encoding="utf-8"))
want = set(counts["countries"]) - NEVER
rd = json.loads((RD / "processed/country_shapes.geojson").read_text(encoding="utf-8"))
geoms = {}
for f in rd["features"]:
    cc = f["properties"]["cc"]
    if cc in want:
        geoms.setdefault(cc, []).append(shapely.geometry.shape(f["geometry"]))
missing = want - set(geoms)
ne = json.loads((RD / "geo/ne_10m_admin_0_countries.geojson").read_text(encoding="utf-8"))
for f in ne["features"]:
    p = f["properties"]
    for key in ("ISO_A2", "ISO_A2_EH"):
        code = str(p.get(key) or "").strip().lower()
        if code in missing and code not in geoms:
            geoms[code] = [shapely.geometry.shape(f["geometry"])]
            print("from Natural Earth:", code, p.get("ADMIN"))
            break
left = sorted(want - set(geoms))
if left:
    raise SystemExit(f"no outline for {left}")


def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(x) for x in o]
    return round(o, 3) if isinstance(o, float) else o


feats = []
for cc in sorted(geoms):
    g = shapely.union_all(geoms[cc]) if len(geoms[cc]) > 1 else geoms[cc][0]
    s = shapely.simplify(g, SIMPLIFY, preserve_topology=True)
    if not shapely.is_valid(s):
        s = shapely.make_valid(s)
    # keep polygons only (make_valid can return collections)
    polys = [q for q in shapely.get_parts(s) if q.geom_type == "Polygon"]
    # islets under MIN_PART deg^2 (~1 km^2) go, unless the whole country is small islands
    big = max(q.area for q in polys)
    floor = MIN_PART if big >= 100 * MIN_PART else 0
    polys = [shapely.Polygon(q.exterior) for q in polys if q.area >= floor]
    s = shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]
    feats.append({"type": "Feature", "properties": {"cc": cc},
                  "geometry": rnd(json.loads(shapely.to_geojson(s)))})
out = LD / "country_shapes.geojson"
out.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                          separators=(",", ":")), encoding="utf-8")
print(f"wrote {out} ({len(feats)} countries, {out.stat().st_size / 1024:.0f} KB)")
