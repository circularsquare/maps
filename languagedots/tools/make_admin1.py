"""Build languagedots/admin1_shapes.geojson: first-level divisions (provinces, states, regions)
for the tooltip's place line (index.html, placeLine; Anita, 2026-10-08).

Natural Earth 10m admin_1_states_provinces (public domain), downloaded as the zip into
data/raw/ne_admin1/. Kept per division: its English name (`name_en`, else `name`), simplified to
SIMPLIFY degrees (topology preserved), outer rings only, islets under MIN_PART deg^2 dropped
unless the division is all small islands, coordinates to 3 decimals. The viewer fetches the file
the first time a tooltip opens.

    python tools/make_admin1.py [simplify_deg] [min_part_deg2]

admin1_shapes.geojson must be published next to index.html (tools/deploy.py GZ_FILES).
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import shapely

LD = Path(r"C:\Users\anita\projects\maps\languagedots")
SRC = LD / "data/raw/ne_admin1/ne_10m_admin_1_states_provinces.zip"
# 0.04 / 0.004 (2026-10-08): 0.84 MB gzipped, against 1.30 at 0.02; a name lookup needs no more
SIMPLIFY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
MIN_PART = float(sys.argv[2]) if len(sys.argv) > 2 else 0.004

g = gpd.read_file(f"zip://{SRC}")
print(f"{len(g):,} divisions in {g['admin'].nunique()} countries")


def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(x) for x in o]
    return round(o, 3) if isinstance(o, float) else o


feats = []
for r in g.itertuples(index=False):
    name = (r.name_en if isinstance(r.name_en, str) and r.name_en.strip() else r.name) or ""
    if not name or r.geometry is None or r.geometry.is_empty:
        continue
    s = shapely.simplify(r.geometry, SIMPLIFY, preserve_topology=True)
    if not shapely.is_valid(s):
        s = shapely.make_valid(s)
    polys = [q for q in shapely.get_parts(s) if q.geom_type == "Polygon" and not q.is_empty]
    if not polys:
        continue
    big = max(q.area for q in polys)
    floor = MIN_PART if big >= 100 * MIN_PART else 0
    polys = [shapely.Polygon(q.exterior) for q in polys if q.area >= floor] or \
            [shapely.Polygon(max(polys, key=lambda q: q.area).exterior)]
    s = shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]
    feats.append({"type": "Feature", "properties": {"n": name.strip()},
                  "geometry": rnd(json.loads(shapely.to_geojson(s)))})
out = LD / "admin1_shapes.geojson"
out.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                          separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(feats):,} divisions, {out.stat().st_size / 1e6:.1f} MB)")
