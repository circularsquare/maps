"""T-4364: Macau's current land, OSM's Macau relation (r1867188, admin_level 3) less OSM's coastline sea, from the
Guangdong extract's admin cache, written to data/raw/worldplan/osm_macau.geojson for world_plan.NE_REPLACE (Natural
Earth's Macau predates the reclamations: the game's pack was 21 km2 of about 33). ODbL, as the packs.

    venv\\Scripts\\python.exe -I trainworld\\notes\\T-4364\\scripts\\macau_outline.py [--write]
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import pyogrio  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import mapping  # noqa: E402
import osm_water  # noqa: E402
import world_plan as wp  # noqa: E402

TW = r"C:\Users\anita\projects\maps\trainworld"
SRC = os.path.join(TW, "data", "raw", "osm", "asia_china_guangdong.admin.fgb")
OUT = os.path.join(TW, "data", "raw", "worldplan", "osm_macau.geojson")
REL = 1867188

df = pyogrio.read_dataframe(SRC, bbox=(113.5, 22.05, 113.65, 22.25))
row = df[df.osm_id.astype(str) == str(REL)]
if len(row) != 1:
    raise SystemExit(f"r{REL} not in {SRC}")
g = wp.polygonal(shapely.make_valid(row.geometry.iloc[0]))
km2 = lambda q: wp.to_moll(q).area / 1e6  # noqa: E731
sea = shapely.union_all([shapely.make_valid(s) for s in osm_water.sea((113.45, 22.0, 113.7, 22.3))])
land = wp.polygonal(g.difference(sea))
ne = wp.territory_whole("MAC")
print(f"r{REL} {row['name'].iloc[0]}: {km2(g):.1f} km2, land {km2(land):.1f} km2; Natural Earth's Macau {ne.area / 1e6:.1f} km2")
if "--write" in sys.argv:
    # land only: with its waters the relation holds two thirds of geoBoundaries' Macau SAR feature, which zhu keeps for
    # Hengqin's east (17 km2 of OSM land that Natural Earth's China misses), so de_facto would give that feature to Macau
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(land), "properties": {
        "iso": "MAC", "source": f"OpenStreetMap relation {REL} (Macau, admin_level 3) less OSM's coastline sea, from Geofabrik's "
                                "Guangdong extract",
        "licence": "ODbL, OpenStreetMap contributors", "task": "T-4364"}}]}
    tmp = OUT + ".part"
    open(tmp, "w", encoding="utf-8").write(json.dumps(gj))
    os.replace(tmp, OUT)
    print("->", OUT)
