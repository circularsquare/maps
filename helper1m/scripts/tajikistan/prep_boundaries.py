"""Build Tajikistan's two boundary levels from the Geofabrik OSM extract.

Reads  helper1m/data/tajikistan/raw/tajikistan-261001.osm.pbf
Writes helper1m/data/tajikistan/boundaries/adm2.gpkg  (65 cities and districts)
       helper1m/data/tajikistan/boundaries/adm1.gpkg  (5 regions, dissolved from adm2)

OSM has the current district map (admin_level 6, post-2018 names: Kushoniyon,
Jayhun, Dusti, Levakant ...) plus Khujand, Guliston and Buston as their own
level-6 units, and Dushanbe at level 4. Three cities the statistics agency
reports on their own have no boundary in OSM and sit inside a district polygon:
Bokhtar (in Kushoniyon), Khorugh (in Shughnon) and Istiqlol (in Bobojon
Ghafurov). Each is carved out of the districts around it using its OSM place
outline (CITY_OUTLINES), which is the built-up town rather than the legal city
limit. Districts that were folded into a city (Isfara, Konibodom, Panjakent,
Istaravshan, Kulob, Norak, Vahdat, Tursunzoda, Hisor, Roghun) keep their OSM
district polygon, which is the territory the merged city now covers.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["OSM_USE_CUSTOM_INDEXING"] = "NO"

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import REGIONS, UNITS  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "tajikistan" / "raw"
PBF = RAW / "tajikistan-261001.osm.pbf"
OUT = HELPER / "data" / "tajikistan" / "boundaries"

# OSM place outlines for the three cities with no boundary relation.
# ("w", id) is a closed way, ("r", id) a multipolygon relation.
CITY_OUTLINES = {
    "bokhtar": ("w", 38153741),     # place=city, 13.5 km2
    "khorugh": ("w", 1367629514),   # place=city, 9.7 km2
    "istiqlol": ("r", 11142161),    # place=town + landuse=residential, 10.3 km2
}


def main():
    rel_ids = sorted({int(u[3][1:]) for u in UNITS if u[3].startswith("r")})
    way_ids = [v[1] for v in CITY_OUTLINES.values() if v[0] == "w"]
    rel_ids_city = [v[1] for v in CITY_OUTLINES.values() if v[0] == "r"]
    ids = ",".join(f"'{i}'" for i in rel_ids + rel_ids_city)
    wids = ",".join(f"'{i}'" for i in way_ids)
    g = pyogrio.read_dataframe(
        PBF, layer="multipolygons",
        where=f"osm_id IN ({ids}) OR osm_way_id IN ({wids})")
    g = g.to_crs(4326)
    rel = {int(r.osm_id): r.geometry for r in g.itertuples() if r.osm_id}
    way = {int(r.osm_way_id): r.geometry for r in g.itertuples() if r.osm_way_id}

    missing = [i for i in rel_ids + rel_ids_city if i not in rel] + [i for i in way_ids if i not in way]
    if missing:
        sys.exit(f"OSM ids not found in the pbf: {missing}")

    # The 62 relations tile the country with no gap or overlap over 0.5 km2
    # (checked against the union of the five admin_level-4 regions).
    nat = shapely.union_all([rel[int(u[3][1:])] for u in UNITS if u[3].startswith("r")])

    city_geom = {}
    for key, (kind, i) in CITY_OUTLINES.items():
        geom = way[i] if kind == "w" else rel[i]
        city_geom[key] = shapely.make_valid(geom).intersection(nat)
    carve = shapely.union_all(list(city_geom.values()))

    rows = []
    for code, name, name_tg, osm, _bul, _cen in UNITS:
        if osm.startswith("r"):
            geom = shapely.make_valid(rel[int(osm[1:])])
            geom = geom.difference(carve)
        else:
            geom = city_geom[osm[2:]]
        rows.append({"code": code, "name": name, "name_cn": name_tg,
                     "parent": code[:5], "group": code[:5], "geometry": geom})
    adm2 = gpd.GeoDataFrame(rows, crs=4326)
    adm2["geometry"] = adm2.geometry.apply(
        lambda gm: shapely.make_valid(gm) if not gm.is_valid else gm)
    # keep polygonal parts only (differences can leave slivers of lines)
    adm2["geometry"] = adm2.geometry.apply(
        lambda gm: shapely.union_all([p for p in getattr(gm, "geoms", [gm])
                                      if p.geom_type in ("Polygon", "MultiPolygon")]))

    eq = adm2.to_crs("ESRI:54009")
    adm2["km2"] = (eq.area / 1e6).round(1)
    # overlaps / sliver checks
    total = eq.area.sum() / 1e6
    union = gpd.GeoSeries([shapely.union_all(eq.geometry.values)], crs="ESRI:54009").area[0] / 1e6
    print(f"adm2: {len(adm2)} units, sum of areas {total:,.0f} km2, union {union:,.0f} km2 "
          f"(overlap {total - union:,.1f} km2)")

    reg = {r[0]: r for r in REGIONS}
    adm1 = adm2.dissolve(by="group", as_index=False)[["group", "geometry"]]
    adm1["code"] = adm1["group"]
    adm1["name"] = adm1["code"].map(lambda c: reg[c][1])
    adm1["name_cn"] = adm1["code"].map(lambda c: reg[c][2])
    adm1 = adm1[["code", "name", "name_cn", "group", "geometry"]]
    adm1["km2"] = (adm1.to_crs("ESRI:54009").area / 1e6).round(0)
    print(adm1[["code", "name", "km2"]].to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    adm2.to_file(OUT / "adm2.gpkg", driver="GPKG")
    adm1.to_file(OUT / "adm1.gpkg", driver="GPKG")
    print(f"wrote {OUT / 'adm2.gpkg'} and adm1.gpkg")


if __name__ == "__main__":
    main()
