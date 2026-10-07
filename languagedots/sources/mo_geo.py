"""Macau: the 23 statistical districts, and the census's own population by residential building.

    python sources/mo_geo.py --fetch    query DSEC's census GIS (about 25 requests, ~5 MB)
    python sources/mo_geo.py            build data/geo/mo/

Writes
    data/geo/mo/mo_units.gpkg       23 district polygons: `unit` (DSEC's district id 1-23), `name`
    data/geo/mo/mo_buildings.gpkg   the placement layer: every building the census found people
                                    in, `unit` (its district), `pop` (its 2021 census population)

SOURCE. DSEC's Statistical Geographic Information System (https://www.dsec.gov.mo/gis/unidade/,
"2021 Population Census" tab) is an ArcGIS 10.0 server with open query endpoints. The service
the page uses for selection queries carries both layers at full resolution:
    .../CensusGIS/rest/services/Production2021/Census-QueryZonaBuilding-2021/MapServer/0
        "c2021_Zona": the 23 statistical districts plus the maritime area (Zona 26), with density
    .../CensusGIS/rest/services/Production2021/Census-QueryZonaBuilding-2021/MapServer/1
        "c2021_Building_P5": 5,682 building polygons, each with the census's resident count
        (`total`, and teen/adult/old, male/female) and `Geocode_txt`, a nine-digit geocode.
NOT the display services (Census2021_Density23_*, Census2021_Population_*): they serve the same
features generalised to a handful of vertices (district 7 as 5 points, 0.156 km2 against its
0.212), and about 960 of their buildings in a second, broken coordinate system.

PROJECTION. The server labels its data wkid 3064 (an Italian UTM zone) and reprojects to 4326
wrongly (Macau comes back at 4.7E 0.16N). The coordinates are the Macao Grid (EPSG:8433, Macao
1920, x 18,575-26,420, y 8,174-21,240), so geometry is fetched native and reprojected here with
EPSG's "Macao 1920 to WGS 84 (1)" (1 m), which moves it about 300 m.

PLACEMENT. Dots go on the buildings, weighted by each building's census population: the census's
own count of where people live, inside the district it counted them in. Nothing published says
where inside a district the speakers of one language live, so every language in a district is
spread the same way.

CHECKS, none a tolerance:
  * 23 land districts and the maritime area, ids 1-23 and 26, each rebuilt at the server's own
    Shape.area (within 0.1%);
  * the buildings' `total` sums to the census's land population (681,293) exactly, and the one
    maritime-area feature equals its 777;
  * the geocode's first two digits, summed, equal the census database's seven parish totals;
  * every district has at least one populated building.
The district of a building is spatial (its largest overlap; 24 buildings straddle a line, and
the nearest district within 200 m would take one in no polygon, of which there are none): the
geocode nests in parishes but not in districts. The witness is the buildings summed by district
against the census database's district totals, a separate release: all 23 agree exactly.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mo"
GEO = ROOT / "data" / "geo" / "mo"

BASE = "https://www.dsec.gov.mo/CensusGIS/rest/services/Production2021/Census-QueryZonaBuilding-2021/MapServer/"
ZONES_URL = BASE + "0/query"
BLDG_URL = BASE + "1/query"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
NATIVE = 8433                  # Macao 1920 / Macao Grid
MARITIME = 26
# Share of the buildings' people that land in a different district from the census's, set at
# 0.5% before reading. On the full-resolution layers it reads 0 (every district exact); on the
# generalised display layers it read 3.58%.
WITNESS_MAX = 0.005
# geocode prefix -> the census database's parish id (freguesias_2021): Santo António, São Lázaro,
# São Lourenço, Sé, Nossa Senhora de Fátima, Taipa, Coloane
PARISH_OF_GEOCODE = {"11": 1, "12": 2, "13": 3, "14": 4, "15": 5, "21": 6, "22": 7}
NEAREST_MAX_M = 200            # a building in no district polygon goes to the nearest, up to this


def get(url, params):
    import requests
    r = requests.get(url, params=params, headers=UA, timeout=120)
    r.raise_for_status()
    d = r.json()
    if "error" in d:
        raise SystemExit(f"!! {url}: {d['error']}")
    return d


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "gis_zones.json"
    if not dest.exists():
        d = get(ZONES_URL, {"where": "1=1", "outFields": "*", "returnGeometry": "true", "f": "json"})
        d["_source"] = ZONES_URL
        dest.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        print(f"got {len(d['features'])} district features")
    dest = RAW / "gis_buildings.json"
    if not dest.exists():
        ids = sorted(get(BLDG_URL, {"where": "1=1", "returnIdsOnly": "true", "f": "json"})["objectIds"])
        feats = []
        for i in range(0, len(ids), 250):
            chunk = ids[i:i + 250]
            d = get(BLDG_URL, {"objectIds": ",".join(map(str, chunk)), "outFields":
                               "OBJECTID,geogrouping,total,teen,adult,old,male,female,Geocode_txt,"
                               "Nome_Edf_P_Unicode,Shape.area", "returnGeometry": "true", "f": "json"})
            if len(d["features"]) != len(chunk):
                raise SystemExit(f"!! asked for {len(chunk)} buildings, got {len(d['features'])}")
            feats += d["features"]
            print(f"  buildings {len(feats):,}/{len(ids):,}")
        out = {"spatialReference": d.get("spatialReference"), "features": feats,
               "_source": BLDG_URL}
        tmp = dest.with_suffix(".part")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)


def esri_polygon(rings):
    """Esri rings -> shapely by the even-odd rule: a point is inside if an odd number of rings
    enclose it (shells and their holes alternate however the rings are wound)."""
    from shapely.geometry import MultiPolygon, Polygon
    if any(abs(p[0]) < 1000 for r in rings for p in r):
        raise SystemExit("!! a feature in degrees, not on the Macao Grid (a display layer?)")
    geom = None
    for r in rings:
        if len(r) < 4:
            continue
        p = Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        geom = p if geom is None else geom.symmetric_difference(p)
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])
    elif geom.geom_type != "MultiPolygon":
        geom = MultiPolygon([g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"])
    return geom


def main():
    if "--fetch" in sys.argv:
        fetch()
    import geopandas as gpd
    import pandas as pd

    # ---- the districts -----------------------------------------------------------------------
    z = json.loads((RAW / "gis_zones.json").read_text(encoding="utf-8"))
    rows = [{"unit": int(f["attributes"]["Zona"]), "name": f["attributes"]["Desc_P"].strip(),
             "density": f["attributes"]["density"], "server_m2": f["attributes"]["Shape.area"],
             "geometry": esri_polygon(f["geometry"]["rings"])}
            for f in z["features"]]
    zones = gpd.GeoDataFrame(rows, crs=NATIVE)
    ids = sorted(zones["unit"])
    if ids != list(range(1, 24)) + [MARITIME]:
        raise SystemExit(f"!! district ids {ids}")
    zones = zones[zones["unit"] != MARITIME].copy()
    zones["km2"] = zones.geometry.area / 1e6
    ra = zones["km2"] * 1e6 / zones["server_m2"]
    print(f"23 districts, {zones['km2'].sum():.1f} km2 (median {zones['km2'].median():.2f} km2); "
          f"polygon area over the server's Shape.area {ra.min():.4f}-{ra.max():.4f}")
    if not ra.between(0.999, 1.001).all():
        for _, r in zones[~ra.between(0.999, 1.001)].iterrows():
            print(f"    district {r['unit']}: {r['km2']:.4f} km2 rebuilt, {r['server_m2'] / 1e6:.4f} server")
        raise SystemExit("!! a district polygon was rebuilt with the wrong area")

    # ---- the census's district populations ----------------------------------------------------
    zl = json.loads((RAW / "zona_lang.json").read_text(encoding="utf-8"))["Value"]
    zids, lids = zl["dimension"]
    census = {}
    for i, zz in enumerate(zids):
        census[int(zz)] = sum(int(x) for x in zl["data"][i * len(lids):(i + 1) * len(lids)])
    # Witness for the district polygons: the GIS's own density (people per km2) times the
    # polygon's area against the census database's count.
    r = zones["density"] * zones["km2"] / zones["unit"].map(census)
    print(f"district polygons: GIS density x polygon area over census population, "
          f"{r.min():.3f}-{r.max():.3f} (district {zones.loc[r.idxmin(), 'unit']} lowest)")

    # ---- the buildings -----------------------------------------------------------------------
    b = json.loads((RAW / "gis_buildings.json").read_text(encoding="utf-8"))
    rows = []
    for f in b["features"]:
        a = f["attributes"]
        rows.append({"oid": a["OBJECTID"], "grp": int(a["geogrouping"]), "pop": int(a["total"] or 0),
                     "name": a.get("Nome_Edf_P_Unicode") or "", "server_m2": a["Shape.area"],
                     "geometry": esri_polygon(f["geometry"]["rings"]) if f.get("geometry") else None})
    bld = gpd.GeoDataFrame(rows, crs=NATIVE)
    ba = bld.geometry.area / bld["server_m2"]
    print(f"{len(bld):,} building features, {bld['pop'].sum():,} people, "
          f"{(bld['pop'] > 0).sum():,} with people; rebuilt area over Shape.area "
          f"{ba.min():.4f}-{ba.max():.4f}")
    if not ba.between(0.99, 1.01).all():
        raise SystemExit("!! a building polygon was rebuilt with the wrong area")
    x = bld.geometry.centroid.x
    if not x.between(15000, 30000).all():
        raise SystemExit(f"!! building x outside the Macao Grid: {x.min():.1f}..{x.max():.1f}")
    mar = bld[bld["grp"] == 0]
    if len(mar) != 1 or int(mar["pop"].iloc[0]) != census[MARITIME]:
        raise SystemExit(f"!! maritime feature {mar[['oid', 'pop']].values.tolist()} vs census {census[MARITIME]}")
    bld = bld[bld["grp"] != 0].copy()
    if bld.geometry.isna().any() or bld.geometry.is_empty.any():
        raise SystemExit("!! a building has no geometry")
    if bld["pop"].sum() != sum(v for k, v in census.items() if k != MARITIME):
        raise SystemExit("!! the buildings do not sum to the census's land population")
    print(f"the land buildings hold {bld['pop'].sum():,} people, the census's land population exactly")
    fl = json.loads((RAW / "freg_lang.json").read_text(encoding="utf-8"))["Value"]
    fids, flids = fl["dimension"]
    parish = {int(f): sum(int(x) for x in fl["data"][i * len(flids):(i + 1) * len(flids)])
              for i, f in enumerate(fids)}
    geo2 = {f["attributes"]["OBJECTID"]: f["attributes"]["Geocode_txt"][:2] for f in b["features"]}
    by_geo = bld.groupby(bld["oid"].map(geo2))["pop"].sum()
    got = {PARISH_OF_GEOCODE[k]: int(v) for k, v in by_geo.items() if v}
    if got != {k: v for k, v in parish.items() if k != 8}:
        raise SystemExit(f"!! geocode parishes {got} vs census {parish}")
    print("the geocode's first two digits reproduce the census's seven parish totals exactly")
    bld = bld[bld["pop"] > 0].copy()       # empty buildings place nobody

    # Each building's district, by where it is: its largest overlap with a district polygon.
    # (`geogrouping` is a nine-digit geocode, not the district.)
    pieces = gpd.overlay(bld[["oid", "geometry"]], zones[["unit", "geometry"]], how="intersection",
                         keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    best = pieces.sort_values("a", ascending=False).drop_duplicates("oid").set_index("oid")
    split = pieces.groupby("oid")["unit"].nunique()
    bld["unit"] = bld["oid"].map(best["unit"])
    lost = bld[bld["unit"].isna()]
    if len(lost):
        # in no district polygon: the nearest district, printed (none on the current fetch)
        near = gpd.sjoin_nearest(lost[["oid", "geometry"]], zones[["unit", "geometry"]],
                                 distance_col="m")
        near = near[~near.index.duplicated()]
        npop = bld.loc[near.index, "pop"]
        print(f"{len(near)} buildings ({int(npop.sum()):,} people) overlap no district polygon: "
              f"nearest district, median {near['m'].median():.0f} m, "
              f"max {near['m'].max():.0f} m")
        for i, r in near.sort_values("m", ascending=False).head(5).iterrows():
            print(f"    building {r['oid']} ({bld.at[i, 'pop']:,} people): district {r['unit']} "
                  f"at {r['m']:.0f} m")
        if near["m"].max() > NEAREST_MAX_M:
            raise SystemExit(f"!! a building over {NEAREST_MAX_M} m from any district")
        bld.loc[near.index, "unit"] = near["unit"]
    bld["unit"] = bld["unit"].astype(int)
    print(f"{int((split > 1).sum())} buildings straddle a district line (each goes to its larger part)")

    # Witness: the buildings, summed by the district they fall in, against the census database's
    # district totals, a separate release (the language cube) of the same census.
    by = bld.groupby("unit")["pop"].sum()
    moved = 0
    for k in range(1, 24):
        got, want = int(by.get(k, 0)), census[k]
        moved += abs(got - want)
        if got != want:
            print(f"    district {k:2d}: buildings {got:,}, census {want:,} ({got - want:+,})")
    share = moved / 2 / bld["pop"].sum()
    print(f"witness: buildings by district against the census's district totals: "
          f"{moved // 2:,} people ({share:.3%}) in a different district")
    if share > WITNESS_MAX:
        raise SystemExit(f"!! more than {WITNESS_MAX:.1%} of people land in another district")

    place = bld[bld["pop"] > 0].copy()
    empty = set(range(1, 24)) - set(place["unit"])
    if empty:
        raise SystemExit(f"!! districts with no populated building: {sorted(empty)}")
    place["unit"] = place["unit"].astype(str)
    place = place[["unit", "pop", "oid", "name", "geometry"]].to_crs(4326)
    zones = zones.to_crs(4326)
    zones["unit"] = zones["unit"].astype(str)
    w, s, e, n = zones.total_bounds
    if not (113.50 < w < e < 113.65 and 22.08 < s < n < 22.24):
        raise SystemExit(f"!! reprojected bounds {zones.total_bounds}")
    per = place.groupby("unit").size()
    print(f"placement: {len(place):,} populated buildings, median {int(per.median())} per district "
          f"(min {per.min()}, district {per.idxmin()}); bounds {[round(x, 4) for x in (w, s, e, n)]}")

    GEO.mkdir(parents=True, exist_ok=True)
    for gdf, name in ((zones[["unit", "name", "km2", "density", "geometry"]], "mo_units.gpkg"),
                      (place, "mo_buildings.gpkg")):
        tmp = GEO / ("tmp_" + name)
        if tmp.exists():
            tmp.unlink()
        gdf.to_file(tmp, driver="GPKG")
        os.replace(tmp, GEO / name)
        print(f"wrote {GEO / name}")


if __name__ == "__main__":
    main()
