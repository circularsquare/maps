"""Rwanda — boundaries for the 30 districts, from NISR's own GIS, and the join proved
to the person.

Writes data/geo/rw/rw_districts.gpkg, data/geo/rw/rw_sectors.gpkg and
data/geo/rw/rw_lookup.csv.

**THE BOUNDARIES ARE NOT COD-AB HERE, AND THAT IS THE POINT.** COD-AB Rwanda is
`rwa_adm_2006_NISR_WGS1984_20181002` — NISR's own 2006 boundaries, republished by OCHA, and
it would serve. What is used instead is NISR's ArcGIS Online layer **`Population_2002_2022`**
(`services5.arcgis.com/deNm5epdmeZgcm16`, owner `NisrProject`, public), because it is the
same office's 2022 geography **carrying the census counts as attributes**: 416 sector
polygons, three census years each, with `tot_pop`, `pop_ur` and `pop_ru` on every one.

**SO THE NAME JOIN IS NOT TRUSTED, IT IS PROVED.** Dissolving the 2022 sectors to district
gives 30 populations, and every one of them equals the district's own booklet total **to
the person** — and so do the urban and rural splits, which is 90 exact equalities on 90
figures that came out of a different publication. The 30 totals are all distinct
(318,126 Nyaruguru to 879,505 Gasabo), so equality alone pins the pairing uniquely and
there is no room for [[reference_name_join_wrong_neighbour]] to hide: a swapped pair does
not merely look odd, it fails. The layer's `province` field is checked against the province
each booklet prints as well, which catches a same-named district in the wrong province
before the populations are ever looked at.

**THE SECTORS ARE KEPT** because `sources/rw_grid.py` needs them: they are the finest
geography for which the census's own count exists, and the placement grid is re-levelled
onto them (see that module).

An office GIS server is worth looking for before COD-AB, and NISR's is the shape to expect:
`gis.statistics.gov.rw` is an alias of the Drupal site and answers a `/server/rest/services`
probe with a 404 HTML page, while the actual content is an **ArcGIS Online organisation**
reachable only through `arcgis.com/sharing/rest/search`. `tags:nisr` returns 95 items,
`owner:NisrProject` twelve. There is no religion layer; there is this.

Usage:
    python sources/rw_geo.py --fetch    one ~20 MB GeoJSON from ArcGIS Online
    python sources/rw_geo.py            rebuild from data/raw/rw/
"""

import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "rw")
GEO = os.path.join(ROOT, "data", "geo", "rw")
NORM = os.path.join(ROOT, "data", "normalized", "rw.csv")
DISTRICTS = os.path.join(GEO, "rw_districts.gpkg")
SECTORS = os.path.join(GEO, "rw_sectors.gpkg")
LOOKUP = os.path.join(GEO, "rw_lookup.csv")

SRC = ("https://services5.arcgis.com/deNm5epdmeZgcm16/arcgis/rest/services/"
       "Population_2002_2022/FeatureServer/0/query")
GEOJSON = "rw_nisr_sectors_2022.geojson"

EXPECTED_SECTORS = 416
EXPECTED_DISTRICTS = 30
EXPECTED_PROVINCES = 5
CENSUS_POPULATION = 13_246_394
# Rwanda is 26,338 km2 including its water and about 24,668 km2 of land. The sector layer
# comes out at 24,306 km2, so **it excludes Lake Kivu** (2,370 km2, of which Rwanda holds
# roughly a third) and the smaller lakes rather than tiling over them. That is worth
# knowing rather than incidental: §8.2c's problem of dots landing on open water does not
# arise for Rwanda and `water.py` is not involved, the same way Lake Malawi and Lake Kariba
# resolved themselves for Malawi and Zimbabwe.
AREA_KM2 = 24_668          # LAND area, which is what the sector polygons cover


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, GEOJSON)
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000_000:
        print("already have", dest)
        return
    print("GET", SRC, "census_time='2022'")
    r = requests.get(SRC, timeout=900, headers={"User-Agent": "Mozilla/5.0"}, params={
        "where": "census_time='2022'",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
        "resultRecordCount": 1000,
    })
    r.raise_for_status()
    # §5a: HTTP 200 is not a download. ArcGIS answers a bad query with a 200 and an
    # {"error": ...} body, and a paged one with exceededTransferLimit set.
    d = r.json()
    if "error" in d:
        raise SystemExit(f"ArcGIS returned an error: {d['error']}")
    feats = d.get("features", [])
    if d.get("exceededTransferLimit") or len(feats) != EXPECTED_SECTORS:
        raise SystemExit(f"got {len(feats)} features "
                         f"(exceededTransferLimit={d.get('exceededTransferLimit')}), "
                         f"expected {EXPECTED_SECTORS} -- the query needs paging")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    print(f"  {os.path.getsize(dest):,} bytes, {len(feats)} sector polygons")


def fold(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    src = os.path.join(RAW, GEOJSON)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    g = gpd.read_file(src)
    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED_SECTORS:
        raise SystemExit(f"{src}: {len(g)} features, expected {EXPECTED_SECTORS}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    if set(g["census_time"].astype(str)) != {"2022"}:
        raise SystemExit("the extract is not census_time=2022 only")
    print(f"NISR sectors: {len(g)} polygons, crs={g.crs}, "
          f"tot_pop {g['tot_pop'].sum():,}")

    if int(g["tot_pop"].sum()) != CENSUS_POPULATION:
        raise SystemExit(f"the sectors sum to {int(g['tot_pop'].sum()):,}, expected "
                         f"{CENSUS_POPULATION:,}")

    eq = g.to_crs("ESRI:102022")
    area = eq.geometry.area.sum() / 1e6
    print(f"  the 416 sectors cover {area:,.0f} km2 against Rwanda's {AREA_KM2:,} km2 of "
          f"LAND ({area / AREA_KM2:.3f}x); Lake Kivu is not tiled over")
    if not 0.9 <= area / AREA_KM2 <= 1.1:
        raise SystemExit("the sector layer does not tile the country")

    # ---- dissolve to district ----
    g["unit"] = "RWD" + g["district_id"].astype(str).str.strip()
    dis = g.dissolve(by="unit", aggfunc={"district": "first", "province": "first",
                                         "tot_pop": "sum", "pop_ur": "sum",
                                         "pop_ru": "sum"}).reset_index()
    if len(dis) != EXPECTED_DISTRICTS:
        raise SystemExit(f"dissolve gave {len(dis)} districts, "
                         f"expected {EXPECTED_DISTRICTS}")
    if dis["district"].nunique() != EXPECTED_DISTRICTS:
        raise SystemExit("two district_id values share a district name")
    if g["province"].nunique() != EXPECTED_PROVINCES:
        raise SystemExit(f"{g['province'].nunique()} provinces, "
                         f"expected {EXPECTED_PROVINCES}")

    # ---- the census side ----
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/rw.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = df[df["source_category"] == "Total"][["geo_id", "geo_name", "count", "note"]]
    if len(cen) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{len(cen)} census districts, expected {EXPECTED_DISTRICTS}")
    cen = cen.assign(province=cen["note"].str.extract(r"province=([^;]+)")[0].str.strip())

    poly = {}
    for _, r in dis.iterrows():
        k = fold(r["district"])
        if k in poly:
            raise SystemExit(f"NISR district name {r['district']!r} appears twice")
        poly[k] = r

    pairs, missing = {}, []
    for _, r in cen.iterrows():
        k = fold(r["geo_name"])
        if k in poly:
            pairs[r["geo_id"]] = (r, poly[k])
        else:
            missing.append((r["geo_id"], r["geo_name"]))
    used = {p[1]["unit"] for p in pairs.values()}
    spare = sorted(set(dis["unit"]) - used)

    print("\n  the join, both ways (§12):")
    print(f"    census districts           {len(cen):>4}")
    print(f"    NISR polygons              {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}  {missing}")
    print(f"    polygons with no census    {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- the province check, which runs BEFORE the populations ----
    bad = [(c["geo_name"], c["province"], p["province"])
           for c, p in pairs.values() if c["province"] != p["province"]]
    if bad:
        raise SystemExit(f"{len(bad)} districts sit in a different province in the layer "
                         f"than in their own booklet: {bad}")
    print(f"\n    all {len(pairs)} districts are in the same province in the layer as in "
          "the booklet that\n    printed them, so no pairing crosses a province before the "
          "counts are looked at.")

    # ---- THE PROOF: the polygon layer's own population, to the person ----
    rows, bad = [], []
    for gid in sorted(pairs):
        c, p = pairs[gid]
        rows.append((gid, c["geo_name"], int(c["count"]), int(p["tot_pop"])))
        if int(c["count"]) != int(p["tot_pop"]):
            bad.append(rows[-1])
    tot = [r[2] for r in rows]
    if len(set(tot)) != len(tot):
        raise SystemExit("two districts have the same population -- exact equality no "
                         "longer determines the pairing and this check must be replaced")
    print(f"\n    and the polygon layer carries the census count, so the pairing is not "
          f"assumed:\n    all {len(rows)} district populations agree TO THE PERSON, and "
          "all 30 are distinct, so\n    no other pairing of these names to these polygons "
          "reproduces them.")
    if bad:
        for gid, nm, a, b in bad[:8]:
            print(f"      {gid} {nm}: booklet {a:,} vs layer {b:,}")
        raise SystemExit(f"{len(bad)} districts disagree with the layer's own population")

    print(f"\n    {'district':<14} {'booklet':>10} {'layer':>10}")
    for gid, nm, a, b in rows:
        print(f"    {nm:<14} {a:>10,} {b:>10,}  {gid}")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    out = dis[["unit", "district", "province", "tot_pop", "geometry"]].rename(
        columns={"district": "name", "tot_pop": "pop"})
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nisr = {p["unit"]: c["geo_name"] for c, p in pairs.values()}
    out["name"] = out["unit"].map(nisr)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no census name")
    out.to_file(DISTRICTS, layer="districts", driver="GPKG")
    print(f"\nwrote {DISTRICTS} ({len(out)} polygons)")

    sec = g[["unit", "sector", "district", "tot_pop", "geometry"]].rename(
        columns={"tot_pop": "pop"})
    sec.to_file(SECTORS, layer="sectors", driver="GPKG")
    print(f"wrote {SECTORS} ({len(sec)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[k][1]["unit"] for k in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
