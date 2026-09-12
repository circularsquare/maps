"""Laos — boundaries for the 8,499 villages of the 2015 census.

Writes data/geo/la/la_units.gpkg and data/geo/la/la_lookup.csv.

**THE COUNTS AND THE POLYGONS COME OFF THE SAME SERVICE, SO THERE IS NO NAME JOIN AND NO
P-CODE JOIN.** Every other country here spends most of its geo module reconciling a
statistical table against somebody else's boundary file. LSB's village layer carries the
religion figures as attributes ON the polygons, keyed by `VCODE`, so the join is by
construction and the interesting question is a different one: **are these polygons the thing
they look like?**

**THEY ARE NOT ADMINISTRATIVE BOUNDARIES AND THE ATLAS SAYS SO OUTRIGHT.** Laos has never had
official digital village boundaries. What the 2005 census did have was a **GPS point for each
of 10,547 villages**, and the Centre for Development and Environment built polygons around
those points with an **accessibility model** — a travel-time surface rather than straight-line
distance, because in mountains the two are very different — so that each polygon is roughly
the territory whose nearest village is that one. Section A.7 of the atlas describes the
construction and adds the warning that matters here: *"this atlas is by no means intended to
be used as a planning tool at the level of single villages"*.

**WHAT THAT COSTS THIS MAP IS SMALL AND IT IS NOT ZERO.** A village's people live at its
point and not across its catchment, so the polygon is a placement envelope rather than a
statement about where anybody is. That is exactly the case §8.2 wants a population grid for,
and `sources/la_grid.py` supplies one; the polygons' job here is only to say which hexes
belong to which village. The counts are unaffected either way (§8.2 — the grid is a
within-unit weight).

**THE POLYGONS TILE THE COUNTRY, WHICH IS WORTH ASSERTING RATHER THAN ASSUMING**: 230,548 km²
against Laos's 236,800, so 97.4%, and the shortfall is the large water bodies and the border
strip that the accessibility model does not assign. The area distribution is the reason
`la_grid.py` exists — median 14.3 km², but 96 polygons over 200 km² holding 68,769 people
between them, and those are the upland districts of Phongsaly, Xekong and Attapeu where the
religion signal is.

**THE INDEPENDENT CHECK IS `Shape_Area` OFF A DIFFERENT SERVICE.** The geometry is downloaded
from `laos_2015_total_population`; the area attribute is read from
`laos_2015_distribution_of_buddhists`, which is a separate map service with its own copy of
the layer. Recomputing the area of every downloaded polygon in the source projection and
comparing it to the other service's stored `Shape_Area` tests that the two services really
publish the same geometry, and that the download did not lose or reorder a ring. Nothing
about this is guaranteed by the VCODE join: two services could carry the same 8,499 codes
over different vintages of the boundary file.

Usage:
    python sources/la_geo.py --fetch    ~13 MB of GeoJSON, about a minute
    python sources/la_geo.py            rebuild from data/raw/la/
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "la")
OUT_DIR = os.path.join(ROOT, "data", "geo", "la")
OUT = os.path.join(OUT_DIR, "la_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "la_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "la.csv")

SERVER = "https://gis.cde.unibe.ch/gis/rest/services/Decide"
GEOM_SERVICE = "laos_2015_total_population"
AREA_SERVICE = "laos_2015_distribution_of_buddhists"   # the independent copy, see docstring
PAGE = 1000                    # smaller than la.py's: these pages carry rings

GEOJSON = os.path.join(RAW, "villages.geojson")
AREAS = os.path.join(RAW, "shape_area.json")

EXPECTED_VILLAGES = 8_499
# The source projection: UTM zone 48N, which is what `Shape_Area` is stored in (m²). The
# comparison below has to be done in it, not in a geographic CRS.
SOURCE_EPSG = 32648
# Laos's land area, CIA World Factbook / FAO: 236,800 km². The polygons are an accessibility
# model rather than a boundary file, so they are asserted to cover MOST of it and not all.
LAOS_KM2 = 236_800
MIN_COVERAGE = 0.95
# Recomputed area against the other service's stored value. A polygon rebuilt from the same
# rings agrees to floating-point noise; anything above this is a different geometry.
AREA_REL_TOL = 1e-3
MAX_AREA_MISMATCH = 0


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch():
    os.makedirs(RAW, exist_ok=True)

    if os.path.exists(GEOJSON) and os.path.getsize(GEOJSON) > 5_000_000:
        print("already have", GEOJSON)
    else:
        feats, offset = [], 0
        while True:
            q = urllib.parse.urlencode({
                "where": "1=1", "outFields": "VCODE", "returnGeometry": "true",
                "outSR": SOURCE_EPSG, "resultOffset": offset, "resultRecordCount": PAGE,
                "orderByFields": "VCODE", "f": "geojson",
            })
            d = _get(f"{SERVER}/{GEOM_SERVICE}/MapServer/0/query?{q}")
            if "error" in d:
                raise SystemExit(f"{GEOM_SERVICE}: {d['error']}")
            got = d.get("features", [])
            feats += got
            print(f"    {len(feats):,} polygons", flush=True)
            if len(got) < PAGE:
                break
            offset += PAGE
            time.sleep(0.2)
        if len(feats) != EXPECTED_VILLAGES:
            raise SystemExit(f"downloaded {len(feats)} polygons, "
                             f"expected {EXPECTED_VILLAGES}")
        # §5a: a 200 is not a download. A GeoJSON with null geometries parses fine.
        null = [f["properties"]["VCODE"] for f in feats if not f.get("geometry")]
        if null:
            raise SystemExit(f"{len(null)} features came back with no geometry: {null[:5]}")
        with open(GEOJSON, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection",
                       "crs": {"type": "name",
                               "properties": {"name": f"EPSG:{SOURCE_EPSG}"}},
                       "features": feats}, fh)
        print(f"  {GEOJSON} ({os.path.getsize(GEOJSON):,} bytes)")

    if os.path.exists(AREAS) and os.path.getsize(AREAS) > 100_000:
        print("already have", AREAS)
        return
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1", "outFields": "VCODE,Shape_Area", "returnGeometry": "false",
            "resultOffset": offset, "resultRecordCount": 2000,
            "orderByFields": "VCODE", "f": "json",
        })
        d = _get(f"{SERVER}/{AREA_SERVICE}/MapServer/0/query?{q}")
        if "error" in d:
            raise SystemExit(f"{AREA_SERVICE}: {d['error']}")
        got = d.get("features", [])
        rows += [f["attributes"] for f in got]
        if len(got) < 2000:
            break
        offset += 2000
        time.sleep(0.2)
    if len(rows) != EXPECTED_VILLAGES:
        raise SystemExit(f"{AREA_SERVICE}: {len(rows)} rows, expected {EXPECTED_VILLAGES}")
    with open(AREAS, "w", encoding="utf-8") as fh:
        json.dump({"service": AREA_SERVICE, "rows": rows}, fh)
    print(f"  {AREAS} ({os.path.getsize(AREAS):,} bytes)")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    for p in (GEOJSON, AREAS):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run with --fetch first")

    g = gpd.read_file(GEOJSON)
    if g.crs is None:
        g = g.set_crs(SOURCE_EPSG)
    if g.crs.to_epsg() != SOURCE_EPSG:
        raise SystemExit(f"the download came back in {g.crs}, expected EPSG:{SOURCE_EPSG}")
    if len(g) != EXPECTED_VILLAGES:
        raise SystemExit(f"{len(g)} polygons, expected {EXPECTED_VILLAGES}")
    g["VCODE"] = g["VCODE"].astype(int)
    print(f"village polygons: {len(g):,}, crs={g.crs}")

    bad = g[~g.geometry.is_valid]
    if len(bad):
        print(f"  repairing {len(bad)} invalid polygons with buffer(0)")
        g.loc[~g.geometry.is_valid, "geometry"] = g.loc[
            ~g.geometry.is_valid, "geometry"].buffer(0)
    empty = g[g.geometry.is_empty | g.geometry.isna()]
    if len(empty):
        raise SystemExit(f"{len(empty)} polygons are empty after repair")

    # ---- 1. the independent check: recomputed area vs the OTHER service's own value ----
    with open(AREAS, encoding="utf-8") as fh:
        stored = {int(r["VCODE"]): float(r["Shape_Area"])
                  for r in json.load(fh)["rows"]}
    if set(stored) != set(g["VCODE"]):
        raise SystemExit(f"the two services carry different village sets: "
                         f"{len(set(stored) - set(g['VCODE']))} / "
                         f"{len(set(g['VCODE']) - set(stored))} either way")
    g["area_m2"] = g.geometry.area
    g["stored_m2"] = g["VCODE"].map(stored)
    rel = ((g["area_m2"] - g["stored_m2"]).abs()
           / g["stored_m2"].where(g["stored_m2"] > 0, 1.0))
    off = g[rel > AREA_REL_TOL]
    print(f"\n  recomputed area against {AREA_SERVICE}'s stored Shape_Area:")
    print(f"    worst relative difference {rel.max():.2e} over {len(g):,} polygons "
          f"(band {AREA_REL_TOL:.0e})")
    print(f"    outside the band: {len(off)}")
    for _, r in off.head(6).iterrows():
        print(f"      VCODE {int(r['VCODE'])}: {r['area_m2'] / 1e6:,.3f} km² "
              f"vs {r['stored_m2'] / 1e6:,.3f}")
    if len(off) > MAX_AREA_MISMATCH:
        raise SystemExit(f"{len(off)} polygons differ from the other service's stored area "
                         "-- the two services are publishing different geometry and the "
                         "counts may not belong to these shapes")
    print("    Two separate map services, one giving the rings and the other its own copy "
          "of the\n    area. Agreement to floating-point noise says they are the same "
          "layer (§12).")

    # ---- 2. the polygons tile the country ----
    km2 = g["area_m2"].sum() / 1e6
    cov = km2 / LAOS_KM2
    print(f"\n  total area {km2:,.0f} km² against Laos's {LAOS_KM2:,} — {cov:.1%}")
    if cov < MIN_COVERAGE or cov > 1.02:
        raise SystemExit(f"the village polygons cover {cov:.1%} of Laos, which is not a "
                         "tiling -- check the download")
    qs = g["area_m2"].div(1e6).quantile([0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    print("    area per village, km²: "
          + "  ".join(f"p{int(q * 100)}={v:,.1f}" for q, v in qs.items()))
    big = g[g["area_m2"] > 200e6]
    print(f"    {len(big)} polygons over 200 km², {big['area_m2'].sum() / km2 / 1e6:.1%} "
          "of the area — why sources/la_grid.py exists")

    # ---- 3. every village in la.csv has a polygon, and nothing is spare ----
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/la.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = df.drop_duplicates("geo_id")[["geo_id", "geo_name", "note"]]
    cen["vcode"] = cen["geo_id"].str.replace("^LA-", "", regex=True).astype(int)
    cen["province"] = cen["note"].str.extract(r"province=([^;]+)")[0].str.strip()
    cen["district"] = cen["note"].str.extract(r"district=([^;]+)")[0].str.strip()

    missing = sorted(set(cen["vcode"]) - set(g["VCODE"]))
    spare = sorted(set(g["VCODE"]) - set(cen["vcode"]))
    print(f"\n  the join, both ways (§12):")
    print(f"    villages in la.csv        {len(cen):>6,}")
    print(f"    polygons downloaded       {len(g):>6,}")
    print(f"    la.csv rows with no polygon {len(missing):>4} {missing[:5]}")
    print(f"    polygons with no la.csv row {len(spare):>4} {spare[:5]}")
    if missing or spare:
        raise SystemExit("join FAILED")
    print("    Exact by construction: both sides are the same LSB layer, keyed on VCODE. "
          "This\n    is a guard against a stale raw file, not a reconciliation.")

    # ---- 4. write, in EPSG:4326 like every other units file here ----
    meta = cen.set_index("vcode")
    out = g[["VCODE", "geometry"]].copy()
    out["unit"] = "LA-" + out["VCODE"].astype(str)
    out["name"] = out["VCODE"].map(meta["geo_name"])
    out["province"] = out["VCODE"].map(meta["province"])
    out["district"] = out["VCODE"].map(meta["district"])
    if out[["name", "province", "district"]].isna().any().any():
        raise SystemExit("a polygon came out of the join with no census attributes")
    out = out.to_crs(4326)

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "district", "province", "geometry"]].to_file(
        OUT, layer="villages", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} polygons, EPSG:4326)")

    lut = pd.DataFrame({"geo_id": sorted(cen["geo_id"])})
    lut["unit"] = lut["geo_id"]
    lut.to_csv(LOOKUP, index=False)
    print(f"wrote {LOOKUP} ({len(lut):,} rows)")


if __name__ == "__main__":
    main()
