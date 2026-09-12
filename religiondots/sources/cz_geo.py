"""Czechia — build the FINEST-UNIT polygon layer, and the list of obce it replaces.

ČSÚ publishes the same 78 religion categories at two overlapping geographies:

    obec           6,254 units, the whole country, median population 435
    city district    142 units, the statutory cities only, median 6,189

The 142 city districts SUBDIVIDE ten-odd obce; they are an alternative to their parent, not
a child of it (sources/cz.md §3).  So the finest complete cover of the country is

    every obec that is not subdivided,  PLUS  every city district

which is what this writes.  Why bother: Prague is 1,301,432 people in ONE obec, 12.4% of the
country in a single polygon, and the six obce over 100,000 hold 22.5% between them.  Placing
dots uniformly across Prague-the-polygon is the worst case for a dot map, and at city-district
level Prague is 57 units instead.  This is MEASURED data, not an allocation -- ČSÚ publishes
the counts at both levels -- so nothing here is `derived` and every row may still ring.

The parent obec of each city district is derived SPATIALLY rather than from a lookup table.
City districts nest exactly inside obce by construction, so a representative-point join is
not an approximation, and it avoids fetching ČSÚ's territorial register for one column.  The
same trick is how countries.py resolves Canada's dissemination areas.

Writes:
    data/geo/cz/cz_finest.gpkg      one layer, `kod` + `nazev` + `level`
    data/geo/cz/cz_replaced.csv     the obec codes that city districts replace

Usage:
    python sources/cz_geo.py
"""

import os
import sys

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "cz")

OBCE = os.path.join(GEO, "obce", "csu_geodb_sde_CISOB_obyvatelstvo_etl_20210326.gpkg")
MC = os.path.join(GEO, "l10044", "csu_geodb_sde_CISMC_obyvatelstvo_etl_20210326.gpkg")
OUT_GPKG = os.path.join(GEO, "cz_finest.gpkg")
OUT_CSV = os.path.join(GEO, "cz_replaced.csv")


def main():
    for p in (OBCE, MC):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — see sources/cz_geo.md for the download")

    obce = gpd.read_file(OBCE)[["kod", "nazev", "geometry"]]
    mc = gpd.read_file(MC)[["kod", "nazev", "geometry"]]
    obce["kod"] = obce["kod"].astype(str)
    mc["kod"] = mc["kod"].astype(str)
    print(f"obce {len(obce):,} ({obce.crs})   city districts {len(mc):,} ({mc.crs})")
    if obce.crs != mc.crs:
        mc = mc.to_crs(obce.crs)

    # Which obec does each city district sit in?  Representative point, so a district whose
    # centroid falls outside its own concave outline still lands in the right parent.
    pts = mc.copy()
    pts["geometry"] = mc.geometry.representative_point()
    joined = gpd.sjoin(pts[["kod", "geometry"]], obce[["kod", "geometry"]],
                       how="left", predicate="within", lsuffix="mc", rsuffix="ob")
    joined = joined[~joined.index.duplicated(keep="first")]

    orphan = joined["kod_ob"].isna().sum()
    if orphan:
        raise SystemExit(f"{orphan} city districts fell outside every obec — the join is "
                         "wrong, not the data")

    parents = sorted(set(joined["kod_ob"]))
    print(f"\n{len(mc):,} city districts sit inside {len(parents)} obce:")
    counts = joined.groupby("kod_ob").size().sort_values(ascending=False)
    names = dict(zip(obce["kod"], obce["nazev"]))
    for kod, n in counts.items():
        print(f"  {names.get(kod, '?'):<24} {kod}   {n:>3} districts")

    keep = obce[~obce["kod"].isin(parents)].copy()
    keep["level"] = "municipality"
    mc = mc.copy()
    mc["level"] = "city_district"
    finest = pd.concat([keep, mc], ignore_index=True)
    finest = gpd.GeoDataFrame(finest, geometry="geometry", crs=obce.crs)

    print(f"\nfinest cover: {len(keep):,} obce + {len(mc):,} city districts "
          f"= {len(finest):,} polygons")
    if finest["kod"].duplicated().any():
        raise SystemExit("duplicate kod in the merged layer")

    finest.to_file(OUT_GPKG, layer="cz_finest", driver="GPKG")
    pd.DataFrame({"kod": parents}).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_GPKG}")
    print(f"wrote {OUT_CSV} ({len(parents)} obec codes)")


if __name__ == "__main__":
    main()
