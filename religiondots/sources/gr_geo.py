"""Greece — placement polygons: the 6,137 LAUs, from the GISCO LAU file already on disk.

Writes data/geo/gr/gr_lau.gpkg with, per LAU:
    lau    8-digit ELSTAT LAU code
    nuts3  its NUTS 3 region, from the GISCO correspondence workbook
    unit   its NUTS 2 region, which is the first four characters of that
    pop    the workbook's POPULATION, used as a placement weight
    name   LAU NAME LATIN, because spec's audience reads Latin characters

Usage:
    python sources/gr_geo.py

TWO FILES THAT WERE ALREADY HERE, AND ONE JOIN. §9y found that Spain's counting geography is
derivable from its placement geography with no join at all, because INE's municipal code
carries the province in its first two digits. Greece is one step short of that: the LAU code
carries nothing, but the same GISCO download ships `EU-27-LAU-2021-NUTS-2021.xlsx`, whose
`EL` sheet maps every LAU to its NUTS 3 — and NUTS 3 to NUTS 2 *is* a prefix. So the join is
LAU-to-LAU between a shapefile and a workbook that were published together, which is the
cheapest kind there is, and the check that it worked is that both sides have 6,137 rows and
the intersection does too.

THE COUNTING GEOGRAPHY IS NUTS 2, WHICH IS 13 UNITS. That is 800,000 people each — better
than Kenya's counties (§9o, 1.01M) and less than half Russia's federal subjects. It is the
finest geography the ESS carries for Greece; the foreign half is available one level finer,
at NUTS 3, and that extra level is used for PLACEMENT rather than for counting (§8.2), which
is the only honest thing to do with a resolution that only half the data has.
"""

import os
import sys

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                       "LAU_RG_01M_2021_4326.shp")
LAU_XLSX = os.path.join(ROOT, "data", "geo", "lau2021",
                        "EU-27-LAU-2021-NUTS-2021.xlsx")
OUT_DIR = os.path.join(ROOT, "data", "geo", "gr")
OUT = os.path.join(OUT_DIR, "gr_lau.gpkg")

# The 13 periféreies. Romanised, because that is what the notes and the legend read in.
NUTS2 = {
    "EL30": "Attiki",
    "EL41": "Voreio Aigaio", "EL42": "Notio Aigaio", "EL43": "Kriti",
    "EL51": "Anatoliki Makedonia, Thraki", "EL52": "Kentriki Makedonia",
    "EL53": "Dytiki Makedonia", "EL54": "Ipeiros",
    "EL61": "Thessalia", "EL62": "Ionia Nisia", "EL63": "Dytiki Ellada",
    "EL64": "Sterea Ellada", "EL65": "Peloponnisos",
    # AND THE ONE THAT IS NOT A REGION. Mount Athos is NUTS `ELZZZ` — "extra-regio", the
    # code NUTS uses for territory belonging to no region — and it is one LAU of 1,811
    # people: an autonomous Orthodox monastic republic inside, but not part of, Kentriki
    # Makedonia. It is kept as its own counting unit rather than folded into the mainland,
    # because drawing it from Macedonia's mixture would put irreligious and Muslim dots on
    # the Holy Mountain. taxonomy/gr2024.py authors it instead.
    "ELZZ": "Agion Oros (Mount Athos)",
}


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (EL only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='EL'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip()
    print(f"  {len(g):,} Greek LAUs, crs={g.crs}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="EL")
    # ZERO-PADDING, AND IT BIT HERE RATHER THAN IN THE SHAPEFILE. Excel stores the LAU code
    # as a number, so the workbook writes Attica's 01010101 as 1010101 while the shapefile
    # keeps the leading zero. 644 of 6,137 codes — every LAU in the regions numbered 01-09 —
    # fail to join without this, and the failure is one-sided in a way that a naive check
    # would report as "the workbook is missing 644 rows".
    x["lau"] = x["LAU CODE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    x["lau"] = x["lau"].str.zfill(8)
    x["nuts3"] = x["NUTS 3 CODE"].astype(str).str.strip()
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    x["name"] = x["LAU NAME LATIN"].astype(str)
    print(f"  {len(x):,} rows, {x['nuts3'].nunique()} NUTS 3, "
          f"population {x['pop'].sum():,.0f}")

    # spec §8.1, both directions. A one-sided check would pass on a file that had silently
    # lost half its rows on the other side.
    only_shp = sorted(set(g["lau"]) - set(x["lau"]))
    only_xls = sorted(set(x["lau"]) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(x['lau'])):,} matched, "
          f"{len(only_shp)} shapefile-only, {len(only_xls)} workbook-only")
    if only_shp or only_xls:
        sys.exit(f"!! LAU codes do not agree: {only_shp[:5]} / {only_xls[:5]}")

    g = g.merge(x[["lau", "nuts3", "pop", "name"]], on="lau", how="left")
    g["unit"] = g["nuts3"].str[:4]

    units = sorted(g["unit"].unique())
    print(f"  {len(units)} NUTS 2 units: {units}")
    if set(units) != set(NUTS2):
        sys.exit(f"!! NUTS 2 set is not the expected 13: "
                 f"{sorted(set(units) ^ set(NUTS2))}")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate LAU codes")

    per_unit = g.groupby("unit")["pop"].sum()
    print("  population per NUTS 2:")
    for u in units:
        print(f"    {u}  {NUTS2[u]:<28} {per_unit[u]:>10,.0f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "nuts3", "unit", "pop", "name", "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} LAUs)")


if __name__ == "__main__":
    main()
