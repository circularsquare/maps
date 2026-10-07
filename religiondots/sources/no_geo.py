"""Norway — the placement layer, and the counting units it rolls up to.

Writes data/geo/no/no_lau.gpkg, one row per municipality (kommune), plus Svalbard's Kontur hexes
(below):

    lau      the SSB kommune number, zero-padded to 4 ("0301" Oslo)
    unit     its NUTS 3 2021 county, THE COUNTING UNIT, 11 of them (the 2020-2023 fylker)
    nuts2    the NUTS 2 2021 region the county sits in, 6 of them
    name     the kommune's name as GISCO spells it
    pop      its 2021 population, which is the dot-placement weight

NO DOWNLOAD. The polygons and the population are the GISCO LAU 2021 shapefile that has been on
disk since Poland (§9e). **The correspondence workbook beside it has no Norway sheet**: it is
`EU-27-LAU-2021-NUTS-2021.xlsx` and carries the 27 member states and nothing else, so the
Sweden and Finland route (se_geo.py, fi_geo.py) of reading the NUTS 3 code off the workbook
does not exist here. The county is read off the kommune number instead, whose first two digits
ARE the county number in SSB's 2020 classification, and the join is then checked against the
census's own county populations rather than trusted (below).

THE COUNTING UNIT IS THE 11 COUNTIES OF 2020-2023, because that is where the 2021 census
counts citizenship (Eurostat `cens_21ctz_r3`, NUTS 3 2021) and so where the foreign half is
measured. Norway went back to 15 counties on 1 January 2024; the census, GISCO LAU 2021 and
NUTS 2021 all predate that and all use the 11, which is spec §8.1's rule (the vintage the data
was published on). ESS gives Norway at NUTS 2 only, and in two vintages that do not nest in
each other; sources/no.py has that problem and says how it is handled.

SVALBARD, ADDED 2026-10-04 (Anita's ruling, ask/RULINGS.md: draw it as part of Norway so its
outline is not an empty hatch). Svalbard is in no county and no kommune, so it is one more
counting unit, `NO-21` (its ISO 3166-2 code), appended after the 356 kommuner. Its placement is
Kontur's SJ layer (H3 r8, 2023-11-01) inside Natural Earth's Svalbard map unit, and KONTUR ALONE
IS WRONG HERE: it puts about 1,900 of its 3,290 Svalbard people at Sveagruva, a mine that closed
in 2017 and has been cleared, and only about 350 at Longyearbyen. So the weight is SSB's count
(table 07430, 1 January 2026) shared over each settlement's hexes in Kontur's proportions, the
Faroes' calibration (fo_geo.py): Longyearbyen takes SSB's "Longyearbyen and Ny-Alesund" row,
Barentsburg its "Barentsburg and Pyramiden" row, Hornsund its own. Every other Svalbard hex is
kept at weight 0, so the unit reaches the places Kontur thinks are lived in (not_drawn.py's test)
and draws no dot there. Ny-Alesund and Pyramiden are inside SSB's group rows and are not split
out by it; at a few dozen people each they get weight 0.

Usage:
    python sources/no_geo.py
"""

import argparse
import os
import sys

import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                       "LAU_RG_01M_2021_4326.shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "no")
OUT = os.path.join(OUT_DIR, "no_lau.gpkg")

N_LAU = 356
N_NUTS3 = 11
N_NUTS2 = 6

# SSB county number (the first two digits of a 2020-2023 kommune number) -> NUTS 3 2021.
FYLKE_TO_NUTS3 = {
    "03": "NO081",   # Oslo
    "30": "NO082",   # Viken
    "34": "NO020",   # Innlandet
    "38": "NO091",   # Vestfold og Telemark
    "42": "NO092",   # Agder
    "11": "NO0A1",   # Rogaland
    "46": "NO0A2",   # Vestland
    "15": "NO0A3",   # Møre og Romsdal
    "50": "NO060",   # Trøndelag
    "18": "NO071",   # Nordland
    "54": "NO074",   # Troms og Finnmark
}

# GISCO'S `POP_2021` FOR NORWAY IS THE 1 JANUARY 2020 POPULATION, NOT 2021's. Its national sum is
# 5,367,580, which is SSB's figure for 1 January 2020 to the person, and the census (1 January
# 2021) is 5,391,370. The first version of this check compared against the census with a 0.5%
# tolerance and failed six counties, every one in the direction a year of growth or decline
# gives (Oslo and Viken low, Nordland and Innlandet high) rather than the equal-and-opposite
# pair a mis-assigned kommune gives. So the check is against SSB table 07459 at 1 January 2020
# and it is EXACT: a kommune number read into the wrong county cannot pass it, and it does not
# look at names at all ([[reference_name_join_wrong_neighbour]]).
SSB_POP_2020 = {
    "NO081": 693_494, "NO082": 1_241_165, "NO020": 371_385, "NO091": 419_396,
    "NO092": 307_231, "NO0A1": 479_892, "NO0A2": 636_531, "NO0A3": 265_238,
    "NO060": 468_702, "NO071": 241_235, "NO074": 243_311,
}
POP_GISCO = 5_367_580

KONTUR_SJ = os.path.join(ROOT, "data", "geo", "kontur", "kontur_population_SJ_20231101.gpkg.gz")
KONTUR_SJ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
                 "kontur_population_SJ_20231101.gpkg.gz")
NE_UNITS = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_map_units.geojson")
SVALBARD_UNIT = "NO-21"
# SSB table 07430, persons in the settlements, 2026H1 (1 January 2026), read 2026-10-04 from
# data.ssb.no/api/v0/en/table/07430. Longyearbyen and Ny-Alesund is the sum of its two rows
# (registered on the mainland 1,648; from abroad 864). The four rows sum to 2,914.
# (settlement point lat, lon), placement radius in km, SSB persons.
SVALBARD_SETTLEMENTS = {
    "Longyearbyen": ((78.2232, 15.6267), 8.0, 1_648 + 864),
    "Barentsburg":  ((78.0648, 14.2335), 5.0, 392),
    "Hornsund":     ((77.0013, 15.5428), 3.0, 10),
}
SVALBARD_POP = 2_914


def svalbard():
    """Svalbard's placement rows: every Kontur hex inside its outline, weighted by SSB's
    settlement counts (module docstring)."""
    import gzip
    import shutil

    import pandas as pd

    gpkg = KONTUR_SJ[:-3]
    if not os.path.exists(gpkg):
        if not os.path.exists(KONTUR_SJ):
            import requests
            r = requests.get(KONTUR_SJ_URL, timeout=300)
            r.raise_for_status()
            with open(KONTUR_SJ + ".part", "wb") as fh:
                fh.write(r.content)
            os.replace(KONTUR_SJ + ".part", KONTUR_SJ)
        with gzip.open(KONTUR_SJ, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)

    mu = gpd.read_file(NE_UNITS)
    outline = mu[mu["GEOUNIT"] == "Svalbard"]
    if len(outline) != 1:
        sys.exit(f"!! Natural Earth map units: {len(outline)} Svalbard features, expected 1")
    k = gpd.read_file(gpkg)
    k = k[k["population"] > 0].to_crs(4326).reset_index(drop=True)
    utm = "EPSG:32633"
    cent = k.to_crs(utm).centroid
    inside = cent.to_crs(4326).within(outline.geometry.iloc[0].buffer(0.05))
    k, cent = k[inside].reset_index(drop=True), cent[inside].reset_index(drop=True)
    print(f"  Kontur SJ: {len(k)} populated hexes inside Svalbard, {k['population'].sum():,.0f} "
          "people by Kontur")

    k["pop"] = 0.0
    k["name"] = "Svalbard"
    taken = pd.Series(False, index=k.index)
    for name, ((lat, lon), km, persons) in SVALBARD_SETTLEMENTS.items():
        pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(utm).iloc[0]
        near = (cent.distance(pt) <= km * 1000) & ~taken
        kp = float(k.loc[near, "population"].sum())
        if kp <= 0:
            sys.exit(f"!! no Kontur hex within {km} km of {name}")
        k.loc[near, "pop"] = k.loc[near, "population"] / kp * persons
        k.loc[near, "name"] = name
        taken |= near
        print(f"    {name}: {int(near.sum())} hexes, Kontur {kp:,.0f} -> SSB {persons:,}")
    left = k.loc[~taken, "population"].sum()
    print(f"    {int((~taken).sum())} other hexes ({left:,.0f} Kontur people, most of them at "
          "Sveagruva) kept at weight 0")
    if round(k["pop"].sum()) != SVALBARD_POP:
        sys.exit(f"!! Svalbard weight sums to {k['pop'].sum():,.1f}, not SSB's {SVALBARD_POP:,}")
    k["lau"] = SVALBARD_UNIT
    k["unit"] = SVALBARD_UNIT
    k["nuts2"] = "NO0B"
    return k[["lau", "unit", "nuts2", "pop", "name", "geometry"]]


def main():
    if not os.path.exists(LAU_SHP):
        sys.exit(f"missing {LAU_SHP} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (NO only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='NO'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(4)
    g["name"] = g["LAU_NAME"].astype(str)
    g["pop"] = g["POP_2021"].astype(float).fillna(0.0)
    print(f"  {len(g):,} Norwegian kommuner, crs={g.crs}, population {g['pop'].sum():,.0f}")

    stray = sorted(set(g["lau"].str[:2]) - set(FYLKE_TO_NUTS3))
    if stray:
        sys.exit(f"!! kommune numbers outside the 11 counties of 2020-2023: {stray}")
    g["unit"] = g["lau"].str[:2].map(FYLKE_TO_NUTS3)
    g["nuts2"] = g["unit"].str[:4]

    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} kommuner, got {len(g)}")
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate kommune numbers")
    if g["unit"].nunique() != N_NUTS3:
        sys.exit(f"!! expected {N_NUTS3} counties, got {g['unit'].nunique()}")
    if g["nuts2"].nunique() != N_NUTS2:
        sys.exit(f"!! expected {N_NUTS2} NUTS 2 regions, got {g['nuts2'].nunique()}")

    if int(g["pop"].sum()) != POP_GISCO:
        sys.exit(f"!! population {g['pop'].sum():,.0f} is not the expected {POP_GISCO:,} — the "
                 "GISCO vintage has changed and SSB_POP_2020 needs re-reading")
    per = g.groupby("unit").agg(kommuner=("lau", "size"), pop=("pop", "sum"))
    print("  population per county against SSB 07459 at 1 January 2020 (exact):")
    bad = []
    for u in sorted(per.index):
        c = SSB_POP_2020[u]
        flag = "  <-- off" if int(per.loc[u, "pop"]) != c else ""
        print(f"    {u}  {per.loc[u, 'pop']:>10,.0f}  SSB {c:>10,}  "
              f"{per.loc[u, 'kommuner']:>3} kommuner{flag}")
        if flag:
            bad.append(u)
    if bad:
        sys.exit(f"!! counties whose kommuner do not add up to SSB's own figure: {bad}")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if (g["pop"] <= 0).any():
        print(f"  !! {int((g['pop'] <= 0).sum())} kommuner have no population and will "
              "attract no dots")

    # Norway reaches 31°E at Vardø and 71°N at Nordkapp; Svalbard is not a kommune, the census
    # counts nobody there (NO0B is zero in cens_21ctz_r3), and it is appended below, after this
    # check, as its own unit.
    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    if not (4 < minx < 5.5 and 57.5 < miny < 58.5 and 30.5 < maxx < 31.5 and 70.5 < maxy < 71.5):
        sys.exit("!! bbox is not mainland Norway's — check the CRS and the country filter")

    print("Svalbard (unit NO-21):")
    sv = svalbard()
    import pandas as pd
    out = gpd.GeoDataFrame(
        pd.concat([g[["lau", "unit", "nuts2", "pop", "name", "geometry"]].to_crs(4326), sv],
                  ignore_index=True), geometry="geometry", crs=4326)

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT + ".part.gpkg", driver="GPKG", layer="lau")
    os.replace(OUT + ".part.gpkg", OUT)
    print(f"wrote {OUT}  ({len(g):,} kommuner and {len(sv):,} Svalbard hexes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="accepted for symmetry; the input is already on disk")
    ap.parse_args()
    main()
