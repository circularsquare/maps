"""American Samoa - the survey's ten counties, the 2020 census count in each, and the placement grid.

Writes:
    data/geo/as/as_counties.gpkg   the 10 units, with census and survey populations
    data/geo/as/as_hexes.gpkg      Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/as/as_lookup.csv      unit -> census 2010, census 2020, survey 2015, Kontur, hexes

Usage:
    python sources/as_geo.py --fetch    TIGER/Line 2020 county subdivisions (32 KB), the 2020
                                        census population table (1 KB), Kontur AS (19 KB)
    python sources/as_geo.py            rebuild from data/raw/as/

## THE UNITS ARE TIGER/Line 2020 COUNTY SUBDIVISIONS

`tl_2020_60_cousub.zip`, U.S. Census Bureau, public domain: 16 features. The survey's ten columns
are the nine counties of Tutuila (Eastern District: Ituau, Ma'oputasi, Sa'ole, Sua, Vaifanua;
Western District: Lealataua, Leasina, Tualatai, Tualauta), joined by folded name, and Manu'a, which
the survey prints as one column and TIGER as the five counties of Manu'a District (COUNTYFP 020),
dissolved. Rose Island (uninhabited) and Swains Island (17 people in 2010, none in 2020) are not
in the survey and are not units. The polygons include each county's territorial water, which
TIGER counts in AWATER; hexes join on their centroid, so coastal hexes fall inside without a snap.

## THE POPULATION BASE IS THE 2020 CENSUS

`american-samoa-phc-table01.csv`, *Population of American Samoa: 2010 and 2020*, by district and
county. The survey's county totals are sample completion times one weight (sources/as.py), not
populations; the census count is. The table is asserted to close (counties to districts, districts
plus Rose and Swains to the territory) in both years.

## A WITNESS THE NAME JOIN CANNOT MAKE TRUE: THE SURVEY AGAINST THE 2010 CENSUS

The survey's county totals are read off Table 1.6 by column position. If a column were shifted or
two swapped, every total check in sources/as.py would still pass; the ratio of each survey county to
the census county of the same name would not (Maoputasi against Saole is 5x). The band is wide
because the survey's totals are known to be off by up to a third.

## KONTUR IS A WEIGHT INSIDE A COUNTY

Tutuila is a steep island whose people live on the coast and the Tafuna plain; Manu'a's villages
are on the shore of three islands.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "as")
GEO = os.path.join(ROOT, "data", "geo", "as")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "as.csv")

OUT_UNITS = os.path.join(GEO, "as_counties.gpkg")
OUT_HEXES = os.path.join(GEO, "as_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "as_lookup.csv")

TIGER = os.path.join(RAW, "tl_2020_60_cousub.zip")
PHC = os.path.join(RAW, "american-samoa-phc-table01.csv")
KONTUR_GPKG = os.path.join(KONTUR, "kontur_population_AS_20231101.gpkg")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
DOWNLOADS = {
    TIGER: ("https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/tl_2020_60_cousub.zip",
            b"PK\x03\x04", 31_892, "ARX2J3O5XCT4UY34STD6DVTJPC4CM737"),
    PHC: ("https://www2.census.gov/programs-surveys/decennial/2020/data/island-areas/"
          "american-samoa/population-and-housing-unit-counts/american-samoa-phc-table01.csv",
          b"\xef\xbb\xbfPopul", 1_145, "UGGNZBLDTO7MQQBEAJCHLPK3QGZ3GFU6"),
    KONTUR_GPKG + ".gz": ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/"
                          "kontur_datasets/kontur_population_AS_20231101.gpkg.gz",
                          b"\x1f\x8b", 18_701, None),
}

UNITS = 10
TIGER_FEATURES = 16
MANUA = "Manu'a"
MANUA_COUNTYFP = "020"
MANUA_COUNTIES = {"Faleasao", "Fitiuta", "Ofu", "Olosega", "Ta'u"}
NOT_UNITS = {"Rose Island", "Swains Island"}
CENSUS_2020 = 49_710
CENSUS_2010 = 55_519

SNAP_M = 500
UTM = "EPSG:32702"
EQ = "EPSG:6933"
SURVEY_OVER_CENSUS_2010 = (0.70, 1.50)     # per county; the survey's totals are sample completion
KONTUR_RATIO = (0.70, 1.40)                # Kontur 2023 over the 2020 census, territory
KONTUR_UNIT_BAND = (0.50, 2.00)            # the same, per county


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests
    from fetch_checks import digest

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    for dst, (url, magic, size, dig) in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            print(f"  have {os.path.basename(dst)} ({size:,} bytes)")
            continue
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        body = r.content
        if not body.startswith(magic):
            raise SystemExit(f"{os.path.basename(dst)} starts {body[:16]!r}, not {magic!r}")
        if len(body) != size or (dig and digest(body) != dig):
            raise SystemExit(f"{os.path.basename(dst)}: {len(body):,} bytes, digest {digest(body)}; "
                             f"pinned {size:,} and {dig}")
        with open(dst + ".part", "wb") as fh:
            fh.write(body)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(body):,} bytes)")


def unpack():
    gz = KONTUR_GPKG + ".gz"
    if not os.path.exists(KONTUR_GPKG):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(KONTUR_GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR_GPKG + ".part", KONTUR_GPKG)
    with open(KONTUR_GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{KONTUR_GPKG} is not a GeoPackage")


def read_phc():
    """{area: (2010, 2020)} from the census table, every closure asserted."""
    with open(PHC, encoding="utf-8-sig", newline="") as fh:
        grid = list(csv.reader(fh))
    out = {}
    for r in grid:
        if len(r) < 3 or not r[0].strip() or not re.fullmatch(r"[\d,]+", r[1].strip()):
            continue
        name = r[0].strip()
        if name in out:
            raise SystemExit(f"census table: {name!r} twice")
        out[name] = (int(r[1].replace(",", "")), int(r[2].replace(",", "")))
    if out.get("American Samoa") != (CENSUS_2010, CENSUS_2020):
        raise SystemExit(f"census table territory row {out.get('American Samoa')}, expected "
                         f"{(CENSUS_2010, CENSUS_2020)}")
    districts = {"Eastern District": ["Ituau", "Ma'oputasi", "Sa'ole", "Sua", "Vaifanua"],
                 "Manu'a District": sorted(MANUA_COUNTIES),
                 "Western District": ["Lealataua", "Leasina", "Tualatai", "Tualauta"]}
    for y in (0, 1):
        for d, counties in districts.items():
            s = sum(out[f"{c} county"][y] for c in counties)
            if s != out[d][y]:
                raise SystemExit(f"census table: {d} counties sum to {s}, row prints {out[d][y]}")
        s = sum(out[d][y] for d in districts) + out["Rose Island"][y] + out["Swains Island"][y]
        if s != out["American Samoa"][y]:
            raise SystemExit(f"census table: districts sum to {s}, territory {out['American Samoa'][y]}")
    return out, districts


def _join_on_centroids(pts, polys, key):
    import geopandas as gpd

    j = gpd.sjoin(pts, polys[[key, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    got = j[key].copy()
    out = got.isna()
    snapped = 0
    if out.any():
        near = gpd.sjoin_nearest(pts.loc[out].to_crs(UTM), polys[[key, "geometry"]].to_crs(UTM),
                                 how="left", max_distance=SNAP_M, distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")]
        s = near[key].dropna()
        got.loc[s.index] = s
        snapped = len(s)
    return got, int(out.sum()), snapped


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack()
    for p in (TIGER, PHC, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run sources/as.py and this script with --fetch")

    survey = pd.read_csv(NORM, keep_default_na=False, na_values=[""]).groupby("geo_id")["count"].sum()
    if len(survey) != UNITS:
        raise SystemExit(f"{len(survey)} counties in as.csv, expected {UNITS}")
    phc, districts = read_phc()

    # ---- 1. units
    cs = geo_checks.read_layer(f"zip://{TIGER}", "TIGER 2020 COUSUB 60")
    if len(cs) != TIGER_FEATURES:
        raise SystemExit(f"TIGER COUSUB 60 has {len(cs)} features, expected {TIGER_FEATURES}")
    cs["key"] = cs["NAME"].map(fold)
    if cs["key"].duplicated().any():
        raise SystemExit(f"repeated county names: {sorted(cs.loc[cs['key'].duplicated(), 'NAME'])}")
    manua = cs[cs["COUNTYFP"] == MANUA_COUNTYFP]
    if set(manua["NAME"]) != MANUA_COUNTIES:
        raise SystemExit(f"Manu'a District's counties are {sorted(manua['NAME'])}, expected "
                         f"{sorted(MANUA_COUNTIES)}")
    parts, used = [], set(manua.index)
    for u in survey.index:
        if u == MANUA:
            g = manua.dissolve()
            census = phc["Manu'a District"]
            names = "+".join(sorted(manua["NAME"]))
        else:
            hits = cs.index[(cs["key"] == fold(u)) & (cs["COUNTYFP"] != MANUA_COUNTYFP)].tolist()
            if len(hits) != 1:
                raise SystemExit(f"survey county {u!r} matched {len(hits)} TIGER features")
            used.add(hits[0])
            g = cs.loc[[hits[0]]]
            census = phc[f"{cs.loc[hits[0], 'NAME']} county"]
            names = cs.loc[hits[0], "NAMELSAD"]
        parts.append(dict(unit=u, tiger=names, census_2010=census[0], census_2020=census[1],
                          survey_2015=int(survey[u]), geometry=g.geometry.union_all()))
    left = set(cs.loc[sorted(set(cs.index) - used), "NAME"])
    if left != NOT_UNITS:
        raise SystemExit(f"TIGER features in no unit: {sorted(left)}, expected {sorted(NOT_UNITS)}")
    units = gpd.GeoDataFrame(parts, geometry="geometry", crs=cs.crs)
    units["area_km2"] = units.to_crs(EQ).area / 1e6
    if int(units["census_2020"].sum()) + phc["Rose Island"][1] + phc["Swains Island"][1] != CENSUS_2020:
        raise SystemExit("the ten units' 2020 census counts do not close on the territory")
    print(f"TIGER 2020 COUSUB 60: {UNITS}/{UNITS} units (9 counties by folded name, Manu'a District "
          f"dissolved from 5); not units: {sorted(NOT_UNITS)} "
          f"({phc['Rose Island'][1] + phc['Swains Island'][1]} people in 2020, "
          f"{phc['Rose Island'][0] + phc['Swains Island'][0]} in 2010)")

    # ---- 2. the survey's columns against the census of the same name
    print("\n  survey 2015 (Table 1.6 total row) against the census:")
    bad = []
    for r in units.itertuples():
        k = r.survey_2015 / r.census_2010
        print(f"    {r.unit:<10} survey {r.survey_2015:>6,}  census 2010 {r.census_2010:>6,} ({k:4.2f}x)  "
              f"2020 {r.census_2020:>6,}  {r.area_km2:7.1f} km2 with water  ({r.tiger})")
        if not SURVEY_OVER_CENSUS_2010[0] <= k <= SURVEY_OVER_CENSUS_2010[1]:
            bad.append(f"{r.unit} {k:.2f}x")
    if bad:
        raise SystemExit(f"survey county totals outside {SURVEY_OVER_CENSUS_2010} of the 2010 census: "
                         f"{bad}; a column of Table 1.6 may be shifted")
    print(f"  witness: every survey county is within {SURVEY_OVER_CENSUS_2010} of its 2010 census count "
          f"(territory {int(units['survey_2015'].sum()) / CENSUS_2010:.3f}x)")

    # ---- 3. Kontur
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur AS")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    unit_of, n_out, n_snap = _join_on_centroids(pts, units, "unit")
    lost_mask = unit_of.isna()
    lost = float(pts.loc[lost_mask, "pop"].sum())
    total_k = float(pts["pop"].sum())
    print(f"\nKontur hexes: {len(hexes):,}, population {total_k:,.0f}; {n_out:,} centroids in no unit, "
          f"{n_snap:,} snapped within {SNAP_M} m, {int(lost_mask.sum()):,} dropped "
          f"({lost:,.0f} people, {100 * lost / total_k:.3f}%)")
    if lost / total_k > 0.02:
        raise SystemExit("more than 2% of Kontur falls outside every unit")
    keep = ~lost_mask
    out = gpd.GeoDataFrame({"unit": unit_of[keep].to_numpy(), "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_2020
    print(f"  Kontur 2023 {tot:,.0f} against the 2020 census {CENSUS_2020:,}: ratio {ratio:.3f}")
    if not KONTUR_RATIO[0] <= ratio <= KONTUR_RATIO[1]:
        raise SystemExit(f"ratio outside {KONTUR_RATIO}; check the download")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print("\n  per unit, census 2020 against Kontur 2023:")
    for r in units.sort_values("census_2020", ascending=False).itertuples():
        print(f"    {r.unit:<10}{r.census_2020:>7,}  Kontur {r.kontur_pop:>7,}  "
              f"{r.kontur_pop / r.census_2020:5.2f}x  {r.hexes:>4,} hexes")
    print(f"  median hexes per unit {units['hexes'].median():.0f}")
    geo_checks.ratio_band(dict(zip(units["unit"], units["census_2020"])),
                          dict(zip(units["unit"], units["kontur_pop"])),
                          *KONTUR_UNIT_BAND, what="county")

    # ---- write
    os.makedirs(GEO, exist_ok=True)
    cols = ["unit", "tiger", "census_2010", "census_2020", "survey_2015", "kontur_pop", "hexes",
            "area_km2", "geometry"]
    units[cols].to_file(OUT_UNITS, layer="counties", driver="GPKG")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "tiger", "census_2010", "census_2020", "survey_2015", "kontur_2023",
                    "hexes", "area_km2"])
        for r in units.sort_values("unit").itertuples():
            w.writerow([r.unit, r.tiger, r.census_2010, r.census_2020, r.survey_2015, r.kontur_pop,
                        r.hexes, round(r.area_km2, 2)])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES} ({len(out):,} hexes)\nwrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
