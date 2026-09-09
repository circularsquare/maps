"""Liberia — boundaries and populations for the 15 counties.

Writes data/geo/lr/lr_counties.gpkg and data/geo/lr/lr_lookup.csv.

  * **boundaries** — OCHA COD-AB Liberia (`cod-ab-lbr`), the **shapefile** bundle rather than
    the geodatabase on §12's Chile rule, read with `engine="fiona"`. 15 ADM1 features,
    `LR01`..`LR15`, alphabetical by name.
  * **populations** — **the 2022 census itself**, Table A4 of LISGIS's *Final Results*, not
    COD-PS.

## WHY THE CENSUS AND NOT COD-PS, WHICH IS WHAT MOST COUNTRIES HERE USE

Liberia is the unusual case where the *religion* margin and the *population* margin come out
of the same table set. `sources/lr.py` fits the county-by-religion table to two exact census
margins: county totals from Table A4 and national religion totals from Table A13. Both sum to
5,250,187 to the person. Substituting a projection for one of them would make the two margins
disagree by whatever the projection has drifted, and the fitted table would then be forced to
absorb that drift as though it were religion.

So the population here is not an estimate of Liberia today; it is **the census night count,
2022-11-10/11**, and `countries.py`'s `grain` says so.

## THE NAMES AGREE ON FOURTEEN OF FIFTEEN AND THE FIFTEENTH IS A SPACE

COD writes `Rivercess`; LISGIS's own table writes `River Cess`. Nothing else differs, so the
join folds out non-letters and is asserted to be a bijection. The pcodes are alphabetical by
COD's spelling, which puts `River Gee` (LR13) before `Rivercess` (LR14) — the order is COD's
and is not used as a key.

Usage:
    python sources/lr_geo.py --fetch    one ~2.2 MB zip from HDX
    python sources/lr_geo.py            rebuild from data/raw/lr/
"""

import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lr")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "lr")
OUT = os.path.join(OUT_DIR, "lr_counties.gpkg")
LOOKUP = os.path.join(OUT_DIR, "lr_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

ZIP_NAME = "lbr_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/bbb2f45c-3d7a-4ad3-afc5-459303dbc8f4/resource/"
           "11dfa07f-3ec3-4729-81dc-2e5abcc228f5/download/lbr_admin_boundaries.shp.zip")

EXPECTED_COUNTIES = 15

# 2022 Liberia Population and Housing Census, FINAL RESULTS, Table A4 "Distribution of the
# Population by Type of Residence and County", page 73. Transcribed from the PDF; the total is
# asserted against Table A1's national figure below, and Table A13's religion column asserts
# the same total independently in sources/lr.py.
CENSUS_2022 = {
    "Bomi": 133_705,
    "Bong": 467_561,
    "Gbarpolu": 95_995,
    "Grand Bassa": 293_689,
    "Grand Cape Mount": 178_867,
    "Grand Gedeh": 216_692,
    "Grand Kru": 109_342,
    "Lofa": 367_376,
    "Margibi": 304_946,
    "Maryland": 172_587,
    "Montserrado": 1_920_965,
    "Nimba": 621_841,
    "River Cess": 90_819,
    "River Gee": 124_653,
    "Sinoe": 151_149,
}

CENSUS_TOTAL = 5_250_187          # Table A1 and Table A2, the 2022 census night count

# COD's spelling -> the census's, where they differ. One entry, and it is a space.
COD_TO_CENSUS = {"Rivercess": "River Cess"}


def fold(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dst = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dst) and os.path.getsize(dst) > 500_000:
        print(f"  have {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    else:
        req = urllib.request.Request(ZIP_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        # §5a: a 200 is not a download.
        if data[:2] != b"PK":
            raise SystemExit(f"{ZIP_NAME} is not a zip — starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    with zipfile.ZipFile(dst) as z:
        z.extractall(SHP_DIR)


def main():
    if "--fetch" in sys.argv:
        fetch()

    shp = os.path.join(SHP_DIR, "lbr_admin1.shp")
    if not os.path.exists(shp):
        raise SystemExit(f"missing {shp} — run with --fetch first")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != EXPECTED_COUNTIES:
        raise SystemExit(f"{shp} has {len(g)} features, expected {EXPECTED_COUNTIES}")
    print(f"COD-AB Liberia admin1: {len(g)} counties, crs={g.crs}, "
          f"valid_on {sorted(set(g['valid_on']))}")

    g = g.rename(columns={"adm1_pcode": "unit", "adm1_name": "cod_name"})
    g["name"] = g["cod_name"].map(lambda n: COD_TO_CENSUS.get(n, n))

    # The join, asserted as a bijection rather than eyeballed.
    census_by_fold = {fold(k): k for k in CENSUS_2022}
    if len(census_by_fold) != len(CENSUS_2022):
        raise SystemExit("two census county names fold to the same key")
    missing = sorted(n for n in g["name"] if fold(n) not in census_by_fold)
    if missing:
        raise SystemExit(f"COD counties with no census population: {missing}")
    extra = sorted(set(CENSUS_2022) - set(g["name"]))
    if extra:
        raise SystemExit(f"census counties with no COD polygon: {extra}")
    if sum(CENSUS_2022.values()) != CENSUS_TOTAL:
        raise SystemExit(f"Table A4 transcription sums to {sum(CENSUS_2022.values()):,}, "
                         f"not the census total {CENSUS_TOTAL:,}")

    g["pop"] = g["name"].map(CENSUS_2022).astype("int64")
    g["geo_id"] = g["unit"]

    # The county names differ from COD's on exactly one row; if that ever grows, the census
    # side has been re-cut and the transcription needs re-reading rather than re-mapping.
    renamed = sorted(n for n in g["cod_name"] if n in COD_TO_CENSUS)
    print(f"  names taken from the census where they differ from COD: {renamed}")

    print(f"\n  {len(g)} counties, {int(g['pop'].sum()):,} people "
          f"(2022 census night, Table A4):")
    for _i, r in g.sort_values("pop", ascending=False).iterrows():
        print(f"    {r['unit']}  {r['name']:<18}{int(r['pop']):>10,}  "
              f"{r['area_sqkm']:>9,.0f} km²")

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = g[["geo_id", "unit", "name", "cod_name", "pop", "area_sqkm", "geometry"]]
    keep.to_file(OUT, layer="counties", driver="GPKG")
    pd.DataFrame(keep.drop(columns="geometry")).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
