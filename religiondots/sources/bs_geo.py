"""The Bahamas — boundaries for the 18 census islands, dissolved from 32 districts.

Writes data/geo/bs/bs_islands.gpkg and data/geo/bs/bs_lookup.csv.

**NOBODY PUBLISHES THE CENSUS'S OWN TIER, SO IT IS BUILT.** BNSI counts religion on 18
islands (`sources/bs.py`); every boundary set for the Bahamas — OCHA COD-AB and
geoBoundaries alike, and they are the same geometry — publishes the **32 local-government
districts** created by the Local Government Act 1996. The districts nest inside the islands
exactly, so the island tier is a dissolve and not an approximation, and `GROUPING` below is
the whole content of this file.

**TWENTY-SIX OF THE THIRTY-TWO CARRY THEIR ISLAND IN THEIR NAME** — `North Abaco`, `Central
Andros`, `South Eleuthera`, `West Grand Bahama`. The six that do not are cays, and each one
was checked against BNSI's own publications rather than against a map:

    Hope Town, Grand Cay, Moore's Island  -> Abaco   named in the 2010 ABACO settlement lists
    Mangrove Cay                          -> Andros  named in ANDROS 2010 CENSUS REPORT p106
    Black Point                           -> Exuma   enumeration district 420201 in EXUMA AND
                                                     CAYS POPULATION BY SETTLEMENT: 2010
    City of Freeport                      -> Grand Bahama

**AND HARBOUR ISLAND AND SPANISH WELLS ARE CENSUS ISLANDS IN THEIR OWN RIGHT**, not part of
Eleuthera, even though both sit off its northern end and are administered from it in every
other context. BNSI numbers them 12 and 18 on the questionnaire's island list, so the census
tier and the geographic intuition disagree here and the census wins. Eleuthera is exactly
its three own districts.

**THE CHECK IS THE PARTITION, ASSERTED BOTH WAYS.** Every district appears in exactly one
island's list, every district in the boundary file is claimed, and the 18 island codes are
the same 18 `sources/bs.py` wrote. A district silently dropped would take its people with
it; a district claimed twice would double its area. Neither can pass.

Usage:
    python sources/bs_geo.py --fetch    one ~1.0 MB shapefile zip from HDX
    python sources/bs_geo.py            rebuild from data/raw/bs/
"""

import os
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bs")
OUT_DIR = os.path.join(ROOT, "data", "geo", "bs")
OUT = os.path.join(OUT_DIR, "bs_islands.gpkg")
LOOKUP = os.path.join(OUT_DIR, "bs_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "bs.csv")

ZIP_URL = ("https://data.humdata.org/dataset/692fc891-1e44-4413-9e88-999349c86ca7/"
           "resource/d521054d-7272-4368-abf2-8a5d9f2b2ae7/download/bhs_adm1_gdams_2009.zip")
ZIP_NAME = "bhs_adm1_gdams_2009.zip"
SHP = "BHS_adm1.shp"
EXPECTED_DISTRICTS = 32

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# census island code -> the local-government districts on it. Codes are BNSI's own, from
# the 2022 questionnaire; see sources/bs.py ISLANDS.
GROUPING = {
    1:  ["New Providence"],
    2:  ["City of Freeport", "West Grand Bahama", "East Grand Bahama"],
    3:  ["North Abaco", "Central Abaco", "South Abaco",
         "Hope Town", "Grand Cay", "Moore's Island"],
    4:  ["Acklins"],
    5:  ["North Andros", "Central Andros", "South Andros", "Mangrove Cay"],
    6:  ["Berry Islands"],
    7:  ["Biminis"],
    8:  ["Cat Island"],
    9:  ["Crooked Island"],
    10: ["North Eleuthera", "Central Eleuthera", "South Eleuthera"],
    11: ["Exuma", "Black Point"],
    12: ["Harbour Island"],
    13: ["Inagua"],
    14: ["Long Island"],
    15: ["Mayaguana"],
    16: ["Ragged Island"],
    17: ["San Salvador", "Rum Cay"],
    18: ["Spanish Wells"],
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=900, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)
    with open(dest, "rb") as fh:
        magic = fh.read(2)
    if magic != b"PK":
        raise SystemExit(f"{dest} is not a zip -- starts {magic!r}")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zp = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zp):
        raise SystemExit(f"missing {zp} -- run with --fetch first")
    with zipfile.ZipFile(zp) as z:
        if SHP not in z.namelist():
            raise SystemExit(f"{zp} has no {SHP} -- it holds {z.namelist()[:8]}")
    g = gpd.read_file(f"zip://{zp}!{SHP}")
    # §12 (Chile): a read that succeeds is not a read that returned data.
    if len(g) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{SHP} returned {len(g)} features, expected "
                         f"{EXPECTED_DISTRICTS}")
    if "NAME_1" not in g.columns:
        raise SystemExit(f"{SHP} has no NAME_1 -- columns are {list(g.columns)}")
    print(f"COD-AB ADM1: {len(g)} districts, crs={g.crs}")

    # ---- the partition, both ways ----
    claimed = [d for ds in GROUPING.values() for d in ds]
    dupes = sorted({d for d in claimed if claimed.count(d) > 1})
    if dupes:
        raise SystemExit(f"GROUPING claims these districts more than once: {dupes}")

    have = {fold(n): n for n in g["NAME_1"]}
    if len(have) != len(g):
        raise SystemExit("two districts in the boundary file fold onto one name")

    want = {fold(d): d for d in claimed}
    missing = sorted(want[k] for k in want if k not in have)
    spare = sorted(have[k] for k in have if k not in want)
    print("\n  the partition, both ways (§12):")
    print(f"    districts in GROUPING       {len(claimed):>4}")
    print(f"    districts in the shapefile  {len(g):>4}")
    print(f"    GROUPING with no polygon    {len(missing):>4}  {missing}")
    print(f"    polygons with no island     {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("the district -> island grouping is not a partition -- a dropped "
                         "district takes its population with it and a doubled one doubles "
                         "its area")

    # ---- the island codes must be the ones sources/bs.py wrote ----
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/bs.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    cen = (df[df["geo_level"] == "island"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    codes = {f"{c:02d}" for c in GROUPING}
    if codes != set(census):
        raise SystemExit(f"GROUPING has islands {sorted(codes)} and bs.csv has "
                         f"{sorted(census)}")

    # ---- dissolve ----
    lut = {fold(d): f"{code:02d}" for code, ds in GROUPING.items() for d in ds}
    g["unit"] = g["NAME_1"].map(lambda n: lut[fold(n)])
    g["name"] = g["unit"].map(census)
    isl = g.dissolve(by="unit", as_index=False)[["unit", "name", "geometry"]]
    if len(isl) != len(GROUPING):
        raise SystemExit(f"dissolve produced {len(isl)} islands, expected {len(GROUPING)}")

    isl = isl.sort_values("unit").reset_index(drop=True)
    areas = isl.to_crs(3857).area / 1e6
    print(f"\n  {len(isl)} islands:")
    for (_, row), a in zip(isl.iterrows(), areas):
        ds = GROUPING[int(row["unit"])]
        print(f"    {row['unit']}  {row['name']:<26} {a:>9,.0f} km2   "
              f"{len(ds)} district(s): {', '.join(ds)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    isl.to_file(OUT, layer="islands", driver="GPKG")
    print(f"\nwrote {OUT} ({len(isl)} polygons)")

    pd.DataFrame({"geo_id": sorted(census), "unit": sorted(census)}).to_csv(
        LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(census)} rows)")


if __name__ == "__main__":
    main()
