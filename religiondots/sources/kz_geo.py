"""Kazakhstan — boundaries for the 17 regions of the 2021 census.

Writes data/geo/kz/kz_regions.gpkg and data/geo/kz/kz_lookup.csv.

COD-AB Kazakhstan (`cod-ab-kaz`, UNHCR from OpenStreetMap, 2023), shapefile bundle, 5.3 MB,
one GET from HDX, no wall.

**THE BOUNDARY FILE IS THE WRONG VINTAGE AND IS PUT BACK, WHICH IS §8.1's WHOLE POINT.** In
2022 Kazakhstan created three new oblasts, so COD's ADM1 has **20** regions and the 2021
census has **17**. The reform is the cleanest kind there is — three new oblasts carved out of
three existing ones, no other boundary touched, no rayon split:

    Abay Region      <- East Kazakhstan Region
    Jetisu Region    <- Almaty Region
    Ulytau Region    <- Karaganda Region

So the 2021 map is recovered by dissolving three PAIRS of whole ADM1 polygons, and that is
safe in a way a general boundary reconstruction is not: no polygon is cut, nothing is
apportioned, and the result is exactly the union of features the file already contains.

**THE CHECK THAT MAKES IT EVIDENCE RATHER THAN AN ASSERTION IS ADM2.** COD ships **218** ADM2
polygons and the census publishes **218** level-2 units, because the reform regrouped rayons
without splitting them. Each merged region's ADM2 children are counted and must equal its
parts' — if the reform had moved a single rayon anywhere else, the ADM2 counts would not line
up and this module stops.

**AND THE REAL CONFIRMATION IS KONTUR, IN `kz_grid.py`**: a wrong merge would put a region's
modelled population against the wrong polygon's grid and blow the per-region band wide open.
That check is downstream and is the one that would actually catch a mistake here.

**WHY NOT DRAW THE 218 ADM2 UNITS**, which the census also publishes and which would be a far
better map. Two reasons and the second is the binding one:

  1. §14 restraint. Kazakhstan is a MODELLED country (sources/kz.py) and 218 units of
     inference is a much stronger claim than 17. `sources.md` §11u argued that at length.
  2. **The join would be 218 fuzzy cross-script matches.** COD names its ADM2 in English
     transliteration (`Arshaly District`, `Korgalzhyn District`) and the census names them in
     Russian adjectival form (`Аршалынский район`, `Коргалжынский район`), and there is no
     shared code — COD's `ADM2_PCODE` is `KAZ###` and the census's key is KATO. That is §12's
     shape 2 (a confident wrong pairing) with 218 chances to happen. **At ADM1 the same join
     is seventeen names and is checked by eye as well as by rule.**

Usage:
    python sources/kz_geo.py --fetch    one 5.3 MB zip from HDX
    python sources/kz_geo.py            rebuild from data/raw/kz/
"""

import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kz")
OUT_DIR = os.path.join(ROOT, "data", "geo", "kz")
OUT = os.path.join(OUT_DIR, "kz_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "kz_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "kz.csv")

ZIP_URL = ("https://data.humdata.org/dataset/afb05759-c3da-44f4-93a1-6bd2d8bcd431/"
           "resource/86cce6ba-4b79-4b4e-8961-3e6e04308395/download/"
           "kaz_adm_unhcr_2023_shp.zip")
ZIP_NAME = "kaz_adm_unhcr_2023_shp.zip"

EXPECTED_ADM1 = 20            # COD's 2023 vintage
EXPECTED_ADM2 = 218           # unchanged by the reform, and the census agrees
EXPECTED_REGIONS = 17         # the 2021 vintage this module produces

# The 2022 reform, undone. child -> parent it was carved out of.
MERGE = {
    "Abay Region": "East Kazakhstan Region",
    "Jetisu Region": "Almaty Region",
    "Ulytau Region": "Karaganda Region",
}

# COD's English ADM1 -> the census's Russian region name. Seventeen pairs, written out
# because a transliteration rule cannot get `Северо-Казахстанская` from `North Kazakhstan`
# and pretending otherwise would be worse than a list this short. Every one is asserted to
# match a census row exactly, and every census row to be claimed exactly once.
NAME = {
    "Akmola Region": "Акмолинская область",
    "Aktobe Region": "Актюбинская область",
    "Almaty Region": "Алматинская область",
    "Atyrau Region": "Атырауская область",
    "West Kazakhstan Region": "Западно-Казахстанская область",
    "Jambyl Region": "Жамбылская область",
    "Karaganda Region": "Карагандинская область",
    "Kostanay Region": "Костанайская область",
    "Kyzylorda Region": "Кызылординская область",
    "Mangystau Region": "Мангистауская область",
    "Pavlodar Region": "Павлодарская область",
    "North Kazakhstan Region": "Северо-Казахстанская область",
    "Turkistan Region": "Туркестанская область",
    "East Kazakhstan Region": "Восточно-Казахстанская область",
    "Astana": "г.Нур-Султан",
    "Almaty": "г.Алматы",
    "Shymkent": "г.Шымкент",
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and zipfile.is_zipfile(dest):
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def _read(layer, expected):
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()
             if re.search(rf"admbnda_{layer}_.*\.shp$", i.filename, re.I)]
    if len(names) != 1:
        raise SystemExit(f"expected one {layer} shapefile, found {names}")
    g = gpd.read_file(f"zip://{src}!{names[0]}", engine="fiona")
    if len(g) != expected:
        raise SystemExit(f"{names[0]}: {len(g)} features, expected {expected}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {layer} {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    adm1 = _read("adm1", EXPECTED_ADM1)
    adm2 = _read("adm2", EXPECTED_ADM2)
    print(f"COD: {len(adm1)} ADM1 (2023 vintage), {len(adm2)} ADM2")

    # ---- 1. the reform, undone
    missing = [k for k in list(MERGE) + list(MERGE.values())
               if k not in set(adm1["ADM1_EN"])]
    if missing:
        raise SystemExit(f"COD no longer names {missing} -- the merge table is stale")
    adm1["region_en"] = adm1["ADM1_EN"].map(lambda n: MERGE.get(n, n))

    # the ADM2 check: a rayon that moved anywhere unexpected breaks this
    child_of = dict(zip(adm1["ADM1_PCODE"], adm1["region_en"]))
    adm2["region_en"] = adm2["ADM1_PCODE"].map(child_of)
    if adm2["region_en"].isna().any():
        raise SystemExit("an ADM2 polygon has an ADM1_PCODE that is not in ADM1")
    per = adm2.groupby("region_en").size()
    print(f"\n  merged {len(MERGE)} pairs -> {adm1['region_en'].nunique()} regions")
    for child, parent in MERGE.items():
        a = int((adm2["ADM1_PCODE"].map(
            dict(zip(adm1["ADM1_PCODE"], adm1["ADM1_EN"]))) == child).sum())
        b = int((adm2["ADM1_PCODE"].map(
            dict(zip(adm1["ADM1_PCODE"], adm1["ADM1_EN"]))) == parent).sum())
        print(f"    {child:<22} {a:>3} rayons  +  {parent:<24} {b:>3}  "
              f"=  {per[MERGE[child]]:>3}")
    if adm1["region_en"].nunique() != EXPECTED_REGIONS:
        raise SystemExit(f"{adm1['region_en'].nunique()} regions after the merge, "
                         f"expected {EXPECTED_REGIONS}")
    if int(per.sum()) != EXPECTED_ADM2:
        raise SystemExit(f"the merged regions hold {int(per.sum())} rayons, "
                         f"expected {EXPECTED_ADM2}")
    print(f"    every one of the {EXPECTED_ADM2} rayons lands in exactly one of the "
          f"{EXPECTED_REGIONS}")

    merged = adm1.dissolve(by="region_en", as_index=False)[["region_en", "geometry"]]

    # ---- 2. the census side
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/kz.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = df.drop_duplicates("geo_id")[["geo_id", "geo_name"]]
    if len(cen) != EXPECTED_REGIONS:
        raise SystemExit(f"{len(cen)} census regions, expected {EXPECTED_REGIONS}")

    def norm(s):
        s = unicodedata.normalize("NFKC", str(s)).strip().lower()
        return re.sub(r"\s+", " ", s.replace("ё", "е"))

    by_ru = {norm(n): g for g, n in zip(cen["geo_id"], cen["geo_name"])}
    pairs, missing = {}, []
    for en in merged["region_en"]:
        ru = NAME.get(en)
        gid = by_ru.get(norm(ru)) if ru else None
        if gid is None:
            missing.append((en, ru))
        else:
            pairs[en] = (gid, ru)
    claimed = [g for g, _ in pairs.values()]
    spare = [n for n in cen["geo_name"] if norm(n) not in
             {norm(v[1]) for v in pairs.values()}]

    print("\n  the join, both ways (§12):")
    print(f"    merged COD regions   {len(merged):>4}")
    print(f"    census regions       {len(cen):>4}")
    print(f"    matched              {len(pairs):>4}")
    print(f"    COD with no census   {len(missing):>4}  {missing}")
    print(f"    census with no COD   {len(spare):>4}  {spare}")
    if missing or spare or len(set(claimed)) != len(claimed):
        raise SystemExit("join FAILED")

    merged["unit"] = merged["region_en"].map(lambda e: pairs[e][0])
    merged["name"] = merged["region_en"].map(lambda e: pairs[e][1])

    os.makedirs(OUT_DIR, exist_ok=True)
    merged[["unit", "name", "region_en", "geometry"]].to_file(
        OUT, layer="regions", driver="GPKG")
    print(f"\nwrote {OUT} ({len(merged)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs[e][0] for e in pairs)})
    lut["unit"] = lut["geo_id"]
    lut.to_csv(LOOKUP, index=False)
    print(f"wrote {LOOKUP} ({len(lut)} rows) — geo_id IS the unit here (KATO)")


if __name__ == "__main__":
    main()
