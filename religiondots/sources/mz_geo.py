"""Mozambique — boundaries for the 11 provinces.

Writes data/geo/mz/mz_provinces.gpkg and data/geo/mz/mz_lookup.csv.

OCHA COD-AB Mozambique (`cod-ab-moz`), the shapefile bundle rather than the geodatabase on
§12's Chile rule, read with `engine="fiona"`. ADM1 is the eleven provinces, with Maputo
City as a province of its own, which is how INE tabulates the census.

THE NAME IS TAKEN FROM THE CENSUS AND NOT FROM THE BOUNDARY FILE (§12, Chile). The join
folds out accents and punctuation and is asserted to be a bijection; `COD_TO_CENSUS` holds
the names that still differ after folding.

Usage:
    python sources/mz_geo.py --fetch    one zip from HDX
    python sources/mz_geo.py            rebuild from data/raw/mz/
"""

import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mz")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mz")
OUT = os.path.join(OUT_DIR, "mz_provinces.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mz_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "mz.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

ZIP_NAME = "moz_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/5e8d83a5-1210-49be-b7d9-cf286dbc15df/resource/"
           "ee532229-14eb-4c3d-9e55-c3f0f6fa402d/download/moz_admin_boundaries.shp.zip")

EXPECTED = 11

# COD's spelling (folded) -> INE's, where folding alone does not make them agree.
COD_TO_CENSUS = {
    "maputo": "Maputo Província",
    "maputocity": "Maputo Cidade",
    "cidadedemaputo": "Maputo Cidade",
    "provinciademaputo": "Maputo Província",
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


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
            raise SystemExit(f"{ZIP_NAME} is not a zip -- starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    with zipfile.ZipFile(dst) as z:
        z.extractall(SHP_DIR)


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    shp = os.path.join(SHP_DIR, "moz_admin1.shp")
    if not os.path.exists(shp):
        have = sorted(os.listdir(SHP_DIR)) if os.path.isdir(SHP_DIR) else []
        raise SystemExit(f"missing {shp} -- run with --fetch first (have {have})")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != EXPECTED:
        raise SystemExit(f"{shp} has {len(g)} features, expected {EXPECTED}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    print(f"COD-AB Mozambique admin1: {len(g)} provinces, columns {list(g.columns)}")
    if "valid_on" in g.columns:
        print(f"  valid_on {sorted(set(map(str, g['valid_on'])))}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/mz.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    cen = df.drop_duplicates("geo_id")[["geo_id", "geo_name"]]
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census provinces, expected {EXPECTED}")
    by_fold = {fold(n): gid for gid, n in zip(cen["geo_id"], cen["geo_name"])}
    name_of = dict(zip(cen["geo_id"], cen["geo_name"]))

    def census_id(cod_name):
        f = fold(cod_name)
        if f in COD_TO_CENSUS:
            f = fold(COD_TO_CENSUS[f])
        return by_fold.get(f)

    g["geo_id"] = g["adm1_name"].map(census_id)
    unmatched = sorted(g.loc[g["geo_id"].isna(), "adm1_name"])
    dup = sorted(g["geo_id"].dropna()[g["geo_id"].dropna().duplicated()])
    spare = sorted(set(cen["geo_id"]) - set(g["geo_id"].dropna()))
    print("\n  the join, both ways (§12):")
    print(f"    COD polygons with no census province  {len(unmatched)}  {unmatched}")
    print(f"    census provinces with no polygon      {len(spare)}  {spare}")
    print(f"    census provinces matched twice        {len(dup)}  {dup}")
    if unmatched or spare or dup:
        raise SystemExit("join FAILED")

    g["unit"] = g["geo_id"]
    g["name"] = g["geo_id"].map(name_of)
    g["cod_pcode"] = g["adm1_pcode"]
    g["cod_name"] = g["adm1_name"]
    for _, r in g.sort_values("geo_id").iterrows():
        print(f"    {r['geo_id']}  {r['name']:<18} <- COD {r['cod_pcode']} {r['cod_name']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = g[["geo_id", "unit", "name", "cod_pcode", "cod_name", "geometry"]]
    keep.to_file(OUT, layer="provinces", driver="GPKG")
    pd.DataFrame(keep.drop(columns="geometry")).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"wrote {LOOKUP}")


UNITS_OUT = os.path.join(OUT_DIR, "mz_units.gpkg")
UNITS_CSV = os.path.join(OUT_DIR, "mz_units.csv")
ADM3_SHP = os.path.join(SHP_DIR, "moz_admin3.shp")
FITTED = os.path.join(ROOT, "data", "normalized", "mz_districts.csv")
# The two provinces with no captured 2007 district volume, drawn whole from 2017's Quadro 11.
WHOLE_PROVINCES = ("MZ02", "MZ06")   # Cabo Delgado, Manica
AREA_TOL = 0.01


def build_units():
    """The drawn units: 2007 districts in nine provinces, and Cabo Delgado and Manica whole.

    Writes data/geo/mz/mz_units.gpkg (layer `units`: unit, level, name, province) and
    mz_units.csv. A 2007 district is the union of the COD-AB administrative posts
    sources/mz_2007.py assigns to it: its COD districts by default, POST_MOVES for the posts
    that changed district after 2007. Checks, all fatal:
      1. every post in the nine provinces has a 2007 district, and every 2007 district a post;
      2. the unit set is DISTRICTS' geo_ids, and data/normalized/mz_districts.csv's if it exists;
      3. per province, the 2007 districts' area is the COD province polygon's within AREA_TOL
         (the posts tile the province, so nothing is double-counted or lost).
    """
    import geopandas as gpd
    import pandas as pd

    sys.path.insert(0, HERE)
    import mz_2007 as M

    posts = gpd.read_file(ADM3_SHP, engine="fiona")
    if len(posts) == 0:
        raise SystemExit(f"{ADM3_SHP}: zero features")
    posts = posts.to_crs(4326)
    assign = M.post_units(list(zip(posts["adm3_pcode"], posts["adm3_name"], posts["adm2_pcode"],
                                   posts["adm2_name"], posts["adm1_pcode"])))
    nine = posts[posts["adm1_pcode"].str[2:].isin(M.DISTRICTS)].copy()
    nine["unit"] = nine["adm3_pcode"].map(assign)
    if nine["unit"].isna().any():
        raise SystemExit(f"posts with no 2007 district: {sorted(nine.loc[nine['unit'].isna(), 'adm3_name'])}")
    names = {p[0]: nm for dl in M.DISTRICTS.values() for nm, p in dl}
    want = set(names)
    got = set(nine["unit"])
    print(f"\n  2007 districts: {len(got)} built from {len(nine)} COD-AB posts, {len(want)} in DISTRICTS")
    if got != want:
        raise SystemExit(f"units differ from DISTRICTS: extra {sorted(got - want)}, "
                         f"missing {sorted(want - got)}")
    if os.path.exists(FITTED):
        fitted = set(pd.read_csv(FITTED, dtype=str)["geo_id"])
        if fitted != want:
            raise SystemExit(f"mz_districts.csv units differ: {sorted(fitted ^ want)}")
        print("  the unit set is mz_districts.csv's")
    moved = nine[nine["adm3_pcode"].isin(M.POST_MOVES)]
    for _, r in moved.iterrows():
        print(f"    moved: {r['adm3_name']} ({r['adm3_pcode']}), COD {r['adm2_name']} -> "
              f"2007 {names[r['unit']]}")

    d = nine[["unit", "geometry"]].dissolve(by="unit").reset_index()
    d["level"] = "district2007"
    d["name"] = d["unit"].map(names)
    d["province"] = d["unit"].str[:4]

    provs = gpd.read_file(OUT, layer="provinces").to_crs(4326)
    ea_d = d.to_crs(6933).area
    ea_p = provs.set_index("unit").to_crs(6933).area
    print("  area of the 2007 districts against the COD province polygon:")
    bad = []
    for p, a in ea_d.groupby(d["province"]).sum().items():
        ratio = a / ea_p[p]
        print(f"    {p}  {ratio:.4f}")
        if abs(ratio - 1) > AREA_TOL:
            bad.append((p, round(ratio, 4)))
    if bad:
        raise SystemExit(f"2007 districts do not tile their provinces: {bad}")

    whole = provs[provs["unit"].isin(WHOLE_PROVINCES)].copy()
    if len(whole) != len(WHOLE_PROVINCES):
        raise SystemExit(f"province polygons missing for {WHOLE_PROVINCES}")
    whole["level"] = "province"
    whole["province"] = whole["unit"]
    units = pd.concat([d[["unit", "level", "name", "province", "geometry"]],
                       whole[["unit", "level", "name", "province", "geometry"]]],
                      ignore_index=True)
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=4326)
    if units["unit"].duplicated().any():
        raise SystemExit("duplicate unit ids")
    units.to_file(UNITS_OUT, layer="units", driver="GPKG")
    pd.DataFrame(units.drop(columns="geometry")).to_csv(UNITS_CSV, index=False, encoding="utf-8")
    print(f"\nwrote {UNITS_OUT} ({len(units)} units: {len(d)} 2007 districts, {len(whole)} provinces)")
    print(f"wrote {UNITS_CSV}")


if __name__ == "__main__":
    main()
    build_units()
