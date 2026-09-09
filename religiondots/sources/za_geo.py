"""South Africa — boundaries for the 213 local and metropolitan municipalities.

Writes data/geo/za/za_munics.gpkg and data/geo/za/za_lookup.csv.

**THIS REPLACED A NINE-PROVINCE LAYER ON 2026-09-09** (ask/answered/002-za). The province
version's whole difficulty was that it joined on nine strings and a transposition would have
moved eleven million people without breaking a single total, so it needed an area rank test
to have any independent evidence at all. None of that applies here: the CS 2016 microdata and
COD-AB carry the **same MDB municipality codes**, so this is a code join, 213 of 213 both
ways with nothing spare on either side.

OCHA COD-AB South Africa from HDX, the **shapefile** bundle rather than the geodatabase on
§12's Chile rule: GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers,
report the right CRS and return ZERO features while raising nothing. Read with
`engine="fiona"`, because pyogrio is geopandas' default when installed and is the engine that
has silently returned zero. The feature count is asserted either way. The same bundle the
province build downloaded is reused; `zaf_admin3.shp` is the 213 municipalities and
`zaf_admin1.shp` the nine provinces.

**THE VINTAGE AGREEING IS NOT LUCK AND IT IS WORTH KNOWING WHY.** CS 2016 was enumerated on
6 March 2016, five months BEFORE the August 2016 demarcation, and every record therefore
carries its geography twice: `MN_CODE_2011` (234 municipalities) and `MN_CODE_2016` (213),
the second being Stats SA's own recode of the same households onto the new boundaries.
COD-AB's ADM3 is the post-2016 set, `valid_on` 2020-11-09, and has exactly those 213. The
2011 vintage is 10% finer in unit count and there is no boundary layer for it on disk, so the
2016 set is what is drawn; `sources/za.py` says how the right label set is chosen.

A CODE JOIN STILL GETS INDEPENDENT EVIDENCE, because the failure a code join has is not a
missed match, it is a match to the wrong polygon after somebody reissues a file with the
codes shifted. Three quantities are checked and none of them is the join key:

  1. **DISTRICT.** Every municipality's district, as Stats SA codes it in the microdata
     (`DC_MDB_C_2016`, carried through `data/normalized/za.csv`'s note column), must be the
     district COD puts its polygon in. 213 of 213, over 52 districts. A pair of municipalities
     swapped between districts fails here.
  2. **PROVINCE.** The same, over the nine provinces, and it catches the coarser version of
     the same fault.
  3. **NAME.** 212 of 213 of Stats SA's own names agree with COD's after folding, and the one
     that does not is `MP326`, which Stats SA calls `City of Mbombela` and COD calls
     `Mbombela`; that is the same place under its official and its short name. Two further
     names come from COD by way of `sources/za.py`'s NAME_OVERRIDE and are excluded from this
     test rather than passing it for free: `LIM345`, whose Stats SA label is the literal word
     `New`, and `NC067`, whose label is corrupt in the .dta itself.

Together those close the within-district swap that (1) and (2) would miss. The exception list
is written out rather than being a fuzzy matcher, on the same reasoning the province build
used for `Nothern Cape`: a matcher good enough to fix a real difference is good enough to pair
two genuinely different municipalities and never say so
([[reference_name_join_wrong_neighbour]]).

Usage:
    python sources/za_geo.py --fetch    one ~90 MB zip from HDX
    python sources/za_geo.py            rebuild from data/raw/za/
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
RAW = os.path.join(ROOT, "data", "raw", "za")
OUT_DIR = os.path.join(ROOT, "data", "geo", "za")
OUT = os.path.join(OUT_DIR, "za_munics.gpkg")
LOOKUP = os.path.join(OUT_DIR, "za_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "za.csv")

ZIP_URL = ("https://data.humdata.org/dataset/061d4492-56e8-458c-a3fb-e7950991adf0/"
           "resource/019208d9-75e0-4aa9-9cc8-ef45f6b8dd54/download/"
           "zaf_admin_boundaries.shp.zip")
ZIP_NAME = "zaf_admin_boundaries.shp.zip"
EXPECTED = 213
EXPECTED_DISTRICTS = 52
EXPECTED_PROVINCES = 9

# The one municipality whose Stats SA name and COD name genuinely differ, and it is the same
# place: Stats SA writes the official `City of Mbombela`, COD writes the short `Mbombela`.
# Asserted to be the ONLY one, so a reissued bundle that renamed anything else fails.
NAME_EXCEPTIONS = {"MP326"}
# Names that sources/za.py already took from COD (its NAME_OVERRIDE), so testing them here
# would be testing COD against itself.
NAME_FROM_COD = {"LIM345", "NC067"}

# COD spells provinces its own way and one of them is misspelt: `ADM1_EN` reads
# `Nothern Cape`, no `r`, in this bundle. Written out rather than fuzzy-matched, for the
# reason in the module docstring. If a reissue spells it correctly the entry stops being used
# and the check still has to reach nine.
PROV_ALIAS = {
    "notherncape": "northerncape",
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
    # §5a: HDX answers the un-redirected URL with a 302 and a small HTML body, which is a
    # perfectly good HTTP 200 to a client that does not follow it.
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def prov_fold(s):
    k = fold(s)
    return PROV_ALIAS.get(k, k)


def _read_adm3():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()]
    shp = [n for n in names if re.search(r"adm(?:in)?3\.shp$", n, re.I)]
    if len(shp) != 1:
        raise SystemExit(f"expected one admin3 shapefile in the bundle, found {shp}")
    g = gpd.read_file(f"zip://{src}!{shp[0]}", engine="fiona")

    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED:
        raise SystemExit(f"{shp[0]}: {len(g)} features, expected {EXPECTED} municipalities. "
                         "COD-AB may have moved to a different demarcation; the CS 2016 "
                         "microdata carries both the 2011 (234) and the 2016 (213) sets, so "
                         "check which one this bundle is before changing anything.")
    cols = {c.lower(): c for c in g.columns}
    need = ["adm3_name", "adm3_name1", "adm3_pcode", "adm2_name1", "adm1_name"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise SystemExit(f"COD ADM3 has no {missing} column; got {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g, {k: cols[k] for k in need}


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, c = _read_adm3()
    print(f"COD ADM3: {len(g)} polygons, crs={g.crs}, valid_on "
          f"{g['valid_on'].iloc[0] if 'valid_on' in g.columns else '?'}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/za.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "municipality"]
    cen = df.drop_duplicates("geo_id")[["geo_id", "geo_name", "note"]].copy()
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} CS 2016 municipalities, expected {EXPECTED}")
    cen["prov"] = cen["note"].str.extract(r"province=([^;]+)")[0].str.strip()
    cen["dist"] = cen["note"].str.extract(r"district=([^;]+)")[0].str.strip()
    if cen[["prov", "dist"]].isna().any().any():
        raise SystemExit("data/normalized/za.csv's note column no longer carries "
                         "`province=` and `district=` -- re-run sources/za.py")

    poly = {}
    for _, row in g.iterrows():
        code = str(row[c["adm3_name1"]]).strip()
        if code in poly:
            raise SystemExit(f"COD municipality code {code!r} appears twice")
        poly[code] = row

    pairs, missing = {}, []
    for _, row in cen.iterrows():
        if row["geo_id"] in poly:
            pairs[row["geo_id"]] = poly[row["geo_id"]]
        else:
            missing.append((row["geo_id"], row["geo_name"]))
    spare = sorted(set(poly) - set(pairs))

    print("\n  the join, both ways (§12), on the MDB municipality code:")
    print(f"    CS 2016 municipalities     {len(cen):>4}")
    print(f"    COD polygons               {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    CS 2016 with no polygon    {len(missing):>4}")
    print(f"    polygons with no CS 2016   {len(spare):>4}")
    for gid, nm in missing:
        print(f"      no polygon: {gid} {nm!r}")
    for code in spare:
        print(f"      no CS 2016: {code} {poly[code][c['adm3_name']]!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- independent check 1: DISTRICT ----
    bad = []
    for _, row in cen.iterrows():
        cod_dc = str(pairs[row["geo_id"]][c["adm2_name1"]]).strip()
        if cod_dc != row["dist"]:
            bad.append((row["geo_id"], row["geo_name"], row["dist"], cod_dc))
    ndist = len({str(pairs[k][c["adm2_name1"]]).strip() for k in pairs})
    if bad:
        for b in bad[:20]:
            print(f"      {b[0]} {b[1]!r}: Stats SA district {b[2]}, COD district {b[3]}")
        raise SystemExit(
            f"{len(bad)} of {EXPECTED} municipalities sit in a different district in COD "
            "than in the CS 2016 microdata. That is what a shifted code join looks like, "
            "and no total anywhere would have caught it.")
    if ndist != EXPECTED_DISTRICTS:
        raise SystemExit(f"{ndist} districts, expected {EXPECTED_DISTRICTS}")
    print(f"\n  DISTRICT CHECK: all {EXPECTED} municipalities sit in the same one of "
          f"{ndist} districts in\n    COD as in the CS 2016 microdata.")

    # ---- independent check 2: PROVINCE ----
    bad = []
    for _, row in cen.iterrows():
        cod_pr = str(pairs[row["geo_id"]][c["adm1_name"]]).strip()
        if prov_fold(cod_pr) != prov_fold(row["prov"]):
            bad.append((row["geo_id"], row["geo_name"], row["prov"], cod_pr))
    nprov = len({prov_fold(pairs[k][c["adm1_name"]]) for k in pairs})
    if bad:
        for b in bad[:20]:
            print(f"      {b[0]} {b[1]!r}: Stats SA {b[2]!r}, COD {b[3]!r}")
        raise SystemExit(f"{len(bad)} municipalities are in a different province in COD")
    if nprov != EXPECTED_PROVINCES:
        raise SystemExit(f"{nprov} provinces, expected {EXPECTED_PROVINCES}")
    print(f"  PROVINCE CHECK: all {EXPECTED} agree across the {nprov} provinces.")

    # ---- independent check 3: NAME ----
    differ = set()
    for _, row in cen.iterrows():
        if row["geo_id"] in NAME_FROM_COD:
            continue
        if fold(row["geo_name"]) != fold(pairs[row["geo_id"]][c["adm3_name"]]):
            differ.add(row["geo_id"])
    print(f"  NAME CHECK: {EXPECTED - len(NAME_FROM_COD) - len(differ)} of "
          f"{EXPECTED - len(NAME_FROM_COD)} testable names agree with COD's after folding.")
    for gid in sorted(differ):
        row = cen[cen["geo_id"] == gid].iloc[0]
        print(f"    {gid}: Stats SA {row['geo_name']!r}, COD "
              f"{pairs[gid][c['adm3_name']]!r}")
    if differ != NAME_EXCEPTIONS:
        raise SystemExit(
            f"the set of municipalities whose names disagree with COD is {sorted(differ)}, "
            f"expected exactly {sorted(NAME_EXCEPTIONS)}. A new disagreement is either a "
            "renamed municipality or a shifted join, and the two look identical from here; "
            "check it against the MDB register before adding it to NAME_EXCEPTIONS.")
    print(f"    {sorted(NAME_EXCEPTIONS)} is the expected exception and the only one.")

    # ---- write ----
    out = g[[c["adm3_name1"], c["adm3_name"], c["adm3_pcode"], "geometry"]].rename(
        columns={c["adm3_name1"]: "unit", c["adm3_name"]: "cod_name",
                 c["adm3_pcode"]: "pcode"})
    out["unit"] = out["unit"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile) -- with
    # the two exceptions sources/za.py's NAME_OVERRIDE names and explains.
    statssa = dict(zip(cen["geo_id"], cen["geo_name"]))
    out["name"] = out["unit"].map(statssa)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no Stats SA name")
    out = out[["unit", "name", "cod_name", "pcode", "geometry"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="munics", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows; geo_id IS the unit here, "
          "the lookup is kept so countries.py's shape does not change)")


if __name__ == "__main__":
    main()
