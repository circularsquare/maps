"""Eswatini — boundaries for the four regions.

Writes data/geo/sz/sz_regions.gpkg and data/geo/sz/sz_lookup.csv.

OCHA COD-AB Eswatini, from HDX, the **shapefile** bundle rather than the geodatabase on
§12's Chile rule — GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers,
report the right CRS and return ZERO features while raising nothing. Read with
`engine="fiona"`, and the feature count is asserted either way.

**FOUR NAMES, AND THEY AGREE CHARACTER FOR CHARACTER.** Hhohho, Manzini, Shiselweni and
Lubombo — no accents, no transliteration, no abbreviation, and no same-named neighbour for
[[reference_name_join_wrong_neighbour]] to catch out. The join is asserted both ways anyway.

**THE ORDER CHECK IS REAL HERE AND IT IS NOT ZIMBABWE'S.** `sources/sz.py` numbers the
regions `SZ01`..`SZ04` from their position in Table 3.2.4, which is the CSO's own order:
Hhohho, Manzini, Shiselweni, Lubombo. COD's `ADM1_PCODE` is a different numbering with a
different origin, and this script prints both and asserts the pairing is a bijection rather
than asserting the two sequences coincide. What actually protects the region columns from a
silent transposition is in `sources/sz.py`: each region's Christians over its Table 5.2.2
population must sit within two points of the national 89.25%, and exactly one of the 24
orderings of the four columns survives that.

**THE BUNDLE SHIPS ADM2 (the tinkhundla) AND IT IS NOT USABLE.** The 2017 census publishes
religion at region and nowhere else — Volume 3's chapter 3 is the entire religion output,
Volume 2 is a tinkhundla atlas with no religion in it, and Volumes 5 and 6 are literacy and
disability. Four units for 1.09M people is the CSO's ceiling rather than a choice made here.
Drawn under spec §3.9b, which sets no minimum unit count. See `sources/sz.md` §2.

Usage:
    python sources/sz_geo.py --fetch    one small zip from HDX
    python sources/sz_geo.py            rebuild from data/raw/sz/
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
RAW = os.path.join(ROOT, "data", "raw", "sz")
OUT_DIR = os.path.join(ROOT, "data", "geo", "sz")
OUT = os.path.join(OUT_DIR, "sz_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "sz_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "sz.csv")

ZIP_URL = ("https://data.humdata.org/dataset/e145e4c6-0a78-42c7-833c-5850ac3a1731/"
           "resource/9828a61f-42cd-4d93-81dd-e14b854ac9ef/download/"
           "swz_admin_boundaries.shp.zip")
ZIP_NAME = "swz_admin_boundaries.shp.zip"
EXPECTED = 4


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


def _read_adm1():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()]
    shp = [n for n in names if re.search(r"adm(?:in)?1\.shp$", n, re.I)]
    if len(shp) != 1:
        raise SystemExit(f"expected one admin1 shapefile in the bundle, found {shp}")
    g = gpd.read_file(f"zip://{src}!{shp[0]}", engine="fiona")

    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED:
        raise SystemExit(f"{shp[0]}: {len(g)} features, expected {EXPECTED} regions")
    cols = {c.upper(): c for c in g.columns}
    name_col = next((cols[k] for k in ("ADM1_NAME", "ADM1_EN") if k in cols), None)
    code_col = cols.get("ADM1_PCODE")
    if not name_col or not code_col:
        raise SystemExit(f"no adm1 name/pcode column in {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g, name_col, code_col


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, name_col, code_col = _read_adm1()
    print(f"COD ADM1: {len(g)} polygons, crs={g.crs}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/sz.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "region"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census regions, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g[name_col], g[code_col].astype(str).str.strip()):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD name {nm!r} appears twice")
        poly[k] = (nm, cd)

    pairs, missing = {}, []
    for gid, nm in zip(cen["geo_id"], cen["geo_name"]):
        k = fold(nm)
        if k in poly:
            pairs[gid] = (nm, poly[k][1], poly[k][0])
        else:
            missing.append((gid, nm))
    used = {v[1] for v in pairs.values()}
    spare = [(nm, cd) for nm, cd in poly.values() if cd not in used]

    print("\n  the join, both ways (§12):")
    print(f"    census regions             {len(cen):>4}")
    print(f"    COD polygons               {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for gid, nm in missing:
        print(f"      no polygon: {gid} {nm!r}")
    for nm, cd in spare:
        print(f"      no census : {cd} {nm!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    print("\n    the two numberings, which have different origins:")
    for gid in sorted(pairs):
        nm, cd, codname = pairs[gid]
        print(f"      {gid}  {nm:<12} <-> {cd}  {codname!r}")
    if len({v[1] for v in pairs.values()}) != EXPECTED:
        raise SystemExit("two census regions were paired to the same polygon")
    print("    A BIJECTION IS ALL THAT IS ASSERTED HERE. Four names cannot be checked by "
          "order\n    the way Zimbabwe's ten were; what rules out a transposed region "
          "column is the\n    Christian-share band in sources/sz.py, which exactly one of "
          "the 24 orderings passes.")

    out = g[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    cso = {v[1]: v[0] for v in pairs.values()}
    out["name"] = out["unit"].map(cso)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no CSO name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="regions", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[g][1] for g in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
