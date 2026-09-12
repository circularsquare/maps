"""Zimbabwe — boundaries for the 10 provinces.

Writes data/geo/zw/zw_provinces.gpkg and data/geo/zw/zw_lookup.csv.

OCHA COD-AB Zimbabwe, from HDX, the **shapefile** bundle rather than the geodatabase on
§12's Chile rule — GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers,
report the right CRS and return ZERO features while raising nothing. Read with
`engine="fiona"`, because pyogrio is geopandas' default when installed and is the engine
that has silently returned zero. The feature count is asserted either way.

**THE JOIN IS THE EASIEST ON THE MAP AND THE CHECK IS STILL FREE.** Ten provinces, ten
polygons, and the names agree character for character — no accents, no transliteration, no
abbreviation. The independent check is Malawi's, and unlike Benin's it holds exactly:

  * On the CENSUS side there is no code at all. `sources/zw.py` numbers each province
    `ZW01`..`ZW10` from its POSITION in Table 2.14.
  * On the BOUNDARY side `adm1_pcode` is an attribute and runs **`ZW10`..`ZW19`** — a
    different numbering with a different origin.

Those two orderings must agree, and they do on all ten: Bulawayo is printed first and coded
`ZW10`, Harare printed last and coded `ZW19`. A single transposed row would break it and
nothing else would, because every total in `zw.py` reconciles whichever polygon a province
is paired with (§9n's `TMA` lesson). **The minted id is deliberately NOT the p-code** —
Benin's lesson — so nothing downstream can quietly assume the two are the same string.

**The bundle also ships ADM2 (districts) and ADM3 (wards), and neither is usable**, because
ZIMSTAT publishes religion at province and nowhere else. Table 2.14 is the only religion
table in the 259-page report and religion got none of the five 2022 PHC thematic reports.
So 10 units for 15.2M people is ZIMSTAT's ceiling rather than a choice made here — and it
is the coarsest counting geography on this map. See `sources/zw.md` §2.

Usage:
    python sources/zw_geo.py --fetch    one ~37 MB zip from HDX
    python sources/zw_geo.py            rebuild from data/raw/zw/
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
RAW = os.path.join(ROOT, "data", "raw", "zw")
OUT_DIR = os.path.join(ROOT, "data", "geo", "zw")
OUT = os.path.join(OUT_DIR, "zw_provinces.gpkg")
LOOKUP = os.path.join(OUT_DIR, "zw_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "zw.csv")

ZIP_URL = ("https://data.humdata.org/dataset/f5c4f4e4-a3c8-4d12-891f-d28d74ce04d5/"
           "resource/cfcbb387-5acd-4869-ae33-45873e7bfb92/download/"
           "zwe_admin_boundaries.shp.zip")
ZIP_NAME = "zwe_admin_boundaries.shp.zip"
EXPECTED = 10


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
        raise SystemExit(f"{shp[0]}: {len(g)} features, expected {EXPECTED} provinces")
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
        raise SystemExit(f"missing {NORM} -- run sources/zw.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "province"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census provinces, expected {EXPECTED}")

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
    print(f"    census provinces           {len(cen):>4}")
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

    # ---- the independent check: print position vs COD's own p-code order ----
    coded = sorted(poly.values(), key=lambda v: v[1])
    order = {cd: i for i, (_, cd) in enumerate(coded)}
    bad = []
    for i, gid in enumerate(sorted(pairs)):
        if order[pairs[gid][1]] != i:
            bad.append((pairs[gid][0], i + 1, order[pairs[gid][1]] + 1))
    print(f"\n    Table 2.14's print order reproduces COD's ADM1_PCODE order on "
          f"{len(pairs) - len(bad)}/{len(pairs)} — the two numberings have different "
          f"origins\n    (ZW01.. from the printed row, ZW10.. from COD), so agreement is "
          "evidence rather than\n    a tautology.")
    for nm, i, j in bad:
        print(f"      {nm!r}: printed at #{i}, coded at #{j}")
    if bad:
        raise SystemExit("Table 2.14's row order is NOT province-code order -- the pairing "
                         "rests on names alone and every province's dots could be misplaced")

    out = g[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    zimstat = {v[1]: v[0] for v in pairs.values()}
    out["name"] = out["unit"].map(zimstat)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no ZIMSTAT name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="provinces", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[g][1] for g in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
