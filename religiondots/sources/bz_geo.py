"""Belize — boundaries for the 6 districts.

Writes data/geo/bz/bz_districts.gpkg and data/geo/bz/bz_lookup.csv.

OCHA COD-AB Belize, from HDX. **The shapefile bundle and not the geodatabase**, on §12's
Chile rule: GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers, report
the right CRS and return ZERO features, raising nothing. The feature count is asserted after
the read either way.

**COD'S ADM1 IS THE CENSUS'S DISTRICT TIER EXACTLY.** Six polygons, six census rows, and
Belize has had the same six districts since 1882 — this is the rare case where there is no
tier ambiguity at all and nothing to check beyond the count and the names.

**THE JOIN IS BY NAME AND THE INDEPENDENT CHECK IS THE P-CODE, WHICH IS NOT REDUNDANT HERE
THE WAY IT WOULD BE IN A COUNTRY WHOSE ORDERS AGREE.** SIB prints its districts **north to
south** — Corozal, Orange Walk, Belize, Cayo, Stann Creek, Toledo — and COD codes them
**alphabetically**, so `BZ01` is Belize District while SIB's first row is Corozal. Five of
the six would be mismatched by position. `sources/bz.py` therefore carries the pcode against
the name rather than numbering by row, and this file asserts the same pairing from the
boundary side: if the name join does not reproduce every hardcoded pcode, the run stops.
Nothing else would catch it — every total in `bz.py` reconciles whichever polygon a district
is paired with (§9n's `TMA` lesson).

**NO NAME DIFFERS.** All six match on `fold()` with no variants at all, which is worth
recording only because it is the first country here of which that is true, so no alias table
is written and none should be added later (§12: a frozen list of renames goes stale in
silence at the next release).

Usage:
    python sources/bz_geo.py --fetch    one ~500 KB zip from HDX
    python sources/bz_geo.py            rebuild from data/raw/bz/
"""

import os
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bz")
OUT_DIR = os.path.join(ROOT, "data", "geo", "bz")
OUT = os.path.join(OUT_DIR, "bz_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "bz_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "bz.csv")

ZIP_URL = ("https://data.humdata.org/dataset/8b2c9a50-82d5-4c31-9b4d-9f7bec2934c6/"
           "resource/113e7382-21bb-4cdb-a5c6-bde8217a0131/download/"
           "blz_admin_boundaries.shp.zip")
ZIP_NAME = "blz_admin_boundaries.shp.zip"
SHP = "blz_admin1.shp"
EXPECTED = 6

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
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


def _read_adm1():
    import geopandas as gpd

    zp = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zp):
        raise SystemExit(f"missing {zp} -- run with --fetch first")
    with zipfile.ZipFile(zp) as z:
        if SHP not in z.namelist():
            raise SystemExit(f"{zp} has no {SHP} -- it holds {z.namelist()[:8]}")
    g = gpd.read_file(f"zip://{zp}!{SHP}")
    # §12 (Chile): a read that succeeds is not a read that returned data.
    if len(g) != EXPECTED:
        raise SystemExit(f"{SHP} returned {len(g)} features, expected {EXPECTED}")
    name_col = "adm1_name"
    code_col = "adm1_pcode"
    for c in (name_col, code_col):
        if c not in g.columns:
            raise SystemExit(f"{SHP} has no {c!r} -- columns are {list(g.columns)}")
    return g, name_col, code_col


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, name_col, code_col = _read_adm1()
    print(f"COD ADM1: {len(g)} polygons, crs={g.crs}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/bz.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    cen = (df[df["geo_level"] == "district"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census districts, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g[name_col], g[code_col]):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD name {nm!r} appears twice")
        poly[k] = (nm, str(cd).strip())

    pairs, missing = {}, []
    for code, nm in census.items():
        k = fold(nm)
        if k in poly:
            pairs[code] = poly[k]
        else:
            missing.append((code, nm))

    used = {v[1] for v in pairs.values()}
    spare = [(nm, cd) for nm, cd in poly.values() if cd not in used]
    print("\n  the join, both ways (§12):")
    print(f"    census districts           {len(census):>4}")
    print(f"    COD polygons               {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for code, nm in missing:
        print(f"      no polygon: {code} {nm!r}")
    for nm, cd in spare:
        print(f"      no census : {cd} {nm!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- the independent check: bz.py's hardcoded pcode vs COD's own ----
    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — bz.py's name->pcode matches COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: bz.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("bz.py's DISTRICTS pcodes disagree with COD -- every district's "
                         "dots would be placed in the wrong district")

    for nm, cod in sorted((census[c], pairs[c][0]) for c in pairs):
        if nm != cod:
            print(f"    name variant, resolved: census {nm!r} / COD {cod!r}")

    out = g[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nso = {pairs[c][1]: census[c] for c in pairs}
    out["name"] = out["pcode"].map(lambda c: nso[str(c).strip()])

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
