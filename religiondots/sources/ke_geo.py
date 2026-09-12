"""Kenya — boundaries for the 47 counties.

Writes data/geo/ke/ke_counties.gpkg and data/geo/ke/ke_lookup.csv.

OCHA COD-AB Kenya, from HDX. **The shapefile bundle and not the geodatabase**, on §12's
Chile rule: GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers, report
the right CRS and return ZERO features, raising nothing. The feature count is asserted after
the read either way.

THE JOIN IS BY NAME, and the independent check is the **P-CODE** — which here is unusually
strong, because the two sides derive the code in completely different ways:

  * On the CENSUS side there is no code at all. KNBS prints Table 2.30's counties in county
    order and nothing else, so `sources/ke.py` numbers them 001..047 **by row position**.
  * On the BOUNDARY side `ADM1_PCODE` is carried as an attribute, `KE001`..`KE047`.

If the census's row order is Kenya's county-code order — which is the assumption the whole
geo_id rests on — then matching by NAME must reproduce the pcode on all 47. It does. A
single transposed or inserted row anywhere in the table would break it, and nothing else
would: the totals reconcile either way (§9n's `TMA` lesson, where a wrong pairing left every
total intact). Kenya's counties are also stable — created by the 2010 constitution, in force
since 2013, unchanged since — so a 2019 census and a COD file are the same vintage in
substance even where the metadata years differ.

Usage:
    python sources/ke_geo.py --fetch    one ~20 MB zip from HDX
    python sources/ke_geo.py            rebuild from data/raw/ke/
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
RAW = os.path.join(ROOT, "data", "raw", "ke")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ke")
OUT = os.path.join(OUT_DIR, "ke_counties.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ke_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ke.csv")

ZIP_URL = ("https://data.humdata.org/dataset/2c0b7571-4bef-4347-9b81-b2174c13f9ef/"
           "resource/c3ade183-9655-4d3e-918a-cc4b2c600e1b/download/"
           "ken_admin_boundaries.shp.zip")
ZIP_NAME = "ken_admin_boundaries.shp.zip"
EXPECTED = 47

# NO HAND-RESOLVED NAMES, AND THAT WAS CHECKED RATHER THAN ASSUMED. KNBS and COD do spell
# five counties differently — TAITA/TAVETA vs Taita Taveta, ELGEYO/MARAKWET vs
# Elgeyo-Marakwet, MURANG'A vs Murang'a, THARAKA-NITHI, NAIROBI vs Nairobi City — but every
# difference is punctuation or case, which `fold()` already removes. A hand-written alias
# table was drafted for these and then deleted after testing that removing it changed
# nothing: a frozen list of five renames goes stale in silence at the next release, and §12
# says derive rather than hard-code. If a real rename ever appears the join fails loudly.
RESOLVED_NAME = {}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and zipfile.is_zipfile(dest):
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=900, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _read_adm1():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    work = os.path.join(RAW, "cod")
    os.makedirs(work, exist_ok=True)
    zipfile.ZipFile(src).extractall(work)

    # COD names it `ken_admin1.shp`, not `ken_adm1.shp`; match both spellings and NOT
    # admin0/admin2/adminlines/adminpoints, which are all in the same bundle.
    pat = re.compile(r"adm(?:in)?1\.shp$", re.I)
    cand = [os.path.join(root, f)
            for root, _, files in os.walk(work) for f in files if pat.search(f)]
    if len(cand) != 1:
        raise SystemExit(f"expected one admin1 shapefile, found {cand}")

    g = gpd.read_file(cand[0])
    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED:
        raise SystemExit(f"{cand[0]}: {len(g)} features, expected {EXPECTED}")
    # COD ships these lowercase in some countries and uppercase in others, and calls the
    # name column adm1_name here and ADM1_EN elsewhere. Resolve rather than assume.
    cols = {c.upper(): c for c in g.columns}
    name_col = next((cols[k] for k in ("ADM1_EN", "ADM1_NAME") if k in cols), None)
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
        raise SystemExit(f"missing {NORM} -- run sources/ke.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    cen = (df[df["geo_level"] == "county"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census counties, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g[name_col], g[code_col]):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD name {nm!r} appears twice")
        poly[k] = (nm, str(cd).strip())

    pairs, missing = {}, []
    for code, nm in census.items():
        k = fold(RESOLVED_NAME.get(nm, nm))
        if k in poly:
            pairs[code] = poly[k]
        else:
            missing.append((code, nm))

    used = {v[1] for v in pairs.values()}
    spare = [(nm, cd) for nm, cd in poly.values() if cd not in used]
    print("\n  the join, both ways (§12):")
    print(f"    census counties            {len(census):>4}")
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

    # ---- the independent check: row position vs COD's own p-code ----
    # ke.py derives geo_id from the row's POSITION in the printed table; COD carries the
    # p-code as an attribute. A name join that reproduces the code on all 47 proves the
    # printed order is county-code order, which nothing else here establishes.
    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != f"KE{c}"]
    print(f"\n    independent check — row position matches COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: ke.py says KE{c}, COD says {pc}")
    if bad:
        raise SystemExit("the census row order is NOT county-code order -- ke.py's geo_id "
                         "is wrong and every county's dots would be in the wrong place")

    for nm, cod in sorted((census[c], pairs[c][0]) for c in pairs):
        if fold(nm) != fold(cod):
            print(f"    name variant, resolved: census {nm!r} / COD {cod!r}")

    out = g[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip().str.replace("^KE", "", regex=True)
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    knbs = {pairs[c][1]: census[c] for c in pairs}
    out["name"] = out["pcode"].map(lambda c: knbs[str(c).strip()])

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="counties", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs), "unit": [c for c in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
