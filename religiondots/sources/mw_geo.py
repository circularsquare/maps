"""Malawi — boundaries for the 32 districts.

Writes data/geo/mw/mw_districts.gpkg and data/geo/mw/mw_lookup.csv.

OCHA COD-AB Malawi, from HDX. **The shapefile bundle and not the geodatabase**, on §12's
Chile rule: GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers, report
the right CRS and return ZERO features, raising nothing. The feature count is asserted after
the read either way.

**COD's ADM2 IS THE CENSUS'S DISTRICT TIER EXACTLY, INCLUDING THE FOUR CITIES.** 32
polygons, split 7 / 10 / 15 across the three regions, which is the same split Table E5
prints. That is not the usual outcome — Malawi is commonly described as having 28 districts,
and a boundary file built to that description would be missing Mzuzu, Lilongwe, Zomba and
Blantyre Cities and would silently absorb 1.85 million people into the four rural districts
that surround them. The count and the per-region split are both asserted.

THE JOIN IS BY NAME, and the independent check is the **P-CODE**, which the two sides derive
in completely different ways:

  * On the CENSUS side there is no code at all. `sources/mw.py` numbers each district
    `MW<region><nn>` from its POSITION in the printed table, region by region.
  * On the BOUNDARY side `adm2_pcode` is carried as an attribute, `MW101`..`MW315`.

If Table E5's print order is Malawi's own district-code order — the assumption the whole
geo_id rests on — then matching by NAME must reproduce the p-code on all 32. It does. A
single transposed row would break this and nothing else would: every total in `mw.py`
reconciles whichever polygon a district is paired with (§9n's `TMA` lesson).

**ONE NAME DIFFERS AND IT IS SPACING.** NSO writes `Nkhata Bay`, COD writes `Nkhatabay`.
`fold()` removes the space, so no alias table is needed and none is written — a frozen list
of renames goes stale in silence at the next release (§12, and the same call `ke_geo.py`
made after testing that its own alias table changed nothing).

Usage:
    python sources/mw_geo.py --fetch    one ~55 MB zip from HDX
    python sources/mw_geo.py            rebuild from data/raw/mw/
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
RAW = os.path.join(ROOT, "data", "raw", "mw")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mw")
OUT = os.path.join(OUT_DIR, "mw_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mw_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "mw.csv")

ZIP_URL = ("https://data.humdata.org/dataset/20eb8e5b-134d-41d8-a56f-4f358f7faf16/"
           "resource/d0a808c1-4ed6-48b3-ba71-0f852f77297d/download/"
           "mwi_admin_boundaries.shp.zip")
ZIP_NAME = "mwi_admin_boundaries.shp.zip"
EXPECTED = 32
PER_REGION = {"MW1": 7, "MW2": 10, "MW3": 15}


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
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: HDX answers the un-redirected URL with a 302 and a 1.5 KB body, which is a
    # perfectly good HTTP 200 to a client that does not follow it.
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _read_adm2():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    work = os.path.join(RAW, "cod")
    os.makedirs(work, exist_ok=True)
    zipfile.ZipFile(src).extractall(work)

    # COD names it `mwi_admin2.shp`, not `mwi_adm2.shp`, and the bundle also holds an
    # `_em` edge-matched twin plus admin0/1/3, lines and points. Match the one.
    pat = re.compile(r"adm(?:in)?2\.shp$", re.I)
    cand = [os.path.join(root, f)
            for root, _, files in os.walk(work) for f in files if pat.search(f)]
    if len(cand) != 1:
        raise SystemExit(f"expected one admin2 shapefile, found {cand}")

    g = gpd.read_file(cand[0])
    # §12: assert the FEATURE COUNT, never the absence of an exception.
    if len(g) != EXPECTED:
        raise SystemExit(f"{cand[0]}: {len(g)} features, expected {EXPECTED} -- if this "
                         "is 28 the file predates the four city districts")
    cols = {c.upper(): c for c in g.columns}
    name_col = next((cols[k] for k in ("ADM2_EN", "ADM2_NAME") if k in cols), None)
    code_col = cols.get("ADM2_PCODE")
    if not name_col or not code_col:
        raise SystemExit(f"no adm2 name/pcode column in {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g, name_col, code_col


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g, name_col, code_col = _read_adm2()
    print(f"COD ADM2: {len(g)} polygons, crs={g.crs}")

    # The 7/10/15 split is what says these are the census's districts and not somebody
    # else's 28. Checked before the join, so a wrong file fails on its own terms.
    got = {}
    for cd in g[code_col]:
        got[str(cd)[:3]] = got.get(str(cd)[:3], 0) + 1
    if got != PER_REGION:
        raise SystemExit(f"districts per region {got}, expected {PER_REGION}")
    print(f"  per region {got} -- matches Table E5")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/mw.py first")
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

    # ---- the independent check: print position vs COD's own p-code ----
    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — Table E5's print order matches COD's ADM2_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: mw.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("Table E5's row order is NOT district-code order -- mw.py's "
                         "geo_id is wrong and every district's dots would be misplaced")

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
