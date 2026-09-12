"""Barbados — boundaries for the 11 parishes.

Writes data/geo/bb/bb_parishes.gpkg and data/geo/bb/bb_lookup.csv.

OCHA COD-AB Barbados, `brb_admbnda_adm1_2019.shp`. **COD'S ADM1 IS BSS'S PARISH TIER
EXACTLY**: 11 polygons against Table 02.06's 11 units, and the parishes are the only
sub-national geography Barbados has — there is no tier below them in any published boundary
set, and the census tabulates on them.

**THE ONLY SYSTEMATIC NAME DIFFERENCE IS `St.` AGAINST `Saint`**, on ten of the eleven
(Christ Church is the exception, and is spelled identically). `fold()` expands the
abbreviation rather than carrying an alias table, because a frozen list of renames goes
stale in silence at the next release (§12).

**THE JOIN IS BY NAME AND THE INDEPENDENT CHECK IS THE P-CODE.** BSS publishes no code at
all in its workbook, so `sources/bb.py` carries COD's `ADM1_PCODE` against each parish name
by hand and this file asserts the same pairing from the boundary side. Nothing else would
catch a transposition, because every total in `bb.py` reconciles whichever polygon a parish
is paired with (§9n's `TMA` lesson, and the same arrangement as Trinidad and Cayman).

COD's pcodes run **BB01 to BB11 in alphabetical order of the `Saint` spelling** — Christ
Church first, then Andrew, George, James, John, Joseph, Lucy, Michael, Peter, Philip,
Thomas — which is worth knowing only because it means the numbering carries no meaning and
must not be assumed to match the census's own row order, which is by population.

Usage:
    python sources/bb_geo.py --fetch    one ~243 KB shapefile zip from HDX
    python sources/bb_geo.py            rebuild from data/raw/bb/
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
RAW = os.path.join(ROOT, "data", "raw", "bb")
OUT_DIR = os.path.join(ROOT, "data", "geo", "bb")
OUT = os.path.join(OUT_DIR, "bb_parishes.gpkg")
LOOKUP = os.path.join(OUT_DIR, "bb_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "bb.csv")

ZIP_URL = ("https://data.humdata.org/dataset/91ee6430-ca2a-446c-8659-8fcf5410591b/"
           "resource/7a45aa8d-f7a4-4a87-ab54-473b32ade717/download/brb_adm_2019_shp.zip")
ZIP_NAME = "brb_adm_2019_shp.zip"
SHP = "brb_adm_2019_shp/brb_admbnda_adm1_2019.shp"
EXPECTED = 11

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# `St.` / `St ` -> `saint`, applied before the alphanumeric fold.
ABBREV = re.compile(r"\bst\.?\s+", re.I)


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
    s = ABBREV.sub("saint ", " ".join(s.split()))
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
    if len(g) != EXPECTED:
        raise SystemExit(f"{SHP} returned {len(g)} features, expected {EXPECTED}")
    for c in ("ADM1_EN", "ADM1_PCODE"):
        if c not in g.columns:
            raise SystemExit(f"{SHP} has no {c!r} -- columns are {list(g.columns)}")
    print(f"COD-AB ADM1: {len(g)} parishes, crs={g.crs}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/bb.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    cen = (df[df["geo_level"] == "parish"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census parishes, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g["ADM1_EN"], g["ADM1_PCODE"]):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD name {nm!r} folds onto an existing one")
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
    print(f"    census parishes         {len(census):>4}")
    print(f"    COD polygons            {len(poly):>4}")
    print(f"    matched                 {len(pairs):>4}")
    print(f"    census with no polygon  {len(missing):>4}  {missing}")
    print(f"    polygons with no census {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("join FAILED")

    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — bb.py's name->pcode matches COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: bb.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("bb.py's PARISHES pcodes disagree with COD -- every parish's dots "
                         "would be placed in the wrong parish")

    variants = [(census[c], pairs[c][0]) for c in pairs if census[c] != pairs[c][0]]
    print(f"    {len(variants)} name variant(s) resolved by fold() — `St.` vs `Saint`")

    out = g[["ADM1_EN", "ADM1_PCODE", "geometry"]].rename(
        columns={"ADM1_EN": "name", "ADM1_PCODE": "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nso = {pairs[c][1]: census[c] for c in pairs}
    out["name"] = out["pcode"].map(lambda c: nso[str(c).strip()])

    areas = out.to_crs(32621).area / 1e6
    print(f"\n  {len(out)} parishes:")
    for (_, row), a in zip(out.sort_values("unit").iterrows(), areas):
        print(f"    {row['unit']}  {row['name']:<16} {a:7.1f} km2")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="parishes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)}).to_csv(
        LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
