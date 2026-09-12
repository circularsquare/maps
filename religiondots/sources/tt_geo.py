"""Trinidad and Tobago — boundaries for the 15 municipalities.

Writes data/geo/tt/tt_municipalities.gpkg and data/geo/tt/tt_lookup.csv.

OCHA COD-AB Trinidad and Tobago, from HDX — the `tto_adm1_v2` shapefile. **COD'S ADM1 IS
THE CENSUS'S MUNICIPALITY TIER EXACTLY**: 15 polygons against Table 8's 15 drawn units, and
they are the same 15 that CSO publishes as separate `... Individuals.xlsx` workbooks on
`cso.gov.tt`, which is an independent confirmation of the tier from the same office.

The tier is the country's local-government structure: **nine regional corporations, three
boroughs, two cities, and the Tobago House of Assembly area**. Table 8 prints `City of` and
`Borough of` in front of five of them and COD does not, which is the only systematic name
difference and is handled by `fold()` rather than by an alias table (§12: a frozen list of
renames goes stale in silence at the next release).

**THE JOIN IS BY NAME AND THE INDEPENDENT CHECK IS THE P-CODE.** CSO publishes no code at
all in Table 8, so `sources/tt.py` carries COD's `ADM1_PCODE` against each municipality name
by hand; this file asserts the same pairing from the boundary side. If the name join does
not reproduce every hardcoded pcode the run stops — nothing else would catch a
transposition, because every total in `tt.py` reconciles whichever polygon a municipality is
paired with (§9n's `TMA` lesson).

**FOUR NAMES DIFFER ONLY IN THEIR SEPARATORS**, which is why `fold()` drops every
non-alphanumeric character rather than just spaces:

    census `Couva/ Tabaquite/ Talparo`   COD `Couva-Tabaquite-Talparo`
    census `Penal/ Debe`                 COD `Penal-Debe`
    census `San Juan/Laventille`         COD `San Juan-Laventille`
    census `Mayaro/ Rio Claro`           COD `Mayaro/Rio Claro`

Usage:
    python sources/tt_geo.py --fetch    one ~93 KB zip from HDX
    python sources/tt_geo.py            rebuild from data/raw/tt/
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
RAW = os.path.join(ROOT, "data", "raw", "tt")
OUT_DIR = os.path.join(ROOT, "data", "geo", "tt")
OUT = os.path.join(OUT_DIR, "tt_municipalities.gpkg")
LOOKUP = os.path.join(OUT_DIR, "tt_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "tt.csv")

ZIP_URL = ("https://data.humdata.org/dataset/eed55f95-183c-48f7-adef-23dff31ec972/"
           "resource/218b72d0-35fb-4026-b8a6-152d87acea0d/download/tto_adm1_v2.zip")
ZIP_NAME = "tto_adm1_v2.zip"
SHP = "tto_adm1/TTO_adm1.shp"
EXPECTED = 15

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# Table 8's own honorifics. Dropped before folding so `City of Port of Spain` reaches COD's
# `Port of Spain`; kept in geo_name, because they are the census's own strings.
HONORIFIC = re.compile(r"^(city|borough|region|regional corporation)\s+of\s+", re.I)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 20_000:
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
    s = HONORIFIC.sub("", " ".join(s.split()))
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
    name_col, code_col = "NAME_1", "ADM1_PCODE"
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
        raise SystemExit(f"missing {NORM} -- run sources/tt.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    cen = (df[df["geo_level"] == "municipality"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census municipalities, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g[name_col], g[code_col]):
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
    print(f"    census municipalities      {len(census):>4}")
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

    # ---- the independent check: tt.py's hardcoded pcode vs COD's own ----
    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — tt.py's name->pcode matches COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: tt.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("tt.py's MUNICIPALITIES pcodes disagree with COD -- every "
                         "municipality's dots would be placed in the wrong municipality")

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
        OUT, layer="municipalities", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
