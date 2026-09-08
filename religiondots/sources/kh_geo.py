"""Cambodia — boundaries for the 25 provinces.

Writes data/geo/kh/kh_provinces.gpkg and data/geo/kh/kh_lookup.csv.

OCHA COD-AB Cambodia, from HDX, the **shapefile** bundle rather than the geodatabase on
§12's Chile rule — GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers,
report the right CRS and return ZERO features while raising nothing. Read with
`engine="fiona"`. The feature count is asserted either way.

**CAMBODIA HANDS OVER TWO INDEPENDENT KEYS AND NEITHER OF THEM IS A NAME**, which is a
better position than most countries here and is why this join is checked rather than
trusted:

  * On the CENSUS side there is no code at all. `sources/kh.py` numbers each province
    `KH-01`..`KH-25` from its POSITION in Tables 2.1.1 and 2.5.1.
  * On the BOUNDARY side `adm1_pcode` runs `KH01`..`KH25`, which is NIS's own official
    province numbering.

Those two orderings agree on all twenty-five, and that is the real check. It is not a
tautology: NIS's printed row order and OCHA's code attribute have different origins, and a
single transposed row would break it while every total in `kh.py` would still reconcile —
§12's `TMA` lesson. **The pairing itself is made on NAMES**, so the code agreement is
genuinely independent evidence rather than the thing being asserted.

**THREE OF THE TWENTY-FIVE NAMES DISAGREE, ALL BY KHMER ROMANISATION, AND THE FOLD IS A
RULE RATHER THAN A LIST** (§12 — derive alias maps, do not hard-code them):

    census            COD                what differs
    Otdar Meanchey    Oddar Meanchey     t / d
    Siem Reap         Siemreap           word break
    Tbong Khmum       Tboung Khmum       ou / o

So the fold strips non-alphanumerics, collapses `ou` to `o`, and maps `d` to `t`. That is
aggressive for a national key and §12 says so — the guard is that **every folded key is
asserted unique on both sides and every match required to be 1:1**, on a set of only 25.
A future release that renames a province into a collision stops the run instead of pairing
two provinces by luck.

**The bundle also ships ADM2 (districts) and ADM3 (communes), and neither is usable**,
because NIS publishes religion at province and nowhere else — Table 2.5.1 is the only
religion table in the 304-page report, and the 2008 census's own priority-table list gives
`A4 Population by Religion` an age and a sex dimension and no geography. See `sources/kh.md`
§2.

Usage:
    python sources/kh_geo.py --fetch    one ~2.5 MB zip from HDX
    python sources/kh_geo.py            rebuild from data/raw/kh/
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
RAW = os.path.join(ROOT, "data", "raw", "kh")
OUT_DIR = os.path.join(ROOT, "data", "geo", "kh")
OUT = os.path.join(OUT_DIR, "kh_provinces.gpkg")
LOOKUP = os.path.join(OUT_DIR, "kh_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "kh.csv")

ZIP_URL = ("https://data.humdata.org/dataset/7472f7e0-3deb-44d9-bd36-38237c666a2e/"
           "resource/2f14a2c7-71c6-4c5c-ba47-1a7d10a0a9fb/download/"
           "khm_admin_boundaries.shp.zip")
ZIP_NAME = "khm_admin_boundaries.shp.zip"
EXPECTED = 25


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
    """Khmer-romanisation-tolerant key. See the module docstring for why each rule is here.

    Only safe because the set is 25 units and both sides are asserted unique under it.
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s.lower())
    return s.replace("ou", "o").replace("d", "t")


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
        raise SystemExit(f"missing {NORM} -- run sources/kh.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "province"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]]
           .sort_values("geo_id"))
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} census provinces, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g[name_col], g[code_col].astype(str).str.strip()):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD folds {nm!r} onto {poly[k][0]!r} -- the romanisation "
                             "fold has collided and is no longer safe as a key")
        poly[k] = (nm, cd)

    seen = {}
    for nm in cen["geo_name"]:
        k = fold(nm)
        if k in seen:
            raise SystemExit(f"the census folds {nm!r} onto {seen[k]!r} -- collision")
        seen[k] = nm

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

    renamed = [(v[0], v[2]) for v in pairs.values() if v[0] != v[2]]
    print(f"\n    {len(pairs) - len(renamed)}/{len(pairs)} names agree character for "
          f"character; {len(renamed)} differ by Khmer romanisation\n    and are matched by "
          "the fold rather than by a hard-coded alias list (§12):")
    for a, b in sorted(renamed):
        print(f"      census {a!r:<20} COD {b!r}")

    # ---- the independent check: the census's PRINT POSITION against COD's own p-code ----
    bad = []
    for gid in sorted(pairs):
        pos = int(gid.split("-")[1])
        code = pairs[gid][1]
        if code != f"KH{pos:02d}":
            bad.append((pairs[gid][0], pos, code))
    print(f"\n    the printed row order reproduces COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)} — NIS's table\n    position and OCHA's "
          "code attribute have different origins, so agreement is evidence\n    rather "
          "than a tautology, and it is what rules out a transposed row.")
    for nm, pos, code in bad:
        print(f"      {nm!r}: printed at #{pos}, coded {code}")
    if bad:
        raise SystemExit("the census row order is NOT province-code order -- the pairing "
                         "rests on a romanisation fold alone and every province's dots "
                         "could be misplaced")

    out = g[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nis = {v[1]: v[0] for v in pairs.values()}
    out["name"] = out["unit"].map(nis)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no NIS name")

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
