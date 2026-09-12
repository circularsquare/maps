"""Cayman Islands — boundaries for the 6 census districts.

Writes data/geo/ky/ky_districts.gpkg and data/geo/ky/ky_lookup.csv.

OCHA COD-AB Cayman Islands, `cym_admbnda_adm1_2020.shp`. **COD'S ADM1 IS ESO'S DISTRICT TIER
EXACTLY** — six polygons, six census tables, and the names match to the letter including
`Sister Islands` for Cayman Brac and Little Cayman together. ESO publishes no code, so
`sources/ky.py` carries COD's `ADM1_PCODE` against each district by hand and this file
asserts the pairing from the boundary side (§9ak's arrangement, and §9n's `TMA` lesson:
nothing else would catch a transposition, because every total in `ky.py` reconciles
whichever polygon a district is paired with).

**AND THE BOUNDARIES ARE NOT RIGHT, WHICH TOOK TWO TESTS TO ESTABLISH AND ONE TO DECIDE.**
The polygon areas are the first clue: COD's Bodden Town is **8.3 km²**, a strip about two
kilometres deep along the south coast, while its North Side is 88.9 km² and reaches down
across the island. That is not the district.

**OSM has the same six districts at `admin_level=8` under the same six names**, which makes
this a choice rather than a complaint. Both were tested:

*Test 1 — settlements.* Every OSM `place` node in the country (69 of them) was located in
both sets. **60 agree; 9 do not**, and neither set wins outright:

    Belford Estates, Breakers, Frank Sound No. 1,   COD says North Side, OSM says Bodden Town
    Midland Acres, Northward, Pease Bay             -- OSM is right, these are Bodden Town
    Savannah, Saint James Pedro Castle              COD says Bodden Town, OSM says George Town
                                                    -- COD is right, these are Bodden Town
    North Sound Estates                             COD North Side, OSM George Town

*Test 2 — population.* Kontur's grid summed per polygon against ESO's own district counts,
which weights each disagreement by how many people it moves:

    district          COD-AB        err        OSM        err
    Bodden Town       11,823     -2,575      8,103     -6,295
    East End             850       -908        776       -982
    George Town       36,211     +2,313     44,064    +10,166
    North Side         3,506     +1,649        495     -1,362
    Sister Islands     1,736       -379      1,566       -549
    West Bay          10,823     -3,961      8,390     -6,394
    total |error|                11,785                25,748

**COD-AB is less wrong by better than two to one, and it is the tier ESO's own table names
match, so COD is used.** What that costs is known and is not hidden: the six eastern Bodden
Town settlements above are inside COD's North Side polygon, so a few hundred Bodden Town
people are drawn in the wrong district and North Side's 1,857 dots are spread over more
ground than the district really covers. `SETTLEMENTS` below re-runs test 1 on every build,
and the KNOWN failures are asserted to be exactly those six — a boundary release that fixes
them, or breaks a different one, stops the run rather than passing quietly.

> The general point: **when two boundary sets disagree, the settlement test says WHICH is
> wrong and the population test says HOW MUCH.** Neither alone decides. Here the first came
> out 6-3 against COD and the second 2:1 for it, because COD's errors are on villages and
> OSM's are on a town.

Usage:
    python sources/ky_geo.py --fetch    one ~196 KB shapefile zip from HDX
    python sources/ky_geo.py            rebuild from data/raw/ky/
"""

import os
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ky")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ky")
OUT = os.path.join(OUT_DIR, "ky_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ky_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ky.csv")

ZIP_URL = ("https://data.humdata.org/dataset/01381a57-754f-46f4-ba1a-763f78796401/"
           "resource/f0fb4f6a-d74a-4f82-84e9-f5624276efd4/download/cym_adm_2020_shp.zip")
ZIP_NAME = "cym_adm_2020_shp.zip"
SHP = "cym_admbnda_adm1_2020.shp"
EXPECTED = 6

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/525.36")

# Settlements with the district they are actually in, and their OSM `place` node position
# (lon, lat). Read off Overpass on 2026-09-07 rather than typed from memory; the query is
#     area["ISO3166-1"="KY"][admin_level=2]; node(area)["place"]; out tags center;
SETTLEMENTS = [
    ("George Town",              -81.3808, 19.2954, "George Town"),
    ("Spotts",                   -81.3172, 19.2755, "George Town"),
    ("Prospect",                 -81.3374, 19.2827, "George Town"),
    ("West Bay",                 -81.3917, 19.3712, "West Bay"),
    ("Hell",                     -81.4079, 19.3755, "West Bay"),
    ("Bodden Town",              -81.2540, 19.2748, "Bodden Town"),
    ("Savannah",                 -81.2975, 19.2725, "Bodden Town"),
    ("Breakers",                 -81.2007, 19.3015, "Bodden Town"),
    ("Northward",                -81.2703, 19.2843, "Bodden Town"),
    ("Frank Sound No. 1",        -81.1895, 19.3006, "Bodden Town"),
    ("North Side",               -81.1850, 19.3353, "North Side"),
    ("Old Man Bay",              -81.1762, 19.3436, "North Side"),
    ("East End",                 -81.1073, 19.2987, "East End"),
    ("Gun Bay",                  -81.0903, 19.3129, "East End"),
    ("Stake Bay",                -79.8239, 19.7183, "Sister Islands"),
    ("West End",                 -79.8829, 19.6888, "Sister Islands"),
]

# The settlements COD-AB gets wrong, and what it says instead. Asserted EXACTLY: a COD
# release that fixes one of these, or breaks another, must stop the run.
KNOWN_BAD = {
    "Breakers": "North Side",
    "Northward": "North Side",
    "Frank Sound No. 1": "North Side",
}


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


def main():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

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
    print(f"COD-AB ADM1: {len(g)} districts, crs={g.crs}")

    # ---- the join, both ways ----
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/ky.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    cen = (df[df["geo_level"] == "district"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census districts, expected {EXPECTED}")

    poly = {fold(n): (n, str(c).strip())
            for n, c in zip(g["ADM1_EN"], g["ADM1_PCODE"])}
    if len(poly) != len(g):
        raise SystemExit("two COD districts fold onto one name")

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
    print(f"    census districts        {len(census):>4}")
    print(f"    COD polygons            {len(poly):>4}")
    print(f"    matched                 {len(pairs):>4}")
    print(f"    census with no polygon  {len(missing):>4}  {missing}")
    print(f"    polygons with no census {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("join FAILED")

    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — ky.py's name->pcode matches COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: ky.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("ky.py's DISTRICTS pcodes disagree with COD -- every district's "
                         "dots would be placed in the wrong district")

    # ---- the settlement test, re-run every build ----
    print(f"\n  the settlement test ({len(SETTLEMENTS)} places whose district is not in "
          "doubt):")
    wrong = {}
    for nm, lon, lat, want in SETTLEMENTS:
        hit = g[g.contains(Point(lon, lat))]
        got = hit["ADM1_EN"].iloc[0] if len(hit) else "(outside every polygon)"
        if got != want:
            wrong[nm] = got
            print(f"      {nm:<20} should be {want:<14} COD says {got}")
    print(f"      {len(SETTLEMENTS) - len(wrong)}/{len(SETTLEMENTS)} correct")
    if wrong != KNOWN_BAD:
        raise SystemExit(
            f"the settlement test changed.\n  expected these to fail: {KNOWN_BAD}\n"
            f"  actually failed:        {wrong}\n"
            "COD has reissued the Cayman boundaries. Re-run the population test in "
            "sources/ky.md §3 against OSM before accepting the new file.")
    print("      the three failures are the KNOWN ones (sources/ky.md §3): COD's North "
          "Side\n      polygon reaches south over Bodden Town's eastern villages. OSM "
          "fixes those\n      three and breaks Savannah and Pedro Castle instead, and "
          "loses the population\n      test 25,748 to 11,785, so COD is used.")

    out = g[["ADM1_EN", "ADM1_PCODE", "geometry"]].rename(
        columns={"ADM1_EN": "name", "ADM1_PCODE": "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nso = {pairs[c][1]: census[c] for c in pairs}
    out["name"] = out["pcode"].map(lambda c: nso[str(c).strip()])

    areas = out.to_crs(32617).area / 1e6
    print(f"\n  {len(out)} districts:")
    for (_, row), a in zip(out.sort_values("unit").iterrows(), areas):
        print(f"    {row['unit']}  {row['name']:<16} {a:8.1f} km2")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)}).to_csv(
        LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
