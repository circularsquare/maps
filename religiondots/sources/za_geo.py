"""South Africa — boundaries for the nine provinces.

Writes data/geo/za/za_provinces.gpkg and data/geo/za/za_lookup.csv.

OCHA COD-AB South Africa, from HDX, the **shapefile** bundle rather than the geodatabase on
§12's Chile rule: GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers,
report the right CRS and return ZERO features while raising nothing. Read with
`engine="fiona"`, because pyogrio is geopandas' default when installed and is the engine
that has silently returned zero. The feature count is asserted either way.

**THE NAME JOIN IS NINE STRINGS AND IT STILL GETS AN INDEPENDENT CHECK**, because a name
join picking the wrong same-named neighbour is invisible to every totals test
([[reference_name_join_wrong_neighbour]]) and here it would be invisible twice over: every
province's figures reconcile against its own report whichever polygon it is paired with, so
a transposition would move eleven million people and break nothing.

Zimbabwe's check was print-order against p-code order. That is not available here, because
Stats SA's province codes 1-9 and COD's `ZA1`..`ZA9` are the same numbering from the same
statute, so agreeing proves nothing. **The independent quantity is AREA**, and South Africa's
provinces are unusually well separated on it: Northern Cape is 372,889 km² and Gauteng is
18,178 km², a factor of twenty, and no two provinces are within 4% of each other except
Free State and Western Cape (129,825 and 129,462, 0.3% apart). So the test is that the nine
polygons' computed areas reproduce the published RANK ORDER exactly, plus a loose magnitude
band per province. Rank is robust to the published figures being a few percent off; the
band catches a wholesale unit or projection error.

The Free State figure is not remembered, it is quoted from that province's own CS 2016
profile, Report 03-01-12 page 10: *"the third largest province in land area (about
129 825 km2)"*. It agrees with the list below, which is why the list is trusted.

**FREE STATE AND WESTERN CAPE ARE 0.3% APART AND THE RANK TEST CANNOT SEPARATE THEM.** That
pair is checked a second way instead, on longitude: the Western Cape contains the country's
western cape and its centroid is around 20°E, the Free State is interior at about 26.5°E,
and no plausible boundary file has them the other way round.

Usage:
    python sources/za_geo.py --fetch    one ~30 MB zip from HDX
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
OUT = os.path.join(OUT_DIR, "za_provinces.gpkg")
LOOKUP = os.path.join(OUT_DIR, "za_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "za.csv")

ZIP_URL = ("https://data.humdata.org/dataset/061d4492-56e8-458c-a3fb-e7950991adf0/"
           "resource/019208d9-75e0-4aa9-9cc8-ef45f6b8dd54/download/"
           "zaf_admin_boundaries.shp.zip")
ZIP_NAME = "zaf_admin_boundaries.shp.zip"
EXPECTED = 9

# Published province land areas, km². Free State's is quoted from Report 03-01-12 p.10; the
# rest are the standard Municipal Demarcation Board figures. Used ONLY as a join check, so
# what matters is the rank order and the order of magnitude, never the third digit.
AREA_KM2 = {
    "Northern Cape": 372_889,
    "Eastern Cape": 168_966,
    "Free State": 129_825,
    "Western Cape": 129_462,
    "Limpopo": 125_754,
    "North West": 104_882,
    "KwaZulu-Natal": 94_361,
    "Mpumalanga": 76_495,
    "Gauteng": 18_178,
}
AREA_TOLERANCE = 0.15

# The pair the rank test cannot resolve, and the axis that does resolve it.
TIED_PAIR = ("Free State", "Western Cape")
# approximate centroid longitude, degrees east
LON_HINT = {"Western Cape": 20.0, "Free State": 26.5}

# South Africa's provinces surround Lesotho and enclose it completely. The COD file is
# South Africa only, so Lesotho is a hole in Free State / KwaZulu-Natal / Eastern Cape
# rather than a tenth polygon; a bundle that had swallowed it would push Free State's area
# up by 30,355 km², 23%, and the tolerance above would catch that.
LESOTHO_KM2 = 30_355


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


# **COD-AB MISSPELLS ONE OF THE NINE.** `ADM1_EN` reads `Nothern Cape`, no `r`, in the
# South Africa bundle. The alias is written out rather than repaired by a fuzzy match,
# because a fuzzy matcher good enough to fix this is also good enough to pair two genuinely
# different provinces and never say so ([[reference_name_join_wrong_neighbour]]). If a
# reissued bundle spells it correctly the entry simply stops being used, and the join still
# has to reach nine.
ALIAS = {
    "notherncape": "northerncape",
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    k = re.sub(r"[^a-z0-9]+", "", s.lower())
    return ALIAS.get(k, k)


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
        raise SystemExit(f"missing {NORM} -- run sources/za.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "province"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    if len(cen) != EXPECTED:
        raise SystemExit(f"{len(cen)} CS 2016 provinces, expected {EXPECTED}")

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
    print(f"    CS 2016 provinces          {len(cen):>4}")
    print(f"    COD polygons               {len(poly):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    CS 2016 with no polygon    {len(missing):>4}")
    print(f"    polygons with no CS 2016   {len(spare):>4}")
    for gid, nm in missing:
        print(f"      no polygon: {gid} {nm!r}")
    for nm, cd in spare:
        print(f"      no CS 2016: {cd} {nm!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- the independent check: AREA rank, not name and not p-code ----
    # An equal-area projection is required; degrees are not area. EPSG:9221 is the South
    # African Albers equal-area, whose standard parallels are chosen for this country.
    eq = g.to_crs("+proj=aea +lat_1=-24 +lat_2=-33 +lat_0=0 +lon_0=25 "
                  "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    area = {}
    for nm, a in zip(g[name_col], eq.geometry.area):
        area[fold(nm)] = a / 1e6

    print("\n  AREA CHECK -- the join's only independent evidence:")
    print(f"    {'province':16s} {'computed km2':>13s} {'published':>11s} {'diff':>7s}")
    bad = []
    for prov in sorted(AREA_KM2, key=lambda p: -AREA_KM2[p]):
        got = area.get(fold(prov))
        if got is None:
            raise SystemExit(f"no polygon area for {prov!r}")
        pub = AREA_KM2[prov]
        d = got / pub - 1.0
        flag = "" if abs(d) <= AREA_TOLERANCE else "   <-- OUT OF BAND"
        print(f"    {prov:16s} {got:13,.0f} {pub:11,} {100 * d:+6.1f}%{flag}")
        if abs(d) > AREA_TOLERANCE:
            bad.append((prov, got, pub))
    if bad:
        raise SystemExit(
            "computed and published province areas disagree by more than "
            f"{100 * AREA_TOLERANCE:.0f}%: {[b[0] for b in bad]}. Either the name join is "
            "transposed, or the bundle has swallowed Lesotho "
            f"({LESOTHO_KM2:,} km2, which lands on Free State), or the projection is wrong.")

    got_rank = sorted(AREA_KM2, key=lambda p: -area[fold(p)])
    pub_rank = sorted(AREA_KM2, key=lambda p: -AREA_KM2[p])
    tied = set(TIED_PAIR)
    if [p for p in got_rank if p not in tied] != [p for p in pub_rank if p not in tied]:
        raise SystemExit(
            "province area RANK ORDER does not reproduce the published one:\n"
            f"  computed:  {got_rank}\n  published: {pub_rank}\n"
            "That is what a transposed name join looks like, and no totals check would "
            "have caught it.")
    print(f"\n    rank order reproduces the published one on "
          f"{len(AREA_KM2) - len(tied)}/{len(AREA_KM2)}; {TIED_PAIR[0]} and "
          f"{TIED_PAIR[1]} are\n    0.3% apart in area and are excluded from the rank test "
          "and checked on longitude instead.")

    lon = {}
    cent = g.to_crs(4326).geometry.representative_point()
    for nm, pt in zip(g[name_col], cent):
        lon[fold(nm)] = pt.x
    for prov, hint in LON_HINT.items():
        got = lon[fold(prov)]
        if abs(got - hint) > 3.0:
            raise SystemExit(f"{prov}'s polygon sits at {got:.1f}E, expected about "
                             f"{hint:.1f}E -- {TIED_PAIR[0]} and {TIED_PAIR[1]} look "
                             "swapped")
    print(f"    {TIED_PAIR[1]} at {lon[fold(TIED_PAIR[1])]:.1f}E and {TIED_PAIR[0]} at "
          f"{lon[fold(TIED_PAIR[0])]:.1f}E, the right way round.")

    out = g[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "name", code_col: "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    statssa = {v[1]: v[0] for v in pairs.values()}
    out["name"] = out["unit"].map(statssa)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no Stats SA name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="provinces", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[k][1] for k in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
