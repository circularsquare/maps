"""Fiji — boundaries for the 15 provinces.

Writes data/geo/fj/fj_provinces.gpkg and data/geo/fj/fj_lookup.csv.

OCHA COD-AB Fiji (`cod-ab-fji`, reviewed October 2024), the **shapefile** bundle rather than
the geodatabase on §12's Chile rule, read with `engine="fiona"` because pyogrio is the engine
that has silently returned zero features.

**FIJI IS THE FIRST COUNTRY ON THIS MAP THAT STRADDLES THE ANTIMERIDIAN, AND THE OBVIOUS
REPROJECTION SILENTLY DESTROYS THREE PROVINCES.** The 180th meridian runs through the country
between Vanua Levu and the Lau group. Converting to EPSG:4326 — the thing every other
`*_geo.py` here does without thinking — gives:

    Cakaudrove   lon -180.000 .. 180.000   span 360.000
    Lau          lon -179.886 .. 179.953   span 359.838
    Macuata      lon -180.000 .. 180.000   span 360.000

That is not a wide province, it is **a torn polygon wrapped the wrong way round the globe**,
and it is the dangerous kind of wrong: the file opens, the feature count is right, the names
are right, and a point-in-polygon join against it puts hexes from the wrong hemisphere inside
Macuata while every total still reconciles. Nothing downstream would complain.

**So this file stores the provinces in EPSG:3832 (WGS 84 / PDC Mercator), and every spatial
operation on Fiji happens there.** PDC Mercator is the Pacific-centred projection made for
exactly this; in it Fiji is one contiguous 543 x 990 km block with no polygon spanning more
than a province's width. COD-AB's own `World_Mercator_150` turns out to *be* PDC Mercator —
identical bounds to the metre — which is the boundary file telling us what it expects.
`sources/fj_grid.py` reads this layer and joins in the same CRS. The dots themselves are
POINTS and are unaffected once placed.

**THE JOIN IS FREE, AND FOR ONCE THAT IS NOT A TRAP.** COD-AB Fiji is sourced from **POPGIS,
Fiji Islands Bureau of Statistics** — the boundary file says so in its own HDX description —
so it carries `FBOS_PID`, the statistics office's own province id, as a first-class column
alongside the OCHA pcode. That is the same identifier the census tabulates on and the same one
SPC's PopGIS returns as `codgeo`. Three sources, one id space, because two of them are the
same office.

So unlike Nicaragua (§9ay, join on name because the codes were renumbered) and unlike Peru
(§9bc, join on code because the names repeat across provinces), Fiji has **no join problem at
all**: 15 units, ids 1-15, and the names agree outright. This file still checks it both ways,
because a join that cannot fail is exactly the one nobody checks.

**WHAT THIS FILE DELIBERATELY DOES NOT DO IS ASSERT A GEOGRAPHIC WITNESS.** `sources/pe_geo.py`
learned the hard way that a witness naming a region can fire on a correct join, and its
replacement — spatial smoothness against random re-pairings — needs enough units to calibrate
on. **Fiji has fifteen**, spread over 500 km of ocean in three groups, so neighbour
correlation is meaningless here and a permutation test on 15 units has no power worth the
name. The honest answer is to say so and to let the check that DOES have power do the work:
`sources/fj_grid.py` correlates each province's census population against an independent
modelled population grid. Stating which check is carrying it is §9ay's rule and it applies
when the answer is "not this one".

Usage:
    python sources/fj_geo.py --fetch    one ~3.9 MB zip from HDX
    python sources/fj_geo.py            rebuild from data/raw/fj/
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fj")
OUT_DIR = os.path.join(ROOT, "data", "geo", "fj")
OUT = os.path.join(OUT_DIR, "fj_provinces.gpkg")
LOOKUP = os.path.join(OUT_DIR, "fj_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "fj.csv")

ZIP_URL = ("https://data.humdata.org/dataset/ae36d1fd-1698-4f53-a4c0-2146a6b5aa29/"
           "resource/8ec2ca4b-973b-409c-8c2f-53e26b816852/download/"
           "fji_polbnda_adm2_province.zip")
ZIP_NAME = "fji_polbnda_adm2_province.zip"
SHP = "fji_polbnda_adm2_province.shp"
EXPECTED = 15

# WGS 84 / PDC Mercator. Pacific-centred, so Fiji is contiguous -- see the docstring.
WORK_CRS = "EPSG:3832"

# COD's spelling -> the census's, where the same province is written differently. Fiji's one
# double-barrelled province is the only case and it is punctuation.
ALIASES = {
    "Nadroga Navosa": "Nadroga/Navosa",
    "Nadroga_Navosa": "Nadroga/Navosa",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    print(f"  {os.path.getsize(dest):,} bytes")
    with open(dest, "rb") as fh:
        if fh.read(2) != b"PK":
            raise SystemExit(f"{dest} is not a zip -- HDX served something else")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zpath):
        raise SystemExit(f"missing {zpath} -- run with --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/fj.py first")

    g = gpd.read_file("zip://" + zpath + "!" + SHP, engine="fiona")
    if len(g) != EXPECTED:
        raise SystemExit(f"COD ADM2 has {len(g)} features, expected {EXPECTED}")
    print(f"COD-AB ADM2: {len(g)} provinces, crs={str(g.crs)[:44]}…")
    # NOT 4326 -- see the module docstring. Fiji straddles the 180th meridian and three
    # provinces come out as torn 360-degree polygons in geographic coordinates.
    torn = g.to_crs("EPSG:4326")
    tears = [(r.ADM2_NAME, r.geometry.bounds[2] - r.geometry.bounds[0])
             for r in torn.itertuples() if r.geometry.bounds[2] - r.geometry.bounds[0] > 180]
    g = g.to_crs(WORK_CRS)
    print(f"  stored in {WORK_CRS} (Pacific-centred), NOT EPSG:4326")
    print(f"  because {len(tears)} provinces tear across the antimeridian in 4326: "
          + ", ".join(f"{n} ({s:.0f}°)" for n, s in tears))
    if not tears:
        raise SystemExit(
            "no province tears across the antimeridian any more. Either COD has re-cut its "
            "geometry or the read has changed -- STOP and check which before simplifying "
            "this file. Do not delete this assertion.")
    wide = [r.ADM2_NAME for r in g.itertuples()
            if r.geometry.bounds[2] - r.geometry.bounds[0] > 2_000_000]
    if wide:
        raise SystemExit(f"provinces still torn in {WORK_CRS}: {wide}")
    print(f"  and none is torn in {WORK_CRS}: the country is "
          f"{(g.total_bounds[2] - g.total_bounds[0]) / 1000:,.0f} x "
          f"{(g.total_bounds[3] - g.total_bounds[1]) / 1000:,.0f} km, contiguous")

    g["fbos"] = g["FBOS_PID"].astype(int).astype(str)
    if g["fbos"].duplicated().any():
        raise SystemExit("COD ADM2 has duplicate FBOS_PID values")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"]
    cen_name = dict(zip(df["geo_id"], df["geo_name"]))
    if len(cen_name) != EXPECTED:
        raise SystemExit(f"{NORM} has {len(cen_name)} provinces, expected {EXPECTED}")

    cod_name = dict(zip(g["fbos"], g["ADM2_NAME"]))
    missing = sorted(set(cen_name) - set(cod_name))
    spare = sorted(set(cod_name) - set(cen_name))

    print("\n  the join, both ways (§12) — ON FBoS's OWN PROVINCE ID:")
    print(f"    census provinces           {len(cen_name):>4}")
    print(f"    COD polygons               {len(cod_name):>4}")
    print(f"    matched                    {len(set(cen_name) & set(cod_name)):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for gid in missing[:6]:
        print(f"      no polygon: {gid} {cen_name[gid]!r}")
    for gid in spare[:6]:
        print(f"      no census : {gid} {cod_name[gid]!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- witness 1: the NAME must agree on every pair ----
    disagree = []
    for gid in sorted(cen_name, key=int):
        a, b = cen_name[gid], cod_name[gid]
        if fold(a) != fold(b) and fold(ALIASES.get(b, b)) != fold(a):
            disagree.append((gid, a, b))
    aliased = sum(1 for gid in cen_name
                  if fold(cen_name[gid]) != fold(cod_name[gid]))
    print(f"\n    witness 1 — the census name equals COD's on "
          f"{EXPECTED - aliased}/{EXPECTED} outright,")
    print(f"    and on {EXPECTED - len(disagree)}/{EXPECTED} once the listed spellings "
          "are allowed:")
    for gid in sorted(cen_name, key=int):
        if fold(cen_name[gid]) != fold(cod_name[gid]):
            print(f"      census {cen_name[gid]!r} = COD {cod_name[gid]!r}")
    for gid, a, b in disagree:
        print(f"      MISMATCH {gid}: census {a!r} vs COD {b!r}")
    if disagree:
        raise SystemExit(f"{len(disagree)} provinces pair on id while naming different "
                         "places -- §12's shape-2 failure; resolve each by hand")

    # ---- witness 2: the division a province sits in must agree with FBoS's own ----
    # COD carries ADM1 (division) independently of the id; Rotuma is its own dependency and
    # FBoS files it under Eastern, which is why this is reported rather than asserted.
    print("\n    witness 2 — COD's own division for each province, reported as a sanity "
          "read:")
    for r in sorted(g.itertuples(), key=lambda r: int(r.fbos)):
        print(f"      {r.fbos:>2}  {cen_name[r.fbos]:<16} {str(r.ADM1_NAME).strip()}")

    print("\n    NO GEOGRAPHIC WITNESS IS ASSERTED HERE, and that is deliberate — see the "
          "module\n    docstring. Fifteen units over 500 km of open ocean cannot calibrate a "
          "neighbour\n    test, so the check with power is the population correlation in "
          "sources/fj_grid.py.")

    out = g[["fbos", "ADM2_NAME", "ADM2_PCODE", "geometry"]].copy()
    out["unit"] = out["ADM2_PCODE"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    out["name"] = out["fbos"].map(cen_name)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no census name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "fbos", "geometry"]].to_file(
        OUT, layer="provinces", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(cen_name, key=int),
                        "unit": [out.set_index("fbos").loc[k, "unit"]
                                 for k in sorted(cen_name, key=int)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
