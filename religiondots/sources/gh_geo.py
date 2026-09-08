"""Ghana — boundaries for the 272 census units, from GSS itself.

Writes data/geo/gh/gh_districts.gpkg and data/geo/gh/gh_lookup.csv.

**THE STATISTICAL OFFICE SHIPS THE BOUNDARIES FOR ITS OWN TABLES, AND BOTH TIERS OF THEM.**
StatsBank's Census Atlas page carries one link, `assets/geofiles.zip` (26 MB), holding
`Districts_261.zip` and `Districts_271.zip`. The second is misnamed: its shapefile is
`District_272.shp` and it has **272** features — which is exactly the 261 MMDAs with the six
metropolitan districts replaced by their 17 sub-metros, i.e. precisely the tier
`sources/gh.py` picks. Neither file is on HDX, COD or geoBoundaries.

Worth the detour before reaching for a general boundary source: geoBoundaries GHA ADM2 is
**260** units on a 2019 vintage, so it would have cost a vintage argument and lost the
sub-metros as well. §12 says prefer a boundary set from the census year; here the census
year's own office publishes it, one click from the table.

THE JOIN IS BY NAME, because the source has no codes on either side — the census cube
publishes names only, and the shapefile carries `Label`, `Region`, `District` and nothing
else. It is **272 of 272 both ways with no spares and no ambiguity**, on a conservative
fold (case, punctuation and `&`/`and` only). Same for the 261 tier.

AND IT IS VERIFIED AGAINST SOMETHING THE JOIN DOES NOT DETERMINE (§12), which matters more
here than usual because a 100% name join is also what a subtly wrong name join looks like.
The check is the REGION. On the census side a unit's region comes from ROW ORDER — the cube
lists each region header followed by its own districts, and nothing else says which region
a district is in. On the boundary side it comes from an attribute column. The two are
independent, and they agree on all 272. That confirms the positional parse in `gh.py` at the
same time as the join, and it is the check that would catch Romania's county-header bug.

Usage:
    python sources/gh_geo.py --fetch    one 26 MB zip
    python sources/gh_geo.py            rebuild from data/raw/gh/
"""

import io
import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gh")
OUT_DIR = os.path.join(ROOT, "data", "geo", "gh")
OUT = os.path.join(OUT_DIR, "gh_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "gh_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "gh.csv")

ZIP_URL = "https://statsbank.statsghana.gov.gh/assets/geofiles.zip"
ZIP_NAME = "geofiles.zip"
VERIFY_TLS = False              # same omitted intermediate as sources/gh.py

# The inner zip is named 271 and its shapefile is named 272 and holds 272 features. Both
# names are GSS's; neither is a typo on this side.
INNER = {"261": "Geo FIles/Districts_261.zip", "272": "Geo FIles/Districts_271.zip"}
SHP = {"261": "District_261.shp", "272": "District_272.shp"}
EXPECTED = {"261": 261, "272": 272}

DRAWN = "272"                   # the tier that becomes gh_districts.gpkg

# LAKE VOLTA IS INSIDE THE DISTRICTS AND HAS TO COME OUT.
#
# water.py subtracts the SEA and says so: "lakes and non-tidal rivers are a different OSM
# layer and are not subtracted here... where an agency has NOT done it the lake will still
# take dots, and that is a known gap rather than a solved problem." Ghana is the first
# country on this map where that gap is large. Lake Volta is 6,045 km² — the largest
# reservoir on earth by surface area — it sits in the middle of the country, and GSS's
# districts run straight across it. Measured on the first build: **397 of 30,750 dots, 1.29%
# of Ghana, in open water**, and they are the most visible dots on the map because there is
# nothing else drawn there. That is a third of the way to the 3.0% in the New York bbox that
# made water.py exist in the first place.
#
# Fixed HERE and not in water.py, deliberately. The gap water.py names is global; the
# knowledge that Ghana specifically needs it filled is local, and no other country on the
# map has yet been shown to. If a second one is, this is the code that should move.
#
# HydroLAKES v1.0 (Messager et al. 2016), CC BY 4.0, already in ../data/ for the river maps.
LAKES = os.path.join(ROOT, os.pardir, "data", "HydroLAKES_polys_v10_shp",
                     "HydroLAKES_polys_v10.shp")
GH_BBOX = (-3.4, 4.4, 1.4, 11.3)
# Same reasoning as water.py's KEEP_WHOLE_ABOVE: a unit that would lose almost everything to
# the clip has its people somewhere, and crushing them onto a shore sliver is worse than
# leaving them spread. Nothing in Ghana comes close — the worst is well under half — so this
# is a guard rather than a working threshold, and it fires loudly if that ever changes.
KEEP_WHOLE_ABOVE = 0.90

# The region cross-check below disagrees on exactly one row, and it is a typo in the
# BOUNDARY FILE rather than a wrong pairing: `District_272.shp` spells Greater Accra
# `Greate Accra` on AMA-Ablekuma South and nowhere else. The unit's own name matched
# exactly, and it is a sub-metro of Accra Metropolitan Area, which cannot be in any other
# region — so this is resolved rather than guessed (§12, Chile). The list is kept HERE, and
# it is exhaustive, so a SECOND disagreement fails the build instead of joining a pile of
# known-harmless ones. Not repaired in gh.py: the normalised CSV reproduces the source, and
# the typo is the boundary file's, not the table's.
RESOLVED_REGION = {
    "AMA-Ablekuma South": ("Greater Accra", "Greate Accra"),
}


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
    r = requests.get(ZIP_URL, timeout=900, verify=VERIFY_TLS, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: assert size AND type, never the absence of an exception.
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    """Conservative fold, for comparison only. Case, accents, punctuation, and the one
    real orthographic variant in these files (`&` for `and`)."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _read_tier(tier):
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    outer = zipfile.ZipFile(src)
    work = os.path.join(RAW, "geo", tier)
    os.makedirs(work, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(outer.read(INNER[tier]))).extractall(work)
    path = None
    for root, _, files in os.walk(work):
        for f in files:
            if f == SHP[tier]:
                path = os.path.join(root, f)
    if path is None:
        raise SystemExit(f"{SHP[tier]} not found under {work}")

    g = gpd.read_file(path)
    # §12: assert the FEATURE COUNT after every read_file, not the absence of an exception
    # — a driver that opens a layer and returns zero rows raises nothing.
    if len(g) != EXPECTED[tier]:
        raise SystemExit(f"{SHP[tier]}: {len(g)} features, expected {EXPECTED[tier]}")
    for col in ("District", "Region"):
        if col not in g.columns:
            raise SystemExit(f"{SHP[tier]}: no {col} column, got {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"{SHP[tier]}: crs is {g.crs}, expected EPSG:4326")
    return g


def _census(levels):
    """{name: region} for the census rows at the given geo_levels, from gh.csv."""
    import pandas as pd

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/gh.py first")
    df = pd.read_csv(NORM, dtype=str, low_memory=False)
    df = df[df["geo_level"].isin(levels)]
    out = {}
    for name, note in zip(df["geo_id"], df["note"]):
        m = re.search(r"region=([^;]+)", note)
        if not m:
            raise SystemExit(f"{name!r} carries no region in its note -- gh.py is stale")
        out[name] = m.group(1).strip()
    return out


def _join(tier, cen, gdf):
    """Print the join BOTH WAYS and fail on either side (§12)."""
    poly, dupes = {}, []
    for name in gdf["District"]:
        k = fold(name)
        if k in poly:
            dupes.append(name)
        poly[k] = name
    if dupes:
        raise SystemExit(f"tier {tier}: folded polygon names collide: {dupes}")

    cens = {}
    for name in cen:
        k = fold(name)
        if k in cens:
            raise SystemExit(f"tier {tier}: folded census names collide: {name!r}")
        cens[k] = name

    miss_c = sorted(k for k in cens if k not in poly)
    miss_p = sorted(k for k in poly if k not in cens)
    print(f"\n  tier {tier}: the join, both ways (§12)")
    print(f"    census rows                {len(cens):>4}")
    print(f"    polygons                   {len(poly):>4}")
    print(f"    matched                    {len(cens) - len(miss_c):>4}")
    print(f"    census with no polygon     {len(miss_c):>4}")
    print(f"    polygons with no census    {len(miss_p):>4}")
    for k in miss_c:
        print(f"      no polygon: {cens[k]!r}")
    for k in miss_p:
        print(f"      no census : {poly[k]!r}")
    if miss_c or miss_p:
        raise SystemExit(f"tier {tier}: join FAILED")
    return {cens[k]: poly[k] for k in cens}


def _verify_region(tier, cen, gdf, pairs):
    """The independent check: census region (from ROW ORDER) vs shapefile region (an
    attribute). A wrong pairing cannot keep these in agreement."""
    shp_region = dict(zip(gdf["District"], gdf["Region"]))
    bad, known = [], []
    for name, poly in pairs.items():
        a, b = cen[name], shp_region[poly]
        if fold(a) == fold(b):
            continue
        if RESOLVED_REGION.get(name) == (a, b):
            known.append((name, a, b))
        else:
            bad.append((name, a, b))
    agree = len(pairs) - len(bad) - len(known)
    print(f"    independent check — region agrees on {agree}/{len(pairs)}"
          + (f", {len(known)} known typo(s) in the boundary file" if known else ""))
    for name, a, b in known:
        print(f"      resolved: {name!r} census {a!r} vs shapefile {b!r} — see "
              "RESOLVED_REGION")
    for name, a, b in bad:
        print(f"      {name!r}: census says {a!r}, shapefile says {b!r}")
    if bad:
        raise SystemExit(f"tier {tier}: the positional region parse and the boundary file "
                         "disagree -- one of them is wrong and it is not safe to guess")


def _drop_lakes(out):
    """Subtract inland water from the placement polygons — see the LAKES note above.

    Placement only. No count moves: every dot stays in the district it was counted in, and
    this changes where inside that district it may land (spec §8.2).
    """
    import geopandas as gpd
    import shapely

    if not os.path.exists(LAKES):
        raise SystemExit(f"missing {LAKES} -- HydroLAKES is expected in ../data/ ; it is "
                         "not downloaded by this script")
    lakes = gpd.read_file(LAKES, bbox=GH_BBOX)
    if len(lakes) == 0:
        raise SystemExit("HydroLAKES returned 0 features in the Ghana bbox -- a read that "
                         "succeeds is not a read that returned data (§12)")
    if lakes.crs is not None and lakes.crs.to_epsg() != 4326:
        lakes = lakes.to_crs(4326)
    volta = lakes.loc[lakes["Lake_name"].fillna("") == "Volta", "Lake_area"]
    if volta.empty or volta.max() < 5000:
        raise SystemExit("Lake Volta is not in the HydroLAKES extract, or is the wrong "
                         "size -- the clip would silently do almost nothing")
    print(f"\n  inland water: {len(lakes)} HydroLAKES polygons in the Ghana bbox, "
          f"Volta {volta.max():,.0f} km²")

    water = shapely.union_all(lakes.geometry.values)
    before = out.geometry.values
    after = shapely.difference(before, water)

    a0, a1 = shapely.area(before), shapely.area(after)
    lost = 1.0 - a1 / a0
    back = (lost > KEEP_WHOLE_ABOVE) | shapely.is_empty(after)
    if back.any():
        after = after.copy()
        after[back] = before[back]
        print(f"    {int(back.sum())} unit(s) lose over {KEEP_WHOLE_ABOVE:.0%} and are "
              f"left UNCLIPPED: {sorted(out['unit'].values[back])}")

    touched = lost > 1e-9
    print(f"    {int(touched.sum())} of {len(out)} districts clipped; "
          f"{(a0.sum() - shapely.area(after).sum()) / a0.sum() * 100:.2f}% of the "
          "country's district area was inland water")
    worst = sorted(zip(lost, out["unit"]), reverse=True)[:5]
    for frac, name in worst:
        print(f"      {frac * 100:5.1f}%  {name}")

    out = out.copy()
    out["geometry"] = after
    return out


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    tiers = {}
    for tier, levels in [("261", ["district", "metro"]),
                         ("272", ["district", "submetro"])]:
        gdf = _read_tier(tier)
        cen = _census(levels)
        print(f"\n=== tier {tier}: {len(cen)} census units, {len(gdf)} polygons ===")
        pairs = _join(tier, cen, gdf)
        _verify_region(tier, cen, gdf, pairs)
        tiers[tier] = (gdf, cen, pairs)

    # The two files must describe the same country: the 272 tier is the 261 tier with six
    # metros subdivided, so their total area has to agree. An overhanging sub-metro layer
    # is the Budapest trap (§12) and would put metro dots outside the metro.
    a261 = _read_tier("261").to_crs(3857).area.sum()
    a272 = tiers["272"][0].to_crs(3857).area.sum()
    rel = abs(a272 - a261) / a261
    print(f"\n  total area, 261 tier vs 272 tier: {rel * 100:.4f}% apart")
    if rel > 1e-4:
        raise SystemExit("the two tiers do not cover the same country -- the sub-metro "
                         "layer is not a subdivision of the metro layer")

    gdf, cen, pairs = tiers[DRAWN]
    out = gdf[["District", "Region", "geometry"]].rename(
        columns={"District": "name", "Region": "region"})
    out["unit"] = out["name"]
    out = _drop_lakes(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "region", "geometry"]].to_file(
        OUT, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons, tier {DRAWN})")

    lut = pd.DataFrame({"geo_id": list(pairs), "unit": [pairs[k] for k in pairs]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
