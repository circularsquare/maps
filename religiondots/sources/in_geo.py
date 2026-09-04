"""India — sub-district boundaries for the Census 2011 religion data.

Writes data/geo/in/in_subdistricts.gpkg, the single layer countries.py reads, and prints the
join against data/normalized/in.csv both ways.

SOURCE: SHRUG (Development Data Lab) open-source polygons, PC11 vintage — the 2011 Census
sub-districts, keyed on the census's own `pc11_s_id` / `pc11_d_id` / `pc11_sd_id`.  Stitched
by DDL from SEDAC, Bharatmaps, Datameet and the Administrative Atlas of India, and it is the
only openly redistributable boundary set that carries census codes at this level.

**Not downloaded from devdatalab.org, which gates the file behind a form.**  It is mirrored
verbatim as parquet in the `india-geodata` repository's `census/2011` release, which is a
plain GitHub asset.  Licence travels with it: CC0 / CC-BY-NC-SA-4.0, so NON-COMMERCIAL —
the first source on the map with that restriction, and worth knowing before anything here is
ever sold.  Recorded in sources.md §6.

WHY NOT THE ALTERNATIVES, all three of which were tried:

  - **The Registrar General's own SubDistricts_2011** is in the same release and is worse:
    its codes split units into `(Pt)` parts, so 1,457 census units and 1,428 polygons fail
    to match and 15.1% of the population drops.  It does carry `Tot_pop`, which is how the
    SHRUG join below is checked against something other than itself.
  - **geoBoundaries IND ADM3** is 6,836 units of 2018 vintage with NO census codes, so it
    would be a name join across 6,836 Indian transliterations against a 2011 table — §8.1's
    hazard and §12's "resolve leftovers by elimination, never by guessing", at a scale where
    neither is possible.
  - **Datameet's 2011_Dist.shp** is districts only, one level too coarse.

THE JOIN IS AN EXACT CODE JOIN and needs no derivation: `geo_id` is state(2) + district(3) +
subdistrict(5), which is exactly how in.py builds it and exactly the widths SHRUG uses.
5,946 of 5,969 polygons match on the first try.  The independent check that this is right
rather than lucky is names — 5,9xx of the matched pairs agree on the sub-district name — and,
better, the Registrar General's file agrees with the census population EXACTLY on 4,504 of
the 4,534 units where its codes line up.  A code join that also reproduces the published
population is not a coincidence.

THE 23 UNITS THAT HAVE NO SUB-DISTRICT POLYGON, and this is the interesting part.

The census emits, in some districts, a unit called **`Area not under any Sub-district`** with
sub-district code `99999`.  There are 23 of them with no SHRUG polygon, and they are not
small: **17,374,936 people, 1.4% of India**, concentrated in West Bengal (18 units, 16.7M),
Tripura (4) and Karnataka (1).  Among them are the whole Kolkata metropolitan fringe —
5.0M in North Twenty Four Parganas alone — which is some of the densest inhabited ground on
earth.  Dropping them was never an option and neither was smearing them.

Three things were established before choosing:

  1. **There is no leftover geometry to give them.**  SHRUG's sub-district polygons TILE
     each district completely, so `district - union(sub-districts)` is empty to four decimal
     places for every one of the 23.  The urban ground is already inside the neighbouring
     rural polygons.
  2. **Spreading them over the district would be badly wrong**, not merely coarse.  Haora's
     1.6M and North Twenty Four Parganas' 5.0M are municipal corporations occupying a few
     hundred km² of a district several thousand km² across.
  3. **The census says exactly what they are made of.**  C-01's TOWN rows inside a `99999`
     unit sum to that unit's population — **100.0% exactly, in all three states** — and they
     are named municipal bodies: Kolkata (M Corp.), Haora, Asansol, Durgapur, Agartala,
     Hubli-Dharwad, BBMP.  All 150 of them have a SHRUG town polygon.

So a `99999` unit's geometry is **the union of its own towns**, which is a fact from the
source rather than an estimate.  The unit keeps its single set of religion counts; only its
shape comes from the towns.  Splitting it into per-town units is a further upgrade available
for free — the town rows carry the full eight categories — and is not taken here because the
count unit stays the census's own.

PLACEMENT (spec §8.2), and India breaks the rule the section is built on.

§8.2 places dots by giving each polygon of a fine layer an equal share, on the grounds that
statistical agencies design their fine units to a population target — US tracts ~3,400
people, Australian SA1s ~406.  **India has no such layer.**  The finer geography available
is 645,828 villages and 4,135 towns, and those are natural settlements ranging from ten
people to two million, not units engineered to a target.  An equal share per village would
weight a hamlet like a small city.

Sub-districts are therefore both the count layer and the placement layer, as in Poland,
Romania and Brazil.  The cost is real and should be stated: the median sub-district holds
about 204,000 people, the coarsest count unit on the map.  In compensation its median AREA,
551 km², is finer than a Brazilian município's 1,527 km², so at national and state zoom the
grain is comparable to a country already drawn; it is at city zoom that India will look
blockier than anywhere else.

THE UPGRADE, recorded because it is specific and now unblocked: `Census_Villages.parquet`
in the same release carries 645,828 village POINTS with `t_pop2011`, summing to 828.9M —
India's entire rural population — and the towns file supplies the urban half.  Weighting
placement by that would put rural dots on real settlements instead of spreading them across
a polygon.  scatter.py's `place_weight` hook (the US's §8.4 weighter) is the right shape
for it; India does not use it today only because its placement layer IS its count layer,
one polygon per unit, so there is nothing inside a unit to weight.  See sources/in_geo.md
§4 for the two pieces of wiring that would change that.

Usage:
    python sources/in_geo.py --fetch    download the parquets (~345MB) if missing
    python sources/in_geo.py            build the gpkg from data/geo/in/
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "in")
NORM = os.path.join(ROOT, "data", "normalized", "in.csv")
OUT = os.path.join(GEO, "in_subdistricts.gpkg")

BASE = ("https://github.com/yashveeeeeeer/india-geodata/releases/download/census%2F2011/")
FILES = {
    "shrug-subdistrict-pc11.parquet": 35_000_000,
    # Needed ONLY for the 150 town polygons that make up the 23 `99999` units. It is a
    # heavy dependency for a small job — 309MB for 17.4M people — and it is taken because
    # the alternative is misplacing the Kolkata metropolitan fringe.
    "shrug-village-pc11.parquet": 300_000_000,
    # Not used to build the layer. Downloaded because it carries Tot_pop per sub-district
    # and is the only independent check available on the join (see check_against_rgi).
    "SubDistricts_2011.parquet": 50_000_000,
}

RESIDUAL_SD = "99999"


def fetch():
    os.makedirs(GEO, exist_ok=True)
    import requests
    for name, min_bytes in FILES.items():
        dest = os.path.join(GEO, name)
        if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
            print(f"  have {name} ({os.path.getsize(dest):,} bytes)")
            continue
        url = BASE + name
        print(f"  getting {name} ...")
        r = requests.get(url, timeout=1800, stream=True)
        if r.status_code != 200:
            raise SystemExit(f"{name}: HTTP {r.status_code}")
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        n = os.path.getsize(dest)
        if n < min_bytes:
            raise SystemExit(f"{name}: {n:,} bytes, expected >= {min_bytes:,}")
        print(f"  got  {name} ({n:,} bytes)")


def town_geometry(need, path):
    """Union of a `99999` unit's own town polygons, one geometry per unit.

    `need` maps geo_id -> list of six-digit town codes, read from C-01. Every town must be
    present: a missing one would silently shrink the unit's footprint rather than fail, and
    the whole point of this path is that the towns account for the unit exactly.
    """
    import pyarrow.parquet as pq
    import shapely
    wanted = {t for codes in need.values() for t in codes}
    # Read the 150 rows wanted rather than all 649,618. The filter is pushed into the
    # parquet reader, and because that bypasses geopandas' metadata the geometry arrives as
    # raw WKB and has to be decoded by hand.
    tbl = pq.read_table(path, columns=["pc11_tv_id", "tv_name", "geometry"],
                        filters=[("pc11_tv_id", "in", sorted(wanted))])
    v = tbl.to_pandas()
    v["geometry"] = shapely.from_wkb(v["geometry"])
    v = gpd.GeoDataFrame(v, geometry="geometry", crs="OGC:CRS84")
    have = set(v["pc11_tv_id"])
    missing = wanted - have
    if missing:
        raise SystemExit(f"{len(missing)} town polygons missing, e.g. {sorted(missing)[:5]}")
    by_town = v.set_index("pc11_tv_id").geometry
    rows = []
    for gid, codes in need.items():
        geom = by_town.loc[list(codes)].union_all()
        if geom.is_empty:
            raise SystemExit(f"{gid}: union of {len(codes)} towns is empty")
        rows.append({"kod": gid, "name": "Area not under any Sub-district",
                     "src": "town-union", "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="OGC:CRS84")


def residual_towns():
    """geo_id -> town codes, for every `99999` unit, straight from the C-01 workbooks.

    Read from raw rather than from in.csv because in.csv deliberately keeps only the `Total`
    rows and town rows are Urban-only (see in.py). This is the one place that needs them.
    """
    import re
    raw = os.path.join(ROOT, "data", "raw", "in")
    pat = re.compile(r"DDW(\d\d)C-01 MDDS\.XLS$", re.I)
    need = {}
    for f in sorted(os.listdir(raw)):
        m = pat.search(f)
        if not m or m.group(1) == "00":
            continue
        df = pd.ExcelFile(os.path.join(raw, f)).parse("C01", header=None, dtype=object)
        df = df[df[0].astype(str).str.strip() == "C0101"]
        if df.empty:
            continue
        d = df[[1, 2, 3, 4, 6]].copy()
        d.columns = ["s", "dd", "sd", "town", "tru"]
        for c in ("s", "dd", "sd", "town"):
            d[c] = d[c].astype(str).str.strip()
        t = d[(d["sd"] == RESIDUAL_SD) & (d["town"] != "000000") & (d["tru"] == "Urban")]
        for _, r in t.iterrows():
            need.setdefault(r["s"] + r["dd"] + r["sd"], []).append(r["town"])
    return need


def check_against_rgi(g, sub):
    """Independent evidence the code join is right: the Registrar General's own population.

    A code join can match on both sides and still be wrong (spec §8.1, Poland's LAU_ID).
    Names are one check; a completely separate agency's population per code is a better one,
    because it cannot agree by accident.
    """
    path = os.path.join(GEO, "SubDistricts_2011.parquet")
    if not os.path.exists(path):
        print("  (SubDistricts_2011.parquet absent — population cross-check skipped)")
        return
    r = gpd.read_parquet(path)
    r["kod"] = (r["stcode11"].astype(str) + r["dtcode11"].astype(str)
                + r["sdtcode11"].astype(str))
    m = sub.merge(r[["kod", "Tot_pop"]].drop_duplicates("kod"),
                  left_on="geo_id", right_on="kod")
    exact = int((m["count"] == m["Tot_pop"]).sum())
    print(f"  cross-check vs the Registrar General's own file: {len(m):,} units share a "
          f"code, {exact:,} agree on population EXACTLY ({exact / len(m):.1%})")


def build():
    src = os.path.join(GEO, "shrug-subdistrict-pc11.parquet")
    g = gpd.read_parquet(src)
    g["kod"] = g["pc11_s_id"] + g["pc11_d_id"] + g["pc11_sd_id"]
    g = g.rename(columns={"sd_name": "name"})[["kod", "name", "geometry"]]
    g["src"] = "shrug-subdistrict"
    print(f"SHRUG sub-districts: {len(g):,} polygons, {g['kod'].nunique():,} unique codes")

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    sub = df[(df["geo_level"] == "subdistrict") & (df["source_category"] == "Total")]
    print(f"census sub-districts: {len(sub):,}, {sub['count'].sum():,} people")

    # ---- the join, both ways, before anything is repaired
    left, right = set(sub["geo_id"]), set(g["kod"])
    only_census, only_poly = left - right, right - left
    lost = sub[sub["geo_id"].isin(only_census)]
    print(f"\n  census units with no polygon: {len(only_census)} "
          f"({lost['count'].sum():,} people, {lost['count'].sum() / sub['count'].sum():.3%})")
    print(f"  polygons with no census unit: {len(only_poly)} -> dropped: "
          f"{sorted(only_poly)}")

    # Every unmatched census unit must be a `99999` residual. Anything else is a real
    # vintage or coding failure and must not be papered over by the town path below.
    odd = {u for u in only_census if not u.endswith(RESIDUAL_SD)}
    if odd:
        raise SystemExit(f"{len(odd)} unmatched units are NOT `Area not under any "
                         f"Sub-district`: {sorted(odd)[:10]}")

    # ---- repair the residuals from their own towns
    if only_census:
        need = {k: v for k, v in residual_towns().items() if k in only_census}
        missing = only_census - set(need)
        if missing:
            raise SystemExit(f"{len(missing)} residual units have no town rows: "
                             f"{sorted(missing)[:5]}")
        extra = gpd.GeoDataFrame(town_geometry(
            need, os.path.join(GEO, "shrug-village-pc11.parquet")))
        ntowns = sum(len(v) for v in need.values())
        print(f"  built {len(extra)} residual polygons from {ntowns} town polygons "
              f"({lost['count'].sum():,} people recovered)")
        g = gpd.GeoDataFrame(pd.concat([g, extra], ignore_index=True),
                             geometry="geometry", crs=g.crs)

    g = g[g["kod"].isin(left)].copy()

    # ---- three-way check (spec §8.1): unmatched data, unmatched polygons, AND matched
    # codes whose geometry is empty or null, which two-way matching cannot see.
    bad = g[g.geometry.is_empty | g.geometry.isna()]
    if len(bad):
        raise SystemExit(f"{len(bad)} matched units have empty geometry: "
                         f"{sorted(bad['kod'])[:10]}")
    still = left - set(g["kod"])
    if still:
        raise SystemExit(f"{len(still)} census units still unplaced: {sorted(still)[:10]}")
    print(f"  matched-but-empty geometry: 0")

    # ---- names, as the independent check on the code join
    nm = sub.merge(g[["kod", "name"]], left_on="geo_id", right_on="kod")
    same = (nm["geo_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
            == nm["name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)).sum()
    print(f"  name agreement: {same:,}/{len(nm):,} ({same / len(nm):.1%}) after folding "
          f"case and punctuation")

    check_against_rgi(g, sub)

    os.makedirs(GEO, exist_ok=True)
    g[["kod", "name", "src", "geometry"]].to_file(OUT, driver="GPKG", layer="subdistricts")
    print(f"\nwrote {OUT}: {len(g):,} polygons covering {sub['count'].sum():,} people")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the parquets first")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    build()


if __name__ == "__main__":
    main()
