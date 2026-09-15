"""Seychelles: the placement grid, Kontur 400 m population hexagons cut to the 26 districts.

Writes data/geo/sc/sc_hexes.gpkg. `countries/sc.py` uses it to weight where a district's dots
land, never to change how many there are.

**THE HEXES ARE CUT BY OVERLAP, NOT ASSIGNED BY CENTROID.** Victoria's districts are tiny:
Mont Buxton is 1.16 km2, English River 1.38 without its island, Saint Louis 1.38, and a
0.74 km2 hexagon's centroid decides badly which of three such districts it belongs to (it
can leave one with no cell at all). So each hex is intersected with the district polygons
and its population is split between the pieces by land area. A hex that is partly sea keeps
its whole population on its land pieces.

**THE STRAYS ARE SNAPPED, NOT DROPPED** ([[reference_archipelago_grid_snap]]). A hex with no
land under COD's generalised shoreline at all goes, whole, to the nearest district within
SNAP_M. Every district but Mont Buxton, Bel Air and Les Mamelles touches the sea.

**PERSEVERANCE ISLAND'S CELLS ARE DROPPED.** The reclaimed island is part of English River in
the 2010 census (sources/sc_geo.py), but its housing estates were built after it: the first
houses were for the Indian Ocean Games of August 2011, and the 2022 census counts 5,410 people
there. Kontur is 2023, so it would put a
large share of English River's 2010 dots on an island that was mostly empty when they were
counted. The island's cells are removed before the cut, and the figure is printed.

**THE VINTAGE GAP IS THIRTEEN YEARS**, counts 2010 and grid 2023, the widest of any country
here. It moves dots within a district and never between districts.

Usage:
    python sources/sc_grid.py --fetch    one ~150 KB gz from Kontur
    python sources/sc_grid.py            rebuild from data/raw/sc/
"""

import gzip
import math
import os
import random
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sc")
GEO = os.path.join(ROOT, "data", "geo", "sc")
UNITS = os.path.join(GEO, "sc_districts.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "sc.csv")
OUT = os.path.join(GEO, "sc_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SC_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SC_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SC_20231101.gpkg"

EXPECTED_UNITS = 26
CENSUS_POPULATION = 90_945

# 2023 grid against a 2010 count; the 2022 census counted 102,612 and NBS estimates 119,878,
# so the grid should be well above the 2010 figure. Looking for a failed download only.
NATIONAL_BAND = (0.8, 1.8)
# Per district, after dividing out the national ratio. Other Islands is printed and not held
# to it: its 1,042 people in 2010 were mostly hotel and construction staff on islands whose
# buildings Kontur models as resorts, and a factor of several either way is not a join error.
UNIT_BAND = 2.0
UNIT_BAND_EXEMPT = {"SC-OI"}
MAX_SPAN_DEG = 12.0            # Aldabra at 46°E to Mahé at 55.5°E

AREA_CRS = "ESRI:102022"       # Africa Albers Equal Area Conic
SNAP_M = 700.0
PERSEVERANCE_BUFFER_M = 100.0


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 50_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg}; run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/sc_geo.py first")

    units = gpd.read_file(UNITS, layer="districts").to_crs("EPSG:4326")
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} districts, expected {EXPECTED_UNITS}")
    pers = gpd.read_file(UNITS, layer="perseverance").to_crs(AREA_CRS)
    name_of = dict(zip(units["unit"], units["name"]))
    census = dict(zip(units["unit"], units["pop"]))

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    hexes = hexes.rename(columns={popcol: "kpop"})[["kpop", "geometry"]]
    hexes = hexes[hexes["kpop"] > 0].to_crs(AREA_CRS).reset_index(drop=True)
    hexes["hid"] = hexes.index
    total_k = float(hexes["kpop"].sum())
    span = hexes.to_crs("EPSG:4326").total_bounds
    print(f"Kontur hexes: {len(hexes):,}, population {total_k:,.0f}, "
          f"lon {span[0]:.2f}-{span[2]:.2f}")
    if span[2] - span[0] > MAX_SPAN_DEG:
        raise SystemExit("the grid is torn [[reference_antimeridian]]")

    # Perseverance Island, built after the 2010 count.
    zone = pers.geometry.buffer(PERSEVERANCE_BUFFER_M).iloc[0]
    on_pers = hexes.geometry.centroid.within(zone)
    pers_pop = float(hexes.loc[on_pers, "kpop"].sum())
    print(f"\n  Perseverance Island: {int(on_pers.sum())} hexes, {pers_pop:,.0f} people in the "
          "2023 grid, DROPPED (built after the 2010 census)")
    hexes = hexes[~on_pers].copy()

    u = units[["unit", "geometry"]].to_crs(AREA_CRS)
    pieces = gpd.overlay(hexes, u, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    pieces = pieces[pieces["a"] > 1.0]
    land = pieces.groupby("hid")["a"].transform("sum")
    pieces["pop"] = pieces["kpop"] * pieces["a"] / land
    covered = set(pieces["hid"])
    strays = hexes[~hexes["hid"].isin(covered)].copy()
    stray_pop = float(strays["kpop"].sum())
    print(f"\n  hexes cut into {len(pieces):,} pieces over {len(covered):,} hexes; "
          f"{len(strays):,} hexes touch no district ({stray_pop:,.0f} people)")

    snapped = gpd.GeoDataFrame(columns=["unit", "pop", "geometry"], crs=AREA_CRS)
    if len(strays):
        pts = gpd.GeoDataFrame({"hid": strays["hid"], "kpop": strays["kpop"]},
                               geometry=strays.geometry.centroid, crs=AREA_CRS)
        near = gpd.sjoin_nearest(pts, u, how="left", max_distance=SNAP_M, distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        ok = near["unit"].notna()
        moved = float(near.loc[ok, "kpop"].sum())
        print(f"     {int(ok.sum()):,} within {SNAP_M:,.0f} m of a district are SNAPPED "
              f"({moved:,.0f} people); {int((~ok).sum()):,} further out are dropped "
              f"({stray_pop - moved:,.0f} people)")
        keep = strays.set_index("hid").loc[near.loc[ok, "hid"]]
        snapped = gpd.GeoDataFrame({"unit": near.loc[ok, "unit"].to_numpy(),
                                    "pop": keep["kpop"].to_numpy()},
                                   geometry=keep.geometry.to_numpy(), crs=AREA_CRS)

    out = pd.concat([pieces[["unit", "pop", "geometry"]], snapped], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=AREA_CRS).to_crs("EPSG:4326")
    out["pop"] = out["pop"].astype(float)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no cell: {[(m, name_of[m]) for m in missing]}")
    if (per["sum"] <= 0).any():
        raise SystemExit(f"districts whose cells sum to zero: {list(per.index[per['sum'] <= 0])}")
    hexcount = pieces.groupby("unit")["hid"].nunique()
    print(f"  every district has cells: {per['size'].min()}-{per['size'].max()} pieces, "
          f"median {int(hexcount.median())} distinct hexes per district")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} (after Perseverance) vs the 2010 census {CENSUS_POPULATION:,}: "
          f"ratio {ratio:.3f}")
    lo, hi = NATIONAL_BAND
    if not lo <= ratio <= hi:
        raise SystemExit(f"ratio {ratio:.2f} outside {lo}-{hi}; check the download")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    if set(df["geo_id"]) != set(units["unit"]):
        raise SystemExit("sc.csv and sc_districts.gpkg disagree about the unit set")

    rows = sorted(((m, name_of[m], census[m], float(per.loc[m, "sum"]),
                    float(per.loc[m, "sum"]) / census[m] / ratio) for m in census),
                  key=lambda r: r[4])
    print(f"\n    {'':<24}{'census 2010':>11}{'kontur':>9}{'norm':>7}")
    for m, nm, c, k, r in rows:
        print(f"    {nm:<24}{c:>11,}{k:>9,.0f}{r:>7.2f}")
    worst = [r for r in rows if r[0] not in UNIT_BAND_EXEMPT
             and not 1 / UNIT_BAND <= r[4] <= UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} districts outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
    print(f"    all but {sorted(UNIT_BAND_EXEMPT)} inside a factor of {UNIT_BAND:g}")

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(r[3]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(pearson(lc, sh))
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  log census against log Kontur over {len(rows)} districts: r = {r_true:.3f}; "
          f"{beat} of 2,000 shuffled pairings reach it")
    if beat > 20:
        raise SystemExit("the pairing in sc_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} pieces)")


if __name__ == "__main__":
    main()
