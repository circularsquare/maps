"""Tonga — the placement grid: Kontur 400 m population hexagons, clipped to villages.

Writes data/geo/to/to_hexes.gpkg. `countries.py` uses it to weight where a village's dots
land, never to change how many there are.

**TONGA NEEDS THIS BECAUSE THE VILLAGE POLYGONS ARE LAND ALLOTMENTS, NOT SETTLEMENTS.** A
Tongan village district on Tongatapu runs from the shore back across its bush allotments, and
the houses are in a band at one end of it. Weighting by area would scatter the dots across the
plantations. On the outer islands the mismatch is bigger still: `Ha'atu'a` on 'Eua is 43.5 km²
of forested plateau with its people on the west coast road.

**THE STRAYS ARE SNAPPED, NOT DROPPED** — Vanuatu's rule (§9bg §9), and it matters more here
than almost anywhere. Tonga is 171 islands and the whole country is coastline; on a 400 m grid
against a detailed shoreline a large share of hex centroids land just seaward of the village
outline, and that loss is *directional*, since the settlements are the coast. Ten COD polygons
are uninhabited islets that are not units at all (`sources/to_geo.py`), so whatever the grid
puts on them is snapped to the nearest village rather than lost.

**THE VINTAGE GAP IS TWO YEARS**, counts 2021 and grid 2023, and it moves dots within a
village, never between villages.

**156 UNITS MAKES THE CORRELATION A REAL CHECK ON `to_geo.py`'s JOIN.** That join is on the
district-qualified name with five witnessed overrides, and its division witness is clean; this
is a third, independent confirmation, because a modelled 2023 grid sharing no lineage with
TSD's census or with OCHA's boundaries has to agree about how many people are in each of 156
villages. It is the check that would catch a Niuafo'ou village paired with its 'Eua twin.

Usage:
    python sources/to_grid.py --fetch    one ~90 KB gz from Kontur
    python sources/to_grid.py            rebuild from data/raw/to/
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
RAW = os.path.join(ROOT, "data", "raw", "to")
GEO = os.path.join(ROOT, "data", "geo", "to")
VILLAGES = os.path.join(GEO, "to_villages.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "to.csv")
OUT = os.path.join(GEO, "to_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TO_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TO_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TO_20231101.gpkg"

EXPECTED_UNITS = 156
CENSUS_POPULATION = 99_408

NATIONAL_TOLERANCE = 0.35
# 156 units and the smallest is 7 people, so a generous band: this is looking for a mispaired
# village, not a modelling difference.
UNIT_BAND = 6.0
MAX_SPAN_DEG = 12.0

# A hex centroid this far outside a village is a coastline-resolution artefact and is snapped
# to the nearest village; anything further is dropped. Metres, in a projected CRS.
SNAP_M = 700.0
SNAP_CRS = "EPSG:3832"


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
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(VILLAGES):
        raise SystemExit(f"missing {VILLAGES} — run sources/to_geo.py first")

    units = gpd.read_file(VILLAGES)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{VILLAGES} has {len(units)} villages, expected {EXPECTED_UNITS}")
    units = units.to_crs("EPSG:4326")
    print(f"villages: {len(units)}, crs={units.crs}")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs("EPSG:4326")
    span = pts.total_bounds[2] - pts.total_bounds[0]
    print(f"  hex centroids span {span:.2f}° of longitude "
          f"(villages {units.total_bounds[2] - units.total_bounds[0]:.2f}°)")
    if span > MAX_SPAN_DEG:
        raise SystemExit(f"the grid spans {span:.1f}°; something is torn "
                         "[[reference_antimeridian]]")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every village: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / pts[popcol].sum():.3f}%)")
    if outside.any():
        m_units = units[["unit", "geometry"]].to_crs(SNAP_CRS)
        m_pts = pts.loc[outside].to_crs(SNAP_CRS)
        near = gpd.sjoin_nearest(m_pts, m_units, how="left", max_distance=SNAP_M,
                                 distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        snapped = joined.loc[outside, "unit"].notna()
        moved = float(pts.loc[outside][snapped.to_numpy()][popcol].sum())
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of a village and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — coastal cells just seaward of a"
              "\n     detailed shoreline, plus the ten uninhabited islets COD carries and the"
              "\n     census does not. Dropping them would pull every shore's dots inland "
              "(§9bg §9).")

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"     {int(outside.sum()):,} cells remain unplaced ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%) and are dropped;")
    print("     these are placement WEIGHTS and not counts, so nobody leaves the map.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=hexes.crs).to_crs(
        "EPSG:4326")

    # ---- A VILLAGE SMALLER THAN ONE HEX GETS ITS OWN POLYGON AS A CELL ----
    #
    # Tonga has villages of 7 and 11 people on lagoon islets, well under one 400 m cell, and
    # the town wards of Nuku'alofa are small too. Where no hex centroid falls inside a village
    # that is the Kontur resolution floor rather than a join failure ([[reference_kontur_
    # resolution_floor]]): the grid is coarser than the unit.
    #
    # Snapping such a village to a neighbour's hex would place its dots in the wrong village,
    # and leaving it out drops it from the map entirely -- `countries.py` has no separate unit
    # layer for Tonga, so the hexes ARE the geography. Instead the village's own polygon is
    # added as a single cell, which is §8.2's equal share over the unit.
    name_of = dict(zip(units["unit"], units["village"]))
    have = set(out["unit"])
    tiny = sorted(set(units["unit"]) - have)
    if tiny:
        add = units[units["unit"].isin(tiny)][["unit", "geometry"]].copy()
        add["pop"] = 1.0
        areas = units.to_crs(SNAP_CRS).set_index("unit").geometry.area / 1e6
        print(f"\n  {len(tiny)} village(s) smaller than one 400 m hex, given their own "
              "polygon as a single cell:")
        for u in tiny:
            print(f"     {name_of[u]} ({u}) — {areas[u]:.2f} km²; dots spread evenly "
                  "inside the village (§8.2)")
        out = gpd.GeoDataFrame(pd.concat([out, add[["unit", "pop", "geometry"]]],
                                         ignore_index=True),
                               crs=out.crs)
        if len(tiny) > 12:
            raise SystemExit(f"{len(tiny)} villages have no hex at all, which is too many to "
                             "be the resolution floor — check the join in to_geo.py")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"villages with no cell after the fallback: "
                         f"{[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"villages whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} villages has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2021 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a 2-year gap — check the download")

    # to.csv has no `Total` row: the 22 categories partition the village exactly (asserted in
    # sources/to.py), so the village total is their sum.
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    census = df.groupby("geo_id")["count"].sum().to_dict()
    if set(census) != set(units["unit"]):
        raise SystemExit("to.csv and to_villages.gpkg disagree about the unit set — "
                         "re-run sources/to_geo.py")

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in census if u not in set(tiny)]
    rows.sort(key=lambda r: r[4])
    print(f"\n  the five worst-agreeing villages each way, census against Kontur, normalised"
          f"\n  ({len(census) - len(rows)} with a synthetic cell excluded):")
    print(f"    {'':<24} {'census 2021':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows[:5] + rows[-5:]:
        print(f"    {nm:<24} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]

    if len(worst) > 8:
        raise SystemExit(f"{len(worst)} villages outside a factor of {UNIT_BAND:g}, which is "
                         "too many to be the grid's blind spots: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
    if worst:
        share = sum(w[2] for w in worst) / sum(r[2] for r in rows)
        print(f"    {len(rows) - len(worst)} of {len(rows)} inside a factor of "
              f"{UNIT_BAND:g}; the {len(worst)} outside are "
              f"{100 * share:.2f}% of the counted population:")
        for u, nm, c, k, r in worst:
            print(f"      {nm:<24} census {c:>6,} vs Kontur {k:>6,.0f}  ({r:.2f}x)")
        print("      Placement weights only; no count moves.")
    else:
        print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g}")

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(r[3]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  and the correlation, on {len(rows)} units: r = {r_true:.4f}, against a best "
          f"of {perm[-1]:.4f}\n  over 2,000 random pairings ({beat} reach it).")
    if beat > 20 or r_true < 0.70:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach — the name join in "
                         "to_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
