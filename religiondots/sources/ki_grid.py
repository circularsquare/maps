"""Kiribati — the placement grid: Kontur 400 m population hexagons, clipped to the 24 islands.

Writes data/geo/ki/ki_hexes.gpkg. `countries.py` uses it to weight where an island's dots land,
never to change how many there are.

**AN ATOLL IS A RING AND ITS MIDDLE IS A LAGOON.** Weighting by polygon area would put most of
Butaritari's and Abaiang's dots on open water inside the reef, and most of Kiritimati's on the
salt flats. The land is a strip a few hundred metres wide, so the grid is doing real work here
even though the units are whole islands.

**SOUTH TARAWA IS THE REASON THE WEIGHTING MATTERS MOST.** 39,058 people on 14 km² of causeway-
linked islets, next to North Tarawa's 6,629 on more land than that. They are separate units and
the grid keeps them apart.

**THE ANTIMERIDIAN CHECK IS PER CELL, NOT PER COUNTRY** — the same reasoning as `ki_geo.py`.
Kiribati's grid legitimately spans 351 degrees of longitude, so the usual width assertion fires
on correct data; a torn hexagon is caught by its own width instead.
[[reference_antimeridian]]

Usage:
    python sources/ki_grid.py --fetch    one small gz from Kontur
    python sources/ki_grid.py            rebuild from data/raw/ki/
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
RAW = os.path.join(ROOT, "data", "raw", "ki")
GEO = os.path.join(ROOT, "data", "geo", "ki")
UNITS = os.path.join(GEO, "ki_islands.gpkg")
LOOKUP = os.path.join(GEO, "ki_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ki.csv")
OUT = os.path.join(GEO, "ki_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KI_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KI_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KI_20231101.gpkg"

EXPECTED_UNITS = 24
CENSUS_POPULATION = 110_136

NATIONAL_TOLERANCE = 0.35
UNIT_BAND = 5.0
MAX_CELL_SPAN_DEG = 0.2

SNAP_M = 700.0
SNAP_CRS = "EPSG:3832"          # PDC Mercator, centred on the Pacific, so 180 is interior


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 20_000:
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
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} — run sources/ki_geo.py first")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} islands, expected {EXPECTED_UNITS}")
    units = units.to_crs("EPSG:4326")
    print(f"islands: {len(units)}, crs={units.crs}")

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
    # Per cell, not per country: Kiribati's grid legitimately spans the antimeridian.
    wide = hexes.to_crs("EPSG:4326").geometry.bounds
    worst = float((wide["maxx"] - wide["minx"]).max())
    print(f"  widest single hexagon spans {worst:.4f}° "
          f"(the country spans {pts.total_bounds[2] - pts.total_bounds[0]:.0f}°, correctly)")
    if worst > MAX_CELL_SPAN_DEG:
        raise SystemExit(f"a hexagon spans {worst:.2f}° and is torn across 180 "
                         "[[reference_antimeridian]]")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every island: {int(outside.sum()):,} "
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
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of an island and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — an atoll is a strip of land a "
              "few hundred\n     metres wide, so nearly every cell is a shoreline cell "
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

    name_of = dict(zip(units["unit"], units["name"]))
    have = set(out["unit"])
    tiny = sorted(set(units["unit"]) - have)
    if tiny:
        add = units[units["unit"].isin(tiny)][["unit", "geometry"]].copy()
        add["pop"] = 1.0
        print(f"\n  {len(tiny)} island(s) smaller than one 400 m hex, given their own polygon "
              f"as a single cell (§8.2): {[name_of[u] for u in tiny]}")
        out = gpd.GeoDataFrame(pd.concat([out, add[["unit", "pop", "geometry"]]],
                                         ignore_index=True), crs=out.crs)
        if len(tiny) > 6:
            raise SystemExit(f"{len(tiny)} islands have no hex at all — check the join")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"islands with no cell: {[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"islands whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} islands has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2015 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for an 8-year gap — check the download")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    census = df.groupby("geo_id")["count"].sum().to_dict()
    if set(census) != set(units["unit"]):
        raise SystemExit("ki.csv and ki_islands.gpkg disagree about the unit set")

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in census if u not in set(tiny)]
    rows.sort(key=lambda r: r[4])
    print(f"\n  census against Kontur, normalised ({len(census) - len(rows)} with a synthetic "
          "cell excluded):")
    print(f"    {'':<18} {'census 2015':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<18} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        print(f"    {len(worst)} outside a factor of {UNIT_BAND:g}: "
              f"{[(w[1], round(w[4], 2)) for w in worst]}")
        if len(worst) > 3:
            raise SystemExit("too many islands disagree with the grid to be modelling noise")
    else:
        print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g}")

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(max(r[3], 1e-9)) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  correlation on {len(rows)} units: r = {r_true:.4f}, against a best of "
          f"{perm[-1]:.4f}\n  over 2,000 random pairings ({beat} reach it).")
    if beat > 20 or r_true < 0.70:
        raise SystemExit(f"census and Kontur correlate at r={r_true:.4f}, which {beat} of "
                         "2,000 random pairings reach — the name join in ki_geo.py is wrong")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
