"""Samoa — the placement grid: Kontur 400 m population hexagons, clipped to the 25 districts.

Writes data/geo/ws/ws_hexes.gpkg. `countries.py` uses it to weight where a district's dots
land, never to change how many there are.

**SAMOA NEEDS THIS MORE THAN MOST.** Upolu and Savai'i are volcanic islands whose interiors are
forest and lava field and whose people live in a ring of villages along the coast road. Savai'i
in particular is 1,700 km² with an empty middle: weighting `Palauli` or `Gagaifomauga` by area
would put most of their dots on the slopes of Mount Silisili, where nobody lives.

**THE STRAYS ARE SNAPPED, NOT DROPPED** — Vanuatu's rule (§9bg §9). The settlement pattern is a
coastal ribbon, so hex centroids landing just seaward of a district outline are exactly the
cells that matter, and the loss would be directional.

**25 UNITS IS A WEAK CORRELATION TEST AND THAT IS WORTH SAYING.** With 183 wards (Solomon
Islands) or 156 villages (Tonga) the census-against-Kontur correlation is a real check on the
join. With 25 it is much easier to pass by luck, so it is reported and asserted but it is not
doing the work it does elsewhere. What actually guarantees Samoa's join is that it is a fold on
names, checked set-against-set in `ws_geo.py`, with no geocoding anywhere in it.

Usage:
    python sources/ws_grid.py --fetch    one ~0.2 MB gz from Kontur
    python sources/ws_grid.py            rebuild from data/raw/ws/
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
RAW = os.path.join(ROOT, "data", "raw", "ws")
GEO = os.path.join(ROOT, "data", "geo", "ws")
UNITS = os.path.join(GEO, "ws_districts.gpkg")
LOOKUP = os.path.join(GEO, "ws_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ws.csv")
OUT = os.path.join(GEO, "ws_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_WS_20231101.gpkg.gz")
GZ_NAME = "kontur_population_WS_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_WS_20231101.gpkg"

EXPECTED_UNITS = 25
CENSUS_POPULATION = 205_557

NATIONAL_TOLERANCE = 0.35
UNIT_BAND = 3.0
MAX_SPAN_DEG = 6.0

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
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} — run sources/ws_geo.py first")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} districts, expected {EXPECTED_UNITS}")
    units = units.to_crs("EPSG:4326")
    print(f"districts: {len(units)}, crs={units.crs}")

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
          f"(districts {units.total_bounds[2] - units.total_bounds[0]:.2f}°)")
    if span > MAX_SPAN_DEG:
        raise SystemExit(f"the grid spans {span:.1f}°; something is torn "
                         "[[reference_antimeridian]]")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every district: {int(outside.sum()):,} "
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
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of a district and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — the settlement pattern is a "
              "coastal ribbon,\n     so dropping them would pull every shore's dots inland "
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
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no cell: {[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} districts has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2021 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a 2-year gap — check the download")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    df["unit"] = df["geo_id"].map(unit_of)
    if df["unit"].isna().any():
        raise SystemExit("ws.csv has villages missing from ws_lookup.csv — re-run ws_geo.py")
    census = df.groupby("unit")["count"].sum().to_dict()
    if set(census) != set(units["unit"]):
        raise SystemExit("ws.csv and ws_districts.gpkg disagree about the unit set")

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print(f"\n  census against Kontur, normalised, all {len(rows)} districts:")
    print(f"    {'':<24} {'census 2021':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<24} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} districts outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]} — on a fold-on-names "
                         "join with 25 units this should not happen at all")
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
    print(f"\n  correlation on {len(rows)} units: r = {r_true:.4f}, against a best of "
          f"{perm[-1]:.4f}\n  over 2,000 random pairings ({beat} reach it). With only 25 "
          "units this is\n  corroboration, not the check that carries the join (see the "
          "module docstring).")
    if beat > 20 or r_true < 0.70:
        raise SystemExit(f"census and Kontur correlate at r={r_true:.4f}, which {beat} of "
                         "2,000 random pairings reach — the fold in ws_geo.py is wrong")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
