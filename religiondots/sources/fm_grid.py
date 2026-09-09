"""Micronesia — the placement grid: Kontur 400 m population hexagons, clipped to the 33 units.

Writes data/geo/fm/fm_hexes.gpkg. `countries.py` uses it to weight where a unit's dots land,
never to change how many there are.

**A UNIT HERE IS MOSTLY OCEAN AND TWO OF THEM ARE ENORMOUS.** Chuuk State is drawn whole and
reaches from Houk in the Western Islands to Ta in the Lower Mortlocks, about 1,000 km; Yap's
municipalities include Satawal and Eauripik, single atolls a kilometre or two across. Weighting
by polygon area would put most of the country's dots on open water, and Chuuk's would land in
the lagoon rather than on Weno, where a third of the state lives. So the grid is doing more work
in Micronesia than in almost any other country here.

**AN ATOLL IS A STRIP OF LAND A FEW HUNDRED METRES WIDE**, so most Kontur cells are shoreline
cells whose centroid falls just outside the polygon. They are SNAPPED to the nearest unit within
`SNAP_M` rather than dropped, on Vanuatu's rule (§9bg) and Kiribati's precedent — dropping them
would quietly move an island's dots onto its neighbour.
[[reference_archipelago_grid_snap]]

**THE UNITS ARE TWO TIERS AND THE BAND CHECK HAS TO KNOW.** Chuuk is 33,883 people in one unit
and Ngulu is 11, so a single ratio band across all 33 would be dominated by the two state-level
units. The band is applied to every unit, and the correlation test runs on all of them, but the
per-unit table prints the tier so a failure can be read.

**MICRONESIA DOES NOT CROSS THE ANTIMERIDIAN** (137°E to 163°E), so the ordinary per-country
width assertion is the right one and is used. Kiribati and Fiji needed per-polygon checks
instead; this country does not. [[reference_antimeridian]]

Usage:
    python sources/fm_grid.py --fetch    one small gz from Kontur
    python sources/fm_grid.py            rebuild from data/raw/fm/
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
RAW = os.path.join(ROOT, "data", "raw", "fm")
GEO = os.path.join(ROOT, "data", "geo", "fm")
UNITS = os.path.join(GEO, "fm_units.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "fm.csv")
OUT = os.path.join(GEO, "fm_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_FM_20231101.gpkg.gz")
GZ_NAME = "kontur_population_FM_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_FM_20231101.gpkg"

EXPECTED_UNITS = 33
CENSUS_POPULATION = 75_817

# KONTUR IS HALF AS BIG AGAIN AS THE COUNTRY AND THAT IS THE COUNTRY'S FAULT, NOT THE GRID'S.
# The 2023 census counted 75,817 people; the 2010 census counted 107,008, and Micronesia lost
# nearly a third of its people to emigration under the Compact in between. Kontur's 2023-11
# extract reads 113,340, which is the 2010 level and not the 2023 one, so the NATIONAL ratio is
# about 1.50 on a correct download. That is tolerated because the grid is only ever a WITHIN-UNIT
# weight and is normalised by this ratio before any unit is judged; what would actually be
# dangerous is a grid that is wrong about the RELATIVE size of one unit, and the per-unit band
# below is what tests for that (spec §12's Eswatini finding). Do not narrow this without
# checking the census year first.
NATIONAL_TOLERANCE = 0.70
UNIT_BAND = 5.0
MAX_CELL_SPAN_DEG = 0.2
MAX_COUNTRY_SPAN_DEG = 27.0

SNAP_M = 700.0
SNAP_CRS = "EPSG:3832"          # PDC Mercator, the Pacific-centred one the region is cut for


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
        raise SystemExit(f"missing {UNITS} — run sources/fm_geo.py first")

    units = gpd.read_file(UNITS).to_crs("EPSG:4326")
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")
    print(f"units: {len(units)} "
          f"({(units['level'] == 'municipality').sum()} municipalities, "
          f"{(units['level'] == 'state').sum()} whole states), crs={units.crs}")

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
    wide = hexes.to_crs("EPSG:4326").geometry.bounds
    worst = float((wide["maxx"] - wide["minx"]).max())
    span = float(pts.total_bounds[2] - pts.total_bounds[0])
    if worst > MAX_CELL_SPAN_DEG or span > MAX_COUNTRY_SPAN_DEG:
        raise SystemExit(f"widest hexagon {worst:.2f}°, country span {span:.1f}° — something "
                         "is torn across 180 [[reference_antimeridian]]")
    print(f"  widest single hexagon {worst:.4f}°, the country {span:.1f}° — neither is torn")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every unit: {int(outside.sum()):,} "
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
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of a unit and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — an atoll is a strip of land a "
              "few hundred\n     metres wide, so nearly every cell is a shoreline cell "
              "(§9bg).")

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"     {int(outside.sum()):,} cells remain unplaced ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%) and are dropped;")
    print("     these are placement WEIGHTS and not counts, so nobody leaves the map.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=hexes.crs).to_crs("EPSG:4326")

    name_of = dict(zip(units["unit"], units["name"]))
    level_of = dict(zip(units["unit"], units["level"]))
    have = set(out["unit"])
    tiny = sorted(set(units["unit"]) - have)
    if tiny:
        add = units[units["unit"].isin(tiny)][["unit", "geometry"]].copy()
        add["pop"] = 1.0
        print(f"\n  {len(tiny)} unit(s) smaller than one 400 m hex, given their own polygon "
              f"as a single cell (§8.2): {[name_of[u] for u in tiny]}")
        out = gpd.GeoDataFrame(pd.concat([out, add[["unit", "pop", "geometry"]]],
                                         ignore_index=True), crs=out.crs)
        if len(tiny) > 6:
            raise SystemExit(f"{len(tiny)} units have no hex at all — check the join")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no cell: {[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"units whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} units has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2023 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much — check the download. Note FSM lost nearly a "
                         "third of its people between 2010 and 2023, so a grid built on "
                         "older inputs reads high rather than low.")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    census = df.groupby("geo_id")["count"].sum().to_dict()
    if set(census) != set(units["unit"]):
        raise SystemExit("fm.csv and fm_units.gpkg disagree about the unit set")

    rows = [(u, name_of[u], level_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in census if u not in set(tiny)]
    rows.sort(key=lambda r: r[5])
    print(f"\n  census against Kontur, normalised ({len(census) - len(rows)} with a synthetic "
          "cell excluded):")
    print(f"    {'':<22} {'tier':<13} {'census':>7} {'kontur':>8} {'norm':>6}")
    for u, nm, lv, c, k, r in rows:
        print(f"    {nm:<22} {lv:<13} {c:>7,} {k:>8,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[5] < 1 / UNIT_BAND or r[5] > UNIT_BAND]
    if worst:
        print(f"    {len(worst)} outside a factor of {UNIT_BAND:g}: "
              f"{[(w[1], round(w[5], 2)) for w in worst]}")
        if len(worst) > 4:
            raise SystemExit("too many units disagree with the grid to be modelling noise "
                             "— [[reference_kontur_resolution_floor]] and spec §12's "
                             "Eswatini finding both start here")
    else:
        print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g}")

    lc = [math.log(r[3]) for r in rows]
    lk = [math.log(max(r[4], 1e-9)) for r in rows]
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
                         "2,000 random pairings reach — the name join in fm_geo.py is wrong")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
