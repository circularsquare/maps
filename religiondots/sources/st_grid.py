"""São Tomé and Príncipe — the placement grid: Kontur 400 m population hexagons.

Writes data/geo/st/st_hexes.gpkg. `countries.py` uses it to weight where a district's dots
land, never to change how many there are.

**SÃO TOMÉ NEEDS THIS BECAUSE FIVE OF THE SEVEN DISTRICTS ARE MOSTLY RAINFOREST.** Caué is
267 km² of the southern massif and the Obô national park, with 7,400 people on a coastal
strip and in a handful of old plantation *roças*; Lembá is the western windward side, 230 km²
at 74 people per km² against Água-Grande's 4,888. Area weighting would scatter dots up Pico de
São Tomé.

**THE STRAYS ARE SNAPPED, NOT DROPPED** — [[reference_archipelago_grid_snap]], and Cabo
Verde's rule at §9ci §5. This is two islands and a scatter of islets and every district
except Mé-Zóchi touches the sea, so a 400 m hex centroid landing just outside a generalised
shoreline is a loss in one direction only. Dropping those cells walks the dots inland, which
here means uphill into the forest the map is trying not to draw people on.

**THIS FILE CARRIES THE MAGNITUDE CHECK ON `st_geo.py`'s JOIN.** Witness 2 there is area and
witness 3 is INE's own 2024 recount, both of them the office's own numbers; Kontur is
modelled from OpenStreetMap building footprints and shares no lineage with either. It has to
agree with the 2012 census about how many people are in each of seven districts.

**THE VINTAGE GAP IS ELEVEN YEARS**, counts 2012 and grid 2023, and it is the largest of any
country here. It moves dots within a district and never between districts, but it is worth
naming: São Tomé's population grew 17% over that span and the growth was not even, so the
grid over-weights Água-Grande's newer periphery relative to 2012. `sources/st.md` §5 says so.

Usage:
    python sources/st_grid.py --fetch    one ~250 KB gz from Kontur
    python sources/st_grid.py            rebuild from data/raw/st/
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
RAW = os.path.join(ROOT, "data", "raw", "st")
GEO = os.path.join(ROOT, "data", "geo", "st")
UNITS = os.path.join(GEO, "st_districts.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "st.csv")
OUT = os.path.join(GEO, "st_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ST_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ST_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ST_20231101.gpkg"

EXPECTED_UNITS = 7
CENSUS_POPULATION = 178_739

# The grid is 2023 and the census is 2012, with 17% of national growth between them, so this
# is deliberately loose: it is looking for a failed download, not for a modelling difference.
NATIONAL_TOLERANCE = 0.45
# Seven units and the smallest is 6,031 people. A mispaired district would show as a factor
# of several; this band is set to catch that and not the forest districts' undercount.
UNIT_BAND = 2.0
# The country spans about 1° of longitude, Príncipe to São Tomé.
MAX_SPAN_DEG = 4.0

# A hex centroid this far outside a district is a coastline-resolution artefact and is
# snapped to the nearest one; anything further is dropped. Metres, in a projected CRS.
# EPSG:32632 is UTM 32N, which covers both islands.
SNAP_M = 700.0
SNAP_CRS = "EPSG:32632"


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
        raise SystemExit(f"missing {UNITS} — run sources/st_geo.py first")

    units = gpd.read_file(UNITS).to_crs("EPSG:4326")
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} districts, expected {EXPECTED_UNITS}")
    name_of = dict(zip(units["unit"], units["name"]))
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
              f"{100.0 * moved / stray:.1f}% of the strays) — coastal cells just seaward of a"
              "\n     generalised shoreline. Dropping them would pull both islands' dots "
              "inland,\n     which here is uphill into the Obô forest "
              "[[reference_archipelago_grid_snap]].")

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
    print(f"\n  Kontur {tot:,.0f} vs the 2012 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much even for an 11-year gap — check the download")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    if set(df["geo_id"]) != set(units["unit"]):
        raise SystemExit("st.csv and st_districts.gpkg disagree about the unit set — "
                         "re-run sources/st_geo.py")
    census = dict(zip(units["unit"], units["pop"]))

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print(f"\n  the seven districts, census against Kontur, normalised:")
    print(f"    {'':<32} {'census 2012':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<32} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} districts outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
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
    if beat > 20 or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach — the pairing in "
                         "st_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
