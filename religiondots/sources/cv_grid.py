"""Cabo Verde — the placement grid: Kontur 400 m population hexagons, clipped to concelhos.

Writes data/geo/cv/cv_hexes.gpkg. `countries.py` uses it to weight where a concelho's dots
land, never to change how many there are.

**CABO VERDE NEEDS THIS BECAUSE THE CONCELHOS ARE MOSTLY EMPTY VOLCANO.** Santa Catarina do
Fogo is the caldera of Pico do Fogo and its people live in two villages on the rim; Porto
Novo is 558 km² of Santo Antão, almost all of it uninhabited lava and ravine, with its
population in the port and along the Ribeira das Patas; Boa Vista and Maio are dune. Area
weighting would scatter dots across ground nobody has ever lived on.

**THE STRAYS ARE SNAPPED, NOT DROPPED** — [[reference_archipelago_grid_snap]], and Vanuatu's
rule at §9bg §9. Cabo Verde is ten islands and the whole country is coastline, and on the
larger islands the settlements are specifically the coast, so a 400 m centroid landing just
seaward of a detailed shoreline is a systematic loss in one direction. Dropping those cells
walks every island's dots inland, towards the interior the map is trying not to draw dots on.

**THIS FILE CARRIES THE MAGNITUDE CHECK ON `cv_geo.py`'s JOIN.** COD-PS, the witness there,
is a projection off the 2010 census and is 16% above the 2021 count nationally, so it can
rank the concelhos and cannot size them. Kontur 2023 is modelled from a different lineage
again, and it has to agree with the census about how many people are in each of 22 units,
including the three whose names are ambiguous. A permuted Ribeira Grande would show here.

**THE VINTAGE GAP IS TWO YEARS**, counts 2021 and grid 2023, and it moves dots within a
concelho, never between concelhos.

Usage:
    python sources/cv_grid.py --fetch    one ~700 KB gz from Kontur
    python sources/cv_grid.py            rebuild from data/raw/cv/
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
RAW = os.path.join(ROOT, "data", "raw", "cv")
GEO = os.path.join(ROOT, "data", "geo", "cv")
UNITS = os.path.join(GEO, "cv_concelhos.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "cv.csv")
OUT = os.path.join(GEO, "cv_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CV_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CV_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CV_20231101.gpkg"

EXPECTED_UNITS = 22
CENSUS_POPULATION = 491_233

NATIONAL_TOLERANCE = 0.35
# 22 units and the smallest is 4,743 people, so this band is looking for a mispaired
# concelho and not for a modelling difference.
UNIT_BAND = 2.2
# Cabo Verde spans about 3.5° of longitude, Santo Antão to Boa Vista.
MAX_SPAN_DEG = 8.0

# A hex centroid this far outside a concelho is a coastline-resolution artefact and is
# snapped to the nearest one; anything further is dropped. Metres, in a projected CRS.
# EPSG:32626 is UTM 26N, which covers the whole archipelago.
SNAP_M = 700.0
SNAP_CRS = "EPSG:32626"


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
        raise SystemExit(f"missing {UNITS} — run sources/cv_geo.py first")

    units = gpd.read_file(UNITS).to_crs("EPSG:4326")
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} concelhos, expected {EXPECTED_UNITS}")
    name_of = dict(zip(units["unit"], units["name"]))
    print(f"concelhos: {len(units)}, crs={units.crs}")

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
          f"(concelhos {units.total_bounds[2] - units.total_bounds[0]:.2f}°)")
    if span > MAX_SPAN_DEG:
        raise SystemExit(f"the grid spans {span:.1f}°; something is torn "
                         "[[reference_antimeridian]]")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every concelho: {int(outside.sum()):,} "
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
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of a concelho and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — coastal cells just seaward of a"
              "\n     detailed shoreline. Dropping them would pull every island's dots "
              "inland,\n     which on these islands is uphill onto empty volcano "
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
        raise SystemExit(f"concelhos with no cell: {[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"concelhos whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} concelhos has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2021 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a 2-year gap — check the download")

    # cv.csv's fifteen categories partition the concelho's 15+ population, not its whole
    # population, so the comparison here is against `pop` on the unit layer, which is the
    # census's own Tabela 1 total of every age. sources/cv.py asserts both.
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    if set(df["geo_id"]) != set(units["unit"]):
        raise SystemExit("cv.csv and cv_concelhos.gpkg disagree about the unit set — "
                         "re-run sources/cv_geo.py")
    census = dict(zip(units["unit"], units["pop"]))

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print(f"\n  the five worst-agreeing concelhos each way, census against Kontur, "
          "normalised:")
    print(f"    {'':<32} {'census 2021':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows[:5] + rows[-5:]:
        print(f"    {nm:<32} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if len(worst) > 4:
        raise SystemExit(f"{len(worst)} concelhos outside a factor of {UNIT_BAND:g}, which "
                         "is too many to be the grid's blind spots: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
    if worst:
        share = sum(w[2] for w in worst) / sum(r[2] for r in rows)
        print(f"    {len(rows) - len(worst)} of {len(rows)} inside a factor of "
              f"{UNIT_BAND:g}; the {len(worst)} outside are "
              f"{100 * share:.2f}% of the counted population:")
        for u, nm, c, k, r in worst:
            print(f"      {nm:<32} census {c:>7,} vs Kontur {k:>7,.0f}  ({r:.2f}x)")
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
    if beat > 20 or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach — the pairing in "
                         "cv.py's PCODE is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
