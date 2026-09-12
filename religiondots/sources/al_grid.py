"""Albania — the placement layer: Kontur 400 m population hexagons, keyed to qark.

Writes data/geo/al/al_hexes.gpkg.

**TWELVE UNITS FOR A COUNTRY THIS SHAPE IS EXACTLY §8.2's CASE.** Tirana qark is 758,513
people on 1,652 km² while Gjirokastër is 60,013 on 2,876 km², a density ratio of twenty-two
to one, and Albania's population moved into the coastal plain and out of the southern
mountains across the whole post-1990 period. Spreading a qark's dots evenly over its own area
would put Tirana's Bektashis in the Skrapar highlands and empty the city.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two qarqe.

**HEXES THAT LAND OUTSIDE EVERY QARK ARE SNAPPED, NOT DROPPED**
([[reference_archipelago_grid_snap]]). Kontur's H3 cells straddle the Adriatic coastline and
the Greek, Macedonian, Kosovan and Montenegrin borders, where INSTAT draws the line slightly
differently; dropping them pulls dots inland off a coast that holds Durrës and Vlorë.

Usage:
    python sources/al_grid.py --fetch    one ~5 MB gzipped gpkg from Kontur
    python sources/al_grid.py            rebuild from data/raw/al/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "al")
GEO = os.path.join(ROOT, "data", "geo", "al")
PREFECTURES = os.path.join(GEO, "al_prefectures.gpkg")
OUT = os.path.join(GEO, "al_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AL_20231101.gpkg.gz")
GZ_NAME = "kontur_population_AL_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_AL_20231101.gpkg"

EXPECTED_QARQE = 12

# The 2023 census's RESIDENT population. Albania's civil register carries roughly 2.76 million
# because it never removes emigrants, so a grid modelled partly off register-derived rasters
# is expected to read well ABOVE the census here rather than near it; that is what the
# tolerance is wide for, and the per-qark band below is the check that actually bites.
CENSUS_POPULATION = 2_402_113
NATIONAL_TOLERANCE = 0.45

UNIT_BAND = 2.0
MAX_OUTSIDE_BAND = 2


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 2_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=3600, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import math
    import random

    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(PREFECTURES):
        raise SystemExit(f"missing {PREFECTURES} -- run sources/al_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    reg = gpd.read_file(PREFECTURES)
    if len(reg) != EXPECTED_QARQE:
        raise SystemExit(f"{PREFECTURES} has {len(reg)} qarqe, expected {EXPECTED_QARQE}")

    # Centroids in Kontur's own projected CRS, then the POINTS are reprojected. Taking them
    # after the conversion to 4326 computes a centroid in degrees and shifts a hex poleward.
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs(reg.crs)
    hexes = hexes.to_crs(reg.crs)

    joined = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls in no qark: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / pts[popcol].sum():.3f}%)")
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside, ["geometry"]],
                                 reg[["unit", "geometry"]], how="left")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        print(f"  snapped all {int(outside.sum()):,} to their nearest qark; none dropped")
    if joined["unit"].isna().any():
        raise SystemExit("hexes still unassigned after the snap")

    out = gpd.GeoDataFrame(
        {"unit": joined["unit"].to_numpy(),
         "pop": pts[popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry.to_numpy(), crs=reg.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    empty = sorted(set(reg["unit"]) - set(per.index))
    if empty:
        raise SystemExit(f"qarqe with no hex at all: {empty}")
    print(f"  every one of the {EXPECTED_QARQE} qarqe has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2023 census's resident {CENSUS_POPULATION:,} "
          f"-- ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    name_of = dict(zip(reg["unit"], reg["name"]))
    cen = dict(zip(reg["unit"], reg["pop_2023"]))
    rows = []
    for u in sorted(reg["unit"]):
        c = float(cen[u])
        k = float(per.loc[u, "sum"])
        rows.append((u, name_of[u], c, k, k / c / ratio))
    rows.sort(key=lambda r: r[4])
    print("\n  per qark, Kontur/census normalised by the national ratio:")
    for u, nm, c, k, r in rows:
        print(f"    {u} {nm[:16]:<16} {c:>10,.0f} {k:>11,.0f} {r:>6.2f}")

    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    print(f"\n  outside the factor-of-{UNIT_BAND:g} band: {len(worst)} of {len(rows)} -- "
          f"{[(w[1][:16], round(w[4], 2)) for w in worst]}")
    if len(worst) > MAX_OUTSIDE_BAND:
        raise SystemExit(f"{len(worst)} qarqe outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")

    # [[reference_check_needs_power]]: the band means nothing unless a WRONG pairing would
    # fail it. Shuffle the Kontur totals against the census ones and count how many land
    # outside; if the shuffle passes too, this is not evidence and must not be reported as it.
    rng = random.Random(0)
    ks = [r[3] for r in rows]
    cs = [r[2] for r in rows]
    fails = []
    for _ in range(2000):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for c, k2 in zip(cs, sh)
                         if not (1 / UNIT_BAND <= k2 / c / ratio <= UNIT_BAND)))
    med = sorted(fails)[len(fails) // 2]
    print(f"\n  BAND control: a shuffled pairing puts a median {med} of {len(rows)} qarqe "
          f"outside the band,\n  against the {len(worst)} the real one has.")
    if med <= MAX_OUTSIDE_BAND:
        raise SystemExit(f"a shuffled pairing puts a median {med} outside the band, which "
                         f"the real one is allowed ({MAX_OUTSIDE_BAND}) -- the band no "
                         "longer discriminates and must not be reported as a check")

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    r = pearson([math.log(x[2]) for x in rows], [math.log(x[3]) for x in rows])
    print(f"  log-log correlation of Kontur against the census, over {len(rows)} qarqe: "
          f"r = {r:.3f}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="hexes")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
