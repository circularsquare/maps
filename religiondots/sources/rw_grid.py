"""Rwanda — the placement layer: Kontur 400 m hexagons, RE-LEVELLED ONTO THE CENSUS SECTORS.

Writes data/geo/rw/rw_hexes.gpkg.

**RWANDA CAN DO SOMETHING MOST COUNTRIES HERE CANNOT.** The usual placement layer is Kontur
alone: a modelled grid, used as a within-unit weight because nothing finer than the counting
unit was counted. Rwanda counted finer. Religion stops at the 30 districts, but NISR's
`Population_2002_2022` layer carries **the census's own population for all 416 sectors**, so
the weight inside a district does not have to be modelled at all — only the weight inside a
sector does.

So the hexes are scaled per sector to make each sector's hexes sum to its census count. What
survives from Kontur is the shape *within* a sector, roughly 15 km2 and 32,000 people, and
what comes from the census is everything coarser than that. A district's dots then land in
its sectors in exactly the proportions the census measured.

**THE CHECK IS RUN ON THE RATIOS BEFORE THE SCALING, not after** — afterwards every ratio is
1.000 by construction and the file would prove nothing. Kontur is built from building
footprints and settlement models and knows nothing about RPHC-5, so the raw agreement
between the two is a quantity the join does not determine, and it is measured against a
permutation control rather than asserted ([[reference_check_needs_power]]).

**416 SECTORS IS WHERE THIS PAYS.** Rwanda is 24,668 km2 of land at 537 people per km2, the
densest country on the African mainland, and its density is genuinely uneven: Kigali's
sectors run past 20,000/km2 while Akagera and the Nyungwe belt hold sectors under 100. An
even spread across a district would put dots in the national parks.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two sectors.

Usage:
    python sources/rw_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/rw_grid.py            rebuild from data/raw/rw/
"""

import gzip
import math
import os
import random
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "rw")
GEO = os.path.join(ROOT, "data", "geo", "rw")
SECTORS = os.path.join(GEO, "rw_sectors.gpkg")
DISTRICTS = os.path.join(GEO, "rw_districts.gpkg")
LOOKUP = os.path.join(GEO, "rw_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "rw.csv")
OUT = os.path.join(GEO, "rw_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_RW_20231101.gpkg.gz")
GZ_NAME = "kontur_population_RW_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_RW_20231101.gpkg"

EXPECTED_SECTORS = 416
EXPECTED_DISTRICTS = 30
CENSUS_POPULATION = 13_246_394

# Kontur is modelled and must not be asserted equal to the census (§12, North Macedonia).
# Its vintage is 2023 against a 2022 census, one year apart, so the national ratio should
# read close to 1.0 the way Zimbabwe's did.
NATIONAL_TOLERANCE = 0.30

# The per-SECTOR band. 416 units of ~32,000 people is a much finer comparison than the
# per-province ones elsewhere in this directory and a wider band is expected: a modelled
# grid disagrees with a count more, not less, as the unit shrinks. Measured, Kontur and
# RPHC-5 agree across all 416 sectors within 0.45x-2.07x, median 0.99, which is remarkably
# tight for this grain; the band below is set just outside that and nothing is allowed
# through it. It is not what carries the check -- the correlation is (see below).
SECTOR_BAND = 2.5
SECTOR_OUTLIER_BUDGET = 0        # of 416


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    # §5a: a 200 is not a download, and a gunzip that runs is not a GeoPackage.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
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
    for p in (gpkg, SECTORS, DISTRICTS):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run rw_geo.py --fetch and rw_grid.py --fetch")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    sec = gpd.read_file(SECTORS)
    if len(sec) != EXPECTED_SECTORS:
        raise SystemExit(f"{SECTORS} has {len(sec)} sectors, "
                         f"expected {EXPECTED_SECTORS}")
    if int(sec["pop"].sum()) != CENSUS_POPULATION:
        raise SystemExit(f"the sectors sum to {int(sec['pop'].sum()):,}, expected "
                         f"{CENSUS_POPULATION:,}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in,
    # then reproject the POINTS -- reprojecting first and taking the centroid after moves
    # it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(sec.crs)
    hexes = hexes.to_crs(sec.crs)

    sec = sec.reset_index(drop=True)
    sec["sid"] = sec.index
    joined = gpd.sjoin(pts, sec[["sid", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["sid"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every sector: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's RW extract overruns into the DRC, Uganda, Tanzania and Burundi "
          "and\n     across Lake Kivu, which the sector layer does not tile; dropped.")

    keep = ~outside
    sid = joined.loc[keep, "sid"].astype(int).to_numpy()
    kpop = pts.loc[keep, popcol].to_numpy(dtype=float)
    geom = hexes.geometry[keep.to_numpy()].to_numpy()

    grid = pd.DataFrame({"sid": sid, "kontur": kpop})
    per = grid.groupby("sid")["kontur"].agg(["size", "sum"]).reindex(sec["sid"])
    empty = sec.loc[per["sum"].fillna(0) <= 0, ["sector", "district"]]
    if len(empty):
        raise SystemExit(f"{len(empty)} sectors have no populated Kontur hex, so there is "
                         f"nothing to re-level onto them: {empty.to_dict('records')[:6]}")
    thin = int((per["size"] < 3).sum())
    print(f"  every one of the {EXPECTED_SECTORS} sectors has populated hexes: "
          f"{int(per['size'].min()):,}-{int(per['size'].max()):,} each, "
          f"{thin} with fewer than three")
    print("     A Kontur cell is H3 resolution 8, about 0.74 km2, and the sectors with one "
          "or two of\n     them are the small dense ones in Kigali; their people are "
          "placed inside that cell,\n     which is finer than anything the district grain "
          "could resolve anyway.")

    tot = float(per["sum"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} -- ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    # ---- the check, on the RAW ratios, before anything is scaled ----
    census = sec["pop"].to_numpy(dtype=float)
    kontur = per["sum"].to_numpy(dtype=float)
    norm = kontur / census / ratio
    out_of_band = [(sec["district"][i], sec["sector"][i], round(norm[i], 2))
                   for i in range(len(sec))
                   if norm[i] < 1 / SECTOR_BAND or norm[i] > SECTOR_BAND]
    print(f"\n  per SECTOR, Kontur/census normalised by the national ratio: "
          f"{norm.min():.2f}-{norm.max():.2f},\n  median {sorted(norm)[len(norm) // 2]:.2f}, "
          f"{len(out_of_band)} outside a factor of {SECTOR_BAND:g}")
    for d, s, v in sorted(out_of_band, key=lambda t: t[2])[:8]:
        print(f"      {d}/{s}: {v}")
    if len(out_of_band) > SECTOR_OUTLIER_BUDGET:
        raise SystemExit(f"{len(out_of_band)} sectors outside a factor of {SECTOR_BAND:g}, "
                         f"budget {SECTOR_OUTLIER_BUDGET} -- the sector join is suspect")

    lc = [math.log(v) for v in census]
    lk = [math.log(v) for v in kontur]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  and the control: r = {r_true:.4f} on {len(sec)} sectors, against a best of "
          f"{perm[-1]:.4f}\n  over 2,000 random pairings ({beat} reach it). Unlike "
          "Zimbabwe's ten provinces, 416 units\n  of similar size do NOT correlate by "
          "luck, so this is the discriminating check here and\n  the band is the loose one "
          "(sources/zw_grid.py's rule, with the answer the other way).")
    if beat > 20 or r_true < 0.70:
        raise SystemExit(f"census and Kontur correlate at r={r_true:.4f}, which {beat} of "
                         "2,000 random pairings reach -- the sector join is not carrying "
                         "information")

    # ---- the re-levelling ----
    scale = (census / kontur)[sid]
    pop = kpop * scale
    unit = sec["unit"].to_numpy()[sid]

    out = gpd.GeoDataFrame({"unit": unit, "pop": pop}, geometry=geom, crs=sec.crs)
    by_unit = out.groupby("unit")["pop"].sum()
    dis = gpd.read_file(DISTRICTS)
    if len(dis) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{DISTRICTS} has {len(dis)} districts")
    bad = [(u, float(by_unit.get(u, 0.0)), int(p))
           for u, p in zip(dis["unit"], dis["pop"])
           if abs(float(by_unit.get(u, 0.0)) - p) > 0.5]
    if bad:
        raise SystemExit(f"after re-levelling, {len(bad)} districts do not sum to their "
                         f"census count: {bad[:5]}")
    print(f"\n  after re-levelling, every one of the {EXPECTED_DISTRICTS} districts' hexes "
          f"sums to its\n  census population exactly, and every one of the "
          f"{EXPECTED_SECTORS} sectors' does too.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
