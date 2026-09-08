"""Mongolia — the placement layer: Kontur 400 m population hexagons, keyed to aimag.

Writes data/geo/mn/mn_hexes.gpkg.

**MONGOLIA IS THE STRONGEST CASE FOR §8.2's GRID ON THIS MAP, ahead of Kazakhstan.** 1.56
million km² and 3.2 million people is a density of **2.1 per km²**, a third of Kazakhstan's
and a seventh of Kenya's, and the distribution is not merely uneven but close to degenerate:
Ulaanbaatar holds 46% of the country on 0.3% of its area, while Ömnögovi is 165,000 km² —
larger than Bangladesh, larger than Greece and Portugal together — holding about 70,000
people. An equal share of dots per polygon would wash the entire Gobi in one colour and put
half of Mongolia's Buddhists in an empty desert.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two aimags.

**THE 270 HEXES THAT LAND OUTSIDE EVERY AIMAG ARE SNAPPED, NOT DROPPED**
(`[[reference_archipelago_grid_snap]]`). They are 0.15% of the grid's population and they sit
on the Russian and Chinese borders, where Kontur's H3 cells straddle a line that COD draws
slightly differently. Dropping them would pull dots inward off the frontier, which on a
country whose Kazakh population lives in the far west is exactly the wrong direction to be
wrong in. Each is assigned to the nearest aimag instead.

Usage:
    python sources/mn_grid.py --fetch    one ~10 MB gzipped gpkg from Kontur
    python sources/mn_grid.py            rebuild from data/raw/mn/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mn")
GEO = os.path.join(ROOT, "data", "geo", "mn")
AIMAGS_GPKG = os.path.join(GEO, "mn_aimags.gpkg")
OUT = os.path.join(GEO, "mn_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MN_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MN_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MN_20231101.gpkg"

EXPECTED_AIMAGS = 22

# The 2020 census's RESIDENT population, which is the universe the aimag volumes and the
# national report's appendix table 1.1 both use. Not the 3,296,866 usually quoted, which
# includes the 99,846 citizens enumerated as living abroad.
CENSUS_POPULATION = 3_197_020

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must
# not be asserted equal to one (§12, North Macedonia). A 2023 grid against a 2020 census at
# roughly 1.8%/yr should read a little above 1.0.
NATIONAL_TOLERANCE = 0.30

# Measured on the real join. Ulaanbaatar's pull makes Kontur over-urban here, so the band is
# generous; the permutation control below is what says whether it discriminates at all.
UNIT_BAND = 2.0
MAX_OUTSIDE_BAND = 3


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
        r = requests.get(GZ_URL, timeout=3600, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import math
    import random

    import geopandas as gpd
    import pandas as pd

    sys.path.insert(0, HERE)
    import mn as M

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(AIMAGS_GPKG):
        raise SystemExit(f"missing {AIMAGS_GPKG} -- run sources/mn_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    reg = gpd.read_file(AIMAGS_GPKG)
    if len(reg) != EXPECTED_AIMAGS:
        raise SystemExit(f"{AIMAGS_GPKG} has {len(reg)} aimags, "
                         f"expected {EXPECTED_AIMAGS}")

    # Centroids are taken in Kontur's own projected CRS and only then reprojected: taking
    # them after the conversion to EPSG:4326 computes a centroid in degrees, which geopandas
    # warns about and which shifts a hex slightly poleward.
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs(reg.crs)
    hexes = hexes.to_crs(reg.crs)

    joined = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls in no aimag: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / pts[popcol].sum():.3f}%)")
    if outside.any():
        # SNAP, DO NOT DROP. These sit on the Russian and Chinese frontiers.
        near = gpd.sjoin_nearest(pts.loc[outside, ["geometry"]],
                                 reg[["unit", "geometry"]], how="left")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        print(f"  snapped all {int(outside.sum()):,} to their nearest aimag "
              f"(§ archipelago grid snap); none dropped")
    if joined["unit"].isna().any():
        raise SystemExit("hexes still unassigned after the snap")

    out = gpd.GeoDataFrame(
        {"unit": joined["unit"].to_numpy(),
         "pop": pts[popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry.to_numpy(), crs=reg.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    empty = sorted(set(reg["unit"]) - set(per.index))
    if empty:
        raise SystemExit(f"aimags with no hex at all: {empty}")
    print(f"  every one of the {EXPECTED_AIMAGS} aimags has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census resident {CENSUS_POPULATION:,} "
          f"-- ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    # The per-aimag comparison is against the census's own resident population, read from
    # the national report's appendix table 1.1 by sources/mn.py.
    pop = M.read_population()
    name_of = dict(zip(reg["unit"], reg["name_en"]))
    rows = []
    for u in reg["unit"]:
        nsoname = next((n for p, (n, _) in M.AIMAGS.items() if p == u), None) \
            or M.MISSING.get(u)
        c = pop[nsoname][0]
        k = float(per.loc[u, "sum"])
        rows.append((u, name_of[u], c, k, k / c / ratio))
    rows.sort(key=lambda r: r[4])
    print("\n  per aimag, Kontur/census normalised by the national ratio:")
    for u, nm, c, k, r in rows:
        print(f"    {nm[:20]:<20} {c:>10,} {k:>11,.0f} {r:>6.2f}")

    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    print(f"\n  outside the factor-of-{UNIT_BAND:g} band: {len(worst)} of {len(rows)} -- "
          f"{[(w[1][:16], round(w[4], 2)) for w in worst]}")
    if len(worst) > MAX_OUTSIDE_BAND:
        raise SystemExit(f"{len(worst)} aimags outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")

    # [[reference_check_needs_power]]: a band only means something if a WRONG join would
    # fail it. Shuffle the Kontur totals against the census ones and count how many land
    # outside; if the shuffled join passes too, the band is not evidence and must not be
    # reported as though it were.
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
    print(f"\n  BAND control: a shuffled join puts a median {med} of {len(rows)} aimags "
          f"outside the band,\n  against the {len(worst)} the real join has.")
    if med <= MAX_OUTSIDE_BAND:
        raise SystemExit(f"a shuffled join puts a median {med} outside the band, which the "
                         f"real join is allowed ({MAX_OUTSIDE_BAND}) -- the band no longer "
                         "discriminates and must not be reported as a check")

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    r = pearson([math.log(x[2]) for x in rows], [math.log(x[3]) for x in rows])
    print(f"  log-log correlation of Kontur against the census, over {len(rows)} aimags: "
          f"r = {r:.3f}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="hexes")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
