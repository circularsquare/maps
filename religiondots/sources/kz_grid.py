"""Kazakhstan — the placement layer: Kontur 400 m population hexagons, keyed to region.

Writes data/geo/kz/kz_hexes.gpkg.

**17 REGIONS FOR 19.19M PEOPLE OVER 2.7 MILLION KM² — the largest units on this map by area
by a very long way.** Karaganda region alone is 428,000 km², bigger than Germany and Poland
together and larger than every country drawn here except Russia, China, India and Brazil. An
equal-share wash would scatter its dots evenly across the Betpak-Dala desert and the Kazakh
Uplands, where essentially nobody lives; the people are in Karaganda city, Temirtau and a
string of mining towns.

**KAZAKHSTAN IS THE STRONGEST CASE FOR §8.2's GRID ANYWHERE IN THE PROJECT**, because the
country is 2.7M km² and 19M people — a density of 7/km², a quarter of Russia's per-unit case
— and the population is almost entirely in a ring round the edge plus two big cities. The
empty middle is genuinely empty.

**AND HERE IT DOES SOMETHING NO OTHER GRID DOES: it is the check on `kz_geo.py`'s merge.**
That module undoes Kazakhstan's 2022 three-oblast reform by dissolving three pairs of COD
polygons (§8.1). If it dissolved the wrong pair, a region's modelled population would be
compared against the wrong polygon's hexes and the per-region band below would blow open —
East Kazakhstan and Karaganda are not remotely alike in population density. The band is
therefore doing double duty and both jobs are asserted.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two regions.

Usage:
    python sources/kz_grid.py --fetch    one ~11 MB gzipped gpkg from Kontur
    python sources/kz_grid.py            rebuild from data/raw/kz/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kz")
GEO = os.path.join(ROOT, "data", "geo", "kz")
REGIONS = os.path.join(GEO, "kz_regions.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "kz.csv")
OUT = os.path.join(GEO, "kz_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KZ_20231101.gpkg"

EXPECTED_REGIONS = 17
CENSUS_POPULATION = 19_186_015

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). 2023 grid against a 2021 census at
# ~1.1%/yr, so this should read a little above 1.0.
NATIONAL_TOLERANCE = 0.30

# Measured on the real join; see the run output. With only 17 very uneven units the BAND is
# the discriminating null here and the correlation is the weaker one -- Zimbabwe's shape.
UNIT_BAND = 1.8
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
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import math
    import random

    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(REGIONS):
        raise SystemExit(f"missing {REGIONS} -- run sources/kz_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    reg = gpd.read_file(REGIONS)
    if len(reg) != EXPECTED_REGIONS:
        raise SystemExit(f"{REGIONS} has {len(reg)} regions, expected {EXPECTED_REGIONS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(reg.crs)
    hexes = hexes.to_crs(reg.crs)

    joined = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every region: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=reg.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(reg["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"regions with no populated hex: {missing}")
    print(f"  every one of the {EXPECTED_REGIONS} regions has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    census = {r["geo_id"]: int(r["count"]) for _, r in
              df[df["source_category"] == "Всего"].iterrows()}
    name_of = dict(zip(reg["unit"], reg["name"]))
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print("\n  per region, Kontur/census normalised by the national ratio:")
    print(f"    {'':<32} {'census':>10} {'kontur':>11} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm[:32]:<32} {c:>10,} {k:>11,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    print(f"\n  outside the factor-of-{UNIT_BAND:g} band: {len(worst)} of {len(rows)} — "
          f"{[(w[1][:18], round(w[4], 2)) for w in worst]}")
    print("     THIS IS ALSO THE CHECK ON kz_geo.py's THREE-OBLAST MERGE (§8.1): a region")
    print("     dissolved onto the wrong parent would be compared against the wrong grid.")
    if len(worst) > MAX_OUTSIDE_BAND:
        raise SystemExit(
            f"{len(worst)} regions outside a factor of {UNIT_BAND:g}: "
            f"{[(w[1], round(w[4], 2)) for w in worst]} -- either Kontur is wrong about "
            "Kazakhstan or kz_geo.py merged the wrong pair")

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
    print(f"\n  BAND control: a shuffled join puts a median {med} of {len(rows)} regions "
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

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(r[3]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    beat = sum(1 for x in perm if x >= r_true)
    print(f"  CORRELATION control: r = {r_true:.4f}, {beat} of 2,000 shuffles reach it.")
    if beat > 100 or r_true < 0.80:
        raise SystemExit(f"census and Kontur correlate at r={r_true:.4f}, which {beat} of "
                         "2,000 random pairings reach -- the join is not carrying "
                         "information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
