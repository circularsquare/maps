"""Nepal — the placement layer: Kontur 400 m population hexagons, keyed to local level.

Writes data/geo/np/np_hexes.gpkg.

**753 local levels is fine geography and Nepal is still the country on this map that needs a
population grid most**, which sounds like a contradiction and is not. The units are fine *in
people* — 38,400 each — and wild *in area*, because Nepal's local-government map was drawn to
equalise population across terrain that ranges from the Terai plain to the 8,000-metre
Himalaya:

    Chandragiri (Kathmandu valley)      ~50 km²   ~90,000 people
    Namkha (Humla, on the Tibet border) 2,290 km²  ~2,500 people

That is a factor of 45 in area between two units of comparable rank. An equal-share wash
would put a rural municipality's dots evenly across glaciers, ridge lines and the floor of the
Kali Gandaki alike, and in the high north it would put most of them where nobody has ever
lived. Kontur knows the valleys. Spec §8.2's trick — fine units make a population layer
unnecessary — is about units that are fine in AREA, and Nepal's are not.

**IT MATTERS FOR WHAT THE WASH WOULD SAY, not only for tidiness.** The trans-Himalayan north
is where `Bon` is (Mustang, Dolpa, Humla, Manang) and where the Buddhist share runs above 80%,
and those units are the emptiest and largest in the country. A wash would spread Nepal's only
Bon dots evenly over several thousand square kilometres of rock and ice, and under-draw the
Kathmandu valley and the Terai where 80% of Nepalis live. The same argument as Cambodia's
Eastern Highlands, with an order of magnitude more relief.

**THE NATIONAL PARKS ARE ALREADY OUT** — `np_geo.py` drops COD's 22 protected-area polygons,
which sit outside the local levels rather than inside them, so no hex is keyed to one and
§8.2c's problem does not arise. What is left of it is handled by Kontur itself: a population
grid has no hexes on the Terai's rivers or on the glaciers.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two local levels.

**BOTH NULLS DISCRIMINATE, AND HARD** (§12 — Benin measured one, Zimbabwe the other, Cambodia
both). 753 units is far more than any previous customer here and it buys a very sharp check:
r = 0.9055 on log populations against a best of 0.1350 over 2,000 shuffles, and 10 units
outside a factor of three against a shuffled median of 224. The ratio's median is exactly 1.00
and its quartiles are 0.84–1.20.

**THE TEN OUTLIERS ARE TWO DIFFERENT THINGS AND ONLY ONE OF THEM IS A BOUNDARY EFFECT**, which
is worth separating because the fix is different and neither is a fault in the join.

  * *A town smeared into its hinterland.* Rohini Gaunpalika reads 4.09 and the municipality it
    wraps, Siddharthanagar, reads 0.26 — the same people, on the other side of a line. **Pool
    Rohini with its neighbours and it falls to 1.31**, which is what a boundary effect looks
    like. Butwal (0.26), Dharan (0.35), Bhimdatta (0.32), Birendranagar (0.33) and Triyuga
    (0.32) are the same shape from the town's side: Kontur's built-up model puts a Nepali
    town's people further out than the municipal boundary does.
  * *A block of the Parsa Terai that Kontur simply over-models.* Kalikamai 8.14, Pakaha
    Mainpur 3.20, Pokhariya 2.43, and it does **not** pool away — Kalikamai with all five
    neighbours is still 2.88, Pakaha Mainpur still 3.18. That is a contiguous area near the
    Indian border where the grid is wrong about the level, and it is Cambodia's Pailin again.

**NEITHER CHANGES A COUNT** (§8.2). The grid is a within-unit weight: Kalikamai receives
exactly NSO's 23,480 people whatever Kontur thinks, and only the shape inside the unit
survives. What to carry is that dots in those ten units sit on this map's least trustworthy
placement surface in Nepal.

Usage:
    python sources/np_grid.py --fetch    one ~30 MB gzipped gpkg from Kontur
    python sources/np_grid.py            rebuild from data/raw/np/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "np")
GEO = os.path.join(ROOT, "data", "geo", "np")
UNITS = os.path.join(GEO, "np_units.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "np.csv")
LOOKUP = os.path.join(GEO, "np_lookup.csv")
OUT = os.path.join(GEO, "np_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_NP_20231101.gpkg.gz")
GZ_NAME = "kontur_population_NP_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_NP_20231101.gpkg"

EXPECTED_UNITS = 753
# The DRAWN population — the 753 local levels, without the district-level institutional rows
# (sources/np.py). Kontur is a within-unit weight and its level divides out, so this is only
# the gross-failure check on the download.
CENSUS_POPULATION = 28_925_480

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). 2023 grid against a 2021 census at
# ~0.9%/yr, so this should read a little above 1.0.
NATIONAL_TOLERANCE = 0.30

# Measured, not guessed. **Ten of the 753 sit outside a factor of three**, and both nulls
# below say that is a real join with ten bad cells rather than a loose band: a shuffled join
# puts a median 224 outside, and the ratio's quartiles are 0.84–1.20 with a median of exactly
# 1.00. Twenty is allowed rather than ten so a Kontur re-release does not break the build on
# an eleventh, and it is nowhere near the 224 that would make the check decoration (§12 —
# Benin's warning).
UNIT_BAND = 3.0
MAX_OUTSIDE_BAND = 20


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
    # sources.md §5a: a 200 is not a download, and a gunzip that runs is not a GeoPackage.
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
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/np_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every local level: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's NP extract overruns into India and Tibet, and the 22 national "
          "parks are\n     carved out of the local levels (np_geo.py), so a little of "
          "each is dropped here.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"local levels with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"local levels whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} local levels has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    # ---- per unit, and BOTH nulls (§12) ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "local"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total Population", "count"].iloc[0])
    if len(census) != EXPECTED_UNITS:
        raise SystemExit(f"{len(census)} census local levels in np.csv, "
                         f"expected {EXPECTED_UNITS}")

    name_of = dict(zip(units["unit"], units["name"]))
    dist_of = dict(zip(units["unit"], units["district"]))
    rows = [(u, name_of[u], dist_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[5])
    print("\n  per local level, Kontur/census normalised by the national ratio — the ten "
          "at each end:")
    print(f"    {'':<26} {'':<16} {'census':>9} {'kontur':>9} {'norm':>6}")
    for r in rows[:10] + rows[-10:]:
        print(f"    {r[1][:26]:<26} {r[2][:16]:<16} {r[3]:>9,} {r[4]:>9,.0f} {r[5]:>6.2f}")
    qs = sorted(r[5] for r in rows)
    print(f"\n  quartiles of the ratio: p05={qs[len(qs) // 20]:.2f} "
          f"p25={qs[len(qs) // 4]:.2f} median={qs[len(qs) // 2]:.2f} "
          f"p75={qs[3 * len(qs) // 4]:.2f} p95={qs[19 * len(qs) // 20]:.2f}")

    worst = [r for r in rows if r[5] < 1 / UNIT_BAND or r[5] > UNIT_BAND]
    print(f"  outside the factor-of-{UNIT_BAND:g} band: {len(worst)} of {len(rows)} "
          f"({len(worst) / len(rows):.1%})")
    if len(worst) > MAX_OUTSIDE_BAND:
        raise SystemExit(
            f"{len(worst)} local levels outside a factor of {UNIT_BAND:g}, more than the "
            f"{MAX_OUTSIDE_BAND} allowed: {[(w[1], round(w[5], 2)) for w in worst[:10]]}")

    rng = random.Random(0)
    ks = [r[4] for r in rows]
    cs = [r[3] for r in rows]
    fails = []
    for _ in range(200):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for c, k2 in zip(cs, sh)
                         if not (1 / UNIT_BAND <= k2 / c / ratio <= UNIT_BAND)))
    med = sorted(fails)[len(fails) // 2]
    band_pass = sum(1 for f in fails if f <= MAX_OUTSIDE_BAND)
    print(f"\n  BAND control: shuffling the populations across the polygons puts a median "
          f"{med} of\n  {len(rows)} local levels outside the band, and {band_pass} of 200 "
          f"shuffles come in at or under\n  the {MAX_OUTSIDE_BAND} the real join is "
          "allowed.")
    if med <= MAX_OUTSIDE_BAND:
        raise SystemExit(
            f"a shuffled join puts a median {med} units outside the band, which the real "
            f"join is allowed ({MAX_OUTSIDE_BAND}) -- the band no longer discriminates "
            "and must not be reported as a check")

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    lc = [math.log(r[3]) for r in rows]
    lk = [math.log(r[4]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"  CORRELATION control: r = {r_true:.4f} against a best of {perm[-1]:.4f} over "
          f"2,000 shuffles\n  ({beat} reach it).")
    if beat > 100 or r_true < 0.70:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "np_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
