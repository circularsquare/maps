"""Laos — the placement layer: Kontur 400 m population hexagons, keyed to village.

Writes data/geo/la/la_hexes.gpkg.

**8,499 villages at 763 people each is the finest counting geography of any mainland Asian
country on this map, and it still needs a population grid, because the units are village
CATCHMENTS rather than villages.** `sources/la_geo.py` has the construction: the census
recorded a GPS point per village and CDE grew polygons around those points on a travel-time
surface, so a polygon is the territory whose nearest village is that one and not the
territory anybody lives on. Its people are at the point.

**THE AREA DISTRIBUTION IS WHY THIS IS NOT ACADEMIC.** Median 14.3 km², which is a 3.8 km
square and fine enough that §8.2's trick would apply — but p95 is 93.8 km² and the tail runs
to 1,080 km², and **96 polygons over 200 km² hold 68,769 people across 13.9% of the country**.
Those are not a random sample of Laos. They are the upland districts of Phongsaly, Houaphan,
Xekong and Attapeu, which is exactly where the 31% category lives: an equal-share wash would
take the most interesting cell on the Laos map and spread it evenly over several hundred
square kilometres of forested mountain per village, most of it uninhabited.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two villages.

**KONTUR'S LAOS EXTRACT IS THIN, AND 365 VILLAGES GET NOTHING FROM IT.** 69,696 hexes for the
whole country is about 9,600 km², so the grid covers the built-up 4% of Laos and no more.
8,134 villages (95.7%) come out with at least one populated hex; the other 365 hold 267,465
people, 4.13%, and are concentrated in Vientiane Capital, Houaphan and Phongsaly. **Each of
those gets one placement polygon, its own**, which is an equal-share wash inside that village
and is exactly what every unit on the map would get if no grid existed. It is written into
the layer with `src='polygon'` rather than left implicit, so the two populations can be told
apart later.

**BOTH NULLS DISCRIMINATE, AND THE BAND IS WIDE ON PURPOSE** (§12):

    outside a factor of three   1,072 of 8,134 (13.2%)   shuffled median 2,995
                                                         0 of 200 shuffles under 1,300
    r on log populations        0.5906                   best of 2,000 shuffles 0.0450

**Both of those numbers are far worse than Nepal's (10 of 753, r = 0.9055) and the reason is
the grain rather than the join.** A unit here holds 763 people, so a village with one Kontur
hex reading `1` against a census `1,205` is a single 400 m cell being asked to carry a whole
settlement's level — and that is 40 rows of the tail on its own. **The level being wrong
there does not move a dot**: a village with one hex puts all its dots in that hex whatever
number the hex carries, and Kontur finding exactly one built-up cell in a mountain catchment
is a statement about WHERE the village is, which is the only thing the weight is used for.
The median ratio is 0.97 and the quartiles 0.63 to 1.35, which is what says the pairing is
real.

**A CATCHMENT MODEL AND A BUILT-UP MODEL DISAGREEING IS THE EXPECTED RESULT, NOT A FAULT.**
Kontur is built from GHSL, HRSL and building footprints and knows where the houses are;
the polygons know only which village is nearest. Where a large catchment contains a second
settlement Kontur has people the census attributes elsewhere, and the per-unit ratio moves.
**None of it changes a count** (§8.2): the grid is a within-unit weight, every village
receives exactly LSB's own figure, and only the shape inside the polygon survives.

Usage:
    python sources/la_grid.py --fetch    one ~6 MB gzipped gpkg from Kontur
    python sources/la_grid.py            rebuild from data/raw/la/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "la")
GEO = os.path.join(ROOT, "data", "geo", "la")
UNITS = os.path.join(GEO, "la_units.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "la.csv")
OUT = os.path.join(GEO, "la_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_LA_20231101.gpkg.gz")
GZ_NAME = "kontur_population_LA_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_LA_20231101.gpkg"

EXPECTED_UNITS = 8_499
# The DRAWN population: the village populations in la.csv, not the 6,492,228 the census
# enumerated. Kontur is a within-unit weight and its level divides out, so this is only the
# gross-failure check on the download.
CENSUS_POPULATION = 6_481_482

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). A 2023 grid against a 2015 census with
# Laos growing at ~1.5%/yr should read well above 1.0.
NATIONAL_TOLERANCE = 0.40

# Measured, not guessed: the real join puts 1,072 of 8,134 outside a factor of three and a
# shuffled one puts a median 2,995, with 0 of 200 shuffles reaching 1,300. The cap is set
# above the real figure with room for a Kontur re-release and is still less than half the
# null, which is the test §12 asks for. The docstring has why the tail is wide here and why
# it does not move a dot.
UNIT_BAND = 3.0
MAX_OUTSIDE_BAND = 1_300


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
        raise SystemExit(f"missing {UNITS} -- run sources/la_geo.py first")

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
    print(f"\n  hexes whose centroid is outside every village polygon: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     The polygons are an accessibility model covering 97.4% of Laos "
          "(sources/la_geo.py),\n     so the border strip and the large water bodies drop "
          "out here.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    out["src"] = "kontur"
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    covered = set(per.index[per["sum"] > 0])
    print(f"  villages with at least one populated hex: {len(covered):,} of "
          f"{EXPECTED_UNITS:,} ({len(covered) / EXPECTED_UNITS:.1%}), "
          f"{per['size'].min():,}–{per['size'].max():,} hexes each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    # ---- per unit, and BOTH nulls (§12) ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    tp = df[df["source_category"] == "Total Population"]
    census = dict(zip(tp["geo_id"], tp["count"].astype(int)))
    if len(census) != EXPECTED_UNITS:
        raise SystemExit(f"{len(census)} villages in la.csv, expected {EXPECTED_UNITS}")

    name_of = dict(zip(units["unit"], units["name"]))
    dist_of = dict(zip(units["unit"], units["district"]))
    prov_of = dict(zip(units["unit"], units["province"]))

    # ---- THE FALLBACK, and it is the honest one: a village Kontur has nothing for gets
    # ---- its own polygon as its single placement shape, which is an equal-share wash
    # ---- inside that village and nothing more. See the module docstring.
    bare = sorted(set(units["unit"]) - covered)
    bare_pop = sum(census.get(u, 0) for u in bare)
    print(f"\n  villages with NO populated Kontur hex: {len(bare):,} "
          f"({bare_pop:,} people, {100.0 * bare_pop / CENSUS_POPULATION:.2f}%)")
    if bare:
        sizes = sorted(census.get(u, 0) for u in bare)
        by_prov = {}
        for u in bare:
            by_prov[prov_of[u]] = by_prov.get(prov_of[u], 0) + 1
        print(f"    census population of those villages: min {sizes[0]}, "
              f"median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
        print("    most affected provinces: " + ", ".join(
            f"{p} {n}" for p, n in sorted(by_prov.items(), key=lambda kv: -kv[1])[:6]))
        fb = units[units["unit"].isin(bare)][["unit", "geometry"]].copy()
        fb["pop"] = fb["unit"].map(census).astype(float)
        fb["src"] = "polygon"
        out = gpd.GeoDataFrame(pd.concat([out, fb[["unit", "pop", "src", "geometry"]]],
                                         ignore_index=True), crs=units.crs)
        print("    Each gets ONE placement polygon, its own, so its dots wash evenly over "
              "the\n    village catchment. That is what every unit would get without a "
              "grid at all.")

    still = sorted(set(units["unit"]) - set(out.loc[out["pop"] > 0, "unit"]))
    if still:
        raise SystemExit(f"{len(still)} villages still have nowhere to place a dot: "
                         f"{still[:6]}")
    print(f"  every one of the {EXPECTED_UNITS:,} villages now has a placement polygon")

    # The controls below run on the Kontur-covered villages only: a fallback polygon
    # carries the census figure by construction, so including it would compare a number
    # with itself and inflate both nulls.
    rows = [(u, name_of[u], dist_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in census if census[u] > 0 and u in covered]
    rows.sort(key=lambda r: r[5])
    print(f"\n  per village, Kontur/census normalised by the national ratio — the six at "
          f"each end of {len(rows):,}:")
    print(f"    {'':<24} {'':<16} {'census':>8} {'kontur':>9} {'norm':>6}")
    for r in rows[:6] + rows[-6:]:
        print(f"    {str(r[1])[:24]:<24} {str(r[2])[:16]:<16} {r[3]:>8,} "
              f"{r[4]:>9,.0f} {r[5]:>6.2f}")
    qs = sorted(r[5] for r in rows)
    print(f"\n  quartiles of the ratio: p05={qs[len(qs) // 20]:.2f} "
          f"p25={qs[len(qs) // 4]:.2f} median={qs[len(qs) // 2]:.2f} "
          f"p75={qs[3 * len(qs) // 4]:.2f} p95={qs[19 * len(qs) // 20]:.2f}")

    worst = [r for r in rows if r[5] < 1 / UNIT_BAND or r[5] > UNIT_BAND]
    print(f"  outside the factor-of-{UNIT_BAND:g} band: {len(worst)} of {len(rows)} "
          f"({len(worst) / len(rows):.1%})")
    if len(worst) > MAX_OUTSIDE_BAND:
        raise SystemExit(
            f"{len(worst)} villages outside a factor of {UNIT_BAND:g}, more than the "
            f"{MAX_OUTSIDE_BAND} allowed: {[(w[1], round(w[5], 2)) for w in worst[:8]]}")

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
          f"{med:,} of\n  {len(rows):,} villages outside the band, and {band_pass} of 200 "
          f"shuffles come in at or under\n  the {MAX_OUTSIDE_BAND:,} the real join is "
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
    if beat > 100 or r_true < 0.50:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "la_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
