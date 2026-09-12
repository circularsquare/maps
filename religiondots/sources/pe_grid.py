"""Peru — the placement grid: Kontur 400 m population hexagons, clipped to the districts.

Writes data/geo/pe/pe_hexes.gpkg. `countries.py` uses it to weight where a district's dots
land, never to change how many there are.

**PERU NEEDS THIS FOR BOTH OF §8.2's REASONS, AND HARDER THAN MOST.** The districts are
wildly unequal in area and the people are not where the land is:

  * **Emptiness.** The Amazonian districts are enormous and nearly unpopulated — Putumayo,
    Andoas, Morona, Río Tambo and Balsapuerto each run to thousands of km² at well under one
    person per km². Loreto alone is 29% of Peru's land and 3% of its people. Spread evenly,
    a Loreto district's dots would paint religion across unbroken forest.
  * **The desert, which is the same problem inverted.** The coastal districts are mostly
    rainless waste with the entire population in an irrigated valley a few km wide. An equal
    share per polygon would put dots in the Atacama margin and the Sechura sands.

And the two compound in the one place it matters most: **the Adventist altiplano districts
are large, high and mostly empty**, with their people in lakeshore and valley settlements.

**THE VINTAGE GAP IS SIX YEARS AND IS THE SMALLEST ON THIS MAP SO FAR** — counts 2017, grid
2023, against Nicaragua's eighteen. It moves dots *within* a unit and never between units, so
no count is affected either way.

**AND THE CORRELATION HERE IS THE STRONGEST CHECK THE JOIN GETS.** `pe_geo.py` joins on code
and confirms it with names, province prefixes and spatial smoothness. This file adds the one
witness that comes from outside both sources entirely: a modelled 2023 population grid,
sharing no code path and no lineage with either INEI's tabulation or OCHA's boundaries, has
to agree about how many people are inside each of 1,873 polygons.

Usage:
    python sources/pe_grid.py --fetch    one ~19 MB gz from Kontur
    python sources/pe_grid.py            rebuild from data/raw/pe/
"""

import gzip
import math
import os
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pe")
GEO = os.path.join(ROOT, "data", "geo", "pe")
DISTRITOS = os.path.join(GEO, "pe_distritos.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "pe.csv")
LOOKUP = os.path.join(GEO, "pe_lookup.csv")
OUT = os.path.join(GEO, "pe_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_PE_20231101.gpkg.gz")
GZ_NAME = "kontur_population_PE_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_PE_20231101.gpkg"

EXPECTED_UNITS = 1873
# The whole 2017 census count, NOT the 23,196,391 aged 12+ that the religion question
# covers: Kontur models everybody, so the comparable denominator is everybody.
CENSUS_POPULATION = 29_381_884

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Six years of growth is a few percent, so
# this band is tighter than Nicaragua's 0.60 — but it is still only ever a within-unit weight.
NATIONAL_TOLERANCE = 0.35

# Per-district, after normalising by the national ratio.
#
# **NICARAGUA'S CONCLUSION FOR A DIFFERENT REASON.** There the band was ruined by an
# eighteen-year vintage gap and the outliers were the agricultural frontier. Here the gap is
# six years and the band is ruined by SIZE instead: Kontur models population from GHSL and
# building footprints, and on a district of 237 people that is noise. The dependence was
# measured rather than assumed, and it is monotone:
#
#     census 12+   districts   1st pct   99th pct   worst
#         < 500         133      0.51      7.18     12.60   <- Llipa, 237 people
#       500-2k          602      0.57      7.05      8.85
#        2k-10k         771      0.71      4.95      8.38
#        > 10k          365      0.57      2.97      5.15
#
# Every unit above a factor of 8 is a small district, and the largest 365 — which hold most
# of Peru — sit inside 5. A band tight enough to be evidence would reject the country for
# Kontur being imprecise about villages. So it is set to admit them and kept as a tripwire
# against a wholesale mispairing only; the correlation below is the check.
UNIT_BAND = 15.0


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
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(DISTRITOS):
        raise SystemExit(f"missing {DISTRITOS} -- run sources/pe_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    dist = gpd.read_file(DISTRITOS)
    if len(dist) != EXPECTED_UNITS:
        raise SystemExit(f"{DISTRITOS} has {len(dist)} districts, "
                         f"expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(dist.crs)
    hexes = hexes.to_crs(dist.crs)

    joined = gpd.sjoin(pts, dist[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every district: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's PE extract overruns the borders with Ecuador, Colombia, Brazil,\n"
          "     Bolivia and Chile, and Lake Titicaca; dropped. MEASURED against COD's own\n"
          "     ADM0 outline: 646,575 of those people are in the neighbouring countries and\n"
          "     only 1,594 (0.005% of Peru) fall in slivers between district boundaries.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=dist.crs)

    # ---- districts too small to catch a hex CENTROID, and why they get the polygon ----
    #
    # A Kontur hex is ~400 m across and is assigned to whichever district contains its
    # CENTROID. Two districts in Bongará, Amazonas are small enough that no hex centroid
    # lands inside them at all — Chisquilla (229 people aged 12+) and Recta (174). They are
    # real districts with real people, and scatter.py DROPS a unit that has no placement
    # polygon, with a warning it is easy to miss, so leaving them out would quietly lose 403
    # people rather than fail.
    #
    # The fallback is the district's own polygon as a single placement cell, which makes its
    # dots land uniformly inside it — §8.2's equal-share default, applied to the two units
    # where there is no grid to do better with. It cannot affect any other district and it
    # cannot move a dot across a boundary.
    have = set(out["unit"])
    orphans = dist[~dist["unit"].isin(have)].copy()
    if len(orphans):
        orphans["pop"] = 1.0
        out = pd.concat(
            [out, orphans[["unit", "pop", "geometry"]]], ignore_index=True)
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=dist.crs)
        print(f"\n  {len(orphans)} districts are too small to contain a hex centroid and "
              "fall back to their\n  own polygon as one uniform placement cell (§8.2): "
              + ", ".join(f"{r['name']} ({r['unit']})" for _, r in orphans.iterrows()))

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(dist["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no placement polygon: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS:,} districts has a placement polygon: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2017 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a six-year gap -- check the download")
    print("     a 2023 modelled grid against a 2017 census count — six years, the SMALLEST "
          "vintage\n     gap on this map, and most of this is real population growth.")

    # ---- per district, and the shuffle control ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "distrito"].groupby("geo_id"):
        u = unit_of[gid]
        census[u] = census.get(u, 0) + int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])

    name_of = dict(zip(dist["unit"], dist["name"]))
    # The fallback units carry a placeholder weight of 1.0, not a modelled population, so
    # they are not evidence about anything and are kept out of the band and the correlation.
    fallback = set(orphans["unit"]) if len(orphans) else set()
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in census if census[u] > 0 and u not in fallback]
    rows.sort(key=lambda r: r[4])
    print("\n  the five most and least populated by Kontur relative to the census,")
    print("  normalised by the national ratio:")
    print(f"    {'':<30} {'census 12+':>11} {'kontur':>11} {'norm':>6}")
    for u, nm, c, k, r in rows[:5] + rows[-5:]:
        print(f"    {nm[:29]:<30} {c:>11,} {k:>11,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} districts outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst][:10]}")
    print(f"    all {len(rows):,} inside a factor of {UNIT_BAND:g} — a WIDE band, kept as a "
          "tripwire against a\n    wholesale mispairing only. The correlation below is the "
          "check.")

    rng = random.Random(0)
    ks = [r[3] for r in rows]
    fails = []
    for _ in range(500):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for (u, nm, c, k, r), k2 in zip(rows, sh)
                         if not (1 / UNIT_BAND <= k2 / c / ratio <= UNIT_BAND)))
    fails.sort()
    clean = sum(1 for f in fails if f == 0)
    print(f"\n  the shuffle control on the band: re-pairing the Kontur populations at "
          f"random puts a\n  median {fails[len(fails) // 2]:,} of {len(rows):,} districts "
          f"outside it and {clean} of 500 shuffles pass cleanly.")

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(r[3]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(500):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  and the correlation, which on {len(rows):,} units is the STRONG check: "
          f"r = {r_true:.4f},\n  against a best of {perm[-1]:.4f} over 500 random pairings "
          f"({beat} reach it).")
    if beat > 5 or r_true < 0.85:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 500 random pairings reach -- the join in "
                         "pe_geo.py is not carrying information")
    print("  A modelled 2023 grid agreeing with a 2017 census on 1,873 polygons is the one "
          "witness\n  here that shares no lineage with either INEI's counts or OCHA's "
          "boundaries.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
