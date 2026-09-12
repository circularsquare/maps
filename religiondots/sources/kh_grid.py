"""Cambodia — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/kh/kh_hexes.gpkg.

**25 provinces for 15.55M people is ~622k each, and the provinces are wildly uneven**, so
this is Kenya's case again (§8.2). Phnom Penh is 2.28M people in 679 km²; Mondul Kiri is
92,213 people in 14,288 km² of the Eastern Highlands; Ratanak Kiri is 217,453 in 10,782.
An equal share per polygon would wash the empty north-east in evenly spaced dots and squash
a seventh of the country into one speck.

**IT MATTERS FOR THE SAME REASON IT MATTERED IN ZIMBABWE — BECAUSE OF WHAT THE WASH WOULD
SAY.** Mondul Kiri and Ratanak Kiri are the two provinces whose composition is nothing like
the national one: 21.2% and 23.2% `Other` against 0.5% nationally, which is the highland
indigenous religion of the Bunong, Tampuan, Jarai, Kreung and Kavet. They are also two of
the emptiest provinces in the country. An even spread would paint the map's only substantial
non-Buddhist, non-Muslim colour evenly across 25,000 km² of forest where almost nobody
lives, and under-draw the Mekong provinces where the people actually are.

**AND THE TONLE SAP IS INSIDE THE PROVINCES.** The lake runs from ~2,700 km² in the dry
season to ~16,000 km² in flood, and the boundaries of Kampong Thom, Kampong Chhnang,
Pursat, Battambang and Siem Reap all run out into it. A population grid has no hexes on open
water, so §8.2c's problem does not arise rather than being patched — the same thing Malawi
found with Lake Malawi and Zimbabwe with Lake Kariba, and the reason `water.py` is not
involved. The floating villages that really are on the lake keep their dots, because Kontur
models people there and an administrative wash would not have known the difference.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two provinces.

Usage:
    python sources/kh_grid.py --fetch    one 5.6 MB gzipped gpkg from Kontur
    python sources/kh_grid.py            rebuild from data/raw/kh/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kh")
GEO = os.path.join(ROOT, "data", "geo", "kh")
PROVINCES = os.path.join(GEO, "kh_provinces.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "kh.csv")
LOOKUP = os.path.join(GEO, "kh_lookup.csv")
OUT = os.path.join(GEO, "kh_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KH_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KH_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KH_20231101.gpkg"

EXPECTED_PROVINCES = 25
CENSUS_POPULATION = 15_552_211

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Only a within-province weight, so the
# level does not matter and the shape does. Kontur's vintage is 2023 against a 2019 census,
# four years at ~1.1%/yr, so this should read a little above 1.0.
NATIONAL_TOLERANCE = 0.30

# Both nulls are measured below (§12 — Benin against Zimbabwe), and for Cambodia it is the
# CORRELATION that carries the check. 25 uneven units (42,665 people in Kep against
# 2,281,951 in Phnom Penh) give it a real null, and the band cannot be the primary check
# here because of PAILIN — see PAILIN_NOTE.
UNIT_BAND = 1.8

# **KONTUR MODELS 374,607 PEOPLE IN PAILIN AND THE CENSUS COUNTS 75,112 — 4.6x, against a
# spread of 0.62-1.72 across the other twenty-four provinces.** It was worth ruling out
# every way that could have been this project's fault, and all of them are ruled out:
#
#   * the polygons TILE CLEANLY — every pairwise intersection is under 1 km2 and the sum of
#     the 25 areas equals the area of their union to within rounding, so this is not §12's
#     Korea trap where overlapping ADM1 polygons hand a `keep="first"` sjoin the wrong unit;
#   * COD's own `area_sqkm` matches each polygon's measured area to 1.00 on all 25, so the
#     Pailin polygon is not oversized in the file;
#   * Pailin's hexes lie inside Pailin's real bounding box (lon 102.49-102.75, lat
#     12.74-13.11, with Pailin town at 102.61/12.85), so they are not Thai border
#     population leaking across;
#   * and the population is SPREAD — 779 hexes, median 176, and the ten densest hold only
#     13% of the total — so it is not one absurd cell.
#
# So Kontur is simply wrong about Pailin, a small former Khmer Rouge stronghold on the Thai
# border whose built-up footprint has grown far faster than its counted population.
#
# **IT CHANGES NO NUMBER ON THE MAP, AND THAT IS THE POINT OF §8.2.** The grid is a
# WITHIN-UNIT weight: Pailin receives exactly NIS's 75,112 people and they are distributed
# across Pailin's hexes in proportion to each hex's share of Pailin. Kontur's level is
# divided out; only its shape inside the province survives. The one thing to carry is that
# Pailin's dots sit on this map's least trustworthy placement surface.
PAILIN_NOTE = "KH24"

# A shuffled join puts a median 16 of 25 provinces outside the band (measured below), so
# tolerating two real outliers still leaves the band a strong gross-failure check while not
# quietly widening it into decoration (§12 — Benin's warning about exactly that).
MAX_OUTSIDE_BAND = 2


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
    if not os.path.exists(PROVINCES):
        raise SystemExit(f"missing {PROVINCES} -- run sources/kh_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    prov = gpd.read_file(PROVINCES)
    if len(prov) != EXPECTED_PROVINCES:
        raise SystemExit(f"{PROVINCES} has {len(prov)} provinces, "
                         f"expected {EXPECTED_PROVINCES}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(prov.crs)
    hexes = hexes.to_crs(prov.crs)

    joined = gpd.sjoin(pts, prov[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every province: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's KH extract overruns into Thailand, Laos and Vietnam; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=prov.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(prov["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"provinces whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_PROVINCES} provinces has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2019 census count, four years apart, and "
          "the census\n     universe excludes Cambodians working abroad (sources/kh.md §2) "
          "— so a ratio a little\n     above 1.0 is what this should read.")

    # ---- per province, and BOTH nulls (§12 — Benin measured one, Zimbabwe the other) ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "province"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])

    name_of = dict(zip(prov["unit"], prov["name"]))
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print("\n  per province, Kontur/census normalised by the national ratio:")
    print(f"    {'':<22} {'census':>10} {'kontur':>10} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<22} {c:>10,} {k:>10,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    print(f"\n  outside the factor-of-{UNIT_BAND:g} band: {len(worst)} of {len(rows)} — "
          f"{[(w[1], round(w[4], 2)) for w in worst]}")
    print("     Pailin is Kontur's failure and not the join's; the module docstring lists "
          "the four\n     ways that were ruled out. It moves dots inside Pailin and no "
          "count anywhere (§8.2).")
    if len(worst) > MAX_OUTSIDE_BAND:
        raise SystemExit(
            f"{len(worst)} provinces outside a factor of {UNIT_BAND:g}, more than the "
            f"{MAX_OUTSIDE_BAND} this country is known to have: "
            f"{[(w[1], round(w[4], 2)) for w in worst]}")

    rng = random.Random(0)
    ks = [r[3] for r in rows]
    fails = []
    for _ in range(2000):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for (u, nm, c, k, r), k2 in zip(rows, sh)
                         if not (1 / UNIT_BAND <= k2 / c / ratio <= UNIT_BAND)))
    band_pass = sum(1 for f in fails if f <= MAX_OUTSIDE_BAND)
    med = sorted(fails)[len(fails) // 2]
    print(f"\n  BAND control: shuffling the populations across the polygons puts a median "
          f"{med} of\n  {len(rows)} provinces outside the factor-of-{UNIT_BAND:g} band, "
          f"and {band_pass} of 2,000 shuffles come in\n  at or under the "
          f"{MAX_OUTSIDE_BAND} this country really has. So the band still discriminates "
          "hard even\n  after Pailin is tolerated — it is not being widened into "
          "decoration.")
    if med <= MAX_OUTSIDE_BAND:
        raise SystemExit(
            f"a shuffled join puts a median {med} provinces outside the band, which the "
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
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"  CORRELATION control: r = {r_true:.4f} against a best of {perm[-1]:.4f} over "
          f"2,000 shuffles\n  ({beat} reach it).")
    if beat > 100 or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "kh_geo.py is not carrying information")
    print("  Cambodia has 25 units and they are very uneven, which is the shape where both "
          "checks\n  work — unlike Benin (77 alike units, band useless) or Zimbabwe (10 "
          "units, correlation\n  useless). Both are asserted here because both discriminate.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
