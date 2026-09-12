"""Benin — the placement layer: Kontur 400 m population hexagons, keyed to commune.

Writes data/geo/bj/bj_hexes.gpkg.

**BENIN NEEDS THIS FOR THE ORDINARY REASON AND ALSO FOR A COASTAL ONE.** The ordinary reason
is Kenya's: the northern communes are large and thinly settled — Karimama and Malanville run
up to the Niger and hold the W National Park, which has essentially nobody in it — while
Cotonou is 679,012 people in 80 km². An equal share per polygon would wash the park in dots
and compress the country's largest city into a flat tile.

The coastal one is §8.2c's, and Benin has it twice over. **Lac Nokoué and the Lagune de
Porto-Novo are inside the communes, not cut out of them**: Sô-Ava's polygon is largely open
lake, and So-Ava is where Ganvié stands — a town of some 30,000 people built on stilts over
that lake (§8.2c-i, and the point Anita raised there is exactly this place: SOME PEOPLE LIVE
ON THE WATER). A population grid handles both ends of that correctly without a clip and
without a special case: it has no hexes on empty water, and it does have hexes over Ganvié,
because Ganvié has buildings and people. `water.py` is not involved.

**AND THIS IS WHERE THE JOIN IS ACTUALLY TESTED.** `sources/bj_geo.py` pairs 77 communes to
77 polygons by name, and five of those pairings needed a transliteration fold or an
elimination. No amount of name agreement can catch a confident wrong pairing (§12); only a
quantity the join does not determine can. Kontur's modelled population per commune is that
quantity — it is built from building footprints and night lights and knows nothing about
INStaD's table — so the ratio against the census population is asserted as a BAND and the
communes the model fits worst are named rather than summarised.

Usage:
    python sources/bj_grid.py --fetch    one 5.8 MB gzipped gpkg from Kontur
    python sources/bj_grid.py            rebuild from data/raw/bj/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bj")
GEO = os.path.join(ROOT, "data", "geo", "bj")
COMMUNES = os.path.join(GEO, "bj_communes.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "bj.csv")
LOOKUP = os.path.join(GEO, "bj_lookup.csv")
OUT = os.path.join(GEO, "bj_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BJ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BJ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BJ_20231101.gpkg"

EXPECTED_COMMUNES = 77
CENSUS_POPULATION = 10_008_749

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). It is a WITHIN-commune weight, so only
# the shape matters — but the level is still worth a band, because a broken download or a
# wrong CRS would blow through it.
#
# THE BAND IS MEASURED FOR BENIN AND NOT COPIED (§9u's rule). Kontur's vintage is 2023 and
# the census is 2013; Benin grew about 2.7%/yr over those ten years, so the grid should read
# HIGH by roughly 30%. A band centred on 1.0 would be the wrong shape of check.
NATIONAL_TOLERANCE = 0.45

# Per commune the ratio is much looser than nationally, because a modelled surface is least
# accurate where its inputs are thinnest and that is not at random. Serbia's rule: a correct
# join keeps every unit inside a factor of a few around a tight median; a scrambled one
# scatters over orders of magnitude.
#
# **FOUR, NOT THREE, AND THE TWO COMMUNES THAT SET IT ARE THE INTERESTING PART.** At a
# factor of three the only failures are `Sô-Ava` at 0.32x and `Aguégués` at 0.30x, and those
# are the two places in Benin where people live ON THE WATER — Sô-Ava is Ganvié, the stilt
# town of some 30,000 on Lac Nokoué, and Aguégués is islands in the Ouémé delta. A
# population grid built from building footprints is worst exactly there, which is spec
# §8.2c-i's question answered from the other end. Both were paired by an exact name fold, so
# there is no doubt about the join; the band is widened to admit them and the control below
# is what keeps it honest.
UNIT_BAND = 4.0


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
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(COMMUNES):
        raise SystemExit(f"missing {COMMUNES} -- run sources/bj_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, never the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    communes = gpd.read_file(COMMUNES)
    if len(communes) != EXPECTED_COMMUNES:
        raise SystemExit(f"{COMMUNES} has {len(communes)} communes, "
                         f"expected {EXPECTED_COMMUNES}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(communes.crs)
    hexes = hexes.to_crs(communes.crs)

    joined = gpd.sjoin(pts, communes[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every commune: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's BJ extract overruns into Nigeria, Togo, Burkina Faso and Niger, "
          "and\n     out to sea; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=communes.crs)

    # A commune with no hex draws nothing, silently — §12, and it is the failure that would
    # empty a real place with no error anywhere.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(communes["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"communes with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"communes whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_COMMUNES} communes has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    # ---- the national band ----
    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2013 census count, so it should and does "
          "read\n     high; used only as a WITHIN-commune weight, so the level does not "
          "matter and the\n     shape does.")

    # ---- the per-commune band: the real test of bj_geo.py's name join ----
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/bj.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "commune"].groupby("geo_id"):
        census[unit_of[gid]] = int(sub["count"].sum())      # the ten plus the residual
    if len(census) != EXPECTED_COMMUNES:
        raise SystemExit(f"{len(census)} census communes in the lookup, "
                         f"expected {EXPECTED_COMMUNES}")

    name_of = dict(zip(communes["unit"], communes["name"]))
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    print(f"\n  per commune, Kontur/census normalised by the national ratio — "
          f"{len(rows) - len(worst)}/{len(rows)}\n  inside a factor of {UNIT_BAND:g}. "
          "A name join that paired two communes wrongly cannot keep this\n  tight, because "
          "nothing in the census determines where Kontur puts buildings.")
    print(f"    {'':<22} {'census':>10} {'kontur':>10} {'norm':>6}")
    for u, nm, c, k, r in rows[:4] + rows[-4:]:
        print(f"    {nm:<22} {c:>10,} {k:>10,.0f} {r:>6.2f}")
    if worst:
        for u, nm, c, k, r in worst:
            print(f"    OUT OF BAND: {nm} ({u}) census {c:,}, Kontur {k:,.0f}, {r:.2f}x")
        raise SystemExit(f"{len(worst)} communes are outside a factor of {UNIT_BAND:g} -- "
                         "that is what a wrong pairing looks like; check bj_geo.py's join")

    print("     The two extremes are the MODEL and not the join, and they are the same two "
          "places:\n     Sô-Ava (Ganvié) and Aguégués are built over Lac Nokoué and the "
          "Ouémé delta, where a\n     footprint-based surface finds least. Spec §8.2c-i, "
          "from the other end.")

    # ---- and the band on its own is NOT the check, which is worth knowing ----
    #
    # Benin's communes are mostly 50,000-250,000 people, so shuffling the census populations
    # across the polygons leaves most of them inside a factor of four anyway: the band alone
    # would pass a badly scrambled join. It is kept because it catches the gross failures a
    # correlation does not care about (an empty commune, a wrong CRS). The statistic that
    # actually discriminates is the CORRELATION of the two populations, compared against
    # what shuffling produces — measured here rather than asserted from intuition.
    import math
    import random

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
    perm, worst_band = [], 0
    for _ in range(500):
        shuffled = list(lk)
        rng.shuffle(shuffled)
        perm.append(abs(pearson(lc, shuffled)))
        worst_band = max(worst_band, sum(
            1 for x, y in zip(lc, shuffled)
            if abs(y - x - math.log(ratio)) > math.log(UNIT_BAND)))
    perm.sort()
    print(f"\n  the band is NOT the discriminating check and the control says so: shuffling "
          f"the\n  populations across the polygons puts at most {worst_band} of "
          f"{len(rows)} outside the same band,\n  because Benin's communes are mostly "
          f"50,000-250,000 people and look alike.")
    print(f"  What discriminates is the correlation: log-log r = {r_true:.4f} for the join "
          f"as built,\n  against {perm[-1]:.4f} for the best of 500 shuffles "
          f"(median {perm[len(perm) // 2]:.4f}).")
    # The load-bearing condition is beating every shuffle; the absolute floor is a second
    # net and is set below the measured 0.905 so an ordinary Kontur re-release cannot fail
    # the build for no reason.
    if r_true <= perm[-1] or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which 500 random pairings reach ({perm[-1]:.4f}) -- the name "
                         "join in bj_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
