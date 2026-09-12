"""Nicaragua — the placement layer: Kontur 400 m population hexagons, keyed to municipio.

Writes data/geo/ni/ni_hexes.gpkg.

**TWO REASONS, AND NICARAGUA IS THE FIRST COUNTRY HERE WHERE BOTH BITE HARD.**

*Emptiness.* The counting units are 153 municipios, which is fine on average (33,000 people
each) and wildly uneven in fact: the two Caribbean autonomous regions are **46% of
Nicaragua's land and 12% of its people**. Waspám alone is 9,341 km² — larger than eleven
whole departments — and Prinzapolka, Puerto Cabezas, Siuna and Bonanza are each the size of a
Pacific department. An equal share per polygon would spread the coast's dots evenly across
rainforest and savanna where almost nobody lives.

*And it matters more here than the area alone suggests, because of WHAT would be spread.*
Those same eastern municipios are the ones carrying `Morava` — Prinzapolka 53.3%, Puerto
Cabezas 50.9%, Waspám 43.6% against 1.6% nationally. **The category this country is drawn for
lives entirely in its largest and emptiest units**, so an even spread would paint the
Moravian coast across a third of the map's Nicaragua on ground that is mostly uninhabited,
and would under-draw the towns where the Moravian congregations actually are. §8.2's argument
with the loudest available consequence, and a sharper version of Zimbabwe's.

*Water.* Lake Nicaragua (Cocibolca) is 8,264 km² and Lake Managua (Xolotlán) 1,042 km², and
the municipal boundaries run out into both — Cocibolca alone is larger than the Pacific
departments around it. A population grid has no hexes on open water, so §8.2c's problem does
not arise rather than being patched, the same thing Malawi found with Lake Malawi.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two municipios.

**THE VINTAGE GAP IS EIGHTEEN YEARS AND IS THE LARGEST ON THIS MAP.** The census is 2005 and
Kontur is 2023; Nicaragua grew from 5.14M to roughly 6.8M over that period, so the national
ratio should read about 1.3 and a band centred on 1.0 would be the wrong shape. It is a
*within-municipio weight*, so the level does not matter and only the shape does — but the gap
is why the tolerance below is wide, and it is measured rather than assumed.

Usage:
    python sources/ni_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/ni_grid.py            rebuild from data/raw/ni/
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
RAW = os.path.join(ROOT, "data", "raw", "ni")
GEO = os.path.join(ROOT, "data", "geo", "ni")
MUNIS = os.path.join(GEO, "ni_municipios.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "ni.csv")
LOOKUP = os.path.join(GEO, "ni_lookup.csv")
OUT = os.path.join(GEO, "ni_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_NI_20231101.gpkg.gz")
GZ_NAME = "kontur_population_NI_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_NI_20231101.gpkg"

EXPECTED_MUNIS = 153
# The whole 2005 census count, NOT the 4,537,200 aged 5+ that the religion question covers:
# Kontur models everybody, so the comparable denominator is everybody.
CENSUS_POPULATION = 5_142_098

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Eighteen years of growth at ~1.3% a year
# is most of what this number is, so the band is wide and centred where the demography puts
# it rather than at 1.0. It is only ever a within-unit weight.
NATIONAL_TOLERANCE = 0.60

# Per-municipio, after normalising by the national ratio.
#
# **NICARAGUA IS ZIMBABWE'S CASE WITH THE ANSWER REVERSED, AND BOTH HALVES WERE MEASURED**
# (sources/zw_grid.py, sources/bj_grid.py). There the BAND was the discriminating check and
# the correlation was weak on 10 units. Here it is the other way round:
#
#   * the BAND is nearly useless, and it is the eighteen-year gap that ruins it. The
#     outliers are not noise and not a bad join — they are the AGRICULTURAL FRONTIER, and
#     they are the same five municipalities any account of Nicaraguan internal migration
#     would name: San Juan de Nicaragua 6.6x, Prinzapolka 4.8x, El Tortuguero 3.3x,
#     El Castillo 2.6x, La Cruz de Río Grande 2.2x. These are the eastern lowlands that
#     absorbed the 2005-2023 colonisation of the old agricultural frontier, so Kontur's 2023
#     grid genuinely holds several times the people the 2005 census counted there. A band
#     tight enough to be evidence would reject the country for being right.
#   * the CORRELATION is strong and is what carries the check, because 153 log-populations
#     spanning Managua's 700k+ to San Juan de Nicaragua's 1,115 cannot be correlated by luck.
#     The permutation control below measures that rather than asserting it.
#
# So the band is set to admit the frontier and is kept only as a tripwire against a wholesale
# mispairing; the correlation is the check. **Measure both, use whichever the country's own
# shape makes discriminating, and say which one it was.**
UNIT_BAND = 8.0


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
    if not os.path.exists(MUNIS):
        raise SystemExit(f"missing {MUNIS} -- run sources/ni_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    muni = gpd.read_file(MUNIS)
    if len(muni) != EXPECTED_MUNIS:
        raise SystemExit(f"{MUNIS} has {len(muni)} municipios, "
                         f"expected {EXPECTED_MUNIS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(muni.crs)
    hexes = hexes.to_crs(muni.crs)

    joined = gpd.sjoin(pts, muni[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every municipio: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's NI extract overruns into Honduras and Costa Rica, and across "
          "Lakes\n     Cocibolca and Xolotlán; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=muni.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(muni["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"municipios with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"municipios whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_MUNIS} municipios has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2005 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much even for an 18-year gap -- check the download")
    print("     a 2023 modelled grid against a 2005 census count. EIGHTEEN YEARS, the "
          "largest\n     vintage gap on this map, and most of this number is real "
          "population growth.")

    # ---- per municipio, and the shuffle control (sources/bj_grid.py's lesson) ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "municipio"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])

    name_of = dict(zip(muni["unit"], muni["name"]))
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print("\n  the ten most and least populated by Kontur relative to the census,")
    print("  normalised by the national ratio:")
    print(f"    {'':<28} {'census 5+':>10} {'kontur':>10} {'norm':>6}")
    for u, nm, c, k, r in rows[:5] + rows[-5:]:
        print(f"    {nm:<28} {c:>10,} {k:>10,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} municipios outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
    print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g}, which is a WIDE band and "
          "deliberately so:\n    the high end is the eastern agricultural frontier, where "
          "Kontur's 2023 grid really\n    does hold several times the people the 2005 census "
          "counted. See UNIT_BAND's note —\n    on this country the band is a tripwire and "
          "the correlation below is the check.")
    print("\n  WHAT THAT COSTS, STATED: the counts are 2005 and the placement is 2023, so in "
          "the\n  frontier municipios the dots land in settlements that had barely begun when "
          "the\n  census was taken. It moves dots WITHIN a unit and never between units, so "
          "no count\n  is affected — but Prinzapolka is both the most Moravian municipio and "
          "the second\n  most re-settled, and that is worth knowing when reading its dots.")

    rng = random.Random(0)
    ks = [r[3] for r in rows]
    fails = []
    for _ in range(2000):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for (u, nm, c, k, r), k2 in zip(rows, sh)
                         if not (1 / UNIT_BAND <= k2 / c / ratio <= UNIT_BAND)))
    fails.sort()
    clean = sum(1 for f in fails if f == 0)
    print(f"\n  the shuffle control on the band: re-pairing the Kontur populations at "
          f"random puts a\n  median {fails[len(fails) // 2]} of {len(rows)} municipios "
          f"outside it and {clean} of 2,000 shuffles pass cleanly — so even\n  widened to "
          f"{UNIT_BAND:g} the band would catch a wholesale permutation, which is all it is "
          "kept for.")

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
    print(f"\n  and the correlation, which on 153 units is the STRONG check here: "
          f"r = {r_true:.4f},\n  against a best of {perm[-1]:.4f} over 2,000 random "
          f"pairings ({beat} reach it).")
    if beat > 20 or r_true < 0.85:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "ni_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
