"""Chile — the placement layer: Kontur 400 m population hexagons, keyed to comuna.

Writes data/geo/cl/cl_hexes.gpkg. `countries.py` uses it to weight where a comuna's dots land,
never to change how many there are.

ADDED 2026-09-14 (session `f95259a4-clht`) after Anita's look at the map: *"population
distributions look a bit more artificial than i'd expect."* Until then Chile was placed on the
345 comuna polygons themselves, so every comuna's dots fell as an even speckle across its whole
area. That is invisible in Santiago, where comunas are 6-20 km² of continuous city, and it is
the loudest thing on the map everywhere else, because **the median comuna is 630 km² and the
largest run to tens of thousands**:

  * **The north is desert with the people in a handful of coastal and oasis towns.** Antofagasta
    comuna is about 30,000 km² and nearly all of its 400,000 people are in one city on the
    shore; spread evenly, they paint the Atacama.
  * **The south is mountains, ice and fjord.** Aysén, Magallanes and the Andean comunas of Los
    Lagos own icefields and the whole Cordillera, with their people in a valley town or on the
    coast. Natales is 49,000 km² and one town.
  * **And the central valley's rural comunas are farmland around a town**, which is where the
    even speckle reads as artificial rather than as wrong: the dots are in the right comuna and
    in none of the places people live.

§8.2e's floor is nowhere near: 630 km² over a 0.74 km² hex is about 850 hexes to a median
comuna, and the smallest, San Ramón at 6.3 km², still takes eight or nine.

THE JOIN IS SPATIAL, ON HEX CENTROIDS, so a hex on a comuna line belongs wholly to one side and
no population is split or double-counted. Hexes whose centroid is outside every comuna are
dropped and reported. A comuna too small to catch a centroid would fall back to its own polygon
as one uniform cell (`sources/pe_grid.py`'s rule); none does today, and the fallback is kept so a
future boundary file cannot silently lose one.

THE VINTAGE GAP IS ONE YEAR: Kontur's 2023 release against the April 2024 census count.

Usage:
    python sources/cl_grid.py --fetch    one ~10 MB gz from Kontur
    python sources/cl_grid.py            rebuild from data/raw/cl/
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
RAW = os.path.join(ROOT, "data", "raw", "cl")
GEO = os.path.join(ROOT, "data", "geo", "cl")
COMUNAS = os.path.join(GEO, "cl_comunas.gpkg")
POP = os.path.join(RAW, "D1_Poblacion-por-sexo-y-edad.xlsx")
OUT = os.path.join(GEO, "cl_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CL_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CL_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CL_20231101.gpkg"

EXPECTED_COMUNAS = 345
# CPV 2024, everyone counted (D1). The religion table covers the 15,205,784 aged 15+; Kontur
# models everybody, so the comparable denominator is everybody.
CENSUS_POPULATION = 18_480_432

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). One year apart, so the national band is
# tight-ish; it is only ever a within-comuna weight.
NATIONAL_TOLERANCE = 0.25

# Per comuna, after normalising by the national ratio. A tripwire against a wholesale
# mispairing, not evidence; the correlation below is the check. MEASURED 2026-09-14, and it is
# Peru's shape (sources/pe_grid.py): the band is ruined by size. The 90 comunas of 50,000+
# people run 0.33-1.90, and everything past a factor of 3 is a comuna of under 7,000:
#
#     Timaukel           157 people   9.19      Magallanes estancias and a park
#     Torres del Paine   203          9.06      hotels and park buildings, not residents
#     Río Verde          102          7.55
#     San Gregorio       241          6.83
#     Pica             6,272          5.92      the Collahuasi mine camp, workers counted at home
#     Guaitecas        1,598          0.31      the one small comuna Kontur under-draws
#
# A band tight enough to be evidence would reject the country for Kontur being imprecise about
# hamlets of a hundred people, which at 1:1,000 draw no dot. So it admits them.
UNIT_BAND = 12.0

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


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
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
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


def census_totals():
    """Whole-population count per comuna from INE's D1 workbook, keyed by 5-digit CUT."""
    import pandas as pd

    d1 = pd.read_excel(POP, sheet_name="2", header=3)
    d1 = d1[d1["Código comuna"].notna()]
    d1["cut"] = d1["Código comuna"].astype(int).astype(str).str.zfill(5)
    return dict(zip(d1["cut"], d1["Población censada"].astype(int)))


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(COMUNAS):
        raise SystemExit(f"missing {COMUNAS} -- run sources/cl_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    com = gpd.read_file(COMUNAS)
    if len(com) != EXPECTED_COMUNAS:
        raise SystemExit(f"{COMUNAS} has {len(com)} comunas, expected {EXPECTED_COMUNAS}")
    com["unit"] = com["comuna"].astype(str)
    if not com["unit"].str.fullmatch(r"\d{5}").all():
        raise SystemExit("comuna codes are not 5-digit CUT strings -- the counts join on those")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(com.crs)
    hexes = hexes.to_crs(com.crs)

    joined = gpd.sjoin(pts, com[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every comuna: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%) -- dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=com.crs)

    # A comuna no hex centroid lands in would be DROPPED by scatter.py with an easy-to-miss
    # warning. Give it its own polygon as one uniform cell instead (sources/pe_grid.py).
    have = set(out["unit"])
    orphans = com[~com["unit"].isin(have)].copy()
    if len(orphans):
        orphans["pop"] = 1.0
        out = pd.concat([out, orphans[["unit", "pop", "geometry"]]], ignore_index=True)
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=com.crs)
        print(f"\n  {len(orphans)} comunas catch no hex centroid and fall back to their own "
              "polygon as one\n  uniform cell (§8.2): "
              + ", ".join(f"{r['name']} ({r['unit']})" for _, r in orphans.iterrows()))

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(com["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"comunas with no placement polygon: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"comunas whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_COMUNAS} comunas has a placement polygon: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs CPV 2024 {CENSUS_POPULATION:,} -- ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a one-year gap -- check the download")

    # ---- per comuna, and the shuffle control ----
    census = census_totals()
    name_of = dict(zip(com["unit"], com["name"]))
    fallback = set(orphans["unit"]) if len(orphans) else set()
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in com["unit"] if census.get(u, 0) > 0 and u not in fallback]
    if len(rows) != EXPECTED_COMUNAS - len(fallback):
        raise SystemExit(f"only {len(rows)} comunas found a census total in {POP}")
    rows.sort(key=lambda r: r[4])
    print("\n  the most and least populated by Kontur relative to the census,")
    print("  normalised by the national ratio:")
    print(f"    {'':<26} {'census':>10} {'kontur':>10} {'norm':>6}")
    for u, nm, c, k, r in rows[:8] + rows[-8:]:
        print(f"    {nm[:25]:<26} {c:>10,} {k:>10,.0f} {r:>6.2f}")
    big = [r[4] for r in rows if r[2] >= 50_000]
    big.sort()
    if big:
        print(f"    the {len(big)} comunas of 50,000+ people run {big[0]:.2f}-{big[-1]:.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} comunas outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst][:10]}")
    print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g} (a tripwire only)")

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
    print(f"\n  log census against log Kontur on {len(rows)} comunas: r = {r_true:.4f}, "
          f"against a best of\n  {perm[-1]:.4f} over 500 random pairings ({beat} reach it)")
    if beat > 0 or r_true < 0.90:
        raise SystemExit(f"census and Kontur correlate at r={r_true:.4f} -- the comuna "
                         "geometry is not the comuna the census counted")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
