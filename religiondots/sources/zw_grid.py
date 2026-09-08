"""Zimbabwe — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/zw/zw_hexes.gpkg.

**THIS IS KENYA'S CASE IN ITS PUREST FORM AND THE MAP WOULD BE UNREADABLE WITHOUT IT.**
Zimbabwe is drawn on **10 provinces for 15.2M people, ~1.5M each — the coarsest counting
geography anywhere on this map** — and the provinces are wildly uneven in habitability.
Matabeleland North is 75,025 km² holding Hwange National Park and much of the Zambezi
escarpment; Matabeleland South is 54,172 km² of dry ranching country; and Harare and
Bulawayo are two metropolitan provinces of 872 km² and 479 km² holding 3.2M people between
them. An equal share per polygon would wash the empty west in evenly spaced dots and squash
a fifth of the country into two specks.

It matters more here than anywhere because of what the wash would SAY. The two Matabeleland
provinces are 129,197 km² — a third of Zimbabwe — holding 1.59M people between them, and
they are the two whose composition is least like the national one: **32-34% `Apostolic
Sect` against 40% nationally, 13-16% `Other Christian` against 8%, and Matabeleland South
has the highest `None` in the country at 13.5%.** An even spread would paint that
distinctive mix across a third of the map's Zimbabwe on ground where almost nobody lives,
and would under-draw the Mashonaland provinces where the people actually are. §8.2's
argument with the loudest available consequence.

**AND LAKE KARIBA IS INSIDE THE PROVINCES.** At 5,580 km² it is one of the largest
reservoirs on earth and the Matabeleland North / Mashonaland West boundary runs out into it.
A population grid has no hexes on open water, so §8.2c's problem does not arise rather than
being patched — the same thing Malawi found with Lake Malawi and Ethiopia with its rift
lakes, and the reason `water.py` is not involved.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two provinces.

Usage:
    python sources/zw_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/zw_grid.py            rebuild from data/raw/zw/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "zw")
GEO = os.path.join(ROOT, "data", "geo", "zw")
PROVINCES = os.path.join(GEO, "zw_provinces.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "zw.csv")
LOOKUP = os.path.join(GEO, "zw_lookup.csv")
OUT = os.path.join(GEO, "zw_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ZW_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ZW_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ZW_20231101.gpkg"

EXPECTED_PROVINCES = 10
CENSUS_POPULATION = 15_178_957

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Only a within-province weight, so the
# level does not matter and the shape does.
#
# THE BAND IS MEASURED FOR ZIMBABWE (§9u's rule). Kontur's vintage is 2023 and the census is
# 2022, one year apart at ~1.5%/yr, so this one should read close to 1.0 — unlike Malawi's
# five-year gap or Benin's ten. A band centred anywhere else would be the wrong shape.
NATIONAL_TOLERANCE = 0.30

# **ZIMBABWE IS BENIN'S LESSON WITH THE ANSWER REVERSED, AND BOTH HALVES WERE MEASURED.**
# bj_grid.py found that Benin's ratio band could not tell a right join from a shuffled one
# (77 similar-sized communes) while the correlation could. Here it is the other way round:
#
#   * the BAND is tight and discriminating. Kontur agrees with the 2022 census to within
#     0.83x-1.11x on every province, because the grid's vintage is one year off the census
#     rather than ten. A factor of 1.6 leaves real headroom and is still far tighter than
#     any mispairing could survive — Harare is 2.4M people in 872 km2 and Matabeleland
#     North is 828k in 75,025 km2, so swapping any two provinces moves a ratio by an order
#     of magnitude. The shuffle control below measures that rather than asserting it.
#   * the CORRELATION is weak, because ten log-populations of similar size are easy to
#     correlate by luck: the best of 2,000 random pairings reaches r = 0.976 against the
#     true join's 0.979. It is kept, and it is not what is carrying the check.
#
# The transferable form: **measure both, use whichever the country's own shape makes
# discriminating, and say which one it was.**
UNIT_BAND = 1.6


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
    if not os.path.exists(PROVINCES):
        raise SystemExit(f"missing {PROVINCES} -- run sources/zw_geo.py first")

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
    print("     Kontur's ZW extract overruns into Zambia, Mozambique, Botswana and South "
          "Africa,\n     and across Lake Kariba; dropped.")

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
    print("     a 2023 modelled grid against a 2022 census count, one year apart, so this "
          "one\n     should and does read close to 1.0.")

    # ---- per province, and the correlation control (sources/bj_grid.py's lesson) ----
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
    print(f"\n  per province, Kontur/census normalised by the national ratio:")
    print(f"    {'':<22} {'census':>10} {'kontur':>10} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<22} {c:>10,} {k:>10,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} provinces outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")

    import math
    import random
    rng = random.Random(0)
    ks = [r[3] for r in rows]
    fails = []
    for _ in range(2000):
        sh = list(ks)
        rng.shuffle(sh)
        fails.append(sum(1 for (u, nm, c, k, r), k2 in zip(rows, sh)
                         if not (1 / UNIT_BAND <= k2 / c / ratio <= UNIT_BAND)))
    print(f"\n  the band IS the discriminating check here, and the control says so: "
          f"shuffling the\n  populations across the polygons puts a median "
          f"{sorted(fails)[len(fails) // 2]} of {len(rows)} provinces outside it, and "
          f"{sum(1 for f in fails if f == 0)}\n  of 2,000 shuffles pass it cleanly. "
          "Zimbabwe's provinces are wildly uneven — Harare is\n  2.4M people in 872 km2 "
          "against Matabeleland North's 828k in 75,025 — which is exactly\n  the shape "
          "Benin lacked (sources/bj_grid.py).")

    # The correlation, kept and reported as the WEAK check here — see UNIT_BAND's note.
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
    print(f"\n  and the correlation, which is the WEAK check on ten units: "
          f"r = {r_true:.4f}, against a best\n  of {perm[-1]:.4f} over 2,000 shuffles "
          f"({beat} reach it). Ten log-populations of similar size\n  correlate by luck, "
          "which is why the band above is what this country is checked on.")
    if beat > 100 or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "zw_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
