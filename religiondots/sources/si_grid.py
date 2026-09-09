"""Slovenia — the placement layer: Kontur 400 m population hexagons, keyed to the občina.

Writes data/geo/si/si_hexes.gpkg.

**SLOVENIA IS 58% FOREST AND ITS MUNICIPALITIES DO NOT KNOW IT.** The drawn units median 72
km², which sounds workable until you look at the big ones: Kočevje is 556 km² of which the
Kočevski Rog is uninhabited beech forest, Bovec and Bohinj are the Julian Alps, and the
Kras municipalities are limestone with the villages strung along the poljes. An equal share
of dots per polygon would draw a serious part of the country onto forest and rock. It also
takes care of Lake Bohinj and Lake Cerknica, which sit inside their municipalities rather
than between them, so §8.2c's problem does not arise rather than being patched.

**THE GRID IS 2023 AND THE CENSUS IS 2002, AND THAT IS FINE FOR ONE REASON ONLY.** Kontur is
a WITHIN-unit weight: how many dots an občina gets is the census's answer and nothing here
can change it. The 21-year gap therefore cannot move a dot between municipalities; it can
only place a dot inside one according to where people live now rather than in 2002. Slovenia
grew from 1.96M to about 2.12M over that span and the growth is not even (the Ljubljana ring
and the coast gained, the Prekmurje and Zasavje lost), so the ratio band below is a fact
about the country and not an error term.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two municipalities.

Usage:
    python sources/si_grid.py --fetch    one 2 MB gzipped gpkg from Kontur
    python sources/si_grid.py            rebuild from data/raw/si/
"""

import gzip
import math
import os
import random
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "si")
GEO = os.path.join(ROOT, "data", "geo", "si")
UNITS = os.path.join(GEO, "si_units.gpkg")
OUT = os.path.join(GEO, "si_hexes.gpkg")
CSV = os.path.join(ROOT, "data", "normalized", "si.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SI_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SI_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SI_20231101.gpkg"

EXPECTED_UNITS = 192
CENSUS_POPULATION = 1_964_036
TOTAL_LABEL = "Veroizpoved - SKUPAJ"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000:
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
    for p in (gpkg, UNITS, CSV):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run si_grid.py --fetch, si.py and si_geo.py")

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

    # Kontur ships in EPSG:3857.  Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["obcina", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["obcina"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's SI extract overruns the border and GISCO's 1:1,000,000 outline is "
          "coarser\n     than the grid, so a thin rim of hexes falls outside; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "obcina"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    df = pd.read_csv(CSV, dtype={"geo_id": str})
    tot_by_unit = (df[(df["geo_level"] == "municipality")
                      & (df["source_category"] == TOTAL_LABEL)]
                   .set_index("geo_id")["count"].to_dict())

    # A unit absent from the placement layer has no geometry to draw into and its people are
    # dropped rather than spread ([[reference_kontur_resolution_floor]]).  Slovenia's
    # smallest municipality is Osilnica at 402 people over 36 km², so the resolution floor
    # should not bite at all here; if it does, that is a finding and not a tolerance.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["obcina"]) - set(per.index))
    if missing:
        raise SystemExit(f"{len(missing)} units with no populated hex: {missing[:12]} -- "
                         "Slovenia's smallest unit is 36 km2, so this is a join failure "
                         "rather than the grid's resolution floor")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"{len(zero)} units whose hexes sum to zero: {zero[:12]}")
    print(f"  every one of the {EXPECTED_UNITS} drawn units has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    print(f"\n  Kontur 2023 {tot:,.0f} vs census 2002 {CENSUS_POPULATION:,} — "
          f"ratio {tot / CENSUS_POPULATION:.3f}")
    print("     Slovenia grew by about a twelfth between the two dates, unevenly, so this "
          "ratio is a\n     fact about the country. The grid is a WITHIN-unit weight only.")

    rows = [(u, float(tot_by_unit[u]), float(per.loc[u, "sum"]))
            for u in units["obcina"] if u in tot_by_unit]
    print(f"\n  ratio band, Kontur vs census, over {len(rows)} units:")
    ratios = sorted((k / c, u) for u, c, k in rows)
    names = dict(zip(units["obcina"], units["name"]))
    for q, label in ((0.01, " 1st pct"), (0.25, "25th pct"), (0.50, "  median"),
                     (0.75, "75th pct"), (0.99, "99th pct")):
        print("    %s  %.2f" % (label, ratios[int(q * (len(ratios) - 1))][0]))
    print("    lowest  %.2f (%s)   highest  %.2f (%s)"
          % (ratios[0][0], names.get(ratios[0][1]), ratios[-1][0], names.get(ratios[-1][1])))

    lc = [math.log(c) for _, c, _ in rows]
    lk = [math.log(k) for _, _, k in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(500):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  CORRELATION control: r = {r_true:.4f} against a best of {perm[-1]:.4f} over "
          f"500 shuffles ({beat} reach it).")
    if beat > 0 or r_true < 0.85:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, which "
                         f"{beat} of 500 random pairings reach -- the join in si_geo.py is "
                         "not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
