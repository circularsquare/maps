"""Mozambique — the placement layer: Kontur 400 m population hexagons, keyed to the drawn units.

Writes data/geo/mz/mz_hexes.gpkg.

Kenya's module (`sources/ke_grid.py`) with Mozambique's files. Since 2026-09-15 the units are
sources/mz_geo.py's `mz_units.gpkg`: 121 districts as they were in 2007 in nine provinces, and
Cabo Delgado and Manica whole (sources/mz.md §7). A unit's dots are spread over Kontur hexagons
by hex population. It is a POPULATION weight and not a religion one.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a unit line belongs wholly to one side.
Hexes whose centroid falls outside every unit (the ocean edge, the lake, the border overrun into
Malawi, Zambia, Zimbabwe, South Africa, Eswatini and Tanzania) are dropped and reported.

THE KONTUR WITNESS. Kontur is a 2023 model, so a unit's Kontur people over its fitted 2017
population (data/normalized/mz_districts.csv, or Quadro 11 for the two whole provinces) is
printed for every unit against the country's ratio. It is the only population check on
Zambézia's districts, which have no 2017 district table, and it is how Maquival post was placed
in Nicoadala (sources/mz_2007.py POST_MOVES). A unit outside KONTUR_UNIT_STOP of the country's
ratio stops the build: that is a polygon or a membership error, not a model's noise.

Usage:
    python sources/mz_grid.py --fetch    one gzipped gpkg from Kontur's public bucket
    python sources/mz_grid.py            rebuild from data/raw/mz/
"""

import csv
import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mz")
GEO = os.path.join(ROOT, "data", "geo", "mz")
UNITS = os.path.join(GEO, "mz_units.gpkg")
OUT = os.path.join(GEO, "mz_hexes.gpkg")
FITTED = os.path.join(ROOT, "data", "normalized", "mz_districts.csv")
NORM17 = os.path.join(ROOT, "data", "normalized", "mz.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MZ_20231101.gpkg"

# Kontur is a 2023 model and the census counted 2017, six years of roughly 2.8% annual growth
# earlier, so the two should differ by about a fifth. Used ONLY as a within-unit weight, so the
# level does not matter; the tolerance is wide enough for the growth and no wider.
CENSUS_POPULATION = 26_899_105
KONTUR_TOLERANCE = 0.40
EXPECTED = 123
# Per unit, Kontur/fitted over the country's. Printed outside the first band; stops outside the
# second. Measured 2026-09-15 against 2007 N before the fit: 0.52 to 1.86 with every post
# placed, Nicoadala 0.55 before Maquival moved.
KONTUR_UNIT_NOTE = (0.67, 1.5)
KONTUR_UNIT_STOP = (0.4, 2.5)


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
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def fitted_population():
    """{unit: 2017 population}: the fitted Total rows, and Quadro 11's for the whole provinces."""
    out = {}
    with open(FITTED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["source_category"] == "Total":
                out[r["geo_id"]] = int(r["count"])
    with open(NORM17, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["source_category"] == "Total" and r["geo_id"] in ("MZ02", "MZ06"):
                out[r["geo_id"]] = int(r["count"])
    return out


def kontur_witness(per, names):
    pop = fitted_population()
    missing = sorted(set(per.index) ^ set(pop))
    if missing:
        raise SystemExit(f"units with a Kontur total and no fitted population, or the reverse: {missing}")
    country = per["sum"].sum() / sum(pop.values())
    print(f"\n  Kontur over fitted 2017 population, per unit, over the country's {country:.3f}:")
    rows = sorted((per.loc[u, "sum"] / pop[u] / country, u) for u in pop)
    lo, hi = KONTUR_UNIT_NOTE
    slo, shi = KONTUR_UNIT_STOP
    stop = []
    for rel, u in rows:
        if not lo <= rel <= hi:
            print(f"    {rel:5.2f}  {u:<7} {names.get(u, u):<24} fitted {pop[u]:>10,}  "
                  f"Kontur {per.loc[u, 'sum']:>12,.0f}")
        if not slo <= rel <= shi:
            stop.append((u, round(rel, 2)))
    inside = sum(lo <= r <= hi for r, _ in rows)
    print(f"    ({inside} of {len(rows)} units inside {KONTUR_UNIT_NOTE}; the rest printed above)")
    if stop:
        raise SystemExit(f"units outside {KONTUR_UNIT_STOP} of the country's Kontur ratio: {stop}")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/mz_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS, layer="units")
    if len(units) != EXPECTED:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED}")
    names = dict(zip(units["unit"], units["name"]))

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"units whose hexes sum to zero population: {zero}")
    print(f"  hexes per unit: median {int(per['size'].median()):,}, fewest "
          f"{int(per['size'].min()):,} ({per['size'].idxmin()} {names[per['size'].idxmin()]})")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs 2017 census {CENSUS_POPULATION:,}, ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is more than six years of growth -- check the download")

    kontur_witness(per, names)

    os.makedirs(GEO, exist_ok=True)
    tmp = OUT + ".part.gpkg"
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
