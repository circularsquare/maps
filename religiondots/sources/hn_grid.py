"""Honduras — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/hn/hn_hexes.gpkg.

18 departments over 112,000 km² is about **6,200 km² per unit**, four times the Dominican
Republic's, and Honduras needs the grid for both of §8.2's reasons. EMPTINESS: Gracias a Dios
and Olancho are 36% of the land and 7% of the people, and the Mosquitia's population sits on
the coast and the rivers with roadless pine savanna and rainforest behind. CONCENTRATION:
Cortés and Francisco Morazán are 37% of the country, most of it in San Pedro Sula and
Tegucigalpa, and Francisco Morazán's polygon runs well past the capital into empty mountains.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, as in `sources/do_grid.py`: a hex on a department
line belongs wholly to one side, so nothing is split or double-counted. Hexes whose centroid
falls outside every department are dropped and reported; that is coastline and the cays.

Usage:
    python sources/hn_grid.py --fetch    one ~4.7 MB gzipped gpkg from Kontur
    python sources/hn_grid.py            rebuild from data/raw/hn/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hn")
GEO = os.path.join(ROOT, "data", "geo", "hn")
DEPTS = os.path.join(GEO, "hn_departamentos.gpkg")
LOOKUP = os.path.join(GEO, "hn_lookup.csv")
OUT = os.path.join(GEO, "hn_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_HN_20231101.gpkg.gz")
GZ_NAME = "kontur_population_HN_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_HN_20231101.gpkg"

EXPECTED_DEPTS = 18

# Kontur is modelled, not counted; used ONLY as a within-department weight, so the level only
# has to be roughly right and the band is wide. The comparison is against INE's 2024
# projection, which is itself eleven years from the last count.
KONTUR_TOLERANCE = 0.35

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
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
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage, starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg}, run with --fetch first")
    if not os.path.exists(DEPTS):
        raise SystemExit(f"missing {DEPTS}, run sources/hn_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    depts = gpd.read_file(DEPTS)
    if len(depts) != EXPECTED_DEPTS:
        raise SystemExit(f"{DEPTS} has {len(depts)} departments, expected {EXPECTED_DEPTS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(depts.crs)
    hexes = hexes.to_crs(depts.crs)

    joined = gpd.sjoin(pts, depts[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every department: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=depts.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(depts["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"departments with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"departments whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DEPTS} departments has hexes: "
          f"{per['size'].min():,} to {per['size'].max():,} each")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    projected = dict(zip(lut["unit"], lut["pop_2024"].astype(int)))
    total = sum(projected.values())
    tot = float(out["pop"].sum())
    ratio = tot / total
    print(f"\n  Kontur {tot:,.0f} vs INE's 2024 projection {total:,}, ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the projection disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight; check the download")

    print("\n  per-department Kontur/projection ratio (the shape check):")
    names = dict(zip(lut["unit"], lut["name"]))
    rows = [(names[u], int(r["size"]), r["sum"] / projected[u]) for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<22} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
