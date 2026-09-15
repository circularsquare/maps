"""Venezuela — the placement layer: Kontur 400 m population hexagons, keyed to federal entity.

Writes data/geo/ve/ve_hexes.gpkg.

§8.2's two reasons, both at full strength. **Emptiness**: Amazonas, Bolívar, Delta Amacuro and
the llanos of Apure and Guárico are most of the land and a small share of the people; Bolívar
alone is over a quarter of the country's area. **Concentration**: 75% of Venezuelans live on the
20% of the land along the coastal mountains (INE, Censo 2011 *Resultados Total Nacional*, p. 14),
and Caracas, Maracaibo, Valencia, Barquisimeto and Maracay hold a large part of that. An equal
share per polygon would paint Bolívar's Catholics across the Gran Sabana.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, `sources/bo_grid.py`'s construction. Hexes whose
centroid is outside every entity (the VE extract overruns into Colombia, Brazil, Guyana and the
Caribbean) are dropped and reported.

Usage:
    python sources/ve_grid.py --fetch    one ~14 MB gzipped gpkg from Kontur
    python sources/ve_grid.py            rebuild from data/raw/ve/ (a minute or two)
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ve")
GEO = os.path.join(ROOT, "data", "geo", "ve")
UNITS = os.path.join(GEO, "ve_estados.gpkg")
POP = os.path.join(GEO, "ve_pop_2011.csv")
OUT = os.path.join(GEO, "ve_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_VE_20231101.gpkg.gz")
GZ_NAME = "kontur_population_VE_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_VE_20231101.gpkg"

EXPECTED = 25

# Kontur is modelled, not a census (§12, North Macedonia); only the relationship is asserted.
# A 2023-11 grid against the 2011 census, twelve years and a mass emigration apart, so the
# per-entity ratios below are printed and not asserted.
CENSUS_POPULATION = 27_227_930
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


def unpack():
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
        return gpkg
    if not os.path.exists(gz):
        raise SystemExit(f"missing {gz} — run with --fetch first")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")
    return gpkg


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} — run sources/ve_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED:
        raise SystemExit(f"{UNITS} has {len(units)} entities, expected {EXPECTED}")

    # centroid in the tiling CRS, then reproject the points (reprojecting first moves it)
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every entity: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    # The four entities no LAPOP wave sampled draw no dots (sources/ve.py NOT_DRAWN), so they
    # have no business in the placement layer. Keeping them would also put Kontur's lone hex at
    # the cap on Cubagua island (29,067 people, nothing populated within 10 km) in front of
    # kontur_cap.py, which can neither cap a block with no ring nor call it real. A later source
    # that draws Nueva Esparta must put them back and review that hex.
    NOT_PLACED = {"VE02": "Amazonas", "VE10": "Delta Amacuro", "VE17": "Nueva Esparta",
                  "VE25": "Dependencias Federales"}
    unplaced = joined["unit"].isin(set(NOT_PLACED))
    print(f"  hexes in the four never-sampled entities: {int(unplaced.sum()):,} "
          f"({float(pts.loc[unplaced, popcol].sum()):,.0f} people); left out, they draw nothing")
    keep = ~outside & ~unplaced
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    names = dict(zip(units["unit"], units["name"]))
    missing = sorted(set(units["unit"]) - set(NOT_PLACED) - set(per.index))
    if missing:
        raise SystemExit(f"drawn entities with no populated hex: {[names[u] for u in missing]}")
    if set(per.index) & set(NOT_PLACED):
        raise SystemExit("a never-sampled entity is still in the placement layer")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"entities whose hexes sum to zero population: {zero}")
    print(f"  entities with hexes: {len(per)}, {per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2011 census {CENSUS_POPULATION:,}, ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")

    print("\n  per-entity Kontur/census ratio (the shape check, printed):")
    cen = pd.read_csv(POP, encoding="utf-8-sig")
    cen = dict(zip(cen["geo_id"].astype(str).str.strip(), cen["pop"]))
    rows = [(names[u], int(r["size"]), r["sum"] / cen[u]) for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm[:34]:<34} {n:>8,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
