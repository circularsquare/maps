"""Mauritius — the placement layer: Kontur 400 m population hexagons, keyed to unit.

Writes data/geo/mu/mu_hexes.gpkg.

**MAURITIUS NEEDS THIS LESS THAN ANY COUNTRY THAT HAS USED IT, AND IT IS STILL WORTH IT.**
The drawn units average 11 km² and 6,800 people, so an equal share per polygon would already
be close to honest — this is nothing like Kenya's deserts or Malawi's lake. Two things still
go wrong without a grid:

  * **The coastal VCAs own their lagoon.** Mauritius is ringed by reef and the VCA
    boundaries run out to it, so a seaside village's polygon is substantially water. Kontur
    has no hexes there, which is §8.2c solved for free again (§9u).
  * **The big rural units are mountain and cane.** Grande Rivière Noire is 43.5 km² and
    Tamarin 48.0 km² against a median of about 6 km², and both are largely gorge, forest and
    estate. Their few thousand people live along one road each.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a boundary belongs wholly to one
side and no population is double-counted. Hexes whose centroid is outside every unit — the
lagoon, the reef and the open sea between the two islands — are dropped and reported.

**RODRIGUES IS 600 KM EAST AND IS THE REASON TO CHECK PER-UNIT COVERAGE RATHER THAN THE
NATIONAL TOTAL.** Its six regions hold 43,604 people, 3.5% of the country; a Kontur extract
that quietly omitted the island would still pass a national ratio test inside any sane band.
Every unit is checked for hexes individually.

**AND KONTUR IS COARSE RELATIVE TO THIS COUNTRY**, which nothing else here has been: 2,072
hexes for 182 units is about 11 each, and three cross-district slivers of 0.09-3.6 km²
contain no hex centroid at all. They are added to the placement layer as one cell covering
their own polygon — **a unit absent from the placement layer has no geometry to put a dot in
at all**, and `place_weight`'s equal-share fallback cannot fire because there is nothing to
share, so leaving them out drops their 890 people instead of scattering them uniformly.

Usage:
    python sources/mu_grid.py --fetch    one 146 KB gzipped gpkg from Kontur
    python sources/mu_grid.py            rebuild from data/raw/mu/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mu")
GEO = os.path.join(ROOT, "data", "geo", "mu")
UNITS = os.path.join(GEO, "mu_units.gpkg")
OUT = os.path.join(GEO, "mu_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MU_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MU_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MU_20231101.gpkg"

EXPECTED_UNITS = 182

# Measured, not copied (§9u). Kontur is 2023 and the census 2022, and Mauritius's population
# is flat to slightly falling, so the two should be close — a much tighter band than Malawi's.
CENSUS_POPULATION = 1_233_097
KONTUR_TOLERANCE = 0.20


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 200_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=900, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
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
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/mu_geo.py first")

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

    # Centroid in the CRS Kontur tiled in, then reproject the POINTS.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     the lagoon, the reef and the sea between the islands; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    empty = sorted(set(units["unit"]) - set(per.index)) + \
        sorted(per.index[per["sum"] <= 0])

    # **KONTUR IS COARSE RELATIVE TO MAURITIUS AND A FEW SLIVERS FALL THROUGH IT.** The whole
    # country is 2,072 hexes for 182 units — about 11 each — so a unit smaller than a hex can
    # contain no centroid at all. Three do: cross-district fragments of 0.09 to 3.6 km².
    # They fall back to an equal share within their own polygon, which at that size is not an
    # approximation of anything. This is reported rather than raised because the failure mode
    # it would otherwise hide — a LARGE unit with no hexes, which would mean a broken join —
    # is caught by the population cap below instead.
    if empty:
        idx = units.set_index("unit")
        print(f"\n  {len(empty)} units contain no hex centroid:")
        for u in empty:
            print(f"      {u}  {idx.loc[u, 'name']}")
        big = [u for u in empty
               if idx.loc[u, "geometry"].area * 111.32 * 111.32 * 0.94 > 10]
        if big:
            raise SystemExit(f"units larger than 10 km2 have no hex: {big} -- that is a "
                             "broken join, not a sliver")
        # **THE WHOLE UNIT IS ADDED AS ITS OWN PLACEMENT CELL, and it has to be.** The
        # placement layer is the ONLY geometry scatter.py sees — a unit absent from it has
        # no polygon to put a dot in, and `place_weight`'s equal-share fallback cannot fire
        # because there is nothing to share. Left out, these three units' 890 people are
        # reported as unplaceable and silently dropped. Given the polygon they get a uniform
        # scatter inside their own boundary, which at 0.09-3.6 km2 is exact enough to be
        # indistinguishable from a finer model.
        add = gpd.GeoDataFrame(
            {"unit": empty, "pop": [1.0] * len(empty)},
            geometry=[idx.loc[u, "geometry"] for u in empty], crs=units.crs)
        out = gpd.GeoDataFrame(pd.concat([out, add], ignore_index=True),
                               geometry="geometry", crs=units.crs)
        print("      each added to the placement layer as one cell covering its own "
              "polygon, so\n      its dots scatter uniformly inside it rather than being "
              "dropped (§8.2)")
    print(f"\n  the {EXPECTED_UNITS - len(empty)} units that do have hexes carry "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    # Rodrigues is the one to name explicitly -- see the module docstring.
    rod = units[units["name"].str.contains("Region", na=False)]["unit"]
    rp = per.loc[per.index.isin(rod)]
    print(f"      Rodrigues' {len(rp)} regions carry {rp['sum'].sum():,.0f} modelled people "
          "against a census 43,604")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2022 census on a country whose population is "
          "flat,\n     so the two should be close and are; used only as a WITHIN-unit "
          "weight.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
