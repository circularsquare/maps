"""Indonesia — the placement layer: Kontur 400 m population hexagons, keyed to the drawn unit.

Writes data/geo/id/id_hexes.gpkg.

**INDONESIA'S PLACEMENT PROBLEM IS NOT KENYA'S, and it shows up in the opposite place.**
Kenya's counties are huge and mostly empty, so uniform placement washed dots over desert.
Indonesia is drawn at kecamatan for 403 of its 492 regencies, and most of those are small
enough that uniform placement is honest. Two things still go wrong without a weight:

  * **the 89 regencies drawn whole**, several of which are enormous and nearly empty —
    Papuan and interior Kalimantan units where the Kenya failure applies exactly; and
  * **the dense urban kecamatan**, which is what you actually notice. Cengkareng is 513,920
    people and Cakung 503,846, each drawn as an even wash across its whole polygon, so a
    city reads as a set of flat-shaded tiles with visible administrative edges instead of a
    built-up area with a shape. Jakarta and Surabaya are where this is most obvious.

**WHAT THIS DOES NOT FIX, and must not be claimed to.** Within a kecamatan the map still
places every religion by the SAME population weight, because that is all the source
supports: BPS publishes religion at kecamatan and nothing below it (the wid space's second
pass is a repeat of the regency tier, not a kelurahan tier — see sources/id.md §2). So a
mixed kecamatan still draws its religions interleaved, and the real street-level
segregation of, say, Kelapa Gading stays invisible. Weighting religions differently inside
a unit would be inventing a magnitude the source does not publish, which spec §14.4 forbids
outright. This layer refines WHERE THE PEOPLE ARE, never who they are.

THE JOIN IS SPATIAL, on hex CENTROIDS — a hex on a boundary belongs wholly to one side, so
none is split and no population is double-counted. Hexes whose centroid falls outside every
drawn unit (the sea edge, the Malaysian/PNG/Timorese border overrun, and the post-2010
territory this map does not paint) are dropped and reported.

**A UNIT WITH NO POPULATED HEX FALLS BACK TO ITS OWN POLYGON** rather than failing the run
or emptying. Kenya could assert that all 47 counties get hexes; 5,212 units cannot be
assumed to, and a unit missing from the placement layer draws NOTHING at all while nothing
errors. Those units are listed on every run.

Usage:
    python sources/id_grid.py --fetch    one 64 MB gzipped gpkg from Kontur
    python sources/id_grid.py            rebuild from data/raw/id_geo/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "id_geo")
GEO = os.path.join(ROOT, "data", "geo", "id")
DRAWN = os.path.join(GEO, "id_drawn.gpkg")
OUT = os.path.join(GEO, "id_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
          "kontur_datasets/kontur_population_ID_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ID_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ID_20231101.gpkg"

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). It is a WITHIN-unit weight only, so
# the level does not matter and the shape does. Note the vintage gap is real and expected:
# Kontur is 2023 and SP2010 is 2010, and Indonesia grew by ~40M in between.
CENSUS_POPULATION = 237_641_326
KONTUR_TOLERANCE = 0.40
EXPECTED_UNITS = 5212     # 5,122 kecamatan + 89 regencies + Kalimantan Utara


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz) or os.path.getsize(gz) < 60_000_000:
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=3600, stream=True,
                         headers={"User-Agent": "religiondots/1.0 (map research)"})
        r.raise_for_status()
        n = 0
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                n += len(chunk)
                if n % (16 << 20) < (1 << 20):
                    print(f"  {n:,} bytes")
        print(f"  {os.path.getsize(gz):,} bytes")
    print("gunzip …")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 22)
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
        if "--build" not in sys.argv:
            return

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(DRAWN):
        raise SystemExit(f"missing {DRAWN} -- run sources/id_geo.py first")

    print("reading the drawn tier …")
    units = gpd.read_file(DRAWN)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{DRAWN} has {len(units)} units, expected {EXPECTED_UNITS}")

    print("reading Kontur …")
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"  {len(hexes):,} hexes, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # Centroids are taken in the CRS Kontur tiled in, then reprojected — not the other way
    # round, or a hex shifts.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    print("spatial join …")
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes outside every drawn unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / float(pts[popcol].sum()):.2f}%)")
    print("     the sea edge, the Malaysian/PNG/Timorese overrun, Kalimantan Utara and the "
          "post-2010\n     territory this map does not paint. Dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # A unit absent from the placement layer draws NOTHING and nothing errors, so any unit
    # without a populated hex gets its own polygon as a single fallback cell. That degrades
    # to today's uniform placement for those units and to nothing worse.
    per = out.groupby("unit")["pop"].sum()
    have = set(per.index[per > 0])
    gaps = units[~units["unit"].isin(have)]
    if len(gaps):
        print(f"\n  {len(gaps)} drawn units have no populated hex — falling back to their "
              "own polygon:")
        for _, g in gaps.head(15).iterrows():
            print(f"     {g['unit']}  {g['name']}  ({g['level']})")
        if len(gaps) > 15:
            print(f"     … and {len(gaps) - 15} more")
        fb = gpd.GeoDataFrame({"unit": gaps["unit"].to_numpy(),
                               "pop": [1.0] * len(gaps)},
                              geometry=gaps.geometry.to_numpy(), crs=units.crs)
        out = gpd.GeoDataFrame(pd.concat([out, fb], ignore_index=True), crs=units.crs)

    still = set(units["unit"]) - set(out["unit"])
    if still:
        raise SystemExit(f"{len(still)} units still unplaced: {sorted(still)[:5]}")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs SP2010 {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "too much even for a weight -- check the download")
    print("     a 2023 modelled grid against a 2010 census; used only as a WITHIN-unit "
          "weight,\n     so the level does not matter and the shape does.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells over {out['unit'].nunique():,} units)")


if __name__ == "__main__":
    main()
