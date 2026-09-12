"""Kyrgyzstan — the placement layer: Kontur 400 m population hexagons, keyed to oblast.

Writes data/geo/kg/kg_hexes.gpkg.

**Kyrgyzstan needs this more than almost any country on the map.** 199,951 km² over nine units
is 22,217 km² per unit, and the country is 94% mountain: Naryn oblast is 45,200 km² of Tien
Shan pasture holding 316,182 people, almost all of them strung along the Naryn river and the
Torugart road, and Issyk-Kul's population is a ring around a lake that is itself 6,236 km² of
water. Spread either oblast's dots evenly over its polygon and most of the country's colour
lands on ice, rock and lake.

The other half of §8.2 bites as well: Chui oblast wraps Bishkek, so its people are a band along
the Chui valley against the Kazakh border while its southern third is the Kyrgyz Ala-Too.

THE JOIN IS SPATIAL, on hex CENTROIDS, so a hex on an oblast line belongs wholly to one side
and no population is double-counted. Hexes whose centroid falls outside every oblast are
dropped and reported: Kontur's KG extract overruns into Kazakhstan, Uzbekistan, Tajikistan and
China, and it also covers the Uzbek and Tajik **enclaves** inside Batken oblast, which are not
in COD-AB's Kyrgyz polygons and correctly fall out here.

Usage:
    python sources/kg_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/kg_grid.py            rebuild from data/raw/kg/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kg")
GEO = os.path.join(ROOT, "data", "geo", "kg")
OBLASTS = os.path.join(GEO, "kg_oblasts.gpkg")
LOOKUP = os.path.join(GEO, "kg_lookup.csv")
OUT = os.path.join(GEO, "kg_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KG_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KG_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KG_20231101.gpkg"

EXPECTED_OBLASTS = 9

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-oblast weight, so all
# that matters is that it is not wildly out.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 against
# the office's 1 January 2026 register, so it should read low by about two years of a
# population growing near 1.7% a year, and Kyrgyzstan's very large seasonal labour migration
# to Russia is counted by the register and not by building footprints, which widens it again.
NSC_POPULATION = 7_404_329
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
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(OBLASTS):
        raise SystemExit(f"missing {OBLASTS} — run sources/kg_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    obl = gpd.read_file(OBLASTS)
    if len(obl) != EXPECTED_OBLASTS:
        raise SystemExit(f"{OBLASTS} has {len(obl)} units, expected {EXPECTED_OBLASTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(obl.crs)
    hexes = hexes.to_crs(obl.crs)

    joined = gpd.sjoin(pts, obl[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every oblast: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's KG extract overruns into Kazakhstan, Uzbekistan, Tajikistan and "
          "China,\n     and covers the enclaves inside Batken; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=obl.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(obl["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"units whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_OBLASTS} units has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / NSC_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the office's {NSC_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the office disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-oblast weight, so the level does not matter and the\n"
          "     shape does.")

    print("\n  per-oblast Kontur/office ratio (the shape check):")
    lut = pd.read_csv(LOOKUP)
    nsc = dict(zip(lut["geo_id"], lut["pop"]))
    name = dict(zip(lut["geo_id"], lut["name"]))
    for u, r in sorted(per.iterrows(), key=lambda t: t[1]["sum"] / nsc[t[0]]):
        print(f"      {name[u]:<14} {int(r['size']):>7,} hexes   {r['sum'] / nsc[u]:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
