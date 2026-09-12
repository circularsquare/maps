"""Jordan — the placement layer: Kontur 400 m population hexagons, keyed to governorate.

Writes data/geo/jo/jo_hexes.gpkg.

**JORDAN IS THE SAME SHAPE OF PROBLEM AS EGYPT AND MOST OF IT IS DESERT.** Three governorates
— Ma'an, Mafraq and Aqaba — are **66,288 km2, 74.7% of the country's land, and 9.5% of its
people**; the badia east and south of the highlands is empty. Spread a governorate's dots
evenly over its polygon and Jordan draws as a rectangle of desert with a faint western edge,
which is the opposite of where anybody lives.

Kontur removes that rather than patching it: an empty hex has no population and takes no dots,
so the badia simply has no weight. `water.py` is not involved; the Dead Sea sits on the
boundary rather than inside a governorate, and Kontur has no hexes on it either way.

**AND IT MATTERS FOR WHAT THIS COUNTRY IS DRAWN FOR.** Jordan's Christians are a Balqa,
Madaba, Ajloun and Amman population, in Fuheis, Husn, Ajloun's villages and the Amman
suburbs — all of them on the highland strip. Mafraq's 686,800 people are almost all in Mafraq
town and the northern villages, inside 2% of its polygon. Drawn flat, the difference between
Madaba and Mafraq would be painted mostly onto empty basalt.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate: a hex on a governorate
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every governorate (Kontur's JO extract overruns into Syria,
Iraq, Saudi Arabia, Israel and the West Bank) are dropped and reported.

Usage:
    python sources/jo_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/jo_grid.py            rebuild from data/raw/jo/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "jo")
GEO = os.path.join(ROOT, "data", "geo", "jo")
GOVERNORATES = os.path.join(GEO, "jo_governorates.gpkg")
OUT = os.path.join(GEO, "jo_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_JO_20231101.gpkg.gz")
GZ_NAME = "kontur_population_JO_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_JO_20231101.gpkg"

EXPECTED_GOVERNORATES = 12

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-governorate weight,
# so all that matters is that it is not wildly out.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 and the
# denominator this country is drawn on is DOS's own estimate for end-2025, so the grid should
# read LOW by roughly two years of Jordanian growth, which is about 2%.
DOS_POPULATION = 11_937_000
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

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(GOVERNORATES):
        raise SystemExit(f"missing {GOVERNORATES} — run sources/jo_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    govs = gpd.read_file(GOVERNORATES)
    if len(govs) != EXPECTED_GOVERNORATES:
        raise SystemExit(f"{GOVERNORATES} has {len(govs)} governorates, "
                         f"expected {EXPECTED_GOVERNORATES}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(govs.crs)
    hexes = hexes.to_crs(govs.crs)

    joined = gpd.sjoin(pts, govs[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every governorate: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's JO extract overruns into Syria, Iraq, Saudi Arabia, Israel and the "
          "West Bank; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=govs.crs)

    # Every governorate must get some hexes, or its dots fall back to an equal share over the
    # whole polygon — which in Jordan means over the badia.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(govs["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"governorates with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"governorates whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_GOVERNORATES} governorates has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / DOS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs DOS {DOS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and DOS disagree by {abs(ratio - 1) * 100:.0f}%, which is "
                         "too much for a weight — check the download")
    print("     used only as a WITHIN-governorate weight, so the level does not matter and "
          "the\n     shape does.")

    # The per-governorate ratio is the one that would bias a share, and it is printed rather
    # than asserted: a governorate Kontur models badly gets its dots on a worse surface, it
    # does not get the wrong number of them (§9t).
    print("\n  per-governorate Kontur/DOS ratio (the shape check):")
    dos = dict(zip(govs["unit"], govs["pop"]))
    names = dict(zip(govs["unit"], govs["name"]))
    rows = [(names[u], u, int(r["size"]), r["sum"] / dos[u]) for u, r in per.iterrows()]
    for nm, unit, n, r in sorted(rows, key=lambda t: t[3]):
        print(f"      {nm:<10} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
