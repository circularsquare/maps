"""Egypt — the placement layer: Kontur 400 m population hexagons, keyed to governorate.

Writes data/geo/eg/eg_hexes.gpkg.

**THIS IS THE COUNTRY THE GRID EXISTS FOR.** Egypt is 1,002,000 km2 and about 96% of it is
uninhabited desert; effectively the whole population lives on the Nile valley and delta, a
green ribbon a few kilometres wide for a thousand kilometres. The governorates are drawn to
contain that ribbon and then run east and west into nothing: **New Valley is 429,151 km2, 43%
of Egypt's land and 0.25% of its people**, and Matrouh, the Red Sea and the two Sinais are the
same shape. Spread a governorate's dots evenly over its polygon and Egypt's map is a Sahara
with a faint stripe, which is the opposite of the country.

Kontur removes that rather than patching it: an empty hex has no population and takes no dots,
so the Western Desert simply has no weight. `water.py` is not involved.

**AND IT MATTERS FOR WHAT THIS COUNTRY IS DRAWN FOR.** The Coptic concentration is an Upper
Egypt one, in Minya, Asyut and Sohag, and those three governorates are each a narrow strip of
valley inside a wide rectangle of desert. Drawn flat, the Christian share of Minya would be
painted mostly onto the Eastern Desert. Drawn on Kontur it lands on the towns along the river,
which is where the people it counts actually are.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate: a hex on a governorate
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every governorate (Kontur's EG extract overruns into Libya,
Sudan, Israel and Gaza) are dropped and reported.

Usage:
    python sources/eg_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/eg_grid.py            rebuild from data/raw/eg/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "eg")
GEO = os.path.join(ROOT, "data", "geo", "eg")
GOVERNORATES = os.path.join(GEO, "eg_governorates.gpkg")
OUT = os.path.join(GEO, "eg_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_EG_20231101.gpkg.gz")
GZ_NAME = "kontur_population_EG_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_EG_20231101.gpkg"

EXPECTED_GOVERNORATES = 27

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-governorate weight,
# so all that matters is that it is not wildly out.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 and the
# denominator this country is drawn on is CAPMAS's own estimate for 2026-01-01, so the grid
# should read LOW by roughly two years of Egyptian growth, which is about 3%.
CAPMAS_POPULATION = 108_528_518
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
        raise SystemExit(f"missing {GOVERNORATES} — run sources/eg_geo.py first")

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
    print("     Kontur's EG extract overruns into Libya, Sudan, Israel and Gaza; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=govs.crs)

    # Every governorate must get some hexes, or its dots fall back to an equal share over the
    # whole polygon — which in Egypt means over the Sahara.
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
    ratio = tot / CAPMAS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs CAPMAS {CAPMAS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and CAPMAS disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")
    print("     used only as a WITHIN-governorate weight, so the level does not matter and "
          "the\n     shape does.")

    # The per-governorate ratio is the one that would bias a share, and it is printed rather
    # than asserted: a governorate Kontur models badly gets its dots on a worse surface, it
    # does not get the wrong number of them (§9t).
    print("\n  per-governorate Kontur/CAPMAS ratio (the shape check):")
    cap = dict(zip(govs["unit"], govs["pop"]))
    names = dict(zip(govs["unit"], govs["name"]))
    rows = [(names[u], u, int(r["size"]), r["sum"] / cap[u]) for u, r in per.iterrows()]
    for nm, unit, n, r in sorted(rows, key=lambda t: t[3]):
        print(f"      {nm:<16} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
