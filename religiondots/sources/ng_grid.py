"""Nigeria — the placement layer: Kontur 400 m population hexagons, keyed to state.

Writes data/geo/ng/ng_hexes.gpkg.

**THIS IS THE MOST EXTREME CASE OF THE ONE THE GRID EXISTS FOR.** 37 states over 924,000 km²
is **25,000 km² per unit**, four times Liberia's counties and the coarsest counting tier on
this map. Nigeria's people are nowhere near evenly spread inside a state:

  * **Lagos is 13.5 million people on 3,671 km²** and Niger State is 6.8 million on 71,934
    km². Spread Lagos's dots over its polygon and the whole state goes solid; spread Niger's
    evenly and the Gurara and Kaduna valleys look as settled as Minna.
  * **Borno, Yobe and Taraba are 44,000 to 72,000 km² each** with most of their people in a
    band along the roads, and the Sahel end of each is close to empty.
  * The Sokoto-Rima and Kano close-settled zones are among the densest rural districts in
    Africa and sit inside states that also hold thinly-peopled bush.

[[reference_kontur_resolution_floor]]'s test is whether the grid is finer than the counting
tier: at 25,000 km² per state against 400 m hexes it is finer by five orders of magnitude, so
Kontur does effectively all of the placement work here and none of the counting. That is the
comfortable direction, and the uncomfortable one is stated in `sources/ng.py`: what a dot's
COLOUR says is a state-wide share, so a Kano dot and a rural Kano dot are drawn from the same
mix even though the state's Christian minority is concentrated in the city.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS — a hex on a state line belongs wholly to one
side, so no hex is split and no population is double-counted. Hexes whose centroid falls
outside every state (Kontur's NG extract overruns into Niger, Chad, Cameroon and Benin, and
across Lake Chad) are dropped and reported.

Usage:
    python sources/ng_grid.py --fetch    one ~48 MB gzipped gpkg from Kontur
    python sources/ng_grid.py            rebuild from data/raw/ng/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ng")
GEO = os.path.join(ROOT, "data", "geo", "ng")
STATES = os.path.join(GEO, "ng_states.gpkg")
OUT = os.path.join(GEO, "ng_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_NG_20231101.gpkg.gz")
GZ_NAME = "kontur_population_NG_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_NG_20231101.gpkg"

EXPECTED_STATES = 37

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-state weight, so what
# matters is that it is not wildly out — assert the RELATIONSHIP, not the level.
#
# THE BAND IS WIDE ON PURPOSE HERE, and the reason is the country rather than the method. Every
# Nigerian population figure descends from the 2006 census, whose state totals are disputed
# (`sources/ng_geo.py`), and Kontur's building-footprint surface is one of the things people
# point at when they argue the census is wrong in either direction. So a disagreement between
# Kontur and COD-PS is not evidence that the download is broken, which is the only thing this
# check is for.
CODPS_2022 = 216_798_930
KONTUR_TOLERANCE = 0.40

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
        r = requests.get(GZ_URL, timeout=3600, stream=True, headers={"User-Agent": UA})
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
    if not os.path.exists(STATES):
        raise SystemExit(f"missing {STATES} — run sources/ng_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    states = gpd.read_file(STATES)
    if len(states) != EXPECTED_STATES:
        raise SystemExit(f"{STATES} has {len(states)} states, expected {EXPECTED_STATES}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(states.crs)
    hexes = hexes.to_crs(states.crs)

    joined = gpd.sjoin(pts, states[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every state: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's NG extract overruns into Niger, Chad, Cameroon and Benin; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=states.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(states["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"states with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"states whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_STATES} states has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CODPS_2022
    print(f"\n  Kontur {tot:,.0f} vs COD-PS 2022 {CODPS_2022:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and COD-PS disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")
    print("     used only as a WITHIN-state weight, so the level does not matter and the "
          "shape does.")

    print("\n  per-state Kontur/COD-PS ratio (the shape check):")
    cod = dict(zip(states["unit"], states["pop"]))
    nm = dict(zip(states["unit"], states["name"]))
    rows = [(nm[u], int(r["size"]), r["sum"] / cod[u]) for u, r in per.iterrows()]
    for name, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {name:<28} {n:>9,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
