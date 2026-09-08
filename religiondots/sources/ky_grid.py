"""Cayman Islands — the placement layer: Kontur 400 m population hexagons, keyed to district.

Writes data/geo/ky/ky_hexes.gpkg.

**THIS IS THE THINNEST KONTUR EXTRACT ON THE MAP: 392 HEXES FOR THE WHOLE COUNTRY.** The
Cayman Islands are 264 km² and a hex is 0.67 km², so there is not much to work with — but
the counting units are 8 to 89 km², which means 12 to 119 hexes each, and the grid is
therefore still FINER than the tier it is weighting. That is the test Saint Vincent failed
(§9ac, where the grid was coarser than the enumeration districts and was dropped), and this
passes it.

It earns its place on Grand Cayman's shape rather than on emptiness: George Town holds
33,898 of 68,811 people, and inside it the population is the Seven Mile Beach corridor and
the town, not the North Sound mangrove that makes up much of the polygon's area.

**THE COASTLINE SNAP, AS IN THE BAHAMAS (§9ar).** A plain `within` join leaves 97 hexes and
4,361 modelled people — **6.29%** — outside every district, and the measured distance to the
nearest district is:

    <100 m       27 hexes   1,404 people
    100-250 m    43 hexes   2,685 people
    250-500 m    27 hexes     272 people
    beyond        0 hexes       0 people

**Nothing at all is genuinely offshore.** All of it is COD-AB's coastline against a 400 m
hex, and Caymanian settlement is coastal, so dropping it would tilt every district's dots
inland. Snapped to the nearest district within 1 km, and the histogram prints on every run.

**THE PER-DISTRICT RATIO IS BAD AND IT IS THE BOUNDARIES, NOT KONTUR.** See
`sources/ky_geo.py`: COD's North Side polygon reaches south across the island over Bodden
Town's eastern villages, so North Side reads about twice its census population and Bodden
Town and West Bay read short. COD was still chosen — it loses less population to the error
than OSM does, by better than two to one — but the ratio table below is the visible symptom
and is printed rather than smoothed. Only the within-district shape is used, so no district
gets the wrong NUMBER of dots (§9t).

Usage:
    python sources/ky_grid.py --fetch    one ~31 KB gzipped gpkg from Kontur
    python sources/ky_grid.py            rebuild from data/raw/ky/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ky")
GEO = os.path.join(ROOT, "data", "geo", "ky")
DISTRICTS = os.path.join(GEO, "ky_districts.gpkg")
OUT = os.path.join(GEO, "ky_hexes.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "ky.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KY_20231101.gpkg"

EXPECTED_DISTRICTS = 6

# The coastline snap; see the docstring. 1 km sits well past the measured maximum (500 m),
# so the threshold is in empty space rather than through the data.
SNAP_M = 1000
UTM = 32617                     # UTM 17N covers the whole country

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-district weight.
CENSUS_POPULATION = 68_811      # ESO's tabular count, which is what ky.csv holds
KONTUR_TOLERANCE = 0.30

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 50_000:
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
    if not os.path.exists(DISTRICTS):
        raise SystemExit(f"missing {DISTRICTS} -- run sources/ky_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    dis = gpd.read_file(DISTRICTS)
    if len(dis) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{DISTRICTS} has {len(dis)} districts, "
                         f"expected {EXPECTED_DISTRICTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(dis.crs)
    hexes = hexes.to_crs(dis.crs)

    joined = gpd.sjoin(pts, dis[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit = joined["unit"].copy()        # NaN where the hex is outside every district

    outside = joined["unit"].isna().to_numpy()
    adrift = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every district: {int(outside.sum()):,} "
          f"({adrift:,.0f} people, {100.0 * adrift / pts[popcol].sum():.3f}%)")

    m_pts = pts[outside].to_crs(UTM)
    near = gpd.sjoin_nearest(m_pts, dis.to_crs(UTM)[["unit", "geometry"]],
                             how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")].reindex(m_pts.index)

    print("     distance from one of those to the nearest district:")
    edges = [0, 100, 250, 500, SNAP_M, float("inf")]
    names = ["<100 m", "100-250 m", "250-500 m", f"500 m-{SNAP_M / 1000:g} km",
             f">{SNAP_M / 1000:g} km"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], names):
        sel = near[(near["dist_m"] >= lo) & (near["dist_m"] < hi)]
        if len(sel):
            print(f"       {lab:<14} {len(sel):>4} hexes  {sel[popcol].sum():>8,.0f} "
                  "people")

    snap = near["dist_m"] <= SNAP_M
    idx = near.index[snap]
    unit.loc[idx] = near.loc[idx, "unit"]
    snapped = float(near.loc[snap, popcol].sum())
    print(f"     snapped to the nearest district within {SNAP_M:,} m: "
          f"{int(snap.sum()):,} hexes ({snapped:,.0f} people) — COD's coastline against a\n"
          "     400 m hex, and Caymanian settlement IS the coast (§9ar).")

    # NaN, not None — see sources/bs_grid.py, where writing this as `!= None` kept two
    # null-unit hexes and the only symptom was one line of scatter.py output.
    keep = unit.notna().to_numpy()
    lost = float(pts.loc[~keep, popcol].sum())
    print(f"     still outside after the snap: {int((~keep).sum()):,} hexes "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")

    out = gpd.GeoDataFrame(
        {"unit": unit[keep].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep].to_numpy(), crs=dis.crs)

    got, want = set(out["unit"].unique()), set(dis["unit"])
    if got != want or out["unit"].isna().any():
        raise SystemExit(f"hex layer carries units {sorted(got, key=str)}, expected "
                         f"{sorted(want)}")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DISTRICTS} districts has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2021 census; used only as a\n"
          "     WITHIN-district weight.")

    cen = pd.read_csv(NORM, dtype={"geo_id": str}, keep_default_na=False, na_values=[])
    cen = cen[(cen["geo_level"] == "district") & (cen["source_category"] == "Total")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    print("\n  per-district Kontur/census ratio — WIDE, and it is the BOUNDARIES:")
    rows = []
    for unit_id, row in per.iterrows():
        nm = dis.loc[dis["unit"] == unit_id, "name"].iloc[0]
        rows.append((row["sum"] / cen[unit_id], nm, int(row["size"]), cen[unit_id]))
    for r, nm, n, c in sorted(rows):
        print(f"      {nm:<16} {r:5.2f}x   {n:>4,} hexes   census {c:>7,}")
    print("      COD's North Side polygon reaches south over Bodden Town's eastern\n"
          "      villages (sources/ky_geo.py), which is why North Side reads high and\n"
          "      Bodden Town and West Bay read short. Only the shape is used.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
