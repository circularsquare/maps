"""Grenada — the placement layer: Kontur 400 m population hexagons, keyed to parish.

Writes data/geo/gd/gd_hexes.gpkg.

**THE GRID IS HERE FOR ST. GEORGE AND FOR CARRIACOU.** The two are opposite problems and
both need it:

  * **St. George holds 44,777 of 108,279 people on 65.8 km²** — 41% of the country in one
    polygon, because the fold in `sources/gd.py` puts the capital back into its parish. Its
    people are on the south-west coast, from St. George's town through Grand Anse to Point
    Salines; the parish's north-east is the ridge up to Mount Sinai. Uniform scatter would
    put a fifth of Grenada on a mountainside.
  * **Carriacou and Petite Martinique are one unit on two islands 2.4 km apart**, 4,747
    people over 33.8 km² of separate land. Uniform scatter inside a multipolygon spreads by
    area, which would put far too many of them on Petite Martinique's 2.4 km².

**THE COASTLINE SNAP, AS IN BARBADOS (§9au), THE BAHAMAS (§9ar) AND CAYMAN (§9at).** A plain
`within` join leaves hexes offshore of COD's coastline; the histogram prints on every run and
anything within 1 km is snapped to the nearest parish.

**THE RATIO IS EXPECTED ABOVE 1 AND THE REASON IS DATED, NOT DEMOGRAPHIC.** Kontur's extract
is the 2023-11-01 vintage and the census is April 2021; Grenada's population was still
growing between them, and Kontur models the *whole* population where `gd.csv` holds the
**non-institutional population in private dwellings**, 108,279 of a census 109,021. Both
push the same way.

Usage:
    python sources/gd_grid.py --fetch    one ~38 KB gzipped gpkg from Kontur
    python sources/gd_grid.py            rebuild from data/raw/gd/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gd")
GEO = os.path.join(ROOT, "data", "geo", "gd")
PARISHES = os.path.join(GEO, "gd_parishes.gpkg")
OUT = os.path.join(GEO, "gd_hexes.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "gd.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_GD_20231101.gpkg.gz")
GZ_NAME = "kontur_population_GD_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_GD_20231101.gpkg"

EXPECTED_PARISHES = 7

SNAP_M = 1000
UTM = 32620                     # UTM 20N covers Grenada

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census (§12,
# North Macedonia). The comparison is against the census's TOTAL population, which is the
# closest published figure to what a building-footprint model estimates.
DRAWN_UNIVERSE = 108_279
TOTAL_POPULATION = 109_021
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
    if not os.path.exists(PARISHES):
        raise SystemExit(f"missing {PARISHES} -- run sources/gd_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    par = gpd.read_file(PARISHES)
    if len(par) != EXPECTED_PARISHES:
        raise SystemExit(f"{PARISHES} has {len(par)} parishes, "
                         f"expected {EXPECTED_PARISHES}")

    # §8.2: the grid has to be FINER than the tier it weights, or it is doing nothing
    # (Saint Vincent, §9ac). Measured rather than assumed.
    hex_km2 = float((hexes.to_crs(UTM).area / 1e6).median())
    unit_km2 = float((par.to_crs(UTM).area / 1e6).median())
    print(f"  hex {hex_km2:.2f} km2 against a median parish of {unit_km2:.1f} km2 "
          f"— {unit_km2 / hex_km2:.0f}x finer")
    if hex_km2 >= unit_km2:
        raise SystemExit("the placement grid is coarser than the counting tier, so it "
                         "cannot weight it")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in,
    # then reproject the POINTS -- reprojecting first and taking the centroid after
    # moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(par.crs)
    hexes = hexes.to_crs(par.crs)

    joined = gpd.sjoin(pts, par[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit = joined["unit"].copy()        # NaN where the hex is outside every parish

    outside = joined["unit"].isna().to_numpy()
    adrift = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every parish: {int(outside.sum()):,} "
          f"({adrift:,.0f} people, {100.0 * adrift / pts[popcol].sum():.3f}%)")

    if outside.any():
        m_pts = pts[outside].to_crs(UTM)
        near = gpd.sjoin_nearest(m_pts, par.to_crs(UTM)[["unit", "geometry"]],
                                 how="left", distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")].reindex(m_pts.index)

        print("     distance from one of those to the nearest parish:")
        edges = [0, 100, 250, 500, SNAP_M, float("inf")]
        names = ["<100 m", "100-250 m", "250-500 m", f"500 m-{SNAP_M / 1000:g} km",
                 f">{SNAP_M / 1000:g} km"]
        for lo, hi, lab in zip(edges[:-1], edges[1:], names):
            sel = near[(near["dist_m"] >= lo) & (near["dist_m"] < hi)]
            if len(sel):
                print(f"       {lab:<14} {len(sel):>4} hexes  "
                      f"{sel[popcol].sum():>8,.0f} people")

        snap = near["dist_m"] <= SNAP_M
        idx = near.index[snap]
        unit.loc[idx] = near.loc[idx, "unit"]
        print(f"     snapped to the nearest parish within {SNAP_M:,} m: "
              f"{int(snap.sum()):,} hexes ({float(near.loc[snap, popcol].sum()):,.0f} "
              "people).\n     Grenada's only inhabited outliers are Carriacou and Petite "
              "Martinique, which\n     both have their own polygons, so this is COD's "
              "coastline against a 400 m hex.")

    # NaN, not None — see sources/bs_grid.py for what `!= None` did here.
    keep = unit.notna().to_numpy()
    lost = float(pts.loc[~keep, popcol].sum())
    print(f"     still outside after the snap: {int((~keep).sum()):,} hexes "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")

    out = gpd.GeoDataFrame(
        {"unit": unit[keep].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep].to_numpy(), crs=par.crs)

    got, want = set(out["unit"].unique()), set(par["unit"])
    if got != want or out["unit"].isna().any():
        raise SystemExit(f"hex layer carries units {sorted(got, key=str)}, expected "
                         f"{sorted(want)}")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"parishes whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_PARISHES} parishes has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / TOTAL_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the census's TOTAL population "
          f"{TOTAL_POPULATION:,} — ratio {ratio:.3f}")
    print(f"     (against the drawn universe {DRAWN_UNIVERSE:,} it is "
          f"{tot / DRAWN_UNIVERSE:.3f}; Kontur is a 2023 vintage against an April 2021\n"
          "      census, and it models everyone rather than only private dwellings)")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census population disagree by "
                         f"{abs(ratio - 1) * 100:.0f}%, which is too much for a weight")

    cen = pd.read_csv(NORM, dtype={"geo_id": str}, keep_default_na=False, na_values=[])
    cen = cen[(cen["geo_level"] == "parish") & (cen["source_category"] == "TOTAL")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    print("\n  per-parish Kontur/census ratio. Only the shape WITHIN a parish is used, so "
          "no parish\n  gets the wrong number of dots whatever this says (§9t):")
    rows = []
    for unit_id, row in per.iterrows():
        nm = par.loc[par["unit"] == unit_id, "name"].iloc[0]
        rows.append((row["sum"] / cen[unit_id], nm, int(row["size"]), cen[unit_id]))
    for r, nm, n, c in sorted(rows):
        print(f"      {nm:<32} {r:5.2f}x   {n:>5,} hexes   census {c:>7,}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
