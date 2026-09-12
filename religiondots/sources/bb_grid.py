"""Barbados — the placement layer: Kontur 400 m population hexagons, keyed to parish.

Writes data/geo/bb/bb_hexes.gpkg.

**BARBADOS IS THE DENSEST COUNTRY ON THIS MAP AND THE PARISHES ARE ALL THE SAME SIZE**, so
this grid is doing something different from every other Caribbean one here. The eleven
parishes run 23.9 km² (St. Joseph) to 62.5 km² (St. Philip) — a 2.6-fold spread, against
Trinidad's 71-fold and the Bahamas' islands, which differ by four orders of magnitude. What
the grid is for here is not empty land and not unit size: it is **St. Michael**, which holds
69,604 of 226,193 tabulable people on 40.7 km², nearly all of them in Bridgetown and the
suburban belt along the south and west coasts. Uniform scatter would spread a third of the
country evenly over a parish that is mostly built up at one end.

**THE COASTLINE SNAP, AS IN THE BAHAMAS (§9ar) AND CAYMAN (§9at).** A plain `within` join
leaves hexes offshore of COD's coastline; the histogram prints on every run and anything
within 1 km is snapped to the nearest parish. Barbados is a single compact island with no
outlying cays, so nothing here is genuinely offshore and the snap is pure recovery.

**THE PER-PARISH RATIO IS NOT A SHAPE CHECK HERE — IT IS A COVERAGE CHECK.** `bb.csv` holds
the census's **tabulable** population, which is 81.4% of the estimated resident population
and runs from 74.6% to 96.1% between parishes (`sources/bb.py`). Kontur models the *actual*
population, so the ratio should come out near 1/0.814 ≈ 1.23 on average and HIGHER in the
worst-covered parishes. That is the expected signal rather than an error, and the band below
is set around it rather than around 1.0.

Usage:
    python sources/bb_grid.py --fetch    one ~180 KB gzipped gpkg from Kontur
    python sources/bb_grid.py            rebuild from data/raw/bb/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bb")
GEO = os.path.join(ROOT, "data", "geo", "bb")
PARISHES = os.path.join(GEO, "bb_parishes.gpkg")
OUT = os.path.join(GEO, "bb_hexes.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "bb.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BB_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BB_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BB_20231101.gpkg"

EXPECTED_PARISHES = 11

SNAP_M = 1000
UTM = 32621                     # UTM 21N covers Barbados

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census (§12,
# North Macedonia). **The comparison here is against the ESTIMATED RESIDENT population, not
# against what bb.csv holds** — bb.csv is the tabulable population, 18% short by
# construction, so comparing the grid to it would flag an 18% error that is really the
# census's undercount. Both are printed; the band is asserted on the resident figure.
TABULABLE_POPULATION = 226_193
ESTIMATED_RESIDENT = 277_821
KONTUR_TOLERANCE = 0.30

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000:
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
        raise SystemExit(f"missing {PARISHES} -- run sources/bb_geo.py first")

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

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
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
              "people).\n     Barbados is one compact island with no outlying cays, so "
              "this is COD's\n     coastline against a 400 m hex and nothing else.")

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
    ratio = tot / ESTIMATED_RESIDENT
    print(f"\n  Kontur {tot:,.0f} vs the census's ESTIMATED RESIDENT population "
          f"{ESTIMATED_RESIDENT:,} — ratio {ratio:.3f}")
    print(f"     (against the TABULABLE {TABULABLE_POPULATION:,} it is "
          f"{tot / TABULABLE_POPULATION:.3f}, which is the census's own 18% undercount\n"
          "      and not a disagreement about population)")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the resident population disagree by "
                         f"{abs(ratio - 1) * 100:.0f}%, which is too much for a weight")

    cen = pd.read_csv(NORM, dtype={"geo_id": str}, keep_default_na=False, na_values=[])
    cen = cen[(cen["geo_level"] == "parish") & (cen["source_category"] == "Total")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    print("\n  per-parish Kontur/tabulable ratio — expected ABOVE 1 and highest where the "
          "census\n  covered least; this is a coverage read, not a shape check:")
    rows = []
    for unit_id, row in per.iterrows():
        nm = par.loc[par["unit"] == unit_id, "name"].iloc[0]
        rows.append((row["sum"] / cen[unit_id], nm, int(row["size"]), cen[unit_id]))
    for r, nm, n, c in sorted(rows):
        print(f"      {nm:<16} {r:5.2f}x   {n:>5,} hexes   tabulable {c:>7,}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
