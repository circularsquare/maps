"""Yemen — the placement layer: Kontur 400 m population hexagons, keyed to governorate.

Writes data/geo/ye/ye_hexes.gpkg.

**Most of Yemen's area is empty and the people are in the western highlands and on the coasts.**
Hadramawt and Al Maharah alone are 227,687 km2, about half the country, and hold 5.6% of its
people; Sana'a City holds 11% on 508 km2. Drawn flat, Hadramawt's dots would cover the Empty
Quarter's southern edge instead of the Wadi Hadramawt towns and Mukalla.

The join is spatial on hex centroids, as `sources/iq_grid.py`: a hex on a governorate line
belongs wholly to one side. Hexes whose centroid falls outside every governorate are dropped and
reported.

**AND IT IS THE INDEPENDENT WITNESS ON THE POPULATION JOIN.** Kontur is modelled from building
footprints and settlement layers and knows nothing about the CSO's projection; the Task Force
table knows nothing about Kontur. Their per-governorate totals are rank-correlated over the 22
and asserted against every one of 5,000 random pairings, which a permuted p-code table cannot
survive.

Usage:
    python sources/ye_grid.py --fetch    one 5.7 MB gzipped gpkg from Kontur
    python sources/ye_grid.py            rebuild from data/raw/ye/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ye")
GEO = os.path.join(ROOT, "data", "geo", "ye")
GOVERNORATES = os.path.join(GEO, "ye_governorates.gpkg")
OUT = os.path.join(GEO, "ye_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_YE_20231101.gpkg.gz")
GZ_NAME = "kontur_population_YE_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_YE_20231101.gpkg"

EXPECTED_GOVERNORATES = 22
PTF_TOTAL = 34_879_018
# Kontur is a weight inside each governorate, not a census (§12, North Macedonia). Its inputs
# for Yemen are settlement rasters scaled to a national total from outside, so the level can
# sit well away from the Task Force's; the band only catches a wrong download.
KONTUR_TOLERANCE = 0.35

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
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
        raise SystemExit(f"missing {gz}; run with --fetch first")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")
    return gpkg


def main():
    import geopandas as gpd
    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    if not os.path.exists(GOVERNORATES):
        raise SystemExit(f"missing {GOVERNORATES}; run sources/ye_geo.py first")

    hexes = geo_checks.read_layer(gpkg, "Kontur YE")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    govs = gpd.read_file(GOVERNORATES)
    if len(govs) != EXPECTED_GOVERNORATES:
        raise SystemExit(f"{GOVERNORATES} has {len(govs)} governorates, expected "
                         f"{EXPECTED_GOVERNORATES}")

    # Centroid in the CRS the hexes were tiled in, then reproject the points.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(govs.crs)
    hexes = hexes.to_crs(govs.crs)
    joined = gpd.sjoin(pts, govs[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every governorate: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=govs.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(govs["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"governorates with no populated hex: {missing}")
    if (per["sum"] <= 0).any():
        raise SystemExit(f"governorates whose hexes sum to zero: "
                         f"{sorted(per.index[per['sum'] <= 0])}")
    print(f"  all {EXPECTED_GOVERNORATES} governorates have hexes: "
          f"{per['size'].min():,} to {per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / PTF_TOTAL
    print(f"\n  Kontur {tot:,.0f} against the Task Force's {PTF_TOTAL:,}: ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the Task Force disagree by {abs(ratio - 1):.0%}; check "
                         "the download")

    # ---- the join witness: two populations that share no input, per p-code ----
    census = dict(zip(govs["unit"], govs["pop"]))
    names = dict(zip(govs["unit"], govs["name"]))
    units = sorted(census)
    a = np.array([per.loc[u, "sum"] for u in units])
    b = np.array([census[u] for u in units], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(5000)])
    beaten = int((perm >= rho).sum())
    print(f"  Kontur against the Task Force per governorate: rho = {rho:+.3f} over "
          f"{len(units)}, and {beaten} of 5,000 random pairings reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("Kontur does not pin the p-code pairing of the population table; STOP")

    print("\n  per-governorate Kontur / Task Force ratio (the weight's shape, printed):")
    rows = sorted(((names[u], int(per.loc[u, "size"]), per.loc[u, "sum"] / census[u])
                   for u in units), key=lambda t: t[2])
    for nm, n, r in rows:
        print(f"    {nm:<14}{n:>8,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
