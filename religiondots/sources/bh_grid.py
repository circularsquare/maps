"""Bahrain: the placement layer, Kontur 400 m population hexagons inside the four governorates.

Writes data/geo/bh/bh_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry).

The units are geoBoundaries' governorates (`sources/bh_geo.py`), so `scatter.py` allocates each
governorate's dots from `sources/bh.py`'s 2020 counts before any weight is read; Kontur only decides
where inside a governorate. It is NOT calibrated: nothing finer than the governorate is used.
Kontur's density cap is handled at scatter time by `kontur_cap.apply` against `kontur_cap.csv`.

THE JOIN IS ON HEX CENTROIDS. Bahrain has no land border (the King Fahd Causeway's border island
is split with Saudi Arabia), so a hex whose centroid is outside every governorate is coast or
reclaimed land newer than the 2017 OSM lines, and is snapped to the nearest governorate within
`SNAP_KM`; beyond that it is dropped and counted.

## CHECKS

  * **the national ratio**, Kontur (November 2023) over the 2020 census, inside
    `EXPECTED_RATIO` +/- `TOLERANCE`;
  * **per governorate**, Kontur over the census count over the national ratio, inside
    `UNIT_BAND` (the witness that neither the name join nor the boundary file decides);
  * the dropped people, under `DROP_MAX` of Kontur's total.

Usage:
    python sources/bh_grid.py --fetch    Kontur BH (gzipped gpkg, 84 KB)
    python sources/bh_grid.py            rebuild from data/raw/bh/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bh")
GEO = os.path.join(ROOT, "data", "geo", "bh")
UNITS_GPKG = os.path.join(GEO, "bh_units.gpkg")
OUT = os.path.join(GEO, "bh_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BH_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BH_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BH_20231101.gpkg"

EXPECTED_UNITS = 4
EXPECTED_RATIO = 1.0
TOLERANCE = 0.2
UNIT_BAND = (0.67, 1.5)
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0)
DROP_MAX = 0.01
METRIC = "EPSG:32639"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000:
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(gz + ".part", gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg):
        fetch()
    if not os.path.exists(UNITS_GPKG):
        raise SystemExit(f"missing {UNITS_GPKG}; run sources/bh_geo.py first")

    hexes = read_layer(gpkg, "Kontur BH")
    popcol = "population"
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")
    units = gpd.read_file(UNITS_GPKG)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS_GPKG} has {len(units)} governorates, expected {EXPECTED_UNITS}")

    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    print(f"\n  hexes whose centroid is outside every governorate: {int(outside.sum()):,} "
          f"({pts.loc[outside, popcol].sum():,.0f} people)")
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km of a governorate: {int(m.sum()):>5,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    share = pts.loc[dropped, popcol].sum() / pts[popcol].sum()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({share:.3%})")
    if share > DROP_MAX:
        raise SystemExit(f"more than {DROP_MAX:.0%} of Kontur's people dropped")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].sum()
    drawn = dict(zip(units["unit"], units["pop"].astype(int)))
    missing = sorted(set(drawn) - set(per.index[per > 0]))
    if missing:
        raise SystemExit(f"governorates with no populated hex: {missing}")
    tot = float(per.sum())
    ratio = tot / sum(drawn.values())
    print(f"\n  Kontur {tot:,.0f} vs the 2020 census {sum(drawn.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census total disagree beyond the band")
    print("  per governorate: Kontur / census over the national ratio")
    bad = []
    for u in sorted(drawn, key=lambda x: -drawn[x]):
        rel = (per[u] / drawn[u]) / ratio
        print(f"      {u:<10} census {drawn[u]:>9,}  Kontur {per[u]:>10,.0f}  {rel:5.2f}")
        if not UNIT_BAND[0] <= rel <= UNIT_BAND[1]:
            bad.append(u)
    if bad:
        raise SystemExit(f"outside UNIT_BAND {UNIT_BAND}: {bad}")
    moved = 0.5 * sum(abs(per[u] / tot - drawn[u] / sum(drawn.values())) for u in drawn)
    print(f"  share of Kontur's people in a different governorate from the census: {moved:.1%}")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
