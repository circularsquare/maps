"""Tajikistan: the placement layer, Kontur 400 m population hexagons keyed to the five units.

Writes data/geo/tj/tj_hexes.gpkg. Copied in shape from `sources/az_grid.py`; `sources/tj.md` §7 is
the record.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every unit is snapped to the nearest
unit within `SNAP_KM` when it is not inside a neighbouring country on Natural Earth (Afghanistan,
China, Kyrgyzstan, Uzbekistan), and dropped otherwise (`playbooks/geography.md`: across a land
border a hex outside the units is the neighbour's town).

THREE CHECKS AGAINST THE CENSUS: the national ratio (Kontur over the 2020 permanent population,
inside `EXPECTED_RATIO` +/- `TOLERANCE`); every unit's own ratio over the national one inside
`UNIT_BAND` (five units are too few for a rank test with any power, so the band is the check); and
every unit holds a populated hex.

Usage:
    python sources/tj_grid.py --fetch    Kontur TJ (3.0 MB gzipped) into data/raw/tj/
    python sources/tj_grid.py            rebuild from data/raw/tj/
"""

import gzip
import json
import os
import shutil
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tj")
GEO = os.path.join(ROOT, "data", "geo", "tj")
UNITS = os.path.join(GEO, "tj_units.gpkg")
OUT = os.path.join(GEO, "tj_hexes.gpkg")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TJ_20231101.gpkg.gz")
GZ = os.path.join(RAW, "kontur_population_TJ_20231101.gpkg.gz")
GPKG = os.path.join(RAW, "kontur_population_TJ_20231101.gpkg")

NEIGHBOURS = ("AFG", "CHN", "KGZ", "UZB")
SNAP_KM = 2.0
EXPECTED_RATIO = 1.0
TOLERANCE = 0.25
UNIT_BAND = (0.75, 1.33)
# Dushanbe reads 1.46 on today's OSM line (1.72 on geoBoundaries' 2017 polygon, which held 180 km2 of
# suburb; sources/tj_geo.py). Kontur's capital is high, not the line: the city put itself at 851,300
# in mid-2019 and the census at 948,251, while Kontur's 2023 surface holds 1.46 million inside 197 km2.
# The census count is what is drawn; Kontur only places it inside the city.
UNIT_PINNED = {"TJ-DU": (1.35, 1.55)}
METRIC = "EPSG:32642"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GZ):
        req = urllib.request.Request(GZ_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=900) as r:
            data = r.read()
        with open(GZ + ".part", "wb") as fh:
            fh.write(data)
        os.replace(GZ + ".part", GZ)
    if not os.path.exists(GPKG):
        with gzip.open(GZ, "rb") as src, open(GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(GPKG + ".part", GPKG)
    with open(GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{GPKG} is not a GeoPackage")


def ne_layer(path, keep):
    import geopandas as gpd
    from shapely.geometry import shape

    d = json.load(open(path, encoding="utf-8"))
    rows = [f for f in d["features"] if keep(f["properties"])]
    return gpd.GeoDataFrame({"a3": [f["properties"].get("ADM0_A3") for f in rows]},
                            geometry=[shape(f["geometry"]) for f in rows], crs=4326)


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    if "--fetch" in sys.argv or not os.path.exists(GPKG):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/tj_geo.py first")

    hexes = read_layer(GPKG, "Kontur TJ")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    print(f"Kontur hexes: {len(hexes):,}, population {hexes[popcol].sum():,.0f}")
    units = gpd.read_file(UNITS)
    if len(units) != 5:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected 5")

    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)

    outside = j["unit"].isna()
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")].reindex(pts.index[outside])
        ne = ne_layer(NE, lambda p: p.get("ADM0_A3") in NEIGHBOURS).to_crs(units.crs)
        w = gpd.sjoin(pts.loc[outside], ne[["a3", "geometry"]], how="left", predicate="within")
        w = w[~w.index.duplicated(keep="first")].reindex(pts.index[outside])
        a3 = w["a3"].fillna("none")
        snap = (a3 == "none") & (near["d"] <= SNAP_KM * 1000)
        tab = pd.DataFrame({"rule": np.where(snap, "snap", "drop"), "ne": a3,
                            "pop": pts.loc[outside, "pop"]})
        print(f"\n  hexes outside every unit: {int(outside.sum()):,}, "
              f"{pts.loc[outside, 'pop'].sum():,.0f} people; by rule and Natural Earth country:")
        for (r_, n_), r in tab.groupby(["rule", "ne"])["pop"].agg(["size", "sum"]).iterrows():
            print(f"      {r_:<5} {n_:<5} {int(r['size']):>6,} hexes  {r['sum']:>10,.0f} people")
        j.loc[near.index[snap], "unit"] = near.loc[snap, "unit"]
    keep = j["unit"].notna()
    print(f"  dropped: {int((~keep).sum()):,} hexes, {pts.loc[~keep, 'pop'].sum():,.0f} people")

    out = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    missing = [names[u] for u in est if u not in per.index or per.loc[u, "sum"] <= 0]
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")

    tot = float(per["sum"].sum())
    ratio = tot / sum(est.values())
    print(f"\n  Kontur {tot:,.0f} vs census 2020 permanent {sum(est.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")
    print("  per unit, Kontur / census over the national ratio:")
    bad = []
    for u in sorted(est, key=lambda x: -est[x]):
        rel = (per.loc[u, "sum"] / est[u]) / ratio
        print(f"      {names[u]:<38} census {est[u]:>10,}  Kontur {per.loc[u, 'sum']:>11,.0f}  "
              f"{int(per.loc[u, 'size']):>7,} hexes  {rel:5.2f}")
        lo, hi = UNIT_PINNED.get(u, UNIT_BAND)
        if not lo <= rel <= hi:
            bad.append((names[u], round(rel, 3)))
    if bad:
        raise SystemExit(f"units outside {UNIT_BAND}: {bad}")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
