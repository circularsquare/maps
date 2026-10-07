"""Taiwan: the 368 townships and districts, and Kontur's hexes keyed to them.

    python sources/tw_geo.py --fetch    MOI's township shapefile (GitHub mirror, ~18 MB)
    python sources/tw_geo.py            build data/geo/tw/

Writes
    data/geo/tw/tw_units.gpkg      368 polygons: `unit` (MOI TOWNCODE), `key` ("<county>|<town>")
    data/geo/tw/tw_lookup.csv      key, unit, county, town
    data/geo/tw/tw_hexes.gpkg      Kontur 2023 hexes: `unit`, `pop`

THE UNITS. MOI's 鄉(鎮、市、區)界線 (National Land Surveying and Mapping Center, TOWN_MOI). The
official download (tgos.tw, data.gov.tw dataset 7441) answers 403 outside Taiwan, as it did for
religiondots; kiang/taiwan_basecode on GitHub mirrors the NLSC open-data release of 2023-03-17
(TOWN_MOI_1120317) unchanged, with NLSC's own change list beside it. Taiwan's 368 townships have
not changed since the 2014 Taoyuan upgrade, so the 2023 file has the 2020 census's units; the join
proves it (368 both ways, by county and township name in the same characters).

PLACEMENT. Kontur 2023 (religiondots' TW extract, read in place), each hex to the township its
centroid falls in (sources/_grid.py). Hexes whose centroid falls just offshore of a coastal
township are snapped to the nearest township within SNAP_M (religiondots playbook: hex centroids
fall offshore of island units); further out they are dropped and printed, which is where Kontur's
TW extract runs into Xiamen beside Kinmen.
"""
import csv
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sources"))
RAW = ROOT / "data" / "raw" / "tw"
GEO = ROOT / "data" / "geo" / "tw"
NORM = ROOT / "data" / "normalized" / "tw.csv"
RD_KONTUR_TW = ROOT.parent / "religiondots" / "data" / "geo" / "tw" / "kontur_population_TW_20231101.gpkg"
MIRROR = "https://raw.githubusercontent.com/kiang/taiwan_basecode/gh-pages/city/shp/20230317/"
STEM = "TOWN_MOI_1120317"
N_TOWNS = 368
SNAP_M = 1000
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for ext in ("shp", "shx", "dbf", "prj"):
        p = RAW / f"{STEM}.{ext}"
        if p.exists() and p.stat().st_size > 100:
            continue
        req = urllib.request.Request(MIRROR + f"{STEM}.{ext}", headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            p.write_bytes(r.read())
        print(f"  {p.name}: {p.stat().st_size:,} bytes")


def main():
    if "--fetch" in sys.argv:
        fetch()
    import geopandas as gpd
    import pandas as pd
    import _grid

    towns = gpd.read_file(RAW / f"{STEM}.shp", encoding="utf-8")
    if len(towns) != N_TOWNS or towns["TOWNCODE"].duplicated().any():
        raise SystemExit(f"{len(towns)} township features (expected {N_TOWNS}, unique codes)")
    towns["key"] = towns["COUNTYNAME"].str.strip() + "|" + towns["TOWNNAME"].str.strip()
    if towns["key"].duplicated().any():
        raise SystemExit("duplicate county|town names in the shapefile")

    df = pd.read_csv(NORM)
    pop = (df[(df["geo_level"] == "town") & (df["question"] == "earliest")]
           .groupby("geo_id")["pop6"].first())
    a, b = set(pop.index), set(towns["key"])
    if a != b:
        raise SystemExit(f"join: {len(a - b)} census townships with no polygon {sorted(a - b)[:10]}; "
                         f"{len(b - a)} polygons with no census row {sorted(b - a)[:10]}")
    print(f"  join: {len(a)} townships, both ways, by county|town name")

    towns = towns.rename(columns={"TOWNCODE": "unit"})
    units = towns[["unit", "key", "COUNTYNAME", "TOWNNAME", "geometry"]].copy()
    units["unit"] = units["unit"].astype(str)
    GEO.mkdir(parents=True, exist_ok=True)
    units.to_file(GEO / "tw_units.gpkg", layer="units", driver="GPKG")
    with open(GEO / "tw_lookup.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "unit", "county", "town"])
        for r in units.itertuples():
            w.writerow([r.key, r.unit, r.COUNTYNAME, r.TOWNNAME])

    # Kontur: centroid join, then snap the offshore centroids of coastal hexes
    hexes = gpd.read_file(RD_KONTUR_TW)
    if len(hexes) == 0:
        raise SystemExit("Kontur TW extract has ZERO features")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    proj = units.to_crs(3826)          # TWD97 / TM2 zone 121, metres
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(3826)
    j = gpd.sjoin(pts, proj[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["unit"].isna()
    near = gpd.sjoin_nearest(pts[out], proj[["unit", "geometry"]], how="left",
                             max_distance=SNAP_M, distance_col="d")
    near = near[~near.index.duplicated(keep="first")]
    snapped = near["unit"].notna()
    j.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
    dropped = j["unit"].isna()
    print(f"  Kontur TW: {len(hexes):,} hexes, {pts['pop'].sum():,.0f} people; "
          f"{int(snapped.sum()):,} offshore centroids snapped ({pts.loc[near.index[snapped], 'pop'].sum():,.0f} people), "
          f"{int(dropped.sum()):,} hexes dropped ({pts.loc[dropped, 'pop'].sum():,.0f} people)")
    if dropped.any():
        d = pts[dropped].to_crs(4326)
        print(f"  dropped hexes lie in lon {d.geometry.x.min():.2f}-{d.geometry.x.max():.2f}, "
              f"lat {d.geometry.y.min():.2f}-{d.geometry.y.max():.2f}; biggest "
              f"{d['pop'].nlargest(3).round().tolist()}")
    keep = ~dropped
    layer = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].astype(str).to_numpy(),
                              "pop": pts.loc[keep, "pop"].to_numpy()},
                             geometry=hexes.geometry[keep.to_numpy()].to_numpy(),
                             crs=hexes.crs).to_crs(4326)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(set(units["unit"]) - set(per.index[per > 0]))
    if empty:
        raise SystemExit(f"{len(empty)} townships with no populated hex: {empty}")

    # Kontur against the census per township (the helper's band and shuffle control)
    key2unit = dict(zip(units["key"], units["unit"]))
    census = {key2unit[k]: int(v) for k, v in pop.items()}
    rows = [(u, c, float(per.get(u, 0.0))) for u, c in census.items()]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    norm = sorted((k / c / ratio, u) for u, c, k in rows)
    name = dict(zip(units["unit"], units["key"]))
    print(f"  Kontur / census 6+ nationally {ratio:.3f}; per township normalised p10 "
          f"{norm[len(norm) // 10][0]:.2f} median {norm[len(norm) // 2][0]:.2f} "
          f"p90 {norm[9 * len(norm) // 10][0]:.2f}")
    print("  lowest: " + ", ".join(f"{name[u]} {r:.2f}" for r, u in norm[:6]))
    print("  highest: " + ", ".join(f"{name[u]} {r:.2f}" for r, u in norm[-6:]))
    import math
    import random
    lc = [math.log(c) for _, c, _ in rows]
    lk = [math.log(k) for _, _, k in rows]

    def pear(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        num = sum((p - mx) * (q - my) for p, q in zip(x, y))
        return num / math.sqrt(sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("the join is not carrying information")
    bad = [(name[u], round(x, 2)) for x, u in norm if not (1 / 3 <= x <= 3)]
    print(f"  {len(bad)} townships outside a factor of 3: {bad}")

    import numpy as np
    area = units.to_crs(3826).area / 1e6
    print(f"  township area median {np.median(area):.1f} km2 "
          f"({np.median(area) / 0.74:.0f} hexes); smallest {area.min():.2f} km2")
    layer.to_file(GEO / "tw_hexes.gpkg", layer="hexes", driver="GPKG")
    print(f"  wrote {GEO / 'tw_hexes.gpkg'} ({len(layer):,} hexes)")


if __name__ == "__main__":
    main()
