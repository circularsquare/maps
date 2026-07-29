"""
Scatter ethnic-origin dots inside Dissemination Area polygons for a region.

Reads:
  - data/raw/origins_<region>.csv           (from fetch_data.py: dguid,cid,origin,count)
  - data/shapefiles/lda_2021/lda_000b21a_e.shp  (StatCan 2021 DA cartographic boundaries)
Writes:
  - data/processed/dots_<region>_1per100.geojson   (one point per 100 responses)

1 dot = 100 responses (not persons): a person reporting multiple origins contributes
to several origin piles, exactly like the US map. Fractional remainders carry forward
across DAs so small origins aren't dropped.

Usage:
    python build_dots.py --region Territories_Territoires
    python build_dots.py --region Ontario
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import geopandas as gpd
import shapely

from labels import canonical

HERE = Path(__file__).parent
SHP = HERE / "data" / "shapefiles" / "lda_2021" / "lda_000b21a_e.shp"
PARQUET = HERE / "data" / "shapefiles" / "da_2021.parquet"  # fast cache (convert_boundaries.py)
COLORS_CSV = HERE / "ancestry_colors_ca.csv"
SCALE = 100
TOTAL_ID = 1698

# Province/territory codes (PRUID) per download region — lets pyogrio read only the
# DAs we need instead of the full 57,932-feature national file.
REGION_PRUIDS = {
    "Territories": ["60", "61", "62"],
    "Atlantic": ["10", "11", "12", "13"],
    "Quebec": ["24"],
    "Ontario": ["35"],
    "Prairies": ["46", "47", "48"],
    "BC": ["59"],
}


def random_points_in_polygon(polygon, n: int) -> list:
    """n random points inside polygon (numpy + shapely vectorized contains).
    Copied from ../scatter_dots.py to keep the Canada pipeline self-contained."""
    if n == 0:
        return []
    minx, miny, maxx, maxy = polygon.bounds
    pts: list = []
    needed = n
    while needed > 0:
        batch = max(needed * 4, 32)
        xs = np.random.uniform(minx, maxx, batch)
        ys = np.random.uniform(miny, maxy, batch)
        mask = shapely.contains_xy(polygon, xs, ys)
        vx, vy = xs[mask], ys[mask]
        take = min(needed, len(vx))
        if take > 0:
            pts.extend(zip(vx[:take].tolist(), vy[:take].tolist()))
            needed -= take
    return pts if pts else [(polygon.centroid.x, polygon.centroid.y)] * n


def load_group_map() -> dict:
    """origin label -> color group, from ancestry_colors_ca.csv if present (else empty)."""
    gm = {}
    if COLORS_CSV.exists():
        with open(COLORS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [n.strip() for n in reader.fieldnames]
            for row in reader:
                gm[row["label"].strip()] = row.get("group", "").strip()
    return gm


def load_origins(region: str) -> dict:
    """dguid -> list of (origin, count), excluding the Total row.
    region='all' merges every data/raw/origins_*.csv."""
    if region == "all":
        paths = sorted((HERE / "data" / "raw").glob("origins_*.csv"))
    else:
        paths = [HERE / "data" / "raw" / f"origins_{region}.csv"]
    by_da = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["cid"]) == TOTAL_ID:
                    continue
                c = int(row["count"])
                if c > 0:
                    by_da[row["dguid"]].append((canonical(row["origin"]), c))
    return by_da


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--region", help="One region key (Ontario, Atlantic, ...)")
    g.add_argument("--all", action="store_true", help="Build all downloaded regions into dots_all")
    args = ap.parse_args()
    region = "all" if args.all else args.region

    by_da = load_origins(region)
    group_map = load_group_map()
    print(f"{region}: {len(by_da)} DAs with origin data")

    pruids = None if region == "all" else REGION_PRUIDS.get(region)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if PARQUET.exists():
            gdf = gpd.read_parquet(PARQUET)
            if pruids:
                gdf = gdf[gdf["PRUID"].isin(pruids)]
        else:
            print("  (no parquet cache — reading .shp directly, ~4 min; run convert_boundaries.py to speed this up)")
            where = "PRUID IN (" + ",".join(f"'{p}'" for p in pruids) + ")" if pruids else None
            gdf = gpd.read_file(SHP, where=where, columns=["DGUID", "PRUID"]).to_crs(epsg=4326)
            gdf["geometry"] = gdf["geometry"].make_valid()
    geom_by_da = dict(zip(gdf["DGUID"], gdf["geometry"]))
    print(f"  loaded {len(geom_by_da)} DA polygons")

    features = []
    remainders: dict = {}
    missing = 0
    for dguid, origins in by_da.items():
        geom = geom_by_da.get(dguid)
        if geom is None or geom.is_empty:
            missing += 1
            continue
        dot_rows = []
        for origin, count in origins:
            total = count / SCALE + remainders.get(origin, 0.0)
            n = int(total)
            remainders[origin] = total - n
            if n > 0:
                dot_rows.append((n, origin))
        if not dot_rows:
            continue
        total_dots = sum(r[0] for r in dot_rows)
        pts = random_points_in_polygon(geom, total_dots)
        idx = 0
        for n, origin in dot_rows:
            grp = group_map.get(origin, "")
            for pt in pts[idx:idx + n]:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(pt[0], 5), round(pt[1], 5)]},
                    "properties": {"label": origin, "group": grp},
                })
            idx += n

    out = HERE / "data" / "processed" / f"dots_{region}_1per{SCALE}.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"  {len(features)} dots -> {out}"
          + (f"  ({missing} DAs had no matching polygon)" if missing else ""))


if __name__ == "__main__":
    main()
