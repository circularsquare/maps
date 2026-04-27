"""Build per-country GeoJSONs from shapefile(s) + long-format population CSV.

Usage: python scripts/build_country.py <country_id>

Reads:
  data/<id>/population.csv       columns: code, level, year, pop
  <boundary shapefiles>          per-country convention — see SHAPEFILES below

Writes:
  countries/<id>/adm{N}.geojson  with properties.populations = {year: pop, ...}
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "helper1m"

# Per-country shapefile locations. Each entry lists candidate paths (first hit wins)
# and the PCODE column used to join with population.csv.
SHAPEFILES = {
    "indonesia": {
        1: {
            "candidates": [
                REPO_ROOT / "data/asia1m/indonesia/idn_admbnda_adm1_bps_20200401.shp",
                HELPER / "data/indonesia/boundaries/idn_admbnda_adm1_bps_20200401.shp",
            ],
            "code_col": "ADM1_PCODE",
            "name_col": "ADM1_EN",
            "parent_col": None,
            "parent_name_col": None,
        },
        2: {
            "candidates": [
                REPO_ROOT / "data/asia1m/indonesia/idn_admbnda_adm2_bps_20200401.shp",
                HELPER / "data/indonesia/boundaries/idn_admbnda_adm2_bps_20200401.shp",
            ],
            "code_col": "ADM2_PCODE",
            "name_col": "ADM2_EN",
            "parent_col": "ADM1_PCODE",
            "parent_name_col": "ADM1_EN",
        },
        3: {
            "candidates": [
                REPO_ROOT / "data/asia1m/indonesia/idn_admbnda_adm3_bps_20200401.shp",
                HELPER / "data/indonesia/boundaries/idn_admbnda_adm3_bps_20200401.shp",
            ],
            "code_col": "ADM3_PCODE",
            "name_col": "ADM3_EN",
            "parent_col": "ADM2_PCODE",
            "parent_name_col": "ADM2_EN",
        },
    },
}


def find_shapefile(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load_populations(country_id):
    """Returns {level: {code: {year: pop}}}. Empty dict if CSV is missing."""
    csv_path = HELPER / "data" / country_id / "population.csv"
    if not csv_path.exists():
        print(f"  (no population.csv at {csv_path} — writing geojsons with empty populations)")
        return {}
    df = pd.read_csv(csv_path, dtype={"code": str, "level": int, "year": int, "pop": "Int64"})
    out = {}
    for (level, code), group in df.groupby(["level", "code"]):
        out.setdefault(level, {})[code] = {
            int(r.year): int(r.pop) for r in group.itertuples() if pd.notna(r.pop)
        }
    return out


def build_level(country_id, level, cfg, pops_by_code):
    shp = find_shapefile(cfg["candidates"])
    if shp is None:
        print(f"  adm{level}: no shapefile found, skipping")
        return
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")

    # Area in km² via equal-area projection (World Mollweide).
    areas_m2 = gdf.to_crs("ESRI:54009").area
    gdf["area_km2"] = (areas_m2 / 1e6).round(2)

    # Slim properties — just what the viewer needs.
    def props(row):
        code = row[cfg["code_col"]]
        p = {
            "code": code,
            "name": row[cfg["name_col"]],
            "area_km2": float(row["area_km2"]),
            "populations": pops_by_code.get(code, {}),
        }
        if cfg["parent_col"]:
            p["parent_code"] = row[cfg["parent_col"]]
            p["parent_name"] = row[cfg["parent_name_col"]]
        return p

    features = []
    for _, row in gdf.iterrows():
        features.append({
            "type": "Feature",
            "properties": props(row),
            "geometry": row["geometry"].__geo_interface__,
        })

    fc = {"type": "FeatureCollection", "features": features}
    out = HELPER / "countries" / country_id / f"adm{level}.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(fc, f, separators=(",", ":"))
    matched = sum(1 for feat in features if feat["properties"]["populations"])
    print(f"  adm{level}: {len(features)} features, {matched} with population data -> {out.relative_to(HELPER)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country_id")
    ap.add_argument("--levels", type=int, nargs="+", default=None,
                    help="Admin levels to build (default: all configured)")
    args = ap.parse_args()

    if args.country_id not in SHAPEFILES:
        print(f"unknown country: {args.country_id}", file=sys.stderr)
        sys.exit(1)

    pops = load_populations(args.country_id)
    print(f"building {args.country_id}")
    levels = args.levels or sorted(SHAPEFILES[args.country_id])
    for level in levels:
        cfg = SHAPEFILES[args.country_id].get(level)
        if cfg is None:
            continue
        build_level(args.country_id, level, cfg, pops.get(level, {}))


if __name__ == "__main__":
    main()
