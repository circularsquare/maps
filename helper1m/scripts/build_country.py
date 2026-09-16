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
import shapely

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "helper1m"

# Geometry simplification tolerance (degrees, ~330 m). Keeps the geojsons small;
# sub-pixel at the zoom levels the viewer uses. A level can override it with
# "simplify" — China's townships are small enough that 330 m mangles them.
SIMPLIFY_TOL = 0.003

# Decimal places kept on coordinates. Five is ~1 m, far finer than the
# simplification above, and it roughly halves the file against full float repr.
COORD_DP = 5

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
    # China's four levels all come out of one township shapefile (see
    # scripts/china/prep_boundaries.py), so the column names are already ours.
    "china": {
        1: {
            "candidates": [HELPER / "data/china/boundaries/adm1.gpkg"],
            "code_col": "code",
            "name_col": "name",
            "parent_col": None,
            "parent_name_col": None,
            "group_col": "group",
            "extra_cols": ["name_cn"],
        },
        2: {
            "candidates": [HELPER / "data/china/boundaries/adm2.gpkg"],
            "code_col": "code",
            "name_col": "name",
            "parent_col": "parent",
            "parent_name_col": None,
            "group_col": "group",
            "extra_cols": ["name_cn"],
        },
        3: {
            "candidates": [HELPER / "data/china/boundaries/adm3.gpkg"],
            "code_col": "code",
            "name_col": "name",
            "parent_col": "parent",
            "parent_name_col": None,
            "group_col": "group",
            "extra_cols": ["name_cn"],
        },
        4: {
            "candidates": [HELPER / "data/china/boundaries/adm4.gpkg"],
            "code_col": "code",
            "name_col": "name",
            "parent_col": "parent",
            "parent_name_col": None,
            "group_col": "group",
            "extra_cols": ["name_cn"],
            # Townships are small — the default 330 m tolerance flattens an
            # urban subdistrict into a triangle.
            "simplify": 0.001,
            # 43,655 features is far too much for one fetch, so this level is
            # written as one file per province and the viewer loads what is ticked.
            "split_by": "group",
        },
    },
    "india": {
        1: {
            "candidates": [REPO_ROOT / "data/asia1m/india/state.shp"],
            "code_col": "pc11_s_id",
            "name_col": "s_name",
            "parent_col": None,
            "parent_name_col": None,
            "group_col": "pc11_s_id",
        },
        2: {
            "candidates": [REPO_ROOT / "data/asia1m/india/district.shp"],
            "code_col": "pc11_d_id",
            "name_col": "d_name",
            "parent_col": "pc11_s_id",
            "parent_name_col": None,
            "group_col": "pc11_s_id",
        },
        3: {
            "candidates": [REPO_ROOT / "data/asia1m/india/subdistrict.shp"],
            "code_col": ["pc11_d_id", "pc11_sd_id"],  # composite: pc11_d_id is global
            "name_col": "sd_name",
            "parent_col": "pc11_d_id",
            "parent_name_col": None,
            "group_col": "pc11_s_id",
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

    # Area in km² via equal-area projection (World Mollweide) — before simplifying.
    areas_m2 = gdf.to_crs("ESRI:54009").area
    gdf["area_km2"] = (areas_m2 / 1e6).round(2)
    # Lighten the geojson — the viewer doesn't need metre-accurate borders.
    gdf["geometry"] = gdf.geometry.simplify(cfg.get("simplify", SIMPLIFY_TOL))
    gdf["geometry"] = shapely.transform(
        gdf.geometry.values, lambda a: a.round(COORD_DP))

    # Slim properties — just what the viewer needs.
    def props(row):
        cc = cfg["code_col"]
        code = "".join(str(row[c]) for c in cc) if isinstance(cc, list) else row[cc]
        p = {
            "code": code,
            "name": row[cfg["name_col"]],
            "area_km2": float(row["area_km2"]),
            "populations": pops_by_code.get(code, {}),
        }
        if cfg["parent_col"]:
            p["parent_code"] = row[cfg["parent_col"]]
            if cfg.get("parent_name_col"):
                p["parent_name"] = row[cfg["parent_name_col"]]
        if cfg.get("group_col"):
            p["group"] = row[cfg["group_col"]]  # adm1 ancestor — viewer's state filter
        for extra in cfg.get("extra_cols", []):
            if row.get(extra) is not None:
                p[extra] = row[extra]
        return p

    features = []
    for _, row in gdf.iterrows():
        features.append({
            "type": "Feature",
            "properties": props(row),
            "geometry": row["geometry"].__geo_interface__,
        })

    matched = sum(1 for feat in features if feat["properties"]["populations"])
    out_dir = HELPER / "countries" / country_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(path, feats):
        with path.open("w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f,
                      separators=(",", ":"))
        return path.stat().st_size

    split_by = cfg.get("split_by")
    if not split_by:
        size = dump(out_dir / f"adm{level}.geojson", features)
        print(f"  adm{level}: {len(features)} features, {matched} with population "
              f"data, {size / 1e6:.1f} MB -> adm{level}.geojson")
        return

    # One file per group, so the viewer fetches only what is ticked.
    parts = {}
    for feat in features:
        parts.setdefault(feat["properties"][split_by], []).append(feat)
    part_dir = out_dir / f"adm{level}"
    part_dir.mkdir(exist_ok=True)
    for stale in part_dir.glob("*.geojson"):
        stale.unlink()
    total = 0
    for key, feats in sorted(parts.items()):
        total += dump(part_dir / f"{key}.geojson", feats)
    print(f"  adm{level}: {len(features)} features, {matched} with population "
          f"data, {total / 1e6:.1f} MB across {len(parts)} files -> adm{level}/")


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
