"""
Pull the japanrail poster's slow-changing inputs into build/ once.

Two jobs, neither of which should run on every render:

  1. Basemap. The OSM water polygons are a 1.26 GB shapefile; a bbox-filtered
     read of Japan is 5 s, which is fine once and tedious per iteration. Lakes
     come from HydroLAKES rather than Natural Earth 10m — at roughly 1 px =
     150 m the NE polygons read as visibly straight-edged.

  2. Line colours. The interactive map at riders/japanriders/index.html holds
     the canonical ラインカラー table, and it is hand-tuned (several entries are
     lifted off their official value for dark-background visibility). Parsing it
     out of the page keeps one source of truth rather than a second copy that
     silently drifts.

    python bake.py

Writes into build/:
    jp_water.gpkg      OSM ocean polygons over the Japan bbox
    jp_lakes.gpkg      HydroLAKES polygons over the same, above --min-lake-km2
    colors.json        {jr: {...}, other: "#…", lines: {"op||line": "#…"}}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
BUILD = HERE / "build"
DATA = HERE.parent.parent / "data"
WATER = DATA / "water-polygons-split-4326" / "water_polygons.shp"
COAST = DATA / "coastlines-split-4326" / "lines.shp"
LAKES = DATA / "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp"
ADM = DATA / "asia1m" / "japan"
INDEX = HERE.parent.parent / "riders" / "japanriders" / "index.html"

# Generous enough to hold every framing, Yonaguni to Cape Soya, so the bake
# never has to be redone when the rotation or the margins move.
BBOX = (122.0, 23.0, 150.0, 47.0)


def bake_basemap(min_lake_km2: float):
    t = time.time()
    water = gpd.read_file(WATER, bbox=BBOX)
    water.to_file(BUILD / "jp_water.gpkg", driver="GPKG")
    print(f"  jp_water.gpkg   {len(water):5d} polygons  ({time.time() - t:.1f}s)")

    # Coastline as LINES, not as the water polygons' edge. Stroking the water
    # polygons draws the seams between OSM's split tiles straight across open
    # sea (a bug ancestrydots hit twice); the split coastline file holds only
    # real coast ways, so it strokes clean.
    t = time.time()
    coast = gpd.read_file(COAST, bbox=BBOX)
    coast[["geometry"]].to_file(BUILD / "jp_coast.gpkg", driver="GPKG")
    print(f"  jp_coast.gpkg   {len(coast):5d} ways      ({time.time() - t:.1f}s)")

    t = time.time()
    adm1 = gpd.read_file(ADM / "jpn_admbnda_adm1_2019.shp")
    adm1.to_file(BUILD / "jp_pref.gpkg", driver="GPKG")
    print(f"  jp_pref.gpkg    {len(adm1):5d} prefectures ({time.time() - t:.1f}s)")

    t = time.time()
    if not LAKES.exists():
        print(f"  !! {LAKES.name} missing — skipping lakes")
        return
    lakes = gpd.read_file(LAKES, bbox=BBOX)
    area_col = next((c for c in ("Lake_area", "lake_area") if c in lakes), None)
    if area_col:
        lakes = lakes[lakes[area_col] >= min_lake_km2]
    lakes[["geometry"]].to_file(BUILD / "jp_lakes.gpkg", driver="GPKG")
    print(f"  jp_lakes.gpkg   {len(lakes):5d} polygons >= {min_lake_km2} km2  "
          f"({time.time() - t:.1f}s)")


def parse_colors():
    """Lift JR, OTHER_COLOR and LINE_COLORS out of the interactive map.

    LINE_COLORS is a two-level JS object literal with comments and mixed
    one-line / multi-line operator blocks, so this tracks brace depth rather
    than trying to match a whole block with one regex.
    """
    src = INDEX.read_text(encoding="utf-8")

    jr = {}
    block = re.search(r"const JR = \{(.*?)\n\};", src, re.S).group(1)
    for op, color, en in re.findall(
            r"'([^']+)':\s*\{\s*color:\s*'(#[0-9a-fA-F]{6})',\s*en:\s*'([^']+)'",
            block):
        jr[op] = {"color": color, "en": en}

    other = re.search(r"const OTHER_COLOR = '(#[0-9a-fA-F]{6})'", src).group(1)

    body = re.search(r"const LINE_COLORS = \{\n(.*?)\n\};", src, re.S).group(1)
    lines, op, depth = {}, None, 1
    for raw in body.split("\n"):
        text = re.sub(r"//.*$", "", raw)
        if depth == 1:
            m = re.search(r"'([^']+)':\s*\{", text)
            if m:
                op = m.group(1)
        if op:
            # The operator key itself carries no colour, so scanning the whole
            # line is safe and keeps one-line operator blocks working.
            for ln, col in re.findall(r"'([^']+)'\s*:\s*'(#[0-9a-fA-F]{6})'",
                                      text):
                lines[f"{op}||{ln}"] = col
        depth += text.count("{") - text.count("}")
        if depth <= 1:
            op = None
    return {"jr": jr, "other": other, "lines": lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-lake-km2", type=float, default=2.0,
                    help="drop lakes smaller than this; 2 km2 is about 0.04 in "
                         "across on a 36 in sheet")
    ap.add_argument("--skip-basemap", action="store_true")
    args = ap.parse_args()

    BUILD.mkdir(exist_ok=True)
    if not args.skip_basemap:
        bake_basemap(args.min_lake_km2)

    colors = parse_colors()
    (BUILD / "colors.json").write_text(
        json.dumps(colors, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  colors.json     {len(colors['jr'])} JR operators, "
          f"{len(colors['lines'])} line colours")


if __name__ == "__main__":
    main()
