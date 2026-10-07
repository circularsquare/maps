"""Moldova: a Kontur placement layer keyed to religiondots' 901 UAT polygons.

    python sources/md_geo.py     -> data/geo/md/md_hexes.gpkg (downloads Kontur MD, ~2 MB, once)

UNITS. religiondots/data/geo/md/md_uat.gpkg, read only: BNS's own commune layer
(`comune_p_distrib_2024_view` on gis.statistica.md), keyed by the CUATM code the census tables
carry, with Chişinău's five sectors cut from OpenStreetMap and clipped to the city
(religiondots/sources/md_geo.py). Its `unit` is the same 7-digit code as data/normalized/md.csv,
so there is no name join; the code sets are asserted equal both ways below.

WHY HEXES. religiondots draws Moldova straight on the UAT polygons, uniformly. A UAT includes its
fields and forest, so Kontur 2023 r8 hexes (sources/_grid.py, by centroid) put the dots in the
villages. Any UAT with no populated hex is given its own polygon with pop=1 so it still draws.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402
from _grid import hex_layer  # noqa: E402

UNITS = RD_GEO / "md" / "md_uat.gpkg"
OUT = ROOT / "data" / "geo" / "md" / "md_hexes.gpkg"


def main():
    units = gpd.read_file(UNITS)
    units["unit"] = units["unit"].astype(str)
    df = pd.read_csv(ROOT / "data" / "normalized" / "md.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "uat"]
    census = df.groupby("geo_id")["count"].sum().to_dict()
    a, b = set(units["unit"]), set(census)
    if a != b or len(a) != 901:
        raise SystemExit(f"units {len(a)}, census {len(b)}; only in polygons {sorted(a - b)[:5]}, "
                         f"only in census {sorted(b - a)[:5]}")
    print(f"901 UAT codes, identical in the polygons and the census table")

    layer = hex_layer("md", units, census=census)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(a - set(per.index[per > 0]))
    if empty:
        extra = units[units["unit"].isin(empty)][["unit", "geometry"]].copy()
        extra["pop"] = 1.0
        layer = pd.concat([layer[layer["unit"].isin(set(per.index[per > 0]))],
                           extra.to_crs(4326)], ignore_index=True)
        layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=4326)
        layer.to_file(OUT, layer="hexes", driver="GPKG")
        print(f"  {len(empty)} UATs with no populated hex drawn on their own polygon: {empty}")
    print(f"  {OUT}: {len(layer):,} features, {layer['unit'].nunique()} units")


if __name__ == "__main__":
    main()
