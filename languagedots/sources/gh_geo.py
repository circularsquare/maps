"""Ghana placement layer: Kontur 400 m hexes keyed to religiondots' 272 district units.

    python sources/gh_geo.py      -> data/geo/gh/gh_hexes.gpkg  (unit, pop)

religiondots draws Ghana on GSS's own 2021 district polygons (255 districts + 17 sub-metros,
Lake Volta cut out; religiondots/sources/gh_geo.py), read-only here, with no population inside
them. This adds Kontur's population so the dots sit where people live inside each district.
The census check is each unit's Ghanaian population from the ethnic table.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from _grid import hex_layer  # noqa: E402

POLY = HERE.parent / "religiondots" / "data" / "geo" / "gh" / "gh_districts.gpkg"


def main():
    units = gpd.read_file(POLY)[["unit", "geometry"]]
    assert len(units) == 272 and units["unit"].is_unique, "religiondots' gh layer changed"
    df = pd.read_csv(HERE / "data" / "normalized" / "gh.csv")
    census = df.groupby("geo_id")["count"].sum().round().astype(int).to_dict()
    assert set(census) == set(units["unit"]), "gh.csv units and the polygons differ"
    hex_layer("gh", units, census=census)


if __name__ == "__main__":
    main()
