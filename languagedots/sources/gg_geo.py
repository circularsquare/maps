"""Guernsey placement layer: Kontur 400 m hexes, one unit `GG` -> data/geo/gg/gg_hexes.gpkg.

    python sources/gg_geo.py

The Electronic Census population (64,091 on 31 March 2023) is Guernsey, Herm and Jethou. Kontur's
"GG" extract covers the whole Bailiwick (155 hexes, 62,849 people), so Alderney (x > -2.30,
y > 49.70; about 2,000 people) and Sark (x about -2.35; about 560) are left out by a longitude cut
at -2.40: every Guernsey, Herm and Jethou hex centroid lies west of it, every Alderney and Sark
one east of it (asserted). One unit, because the census publishes country of birth only for the
island as a whole (parish populations exist, sources/gg.md, but no parish table of birthplace).

Kontur against the census: 60,362 people in the 120 hexes kept, against 64,091 (ratio 0.94).
"""
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid import hex_layer, kontur_path  # noqa: E402

CUT = -2.40
CENSUS = 64_091


def main():
    k = gpd.read_file(kontur_path("gg"))
    c = k.geometry.to_crs(4326).centroid
    west = c.x < CUT
    # Alderney and Sark sit east of the cut, Guernsey / Herm / Jethou west of it
    assert (c[west].y < 49.56).all() and (c[~west].x > -2.40).all()
    assert abs(k.loc[west, "population"].sum() - 60_362) < 1, k.loc[west, "population"].sum()
    units = gpd.GeoDataFrame({"unit": ["GG"]}, geometry=[box(-2.80, 49.35, CUT, 49.60)], crs=4326)
    hex_layer("gg", units, census={"GG": CENSUS})


if __name__ == "__main__":
    main()
