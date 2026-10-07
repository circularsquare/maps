"""OSM place nodes for Iran (city, town, village, hamlet), from the Geofabrik extract.

Writes helper1m/data/iran/osm_places.gpkg with osm_id, name, place, name_fa (from name:fa when
the plain name is not Persian). bakhsh.py uses them to locate 1395 settlements by name inside
their county, which gives each OSM district piece a composition that needs no district names.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pyogrio  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "iran" / "raw" / "osm"
OUT = HELPER / "data" / "iran" / "osm_places.gpkg"


def main():
    pbf = sorted(RAW.glob("iran-*.osm.pbf"))[-1]
    g = pyogrio.read_dataframe(pbf, layer="points",
                               where="place IN ('city','town','village','hamlet')")
    g["name_fa"] = g["other_tags"].map(
        lambda s: (re.search(r'"name:fa"=>"([^"]*)"', s).group(1)
                   if isinstance(s, str) and '"name:fa"' in s else None))
    g = g[["osm_id", "name", "name_fa", "place", "geometry"]]
    print(g["place"].value_counts())
    if OUT.exists():
        OUT.unlink()
    g.to_file(OUT, layer="places", driver="GPKG")
    print("wrote", OUT, len(g))


if __name__ == "__main__":
    main()
