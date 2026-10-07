"""Pull Iran's administrative polygons out of the Geofabrik extract.

Reads  helper1m/data/iran/raw/osm/iran-*.osm.pbf  (Geofabrik, 2026-10-05)
Writes helper1m/data/iran/osm_admin.gpkg  layer `admin`: every boundary=administrative
       multipolygon at admin_level 4-8 with osm_id, admin_level, name, name:en, name:fa and the
       raw other_tags, through GDAL's OSM driver (it assembles relations into polygons).

Overpass answered 504 on three endpoints on 2026-10-06, hence the extract.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OSM_CONFIG_FILE", "")

import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pyogrio  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "iran" / "raw" / "osm"
OUT = HELPER / "data" / "iran" / "osm_admin.gpkg"


def tag(s, key):
    if not isinstance(s, str):
        return None
    m = re.search(r'"' + re.escape(key) + r'"=>"((?:[^"\\]|\\.)*)"', s)
    return m.group(1) if m else None


def main():
    pbf = sorted(RAW.glob("iran-*.osm.pbf"))[-1]
    print("reading", pbf)
    os.environ.pop("OSM_CONFIG_FILE", None)
    g = pyogrio.read_dataframe(pbf, layer="multipolygons",
                               where="boundary = 'administrative' AND admin_level IN "
                                     "('4','5','6','7','8')")
    print(len(g), "admin multipolygons")
    g["name_en"] = g["other_tags"].map(lambda s: tag(s, "name:en"))
    g["name_fa"] = g["other_tags"].map(lambda s: tag(s, "name:fa"))
    g["ref"] = g["other_tags"].map(lambda s: tag(s, "ref"))
    g["wikidata"] = g["other_tags"].map(lambda s: tag(s, "wikidata"))
    keep = ["osm_id", "osm_way_id", "admin_level", "name", "name_en", "name_fa", "ref", "wikidata",
            "other_tags", "geometry"]
    g = g[[c for c in keep if c in g.columns]]
    print(g["admin_level"].value_counts().sort_index())
    if OUT.exists():
        OUT.unlink()
    g.to_file(OUT, layer="admin", driver="GPKG")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
