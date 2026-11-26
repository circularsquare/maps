"""kit/dry.py for specs that touch Zhuhai's metro (zhu): its unit list holds geoBoundaries' Macau SAR ADM1 feature
(43563684B97104103456250: Hengqin once Macau is cut out, T-897's recut), which the plan state lacks, so plan_edit
raises KeyError on zhu. This adds that unit from CHN_ADM1.geojson, de facto cut, people 0 (zhu's people go by delta).
Applying such a spec with plan_edit --go needs the same unit in the state (or plan_edit to skip it for geometry).

    venv\\Scripts\\python.exe -I trainworld\\notes\\T-4364\\cn_hengqin.py <spec> --out <folder> [--png <sheet>]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kit"))
import dry  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import shape  # noqa: E402
import world_plan as wp  # noqa: E402

UID = "43563684B97104103456250"
GB = r"C:\Users\anita\projects\maps\trainworld\data\raw\worldplan\geoboundaries\CHN_ADM1.geojson"
_orig = dry.pe.load_units


def load_units(iso):
    by = _orig(iso)
    if iso == "CHN" and UID not in by:
        f = next(f for f in json.load(open(GB, encoding="utf-8"))["features"] if f["properties"]["shapeID"] == UID)
        geom = wp.polygonal(wp.to_moll(shapely.make_valid(shape(f["geometry"]))))
        by[UID] = wp.Unit(UID, "Hengqin (Macau SAR feature less Macau)", "CHN", wp.de_facto("CHN", geom) or geom, [])
    return by


dry.pe.load_units = load_units
if __name__ == "__main__":
    dry.main()
