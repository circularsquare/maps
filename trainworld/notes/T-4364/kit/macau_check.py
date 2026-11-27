"""T-4364 job, after China's waves: Macau's land (data/raw/worldplan/osm_macau.geojson) against the built mcu and zhu
outlines. Exit 1 when Zhuhai's outline holds over ZHU_MAX_KM2 of Macau's land (Anita: none of Macau in Zhuhai's zone);
Macau land outside mcu and mcu's area are printed. Read-only.

    macau_check.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shapely  # noqa: E402
from shapely.geometry import shape  # noqa: E402
import world_plan as wp  # noqa: E402

ZHU_MAX_KM2 = 0.5


def outline(path):
    gj = json.load(open(path, encoding="utf-8"))
    return wp.polygonal(shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in gj["features"] if f["geometry"]]))


km2 = lambda g: wp.to_moll(g).area / 1e6  # noqa: E731
mac = outline(str(wp.NE_REPLACE["MAC"]))
mcu = outline(os.path.join(TW, "data", "packs", "mcu.boundary.geojson"))
zhu = outline(os.path.join(TW, "data", "packs", "zhu.boundary.geojson"))
inz = km2(zhu.intersection(mac))
out = mac.difference(mcu)
print(f"Macau land {km2(mac):.1f} km2; mcu outline {km2(mcu):.1f} km2; Macau land outside mcu {km2(out):.2f} km2; "
      f"in zhu {inz:.2f} km2; mcu over zhu {km2(mcu.intersection(zhu)):.2f} km2")
for p in sorted(shapely.get_parts(out), key=lambda q: -q.area)[:5]:
    c = p.representative_point()
    print(f"  outside mcu: {km2(p):.2f} km2 at {c.x:.4f},{c.y:.4f}")
sys.exit(1 if inz > ZHU_MAX_KM2 else 0)
