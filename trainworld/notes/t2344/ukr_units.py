"""geoBoundaries UKR levels: counts, sample names, and how the ISW line cuts ADM3 units."""
import json, sys
import shapely
from shapely.geometry import shape
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\anita\projects\maps\trainworld\data\raw\worldplan\geoboundaries"
K = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8509086d-46cd-4c5f-8d59-da9348f4a5e0\scratchpad\countries2"
isw = shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in json.load(open(r"C:\Users\anita\projects\maps\trainworld\data\raw\worldplan\isw_ukr_control_2026-09-29.geojson", encoding="utf-8"))["features"]])
shapely.prepare(isw)
crimea = shapely.box(32.4, 44.3, 36.7, 46.3)
for lvl in ("ADM1", "ADM2", "ADM3"):
    fs = json.load(open(f"{T}\\UKR_{lvl}.geojson", encoding="utf-8"))["features"]
    print(lvl, len(fs), "props", list(fs[0]["properties"]), [f["properties"]["shapeName"] for f in fs[:8]])
    if lvl == "ADM1":
        print("  ", sorted(f["properties"]["shapeName"] for f in fs))
    if lvl == "ADM3":
        whole = part = 0
        cr = 0
        for f in fs:
            g = shapely.make_valid(shape(f["geometry"]))
            if g.is_empty:
                continue
            a = g.intersection(isw).area / g.area if g.intersects(isw) else 0.0
            if a > 0.5:
                whole += 1
            elif a > 0.02:
                part += 1
            if crimea.contains(g.representative_point()) and g.representative_point().y > 44.3 and a > 0.5:
                cr += 1
        print(f"  ADM3 mostly inside ISW: {whole}; cut 2-50%: {part}; inside ISW and in the Crimea box: {cr}")
