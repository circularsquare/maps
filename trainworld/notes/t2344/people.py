"""GHS-POP 2025 people: Jakarta's and Tangerang's plan units; Ukraine held, occupied (ISW), Crimea, by oblast."""
import json, sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import shapely
from shapely.geometry import shape
import world_plan as wp

T = r"C:\Users\anita\projects\maps\trainworld"
K = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8509086d-46cd-4c5f-8d59-da9348f4a5e0\scratchpad\countries2"
pop = wp.Pop()


def people(g_ll):
    return float(pop.inside(wp.polygonal(wp.to_moll(g_ll)))[2].sum())


p = json.load(open(T + r"\pipeline\world_plan.json", encoding="utf-8"))
rows = {r["id"]: r for r in p["metros"]}
gb = {f["properties"]["shapeID"]: f for f in json.load(open(T + r"\data\raw\worldplan\geoboundaries\IDN_ADM2.geojson", encoding="utf-8"))["features"]}
for rid in ("jak", "tan"):
    tot = 0
    for u in rows[rid]["units"]["IDN"]:
        n = people(shape(gb[u]["geometry"]))
        tot += n
        print(rid, gb[u]["properties"]["shapeName"], f"{n / 1e6:.2f}M")
    print(rid, "sum", f"{tot / 1e6:.2f}M")

isw = shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in json.load(open(r"C:\Users\anita\projects\maps\trainworld\data\raw\worldplan\isw_ukr_control_2026-09-29.geojson", encoding="utf-8"))["features"]])
crimea_box = shapely.box(*wp.CRIMEA)
ne = wp.countries()
ukr = shapely.union_all([wp.to_ll(g) for c, g in ne if c == "UKR"])
rus_crimea = shapely.union_all([wp.to_ll(g) for c, g in ne if c == "RUS"]).intersection(crimea_box)
print("Crimea (NE RUS in box)", f"{people(rus_crimea) / 1e6:.2f}M")
print("Ukraine NE", f"{people(ukr) / 1e6:.2f}M")
occ = ukr.intersection(isw)
print("Ukraine NE inside ISW", f"{people(occ) / 1e6:.2f}M")
held = ukr.difference(isw)
print("Ukraine NE held", f"{people(held) / 1e6:.2f}M")
a1 = json.load(open(T + r"\data\raw\worldplan\geoboundaries\UKR_ADM1.geojson", encoding="utf-8"))["features"]
for f in a1:
    g = shapely.make_valid(shape(f["geometry"]))
    o = g.intersection(isw)
    print(f"  {f['properties']['shapeName']}: all {people(g) / 1e6:.2f}M, inside ISW {people(o) / 1e6:.2f}M, area share {o.area / g.area:.2f}")
