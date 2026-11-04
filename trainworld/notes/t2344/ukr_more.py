"""Ukraine's held metro rows against the ISW line; Crimea's people by old raion (GHS-POP 2025)."""
import json, sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import shapely
from shapely.geometry import shape
import world_plan as wp
T = r"C:\Users\anita\projects\maps\trainworld"
G = T + r"\data\raw\worldplan\geoboundaries"
K = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8509086d-46cd-4c5f-8d59-da9348f4a5e0\scratchpad\countries2"
isw = shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in json.load(open(r"C:\Users\anita\projects\maps\trainworld\data\raw\worldplan\isw_ukr_control_2026-09-29.geojson", encoding="utf-8"))["features"]])
pop = wp.Pop()


def people(g):
    g = wp.polygonal(wp.to_moll(g))
    return 0.0 if g.is_empty else float(pop.inside(g)[2].sum())


a3 = {f["properties"]["shapeID"]: f for f in json.load(open(G + r"\UKR_ADM3.geojson", encoding="utf-8"))["features"]}
plan = json.load(open(T + r"\pipeline\world_plan.json", encoding="utf-8"))
for r in plan["metros"]:
    if r.get("iso2") == "UA":
        us = r["units"]["UKR"]
        inside = [u for u in us if u in a3 and shapely.make_valid(shape(a3[u]["geometry"])).intersection(isw).area > 0.5 * shapely.make_valid(shape(a3[u]["geometry"])).area]
        print(r["id"], r["name"], len(us), "units; mostly inside ISW:", len(inside), [a3[u]["properties"]["shapeName"] for u in inside][:10])
a1 = json.load(open(G + r"\UKR_ADM1.geojson", encoding="utf-8"))["features"]
cr = shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in a1 if f["properties"]["shapeName"] in ("Autonomous Republic of Crimea", "Sevastopol")])
print("Crimea + Sevastopol (geoBoundaries ADM1)", f"{people(cr) / 1e6:.2f}M")
a2 = json.load(open(G + r"\UKR_ADM2.geojson", encoding="utf-8"))["features"]
for f in a2:
    g = shapely.make_valid(shape(f["geometry"]))
    if cr.contains(g.representative_point()):
        print(f"  {f['properties']['shapeName']}: {people(g) / 1e3:.0f}k")
