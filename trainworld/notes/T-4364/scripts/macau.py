"""Macau's pack outline against OSM's land (the coastline sea's complement) round it; read-only. Draws macau_check.png."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import shapely  # noqa: E402
from shapely.geometry import shape  # noqa: E402
import osm_water  # noqa: E402
import world_plan as wp  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = r"C:\Users\anita\projects\maps\trainworld\data\packs"
HERE = os.path.dirname(os.path.abspath(__file__))


def outline(cid):
    gj = json.load(open(os.path.join(P, f"{cid}.boundary.geojson"), encoding="utf-8"))
    return wp.polygonal(shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in gj["features"]]))


mcu, zhu = outline("mcu"), outline("zhu")
box = shapely.box(113.50, 22.08, 113.62, 22.23)
sea = shapely.union_all([shapely.make_valid(s) for s in osm_water.sea(box.bounds)])
land = box.difference(sea)
# Macau's land: land closer to Macau's outline than to Zhuhai's, east of Hengqin's channel; take land within 600 m of mcu
near = land.intersection(mcu.buffer(0.006))
gap = near.difference(mcu)
km2 = lambda g: wp.to_moll(g).area / 1e6  # noqa: E731
print(f"mcu outline {km2(mcu):.2f} km2; OSM land within 600 m of it {km2(near):.2f} km2; land near it outside the outline "
      f"{km2(gap):.2f} km2; outline over sea {km2(mcu.intersection(sea)):.2f} km2")
parts = sorted(getattr(gap, "geoms", [gap]), key=lambda q: -q.area)
for q in parts[:8]:
    c = q.representative_point()
    print(f"  gap piece {km2(q):.3f} km2 at {c.x:.4f},{c.y:.4f}; overlaps zhu {km2(q.intersection(zhu)):.3f}")
fig, ax = plt.subplots(figsize=(9, 11))
for p in getattr(sea, "geoms", [sea]):
    if p.geom_type == "Polygon":
        ax.fill(*p.exterior.xy, color="#cfe3f3", ec="none")
        for h in p.interiors:
            ax.fill(*h.xy, color="#e8e4d8", ec="none")
for g, ec, lw in ((zhu, "#3b6ea5", 1.0), (mcu, "#c0501f", 2.0)):
    for p in getattr(g, "geoms", [g]):
        if p.geom_type == "Polygon":
            ax.plot(*p.exterior.xy, color=ec, lw=lw)
            for h in p.interiors:
                ax.plot(*h.xy, color=ec, lw=lw)
ax.set_facecolor("#e8e4d8")
ax.set_xlim(113.50, 113.62)
ax.set_ylim(22.08, 22.23)
ax.set_aspect(1.08)
ax.set_title("Macau: game outline (orange), Zhuhai's (blue), OSM land (beige), sea (blue fill)", fontsize=9)
fig.savefig(os.path.join(HERE, "macau_check.png"), dpi=110)
