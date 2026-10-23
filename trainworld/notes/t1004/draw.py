"""Picture of the dry run: lower 48 (and an inset list for AK/HI/PR skipped), metro rows dark, zones in colours,
built packs grey. Out: artifacts/trainworld/t1004-zones.png"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shapely  # noqa: E402
from shapely import union_all  # noqa: E402
import world_plan as wp  # noqa: E402

st = pickle.load(open(os.path.join(HERE, "dry_state.pkl"), "rb"))
by = {u.id: u for u in st["units"]}
rows = json.load(open(os.path.join(HERE, "dry2_rows.json")))["rows"]
zones = json.load(open(os.path.join(HERE, "dry2_zones_500000.json")))["zones"]
SKIP = ("02", "15", "72", "60", "66", "69", "78")
fig, ax = plt.subplots(figsize=(22, 13), dpi=110)
cols = plt.cm.tab20.colors


def draw(g, fc, ec="white", lw=0.4):
    g = shapely.simplify(g, 2000)
    for p in shapely.get_parts(g):
        if p.geom_type != "Polygon":
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, fc=fc, ec=ec, lw=lw)


us48 = [u for u in st["units"] if u.parents[0][0] not in SKIP and u.iso == "USA"]
draw(union_all([u.geom for u in us48]), "#dddddd", "none")
for q in st["packs"].values():
    if q["geom"].bounds[0] > -1.4e7 and q["geom"].bounds[2] < -5e6 and q["geom"].bounds[1] > 2.5e6:
        draw(q["geom"], "#888888", "white", 0.6)
for k, z in enumerate(sorted(zones, key=lambda z: z["lead"])):
    if z["state"] in SKIP:
        continue
    g = union_all([by[i].geom for i in z["units"]])
    draw(g, cols[k % 20])
    c = g.representative_point()
    ax.text(c.x, c.y, f"{z['name'].split(',')[0].replace(' County', '').replace(' Parish', '')}\n{z['people'] / 1e6:.2f}",
            fontsize=5.5, ha="center", va="center")
for r in rows:
    if r["lead_state"] in SKIP:
        continue
    g = union_all([by[i].geom for i in r["units"]])
    draw(g, "#2b2b55", "white", 0.6)
    c = g.representative_point()
    ax.text(c.x, c.y, f"{r['name'].split(' [')[0]}\n{r['people'] / 1e6:.1f}", fontsize=6, ha="center", va="center", color="white", weight="bold")
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("T-1004 dry run: 44 metro rows (dark), 197 zones at the 0.5M floor (colours; label M people), built packs grey. Lower 48 only.", fontsize=13)
out = r"C:\Users\anita\projects\maps\artifacts\trainworld\t1004-zones.png"
fig.savefig(out, bbox_inches="tight")
print(out)
