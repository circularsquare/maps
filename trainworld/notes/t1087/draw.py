"""T-1087 pictures: zones before and after, metro rows dark, rings outlined red.

    python draw.py us <variant> <out.png> [title]
    python draw.py cn <variant> <out.png> <province> [title]
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shapely  # noqa: E402
from shapely import union_all  # noqa: E402

which, variant, out = sys.argv[1], sys.argv[2], sys.argv[3]
rest = sys.argv[4:]
if which == "us":
    D = pickle.load(open(os.path.join(HERE, "us_data.pkl"), "rb"))
    R = pickle.load(open(os.path.join(HERE, "us_out.pkl"), "rb"))
    prov = None
else:
    D = pickle.load(open(os.path.join(HERE, f"{which}_data.pkl"), "rb"))
    R = pickle.load(open(os.path.join(HERE, f"{which}_out.pkl"), "rb"))
    prov = rest.pop(0)
title = rest[0] if rest else variant
U = D["units"]
V = R[variant]
SKIP = ("02", "15", "72")
fig, ax = plt.subplots(figsize=(22, 13) if which == "us" else (14, 14), dpi=110)
cols = plt.cm.tab20.colors


def draw(g, fc, ec="white", lw=0.4, z=1):
    g = shapely.simplify(g, 1500 if which == "us" else 300)
    for p in shapely.get_parts(g):
        if p.geom_type != "Polygon":
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, fc=fc, ec=ec, lw=lw, zorder=z)


def keep(ids):
    if which == "us":
        return U[ids[0]]["state"] not in SKIP
    return any(U[i]["state"] == prov for i in ids)


ring_a = {a for a, *_ in V["rings"]}
shown = []
for k, (lead, ids) in enumerate(sorted(V["groups"].items(), key=lambda kv: str(kv[0]))):
    if not keep(ids):
        continue
    g = union_all([U[i]["simple"] for i in ids])
    shown.append(g)
    draw(g, cols[k % 20])
    if lead in ring_a:
        draw(g, "none", "#d00000", 2.2, 3)
    c = g.representative_point()
    nm = max(ids, key=lambda i: U[i]["free"])
    label = U[nm]["name"].split(",")[0].replace(" County", "").replace(" Parish", "")
    ppl = sum(U[i]["free"] for i in ids)
    ax.text(c.x, c.y, f"{label}\n{ppl / 1e6:.2f}", fontsize=5.5 if which == "us" else 7, ha="center", va="center", zorder=4)
for r in D["rows"]:
    if r["kind"] == "zone" or not r["units"]:
        continue
    if not keep(r["units"]):
        continue
    g = union_all([U[i]["simple"] for i in r["units"]])
    if which != "us" and prov:
        bb = union_all(shown).buffer(30000) if shown else g
        if not g.intersects(bb):
            continue
    draw(g, "#2b2b55", "white", 0.6, 2)
    c = g.representative_point()
    ax.text(c.x, c.y, r["name"].split(" [")[0], fontsize=6 if which == "us" else 8, ha="center", va="center",
            color="white", weight="bold", zorder=4)
if which == "us":
    for k, p in D["packs"].items():
        draw(p["geom"], "#888888", "white", 0.6, 2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(title, fontsize=13)
fig.savefig(out, bbox_inches="tight")
print(out)
