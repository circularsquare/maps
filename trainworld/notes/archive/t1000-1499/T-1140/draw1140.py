"""T-1140 pictures: zones before (the plan) and after (a dry run), side by side; zones that change in
colour, kept zones grey, metro rows dark.

    python draw1140.py <data> <result.pkl> <variant|-> <out.png> <title> [--isos A,B] [--regions r1,r2] [--bbox x0,y0,x1,y1]
"""
import os
import math
import sys
LATS = []

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shapely  # noqa: E402
from shapely import union_all  # noqa: E402

import common as C  # noqa: E402

args = sys.argv[1:]
data, res, variant, out, title = args[:5]
opt = {args[i]: args[i + 1] for i in range(5, len(args) - 1, 2)}
D = C.load(f"{data}_data.pkl")
Rz = C.load(res)
Rz = Rz[variant] if variant != "-" else Rz
isos = opt.get("--isos", ",".join(D)).split(",")
regions = set(opt["--regions"].split(",")) if "--regions" in opt else None
bbox = [float(v) for v in opt["--bbox"].split(",")] if "--bbox" in opt else None
PAL = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
       "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#fabebe"]


def reg_of(d, i):
    u = d["units"][i]
    if "region" in d and d["region"]:
        r = d["region"][i]
    else:
        r = u.parents[0]
    return r[1] if isinstance(r, tuple) else str(r)


fig, axes = plt.subplots(1, 2, figsize=(26, 13), dpi=100)
for iso in isos:
    d = D[iso]
    U = d["units"]
    S = C.Shapes(U)
    plan = {r["id"]: [i for i in r["units"][iso] if i in U] for r in d["rows"] if r.get("kind") == "zone"}
    plan = {k: v for k, v in plan.items() if v}
    after = Rz[iso]["groups"]
    pset = {frozenset(v) for v in plan.values()}
    aset = {frozenset(v) for v in after.values()}
    keep = (lambda ids: any(reg_of(d, i) in regions for i in ids)) if regions else (lambda ids: True)
    for ax, gr, other in ((axes[0], list(plan.values()), aset), (axes[1], list(after.values()), pset)):
        gr = [g for g in gr if keep(g)]
        shapes = [S.zone(g) for g in gr]
        if bbox:
            box = shapely.box(*bbox)
            sel = [k for k, s in enumerate(shapes) if s.intersects(box)]
            gr, shapes = [gr[k] for k in sel], [shapes[k] for k in sel]
        changed = [frozenset(g) not in other for g in gr]
        # greedy colours for changed zones, distinct from touching ones
        col = {}
        tree = shapely.STRtree(shapes)
        for k in sorted(range(len(gr)), key=lambda k: -shapes[k].area):
            if not changed[k]:
                continue
            near = {col.get(int(j)) for j in tree.query(shapes[k].buffer(2000), predicate="intersects")}
            col[k] = next((c for c in PAL if c not in near), PAL[k % len(PAL)])
        for k, (g, s) in enumerate(zip(gr, shapes)):
            s = C.wp.to_ll(shapely.simplify(s, 1500))
            LATS.append(s.centroid.y)
            fc = col.get(k, "#d9d9d9")
            for p in shapely.get_parts(s):
                if p.geom_type == "Polygon":
                    x, y = p.exterior.xy
                    ax.fill(x, y, fc=fc, ec="white", lw=0.5, zorder=1)
            if changed[k]:
                c = s.representative_point()
                lead = max(g, key=lambda i: U[i].free)
                nm = U[lead].name.split(",")[0].replace(" County", "").replace(" Parish", "")
                ax.text(c.x, c.y, f"{nm}\n{sum(U[i].free for i in g) / 1e6:.2f}M", fontsize=6.5, ha="center",
                        va="center", zorder=4)
        for f in d["fixed"]:
            if not f["units"] or (regions and not keep(f["units"])):
                continue
            g = f["geom"]
            if bbox and not g.intersects(shapely.box(*bbox)):
                continue
            for p in shapely.get_parts(C.wp.to_ll(shapely.simplify(g, 1500))):
                if p.geom_type == "Polygon":
                    x, y = p.exterior.xy
                    ax.fill(x, y, fc="#2b2b55", ec="white", lw=0.5, zorder=2)
for ax, t in zip(axes, ("Now (the plan)", "With the rule")):
    ax.set_aspect(1 / math.cos(math.radians(sum(LATS) / max(len(LATS), 1))))
    ax.axis("off")
    ax.set_title(t, fontsize=14)
    if bbox:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
fig.suptitle(title, fontsize=16)
fig.savefig(out, bbox_inches="tight")
print(out)
