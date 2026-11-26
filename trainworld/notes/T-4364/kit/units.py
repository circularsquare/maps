"""A working map of a country's plan units in a box: each unit coloured by its row, labelled with its name and people,
and a table (id, name, people, row, parents) printed. Read-only.
usage: units.py <ISO3> <w,s,e,n> <out.png> [row,row,...: only list these rows' units in the table]"""
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
import world_plan as wp  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
       "#ccebc5", "#ffed6f", "#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6"]


def main():
    iso, box, out = sys.argv[1], [float(v) for v in sys.argv[2].split(",")], sys.argv[3]
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    by = pe.load_units(iso)
    doc = shared_lock.read_json(pe.PLAN)
    owner = {u: r["id"] for r in doc["metros"] for u in (r.get("units") or {}).get(iso, [])}
    w, s, e, n = box
    bx = shapely.box(w, s, e, n)
    rows = sorted(set(owner.values()))
    col = {k: PAL[i % len(PAL)] for i, k in enumerate(rows)}
    asp = 1 / math.cos(math.radians((s + n) / 2))
    fig, ax = plt.subplots(figsize=(16, max(6, 16 * (n - s) * asp / (e - w))))
    table = []
    for u in by.values():
        g = wp.to_ll(u.geom)
        if not g.intersects(bx):
            continue
        k = owner.get(u.id, "-")
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon":
                ax.fill(*p.exterior.xy, color=col.get(k, "#ffffff"), ec="#555", lw=0.3)
        lp = g.intersection(bx)
        if not lp.is_empty and lp.geom_type in ("Polygon", "MultiPolygon"):
            big = max(getattr(lp, "geoms", [lp]), key=lambda q: q.area)
            pt = shapely.ops.polylabel(big, 0.002)
            ax.text(pt.x, pt.y, f"{u.name}\n{k} {u.pop / 1e3:.0f}k", ha="center", va="center", fontsize=4.5)
        if only is None or k in only:
            c = g.representative_point()
            table.append((k, u.name, u.id, u.pop, f"{c.x:.3f},{c.y:.3f}", "/".join(str(p) for p in (u.parents or [])[:2])))
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(asp)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    for t in sorted(table):
        print(f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]:,.0f}\t{t[4]}\t{t[5]}")


if __name__ == "__main__":
    main()
