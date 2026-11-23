"""Before/after of plan-row redraws: each spec's region, units coloured by row, rows labelled with name and people.
usage: render.py <out.png> <spec.json> [<spec.json> ...]   (dry: reads the plan, writes only the picture)
A spec may carry "view": [w, s, e, n] in lon/lat; else the touched rows' box plus a margin."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
import world_plan as wp  # noqa: E402

PAL = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
       "#ccebc5", "#ffed6f", "#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6"]


def draw(ax, rows, iso, by, view, touched, title):
    w, s, e, n = view
    box = shapely.box(w, s, e, n)
    ids = []
    for k, r in rows.items():
        us = [u for u in r["units"].get(iso, []) if u in by]
        if not us:
            continue
        g = wp.to_ll(shapely.union_all([by[u].geom for u in us]))
        if g.intersects(box):
            ids.append((k, g, sum(by[u].pop for u in us)))
    for i, (k, g, p) in enumerate(sorted(ids)):
        c = PAL[i % len(PAL)]
        for part in getattr(g, "geoms", [g]):
            x, y = part.exterior.xy
            ax.fill(x, y, color=c, ec="#333" if k in touched else "#999", lw=1.6 if k in touched else 0.5, zorder=2 if k in touched else 1)
        lp = g.intersection(box)
        if lp.is_empty:
            continue
        pt = shapely.ops.polylabel(max(getattr(lp, "geoms", [lp]), key=lambda q: q.area), 0.01) if lp.geom_type in ("Polygon", "MultiPolygon") else lp.representative_point()
        ax.text(pt.x, pt.y, f"{rows[k]['name']}\n{p / 1e6:.2f}M", ha="center", va="center", fontsize=7 if k in touched else 5.5,
                weight="bold" if k in touched else "normal", zorder=5)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / max(0.2, __import__("math").cos(__import__("math").radians((s + n) / 2))))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(len(specs), 2, figsize=(14, 6.5 * len(specs)), squeeze=False)
    doc = shared_lock.read_json(pe.PLAN)
    for i, path in enumerate(specs):
        spec = json.load(open(path, encoding="utf-8"))
        iso = spec["iso"]
        by = pe.load_units(iso)
        before = {r["id"]: r for r in doc["metros"] if r.get("units", {}).get(iso)}
        new, gone, waves, others, lines = pe.plan_change(spec, json.loads(json.dumps(doc)), by)
        print("\n".join(lines))
        after = {k: v for k, v in before.items() if k not in gone}
        after.update({k: v for k, v in new.items() if k not in gone})
        touched = set(new) | set(gone)
        if "view" in spec:
            view = spec["view"]
        else:
            g = shapely.union_all([wp.to_ll(shapely.union_all([by[u].geom for u in before[k]["units"][iso]])) for k in touched if k in before])
            w, s, e, n = g.bounds
            m = 0.15 * max(e - w, n - s)
            view = (w - m, s - m, e + m, n + m)
        label = spec.get("label", spec["task"])
        draw(axes[i][0], before, iso, by, view, touched, f"{label}: before")
        draw(axes[i][1], after, iso, by, view, touched, f"{label}: after")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("->", out)


if __name__ == "__main__":
    main()
