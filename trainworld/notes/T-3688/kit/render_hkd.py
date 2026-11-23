"""Hokkaido's zones before and after the T-3688 design, from the cached subprefecture municipalities.
usage: render_hkd.py <out.png>"""
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
from shapely.geometry import shape  # noqa: E402

TW = r"C:\Users\anita\projects\maps\trainworld"
RAW = os.path.join(TW, "data", "raw", "tier0", "japan")
POP = json.load(open(os.path.join(TW, "pipeline", "japan_zones.json"), encoding="utf-8"))["unit_population"]
SAPPORO = ["01-ishikari", "01-01203"]
BEFORE = {"Sapporo (spk)": SAPPORO, "Kamikawa (kmk)": ["01-kamikawa", "01-soya", "01-rumoi"], "Sorachi (sor)": ["01-sorachi"],
          "Iburi (ibr)": ["01-iburi", "01-shiribeshi~01203"], "Okhotsk (okh)": ["01-okhotsk", "01-nemuro"],
          "Tokachi (tkc)": ["01-tokachi", "01-hidaka"], "Kushiro (ksr)": ["01-kushiro"], "Oshima (osh)": ["01-oshima", "01-hiyama"]}
AFTER = {"Sapporo (spk)": SAPPORO, "Oshima (osh)": ["01-oshima", "01-hiyama", "01-shiribeshi~01203"],
         "Iburi (ibr)": ["01-iburi", "01-hidaka"], "Kamikawa (kmk)": ["01-kamikawa", "01-sorachi", "01-rumoi", "01-soya"],
         "Tokachi (tkc)": ["01-tokachi", "01-kushiro", "01-okhotsk", "01-nemuro"]}
PAL = ["#d9d9d9", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69"]


def unit(u):
    fc = json.load(open(os.path.join(RAW, f"municipalities_{u}.geojson"), encoding="utf-8"))
    return shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in fc["features"]]).buffer(0.002).simplify(0.005)


def main():
    geo = {u: unit(u) for g in BEFORE.values() for u in g}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (title, groups) in zip(axes, [("Hokkaido: before (8 zones)", BEFORE), ("Hokkaido: after (Sapporo + 4)", AFTER)]):
        for i, (name, us) in enumerate(groups.items()):
            g = shapely.make_valid(shapely.union_all([shapely.set_precision(geo[u], 1e-4) for u in us])).buffer(0)
            g = shapely.MultiPolygon([p for p in getattr(g, "geoms", [g]) if p.area > 0.002]) if g.geom_type == "MultiPolygon" else g
            for p in getattr(g, "geoms", [g]):
                ax.fill(*p.exterior.xy, color=PAL[i % len(PAL)], ec="#333", lw=0.8)
            big = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
            hull = big.area / big.convex_hull.area
            people = sum(POP.get(u, 0) for u in us)
            pt = shapely.ops.polylabel(big, 0.01)
            ax.text(pt.x, pt.y, f"{name}\n{people / 1e6:.2f}M" + ("" if name.startswith("Sapporo") else f"\nhull {hull:.2f}"),
                    ha="center", va="center", fontsize=7.5, weight="bold")
        ax.set_xlim(139.3, 146.0)
        ax.set_ylim(41.3, 45.6)
        ax.set_aspect(1 / math.cos(math.radians(43.4)))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=110)
    print("->", sys.argv[1])


if __name__ == "__main__":
    main()
