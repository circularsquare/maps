"""A spec's before/after as zone outlines and names over a Carto basemap (tiles cached in the scratch folder given).
usage: render_map.py <out.png> <spec.json> <tile cache folder> [zoom]   (dry: reads the plan, writes only the picture)"""
import io
import json
import math
import os
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
from PIL import Image  # noqa: E402
import world_plan as wp  # noqa: E402

COL = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#1b9e77", "#d95f02", "#7570b3",
       "#e7298a", "#66a61e", "#e6ab02", "#666666"]


def my(lat):
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def merc(g):
    return shapely.ops.transform(lambda x, y: (x, np.degrees(np.log(np.tan(np.pi / 4 + np.radians(y) / 2)))), g)


def basemap(ax, view, cache, z):
    w, s, e, n = view
    n2 = 2 ** z
    tx = lambda lon: (lon + 180) / 360 * n2  # noqa: E731
    ty = lambda lat: (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n2  # noqa: E731
    x0, x1, y0, y1 = int(tx(w)), int(tx(e)), int(ty(n)), int(ty(s))
    rows = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            tile = None
            for layer in ("Canvas/World_Light_Gray_Base", "Canvas/World_Light_Gray_Reference"):
                p = os.path.join(cache, f"{layer.split('/')[1]}_{z}_{x}_{y}.png")
                if not os.path.exists(p):
                    req = urllib.request.Request(f"https://server.arcgisonline.com/ArcGIS/rest/services/{layer}/MapServer/tile/{z}/{y}/{x}",
                                                 headers={"User-Agent": "map-render/1.0"})
                    open(p, "wb").write(urllib.request.urlopen(req, timeout=30).read())
                im = Image.open(p).convert("RGBA")
                tile = im if tile is None else Image.alpha_composite(tile, im)
            row.append(np.asarray(tile.convert("RGB")))
        rows.append(np.concatenate(row, axis=1))
    img = np.concatenate(rows, axis=0)
    lon = lambda x: x / n2 * 360 - 180  # noqa: E731
    lat = lambda y: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n2))))  # noqa: E731
    ax.imshow(img, extent=(lon(x0), lon(x1 + 1), my(lat(y1 + 1)), my(lat(y0))), zorder=0, interpolation="lanczos")


def draw(ax, rows, iso, by, view, touched, title, cache, z):
    basemap(ax, view, cache, z)
    w, s, e, n = view
    box = shapely.box(w, s, e, n)
    shown = []
    for k, r in sorted(rows.items()):
        us = [u for u in r["units"].get(iso, []) if u in by]
        if not us:
            continue
        g = wp.to_ll(shapely.union_all([by[u].geom for u in us]))
        if g.intersects(box):
            shown.append((k, g, sum(by[u].pop for u in us)))
    for k, g, p in shown:
        hot = k in touched
        c = COL[sorted(touched).index(k) % len(COL)] if hot else "#555555"
        gm = merc(g)
        for part in getattr(gm, "geoms", [gm]):
            x, y = part.exterior.xy
            if hot:
                ax.fill(x, y, color=c, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, y, color=c, lw=2.2 if hot else 0.8, zorder=3 if hot else 2)
        lp = g.intersection(box)
        if lp.is_empty or lp.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        pt = shapely.ops.polylabel(max(getattr(lp, "geoms", [lp]), key=lambda q: q.area), 0.005)
        ax.text(pt.x, my(pt.y), f"{rows[k]['name']}\n{p / 1e6:.2f}M", ha="center", va="center", zorder=5,
                fontsize=11 if hot else 8, weight="bold" if hot else "normal", color="#111" if hot else "#444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c if hot else "none", alpha=0.8))
    ax.set_xlim(w, e)
    ax.set_ylim(my(s), my(n))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    out, path, cache = sys.argv[1], sys.argv[2], sys.argv[3]
    z = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    os.makedirs(cache, exist_ok=True)
    spec = json.load(open(path, encoding="utf-8"))
    iso = spec["iso"]
    by = pe.load_units(iso)
    doc = shared_lock.read_json(pe.PLAN)
    before = {r["id"]: r for r in doc["metros"] if r.get("units", {}).get(iso)}
    new, gone, _, _, lines = pe.plan_change(spec, json.loads(json.dumps(doc)), by)
    print("\n".join(l for l in lines if not l.startswith("move ")))
    after = {k: v for k, v in before.items() if k not in gone}
    after.update({k: v for k, v in new.items() if k not in gone})
    touched = set(new) | set(gone)
    view = spec["view"]
    fig, axes = plt.subplots(1, 2, figsize=(26, 15))
    label = spec.get("label", spec["task"])
    draw(axes[0], before, iso, by, view, touched, f"{label}: now", cache, z)
    draw(axes[1], after, iso, by, view, touched, f"{label}: proposed", cache, z)
    fig.text(0.5, 0.01, "Basemap: Esri World Light Gray. Zones in colour are the ones that change.",
             ha="center", fontsize=9, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=90)
    print("->", out)


if __name__ == "__main__":
    main()
