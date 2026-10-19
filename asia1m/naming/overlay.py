"""Admin outlines from helper1m for each batch thumbnail -> batches/<name>/overlay.json

    python asia1m/naming/overlay.py <batch name>      (batch.py runs this itself)

For every region, the helper1m units (China: counties) touching its thumbnail, as SVG paths in
thumbnail pixels, with names, parent and helper1m's latest population estimate. The viewer draws
them faintly over the thumbnail with a hover like helper1m's.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "regionnaming"))
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform
from geo import lonlat_to_px, px_to_lonlat

H1 = os.path.join(HERE, "..", "..", "helper1m", "countries")
# country in regions.json -> (helper1m id, level drawn, parent level shown in the hover)
LEVELS = {"China": ("china", 3, 2), "India": ("india", 3, 2), "Bangladesh": ("bangladesh", 3, 2),
          "Kazakhstan": ("kazakhstan", 2, 1), "Kyrgyzstan": ("kyrgyzstan", 2, 1),
          "Tajikistan": ("tajikistan", 2, 1), "Uzbekistan": ("uzbekistan", 2, 1),
          "Nepal": ("nepal", 2, 1), "Russia": ("russia", 2, 1)}

name = sys.argv[1]
B = os.path.join(HERE, "batches", name)
facts = json.load(open(f"{B}/facts.json", encoding="utf-8"))
regions = [r for r in facts["regions"] if "thumb" in r]

def latest(pops):
    if not pops: return None
    y = max(pops, key=int)
    return dict(year=int(y), pop=pops[y])

# whole batch extent in map pixels -> lon/lat window to pre-filter features
x0 = min(r["thumb"]["x0"] for r in regions); y0 = min(r["thumb"]["y0"] for r in regions)
x1 = max(r["thumb"]["x0"] + r["thumb"]["w"] / r["thumb"]["k"] for r in regions)
y1 = max(r["thumb"]["y0"] + r["thumb"]["h"] / r["thumb"]["k"] for r in regions)
xs = np.linspace(x0, x1, 9); ys = np.linspace(y0, y1, 9)
gx, gy = np.meshgrid(xs, ys)
lo, la = px_to_lonlat(gx.ravel(), gy.ravel())
win = (min(lo) - 1, min(la) - 1, max(lo) + 1, max(la) + 1)

feats = []
for country in sorted({r["country"] for r in regions}):
    if country not in LEVELS: continue
    cid, lvl, plvl = LEVELS[country]
    meta = json.load(open(f"{H1}/{cid}/meta.json", encoding="utf-8"))
    files = {l["level"]: l.get("file") for l in meta["admin_levels"]}
    g = gpd.read_file(f"{H1}/{cid}/{files[lvl]}", bbox=win)
    parents = {}
    if plvl and files.get(plvl):
        pg = gpd.read_file(f"{H1}/{cid}/{files[plvl]}", ignore_geometry=True)
        parents = {p["code"]: p.get("name") for _, p in pg.iterrows()}
    for _, f in g.iterrows():
        if f.geometry is None or f.geometry.is_empty: continue
        geom = transform(lambda x, y, z=None: lonlat_to_px(np.asarray(x), np.asarray(y)), f.geometry)
        if not geom.is_valid: geom = geom.buffer(0)   # a few helper1m shapes have stray holes
        pops = f.get("populations")
        if isinstance(pops, str): pops = json.loads(pops)
        feats.append(dict(geom=geom, name=f.get("name"), local=f.get("name_cn") or f.get("name_local"),
                          parent=parents.get(f.get("parent_code")), est=latest(pops)))
print(f"{len(feats)} helper1m units in the batch window")

def path(geom, ox, oy, k):
    out = []
    polys = getattr(geom, "geoms", [geom])
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)
            c = (c - [ox, oy]) * k
            out.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in c) + "Z")
    return "".join(out)

result = {}
for r in regions:
    t = r["thumb"]
    crop = box(t["x0"], t["y0"], t["x0"] + t["w"] / t["k"], t["y0"] + t["h"] / t["k"])
    units = []
    for f in feats:
        if not f["geom"].intersects(crop): continue
        g = f["geom"].intersection(crop).simplify(0.4 / max(t["k"], 0.25))
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            g = g.buffer(0) if not g.is_empty else g
            if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"): continue
        lp = g.representative_point()
        units.append(dict(name=f["name"], local=f["local"], parent=f["parent"], est=f["est"],
                          d=path(g, t["x0"], t["y0"], t["k"]),
                          lx=round((lp.x - t["x0"]) * t["k"], 1), ly=round((lp.y - t["y0"]) * t["k"], 1),
                          a=round(g.area * t["k"] ** 2)))
    result[r["n"]] = dict(w=t["w"], h=t["h"], units=units)
json.dump(result, open(f"{B}/overlay.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"wrote {B}/overlay.json ({os.path.getsize(f'{B}/overlay.json') // 1024} KB)")
