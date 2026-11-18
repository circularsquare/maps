"""T-3506: every registered pack's border, split by who is on the other side: each neighbouring pack, dead space in
its own country, another country's land (no pack), and coast. Read-only on data. Output borders.json beside this file.
usage: borders.py [--only id,id]"""
import json, os, sys, time, pickle, math
os.environ["OMP_NUM_THREADS"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
import shapely
import numpy as np
from shapely.geometry import shape, box
from shapely import STRtree
import pyogrio

HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:/Users/anita/projects/maps/trainworld/"
P = TW + "data/packs/"
REG = TW + "app/src/game/registry.json"
NE_LAND = TW + "data/raw/world_field/ne_10m_land.zip"
NE_ADM = TW + "data/raw/world_field/ne_10m_admin_0_countries.zip"
TOL_KM = 0.4     # a neighbour's outline within this of ours shares the border
INNER_KM = 1.5   # land this far inside Natural Earth's coast is land, not coast

reg = json.load(open(REG, encoding="utf8"))["cities"]
regby = {c["id"]: c for c in reg}
only = None
WATER = "--water" in sys.argv
if "--only" in sys.argv:
    only = set(sys.argv[sys.argv.index("--only") + 1].split(","))

cache = os.path.join(HERE, "geoms.pkl")
t0 = time.time()
if os.path.exists(cache):
    geoms, pops = pickle.load(open(cache, "rb"))
else:
    geoms, pops = {}, {}
    for c in reg:
        i = c["id"]
        f = P + i + ".boundary.geojson"
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf8") as fh:
            fc = json.load(fh)
        g = shapely.union_all([shapely.make_valid(shape(ft["geometry"])) for ft in fc["features"]])
        g = shapely.union_all([x for x in getattr(g, "geoms", [g]) if x.geom_type in ("Polygon", "MultiPolygon")]) if g.geom_type == "GeometryCollection" else g
        geoms[i] = g
        try:
            with open(P + i + ".json", encoding="utf8") as fh:
                pops[i] = json.load(fh).get("totals", {}).get("pop")
        except Exception:
            pops[i] = None
    pickle.dump((geoms, pops), open(cache, "wb"))
print("geoms", len(geoms), round(time.time() - t0), "s", flush=True)

land = pyogrio.read_dataframe(f"zip://{NE_LAND}!ne_10m_land.shp")
landg = shapely.union_all(list(land.geometry.values))
adm = pyogrio.read_dataframe(f"zip://{NE_ADM}!ne_10m_admin_0_countries.shp", columns=["ISO_A2_EH"])
admby = {}
for iso, g in zip(adm["ISO_A2_EH"], adm.geometry):
    admby[iso] = g if iso not in admby else admby[iso].union(g)
print("NE loaded", round(time.time() - t0), "s", flush=True)

OUTL = json.load(open(P + "outlines.json", encoding="utf8"))
QQ = OUTL["q"]


def water_of(i):
    """The zone's own sea from outlines.json (rings combined even-odd), or None."""
    rs = OUTL["water"].get(i)
    if not rs:
        return None
    g = None
    for e in rs:
        x, y = e[0], e[1]  # the file stores lon and lat times q, deltas after the first
        pts = [(x / QQ, y / QQ)]
        for k in range(2, len(e), 2):
            x += e[k]; y += e[k + 1]
            pts.append((x / QQ, y / QQ))
        if len(pts) < 3:
            continue
        p = shapely.make_valid(shapely.Polygon(pts))
        g = p if g is None else g.symmetric_difference(p)
    return g


ids = list(geoms)
tree = STRtree([geoms[i] for i in ids])


def proj(lat0):
    kx = 111.32 * math.cos(math.radians(lat0))
    ky = 110.57
    return lambda g: shapely.transform(g, lambda a: np.column_stack([a[:, 0] * kx, a[:, 1] * ky]))


out = {}
todo = [i for i in ids if (only is None or i in only)]
for n, i in enumerate(todo):
    g = geoms[i]
    b = g.bounds
    lat0 = (b[1] + b[3]) / 2
    pj = proj(lat0)
    margin = 0.05
    near = [ids[k] for k in tree.query(box(b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)) if ids[k] != i]
    gk = pj(g)
    ring = gk.boundary
    L = ring.length
    nb = {}
    covered = []
    for j in near:
        hk = pj(geoms[j]).buffer(TOL_KM)
        s = ring.intersection(hk).length
        if s > 0.05:
            nb[j] = round(s, 2)
            covered.append(hk)
    rest = ring.difference(shapely.union_all(covered)) if covered else ring
    if WATER:
        w = water_of(i)
        if w is not None:
            rest = rest.difference(pj(w).buffer(TOL_KM))
    win = box(b[0] - 0.3, b[1] - 0.3, b[2] + 0.3, b[3] + 0.3)
    lk = pj(landg.intersection(win)).buffer(-INNER_KM)
    restland = rest.intersection(lk)
    iso = regby.get(i, {}).get("country")
    own = admby.get(iso)
    if own is not None:
        ok = pj(own.intersection(win)).buffer(-INNER_KM)
        dead = restland.intersection(ok).length
    else:
        dead = 0.0
    landlen = restland.length
    out[i] = {"L": round(L, 2), "nb": nb, "dead": round(dead, 2), "foreign": round(landlen - dead, 2),
              "coast": round(L - sum(nb.values()) - landlen, 2) if WATER else round(rest.length - landlen, 2), "parts": len(getattr(g, "geoms", [g])),
              "area": round(gk.area, 1), "pop": pops.get(i)}
    if n % 200 == 0:
        print(n, len(todo), i, round(time.time() - t0), "s", flush=True)

name = ("borders_w.json" if WATER else "borders.json") if only is None else ("borders_only_w.json" if WATER else "borders_only.json")
json.dump(out, open(os.path.join(HERE, name), "w", encoding="utf8"), ensure_ascii=False)
print("done", len(out), round(time.time() - t0), "s", flush=True)



