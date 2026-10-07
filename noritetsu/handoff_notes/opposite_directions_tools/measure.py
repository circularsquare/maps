"""Measure, per built country, from the shipped dist/data and data/proc (read only):
1. ways a line's two directions use as a pair (one variant over one, another variant over the
   other in the opposite direction, side by side with no drawn way between) whose owners
   differ;
2. ways left 'abroad' by ownership (outside the coarse regions.json outline + 150 m, no
   register owner), classed by where they really are.
"""
import json, math, pickle, sys, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, shapely
from shapely import STRtree
from shapely.geometry import Polygon, LineString, Point, shape
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
sys.path.insert(0, str(ROOT))
from ownership import merc, scale_at, ABROAD_M
import borders
from build_model import group_lines, assemble, Coords

PAIR_M = 25.0
R = 20037508.34 / 180.0
NE_LOADED = {}


def ne():
    if not NE_LOADED:
        iso, geo = [], []
        for f in json.loads(borders.NAMES.read_text(encoding="utf-8"))["features"]:
            q = f["properties"]
            g = shape(f["geometry"])
            for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                iso.append(q.get("ISO_A2_EH"))
                geo.append(p)
        NE_LOADED.update(iso=iso, tree=STRtree(geo))
    return NE_LOADED


def run(cc, out, detail=False):
    t0 = time.time()
    import os
    D = Path(os.environ["MEASURE_DIST"]) / cc if os.environ.get("MEASURE_DIST") else ROOT / "dist" / "data" / cc
    P = ROOT / "data" / "proc" / cc
    lines = json.loads((D / "lines.json").read_text(encoding="utf-8"))["lines"]
    byid = {l["id"]: l for l in lines}
    wj = json.loads((D / "ways.json").read_text(encoding="utf-8"))
    wl = wj["lines"]
    unowned = set(wj["unowned"])
    with open(P / "ways.pkl", "rb") as f:
        W = pickle.load(f)
    with open(P / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    c = np.load(P / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    coords = Coords(cid, cx, cy)
    reg = json.loads((ROOT / "dist" / "regions.json").read_text(encoding="utf-8"))["regions"][cc]
    polys = []
    for ring in reg["parts"]:
        r = ring[0] if ring and isinstance(ring[0][0], list) else ring
        arr = np.asarray(r, dtype=np.float64)
        if len(arr) >= 3:
            polys.append(Polygon(merc(arr[:, 0], arr[:, 1])).buffer(0))
    lat_mid = (reg["bbox"][1] + reg["bbox"][3]) / 2
    coarse = shapely.union_all(polys).buffer(ABROAD_M * scale_at(lat_mid))
    shapely.prepare(coarse)

    wids = [int(w) for w in wj["ways"]]
    geo, lens, scs = {}, {}, {}
    for w in wids:
        t = W.get(w)
        if t is None:
            continue
        nodes = np.asarray(t[1], dtype=np.int64)
        p = np.clip(np.searchsorted(cid, nodes), 0, cid.size - 1)
        ok = cid[p] == nodes
        if ok.sum() < 2:
            continue
        lon, lat = cx[p[ok]] / 1e7, cy[p[ok]] / 1e7
        g = LineString(merc(lon, lat))
        sc = scale_at(lat.mean())
        geo[w], scs[w], lens[w] = g, sc, g.length / sc
    gw = sorted(geo)
    all_tree = STRtree([geo[w] for w in gw])
    mids = {w: geo[w].interpolate(0.5, normalized=True) for w in gw}

    def users(w):
        return [wl[i] for i in wj["ways"][str(w)]]

    def is_reg(lid):
        l = byid.get(lid)
        return l is not None and l.get("src", "osm") != "osm"

    def is_svc(lid):
        l = byid.get(lid)
        return l is not None and bool(l.get("service"))

    outside = {w for w in gw if not coarse.contains(mids[w])}

    def owner_class(w):
        us = users(w)
        if str(w) in unowned or not us:
            return ("none", None)
        o = us[0]
        if is_reg(o):
            return ("reg", o)
        if is_svc(o) or w in outside:
            return ("none", None)
        return ("osm", o)

    # ---------------- 1. directional pairs
    groups, routes = group_lines(rels, lambda *_: None)
    pair_km = 0.0
    diff = Counter()
    diff_lines = Counter()
    seen = set()
    rows = []
    for lid, mtags, rids in groups:
        if len(rids) < 2:
            continue
        vdir = []          # per variant: way -> travel unit vector (Mercator)
        for r in rids:
            mem = routes[r][1]
            runs = assemble(mem, W, coords)
            pos = {}
            for ri, (ids, _xy) in enumerate(runs):
                for k, n in enumerate(ids.tolist()):
                    pos.setdefault(n, (ri, k))
            d = {}
            for ty, ref, role in mem:
                if ty != "w" or (role and not role.startswith(("forward", "backward"))) or ref not in geo:
                    continue
                ns = W[ref][1]
                a, b = pos.get(ns[0]), pos.get(ns[-1])
                if a is None or b is None or a[0] != b[0] or a[1] == b[1]:
                    continue
                cs = np.asarray(geo[ref].coords)
                v = cs[-1] - cs[0]
                nv = np.hypot(*v)
                if nv <= 0:
                    continue
                d[ref] = v / nv * (1 if b[1] > a[1] else -1)
            vdir.append(d)
        allw = sorted(set().union(*[set(d) for d in vdir]))
        if len(allw) < 2:
            continue
        tree = STRtree([geo[w] for w in allw])
        for a in allw:
            sc = scs[a]
            ma = mids[a]
            for k in tree.query(ma, predicate="dwithin", distance=PAIR_M * sc):
                b = allw[int(k)]
                if b == a:
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                ok = False
                for da in vdir:
                    if a not in da or b in da:
                        continue
                    for db in vdir:
                        if b not in db or a in db:
                            continue
                        if float(np.dot(da[a], db[b])) < -0.7:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    continue
                ga, gb = geo[a], geo[b]
                share = ga.intersection(gb.buffer(PAIR_M * sc, quad_segs=2)).length / max(ga.length, 1e-9)
                if share < 0.5:
                    continue
                # adjacent: no other drawn way crosses the line from a's middle to b
                pb = gb.interpolate(gb.project(ma))
                conn = LineString([ma.coords[0], pb.coords[0]])
                between = False
                if conn.length > 0:
                    for kk in all_tree.query(conn, predicate="intersects"):
                        w2 = gw[int(kk)]
                        if w2 in (a, b):
                            continue
                        x = geo[w2].intersection(conn)
                        if x.is_empty:
                            continue
                        if x.distance(ma) > 0.3 * sc and x.distance(pb) > 0.3 * sc:
                            between = True
                            break
                if between:
                    continue
                seen.add(key)
                L = min(lens[a], lens[b])
                pair_km += L / 1000
                ca, cb = owner_class(a), owner_class(b)
                if ca != cb:
                    k2 = "/".join(sorted([ca[0], cb[0]]))
                    if ca[0] == cb[0]:
                        k2 += "(diff)"
                    diff[k2] += L / 1000
                    diff_lines[lid] += L / 1000
                    if detail:
                        lon = ma.x / R
                        lat = math.degrees(2 * math.atan(math.exp(ma.y / R * math.pi / 180)) - math.pi / 2)
                        rows.append(f"{mtags.get('name')} | {a}:{ca} {b}:{cb} {L:.0f} m at {lat:.5f},{lon:.5f}")
    # ---------------- 2. abroad
    ab = Counter()
    home_full = borders.outline(cc)
    iso_home = borders.ISO.get(cc, cc.upper())
    override = cc in borders.OUTLINE
    N = ne()
    for w in outside:
        us = users(w)
        if str(w) not in unowned and us and is_reg(us[0]):
            continue
        osm_users = [u for u in us if not is_reg(u) and not is_svc(u)]
        if not osm_users:
            continue
        m = mids[w]
        lon = m.x / R
        lat = math.degrees(2 * math.atan(math.exp(m.y / R * math.pi / 180)) - math.pi / 2)
        pt = Point(lon, lat)
        km = lens[w] / 1000
        if home_full is not None and home_full.distance(pt) * 111320 <= ABROAD_M:
            ab["home full outline"] += km
            continue
        hit = [N["iso"][int(i)] for i in N["tree"].query(pt, predicate="within")]
        if not hit:
            ab["water"] += km
        elif iso_home in hit and not override:
            ab["home in NE"] += km
        else:
            ab["foreign"] += km
    out.write(json.dumps({"cc": cc, "pair_km": round(pair_km, 1),
                          "diff": {k: round(v, 2) for k, v in diff.items()},
                          "diff_lines": [(byid.get(k, {}).get("name") or k, round(v, 2)) for k, v in diff_lines.most_common(6)],
                          "abroad": {k: round(v, 2) for k, v in ab.items()},
                          "secs": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
    for r in rows:
        out.write("   " + r + "\n")
    out.flush()


if __name__ == "__main__":
    args = sys.argv[2:]
    detail = "--detail" in args
    args = [a for a in args if a != "--detail"]
    if args == ["--all"]:
        args = sorted(json.loads((ROOT / "dist" / "regions.json").read_text(encoding="utf-8"))["regions"])
    with open(sys.argv[1], "a", encoding="utf-8") as out:
        for cc in args:
            try:
                run(cc, out, detail)
            except Exception as e:
                import traceback
                out.write(json.dumps({"cc": cc, "error": traceback.format_exc()}) + "\n")
            print(cc, flush=True)
