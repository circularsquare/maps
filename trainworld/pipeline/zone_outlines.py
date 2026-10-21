"""T-800: the coarse outline of every registered zone, `data/packs/outlines.json`.

    python pipeline/zone_outlines.py [--tol METRES] [--out FILE]

The map draws a zone's outline from its pack's `<id>.boundary.geojson` once the view comes near it
(map/cityBoundaries.ts); this file is what it draws until then, so every zone has an outline from the
start. It is cut from the same boundary files, for the registry's zones in registry order, and written
whole each time (about 15 s for 600 zones), so a rerun after zones are added is all the update there
is; world_run.py's join and docs/adding-a-city.md say when.

Rings are simplified (Douglas-Peucker, TOL_M) without breaking the draw-once rule: a run of vertices
that several zones hold (the same vertices on a 1e-5 degree grid, as the app compares them) is
simplified once, in one direction, with its ends kept, so neighbours still share exactly the same
points. Format: {"tol": metres, "q": grid units per degree, "zones": {id: [ring, ...]}}; a ring is
[x0, y0, dx1, dy1, ...] in grid units, closed (the first point is not repeated at the end).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "data/packs"
REG = ROOT / "app/src/game/registry.json"
OUT = PACKS / "outlines.json"
TOL_M = 300
Q = 10000  # the file's grid, 1/Q degree (about 11 m)
G = 100000  # the app's identity grid for a shared vertex (cityBoundaries' OutlineEdges)
M_LAT = 110574.0
M_LON = 111320.0


def pt(p):
    """A vertex on the identity grid, as integers."""
    return (round((p[0] + 180) * G), round((p[1] + 90) * G))


def dp(xy, tol):
    """Indices kept by Douglas-Peucker on an (n, 2) array in metres; both ends are kept."""
    n = len(xy)
    keep = np.zeros(n, bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = xy[b] - xy[a]
        d = xy[a + 1:b] - xy[a]
        L = math.hypot(seg[0], seg[1])
        dist = np.hypot(d[:, 0], d[:, 1]) if L == 0 else np.abs(d[:, 0] * seg[1] - d[:, 1] * seg[0]) / L
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.nonzero(keep)[0]


def metres(run):
    """A run of identity-grid vertices as metres from its first, scaled by its own mean latitude, so
    a run held by two zones is simplified identically."""
    a = np.array(run, dtype=float)
    lat = a[:, 1] / G - 90
    k = math.cos(math.radians(float(lat.mean())))
    return np.column_stack([(a[:, 0] - a[0, 0]) / G * M_LON * k, (a[:, 1] - a[0, 1]) / G * M_LAT])


def simplify_run(run, tol):
    if len(run) < 3:
        return list(run)
    return [run[i] for i in dp(metres(run), tol)]


def canonical(run):
    """The run in the direction two zones holding it agree on."""
    return run if run[::-1] >= run else run[::-1]


def load(ids):
    zones = {}
    for cid in ids:
        f = PACKS / f"{cid}.boundary.geojson"
        if not f.exists():
            print(f"  warning: {cid}: no boundary file, no coarse outline", flush=True)
            continue
        rings = []
        for feat in json.loads(f.read_text(encoding="utf-8"))["features"]:
            g = feat["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                for r in poly:
                    pts = [pt(p) for p in r]
                    if len(pts) > 1 and pts[0] == pts[-1]:
                        pts.pop()
                    # consecutive repeats on the grid carry no segment
                    pts = [p for i, p in enumerate(pts) if p != pts[i - 1]] if len(pts) > 1 else pts
                    if len(pts) >= 3:
                        rings.append(pts)
        zones[cid] = rings
    return zones


def segkey(a, b):
    return (a, b) if a < b else (b, a)


def outline_rings(zones, tol):
    """{id: [ring of identity-grid vertices, open]} simplified with shared runs kept in step."""
    holders = {}
    for zi, rings in enumerate(zones.values()):
        bit = 1 << zi
        for r in rings:
            for i in range(len(r)):
                k = segkey(r[i], r[(i + 1) % len(r)])
                holders[k] = holders.get(k, 0) | bit
    out = {}
    for cid, rings in zones.items():
        res = []
        for r in rings:
            n = len(r)
            seg = [holders[segkey(r[i], r[(i + 1) % n])] for i in range(n)]  # seg[i]: r[i] to r[i+1]
            cuts = [i for i in range(n) if seg[i] != seg[i - 1]]
            if not cuts:
                # one holder set all round: closed, cut at its first vertex and the one farthest from it
                run = canonical_ring(r)
                far = int(np.argmax(np.hypot(*metres(run).T)))
                ring = simplify_run(run[:far + 1], tol)[:-1] + simplify_run(run[far:], tol)[:-1]
            else:
                r = r[cuts[0]:] + r[:cuts[0]]
                starts = [(c - cuts[0]) % n for c in cuts]
                ring = []
                for j, a in enumerate(starts):
                    b = starts[j + 1] if j + 1 < len(starts) else n
                    run = r[a:b + 1] if b < n else r[a:] + [r[0]]
                    c = canonical(run)
                    sim = simplify_run(c, tol)
                    ring += (sim if c is run else sim[::-1])[:-1]
            if len(ring) >= 3:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                k = math.cos(math.radians(ys[0] / G - 90))
                res.append((max((max(xs) - min(xs)) / G * M_LON * k, (max(ys) - min(ys)) / G * M_LAT), ring))
        # a ring smaller than the tolerance is a dot at any zoom this file serves; a zone keeps its largest
        keep = [ring for ext, ring in res if ext >= tol]
        if not keep and res:
            keep = [max(res, key=lambda e: e[0])[1]]
        out[cid] = keep
    return out


def canonical_ring(r):
    """A ring wholly shared: opened at its least vertex, in the direction two holders agree on."""
    i = r.index(min(r))
    a = r[i:] + r[:i]
    b = [a[0]] + a[:0:-1]
    c = min(a, b)
    return c + [c[0]]


def encode(ring):
    q = [(round(p[0] / G * Q), round(p[1] / G * Q)) for p in ring]
    q = [p for i, p in enumerate(q) if p != q[i - 1]]
    if len(q) < 3:
        return None
    flat = [q[0][0] - 180 * Q, q[0][1] - 90 * Q]
    for i in range(1, len(q)):
        flat += [q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]]
    return flat


def build(tol=TOL_M, out=OUT, root=None):
    reg = json.loads(REG.read_text(encoding="utf-8"))
    ids = [c["id"] for c in reg["cities"]]
    t0 = time.time()
    zones = load(ids)
    rings = outline_rings(zones, tol)
    doc = {"tol": tol, "q": Q, "zones": {}}
    for cid, rs in rings.items():
        enc = [e for e in map(encode, rs) if e]
        if enc:
            doc["zones"][cid] = enc
    text = json.dumps(doc, separators=(",", ":"))
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)
    pts = sum(len(e) // 2 for z in doc["zones"].values() for e in z)
    print(f"{out.name}: {len(doc['zones'])} of {len(ids)} zones, {pts:,} points, {len(text) / 1e6:.2f} MB, "
          f"{time.time() - t0:.0f} s", flush=True)
    return doc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    tol = float(a[a.index("--tol") + 1]) if "--tol" in a else TOL_M
    out = Path(a[a.index("--out") + 1]) if "--out" in a else OUT
    build(tol, out)
